package qupath.ext.celltune.classifier;

import com.microsoft.ml.lightgbm.PredictionType;
import io.github.metarank.lightgbm4j.LGBMBooster;
import io.github.metarank.lightgbm4j.LGBMDataset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.UUID;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.celltune.util.TrainingThreads;

/**
 * Wraps LightGBM4J training and prediction behind the same style interface
 * as {@link XGBoostModel}.
 * <p>
 * Supports binary and multiclass objectives. For multiclass, uses
 * {@code objective=multiclass} with {@code num_class=N}.
 * Probability vectors are always returned as {@code float[nClasses]}.
 */
public class LightGBMModel {

    private static final Logger logger = LoggerFactory.getLogger(LightGBMModel.class);

    /**
     * LightGBM4J extracts its native library by reading {@code java.io.tmpdir} during
     * static initialisation of {@link LGBMBooster}. On shared/HPC systems the default
     * {@code /tmp/lib_lightgbm.so} path frequently fails: another user owns the file,
     * or {@code /tmp} is mounted noexec. Redirect the extraction to a per-user
     * directory under {@code $HOME/.celltune/native/<pid>-<uuid>/} before the LGBM
     * classes are referenced, then restore {@code java.io.tmpdir} so the rest of the
     * JVM (QuPath, etc.) keeps using its original temp dir.
     */
    static {
        String originalTmp = System.getProperty("java.io.tmpdir");
        try {
            String userHome = System.getProperty("user.home");
            Path baseDir = (userHome != null && !userHome.isBlank())
                    ? Paths.get(userHome, ".celltune", "native")
                    : Paths.get(originalTmp == null ? "." : originalTmp, "celltune-native");
            // Use a unique sub-directory per JVM invocation so concurrent QuPath
            // processes never collide on the extracted .so file.
            Path nativeDir =
                    baseDir.resolve(String.valueOf(ProcessHandle.current().pid()) + "-" + UUID.randomUUID());
            Files.createDirectories(nativeDir);
            // Best-effort cleanup at JVM exit; ignore failures (file may be in use).
            nativeDir.toFile().deleteOnExit();
            System.setProperty("java.io.tmpdir", nativeDir.toString());
            // Force LGBMBooster's static initializer (which extracts the native lib
            // using the now-redirected java.io.tmpdir) to run while the redirect
            // is still in effect.
            Class.forName(LGBMBooster.class.getName());
            logger.info("LightGBM native library extracted to user-scoped tmp dir: {}", nativeDir);
        } catch (Throwable t) {
            // Don't fail extension load just because LightGBM init failed; the actual
            // training call will surface the original error if the lib never loaded.
            logger.warn("LightGBM native init redirect failed: {}", t.getMessage());
        } finally {
            if (originalTmp != null) {
                System.setProperty("java.io.tmpdir", originalTmp);
            }
        }
    }

    private LGBMBooster booster;
    private int nClasses;
    private List<String> classNames;
    private List<String> featureNames;
    private String lastDevice = "unknown";

    // ── Training ────────────────────────────────────────────────────────────────

    /**
     * Train a new LightGBM model.
     *
     * @param flatData     row-major feature matrix (nSamples × nFeatures)
     * @param labels       integer class labels as float (0-indexed)
     * @param nSamples     number of training samples
     * @param nFeatures    number of features per sample
     * @param classNames   ordered list of class names
     * @param featureNames ordered list of feature names
     * @param numRounds    boosting iterations
     * @param maxDepth     max tree depth (-1 for unlimited)
     * @param learningRate learning rate
     * @param subsample    row subsampling ratio per round
     * @throws Exception if training fails
     */
    public void train(
            float[] flatData,
            float[] labels,
            int nSamples,
            int nFeatures,
            List<String> classNames,
            List<String> featureNames,
            int numRounds,
            int maxDepth,
            float learningRate,
            float subsample)
            throws Exception {

        this.nClasses = classNames.size();
        this.classNames = List.copyOf(classNames);
        this.featureNames = List.copyOf(featureNames);

        // Create dataset from matrix
        LGBMDataset dataset = LGBMDataset.createFromMat(flatData, nSamples, nFeatures, true, "", null);
        // LightGBM also rejects special/non-ASCII chars in feature names
        String[] safeNames =
                featureNames.stream().map(XGBoostModel::sanitiseFeatureName).toArray(String[]::new);
        dataset.setFeatureNames(safeNames);
        dataset.setField("label", labels);

        // Build parameter string
        String params = buildParams(nClasses, maxDepth, learningRate, subsample);

        // This build pins the CPU-only lightgbm4j artifact (see build.gradle.kts:
        // "io.github.metarank:lightgbm4j"). No GPU kernels are shipped, so device_type=gpu can
        // never succeed. The previous probe-and-fallback ran a *full* numRounds loop under GPU
        // params on every train() call before failing — and if it failed part-way through, that
        // work was discarded and CPU training restarted from round 0. Always train on CPU here.
        // If/when a GPU artifact is wired in, restore the probe-and-fallback logic.
        // (Same reasoning, same conclusion, as XGBoostModel.train.)
        booster = LGBMBooster.create(dataset, params);
        int rounds = 0;
        for (int i = 0; i < numRounds; i++) {
            rounds++;
            // updateOneIter returns LightGBM's is_finished: with min_gain_to_split set, no split
            // may be worth making any more, and every further call is a no-op.
            if (booster.updateOneIter()) {
                break;
            }
        }
        logger.info(
                "LightGBM training: CPU — {} samples, {} features, {} classes, {} of {} rounds",
                nSamples,
                nFeatures,
                nClasses,
                rounds,
                numRounds);

        // Close dataset — booster keeps its own copy
        dataset.close();

        lastDevice = "CPU";
    }

    /** @return the device used for the last training run */
    public String getLastDevice() {
        return lastDevice;
    }

    // ── Early Stopping ──────────────────────────────────────────────────────────

    /**
     * Find the optimal number of boosting rounds by training on a subset and
     * monitoring validation loss. Uses CPU only for speed.
     *
     * @return optimal number of rounds (1-indexed)
     */
    static int findBestRounds(
            float[] trainData,
            float[] trainLabels,
            int trainSize,
            float[] valData,
            float[] valLabels,
            int valSize,
            int nFeatures,
            int nClasses,
            int maxRounds,
            int maxDepth,
            float learningRate,
            float subsample,
            int patience,
            Consumer<String> log)
            throws Exception {

        LGBMDataset dataset = LGBMDataset.createFromMat(trainData, trainSize, nFeatures, true, "", null);
        dataset.setField("label", trainLabels);

        String params = buildParams(nClasses, maxDepth, learningRate, subsample);
        LGBMBooster booster = LGBMBooster.create(dataset, params);

        // Register the validation fold with the booster so LightGBM scores it natively. It keeps
        // a running raw score per validation row and adds only the new trees each iteration, so
        // the whole search costs O(rounds); calling predictForMat per round instead re-scored
        // every tree from scratch and re-marshalled the fold across JNI, making it O(rounds²).
        //
        // The reference dataset is mandatory, not cosmetic: LightGBM rejects a validation set
        // whose feature bin mappers differ from the training set's.
        LGBMDataset valDataset = null;
        int metricIdx = -1;
        try {
            valDataset = LGBMDataset.createFromMat(valData, valSize, nFeatures, true, "", dataset);
            valDataset.setField("label", valLabels);
            booster.addValidData(valDataset);
            metricIdx = resolveMetricIndex(booster.getEvalNames(), nClasses);
        } catch (Exception ex) {
            logger.info("LightGBM native validation metric unavailable ({}); scoring manually", ex.getMessage());
            metricIdx = -1;
        }

        try {
            double bestLoss = Double.MAX_VALUE;
            int bestRound = 0;
            // One-way latch: native scoring is used until it misbehaves, after which the rest of
            // the search is scored manually and never switches back. It cannot oscillate, so the
            // change can never end up slower than the old always-manual path.
            //
            // A mid-search fallback does compare a manual loss against a best-so-far that came
            // from LightGBM, which is only sound because they are the same quantity: mean log-loss
            // over the same validation rows, same formula, same /n. They differ at most by
            // LightGBM's internal probability clamp. Nothing here relies on tighter agreement than
            // that, and reaching this path at all means getEval already returned a non-finite
            // value, which does not happen in a healthy run.
            boolean useNative = metricIdx >= 0;

            for (int round = 0; round < maxRounds; round++) {
                boolean finished = booster.updateOneIter();

                double loss = Double.NaN;
                if (useNative) {
                    try {
                        double[] evals = booster.getEval(1); // 0 = training data, 1 = first valid set
                        loss = (metricIdx < evals.length) ? evals[metricIdx] : Double.NaN;
                    } catch (Exception ex) {
                        loss = Double.NaN;
                    }
                    if (!Double.isFinite(loss)) {
                        logger.info("LightGBM getEval returned no usable value; falling back to manual scoring");
                        useNative = false;
                    }
                }
                if (!useNative) {
                    double[] preds = booster.predictForMat(
                            valData, valSize, nFeatures, true, PredictionType.C_API_PREDICT_NORMAL);
                    loss = computeLogloss(preds, valLabels, valSize, nClasses);
                }

                if (loss < bestLoss) {
                    bestLoss = loss;
                    bestRound = round;
                }
                if (round - bestRound >= patience) break;
                // LightGBM has nothing left to add — further rounds cannot change the result.
                if (finished) break;
            }

            int actualRounds = bestRound + 1;
            log.accept(String.format(
                    "LightGBM early stopping: best round %d/%d (val loss: %.6f)", actualRounds, maxRounds, bestLoss));
            return actualRounds;

        } finally {
            booster.close();
            if (valDataset != null) {
                try {
                    valDataset.close();
                } catch (Exception ignore) {
                }
            }
            dataset.close();
        }
    }

    /**
     * Picks the index of the log-loss metric among the booster's configured eval metrics.
     * <p>
     * {@code buildParams} configures exactly one ({@code binary_logloss} or {@code multi_logloss},
     * matching {@link #computeLogloss}), but the name is matched rather than assumed so that
     * adding a metric later cannot silently start early-stopping on the wrong one.
     *
     * @return the index, or {@code -1} if no log-loss metric is present
     */
    static int resolveMetricIndex(String[] evalNames, int nClasses) {
        if (evalNames == null || evalNames.length == 0) {
            return -1;
        }
        String expected = nClasses == 2 ? "binary_logloss" : "multi_logloss";
        for (int i = 0; i < evalNames.length; i++) {
            if (expected.equals(evalNames[i])) {
                return i;
            }
        }
        for (int i = 0; i < evalNames.length; i++) {
            if (evalNames[i] != null && evalNames[i].endsWith("logloss")) {
                return i;
            }
        }
        return -1;
    }

    /** Compute mean log-loss from raw LightGBM predictions. */
    private static double computeLogloss(double[] preds, float[] labels, int n, int nClasses) {
        double loss = 0;
        if (nClasses == 2 && preds.length == n) {
            for (int i = 0; i < n; i++) {
                int trueClass = (int) labels[i];
                double p = trueClass == 1 ? preds[i] : 1 - preds[i];
                loss += -Math.log(Math.max(p, 1e-15));
            }
        } else {
            for (int i = 0; i < n; i++) {
                int trueClass = (int) labels[i];
                double p = preds[i * nClasses + trueClass];
                loss += -Math.log(Math.max(p, 1e-15));
            }
        }
        return loss / n;
    }

    // ── Prediction ──────────────────────────────────────────────────────────────

    /**
     * Predict class probabilities for multiple cells.
     *
     * @param flatData  row-major feature matrix (nSamples × nFeatures)
     * @param nSamples  number of samples
     * @param nFeatures number of features
     * @return probability matrix [nSamples][nClasses]
     * @throws Exception if prediction fails
     */
    public float[][] predictProba(float[] flatData, int nSamples, int nFeatures) throws Exception {

        double[] rawPreds =
                booster.predictForMat(flatData, nSamples, nFeatures, true, PredictionType.C_API_PREDICT_NORMAL);

        float[][] result = new float[nSamples][nClasses];

        if (nClasses == 2 && rawPreds.length == nSamples) {
            // Binary: raw returns P(class=1), one value per sample
            for (int i = 0; i < nSamples; i++) {
                result[i][1] = (float) rawPreds[i];
                result[i][0] = 1f - result[i][1];
            }
        } else {
            // Multiclass: raw is flat [nSamples * nClasses], row-major
            for (int i = 0; i < nSamples; i++) {
                for (int c = 0; c < nClasses; c++) {
                    result[i][c] = (float) rawPreds[i * nClasses + c];
                }
            }
        }
        return result;
    }

    /**
     * Predict the single best class index for each sample.
     *
     * @param flatData  row-major feature matrix
     * @param nSamples  number of samples
     * @param nFeatures number of features
     * @return array of predicted class indices
     * @throws Exception if prediction fails
     */
    public int[] predict(float[] flatData, int nSamples, int nFeatures) throws Exception {

        float[][] probs = predictProba(flatData, nSamples, nFeatures);
        int[] preds = new int[nSamples];
        for (int i = 0; i < nSamples; i++) {
            int best = 0;
            for (int c = 1; c < nClasses; c++) {
                if (probs[i][c] > probs[i][best]) best = c;
            }
            preds[i] = best;
        }
        return preds;
    }

    // ── SHAP / Feature Importance ────────────────────────────────────────────────

    /**
     * Compute per-class mean absolute SHAP values using LightGBM's native
     * TreeSHAP implementation ({@code C_API_PREDICT_CONTRIB}).
     * <p>
     * For binary models the same values are reflected for both classes.
     * For multiclass ({@code objective=multiclass}) values are class-specific.
     *
     * @param flatData  row-major feature matrix (nSamples × nFeatures)
     * @param nSamples  number of samples
     * @param nFeatures number of features
     * @return mean absolute SHAP matrix [nClasses][nFeatures]
     * @throws Exception if prediction fails
     */
    public double[][] computeMeanAbsShap(float[] flatData, int nSamples, int nFeatures) throws Exception {
        // C_API_PREDICT_CONTRIB → TreeSHAP contributions
        // Binary:     flat [nSamples * (nFeatures + 1)]
        // Multiclass: flat [nSamples * nClasses * (nFeatures + 1)]  (class-major order)
        double[] raw = booster.predictForMat(flatData, nSamples, nFeatures, true, PredictionType.C_API_PREDICT_CONTRIB);

        double[][] result = new double[nClasses][nFeatures];
        int stride = nFeatures + 1; // +1 for the bias term

        if (nClasses == 2) {
            // Binary: raw length = nSamples * (nFeatures + 1)
            for (int i = 0; i < nSamples; i++) {
                for (int f = 0; f < nFeatures; f++) {
                    double s = Math.abs(raw[i * stride + f]);
                    result[0][f] += s;
                    result[1][f] += s;
                }
            }
        } else {
            // Multiclass: class k, feature f → raw[i * nClasses*(nFeatures+1) + k*(nFeatures+1) + f]
            int classStride = nClasses * stride;
            for (int i = 0; i < nSamples; i++) {
                for (int c = 0; c < nClasses; c++) {
                    for (int f = 0; f < nFeatures; f++) {
                        result[c][f] += Math.abs(raw[i * classStride + c * stride + f]);
                    }
                }
            }
        }

        // Average over samples
        for (int c = 0; c < nClasses; c++) {
            for (int f = 0; f < nFeatures; f++) {
                result[c][f] /= nSamples;
            }
        }
        return result;
    }

    // ── Serialisation ───────────────────────────────────────────────────────────

    /**
     * Serialise the trained model to a byte array (UTF-8 model string).
     *
     * @return model bytes
     * @throws Exception if serialisation fails
     */
    public byte[] toBytes() throws Exception {
        String modelStr = booster.saveModelToString(0, 0, LGBMBooster.FeatureImportanceType.SPLIT);
        return modelStr.getBytes(StandardCharsets.UTF_8);
    }

    /**
     * Load a model from a byte array.
     *
     * @param bytes        model bytes (UTF-8 model string)
     * @param classNames   ordered class names
     * @param featureNames ordered feature names
     * @throws Exception if loading fails
     */
    public void loadFromBytes(byte[] bytes, List<String> classNames, List<String> featureNames) throws Exception {

        this.classNames = List.copyOf(classNames);
        this.featureNames = List.copyOf(featureNames);
        this.nClasses = classNames.size();

        String modelStr = new String(bytes, StandardCharsets.UTF_8);
        booster = LGBMBooster.loadModelFromString(modelStr);
    }

    /**
     * Release native resources held by the booster.
     * Call this when the model is no longer needed.
     */
    public void close() {
        if (booster != null) {
            try {
                booster.close();
            } catch (Exception e) {
                logger.warn("Error closing LightGBM booster", e);
            }
            booster = null;
        }
    }

    // ── Accessors ───────────────────────────────────────────────────────────────

    public boolean isTrained() {
        return booster != null;
    }

    public int getNumClasses() {
        return nClasses;
    }

    public List<String> getClassNames() {
        return classNames;
    }

    public List<String> getFeatureNames() {
        return featureNames;
    }

    // ── Private helpers ─────────────────────────────────────────────────────────

    private static String buildParams(int nClasses, int maxDepth, float learningRate, float subsample) {
        return buildParams(nClasses, maxDepth, learningRate, subsample, TrainingThreads.total());
    }

    /**
     * The single definition of how a LightGBM booster in this extension is configured.
     * <p>
     * {@link HyperparameterTuner} must call this rather than assembling its own string. It used to
     * do the latter and omitted {@code min_gain_to_split}, so cross-validation scored unconstrained
     * boosters and then handed the winning depth/rate/subsample to a constrained one — the tuner
     * was optimising a model that never got built. Any parameter added here must reach both paths
     * or the same class of bug comes back.
     *
     * @param numThreads thread budget for this booster; the tuner divides the total across its
     *                   concurrently-evaluated folds rather than letting each request every core
     */
    static String buildParams(int nClasses, int maxDepth, float learningRate, float subsample, int numThreads) {
        StringBuilder sb = new StringBuilder();
        if (nClasses == 2) {
            sb.append("objective=binary metric=binary_logloss");
        } else {
            sb.append("objective=multiclass metric=multi_logloss num_class=").append(nClasses);
        }
        sb.append(" max_depth=").append(maxDepth);
        sb.append(" learning_rate=").append(learningRate);
        sb.append(" bagging_fraction=").append(subsample);
        sb.append(" bagging_freq=1");
        sb.append(" feature_fraction=0.8");
        sb.append(" min_gain_to_split=10");
        sb.append(" num_threads=").append(numThreads);
        sb.append(" seed=42");
        sb.append(" verbosity=-1"); // suppress LightGBM logs
        return sb.toString();
    }
}
