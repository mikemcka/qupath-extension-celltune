package qupath.ext.celltune.classifier;

import java.io.*;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.celltune.util.TrainingThreads;

/**
 * Wraps XGBoost4J training and prediction behind a simple interface.
 * <p>
 * Supports both binary ({@code binary:logistic}) and multiclass
 * ({@code multi:softprob}) objectives. Probability vectors are always
 * returned as {@code float[nClasses]}.
 */
public class XGBoostModel {

    private static final Logger logger = LoggerFactory.getLogger(XGBoostModel.class);

    private Booster booster;
    private int nClasses;
    private List<String> classNames;
    private List<String> featureNames;
    private String lastDevice = "unknown";

    // ── Training ────────────────────────────────────────────────────────────────

    /**
     * Train a new XGBoost model.
     *
     * @param flatData     row-major feature matrix (nSamples × nFeatures)
     * @param labels       integer class labels (0-indexed)
     * @param nSamples     number of training samples
     * @param nFeatures    number of features per sample
     * @param classNames   ordered list of class names
     * @param featureNames ordered list of feature names
     * @param numRounds    boosting iterations
     * @param maxDepth     max tree depth
     * @param eta          learning rate
     * @param subsample    row subsampling ratio per round
     * @param colsample    column subsampling ratio per tree ({@link #DEFAULT_COLSAMPLE} when untuned)
     * @throws XGBoostError if training fails
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
            float eta,
            float subsample,
            float colsample)
            throws XGBoostError {

        this.nClasses = classNames.size();
        this.classNames = List.copyOf(classNames);
        this.featureNames = List.copyOf(featureNames);

        DMatrix trainMat = new DMatrix(flatData, nSamples, nFeatures, Float.NaN);
        try {
            trainMat.setLabel(labels);

            Map<String, Object> params =
                    buildParams(nClasses, maxDepth, eta, subsample, colsample, TrainingThreads.total());
            // No watches: XGBoost evaluates every watched matrix once per round, and a "train"
            // watch here scored the full training matrix on all N rounds with nothing consuming
            // the result (no callback is passed, and evaluation never feeds back into boosting).
            // Dropping it leaves the model bit-identical.
            Map<String, DMatrix> watches = new LinkedHashMap<>();

            // device=cpu and tree_method=hist come from buildParams. This build pins the CPU-only
            // xgboost4j artifact (see build.gradle.kts: "ml.dmlc:xgboost4j_2.13"), whose CUDA
            // kernels are NOT shipped — device=cuda is silently ignored by XGBoost and falls back
            // to CPU without throwing, which previously caused us to mis-report GPU. If/when a
            // -gpu artifact is wired in, restore the probe-and-fallback logic.
            booster = XGBoost.train(trainMat, params, numRounds, watches, null, null);
            logger.info(
                    "XGBoost training: CPU — {} samples, {} features, {} classes, {} rounds",
                    nSamples,
                    nFeatures,
                    nClasses,
                    numRounds);

            this.lastDevice = "CPU";
        } finally {
            trainMat.dispose();
        }

        // Embed metadata for later serialisation
        // XGBoost rejects special chars (µ, ^, :, /, etc.) in feature names
        String[] safeNames =
                featureNames.stream().map(XGBoostModel::sanitiseFeatureName).toArray(String[]::new);
        booster.setFeatureNames(safeNames);
        booster.setAttr("class_names", String.join(",", classNames));

        logger.info("XGBoost training complete ({})", lastDevice);
    }

    /** @return the device used for the last training run */
    public String getLastDevice() {
        return lastDevice;
    }

    // ── Early Stopping ──────────────────────────────────────────────────────────

    /**
     * Outcome of a round search.
     *
     * @param bestRounds the round count with the lowest validation loss (1-indexed)
     * @param bestModel  the booster serialised as it stood at {@code bestRounds}, or {@code null}
     *                   if snapshotting was disabled. Boosting is sequential and seeded, so this
     *                   is the same model a fresh {@code bestRounds}-round fit on the same fold
     *                   would produce — the rounds that followed only appended trees.
     */
    record RoundSearch(int bestRounds, byte[] bestModel) {}

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
            float eta,
            float subsample,
            int patience,
            Consumer<String> log)
            throws Exception {
        return searchRounds(
                        trainData,
                        trainLabels,
                        trainSize,
                        valData,
                        valLabels,
                        valSize,
                        nFeatures,
                        nClasses,
                        maxRounds,
                        maxDepth,
                        eta,
                        subsample,
                        patience,
                        false,
                        log)
                .bestRounds();
    }

    /**
     * Round search that also hands back the model at the winning round.
     * <p>
     * Without this the search discards its booster and returns only a count, so the caller has to
     * rebuild the identical model from scratch on the identical fold — which on a wide
     * multi-class panel costs as much as the search itself. Snapshotting captures it instead.
     * <p>
     * The snapshot is taken only when the validation loss improves, which is most rounds early on
     * and rare later. {@code toByteArray} is not free on a large multi-class model, so
     * {@code snapshot} is a parameter rather than always-on: the caller decides whether a
     * downstream fit is actually going to reuse it.
     */
    static RoundSearch searchRounds(
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
            float eta,
            float subsample,
            int patience,
            boolean snapshot,
            Consumer<String> log)
            throws Exception {

        DMatrix trainMat = null;
        DMatrix valMat = null;
        Booster booster = null;
        try {
            trainMat = new DMatrix(trainData, trainSize, nFeatures, Float.NaN);
            trainMat.setLabel(trainLabels);
            valMat = new DMatrix(valData, valSize, nFeatures, Float.NaN);
            valMat.setLabel(valLabels);

            Map<String, Object> params = buildParams(nClasses, maxDepth, eta, subsample);
            params.put("verbosity", 0);

            booster = XGBoost.train(trainMat, params, 1, new LinkedHashMap<>(), null, null);

            String evalStr = booster.evalSet(new DMatrix[] {valMat}, new String[] {"val"}, 0);
            double bestLoss = parseEvalMetric(evalStr);
            int bestRound = 0;
            byte[] bestModel = snapshot ? booster.toByteArray() : null;
            long snapshotNanos = 0;
            int snapshots = snapshot ? 1 : 0;

            for (int round = 1; round < maxRounds; round++) {
                booster.update(trainMat, round);
                evalStr = booster.evalSet(new DMatrix[] {valMat}, new String[] {"val"}, round);
                double loss = parseEvalMetric(evalStr);

                if (loss < bestLoss) {
                    bestLoss = loss;
                    bestRound = round;
                    if (snapshot) {
                        long t0 = System.nanoTime();
                        bestModel = booster.toByteArray();
                        snapshotNanos += System.nanoTime() - t0;
                        snapshots++;
                    }
                }
                if (round - bestRound >= patience) break;
            }

            int actualRounds = bestRound + 1;
            log.accept(String.format(
                    "XGBoost early stopping: best round %d/%d (val loss: %.6f)", actualRounds, maxRounds, bestLoss));
            if (snapshot) {
                log.accept(String.format(
                        "  kept the round-%d model (%d snapshots, %.2fs, %,d KB) \u2014 saves retraining it",
                        actualRounds, snapshots, snapshotNanos / 1e9, bestModel == null ? 0 : bestModel.length / 1024));
            }
            return new RoundSearch(actualRounds, bestModel);

        } finally {
            try {
                if (booster != null) booster.dispose();
            } catch (Exception ignore) {
            }
            try {
                if (trainMat != null) trainMat.dispose();
            } catch (Exception ignore) {
            }
            try {
                if (valMat != null) valMat.dispose();
            } catch (Exception ignore) {
            }
        }
    }

    /** Parse the last metric value from an XGBoost evalSet string. */
    private static double parseEvalMetric(String evalStr) {
        int lastColon = evalStr.lastIndexOf(':');
        if (lastColon < 0) return Double.MAX_VALUE;
        String valStr = evalStr.substring(lastColon + 1).trim();
        try {
            return Double.parseDouble(valStr);
        } catch (NumberFormatException e) {
            return Double.MAX_VALUE;
        }
    }

    // ── Prediction ──────────────────────────────────────────────────────────────

    /**
     * Predict class probabilities for multiple cells.
     *
     * @param flatData  row-major feature matrix (nSamples × nFeatures)
     * @param nSamples  number of samples
     * @param nFeatures number of features
     * @return probability matrix [nSamples][nClasses]
     */
    public float[][] predictProba(float[] flatData, int nSamples, int nFeatures) throws XGBoostError {

        DMatrix predMat = new DMatrix(flatData, nSamples, nFeatures, Float.NaN);
        float[][] rawPreds;
        try {
            rawPreds = booster.predict(predMat);
        } finally {
            predMat.dispose();
        }

        // binary:logistic returns [n][1]; multi:softprob returns [n][nClasses]
        if (nClasses == 2 && rawPreds[0].length == 1) {
            float[][] expanded = new float[nSamples][2];
            for (int i = 0; i < nSamples; i++) {
                expanded[i][1] = rawPreds[i][0];
                expanded[i][0] = 1f - rawPreds[i][0];
            }
            return expanded;
        }
        return rawPreds;
    }

    /**
     * Predict the single best class index for each sample.
     *
     * @param flatData  row-major feature matrix
     * @param nSamples  number of samples
     * @param nFeatures number of features
     * @return array of predicted class indices
     */
    public int[] predict(float[] flatData, int nSamples, int nFeatures) throws XGBoostError {

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
     * Compute per-class mean absolute SHAP values using XGBoost's native
     * TreeSHAP implementation ({@code predContrib=true}).
     * <p>
     * For binary models ({@code binary:logistic}) SHAP values are for the
     * decision margin; the same values are reflected for both classes.
     * For multiclass ({@code multi:softprob}) values are class-specific.
     *
     * @param flatData  row-major feature matrix (nSamples × nFeatures)
     * @param nSamples  number of samples
     * @param nFeatures number of features
     * @return mean absolute SHAP matrix [nClasses][nFeatures]
     * @throws XGBoostError if prediction fails
     */
    public double[][] computeMeanAbsShap(float[] flatData, int nSamples, int nFeatures) throws XGBoostError {
        DMatrix dmat = new DMatrix(flatData, nSamples, nFeatures, Float.NaN);
        try {
            // predictContrib → TreeSHAP contributions (public API in XGBoost4J 3.x)
            // Binary:     raw[nSamples][nFeatures + 1]                   (last = bias)
            // Multiclass: raw[nSamples][nClasses * (nFeatures + 1)]       (class-major)
            float[][] raw = booster.predictContrib(dmat, 0);

            double[][] result = new double[nClasses][nFeatures];
            int stride = nFeatures + 1; // +1 for the bias term

            if (nClasses == 2 && raw[0].length == stride) {
                // Binary: contributions are for the positive class (class 1)
                for (int i = 0; i < nSamples; i++) {
                    for (int f = 0; f < nFeatures; f++) {
                        double s = Math.abs(raw[i][f]);
                        result[0][f] += s;
                        result[1][f] += s;
                    }
                }
            } else {
                // Multiclass: class k, feature f → raw[i][k*(nFeatures+1) + f]
                for (int i = 0; i < nSamples; i++) {
                    for (int c = 0; c < nClasses; c++) {
                        for (int f = 0; f < nFeatures; f++) {
                            result[c][f] += Math.abs(raw[i][stride * c + f]);
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
        } finally {
            dmat.dispose();
        }
    }

    // ── Serialisation ───────────────────────────────────────────────────────────

    /**
     * Serialise the trained model to a byte array.
     *
     * @return raw model bytes
     * @throws XGBoostError if serialisation fails
     */
    public byte[] toBytes() throws XGBoostError {
        return booster.toByteArray();
    }

    /**
     * Load a model from a byte array.
     *
     * @param bytes       raw model bytes
     * @param classNames  ordered class names
     * @param featureNames ordered feature names
     * @throws XGBoostError if loading fails
     */
    public void loadFromBytes(byte[] bytes, List<String> classNames, List<String> featureNames)
            throws XGBoostError, IOException {

        this.classNames = List.copyOf(classNames);
        this.featureNames = List.copyOf(featureNames);
        this.nClasses = classNames.size();

        // XGBoost4J loadModel expects an InputStream
        booster = XGBoost.loadModel(new ByteArrayInputStream(bytes));
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

    /**
     * Replace characters that XGBoost4J rejects in feature names.
     * XGBoost forbids: [ ] < > , " and any non-ASCII.
     */
    static String sanitiseFeatureName(String name) {
        // Replace known problematic chars with ASCII equivalents
        String s = name.replace("\u00b5", "u") // micro sign µ
                .replace("\u03bc", "u") // greek mu μ
                .replace("^", "_pow_")
                .replace(":", "_")
                .replace("/", "_per_")
                .replace(" ", "_")
                .replace("[", "_")
                .replace("]", "_")
                .replace("<", "_lt_")
                .replace(">", "_gt_")
                .replace(",", "_")
                .replace("\"", "_")
                .replace(".", "_")
                .replace("{", "_")
                .replace("}", "_");
        // Strip any remaining non-ASCII
        StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            sb.append(c >= 0x20 && c <= 0x7E ? c : '_');
        }
        return sb.toString();
    }

    /**
     * Column-subsampling rate used unless a caller supplies one.
     * <p>
     * This was a bare literal until {@code colsample_bytree} became a tuned dimension. Naming it
     * keeps every untuned path — the early-stopping round search, the metrics copies, a run with
     * auto-tune switched off — on exactly the value they used before, so only a deliberate search
     * result can move it.
     */
    static final float DEFAULT_COLSAMPLE = 0.8f;

    private static Map<String, Object> buildParams(int nClasses, int maxDepth, float eta, float subsample) {
        return buildParams(nClasses, maxDepth, eta, subsample, DEFAULT_COLSAMPLE, TrainingThreads.total());
    }

    /**
     * The single definition of how an XGBoost booster in this extension is configured.
     * <p>
     * {@link HyperparameterTuner} must call this rather than assembling its own map. Its
     * hand-built copy was equivalent right up until {@code max_bin} was added here — at which
     * point cross-validation would have scored 256-bin boosters and handed the winner to a
     * 64-bin one, exactly the bug the LightGBM path already had with
     * {@code min_gain_to_split}. Model-affecting parameters belong here, where there is one of
     * them; {@code verbosity} stays at the call sites because it does not touch the model.
     *
     * @param nThreads thread budget for this booster; the tuner divides the total across its
     *                 concurrently-evaluated folds rather than letting each request every core
     */
    static Map<String, Object> buildParams(int nClasses, int maxDepth, float eta, float subsample, int nThreads) {
        return buildParams(nClasses, maxDepth, eta, subsample, DEFAULT_COLSAMPLE, nThreads);
    }

    /**
     * @param colsample per-tree column-sampling rate. A searched dimension: on a wide panel the
     *                  fraction of features each tree may look at moves accuracy at least as much
     *                  as tree depth does, so the tuner varies it and every untuned path passes
     *                  {@link #DEFAULT_COLSAMPLE}.
     */
    static Map<String, Object> buildParams(
            int nClasses, int maxDepth, float eta, float subsample, float colsample, int nThreads) {

        Map<String, Object> p = new LinkedHashMap<>();
        p.put("max_depth", maxDepth);
        p.put("eta", (double) eta);
        p.put("subsample", (double) subsample);
        p.put("colsample_bytree", (double) colsample);
        p.put("objective", nClasses == 2 ? "binary:logistic" : "multi:softprob");
        p.put("eval_metric", nClasses == 2 ? "logloss" : "mlogloss");
        p.put("nthread", nThreads);
        p.put("seed", 42);
        if (nClasses > 2) p.put("num_class", nClasses);
        // This build pins the CPU-only xgboost4j artifact, and hist is what every measurement in
        // this extension assumes. Both are model-affecting, so they live here rather than being
        // re-stated at each call site.
        p.put("device", "cpu");
        p.put("tree_method", "hist");
        int bins = maxBin;
        if (bins > 0) p.put("max_bin", bins);
        return p;
    }

    // ── Histogram resolution ────────────────────────────────────────────────────

    /** 0 means "leave XGBoost's default" (256), which is the behaviour this shipped with. */
    private static volatile int maxBin = 0;

    /**
     * Sets the number of histogram bins {@code tree_method=hist} builds per feature.
     * <p>
     * This is the one remaining lever with real leverage on training time, and it is opt-in
     * because it <b>changes the model</b>. After the phase-18 work XGBoost is ~87% of a training
     * run and a field log confirmed the cost is tree construction, not evaluation — and the split
     * search inside that scales with the bin count. Measured by {@code XGBoostTuningBenchmark} at
     * the real field shape (8,646 rows, 1,886 features, 35 classes, 16 cores):
     * <pre>
     *   max_bin=256 (default)   68.0 s   1.00x
     *   max_bin=128             33.9 s   2.01x
     *   max_bin=64              26.5 s   2.57x
     * </pre>
     * Coarser bins are also a form of regularisation, so accuracy does not simply degrade — on
     * that synthetic fixture 64 bins scored marginally <em>higher</em>. Do not read that as a free
     * win: the fixture's signal structure is not real marker data, and only the timing ratios
     * transfer reliably. Anyone lowering this should diff the predicted class column on their own
     * data first.
     * <p>
     * Applies to the round search and the final fit alike, so the round count chosen by early
     * stopping always matches the model that gets built.
     *
     * @param bins bins per feature; {@code 0} (or negative) restores XGBoost's default of 256.
     *             XGBoost requires at least 2.
     */
    public static void setMaxBin(int bins) {
        maxBin = bins <= 0 ? 0 : Math.max(2, bins);
    }

    /** @return the configured bin count, or {@code 0} when XGBoost's default is in use */
    public static int getMaxBin() {
        return maxBin;
    }
}
