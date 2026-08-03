package qupath.ext.celltune.classifier;

import com.microsoft.ml.lightgbm.PredictionType;
import io.github.metarank.lightgbm4j.LGBMBooster;
import io.github.metarank.lightgbm4j.LGBMDataset;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Consumer;
import java.util.stream.IntStream;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Bayesian (TPE) hyperparameter tuner with stratified k-fold cross-validation.
 * <p>
 * Uses Tree-structured Parzen Estimator (TPE) to guide the search toward
 * promising hyperparameter regions. After an initial warm-up period of random
 * trials, each subsequent trial is sampled from a kernel density estimate
 * fitted to the best-performing observations.
 * <p>
 * XGBoost and LightGBM are tuned independently — each model gets its own
 * set of optimal hyperparameters.
 * <p>
 * All training during tuning uses CPU to avoid GPU probe overhead.
 */
public final class HyperparameterTuner {

    private static final Logger logger = LoggerFactory.getLogger(HyperparameterTuner.class);

    /** Default number of search trials per model. */
    public static final int DEFAULT_TRIALS = 20;

    /** Default number of CV folds. */
    public static final int DEFAULT_FOLDS = 5;

    /** Minimum training samples required for tuning. */
    private static final int MIN_SAMPLES = 20;

    // ── Search space bounds ─────────────────────────────────────────────────
    static final int ROUNDS_MIN = 50, ROUNDS_MAX = 500;
    static final int DEPTH_MIN = 2, DEPTH_MAX = 12;
    static final double ETA_MIN = 0.01, ETA_MAX = 0.3;
    static final double SUB_MIN = 0.5, SUB_MAX = 1.0;

    // Transformed space bounds (log for eta)
    private static final double[] LOWER = {ROUNDS_MIN, DEPTH_MIN, Math.log(ETA_MIN), SUB_MIN};
    private static final double[] UPPER = {ROUNDS_MAX, DEPTH_MAX, Math.log(ETA_MAX), SUB_MAX};

    private HyperparameterTuner() {} // utility class

    // ── Result types ────────────────────────────────────────────────────────

    /** A set of hyperparameters for boosted tree models. */
    public record HyperParams(int numRounds, int maxDepth, float eta, float subsample) {
        @Override
        public String toString() {
            return String.format("rounds=%d, depth=%d, eta=%.4f, subsample=%.2f", numRounds, maxDepth, eta, subsample);
        }
    }

    /** Result of a tuning run: best parameters for each model and their CV scores. */
    public record TuningResult(HyperParams xgbParams, double xgbScore, HyperParams lgbParams, double lgbScore) {}

    private record ModelTuneResult(HyperParams params, double score) {}

    // ── Main API ────────────────────────────────────────────────────────────

    /**
     * Tune hyperparameters independently for XGBoost and LightGBM using TPE
     * with stratified k-fold CV.
     *
     * @param flatData  row-major training feature matrix
     * @param labels    float class labels (0-indexed)
     * @param nSamples  number of training samples
     * @param nFeatures number of features per sample
     * @param nClasses  number of classes
     * @param nTrials   number of search iterations per model
     * @param nFolds    number of CV folds (typically 5)
     * @param log       progress callback (may be null)
     * @return best parameters for each model and their scores
     */
    public static TuningResult tune(
            float[] flatData,
            float[] labels,
            int nSamples,
            int nFeatures,
            int nClasses,
            int nTrials,
            int nFolds,
            Consumer<String> log) {
        Consumer<String> out = log != null ? log : s -> {};

        if (nSamples < MIN_SAMPLES) {
            out.accept("Too few samples (" + nSamples + ") for auto-tuning — using defaults");
            HyperParams defaults = new HyperParams(200, 6, 0.1f, 0.8f);
            return new TuningResult(defaults, 0, defaults, 0);
        }

        out.accept(
                "Auto-tuning: " + nTrials + " TPE trials × " + nFolds + " folds per model on " + nSamples + " samples");

        int[] intLabels = new int[nSamples];
        for (int i = 0; i < nSamples; i++) intLabels[i] = (int) labels[i];

        List<int[][]> folds = stratifiedKFold(intLabels, nClasses, nFolds, new Random(42));

        // Tune each model independently
        out.accept("── Tuning XGBoost ──");
        ModelTuneResult xgb = tuneModel(
                "XGBoost",
                flatData,
                labels,
                intLabels,
                nSamples,
                nFeatures,
                nClasses,
                folds,
                nTrials,
                new Random(42),
                out);

        out.accept("── Tuning LightGBM ──");
        ModelTuneResult lgb = tuneModel(
                "LightGBM",
                flatData,
                labels,
                intLabels,
                nSamples,
                nFeatures,
                nClasses,
                folds,
                nTrials,
                new Random(43),
                out);

        out.accept(String.format("Best XGBoost:  %s → F1 = %.4f", xgb.params(), xgb.score()));
        out.accept(String.format("Best LightGBM: %s → F1 = %.4f", lgb.params(), lgb.score()));

        return new TuningResult(xgb.params(), xgb.score(), lgb.params(), lgb.score());
    }

    // ── Per-model TPE tuning loop ───────────────────────────────────────────

    private static ModelTuneResult tuneModel(
            String modelName,
            float[] flatData,
            float[] labels,
            int[] intLabels,
            int nSamples,
            int nFeatures,
            int nClasses,
            List<int[][]> folds,
            int nTrials,
            Random rng,
            Consumer<String> log) {
        TPESampler sampler = new TPESampler(rng);
        HyperParams bestParams = null;
        double bestScore = -1;

        int totalCores = Runtime.getRuntime().availableProcessors();
        int nFolds = folds.size();
        // Parallelize folds when we have enough cores (at least 2 per fold)
        boolean parallelFolds = totalCores >= nFolds * 2;
        int threadsPerFoldXGB = parallelFolds ? Math.max(1, totalCores / nFolds) : totalCores;
        int threadsPerFoldLGB = totalCores; // Always use all cores for LightGBM

        if (parallelFolds) {
            log.accept(String.format(
                    "  Parallel CV: %d folds × %d threads/fold (XGB), %d threads/fold (LGB) (%d cores)",
                    nFolds, threadsPerFoldXGB, threadsPerFoldLGB, totalCores));
        }

        ExecutorService foldPool = parallelFolds ? Executors.newFixedThreadPool(nFolds) : null;

        try {
            for (int trial = 0; trial < nTrials; trial++) {
                HyperParams hp = sampler.suggest();

                double totalScore = 0;
                int validFolds = 0;

                if (parallelFolds) {
                    // Submit all folds in parallel
                    List<Future<Double>> futures = new ArrayList<>(nFolds);
                    for (var fold : folds) {
                        int[] trainIdx = fold[0];
                        int[] testIdx = fold[1];

                        float[] foldTrainData = extractRows(flatData, trainIdx, nFeatures);
                        float[] foldTrainLabels = extractLabels(labels, trainIdx);
                        float[] foldTestData = extractRows(flatData, testIdx, nFeatures);
                        int[] foldTestTruth = extractIntLabels(intLabels, testIdx);

                        futures.add(foldPool.submit(() -> "XGBoost".equals(modelName)
                                ? evaluateXGBoostFold(
                                        foldTrainData,
                                        foldTrainLabels,
                                        trainIdx.length,
                                        foldTestData,
                                        foldTestTruth,
                                        testIdx.length,
                                        nFeatures,
                                        nClasses,
                                        hp,
                                        threadsPerFoldXGB)
                                : evaluateLightGBMFold(
                                        foldTrainData,
                                        foldTrainLabels,
                                        trainIdx.length,
                                        foldTestData,
                                        foldTestTruth,
                                        testIdx.length,
                                        nFeatures,
                                        nClasses,
                                        hp,
                                        threadsPerFoldLGB)));
                    }

                    for (Future<Double> f : futures) {
                        try {
                            double f1 = f.get();
                            if (f1 >= 0) {
                                totalScore += f1;
                                validFolds++;
                            }
                        } catch (InterruptedException | ExecutionException e) {
                            logger.debug("Parallel fold failed: {}", e.getMessage());
                        }
                    }
                } else {
                    // Sequential fallback
                    for (var fold : folds) {
                        int[] trainIdx = fold[0];
                        int[] testIdx = fold[1];

                        float[] foldTrainData = extractRows(flatData, trainIdx, nFeatures);
                        float[] foldTrainLabels = extractLabels(labels, trainIdx);
                        float[] foldTestData = extractRows(flatData, testIdx, nFeatures);
                        int[] foldTestTruth = extractIntLabels(intLabels, testIdx);

                        double f1 = "XGBoost".equals(modelName)
                                ? evaluateXGBoostFold(
                                        foldTrainData,
                                        foldTrainLabels,
                                        trainIdx.length,
                                        foldTestData,
                                        foldTestTruth,
                                        testIdx.length,
                                        nFeatures,
                                        nClasses,
                                        hp,
                                        threadsPerFoldXGB)
                                : evaluateLightGBMFold(
                                        foldTrainData,
                                        foldTrainLabels,
                                        trainIdx.length,
                                        foldTestData,
                                        foldTestTruth,
                                        testIdx.length,
                                        nFeatures,
                                        nClasses,
                                        hp,
                                        threadsPerFoldLGB);

                        if (f1 >= 0) {
                            totalScore += f1;
                            validFolds++;
                        }
                    }
                }

                double meanScore = validFolds > 0 ? totalScore / validFolds : 0;
                sampler.observe(hp, meanScore);

                String marker = meanScore > bestScore ? " ★" : "";
                log.accept(
                        String.format("  Trial %2d/%d: %s → F1 = %.4f%s", trial + 1, nTrials, hp, meanScore, marker));

                if (meanScore > bestScore) {
                    bestScore = meanScore;
                    bestParams = hp;
                }
            }
        } finally {
            if (foldPool != null) {
                foldPool.shutdown();
            }
        }

        if (bestParams == null) {
            bestParams = new HyperParams(200, 6, 0.1f, 0.8f);
        }

        return new ModelTuneResult(bestParams, bestScore);
    }

    // ── TPE Sampler ─────────────────────────────────────────────────────────

    /**
     * Tree-structured Parzen Estimator (TPE) sampler.
     * After a warm-up period of random trials, suggests new points by:
     * <ol>
     *   <li>Splitting observed points into "good" (top γ fraction) and "bad"</li>
     *   <li>Fitting independent 1D Gaussian KDEs to each group</li>
     *   <li>Sampling candidates from the "good" KDE</li>
     *   <li>Selecting the candidate that maximises l(x)/g(x)</li>
     * </ol>
     */
    private static final class TPESampler {

        private static final double GAMMA = 0.25; // top 25% = "good"
        private static final int N_CANDIDATES = 24;
        private static final int WARM_UP = 5;
        private static final int N_DIMS = 4;

        private final Random rng;
        private final List<double[]> history = new ArrayList<>();
        private final List<Double> scores = new ArrayList<>();

        TPESampler(Random rng) {
            this.rng = rng;
        }

        void observe(HyperParams hp, double score) {
            history.add(toTransformed(hp));
            scores.add(score);
        }

        HyperParams suggest() {
            if (history.size() < WARM_UP) {
                return sampleUniform();
            }
            return tpeSuggest();
        }

        private HyperParams tpeSuggest() {
            int n = history.size();
            int nGood = Math.max(1, (int) (n * GAMMA));

            // Sort indices by score descending (higher F1 is better)
            Integer[] sortedIdx = IntStream.range(0, n)
                    .boxed()
                    .sorted((a, b) -> Double.compare(scores.get(b), scores.get(a)))
                    .toArray(Integer[]::new);

            List<double[]> good = new ArrayList<>();
            List<double[]> bad = new ArrayList<>();
            for (int i = 0; i < sortedIdx.length; i++) {
                (i < nGood ? good : bad).add(history.get(sortedIdx[i]));
            }
            if (bad.isEmpty()) bad.add(good.getLast());

            // Sample candidates from good KDE, pick best l(x)/g(x)
            double bestRatio = Double.NEGATIVE_INFINITY;
            double[] bestCandidate = null;

            for (int c = 0; c < N_CANDIDATES; c++) {
                double[] candidate = sampleFromKDE(good);
                double lx = evaluateKDE(candidate, good);
                double gx = evaluateKDE(candidate, bad);
                double ratio = lx / (gx + 1e-12);
                if (ratio > bestRatio) {
                    bestRatio = ratio;
                    bestCandidate = candidate;
                }
            }

            return fromTransformed(bestCandidate);
        }

        /** Sample a point from a Gaussian KDE fitted to the given observations. */
        private double[] sampleFromKDE(List<double[]> points) {
            double[] sample = new double[N_DIMS];
            for (int d = 0; d < N_DIMS; d++) {
                double[] center = points.get(rng.nextInt(points.size()));
                double bw = silvermanBW(points, d);
                sample[d] = center[d] + rng.nextGaussian() * bw;
                sample[d] = Math.max(LOWER[d], Math.min(UPPER[d], sample[d]));
            }
            return sample;
        }

        /** Evaluate the KDE density at a point (product of per-dimension densities). */
        private double evaluateKDE(double[] x, List<double[]> points) {
            double logDensity = 0;
            for (int d = 0; d < N_DIMS; d++) {
                double bw = silvermanBW(points, d);
                double density = 0;
                for (double[] pt : points) {
                    double z = (x[d] - pt[d]) / bw;
                    density += Math.exp(-0.5 * z * z);
                }
                density /= (points.size() * bw * Math.sqrt(2 * Math.PI));
                logDensity += Math.log(Math.max(density, 1e-300));
            }
            return Math.exp(logDensity);
        }

        /** Silverman's rule of thumb bandwidth: h = 1.06 σ n^{-1/5}. */
        private double silvermanBW(List<double[]> points, int dim) {
            int n = points.size();
            if (n < 2) return (UPPER[dim] - LOWER[dim]) / 4.0;

            double mean = 0;
            for (double[] p : points) mean += p[dim];
            mean /= n;

            double variance = 0;
            for (double[] p : points) {
                double diff = p[dim] - mean;
                variance += diff * diff;
            }
            variance /= (n - 1);
            double std = Math.sqrt(variance);
            if (std < 1e-10) std = (UPPER[dim] - LOWER[dim]) / 4.0;

            return 1.06 * std * Math.pow(n, -0.2);
        }

        private HyperParams sampleUniform() {
            int rounds = ROUNDS_MIN + rng.nextInt(ROUNDS_MAX - ROUNDS_MIN + 1);
            rounds = ((rounds + 5) / 10) * 10;
            int depth = DEPTH_MIN + rng.nextInt(DEPTH_MAX - DEPTH_MIN + 1);
            float eta =
                    (float) Math.exp(Math.log(ETA_MIN) + rng.nextDouble() * (Math.log(ETA_MAX) - Math.log(ETA_MIN)));
            float sub = (float) (SUB_MIN + rng.nextDouble() * (SUB_MAX - SUB_MIN));
            return new HyperParams(rounds, depth, eta, sub);
        }

        private static double[] toTransformed(HyperParams hp) {
            return new double[] {hp.numRounds(), hp.maxDepth(), Math.log(hp.eta()), hp.subsample()};
        }

        private HyperParams fromTransformed(double[] x) {
            int rounds = (int) Math.round(x[0]);
            rounds = Math.max(ROUNDS_MIN, Math.min(ROUNDS_MAX, ((rounds + 5) / 10) * 10));
            int depth = Math.max(DEPTH_MIN, Math.min(DEPTH_MAX, (int) Math.round(x[1])));
            float eta = (float) Math.max(ETA_MIN, Math.min(ETA_MAX, Math.exp(x[2])));
            float sub = (float) Math.max(SUB_MIN, Math.min(SUB_MAX, x[3]));
            return new HyperParams(rounds, depth, eta, sub);
        }
    }

    // ── Stratified k-fold ───────────────────────────────────────────────────

    private static List<int[][]> stratifiedKFold(int[] labels, int nClasses, int k, Random rng) {
        List<List<Integer>> classGroups = new ArrayList<>();
        for (int c = 0; c < nClasses; c++) classGroups.add(new ArrayList<>());
        for (int i = 0; i < labels.length; i++) classGroups.get(labels[i]).add(i);

        for (var group : classGroups) Collections.shuffle(group, rng);

        List<List<Integer>> foldLists = new ArrayList<>();
        for (int f = 0; f < k; f++) foldLists.add(new ArrayList<>());

        for (var group : classGroups) {
            for (int i = 0; i < group.size(); i++) {
                foldLists.get(i % k).add(group.get(i));
            }
        }

        List<int[][]> folds = new ArrayList<>();
        for (int f = 0; f < k; f++) {
            List<Integer> testList = foldLists.get(f);
            List<Integer> trainList = new ArrayList<>();
            for (int g = 0; g < k; g++) {
                if (g != f) trainList.addAll(foldLists.get(g));
            }
            folds.add(new int[][] {
                trainList.stream().mapToInt(Integer::intValue).toArray(),
                testList.stream().mapToInt(Integer::intValue).toArray()
            });
        }

        return folds;
    }

    // ── XGBoost fold evaluation ─────────────────────────────────────────────

    private static double evaluateXGBoostFold(
            float[] trainData,
            float[] trainLabels,
            int trainSize,
            float[] testData,
            int[] testTruth,
            int testSize,
            int nFeatures,
            int nClasses,
            HyperParams hp,
            int nThreads) {

        DMatrix trainMat = null;
        DMatrix testMat = null;
        Booster booster = null;
        try {
            trainMat = new DMatrix(trainData, trainSize, nFeatures, Float.NaN);
            trainMat.setLabel(trainLabels);

            Map<String, Object> params = new LinkedHashMap<>();
            params.put("max_depth", hp.maxDepth());
            params.put("eta", (double) hp.eta());
            params.put("subsample", (double) hp.subsample());
            params.put("colsample_bytree", 0.8);
            params.put("objective", nClasses == 2 ? "binary:logistic" : "multi:softprob");
            params.put("eval_metric", nClasses == 2 ? "logloss" : "mlogloss");
            params.put("device", "cpu");
            params.put("tree_method", "hist");
            params.put("nthread", nThreads);
            params.put("seed", 42);
            params.put("verbosity", 0);
            if (nClasses > 2) params.put("num_class", nClasses);

            booster = XGBoost.train(trainMat, params, hp.numRounds(), new LinkedHashMap<>(), null, null);

            testMat = new DMatrix(testData, testSize, nFeatures, Float.NaN);
            float[][] preds = booster.predict(testMat);

            int[] predClasses = toPredictedClasses(preds, testSize, nClasses);
            return macroF1(predClasses, testTruth, nClasses);

        } catch (Exception e) {
            logger.debug("XGBoost CV fold failed: {}", e.getMessage());
            return -1;
        } finally {
            // One booster per fold per trial — 100 per tuned model. Each holds a native handle
            // that Booster.finalize is not a reliable path for releasing on JDK 25.
            try {
                if (booster != null) booster.dispose();
            } catch (Exception ignore) {
            }
            try {
                if (trainMat != null) trainMat.dispose();
            } catch (Exception ignore) {
            }
            try {
                if (testMat != null) testMat.dispose();
            } catch (Exception ignore) {
            }
        }
    }

    // ── LightGBM fold evaluation ────────────────────────────────────────────

    private static double evaluateLightGBMFold(
            float[] trainData,
            float[] trainLabels,
            int trainSize,
            float[] testData,
            int[] testTruth,
            int testSize,
            int nFeatures,
            int nClasses,
            HyperParams hp,
            int nThreads) {

        LGBMDataset dataset = null;
        LGBMBooster booster = null;
        try {
            dataset = LGBMDataset.createFromMat(trainData, trainSize, nFeatures, true, "", null);
            dataset.setField("label", trainLabels);

            StringBuilder sb = new StringBuilder();
            if (nClasses == 2) {
                sb.append("objective=binary metric=binary_logloss");
            } else {
                sb.append("objective=multiclass metric=multi_logloss num_class=")
                        .append(nClasses);
            }
            sb.append(" max_depth=").append(hp.maxDepth());
            sb.append(" learning_rate=").append(hp.eta());
            sb.append(" bagging_fraction=").append(hp.subsample());
            sb.append(" bagging_freq=1");
            sb.append(" feature_fraction=0.8");
            sb.append(" num_threads=").append(nThreads);
            sb.append(" seed=42");
            sb.append(" verbosity=-1");

            booster = LGBMBooster.create(dataset, sb.toString());
            for (int i = 0; i < hp.numRounds(); i++) {
                booster.updateOneIter();
            }

            double[] rawPreds =
                    booster.predictForMat(testData, testSize, nFeatures, true, PredictionType.C_API_PREDICT_NORMAL);

            int[] predClasses = toLGBPredictedClasses(rawPreds, testSize, nClasses);
            return macroF1(predClasses, testTruth, nClasses);

        } catch (Exception e) {
            logger.debug("LightGBM CV fold failed: {}", e.getMessage());
            return -1;
        } finally {
            try {
                if (booster != null) booster.close();
            } catch (Exception ignore) {
            }
            try {
                if (dataset != null) dataset.close();
            } catch (Exception ignore) {
            }
        }
    }

    // ── Prediction helpers ──────────────────────────────────────────────────

    private static int[] toPredictedClasses(float[][] preds, int n, int nClasses) {
        int[] result = new int[n];
        if (nClasses == 2 && preds[0].length == 1) {
            for (int i = 0; i < n; i++) {
                result[i] = preds[i][0] >= 0.5f ? 1 : 0;
            }
        } else {
            for (int i = 0; i < n; i++) {
                int best = 0;
                for (int c = 1; c < nClasses; c++) {
                    if (preds[i][c] > preds[i][best]) best = c;
                }
                result[i] = best;
            }
        }
        return result;
    }

    private static int[] toLGBPredictedClasses(double[] raw, int n, int nClasses) {
        int[] result = new int[n];
        if (nClasses == 2 && raw.length == n) {
            for (int i = 0; i < n; i++) {
                result[i] = raw[i] >= 0.5 ? 1 : 0;
            }
        } else {
            for (int i = 0; i < n; i++) {
                int best = 0;
                for (int c = 1; c < nClasses; c++) {
                    if (raw[i * nClasses + c] > raw[i * nClasses + best]) best = c;
                }
                result[i] = best;
            }
        }
        return result;
    }

    // ── Scoring ─────────────────────────────────────────────────────────────

    private static double macroF1(int[] predicted, int[] truth, int nClasses) {
        int[] tp = new int[nClasses];
        int[] fp = new int[nClasses];
        int[] fn = new int[nClasses];

        for (int i = 0; i < predicted.length; i++) {
            if (predicted[i] == truth[i]) {
                tp[predicted[i]]++;
            } else {
                fp[predicted[i]]++;
                fn[truth[i]]++;
            }
        }

        double sumF1 = 0;
        int counted = 0;
        for (int c = 0; c < nClasses; c++) {
            int support = tp[c] + fn[c];
            if (support == 0) continue;
            double precision = (tp[c] + fp[c]) > 0 ? (double) tp[c] / (tp[c] + fp[c]) : 0;
            double recall = (double) tp[c] / (tp[c] + fn[c]);
            double f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;
            sumF1 += f1;
            counted++;
        }

        return counted > 0 ? sumF1 / counted : 0;
    }

    // ── Data extraction helpers ─────────────────────────────────────────────

    private static float[] extractRows(float[] flatData, int[] indices, int nFeatures) {
        float[] result = new float[indices.length * nFeatures];
        for (int i = 0; i < indices.length; i++) {
            System.arraycopy(flatData, indices[i] * nFeatures, result, i * nFeatures, nFeatures);
        }
        return result;
    }

    private static float[] extractLabels(float[] labels, int[] indices) {
        float[] result = new float[indices.length];
        for (int i = 0; i < indices.length; i++) result[i] = labels[indices[i]];
        return result;
    }

    private static int[] extractIntLabels(int[] labels, int[] indices) {
        int[] result = new int[indices.length];
        for (int i = 0; i < indices.length; i++) result[i] = labels[indices[i]];
        return result;
    }
}
