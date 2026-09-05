package qupath.ext.spclassify.classifier;

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
import qupath.ext.spclassify.util.BackgroundExecutors;
import qupath.ext.spclassify.util.TrainingThreads;

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
 */
public final class HyperparameterTuner {

    private static final Logger logger = LoggerFactory.getLogger(HyperparameterTuner.class);

    /** Default number of search trials per model. */
    public static final int DEFAULT_TRIALS = 20;

    /** Default number of CV folds. */
    public static final int DEFAULT_FOLDS = 5;

    /** Minimum training samples required for tuning. */
    private static final int MIN_SAMPLES = 20;

    /** Fewest folds a cross-validation can be reduced to and still be one. */
    private static final int MIN_FOLDS = 2;

    // ── Search space bounds ─────────────────────────────────────────────────
    static final int ROUNDS_MIN = 50, ROUNDS_MAX = 500;
    static final int DEPTH_MIN = 2, DEPTH_MAX = 12;
    static final double ETA_MIN = 0.01, ETA_MAX = 0.3;
    static final double SUB_MIN = 0.5, SUB_MAX = 1.0;
    // Column sampling. Bottoms out well below the row-sampling floor because the panels this runs
    // on are wide and highly correlated — at ~1,900 features, showing each tree a third of them is
    // a reasonable candidate, where sampling a third of the *rows* would not be.
    static final double COL_MIN = 0.3, COL_MAX = 1.0;

    // Transformed space bounds (log for eta)
    private static final double[] LOWER = {ROUNDS_MIN, DEPTH_MIN, Math.log(ETA_MIN), SUB_MIN, COL_MIN};
    private static final double[] UPPER = {ROUNDS_MAX, DEPTH_MAX, Math.log(ETA_MAX), SUB_MAX, COL_MAX};

    private HyperparameterTuner() {} // utility class

    // ── Result types ────────────────────────────────────────────────────────

    /** A set of hyperparameters for boosted tree models. */
    public record HyperParams(int numRounds, int maxDepth, float eta, float subsample, float colsample) {
        @Override
        public String toString() {
            return String.format(
                    "rounds=%d, depth=%d, eta=%.4f, subsample=%.2f, colsample=%.2f",
                    numRounds, maxDepth, eta, subsample, colsample);
        }
    }

    /** The parameters a run uses when the search is skipped or produces nothing usable. */
    static HyperParams defaults() {
        return new HyperParams(200, 6, 0.1f, 0.8f, XGBoostModel.DEFAULT_COLSAMPLE);
    }

    /** Result of a tuning run: best parameters for each model and their CV scores. */
    public record TuningResult(HyperParams xgbParams, double xgbScore, HyperParams lgbParams, double lgbScore) {}

    private record ModelTuneResult(HyperParams params, double score) {}

    /**
     * A resampled fold-training set: row-major rows, one label per row, and the row count.
     * <p>
     * Package-private, like {@link FoldResampler}: both are handed straight between
     * {@code DualModelClassifier} and this class and never escape the package. Publishing a record
     * that wraps caller-owned arrays would mean either defensive copies of a matrix sized in
     * hundreds of MB, or a documented aliasing contract — neither worth it for a two-class seam.
     */
    record Resampled(float[] data, float[] labels, int size) {}

    /**
     * Class-balances ONE fold's training rows. Applied during fold preparation, never to the whole
     * dataset before splitting.
     * <p>
     * The distinction is not cosmetic. SMOTE builds each synthetic row by interpolating between two
     * real rows of the same class, so a synthetic row generated <em>before</em> the split can land
     * in the fold being scored while both of its parents sit in the fold being trained on — the
     * model is then graded on points it can reconstruct from what it memorised. The inflation is
     * worst exactly where macro-F1 is most sensitive (a rare class padded from tens of real rows to
     * thousands of synthetic ones still counts once in the average), and it is not uniform across
     * the search space: it rewards whichever settings fit the minority neighbourhoods tightest,
     * which is the very axis being searched. So it does not merely flatter the score, it picks the
     * wrong winner.
     * <p>
     * Same rule the early-stopping split and the train/val metrics split in
     * {@code DualModelClassifier} already follow: score against real rows only.
     */
    @FunctionalInterface
    interface FoldResampler {
        /**
         * @param trainData  row-major training rows for one fold
         * @param trainLabels one label per row
         * @param trainSize  number of rows
         * @return the balanced training set for that fold
         */
        Resampled resample(float[] trainData, float[] trainLabels, int trainSize);
    }

    /**
     * One materialised CV fold: the rows to train on (already balanced, if a {@link FoldResampler}
     * was supplied) and the untouched real rows to score against.
     */
    private record PreparedFold(
            float[] trainData, float[] trainLabels, int trainSize, float[] testData, int[] testTruth, int testSize) {}

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
        return tune(flatData, labels, nSamples, nFeatures, nClasses, nTrials, nFolds, null, null, null, log);
    }

    /**
     * @param xgbFixedRounds round count to hold fixed for XGBoost instead of searching over it,
     *                       or {@code null} to search. Pass the value early stopping chose: it
     *                       measured the round count against a real validation fold, which a
     *                       20-trial search over a 50–500 range cannot match, and leaving both to
     *                       decide meant the round search ran and was then discarded.
     * @param lgbFixedRounds the same for LightGBM
     */
    public static TuningResult tune(
            float[] flatData,
            float[] labels,
            int nSamples,
            int nFeatures,
            int nClasses,
            int nTrials,
            int nFolds,
            Integer xgbFixedRounds,
            Integer lgbFixedRounds,
            Consumer<String> log) {
        return tune(
                flatData,
                labels,
                nSamples,
                nFeatures,
                nClasses,
                nTrials,
                nFolds,
                xgbFixedRounds,
                lgbFixedRounds,
                null,
                log);
    }

    /**
     * @param resampler class balancing to apply inside each fold's training rows, or {@code null}
     *                  for none. Pass the real (un-balanced) rows as {@code flatData} together with
     *                  this: balancing the whole dataset first and splitting afterwards leaks
     *                  synthetic rows into the fold being scored — see {@link FoldResampler}.
     */
    static TuningResult tune(
            float[] flatData,
            float[] labels,
            int nSamples,
            int nFeatures,
            int nClasses,
            int nTrials,
            int nFolds,
            Integer xgbFixedRounds,
            Integer lgbFixedRounds,
            FoldResampler resampler,
            Consumer<String> log) {
        Consumer<String> out = log != null ? log : s -> {};

        if (nSamples < MIN_SAMPLES) {
            out.accept("Too few samples (" + nSamples + ") for auto-tuning — using defaults");
            HyperParams defaults = defaults();
            return new TuningResult(defaults, 0, defaults, 0);
        }

        int[] intLabels = new int[nSamples];
        for (int i = 0; i < nSamples; i++) intLabels[i] = (int) labels[i];

        // A k-fold split can only give every class a presence in every training fold if the rarest
        // class has at least k members. Above that, k folds means each class sits out of one fold's
        // training set at a time; below it, some fold trains without the class entirely and then
        // scores rows of it, taking a guaranteed zero on that class's F1. That zero is identical
        // for every trial, so it does not corrupt the ranking TPE works from — but it drags the
        // reported score down for a reason that has nothing to do with hyperparameters, and a
        // reader comparing it against the Training Metrics report has no way to know that. Clamp
        // to what the data supports and say so, rather than reporting a confident number from
        // folds too thin to carry one.
        int effectiveFolds = resolveFolds(intLabels, nClasses, nFolds, out);

        // Say what this is actually going to do. "Several minutes" was the old wording, and at a
        // wide panel with many classes a single fit runs into minutes on its own — 2 x trials x
        // folds of them is hours, which reads as a hang unless the count is stated up front.
        out.accept(String.format(
                "Auto-tuning: %d TPE trials × %d folds per model on %,d samples × %,d features"
                        + " — %d model fits in total",
                nTrials, effectiveFolds, nSamples, nFeatures, 2 * nTrials * effectiveFolds));
        if (xgbFixedRounds != null || lgbFixedRounds != null) {
            out.accept(String.format(
                    "  Holding rounds at the early-stopping result (XGBoost %s, LightGBM %s);"
                            + " searching depth, learning rate, subsample and colsample",
                    xgbFixedRounds == null ? "searched" : xgbFixedRounds,
                    lgbFixedRounds == null ? "searched" : lgbFixedRounds));
        }

        List<int[][]> foldIndices = stratifiedKFold(intLabels, nClasses, effectiveFolds, new Random(42));

        // Materialised once and shared by both models. The folds are identical on every trial, so
        // re-deriving them per trial re-ran the row copies 2 x nTrials times over — and now that
        // balancing happens per fold, it would re-run SMOTE just as often, which used to be the
        // single most expensive phase of a training run. The trade is that all folds stay resident
        // for the whole search rather than being rebuilt and dropped per trial; the footprint is
        // logged below so a run that runs out of memory says why.
        List<PreparedFold> folds = prepareFolds(flatData, labels, intLabels, nFeatures, foldIndices, resampler, out);

        // Tune each model independently
        out.accept("── Tuning XGBoost ──");
        ModelTuneResult xgb = tuneModel(
                "XGBoost", folds, nFeatures, nClasses, nTrials, new Random(42), xgbFixedRounds, DEPTH_MAX, out);

        // LightGBM's depth ceiling is lower than XGBoost's because its leaf cap binds first.
        out.accept("── Tuning LightGBM ──");
        ModelTuneResult lgb = tuneModel(
                "LightGBM",
                folds,
                nFeatures,
                nClasses,
                nTrials,
                new Random(43),
                lgbFixedRounds,
                leafBoundedDepth(LightGBMModel.NUM_LEAVES),
                out);

        out.accept(String.format("Best XGBoost:  %s → F1 = %.4f", xgb.params(), xgb.score()));
        out.accept(String.format("Best LightGBM: %s → F1 = %.4f", lgb.params(), lgb.score()));

        return new TuningResult(xgb.params(), xgb.score(), lgb.params(), lgb.score());
    }

    // ── Fold count ──────────────────────────────────────────────────────────

    /**
     * Reduce the requested fold count to what the rarest class can support, warning when it does.
     * <p>
     * Mirrors the warning scikit-learn's {@code StratifiedKFold} emits — "the least populated class
     * has n members, which is less than n_splits" — except that this clamps as well as warns, since
     * nothing downstream can act on the warning. Classes with no labelled rows at all are ignored:
     * a class the user has declared but not yet labelled says nothing about how finely the labelled
     * data can be split.
     *
     * @return the fold count to actually use, never below {@link #MIN_FOLDS}
     */
    static int resolveFolds(int[] intLabels, int nClasses, int requestedFolds, Consumer<String> out) {
        int[] support = new int[nClasses];
        for (int label : intLabels) {
            if (label >= 0 && label < nClasses) support[label]++;
        }

        int rarest = Integer.MAX_VALUE;
        int rarestClass = -1;
        for (int c = 0; c < nClasses; c++) {
            if (support[c] > 0 && support[c] < rarest) {
                rarest = support[c];
                rarestClass = c;
            }
        }
        if (rarestClass < 0) {
            return requestedFolds; // no labelled rows at all; MIN_SAMPLES already gates this
        }

        if (rarest >= requestedFolds) {
            return requestedFolds;
        }

        int folds = Math.max(MIN_FOLDS, rarest);
        if (rarest < MIN_FOLDS) {
            // Even two folds cannot train and score a single-member class. Nothing to clamp to, so
            // the honest move is to say the class is unscoreable and carry on with the rest.
            out.accept(String.format(
                    "  Note: class %d has only %d labelled row(s), too few for cross-validation to"
                            + " score it at all — its F1 counts as 0 in every trial. Label a few more"
                            + " for the search to see it.",
                    rarestClass, rarest));
        } else {
            out.accept(String.format(
                    "  Note: the least populated class (%d) has %d labelled rows, fewer than the %d"
                            + " folds requested — using %d folds so every fold can train on it.",
                    rarestClass, rarest, requestedFolds, folds));
        }
        return folds;
    }

    // ── Fold preparation ────────────────────────────────────────────────────

    /**
     * Cut every fold out of the dataset once, balancing each fold's training rows in isolation.
     *
     * @param resampler applied to the training rows only; the scored rows are always the real ones
     */
    private static List<PreparedFold> prepareFolds(
            float[] flatData,
            float[] labels,
            int[] intLabels,
            int nFeatures,
            List<int[][]> foldIndices,
            FoldResampler resampler,
            Consumer<String> out) {

        List<PreparedFold> prepared = new ArrayList<>(foldIndices.size());
        long realTrainRows = 0;
        long trainRows = 0;
        long testRows = 0;

        for (var fold : foldIndices) {
            int[] trainIdx = fold[0];
            int[] testIdx = fold[1];

            float[] trainData = extractRows(flatData, trainIdx, nFeatures);
            float[] trainLabels = extractLabels(labels, trainIdx);
            int trainSize = trainIdx.length;
            realTrainRows += trainSize;

            if (resampler != null) {
                Resampled balanced = resampler.resample(trainData, trainLabels, trainSize);
                trainData = balanced.data();
                trainLabels = balanced.labels();
                trainSize = balanced.size();
            }

            trainRows += trainSize;
            testRows += testIdx.length;
            prepared.add(new PreparedFold(
                    trainData,
                    trainLabels,
                    trainSize,
                    extractRows(flatData, testIdx, nFeatures),
                    extractIntLabels(intLabels, testIdx),
                    testIdx.length));
        }

        long mb = (trainRows + testRows) * nFeatures * (long) Float.BYTES / (1024 * 1024);
        if (resampler != null) {
            out.accept(String.format(
                    "  Folds prepared once and reused: %,d training rows balanced to %,d inside the folds,"
                            + " scored against %,d real rows (~%,d MB held)",
                    realTrainRows, trainRows, testRows, mb));
        } else {
            out.accept(String.format(
                    "  Folds prepared once and reused: %,d training rows, %,d scored rows (~%,d MB held)",
                    trainRows, testRows, mb));
        }
        return prepared;
    }

    /**
     * Highest {@code max_depth} worth searching for a leaf-capped learner.
     * <p>
     * LightGBM grows leaf-wise under a {@code num_leaves} cap, so {@code max_depth} constrains the
     * tree only while a depth-d tree's ceiling of 2^d leaves stays below that cap. At the default
     * 31 leaves every depth from 5 up produces the same model, and searching 5..12 spends trials
     * re-scoring one configuration. Derived from the cap rather than hard-coded so the two cannot
     * drift apart.
     *
     * @param numLeaves the leaf cap the model is configured with
     * @return the smallest depth whose leaf ceiling reaches the cap, clamped to {@link #DEPTH_MAX}
     */
    static int leafBoundedDepth(int numLeaves) {
        int depth = DEPTH_MIN;
        while (depth < DEPTH_MAX && (1L << depth) < numLeaves) {
            depth++;
        }
        return depth;
    }

    // ── Per-model TPE tuning loop ───────────────────────────────────────────

    private static ModelTuneResult tuneModel(
            String modelName,
            List<PreparedFold> folds,
            int nFeatures,
            int nClasses,
            int nTrials,
            Random rng,
            Integer fixedRounds,
            int depthMax,
            Consumer<String> log) {
        TPESampler sampler = new TPESampler(rng, fixedRounds, depthMax);
        HyperParams bestParams = null;
        double bestScore = -1;
        boolean anyTrialScored = false;

        // The training thread budget, not the core count: a user who caps CPU threads to keep the
        // machine usable meant it to apply here too, and this is the heaviest thing the extension
        // does. It used to call availableProcessors() directly and ignore the cap entirely.
        int budget = TrainingThreads.total();
        int nFolds = folds.size();
        // Parallelise folds when there is enough budget for at least 2 threads each.
        boolean parallelFolds = budget >= nFolds * 2;
        // Both models divide the budget across concurrent folds. LightGBM used to take the whole
        // budget per fold while XGBoost divided it, so five concurrent folds asked for five times
        // the machine.
        int threadsPerFold = parallelFolds ? TrainingThreads.forConcurrentTasks(nFolds) : budget;

        if (parallelFolds) {
            log.accept(String.format(
                    "  Parallel CV: %d folds × %d threads/fold (%d thread budget)", nFolds, threadsPerFold, budget));
        }

        ExecutorService foldPool = parallelFolds ? BackgroundExecutors.newFixedPool(nFolds, "SpClassify-Tune") : null;

        try {
            for (int trial = 0; trial < nTrials; trial++) {
                HyperParams hp = sampler.suggest();

                double totalScore = 0;
                int validFolds = 0;

                if (parallelFolds) {
                    // Submit all folds in parallel
                    List<Future<Double>> futures = new ArrayList<>(nFolds);
                    for (PreparedFold fold : folds) {
                        futures.add(foldPool.submit(
                                () -> evaluateFold(modelName, fold, nFeatures, nClasses, hp, threadsPerFold)));
                    }

                    for (Future<Double> f : futures) {
                        try {
                            double f1 = f.get();
                            if (f1 >= 0) {
                                totalScore += f1;
                                validFolds++;
                            }
                        } catch (InterruptedException | ExecutionException e) {
                            logger.warn("Parallel CV fold failed: {}", e.toString());
                        }
                    }
                } else {
                    // Sequential fallback
                    for (PreparedFold fold : folds) {
                        double f1 = evaluateFold(modelName, fold, nFeatures, nClasses, hp, threadsPerFold);
                        if (f1 >= 0) {
                            totalScore += f1;
                            validFolds++;
                        }
                    }
                }

                int failedFolds = nFolds - validFolds;
                if (validFolds == 0) {
                    // No information at all. Feeding 0 to the sampler would be worse than useless:
                    // TPE would read it as "this region scores terribly" and steer away from
                    // hyperparameters that were never actually evaluated. Skip the observation.
                    log.accept(String.format(
                            "  Trial %2d/%d: %s → ALL %d FOLDS FAILED (see the log for the cause;"
                                    + " out of memory is the usual one at this width)",
                            trial + 1, nTrials, hp, nFolds));
                    continue;
                }

                double meanScore = totalScore / validFolds;
                sampler.observe(hp, meanScore);
                anyTrialScored = true;

                String marker = meanScore > bestScore ? " ★" : "";
                String failures = failedFolds > 0 ? String.format("  [%d/%d folds failed]", failedFolds, nFolds) : "";
                log.accept(String.format(
                        "  Trial %2d/%d: %s → F1 = %.4f%s%s", trial + 1, nTrials, hp, meanScore, marker, failures));

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

        if (!anyTrialScored || bestParams == null) {
            // Every trial failed, so the search learned nothing. Returning the first random draw
            // would look like a tuned result; the documented defaults are the honest answer.
            bestParams = defaults();
            bestScore = -1;
            log.accept("  No trial produced a usable score — falling back to defaults: " + bestParams);
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
        private static final int N_DIMS = 5;

        /** Index of the rounds dimension in the transformed vector. */
        private static final int DIM_ROUNDS = 0;

        /** Index of the depth dimension in the transformed vector. */
        private static final int DIM_DEPTH = 1;

        private final Random rng;
        private final List<double[]> history = new ArrayList<>();
        private final List<Double> scores = new ArrayList<>();

        /**
         * When early stopping already chose a round count, that value is used verbatim and the
         * rounds dimension drops out of the search entirely — it is not merely overwritten
         * afterwards. Leaving it in would have the KDE score trials at one round count and the
         * deployed model use another, which is the same "tuned a model that never got built"
         * failure the shared parameter builders exist to prevent.
         */
        private final Integer fixedRounds;

        /** The dimensions actually being searched — all four, or the three besides rounds. */
        private final int[] dims;

        /**
         * Per-model bounds. Only the depth ceiling varies: a leaf-capped learner stops responding
         * to depth well before {@link #DEPTH_MAX}, so its search space is narrowed to the range
         * that changes the model rather than being padded with equivalent configurations.
         */
        private final double[] lower;

        private final double[] upper;

        TPESampler(Random rng, Integer fixedRounds, int depthMax) {
            this.rng = rng;
            this.fixedRounds = fixedRounds;
            this.lower = LOWER;
            this.upper = UPPER.clone();
            this.upper[DIM_DEPTH] = depthMax;
            if (fixedRounds == null) {
                this.dims = new int[] {0, 1, 2, 3, 4};
            } else {
                this.dims = new int[] {1, 2, 3, 4};
            }
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
            if (fixedRounds != null) sample[DIM_ROUNDS] = fixedRounds;
            for (int d : dims) {
                double[] center = points.get(rng.nextInt(points.size()));
                double bw = silvermanBW(points, d);
                sample[d] = center[d] + rng.nextGaussian() * bw;
                sample[d] = Math.max(lower[d], Math.min(upper[d], sample[d]));
            }
            return sample;
        }

        /** Evaluate the KDE density at a point (product of per-dimension densities). */
        private double evaluateKDE(double[] x, List<double[]> points) {
            double logDensity = 0;
            for (int d : dims) {
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
            if (n < 2) return (upper[dim] - lower[dim]) / 4.0;

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
            if (std < 1e-10) std = (upper[dim] - lower[dim]) / 4.0;

            return 1.06 * std * Math.pow(n, -0.2);
        }

        private HyperParams sampleUniform() {
            int rounds;
            if (fixedRounds != null) {
                rounds = fixedRounds;
            } else {
                rounds = ROUNDS_MIN + rng.nextInt(ROUNDS_MAX - ROUNDS_MIN + 1);
                rounds = ((rounds + 5) / 10) * 10;
            }
            int depth = DEPTH_MIN + rng.nextInt((int) upper[DIM_DEPTH] - DEPTH_MIN + 1);
            float eta =
                    (float) Math.exp(Math.log(ETA_MIN) + rng.nextDouble() * (Math.log(ETA_MAX) - Math.log(ETA_MIN)));
            float sub = (float) (SUB_MIN + rng.nextDouble() * (SUB_MAX - SUB_MIN));
            float col = (float) (COL_MIN + rng.nextDouble() * (COL_MAX - COL_MIN));
            return new HyperParams(rounds, depth, eta, sub, col);
        }

        private static double[] toTransformed(HyperParams hp) {
            return new double[] {hp.numRounds(), hp.maxDepth(), Math.log(hp.eta()), hp.subsample(), hp.colsample()};
        }

        private HyperParams fromTransformed(double[] x) {
            int rounds;
            if (fixedRounds != null) {
                rounds = fixedRounds;
            } else {
                rounds = (int) Math.round(x[DIM_ROUNDS]);
                rounds = Math.max(ROUNDS_MIN, Math.min(ROUNDS_MAX, ((rounds + 5) / 10) * 10));
            }
            int depth = Math.max(DEPTH_MIN, Math.min((int) upper[DIM_DEPTH], (int) Math.round(x[DIM_DEPTH])));
            float eta = (float) Math.max(ETA_MIN, Math.min(ETA_MAX, Math.exp(x[2])));
            float sub = (float) Math.max(SUB_MIN, Math.min(SUB_MAX, x[3]));
            float col = (float) Math.max(COL_MIN, Math.min(COL_MAX, x[4]));
            return new HyperParams(rounds, depth, eta, sub, col);
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

    /** Score one prepared fold with whichever library is being tuned. */
    private static double evaluateFold(
            String modelName, PreparedFold fold, int nFeatures, int nClasses, HyperParams hp, int nThreads) {
        return "XGBoost".equals(modelName)
                ? evaluateXGBoostFold(
                        fold.trainData(),
                        fold.trainLabels(),
                        fold.trainSize(),
                        fold.testData(),
                        fold.testTruth(),
                        fold.testSize(),
                        nFeatures,
                        nClasses,
                        hp,
                        nThreads)
                : evaluateLightGBMFold(
                        fold.trainData(),
                        fold.trainLabels(),
                        fold.trainSize(),
                        fold.testData(),
                        fold.testTruth(),
                        fold.testSize(),
                        nFeatures,
                        nClasses,
                        hp,
                        nThreads);
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

            // Shared with the real fit rather than restated here: cross-validation has to score
            // the model that will actually be built, and a locally-maintained copy silently stops
            // matching the moment a parameter is added on the other side.
            Map<String, Object> params = XGBoostModel.buildParams(
                    nClasses, hp.maxDepth(), hp.eta(), hp.subsample(), hp.colsample(), nThreads);
            params.put("verbosity", 0);

            booster = XGBoost.train(trainMat, params, hp.numRounds(), new LinkedHashMap<>(), null, null);

            testMat = new DMatrix(testData, testSize, nFeatures, Float.NaN);
            float[][] preds = booster.predict(testMat);

            int[] predClasses = toPredictedClasses(preds, testSize, nClasses);
            return macroF1(predClasses, testTruth, nClasses);

        } catch (Exception e) {
            logger.warn("XGBoost CV fold failed: {}", e.toString());
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

            // Shared with the real fit rather than restated here. The local copy omitted
            // min_gain_to_split, so every trial was scored on an unconstrained booster and the
            // winning depth/rate/subsample was then handed to a constrained one — the tuner was
            // optimising a model that never got built.
            String params = LightGBMModel.buildParams(
                    nClasses, hp.maxDepth(), hp.eta(), hp.subsample(), hp.colsample(), nThreads);

            booster = LGBMBooster.create(dataset, params);
            for (int i = 0; i < hp.numRounds(); i++) {
                // is_finished is deliberately ignored — see LightGBMModel.train(). It marks a
                // barren iteration, not convergence, so breaking on it would score a truncated
                // model and hand its round count to a full one.
                booster.updateOneIter();
            }

            double[] rawPreds =
                    booster.predictForMat(testData, testSize, nFeatures, true, PredictionType.C_API_PREDICT_NORMAL);

            int[] predClasses = toLGBPredictedClasses(rawPreds, testSize, nClasses);
            return macroF1(predClasses, testTruth, nClasses);

        } catch (Exception e) {
            logger.warn("LightGBM CV fold failed: {}", e.toString());
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
