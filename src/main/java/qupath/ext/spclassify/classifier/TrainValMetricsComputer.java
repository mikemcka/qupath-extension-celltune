package qupath.ext.spclassify.classifier;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;

/**
 * Computes per-class train/validation metrics for the two models on a fresh 80/20
 * stratified split of the real (pre-resampling) labelled data.
 * <p>
 * Extracted from {@code DualModelClassifier}. The models are supplied as injected
 * {@link ModelTrainer}/{@link ModelPredictor} callbacks, so the orchestration
 * (split → resample the train fold → build flat arrays → train eval copies →
 * predict → score) holds no classifier state and is unit-testable with stub
 * trainers/predictors — no native ML models required.
 * <p>
 * The trainers train <em>evaluation copies</em> on the 80% only; in production these
 * overwrite the caller's models, which is intentional — the final full-data training
 * step runs immediately afterwards and overwrites them again.
 */
final class TrainValMetricsComputer {

    private TrainValMetricsComputer() {} // utility class

    /** Trains a model on the given flat feature matrix + float labels. */
    @FunctionalInterface
    interface ModelTrainer {
        void train(float[] flatData, float[] labels, int nSamples) throws Exception;
    }

    /** Predicts {@code [nSamples][nClasses]} probabilities for a flat feature matrix. */
    @FunctionalInterface
    interface ModelPredictor {
        float[][] predict(float[] flatData, int nSamples) throws Exception;
    }

    /** The four metric sets; fields are null when the split was degenerate. */
    record Result(
            TrainingMetrics model1Train,
            TrainingMetrics model1Val,
            TrainingMetrics model2Train,
            TrainingMetrics model2Val) {
        static Result empty() {
            return new Result(null, null, null, null);
        }
    }

    /**
     * An 80/20 stratified split with the train fold resampled and both folds flattened, ready to
     * hand to a model.
     *
     * @param split      {@code [trainIndices, valIndices]} into the original real rows
     * @param trainData  row-major resampled train fold
     * @param trainSize  rows in {@code trainData} (post-resampling, so &ge; {@code split[0].length})
     * @param valData    row-major validation fold — real rows only, never resampled
     */
    record PreparedFold(
            int[][] split,
            float[] trainData,
            float[] trainLabels,
            int trainSize,
            float[] valData,
            float[] valLabels,
            int valSize) {}

    /**
     * Builds the 80/20 fold used by both early stopping and the train/val metrics.
     * <p>
     * Both callers previously did this independently with the same inputs, the same ratio and the
     * same seed, so they produced byte-identical folds — including two identical runs of an
     * O(n²) resampling pass. Sharing one {@code PreparedFold} removes that duplicate work
     * without changing any value it produces.
     *
     * @param strategy resampling applied to the train fold only; the validation fold stays real
     * @param out      log sink for the resampling messages
     * @return the prepared fold, or {@code null} if the split was degenerate (an empty fold)
     */
    static PreparedFold prepare(
            List<float[]> realRows,
            List<Integer> realLabels,
            int nRealSamples,
            int nClasses,
            int nFeatures,
            ResamplingStrategy strategy,
            Consumer<String> out) {
        int[] realIntLabels = new int[nRealSamples];
        for (int i = 0; i < nRealSamples; i++) realIntLabels[i] = realLabels.get(i);

        int[][] split = stratifiedSplit(realIntLabels, nClasses, 0.8, new Random(42));
        int valSize = split[1].length;
        if (valSize == 0 || split[0].length == 0) {
            return null;
        }

        // Build the 80% train fold (eligible for resampling); the 20% stays real.
        List<float[]> trainRows = new ArrayList<>(split[0].length);
        List<Integer> trainLabelsList = new ArrayList<>(split[0].length);
        for (int idx : split[0]) {
            trainRows.add(realRows.get(idx));
            trainLabelsList.add(realLabels.get(idx));
        }

        float[] trainData;
        float[] trainLabels;
        int trainSize;
        if (strategy != ResamplingStrategy.NONE) {
            Resampler.Result res = Resampler.apply(trainRows, trainLabelsList, nClasses, strategy, out);
            trainSize = res.size();
            // Straight to the flat matrix — skips materialising an ArrayList of rows and boxing
            // every label, only to copy them out again.
            float[][] rows = res.rowArray();
            int[] labels = res.labelArray();
            trainData = new float[trainSize * nFeatures];
            trainLabels = new float[trainSize];
            for (int i = 0; i < trainSize; i++) {
                System.arraycopy(rows[i], 0, trainData, i * nFeatures, nFeatures);
                trainLabels[i] = labels[i];
            }
        } else {
            trainSize = trainRows.size();
            trainData = new float[trainSize * nFeatures];
            trainLabels = new float[trainSize];
            for (int i = 0; i < trainSize; i++) {
                System.arraycopy(trainRows.get(i), 0, trainData, i * nFeatures, nFeatures);
                trainLabels[i] = trainLabelsList.get(i);
            }
        }

        float[] valData = new float[valSize * nFeatures];
        float[] valLabels = new float[valSize];
        for (int i = 0; i < valSize; i++) {
            System.arraycopy(realRows.get(split[1][i]), 0, valData, i * nFeatures, nFeatures);
            valLabels[i] = realLabels.get(split[1][i]);
        }

        return new PreparedFold(split, trainData, trainLabels, trainSize, valData, valLabels, valSize);
    }

    /**
     * @param realRows        feature rows for the real labelled cells (pre-resampling)
     * @param realLabels      integer class labels parallel to {@code realRows}
     * @param nRealSamples    number of real samples
     * @param nClasses        class count
     * @param nFeatures       feature count per row
     * @param strategy        resampling strategy applied to the 80% train fold only
     * @param model1Trainer   trains model-1 eval copy on the 80%
     * @param model1Predictor predicts with model-1
     * @param model1Name      label prefix for model-1 metrics, e.g. {@code "Model 1 (XGBOOST)"}
     * @param model2Trainer   trains model-2 eval copy on the 80%
     * @param model2Predictor predicts with model-2
     * @param model2Name      label prefix for model-2 metrics
     * @param classNames      ordered class names
     * @param out             progress/log sink
     * @return the four metric sets, or {@link Result#empty()} if the split was degenerate
     */
    static Result compute(
            List<float[]> realRows,
            List<Integer> realLabels,
            int nRealSamples,
            int nClasses,
            int nFeatures,
            ResamplingStrategy strategy,
            ModelTrainer model1Trainer,
            ModelPredictor model1Predictor,
            String model1Name,
            ModelTrainer model2Trainer,
            ModelPredictor model2Predictor,
            String model2Name,
            List<String> classNames,
            Consumer<String> out)
            throws Exception {
        return compute(
                realRows,
                realLabels,
                nRealSamples,
                nClasses,
                nFeatures,
                strategy,
                model1Trainer,
                model1Predictor,
                model1Name,
                model2Trainer,
                model2Predictor,
                model2Name,
                classNames,
                out,
                null);
    }

    /**
     * @param cached a fold already built by {@link #prepare} for the <em>same</em> rows, labels,
     *               class count and strategy — reused as-is; pass {@code null} to build one here
     */
    static Result compute(
            List<float[]> realRows,
            List<Integer> realLabels,
            int nRealSamples,
            int nClasses,
            int nFeatures,
            ResamplingStrategy strategy,
            ModelTrainer model1Trainer,
            ModelPredictor model1Predictor,
            String model1Name,
            ModelTrainer model2Trainer,
            ModelPredictor model2Predictor,
            String model2Name,
            List<String> classNames,
            Consumer<String> out,
            PreparedFold cached)
            throws Exception {
        // Resampling is silent here: when this builds its own fold the caller has not announced
        // one, and when it reuses a cached fold the messages were already emitted upstream.
        PreparedFold fold = cached != null
                ? cached
                : prepare(realRows, realLabels, nRealSamples, nClasses, nFeatures, strategy, s -> {});
        if (fold == null) {
            out.accept("Skipping metrics: stratified split produced empty fold.");
            return Result.empty();
        }

        int valSize = fold.valSize();
        int evTrainSize = fold.trainSize();
        float[] evTrainData = fold.trainData();
        float[] evTrainLabels = fold.trainLabels();
        float[] evValData = fold.valData();
        float[] evValLabels = fold.valLabels();

        // ── Eval Model 1 ────────────────────────────────────────────────────
        model1Trainer.train(evTrainData, evTrainLabels, evTrainSize);
        float[][] m1TrainProba = model1Predictor.predict(evTrainData, evTrainSize);
        float[][] m1ValProba = model1Predictor.predict(evValData, valSize);
        TrainingMetrics model1Train =
                TrainingMetrics.compute(model1Name + " — Train (80%)", classNames, evTrainLabels, m1TrainProba);
        TrainingMetrics model1Val =
                TrainingMetrics.compute(model1Name + " — Validation (20%)", classNames, evValLabels, m1ValProba);

        // ── Eval Model 2 ────────────────────────────────────────────────────
        model2Trainer.train(evTrainData, evTrainLabels, evTrainSize);
        float[][] m2TrainProba = model2Predictor.predict(evTrainData, evTrainSize);
        float[][] m2ValProba = model2Predictor.predict(evValData, valSize);
        TrainingMetrics model2Train =
                TrainingMetrics.compute(model2Name + " — Train (80%)", classNames, evTrainLabels, m2TrainProba);
        TrainingMetrics model2Val =
                TrainingMetrics.compute(model2Name + " — Validation (20%)", classNames, evValLabels, m2ValProba);

        out.accept(String.format(
                "Macro F1: M1 train=%.3f val=%.3f | M2 train=%.3f val=%.3f",
                model1Train.macroF1(), model1Val.macroF1(), model2Train.macroF1(), model2Val.macroF1()));

        return new Result(model1Train, model1Val, model2Train, model2Val);
    }

    /**
     * Stratified train/val split: shuffles each class's indices and assigns
     * {@code trainRatio} of each to train (at least 1), the rest to validation.
     * Shared with {@code DualModelClassifier}'s early-stopping split.
     */
    static int[][] stratifiedSplit(int[] labels, int nClasses, double trainRatio, Random rng) {
        List<List<Integer>> groups = new ArrayList<>();
        for (int c = 0; c < nClasses; c++) groups.add(new ArrayList<>());
        for (int i = 0; i < labels.length; i++) groups.get(labels[i]).add(i);

        List<Integer> trainList = new ArrayList<>();
        List<Integer> valList = new ArrayList<>();

        for (var group : groups) {
            if (group.isEmpty()) continue; // class has no samples in pooled data — skip
            Collections.shuffle(group, rng);
            int trainCount = Math.max(1, (int) (group.size() * trainRatio));
            trainList.addAll(group.subList(0, trainCount));
            if (trainCount < group.size()) {
                valList.addAll(group.subList(trainCount, group.size()));
            }
        }

        return new int[][] {
            trainList.stream().mapToInt(Integer::intValue).toArray(),
            valList.stream().mapToInt(Integer::intValue).toArray()
        };
    }
}
