package qupath.ext.spclassify.classifier;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/**
 * Tests the {@link TrainValMetricsComputer} orchestration with stub trainers and a
 * "perfect" predictor (reads the true class encoded in feature 0), so the split →
 * array-build → score pipeline is exercised without native ML models.
 */
class TrainValMetricsComputerTest {

    private static final List<String> CLASSES = List.of("A", "B");
    private static final double EPS = 1e-9;

    /** Two features per row; feature 0 holds the true class index. */
    private static List<float[]> rowsEncoding(List<Integer> labels) {
        List<float[]> rows = new ArrayList<>();
        for (int lbl : labels) rows.add(new float[] {lbl, 0f});
        return rows;
    }

    /** Predictor that returns a one-hot for the class encoded in feature 0 of each row. */
    private static TrainValMetricsComputer.ModelPredictor perfectPredictor(int nFeatures) {
        return (data, n) -> {
            float[][] out = new float[n][CLASSES.size()];
            for (int i = 0; i < n; i++) {
                int cls = (int) data[i * nFeatures];
                out[i][cls] = 1f;
            }
            return out;
        };
    }

    @Test
    void computesPerfectMetricsWithMatchingPredictor() throws Exception {
        List<Integer> labels = new ArrayList<>(List.of(0, 0, 0, 0, 0, 1, 1, 1, 1, 1));
        List<float[]> rows = rowsEncoding(labels);
        AtomicInteger trainCalls = new AtomicInteger(0);
        TrainValMetricsComputer.ModelTrainer trainer = (data, lbls, n) -> trainCalls.incrementAndGet();

        var result = TrainValMetricsComputer.compute(
                rows,
                labels,
                labels.size(),
                2,
                2,
                ResamplingStrategy.NONE,
                trainer,
                perfectPredictor(2),
                "Model 1 (XGBOOST)",
                trainer,
                perfectPredictor(2),
                "Model 2 (LIGHTGBM)",
                CLASSES,
                s -> {});

        assertNotNull(result.model1Train());
        assertNotNull(result.model1Val());
        assertNotNull(result.model2Train());
        assertNotNull(result.model2Val());
        // Perfect predictor → macro F1 of 1.0 on both folds for both models.
        assertEquals(1.0, result.model1Train().macroF1(), EPS);
        assertEquals(1.0, result.model1Val().macroF1(), EPS);
        assertEquals(1.0, result.model2Val().macroF1(), EPS);
        assertEquals(2, trainCalls.get(), "both models' eval copies should be trained once");
    }

    @Test
    void degenerateSplitReturnsEmptyResultAndLogs() throws Exception {
        // One sample per class → the 20% val fold is empty.
        List<Integer> labels = new ArrayList<>(List.of(0, 1));
        List<float[]> rows = rowsEncoding(labels);
        List<String> log = new ArrayList<>();
        AtomicInteger trainCalls = new AtomicInteger(0);
        TrainValMetricsComputer.ModelTrainer trainer = (data, lbls, n) -> trainCalls.incrementAndGet();

        var result = TrainValMetricsComputer.compute(
                rows,
                labels,
                labels.size(),
                2,
                2,
                ResamplingStrategy.NONE,
                trainer,
                perfectPredictor(2),
                "M1",
                trainer,
                perfectPredictor(2),
                "M2",
                CLASSES,
                log::add);

        assertNull(result.model1Train());
        assertNull(result.model1Val());
        assertNull(result.model2Train());
        assertNull(result.model2Val());
        assertEquals(0, trainCalls.get(), "no training when the split is degenerate");
        assertTrue(log.stream().anyMatch(m -> m.contains("Skipping metrics")), "should log that metrics were skipped");
    }

    @Test
    void stratifiedSplitIsProportionalDistinctAndComplete() {
        // 10 of class 0, 4 of class 1.
        int[] labels = new int[14];
        Arrays.fill(labels, 0, 10, 0);
        Arrays.fill(labels, 10, 14, 1);

        int[][] split = TrainValMetricsComputer.stratifiedSplit(labels, 2, 0.8, new Random(1));
        int[] train = split[0];
        int[] val = split[1];

        // 80% of each class to train: 8 of class0, 3 of class1 = 11 train; 3 val.
        assertEquals(11, train.length);
        assertEquals(3, val.length);

        // Distinct and disjoint, covering all 14 indices exactly once.
        var all = new java.util.TreeSet<Integer>();
        for (int i : train) all.add(i);
        for (int i : val) all.add(i);
        assertEquals(14, all.size(), "train ∪ val must cover every index with no overlap");
        assertEquals(0, all.first());
        assertEquals(13, all.last());
    }

    @Test
    void stratifiedSplitKeepsAtLeastOneTrainPerClass() {
        int[] labels = {0, 1}; // one each
        int[][] split = TrainValMetricsComputer.stratifiedSplit(labels, 2, 0.8, new Random(1));
        assertEquals(2, split[0].length, "each class contributes at least one train sample");
        assertEquals(0, split[1].length);
    }
}
