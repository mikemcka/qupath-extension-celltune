package qupath.ext.spclassify.classifier;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import org.junit.jupiter.api.Test;

class TrainingMetricsTest {

    private static final double DELTA = 1e-9;

    // ── perfect binary classifier ────────────────────────────────────────────

    @Test
    void perfectBinaryClassifierHasAccuracyAndF1OfOne() {
        // 2 classes: 0=Neg, 1=Pos
        var classNames = List.of("Neg", "Pos");
        float[] trueLabels = {0f, 0f, 1f, 1f};
        float[][] proba = {
            {0.9f, 0.1f}, // true=Neg, pred=Neg ✓
            {0.8f, 0.2f}, // true=Neg, pred=Neg ✓
            {0.1f, 0.9f}, // true=Pos, pred=Pos ✓
            {0.2f, 0.8f}, // true=Pos, pred=Pos ✓
        };

        var metrics = TrainingMetrics.compute("test", classNames, trueLabels, proba);

        assertEquals(1.0, metrics.accuracy(), DELTA);
        assertEquals(1.0, metrics.macroF1(), DELTA);
        assertEquals(1.0, metrics.weightedF1(), DELTA);
        assertEquals(4, metrics.total());
    }

    @Test
    void perfectClassifierHasPerClassF1OfOne() {
        var classNames = List.of("Neg", "Pos");
        float[] trueLabels = {0f, 1f};
        float[][] proba = {
            {0.9f, 0.1f},
            {0.1f, 0.9f},
        };

        var metrics = TrainingMetrics.compute("test", classNames, trueLabels, proba);

        assertArrayEquals(new double[] {1.0, 1.0}, metrics.f1(), DELTA);
        assertArrayEquals(new double[] {1.0, 1.0}, metrics.precision(), DELTA);
        assertArrayEquals(new double[] {1.0, 1.0}, metrics.recall(), DELTA);
    }

    // ── known binary case ────────────────────────────────────────────────────

    @Test
    void binaryCaseWithTwoErrorsHasCorrectAccuracy() {
        // 4 samples: 2 correct, 2 wrong
        var classNames = List.of("A", "B");
        float[] trueLabels = {0f, 0f, 1f, 1f};
        float[][] proba = {
            {0.9f, 0.1f}, // true=A, pred=A ✓
            {0.2f, 0.8f}, // true=A, pred=B ✗
            {0.8f, 0.2f}, // true=B, pred=A ✗
            {0.1f, 0.9f}, // true=B, pred=B ✓
        };

        var metrics = TrainingMetrics.compute("test", classNames, trueLabels, proba);

        assertEquals(0.5, metrics.accuracy(), DELTA);
        assertEquals(4, metrics.total());
    }

    @Test
    void confusionMatrixDiagonalMatchesCorrectPredictions() {
        var classNames = List.of("A", "B");
        float[] trueLabels = {0f, 0f, 1f, 1f};
        float[][] proba = {
            {0.9f, 0.1f}, // true=A, pred=A  → cm[0][0]++
            {0.2f, 0.8f}, // true=A, pred=B  → cm[0][1]++
            {0.8f, 0.2f}, // true=B, pred=A  → cm[1][0]++
            {0.1f, 0.9f}, // true=B, pred=B  → cm[1][1]++
        };

        var metrics = TrainingMetrics.compute("test", classNames, trueLabels, proba);

        int[][] cm = metrics.confusionMatrix();
        assertEquals(1, cm[0][0]); // A predicted as A
        assertEquals(1, cm[0][1]); // A predicted as B
        assertEquals(1, cm[1][0]); // B predicted as A
        assertEquals(1, cm[1][1]); // B predicted as B
    }

    // ── multi-class ──────────────────────────────────────────────────────────

    @Test
    void multiClassPerfectClassifier() {
        var classNames = List.of("T", "NK", "B");
        float[] trueLabels = {0f, 1f, 2f};
        float[][] proba = {
            {0.9f, 0.05f, 0.05f}, // true=T, pred=T ✓
            {0.05f, 0.9f, 0.05f}, // true=NK, pred=NK ✓
            {0.05f, 0.05f, 0.9f}, // true=B, pred=B ✓
        };

        var metrics = TrainingMetrics.compute("test", classNames, trueLabels, proba);

        assertEquals(1.0, metrics.accuracy(), DELTA);
        assertEquals(3, metrics.total());
        assertArrayEquals(new double[] {1.0, 1.0, 1.0}, metrics.f1(), DELTA);
    }

    @Test
    void supportArraySumsToTotal() {
        var classNames = List.of("A", "B", "C");
        float[] trueLabels = {0f, 0f, 1f, 2f, 2f};
        float[][] proba = {
            {0.9f, 0.05f, 0.05f},
            {0.9f, 0.05f, 0.05f},
            {0.05f, 0.9f, 0.05f},
            {0.05f, 0.05f, 0.9f},
            {0.05f, 0.05f, 0.9f},
        };

        var metrics = TrainingMetrics.compute("test", classNames, trueLabels, proba);

        int supportSum = 0;
        for (int s : metrics.support()) supportSum += s;
        assertEquals(metrics.total(), supportSum);
        assertEquals(2, metrics.support()[0]);
        assertEquals(1, metrics.support()[1]);
        assertEquals(2, metrics.support()[2]);
    }

    // ── edge cases ───────────────────────────────────────────────────────────

    @Test
    void emptyInputReturnsZeroAccuracy() {
        var metrics = TrainingMetrics.compute("empty", List.of("A", "B"), new float[] {}, new float[][] {});
        assertEquals(0.0, metrics.accuracy(), DELTA);
        assertEquals(0, metrics.total());
    }

    @Test
    void labelContainsCorrectClassNames() {
        var classNames = List.of("T", "NK");
        var metrics = TrainingMetrics.compute("label-test", classNames, new float[] {0f}, new float[][] {{0.9f, 0.1f}});
        assertEquals("label-test", metrics.label());
        assertEquals(classNames, metrics.classNames());
    }

    // ── toFormattedReport ────────────────────────────────────────────────────

    @Test
    void allWrongClassifierHasZeroAccuracy() {
        var classNames = List.of("A", "B");
        float[] trueLabels = {0f, 1f};
        float[][] proba = {
            {0.1f, 0.9f}, // true=A, pred=B ✗
            {0.9f, 0.1f}, // true=B, pred=A ✗
        };

        var metrics = TrainingMetrics.compute("all-wrong", classNames, trueLabels, proba);
        assertEquals(0.0, metrics.accuracy(), DELTA);
    }

    @Test
    void zeroSupportClassDoesNotThrow() {
        // Only class 0 appears in trueLabels — class 1 has zero support
        var classNames = List.of("A", "B");
        float[] trueLabels = {0f, 0f};
        float[][] proba = {
            {0.9f, 0.1f},
            {0.8f, 0.2f},
        };

        assertDoesNotThrow(() -> {
            var metrics = TrainingMetrics.compute("zero-support", classNames, trueLabels, proba);
            assertEquals(0, metrics.support()[1]);
            assertEquals(0.0, metrics.precision()[1], DELTA);
            assertEquals(0.0, metrics.recall()[1], DELTA);
            assertEquals(0.0, metrics.f1()[1], DELTA);
        });
    }

    @Test
    void weightedF1IsZeroForAllWrongWithEqualSupport() {
        // All wrong → every TP = 0 → every per-class F1 = 0 → weightedF1 = 0
        var classNames = List.of("A", "B");
        float[] trueLabels = {0f, 1f};
        float[][] proba = {
            {0.1f, 0.9f},
            {0.9f, 0.1f},
        };

        var metrics = TrainingMetrics.compute("weighted-f1", classNames, trueLabels, proba);
        assertEquals(0.0, metrics.weightedF1(), DELTA);
        assertEquals(0.0, metrics.macroF1(), DELTA);
    }

    @Test
    void toFormattedReportContainsClassNamesAndMetrics() {
        var classNames = List.of("T", "NK");
        float[] trueLabels = {0f, 1f};
        float[][] proba = {{0.9f, 0.1f}, {0.1f, 0.9f}};

        var metrics = TrainingMetrics.compute("report-test", classNames, trueLabels, proba);
        String report = metrics.toFormattedReport();

        assertTrue(report.contains("T"));
        assertTrue(report.contains("NK"));
        assertTrue(report.contains("accuracy"));
        assertTrue(report.contains("macro F1"));
    }
}
