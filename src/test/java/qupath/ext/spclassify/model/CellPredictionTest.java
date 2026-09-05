package qupath.ext.spclassify.model;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import org.junit.jupiter.api.Test;

class CellPredictionTest {

    private static final List<String> CLASSES = List.of("T", "NK", "B");

    @Test
    void isDisagreementReturnsFalseWhenModelsAgree() {
        var pred = new CellPrediction(
                "cell-1", "T", "T", new float[] {0.9f, 0.05f, 0.05f}, new float[] {0.85f, 0.1f, 0.05f}, CLASSES);
        assertFalse(pred.isDisagreement());
    }

    @Test
    void isDisagreementReturnsTrueWhenModelsDisagree() {
        var pred = new CellPrediction(
                "cell-2", "T", "NK", new float[] {0.7f, 0.2f, 0.1f}, new float[] {0.2f, 0.7f, 0.1f}, CLASSES);
        assertTrue(pred.isDisagreement());
    }

    @Test
    void avgLabelReturnsClassWithHighestAverageProbability() {
        // Avg probs: T=0.5, NK=0.4, B=0.1  → T wins
        var pred = new CellPrediction(
                "cell-3", "T", "NK", new float[] {0.6f, 0.3f, 0.1f}, new float[] {0.4f, 0.5f, 0.1f}, CLASSES);
        assertEquals("T", pred.avgLabel());
    }

    @Test
    void avgLabelReturnsCorrectClassWhenModel2Wins() {
        // Avg probs: T=0.2, NK=0.7, B=0.1  → NK wins
        var pred = new CellPrediction(
                "cell-4", "T", "NK", new float[] {0.3f, 0.6f, 0.1f}, new float[] {0.1f, 0.8f, 0.1f}, CLASSES);
        assertEquals("NK", pred.avgLabel());
    }

    @Test
    void allLabelReturnsSingleLabelWhenModelsAgree() {
        var pred = new CellPrediction(
                "cell-5", "B", "B", new float[] {0.1f, 0.1f, 0.8f}, new float[] {0.05f, 0.15f, 0.8f}, CLASSES);
        assertEquals("B", pred.allLabel());
    }

    @Test
    void allLabelReturnsCombinedLabelWhenModelsDisagree() {
        var pred = new CellPrediction(
                "cell-6", "T", "NK", new float[] {0.7f, 0.2f, 0.1f}, new float[] {0.2f, 0.7f, 0.1f}, CLASSES);
        assertEquals("T/NK", pred.allLabel());
    }

    @Test
    void constructorClonesProbs() {
        float[] probs1 = {0.8f, 0.1f, 0.1f};
        float[] probs2 = {0.1f, 0.8f, 0.1f};
        var pred = new CellPrediction("cell-7", "T", "NK", probs1, probs2, CLASSES);
        probs1[0] = 0.0f; // mutate original
        // prediction should be unaffected
        assertEquals("T", pred.avgLabel());
    }

    @Test
    void gettersReturnCorrectValues() {
        var pred = new CellPrediction(
                "cell-8", "NK", "B", new float[] {0.1f, 0.8f, 0.1f}, new float[] {0.1f, 0.1f, 0.8f}, CLASSES);
        assertEquals("cell-8", pred.getCellId());
        assertEquals("NK", pred.getModel1Label());
        assertEquals("B", pred.getModel2Label());
        assertEquals(CLASSES, pred.getClassNames());
    }

    @Test
    void model1ConfidenceReturnsCorrectProbability() {
        // model1 predicts NK (index 1) with prob 0.8
        var pred = new CellPrediction(
                "cell-9", "NK", "NK", new float[] {0.1f, 0.8f, 0.1f}, new float[] {0.1f, 0.8f, 0.1f}, CLASSES);
        assertEquals(0.8f, pred.model1Confidence(), 1e-6f);
    }

    @Test
    void model2ConfidenceReturnsCorrectProbability() {
        // model2 predicts B (index 2) with prob 0.9
        var pred = new CellPrediction(
                "cell-10", "T", "B", new float[] {0.9f, 0.05f, 0.05f}, new float[] {0.05f, 0.05f, 0.9f}, CLASSES);
        assertEquals(0.9f, pred.model2Confidence(), 1e-6f);
    }

    @Test
    void getModel1ProbsReturnsClomedArray() {
        float[] probs = {0.7f, 0.2f, 0.1f};
        var pred = new CellPrediction("cell-11", "T", "T", probs, probs.clone(), CLASSES);
        float[] returned = pred.getModel1Probs();
        returned[0] = 0.0f; // mutate the returned copy
        // original prediction should be unaffected
        assertEquals(0.7f, pred.model1Confidence(), 1e-6f);
    }
}
