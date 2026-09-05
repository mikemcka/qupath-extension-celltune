package qupath.ext.spclassify.classifier;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import qupath.ext.spclassify.model.CellFeatureExtractor;
import qupath.ext.spclassify.model.CellPrediction;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.ROIs;

/**
 * Unit tests for {@link PredictionBatcher#predict}. The two models are supplied
 * as stub predictors returning canned probabilities, so the chunking, argmax →
 * label mapping, disagreement counting, and population-sink wiring are exercised
 * without any native XGBoost/LightGBM models. Covers logic that previously had no
 * automated coverage.
 */
class PredictionBatcherTest {

    private static final List<String> CLASSES = List.of("A", "B");

    /** Returns the next {@code chunkSize} rows of a pre-built probability list. */
    private static final class SeqPredictor implements PredictionBatcher.ChunkPredictor {
        private final float[][] rows;
        private int pos = 0;

        SeqPredictor(float[][] rows) {
            this.rows = rows;
        }

        @Override
        public float[][] predict(float[] chunkData, int chunkSize) {
            float[][] out = new float[chunkSize][];
            for (int i = 0; i < chunkSize; i++) out[i] = rows[pos++];
            return out;
        }
    }

    private static PathObject detection(double x) {
        return PathObjects.createDetectionObject(ROIs.createRectangleROI(x, 0, 10, 10, ImagePlane.getDefaultPlane()));
    }

    private static List<PathObject> detections(int n) {
        List<PathObject> cells = new ArrayList<>();
        for (int i = 0; i < n; i++) cells.add(detection(i * 20));
        return cells;
    }

    private static CellFeatureExtractor extractor() {
        // The stub predictors ignore the feature matrix, but the batcher still
        // calls extractMatrix; any marker works (missing measurements read as 0).
        return new CellFeatureExtractor(List.of("CD3: Cell: Mean"), null);
    }

    @Test
    void buildsPredictionsAndCountsDisagreements() throws Exception {
        List<PathObject> cells = detections(3);
        // model1: A, B, A   model2: A, A, B  → disagreements on cells 1 and 2
        var model1 = new SeqPredictor(new float[][] {{0.9f, 0.1f}, {0.1f, 0.9f}, {0.8f, 0.2f}});
        var model2 = new SeqPredictor(new float[][] {{0.7f, 0.3f}, {0.6f, 0.4f}, {0.2f, 0.8f}});

        Map<String, CellPrediction> sink = new LinkedHashMap<>();
        PredictionBatcher.Batch batch =
                PredictionBatcher.predict(cells, extractor(), 100, model1, model2, CLASSES, sink::put, end -> {});

        assertEquals(3, batch.objects().size());
        assertEquals(3, batch.classes().size());
        assertEquals(2, batch.disagreements(), "cells 1 and 2 should disagree");
        assertEquals(3, sink.size(), "every cell must reach the sink");

        // avgLabel: cell0 [.8,.2]→A, cell1 [.35,.65]→B, cell2 [.5,.5]→A (first max)
        assertEquals("A", batch.classes().get(0).getName());
        assertEquals("B", batch.classes().get(1).getName());
        assertEquals("A", batch.classes().get(2).getName());

        // The applied class must match each prediction's avgLabel.
        for (int i = 0; i < cells.size(); i++) {
            CellPrediction pred = sink.get(cells.get(i).getID().toString());
            assertEquals(pred.avgLabel(), batch.classes().get(i).getName());
        }
    }

    @Test
    void chunksCellsAndReportsCumulativeProgress() throws Exception {
        List<PathObject> cells = detections(5);
        float[][] probs = {{0.9f, 0.1f}, {0.9f, 0.1f}, {0.9f, 0.1f}, {0.9f, 0.1f}, {0.9f, 0.1f}};
        var model1 = new SeqPredictor(probs);
        var model2 = new SeqPredictor(probs);

        List<Integer> progress = new ArrayList<>();
        PredictionBatcher.Batch batch = PredictionBatcher.predict(
                cells,
                extractor(),
                2, // chunk size 2 → chunks of 2,2,1
                model1,
                model2,
                CLASSES,
                (id, p) -> {},
                progress::add);

        assertEquals(5, batch.objects().size());
        assertEquals(0, batch.disagreements());
        assertEquals(List.of(2, 4, 5), progress, "onChunkDone gets cumulative counts");
    }

    @Test
    void emptyCellListProducesEmptyBatch() throws Exception {
        List<Integer> progress = new ArrayList<>();
        PredictionBatcher.Batch batch = PredictionBatcher.predict(
                List.of(),
                extractor(),
                100,
                (d, s) -> new float[0][],
                (d, s) -> new float[0][],
                CLASSES,
                (id, p) -> fail("sink must not be called for empty input"),
                progress::add);

        assertTrue(batch.objects().isEmpty());
        assertTrue(batch.classes().isEmpty());
        assertEquals(0, batch.disagreements());
        assertTrue(progress.isEmpty(), "no chunks → no progress callbacks");
    }
}
