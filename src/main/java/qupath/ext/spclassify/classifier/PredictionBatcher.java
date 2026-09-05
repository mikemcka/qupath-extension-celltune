package qupath.ext.spclassify.classifier;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.function.IntConsumer;
import javafx.application.Platform;
import qupath.ext.spclassify.model.CellFeatureExtractor;
import qupath.ext.spclassify.model.CellPrediction;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;

/**
 * Shared chunked-prediction loop for {@link DualModelClassifier}.
 * <p>
 * Three call sites (train-then-predict, predict-only, predict-and-collect) used to
 * carry byte-for-byte copies of the same loop: walk the cells in fixed-size chunks,
 * extract the feature matrix, run both models, build a {@link CellPrediction} per
 * cell, record the average-label {@link PathClass} for later application, and count
 * model disagreements. This class owns that loop once.
 * <p>
 * The two models are supplied as {@link ChunkPredictor} callbacks and the
 * population of result sets via a {@link PredictionSink}, so the loop holds no
 * classifier state and never touches the JavaFX thread — making it unit-testable
 * with stub predictors and no native ML models. Applying the resulting
 * classifications to the live hierarchy is a separate step
 * ({@link #applyOnFxThreadBlocking}).
 */
final class PredictionBatcher {

    private PredictionBatcher() {} // utility class

    /** Predicts class probabilities for one chunk of {@code chunkSize} rows. */
    @FunctionalInterface
    interface ChunkPredictor {
        /**
         * @param chunkData flat {@code [chunkSize * nFeatures]} feature matrix
         * @param chunkSize number of rows in this chunk
         * @return {@code [chunkSize][nClasses]} class probabilities
         */
        float[][] predict(float[] chunkData, int chunkSize) throws Exception;
    }

    /** Receives each cell's prediction so callers decide which population sets to fill. */
    @FunctionalInterface
    interface PredictionSink {
        void accept(String cellId, CellPrediction prediction);
    }

    /**
     * Result of a batch: the cells and the path classes to assign to them (parallel
     * lists, applied later on the FX thread) plus the inter-model disagreement count.
     */
    record Batch(List<PathObject> objects, List<PathClass> classes, int disagreements) {}

    /**
     * Run the chunked prediction loop.
     *
     * @param cellList    cells to predict (indexed access; chunked via {@code subList})
     * @param extractor   feature extractor (same feature columns as training)
     * @param chunkSize   rows per chunk (e.g. {@code PREDICT_CHUNK_SIZE})
     * @param model1      model-1 chunk predictor
     * @param model2      model-2 chunk predictor
     * @param classNames  class index → name
     * @param sink        receives every cell's prediction
     * @param onChunkDone called with the cumulative cell count after each chunk
     * @return the collected objects/classes to apply and the disagreement count
     */
    static Batch predict(
            List<PathObject> cellList,
            CellFeatureExtractor extractor,
            int chunkSize,
            ChunkPredictor model1,
            ChunkPredictor model2,
            List<String> classNames,
            PredictionSink sink,
            IntConsumer onChunkDone)
            throws Exception {
        int total = cellList.size();
        List<PathObject> objects = new ArrayList<>(total);
        List<PathClass> classes = new ArrayList<>(total);
        int disagreements = 0;

        for (int start = 0; start < total; start += chunkSize) {
            int end = Math.min(start + chunkSize, total);
            int size = end - start;
            List<PathObject> chunk = cellList.subList(start, end);

            float[] chunkData = extractor.extractMatrix(chunk);
            float[][] mdl1Probs = model1.predict(chunkData, size);
            float[][] mdl2Probs = model2.predict(chunkData, size);

            for (int i = 0; i < size; i++) {
                PathObject cell = chunk.get(i);
                String cellId = cell.getID().toString();

                String mdl1Label = classNames.get(argmax(mdl1Probs[i]));
                String mdl2Label = classNames.get(argmax(mdl2Probs[i]));

                CellPrediction pred =
                        new CellPrediction(cellId, mdl1Label, mdl2Label, mdl1Probs[i], mdl2Probs[i], classNames);

                objects.add(cell);
                classes.add(PathClass.fromString(pred.avgLabel()));
                sink.accept(cellId, pred);

                if (pred.isDisagreement()) disagreements++;
            }

            onChunkDone.accept(end);
        }

        return new Batch(objects, classes, disagreements);
    }

    /**
     * Apply the path classes to their objects on the JavaFX thread, blocking until
     * done so callers can persist immediately afterwards. Any {@link RuntimeException}
     * thrown on the FX thread is rethrown to the caller. Runs inline if already on
     * the FX thread.
     */
    static void applyOnFxThreadBlocking(List<PathObject> objects, List<PathClass> classes) {
        if (Platform.isFxApplicationThread()) {
            applyAll(objects, classes);
            return;
        }
        CountDownLatch done = new CountDownLatch(1);
        RuntimeException[] fxError = new RuntimeException[1];
        Platform.runLater(() -> {
            try {
                applyAll(objects, classes);
            } catch (RuntimeException ex) {
                fxError[0] = ex;
            } finally {
                done.countDown();
            }
        });
        try {
            done.await();
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while applying predictions.", ie);
        }
        if (fxError[0] != null) {
            throw fxError[0];
        }
    }

    private static void applyAll(List<PathObject> objects, List<PathClass> classes) {
        for (int i = 0; i < objects.size(); i++) {
            objects.get(i).setPathClass(classes.get(i));
        }
    }

    /** Index of the first maximum (matches {@code DualModelClassifier.argmax}). */
    private static int argmax(float[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[best]) best = i;
        }
        return best;
    }
}
