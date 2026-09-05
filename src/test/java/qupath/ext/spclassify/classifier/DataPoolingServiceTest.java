package qupath.ext.spclassify.classifier;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;
import qupath.ext.spclassify.io.GroundTruthIO;

/**
 * Unit coverage for the imported-row pooling extracted from {@code ClassificationPanel.doTrain}.
 */
class DataPoolingServiceTest {

    private static GroundTruthIO.TrainingRow row(String label, float... features) {
        return new GroundTruthIO.TrainingRow(label, features);
    }

    @Test
    void alignsRowsToTargetColumnsAndCountsThem() {
        var pooled = DataPoolingService.poolImportedRows(
                List.of(row("Tumour", 1f, 2f, 3f), row("Stroma", 4f, 5f, 6f)),
                List.of("CD3", "CD8", "CD4"), // imported column order
                List.of("CD4", "CD3")); // target column order (subset)

        assertEquals(2, pooled.addedCount());
        assertEquals(2, pooled.mappedFeatureCount()); // both target cols found in import
        assertEquals(List.of("Tumour", "Stroma"), pooled.labels());
        // CD4=3rd import col, CD3=1st import col
        assertArrayEquals(new float[] {3f, 1f}, pooled.rows().get(0), 0f);
        assertArrayEquals(new float[] {6f, 4f}, pooled.rows().get(1), 0f);
    }

    @Test
    void zeroFillsTargetColumnsMissingFromImport() {
        var pooled = DataPoolingService.poolImportedRows(
                List.of(row("A", 9f)), List.of("CD3"), List.of("CD3", "FOXP3")); // FOXP3 absent from import

        assertEquals(1, pooled.addedCount());
        assertEquals(1, pooled.mappedFeatureCount());
        assertArrayEquals(new float[] {9f, 0f}, pooled.rows().get(0), 0f);
    }

    @Test
    void skipsRowsWithBlankLabelOrNullFeatures() {
        var pooled = DataPoolingService.poolImportedRows(
                java.util.Arrays.asList(
                        row("Good", 1f),
                        row("  ", 2f), // blank label -> skipped
                        new GroundTruthIO.TrainingRow("NoFeatures", null)), // null features -> skipped
                List.of("CD3"),
                List.of("CD3"));
        assertEquals(1, pooled.addedCount());
        assertEquals(List.of("Good"), pooled.labels());
    }

    @Test
    void returnsEmptyWhenNoImportedFeaturesMapToTarget() {
        var pooled = DataPoolingService.poolImportedRows(
                List.of(row("A", 1f)), List.of("PANEL_X"), List.of("CD3", "CD8")); // no overlap
        assertEquals(0, pooled.addedCount());
        assertEquals(0, pooled.mappedFeatureCount());
        assertTrue(pooled.rows().isEmpty());
        assertTrue(pooled.labels().isEmpty());
    }

    @Test
    void handlesNullAndEmptyInputs() {
        assertEquals(
                0,
                DataPoolingService.poolImportedRows(null, List.of("CD3"), List.of("CD3"))
                        .addedCount());
        assertEquals(
                0,
                DataPoolingService.poolImportedRows(List.of(), List.of("CD3"), List.of("CD3"))
                        .addedCount());
        assertEquals(
                0,
                DataPoolingService.poolImportedRows(List.of(row("A", 1f)), null, List.of("CD3"))
                        .addedCount());
        assertEquals(
                0,
                DataPoolingService.poolImportedRows(List.of(row("A", 1f)), List.of(), List.of("CD3"))
                        .addedCount());
    }
}
