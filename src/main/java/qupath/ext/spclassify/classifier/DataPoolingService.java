package qupath.ext.spclassify.classifier;

import java.util.ArrayList;
import java.util.List;
import qupath.ext.spclassify.io.GroundTruthIO;

/**
 * Pure helpers for assembling supplementary training data from explicitly imported
 * ground-truth rows.
 * <p>
 * Imported rows carry their own feature panel; this aligns each row to the current
 * model's feature-column order (via {@link FeatureMappingService}), dropping rows with
 * no usable label/features and skipping the whole import when none of the imported
 * features map onto the current panel.
 * <p>
 * Extracted from {@code ClassificationPanel.doTrain} so the import-pooling logic is
 * directly unit-testable. The IO-bound cross-image label pooling (which must open other
 * project images) stays in {@code doTrain}.
 */
public final class DataPoolingService {

    private DataPoolingService() {} // utility class

    /**
     * The aligned supplementary rows produced from imported training data, with the
     * counts {@code doTrain} surfaces in its progress log.
     *
     * @param rows               aligned feature rows in target column order (never null)
     * @param labels             class names parallel to {@code rows} (never null)
     * @param mappedFeatureCount how many target columns had an imported counterpart
     * @param addedCount         number of rows actually emitted ({@code == rows.size()})
     */
    public record PooledRows(List<float[]> rows, List<String> labels, int mappedFeatureCount, int addedCount) {}

    /**
     * Align and collect imported training rows into the current model's feature layout.
     * <p>
     * Returns empty results (no rows, {@code mappedFeatureCount == 0}) when there is
     * nothing to import or when none of the imported features map onto the target panel.
     * Rows with a null/blank label or null features are skipped.
     *
     * @param importedRows        imported labelled feature rows (nullable)
     * @param importedFeatureNames feature column names for the imported rows (nullable)
     * @param targetFeatureNames   the current model's feature column ordering
     * @return the aligned rows/labels and the mapped/added counts
     */
    public static PooledRows poolImportedRows(
            List<GroundTruthIO.TrainingRow> importedRows,
            List<String> importedFeatureNames,
            List<String> targetFeatureNames) {
        List<float[]> rows = new ArrayList<>();
        List<String> labels = new ArrayList<>();

        if (importedRows == null
                || importedRows.isEmpty()
                || importedFeatureNames == null
                || importedFeatureNames.isEmpty()) {
            return new PooledRows(rows, labels, 0, 0);
        }

        int[] featureMap = FeatureMappingService.buildFeatureIndexMap(importedFeatureNames, targetFeatureNames);
        int mappedFeatureCount = 0;
        for (int idx : featureMap) {
            if (idx >= 0) mappedFeatureCount++;
        }
        if (mappedFeatureCount == 0) {
            return new PooledRows(rows, labels, 0, 0);
        }

        for (var row : importedRows) {
            if (row == null || row.label() == null || row.label().isBlank()) continue;
            float[] src = row.features();
            if (src == null) continue;
            rows.add(FeatureMappingService.alignRow(src, featureMap));
            labels.add(row.label());
        }
        return new PooledRows(rows, labels, mappedFeatureCount, rows.size());
    }
}
