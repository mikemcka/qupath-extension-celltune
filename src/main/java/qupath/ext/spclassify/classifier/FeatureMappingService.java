package qupath.ext.spclassify.classifier;

import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Pure helpers for aligning imported/pooled training rows to the current model's
 * feature-column ordering.
 * <p>
 * Imported ground-truth rows (or rows pooled from another project) carry feature
 * values in <em>their</em> column order, which need not match the column order the
 * current model trains on. These helpers build a column index map by feature name
 * (case-insensitive, trimmed) and re-order a row's values into the target layout,
 * zero-filling any target column the source lacks.
 * <p>
 * Extracted from the duplicated {@code buildFeatureIndexMap} copies in
 * {@code ClassificationPanel} and {@code SpClassifyExtension} so the logic lives in one
 * directly unit-testable place.
 */
public final class FeatureMappingService {

    private FeatureMappingService() {} // utility class

    /**
     * Map each target feature column to its index in the source feature list.
     * <p>
     * Matching is case-insensitive and ignores surrounding whitespace. A target
     * feature that has no source counterpart maps to {@code -1}.
     *
     * @param sourceFeatureNames the source row's feature column names (e.g. imported panel)
     * @param targetFeatureNames the current model's feature column names
     * @return an array of length {@code targetFeatureNames.size()} where entry {@code i}
     *         is the source index for target column {@code i}, or {@code -1} if absent
     */
    public static int[] buildFeatureIndexMap(List<String> sourceFeatureNames, List<String> targetFeatureNames) {
        Map<String, Integer> sourceByName = new HashMap<>();
        for (int i = 0; i < sourceFeatureNames.size(); i++) {
            sourceByName.put(sourceFeatureNames.get(i).strip().toLowerCase(Locale.ROOT), i);
        }

        int[] map = new int[targetFeatureNames.size()];
        for (int i = 0; i < targetFeatureNames.size(); i++) {
            String key = targetFeatureNames.get(i).strip().toLowerCase(Locale.ROOT);
            map[i] = sourceByName.getOrDefault(key, -1);
        }
        return map;
    }

    /**
     * Re-order a source feature row into the target column layout described by a
     * {@code featureMap} from {@link #buildFeatureIndexMap}.
     * <p>
     * The returned array has one entry per target column. For each target column,
     * its source value is copied when the map points at a valid in-range source
     * index and the value is finite; otherwise the column is left as {@code 0f}
     * (so missing columns and non-finite source values are both zero-filled).
     *
     * @param sourceRow  the source row's feature values (in source column order)
     * @param featureMap target→source index map from {@link #buildFeatureIndexMap}
     * @return a new array of length {@code featureMap.length} in target column order
     */
    public static float[] alignRow(float[] sourceRow, int[] featureMap) {
        float[] aligned = new float[featureMap.length];
        for (int f = 0; f < aligned.length; f++) {
            int srcIdx = featureMap[f];
            if (srcIdx >= 0 && srcIdx < sourceRow.length) {
                float val = sourceRow[srcIdx];
                aligned[f] = Float.isFinite(val) ? val : 0f;
            }
        }
        return aligned;
    }
}
