package qupath.ext.spclassify.classifier;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Unit coverage for the pure feature-alignment helpers extracted from the duplicated
 * {@code buildFeatureIndexMap} copies in {@code ClassificationPanel}/{@code SpClassifyExtension}.
 */
class FeatureMappingServiceTest {

    @Test
    void buildsIndexMapInTargetOrder() {
        int[] map = FeatureMappingService.buildFeatureIndexMap(
                List.of("CD3", "CD8", "CD4"), // source order
                List.of("CD4", "CD3", "CD8")); // target order
        assertArrayEquals(new int[] {2, 0, 1}, map);
    }

    @Test
    void matchingIsCaseInsensitiveAndTrimmed() {
        int[] map = FeatureMappingService.buildFeatureIndexMap(List.of("  cd3 ", "CD8"), List.of("CD3", "cd8"));
        assertArrayEquals(new int[] {0, 1}, map);
    }

    @Test
    void missingTargetFeaturesMapToMinusOne() {
        int[] map = FeatureMappingService.buildFeatureIndexMap(List.of("CD3"), List.of("CD3", "CD8", "FOXP3"));
        assertArrayEquals(new int[] {0, -1, -1}, map);
    }

    @Test
    void emptyTargetGivesEmptyMap() {
        assertEquals(0, FeatureMappingService.buildFeatureIndexMap(List.of("CD3"), List.of()).length);
    }

    @Test
    void alignRowReordersAndZeroFillsMissing() {
        // target columns: [CD4, CD3, CD8, FOXP3]; source order: [CD3, CD8, CD4]
        int[] map = FeatureMappingService.buildFeatureIndexMap(
                List.of("CD3", "CD8", "CD4"), List.of("CD4", "CD3", "CD8", "FOXP3"));
        float[] src = {1f /*CD3*/, 2f /*CD8*/, 3f /*CD4*/};
        float[] aligned = FeatureMappingService.alignRow(src, map);
        // CD4=3, CD3=1, CD8=2, FOXP3 missing -> 0
        assertArrayEquals(new float[] {3f, 1f, 2f, 0f}, aligned, 0f);
    }

    @Test
    void alignRowZeroFillsNonFiniteSourceValues() {
        int[] map = {0, 1, 2};
        float[] src = {Float.NaN, Float.POSITIVE_INFINITY, 5f};
        assertArrayEquals(new float[] {0f, 0f, 5f}, FeatureMappingService.alignRow(src, map), 0f);
    }

    @Test
    void alignRowTreatsOutOfRangeSourceIndexAsMissing() {
        // featureMap points at a source index beyond the (shorter) source row.
        int[] map = {0, 5};
        float[] src = {7f};
        assertArrayEquals(new float[] {7f, 0f}, FeatureMappingService.alignRow(src, map), 0f);
    }
}
