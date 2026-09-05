package qupath.ext.spclassify.classifier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Unit coverage for the pre-extracted ({@code float[][]}) pruning overload used to
 * prune the full, normalised cross-image training matrix (current + pooled rows)
 * rather than one image's raw measurements.
 */
class FeaturePrunerTest {

    // minNonZeroCells=1, within-marker r>0.95, cross-marker off, whitelist off (minKeptPerGroup=0)
    private static final FeaturePruner.PruneOptions OPTS = new FeaturePruner.PruneOptions(1, 0.95, 1.0, 0);

    @Test
    void dropsNearConstantColumn() {
        List<String> names = List.of("CD3: Mean", "CD8: Mean");
        // CD3 varies across rows; CD8 is constant (zero variance).
        float[][] rows = {
            {1f, 5f}, {2f, 5f}, {3f, 5f}, {4f, 5f}, {5f, 5f}, {6f, 5f},
        };
        FeaturePruner.PruneResult pr = FeaturePruner.prune(rows, names, OPTS, null);
        assertEquals(List.of("CD3: Mean"), pr.keptFeatures());
        assertEquals(1, pr.droppedConstant());
    }

    @Test
    void dropsRedundantWithinMarker() {
        List<String> names = List.of("CD3: Mean", "CD3: Median", "CD8: Mean");
        // CD3 Median == 2 * CD3 Mean -> perfectly correlated within the CD3 group.
        // CD8 is independent and in its own group.
        float[][] rows = {
            {1f, 2f, 9f}, {2f, 4f, 1f}, {3f, 6f, 7f}, {4f, 8f, 2f}, {5f, 10f, 8f}, {6f, 12f, 3f},
        };
        FeaturePruner.PruneResult pr = FeaturePruner.prune(rows, names, OPTS, null);
        assertEquals(1, pr.droppedWithinMarker());
        assertEquals(2, pr.keptFeatures().size());
        assertTrue(pr.keptFeatures().contains("CD8: Mean"));
        // Exactly one of the two correlated CD3 features survives.
        assertTrue(pr.keptFeatures().contains("CD3: Mean") ^ pr.keptFeatures().contains("CD3: Median"));
    }

    @Test
    void keepsFeatureThatOnlyVariesAcrossPooledRows() {
        // The whole point of pruning after pooling: a feature flat within the first
        // image's cells but varying once other images are pooled must be kept.
        List<String> names = List.of("CD3: Mean", "CD20: Mean");
        // CD20 is constant in the first 4 rows (the "current image") but varies in the
        // last 4 ("pooled" rows) -> across the full matrix it is NOT near-constant.
        float[][] rows = {
            {1f, 7f}, {2f, 7f}, {3f, 7f}, {4f, 7f}, {5f, 2f}, {6f, 9f}, {7f, 1f}, {8f, 8f},
        };
        FeaturePruner.PruneResult pr = FeaturePruner.prune(rows, names, OPTS, null);
        assertTrue(pr.keptFeatures().contains("CD20: Mean"), "feature varying across pooled rows must survive");
        assertEquals(0, pr.droppedConstant());

        // Sanity: judged on only the first image's rows, CD20 IS near-constant and dropped.
        float[][] currentOnly = {{1f, 7f}, {2f, 7f}, {3f, 7f}, {4f, 7f}};
        FeaturePruner.PruneResult prCurrent = FeaturePruner.prune(currentOnly, names, OPTS, null);
        assertFalse(
                prCurrent.keptFeatures().contains("CD20: Mean"),
                "feature flat within current image alone is dropped — the bug this reorder fixes");
    }

    @Test
    void emptyRowsKeepsAllFeatures() {
        List<String> names = List.of("CD3: Mean", "CD8: Mean");
        FeaturePruner.PruneResult pr =
                FeaturePruner.prune(new float[0][], names, FeaturePruner.PruneOptions.defaults(), null);
        assertEquals(names, pr.keptFeatures());
    }

    @Test
    void rejectsRaggedRows() {
        List<String> names = List.of("a", "b");
        float[][] rows = {{1f, 2f}, {3f}};
        assertThrows(
                IllegalArgumentException.class,
                () -> FeaturePruner.prune(rows, names, FeaturePruner.PruneOptions.defaults(), null));
    }

    @Test
    void groupsByColonThenUnderscoreThenSpace() {
        // Group keys are lower-cased (case-insensitive grouping).
        // Marker convention: text before the first ": ".
        assertEquals("cd3", FeaturePruner.extractGroup("CD3: Cell: Mean"));
        // ": " wins even when a space appears earlier.
        assertEquals("some marker", FeaturePruner.extractGroup("Some Marker: Mean"));
        // No colon -> token before the first underscore.
        assertEquals("kronos", FeaturePruner.extractGroup("kronos_emb_0"));
        assertEquals("embedding", FeaturePruner.extractGroup("embedding_12"));
        // No colon -> token before the first space.
        assertEquals("distance", FeaturePruner.extractGroup("Distance to tumor"));
        // Underscore and space both present, no colon -> whichever is earlier wins.
        assertEquals("kronos", FeaturePruner.extractGroup("kronos_emb dim"));
        assertEquals("distance", FeaturePruner.extractGroup("Distance to tumor_region"));
        // No recognised separator -> own singleton group.
        assertEquals("solo", FeaturePruner.extractGroup("solo"));
        // Separator at position 0 -> whole name (empty prefix not allowed).
        assertEquals("_lead", FeaturePruner.extractGroup("_lead"));
    }

    @Test
    void groupingIsCaseInsensitive() {
        // Different-case prefixes collapse to the same group key.
        assertEquals(FeaturePruner.extractGroup("CD3: Mean"), FeaturePruner.extractGroup("cd3: Max"));
        assertEquals(FeaturePruner.extractGroup("Kronos_emb_0"), FeaturePruner.extractGroup("kronos_emb_1"));
        // So two case-variant, correlated features land in one group and get deduped.
        List<String> names = List.of("Kronos_emb_0", "kronos_emb_1");
        float[][] rows = {{1f, 2f}, {2f, 4f}, {3f, 6f}, {4f, 8f}, {5f, 10f}, {6f, 12f}};
        FeaturePruner.PruneResult pr = FeaturePruner.prune(rows, names, OPTS, null);
        assertEquals(1, pr.keptFeatures().size());
        assertEquals(1, pr.droppedWithinMarker());
    }

    @Test
    void whitelistKeepsTopFivePerGroupEvenIfRedundant() {
        // Two perfectly-correlated CD3 features. With the default top-5 whitelist and a
        // group of <=5 features, BOTH survive despite the within-marker redundancy.
        List<String> names = List.of("CD3: Mean", "CD3: Median");
        float[][] rows = {{1f, 2f}, {2f, 4f}, {3f, 6f}, {4f, 8f}, {5f, 10f}, {6f, 12f}};
        FeaturePruner.PruneResult pr = FeaturePruner.prune(rows, names, FeaturePruner.PruneOptions.defaults(), null);
        assertEquals(2, pr.keptFeatures().size());
        assertTrue(pr.keptFeatures().contains("CD3: Mean"));
        assertTrue(pr.keptFeatures().contains("CD3: Median"));
        // Net redundancy count is 0: the dropped peer was re-added by the whitelist.
        assertEquals(0, pr.droppedWithinMarker());
    }

    @Test
    void whitelistKeepsTopFiveInLargerGroupAndPrunesBeyond() {
        // One "M" group of 6 mutually-correlated features with distinct variances.
        // Default whitelist protects the top 5 by variance; the 6th (lowest variance)
        // is redundant, outside the whitelist, and dropped.
        List<String> names = List.of("M: a", "M: b", "M: c", "M: d", "M: e", "M: f");
        float[][] rows = {
            {10f, 8f, 6f, 4f, 3f, 0.1f},
            {-10f, -8f, -6f, -4f, -3f, -0.1f},
            {10f, 8f, 6f, 4f, 3f, 0.1f},
            {-10f, -8f, -6f, -4f, -3f, -0.1f},
            {10f, 8f, 6f, 4f, 3f, 0.1f},
            {-10f, -8f, -6f, -4f, -3f, -0.1f},
        };
        FeaturePruner.PruneResult pr = FeaturePruner.prune(rows, names, FeaturePruner.PruneOptions.defaults(), null);
        assertEquals(5, pr.keptFeatures().size());
        assertTrue(pr.keptFeatures().contains("M: a"));
        assertFalse(pr.keptFeatures().contains("M: f"), "lowest-variance redundant feature is not in the top 5");
    }

    @Test
    void underscoreGroupedFeaturesAreDeduplicated() {
        // Two perfectly-correlated kronos embedding dims now share the "kronos" group,
        // so within-marker correlation removal drops one (previously each was a
        // singleton and neither was ever compared).
        List<String> names = List.of("kronos_emb_0", "kronos_emb_1", "Distance to tumor");
        // emb_1 == 3 * emb_0 -> |r| = 1 within the kronos group; distance is independent.
        float[][] rows = {
            {1f, 3f, 9f}, {2f, 6f, 1f}, {3f, 9f, 7f}, {4f, 12f, 2f}, {5f, 15f, 8f}, {6f, 18f, 3f},
        };
        FeaturePruner.PruneResult pr = FeaturePruner.prune(rows, names, OPTS, null);
        assertEquals(1, pr.droppedWithinMarker());
        assertTrue(pr.keptFeatures().contains("Distance to tumor"));
        assertTrue(
                pr.keptFeatures().contains("kronos_emb_0") ^ pr.keptFeatures().contains("kronos_emb_1"),
                "exactly one of the two correlated kronos embeddings survives");
    }
}
