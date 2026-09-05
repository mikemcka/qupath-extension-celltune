package qupath.ext.spclassify.model;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Test;
import qupath.ext.spclassify.model.NeighborhoodModel.ClusterResult;

/**
 * Unit tests for {@link NeighborhoodModel}: the pure neighbor/composition/cluster
 * math behind cellular-neighborhood analysis (CN-01, CN-02, CN-04). Synthetic
 * clouds only — no QuPath/JavaFX APIs — mirroring {@code DistanceMeasurementsDialogTest}
 * and {@code ScatterMathTest}.
 */
class NeighborhoodModelTest {

    private static final double EPS = 1e-9;

    // ── kNN ──────────────────────────────────────────────────────────────────

    @Test
    void knnMatchesBruteForceSetOnRandomCloud() {
        Random rng = new Random(42);
        int n = 200;
        int k = 8;
        double[] xs = new double[n];
        double[] ys = new double[n];
        for (int i = 0; i < n; i++) {
            xs[i] = rng.nextDouble() * 1000.0;
            ys[i] = rng.nextDouble() * 1000.0;
        }
        int[][] actual = NeighborhoodModel.kNearestNeighborIndices(xs, ys, k);
        for (int i = 0; i < n; i++) {
            assertEquals(k, actual[i].length, "Expected exactly k neighbours at index " + i);
            assertFalse(toSet(actual[i]).contains(i), "Self must be excluded at index " + i);
            Set<Integer> expected = bruteForceKnn(xs, ys, i, k);
            assertEquals(expected, toSet(actual[i]), "kNN set mismatch at index " + i);
        }
    }

    @Test
    void knnExcludesSelfWithDuplicateCoordinates() {
        double[] xs = {7.0, 7.0, 100.0};
        double[] ys = {7.0, 7.0, 100.0};
        int[][] out = NeighborhoodModel.kNearestNeighborIndices(xs, ys, 1);
        assertEquals(1, out[0].length);
        assertEquals(1, out[0][0], "Duplicate at same location must match its twin, not itself");
        assertEquals(1, out[1].length);
        assertEquals(0, out[1][0]);
    }

    @Test
    void knnReturnsNMinusOneWhenFewerThanK() {
        double[] xs = {0.0, 1.0, 2.0};
        double[] ys = {0.0, 0.0, 0.0};
        int[][] out = NeighborhoodModel.kNearestNeighborIndices(xs, ys, 10);
        for (int i = 0; i < 3; i++) {
            assertEquals(2, out[i].length, "n-1 neighbours when n-1 < k at index " + i);
            assertFalse(toSet(out[i]).contains(i));
        }
    }

    @Test
    void knnNanCoordinatesAreEmptyAndIgnoredByOthers() {
        double[] xs = {0.0, Double.NaN, 10.0};
        double[] ys = {0.0, Double.NaN, 0.0};
        int[][] out = NeighborhoodModel.kNearestNeighborIndices(xs, ys, 1);
        assertEquals(0, out[1].length, "NaN-coord point has no window");
        assertEquals(1, out[0].length);
        assertEquals(2, out[0][0], "Valid point ignores the NaN neighbour");
        assertEquals(0, out[2][0], "Valid point ignores the NaN neighbour");
    }

    // ── radius ─────────────────────────────────────────────────────────────────

    @Test
    void radiusMatchesBruteForceOnRandomCloud() {
        Random rng = new Random(7);
        int n = 300;
        double radius = 80.0;
        double[] xs = new double[n];
        double[] ys = new double[n];
        for (int i = 0; i < n; i++) {
            xs[i] = rng.nextDouble() * 1000.0;
            ys[i] = rng.nextDouble() * 1000.0;
        }
        int[][] actual = NeighborhoodModel.radiusNeighborIndices(xs, ys, radius);
        double r2 = radius * radius;
        for (int i = 0; i < n; i++) {
            Set<Integer> expected = new HashSet<>();
            for (int j = 0; j < n; j++) {
                if (j == i) {
                    continue;
                }
                double dx = xs[i] - xs[j];
                double dy = ys[i] - ys[j];
                if (dx * dx + dy * dy <= r2) {
                    expected.add(j);
                }
            }
            assertEquals(expected, toSet(actual[i]), "radius window mismatch at index " + i);
            assertFalse(toSet(actual[i]).contains(i), "Self must be excluded at index " + i);
        }
    }

    @Test
    void radiusTooSmallYieldsEmptyWindows() {
        double[] xs = {0.0, 10.0, 20.0};
        double[] ys = {0.0, 0.0, 0.0};
        int[][] out = NeighborhoodModel.radiusNeighborIndices(xs, ys, 1.0);
        for (int[] w : out) {
            assertEquals(0, w.length);
        }
    }

    // ── Delaunay ─────────────────────────────────────────────────────────────────

    @Test
    void delaunaySquareHasFourSidesAndOneDiagonal() {
        // Unit square: Delaunay = the 4 sides + exactly one of the two (degenerate) diagonals.
        double[] xs = {0.0, 1.0, 1.0, 0.0};
        double[] ys = {0.0, 0.0, 1.0, 1.0};
        int[][] adj = NeighborhoodModel.delaunayNeighborIndices(xs, ys, Double.POSITIVE_INFINITY);
        assertSymmetricAndSelfExcluded(adj);
        // The four sides are always present regardless of which diagonal is chosen.
        assertTrue(toSet(adj[0]).contains(1) && toSet(adj[0]).contains(3), "corner 0 sides");
        assertTrue(toSet(adj[1]).contains(0) && toSet(adj[1]).contains(2), "corner 1 sides");
        assertTrue(toSet(adj[2]).contains(1) && toSet(adj[2]).contains(3), "corner 2 sides");
        assertTrue(toSet(adj[3]).contains(2) && toSet(adj[3]).contains(0), "corner 3 sides");
        assertEquals(5, undirectedEdgeCount(adj), "4 sides + 1 diagonal");
    }

    @Test
    void delaunayNearestNeighbourIsAlwaysAnEdge() {
        // A well-known Delaunay property: each point's single nearest neighbour is a
        // Delaunay neighbour. Cheap, strong correctness check without a reference triangulator.
        Random rng = new Random(19);
        int n = 150;
        double[] xs = new double[n];
        double[] ys = new double[n];
        for (int i = 0; i < n; i++) {
            xs[i] = rng.nextDouble() * 1000.0;
            ys[i] = rng.nextDouble() * 1000.0;
        }
        int[][] adj = NeighborhoodModel.delaunayNeighborIndices(xs, ys, Double.POSITIVE_INFINITY);
        assertSymmetricAndSelfExcluded(adj);
        for (int i = 0; i < n; i++) {
            int nn = -1;
            double best = Double.POSITIVE_INFINITY;
            for (int j = 0; j < n; j++) {
                if (j == i) {
                    continue;
                }
                double dx = xs[i] - xs[j];
                double dy = ys[i] - ys[j];
                double d2 = dx * dx + dy * dy;
                if (d2 < best) {
                    best = d2;
                    nn = j;
                }
            }
            assertTrue(toSet(adj[i]).contains(nn), "nearest neighbour of " + i + " must be a Delaunay edge");
        }
    }

    @Test
    void delaunayFixedPruneIsolatesFarOutlier() {
        // A tight cluster plus one point far beyond the cutoff: the outlier's edges are
        // all pruned (empty window → CN = -1 downstream) while the cluster stays connected.
        double[] xs = {0.0, 1.0, 0.0, 1.0, 0.5, 500.0};
        double[] ys = {0.0, 0.0, 1.0, 1.0, 0.5, 500.0};
        int[][] adj = NeighborhoodModel.delaunayNeighborIndices(xs, ys, 5.0);
        assertSymmetricAndSelfExcluded(adj);
        assertEquals(0, adj[5].length, "far outlier must have no neighbours after pruning");
        for (int i = 0; i < 5; i++) {
            assertFalse(toSet(adj[i]).contains(5), "no cluster cell links to the outlier");
            assertTrue(adj[i].length > 0, "cluster cell " + i + " keeps its short edges");
        }
    }

    @Test
    void delaunayAutoPrunesLongBorderEdge() {
        // 3x3 grid (spacing 10) + one distant point. The auto (Q3+1.5·IQR) cutoff is set
        // by the grid's ~10–14 unit edges, so the ~hundreds-of-units links to the outlier
        // are pruned automatically, isolating it.
        int gi = 0;
        double[] xs = new double[10];
        double[] ys = new double[10];
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                xs[gi] = c * 10.0;
                ys[gi] = r * 10.0;
                gi++;
            }
        }
        xs[9] = 400.0;
        ys[9] = 400.0;
        int[][] adj = NeighborhoodModel.delaunayNeighborIndicesAuto(xs, ys);
        assertSymmetricAndSelfExcluded(adj);
        assertEquals(0, adj[9].length, "distant point pruned by the auto whisker");
        for (int i = 0; i < 9; i++) {
            assertTrue(adj[i].length > 0, "grid cell " + i + " keeps neighbours");
            assertFalse(toSet(adj[i]).contains(9), "no grid cell links across the void");
        }
    }

    @Test
    void delaunayNanCoordinatesAreEmptyAndIgnored() {
        double[] xs = {0.0, 1.0, Double.NaN, 0.0};
        double[] ys = {0.0, 0.0, Double.NaN, 1.0};
        int[][] adj = NeighborhoodModel.delaunayNeighborIndices(xs, ys, Double.POSITIVE_INFINITY);
        assertEquals(0, adj[2].length, "NaN-coord point has no window");
        for (int i = 0; i < 4; i++) {
            assertFalse(toSet(adj[i]).contains(2), "NaN point is not a neighbour of anyone");
        }
        assertSymmetricAndSelfExcluded(adj);
    }

    @Test
    void delaunayDuplicateCoordinatesConnect() {
        // Two coincident cells (0 and 1) collapse to one JTS vertex; both must still connect
        // to each other and share the triangulation neighbours of that vertex.
        double[] xs = {5.0, 5.0, 0.0, 10.0, 5.0};
        double[] ys = {5.0, 5.0, 0.0, 0.0, 12.0};
        int[][] adj = NeighborhoodModel.delaunayNeighborIndices(xs, ys, Double.POSITIVE_INFINITY);
        assertSymmetricAndSelfExcluded(adj);
        assertTrue(toSet(adj[0]).contains(1), "coincident twins must connect to each other");
        assertTrue(toSet(adj[1]).contains(0));
        // Whatever the shared vertex links to, both twins carry it identically.
        Set<Integer> a0 = toSet(adj[0]);
        Set<Integer> a1 = toSet(adj[1]);
        a0.remove(1);
        a1.remove(0);
        assertEquals(a0, a1, "twins share the same external neighbours");
    }

    @Test
    void delaunayNonPositiveMaxEdgeMeansNoPruning() {
        double[] xs = {0.0, 1.0, 1.0, 0.0};
        double[] ys = {0.0, 0.0, 1.0, 1.0};
        int[][] pruned = NeighborhoodModel.delaunayNeighborIndices(xs, ys, 0.0);
        int[][] raw = NeighborhoodModel.delaunayNeighborIndices(xs, ys, Double.POSITIVE_INFINITY);
        assertEquals(undirectedEdgeCount(raw), undirectedEdgeCount(pruned), "maxEdge<=0 must not prune");
    }

    @Test
    void tukeyUpperWhiskerMatchesFivenumFence() {
        // fivenum(1..9) hinges = Q1=3, Q3=7 → IQR=4 → fence = 7 + 1.5*4 = 13.
        double[] vals = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        assertEquals(13.0, NeighborhoodModel.tukeyUpperWhisker(vals), EPS);
        // Non-finite entries are ignored; empty → NaN (callers treat as "no prune").
        assertTrue(Double.isNaN(NeighborhoodModel.tukeyUpperWhisker(new double[] {Double.NaN})));
        assertEquals(
                13.0, NeighborhoodModel.tukeyUpperWhisker(new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, Double.NaN}), EPS);
    }

    // ── composition ─────────────────────────────────────────────────────────────

    @Test
    void compositionRowsSumToOneAndHonorIncludeCenter() {
        // typeId: cells 0,1 = type0; cell2 = type1; cell3 = ignored.
        int[] typeId = {0, 0, 1, -1};
        int[][] neighbors = {{1, 2, 3}, {0}, {0, 1}, {}};
        int nTypes = 2;

        double[][] noCenter = NeighborhoodModel.compositionMatrix(neighbors, typeId, nTypes, false);
        // cell0 window {1(t0), 2(t1), 3(ignored)} → counts t0=1,t1=1, base=2.
        assertArrayEqualsD(new double[] {0.5, 0.5}, noCenter[0]);
        assertEquals(1.0, noCenter[0][0] + noCenter[0][1], EPS);
        // cell3 is isolated (empty window) → all-zero row.
        assertArrayEqualsD(new double[] {0.0, 0.0}, noCenter[3]);

        double[][] withCenter = NeighborhoodModel.compositionMatrix(neighbors, typeId, nTypes, true);
        // cell0 + center(t0): t0=2,t1=1, base=3.
        assertArrayEqualsD(new double[] {2.0 / 3.0, 1.0 / 3.0}, withCenter[0]);
    }

    @Test
    void compositionExcludesIgnoredFromCountsAndBase() {
        int[] typeId = {0, -1, -1};
        int[][] neighbors = {{1, 2}, {}, {}};
        double[][] comp = NeighborhoodModel.compositionMatrix(neighbors, typeId, 1, false);
        // cell0's only neighbours are ignored → empty effective window → zero row.
        assertArrayEqualsD(new double[] {0.0}, comp[0]);
    }

    // ── clustering ───────────────────────────────────────────────────────────────

    @Test
    void clusterRecoversTwoSeparatedBlobsByPurity() {
        Random rng = new Random(11);
        int per = 60;
        int n = per * 2;
        double[][] comp = new double[n][2];
        for (int i = 0; i < per; i++) {
            comp[i][0] = 0.95 + rng.nextDouble() * 0.05; // blob A near (1,0)
            comp[i][1] = 1.0 - comp[i][0];
        }
        for (int i = per; i < n; i++) {
            comp[i][1] = 0.95 + rng.nextDouble() * 0.05; // blob B near (0,1)
            comp[i][0] = 1.0 - comp[i][1];
        }
        ClusterResult res = NeighborhoodModel.clusterCompositions(comp, 2);
        assertEquals(2, res.kEffective());
        // Each blob should be internally pure: one dominant label covering it.
        assertTrue(purity(res.labels(), 0, per) > 0.95, "Blob A not pure");
        assertTrue(purity(res.labels(), per, n) > 0.95, "Blob B not pure");
        // And the two blobs must land in different clusters.
        assertFalse(
                res.labels()[0] == res.labels()[n - 1] && purity(res.labels(), 0, n) > 0.95,
                "Blobs should be separated into distinct clusters");
    }

    @Test
    void clusterKEffectiveCappedAtRowCount() {
        double[][] comp = {{1, 0}, {0, 1}, {0.5, 0.5}};
        ClusterResult res = NeighborhoodModel.clusterCompositions(comp, 10);
        assertEquals(3, res.kEffective(), "kEffective must be min(k, nRows)");
    }

    @Test
    void multiRestartNeverWorseThanSingleInitOnHardProblem() {
        // Uniform-random high-dim cloud with k=10 has many local optima, so a
        // single k-means run is init-sensitive. Keeping the best of nInit restarts
        // must yield a partition at least as tight (lower/equal inertia) as one run.
        Random rng = new Random(7);
        int n = 600;
        int dim = 12;
        double[][] comp = new double[n][dim];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dim; j++) {
                comp[i][j] = rng.nextDouble();
            }
        }
        double singleInertia = inertia(NeighborhoodModel.clusterCompositions(comp, 10, 1), comp);
        double multiInertia = inertia(NeighborhoodModel.clusterCompositions(comp, 10, 20), comp);
        assertTrue(
                multiInertia <= singleInertia + EPS,
                "n_init=20 (" + multiInertia + ") should not be worse than n_init=1 (" + singleInertia + ")");
    }

    @Test
    void clusterCompositionsIsSeededAndReproducibleAcrossRuns() {
        // NeighborhoodModel.SEED must be set once before the restart loop (mirroring
        // ScatterPlotView's KMEANS_SEED), so two independent calls over the SAME
        // composition matrix must yield EXACT-equality identical per-cell labels.
        Random rng = new Random(55);
        int n = 400;
        int dim = 6;
        double[][] comp = new double[n][dim];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dim; j++) {
                comp[i][j] = rng.nextDouble();
            }
        }

        ClusterResult a = NeighborhoodModel.clusterCompositions(comp, 5, 8);
        ClusterResult b = NeighborhoodModel.clusterCompositions(comp, 5, 8);

        assertArrayEquals(a.labels(), b.labels(), "seeded k-means must produce identical labels run-to-run");
        assertEquals(a.kEffective(), b.kEffective());
        for (int c = 0; c < a.centroids().length; c++) {
            assertArrayEqualsD(a.centroids()[c], b.centroids()[c]);
        }
    }

    @Test
    void clusterTreatsNonPositiveNInitAsSingleRun() {
        // nInit is clamped to >= 1: zero/negative must still cluster, not crash.
        double[][] comp = {{1, 0}, {0.95, 0.05}, {0, 1}, {0.05, 0.95}};
        ClusterResult res = NeighborhoodModel.clusterCompositions(comp, 2, 0);
        assertEquals(2, res.kEffective());
        assertTrue(purity(res.labels(), 0, 2) > 0.95 && purity(res.labels(), 2, 4) > 0.95, "Blobs not recovered");
    }

    /** Within-cluster sum of squared distances of rows to their assigned cluster mean. */
    private static double inertia(ClusterResult res, double[][] data) {
        double[][] c = res.centroids();
        int[] labels = res.labels();
        double sum = 0;
        for (int i = 0; i < data.length; i++) {
            int g = labels[i];
            if (g < 0 || g >= c.length) {
                continue;
            }
            for (int j = 0; j < data[i].length; j++) {
                double d = data[i][j] - c[g][j];
                sum += d * d;
            }
        }
        return sum;
    }

    // ── cnMeanComposition ──────────────────────────────────────────────────────

    @Test
    void cnMeanCompositionEqualsHandComputedColumnMeans() {
        int[] labels = {0, 0, 1};
        double[][] comp = {{1.0, 0.0}, {0.5, 0.5}, {0.0, 1.0}};
        double[][] mean = NeighborhoodModel.cnMeanComposition(labels, comp, 2);
        assertArrayEqualsD(new double[] {0.75, 0.25}, mean[0]);
        assertArrayEqualsD(new double[] {0.0, 1.0}, mean[1]);
    }

    // ── diversity + adjacency ─────────────────────────────────────────────────────

    @Test
    void compositionDiversityRewardsEvennessAndRichness() {
        assertEquals(0.0, NeighborhoodModel.compositionDiversity(new double[] {1, 0, 0, 0}), EPS, "single type → 0");
        assertEquals(0.0, NeighborhoodModel.compositionDiversity(new double[] {0, 0, 0, 0}), EPS, "empty → 0");
        assertEquals(
                1.0,
                NeighborhoodModel.compositionDiversity(new double[] {0.25, 0.25, 0.25, 0.25}),
                EPS,
                "even over all types → 1");
        // Even over 2 of 4 types: ln(2)/ln(4) = 0.5.
        assertEquals(0.5, NeighborhoodModel.compositionDiversity(new double[] {0.5, 0.5, 0, 0}), EPS);
        // More even mixes score higher than dominated ones.
        double mixed = NeighborhoodModel.compositionDiversity(new double[] {0.4, 0.3, 0.3, 0.0});
        double dominated = NeighborhoodModel.compositionDiversity(new double[] {0.9, 0.05, 0.05, 0.0});
        assertTrue(mixed > dominated, "more even mix should be more diverse");
    }

    @Test
    void cnAdjacencyCountsTouchingPairsBothDirections() {
        int[] cn = {1, 2, 1};
        int[][] neighbors = {{1}, {0, 2}, {1}};
        double[][] adj = NeighborhoodModel.cnAdjacency(neighbors, cn, 2);
        assertEquals(2.0, adj[0][1], EPS); // CN1 sees CN2: cell0→1 and cell2→1
        assertEquals(2.0, adj[1][0], EPS); // CN2 sees CN1: cell1→0 and cell1→2
        // Same-CN neighbours and out-of-range ids contribute nothing.
        int[] cnWithEmpty = {1, -1, 1};
        double[][] adj2 = NeighborhoodModel.cnAdjacency(new int[][] {{1, 2}, {0}, {0}}, cnWithEmpty, 2);
        assertEquals(0.0, adj2[0][0], EPS);
        assertEquals(0.0, adj2[0][1], EPS);
    }

    // ── helpers ─────────────────────────────────────────────────────────────────

    private static Set<Integer> bruteForceKnn(double[] xs, double[] ys, int i, int k) {
        int n = xs.length;
        Integer[] order = new Integer[n];
        double[] d2 = new double[n];
        for (int j = 0; j < n; j++) {
            order[j] = j;
            double dx = xs[i] - xs[j];
            double dy = ys[i] - ys[j];
            d2[j] = (j == i) ? Double.POSITIVE_INFINITY : dx * dx + dy * dy;
        }
        Arrays.sort(order, (p, q) -> Double.compare(d2[p], d2[q]));
        Set<Integer> set = new HashSet<>();
        for (int t = 0; t < k && t < n; t++) {
            set.add(order[t]);
        }
        return set;
    }

    private static void assertSymmetricAndSelfExcluded(int[][] adj) {
        for (int i = 0; i < adj.length; i++) {
            for (int j : adj[i]) {
                assertFalse(j == i, "self edge at " + i);
                assertTrue(toSet(adj[j]).contains(i), "asymmetric edge " + i + "-" + j);
            }
        }
    }

    /** Distinct undirected edges (counts each {i,j} once, tolerant of repeated list entries). */
    private static int undirectedEdgeCount(int[][] adj) {
        int count = 0;
        for (int i = 0; i < adj.length; i++) {
            for (int j : toSet(adj[i])) {
                if (j > i) {
                    count++;
                }
            }
        }
        return count;
    }

    private static Set<Integer> toSet(int[] a) {
        Set<Integer> s = new HashSet<>();
        for (int v : a) {
            s.add(v);
        }
        return s;
    }

    /** Fraction of rows in [from,to) that carry the most common label in that range. */
    private static double purity(int[] labels, int from, int to) {
        java.util.Map<Integer, Integer> counts = new java.util.HashMap<>();
        for (int i = from; i < to; i++) {
            counts.merge(labels[i], 1, Integer::sum);
        }
        int max = counts.values().stream().mapToInt(Integer::intValue).max().orElse(0);
        return max / (double) (to - from);
    }

    private static void assertArrayEqualsD(double[] expected, double[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], EPS, "Mismatch at column " + i);
        }
    }
}
