package qupath.ext.spclassify.model;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link HnswKnnIndex}: the ANN (HNSW) kNN graph builder that
 * Leiden routes through instead of {@link LeidenModel}'s brute-force {@code
 * featureKnn} (LEI-07). Synthetic clouds only, mirroring {@link
 * LeidenModelTest}'s conventions.
 */
class HnswKnnIndexTest {

    // ── Test A: recall vs exact ──────────────────────────────────────────────

    @Test
    void knnRecallsAtLeast95PercentOfExactNeighboursOnMultiBlobCloud() {
        Random rng = new Random(7);
        int perBlob = 60;
        int blobs = 5;
        int n = perBlob * blobs;
        int d = 5;
        int k = 10;
        double[][] rows = new double[n][d];
        for (int b = 0; b < blobs; b++) {
            fillBlob(rows, rng, b * perBlob, (b + 1) * perBlob, d, b * 12.0, 0.6);
        }

        int[][] approx = HnswKnnIndex.knn(rows, k, 42L, true);
        int[][] exact = LeidenModel.featureKnn(rows, k);

        double totalOverlap = 0;
        for (int i = 0; i < n; i++) {
            Set<Integer> a = toSet(approx[i]);
            Set<Integer> e = toSet(exact[i]);
            int intersection = 0;
            for (int v : approx[i]) {
                if (e.contains(v)) {
                    intersection++;
                }
            }
            totalOverlap += intersection / (double) e.size();
            assertEquals(0, countIfPresent(a, i), "Self must not appear in approximate neighbours at row " + i);
        }
        double meanRecall = totalOverlap / n;
        assertTrue(meanRecall >= 0.95, "Mean recall " + meanRecall + " below 0.95 threshold");
    }

    // ── Test B: self-exclusion + length ──────────────────────────────────────

    @Test
    void knnExcludesSelfAndHasExpectedLength() {
        Random rng = new Random(3);
        int n = 120;
        int d = 4;
        int k = 15;
        double[][] rows = randomCloud(rng, n, d);

        int[][] out = HnswKnnIndex.knn(rows, k, 1L, false);
        for (int i = 0; i < n; i++) {
            assertEquals(Math.min(k, n - 1), out[i].length, "Unexpected neighbour count at row " + i);
            assertFalse(toSet(out[i]).contains(i), "Self must be excluded at row " + i);
        }
    }

    @Test
    void knnReturnsNMinusOneWhenFewerThanK() {
        double[][] rows = {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}};
        int[][] out = HnswKnnIndex.knn(rows, 10, 5L, true);
        for (int i = 0; i < 3; i++) {
            assertEquals(2, out[i].length, "n-1 neighbours expected when n-1 < k at row " + i);
            assertFalse(toSet(out[i]).contains(i));
        }
    }

    // ── Test C: determinism ──────────────────────────────────────────────────

    @Test
    void reproducibleBuildIsByteIdenticalAcrossTwoConsecutiveRuns() {
        Random rng = new Random(11);
        int n = 100;
        int d = 3;
        int k = 8;
        double[][] rows = new double[n][d];
        fillBlob(rows, rng, 0, 50, d, 0.0, 0.4);
        fillBlob(rows, rng, 50, 100, d, 15.0, 0.4);

        int[][] a = HnswKnnIndex.knn(rows, k, 42L, true);
        int[][] b = HnswKnnIndex.knn(rows, k, 42L, true);

        for (int i = 0; i < n; i++) {
            assertArrayEquals(a[i], b[i], "Reproducible build must return identical neighbours at row " + i);
        }
    }

    // ── Test D: degenerate inputs ─────────────────────────────────────────────

    @Test
    void knnHandlesEmptyInputWithoutThrowing() {
        double[][] rows = new double[0][0];
        int[][] out = assertDoesNotThrow(() -> HnswKnnIndex.knn(rows, 5, 1L, true));
        assertEquals(0, out.length);
    }

    @Test
    void knnHandlesSingleRowWithoutThrowing() {
        double[][] rows = {{1.0, 2.0, 3.0}};
        int[][] out = assertDoesNotThrow(() -> HnswKnnIndex.knn(rows, 5, 1L, true));
        assertEquals(1, out.length);
        assertEquals(0, out[0].length, "A single row has no neighbours");
    }

    // ── Test E: setEf escalation raises recall, no rebuild ───────────────────

    @Test
    void setEfEscalationRaisesOrMaintainsRecallWithoutRebuilding() {
        Random rng = new Random(23);
        int perBlob = 80;
        int blobs = 4;
        int n = perBlob * blobs;
        int d = 6;
        int k = 12;
        double[][] rows = new double[n][d];
        for (int b = 0; b < blobs; b++) {
            fillBlob(rows, rng, b * perBlob, (b + 1) * perBlob, d, b * 10.0, 0.8);
        }
        int[][] exact = LeidenModel.featureKnn(rows, k);

        HnswKnnIndex live = HnswKnnIndex.build(rows, 5L, false);
        live.setEf(1); // deliberately tiny -- low recall
        double lowRecall = meanRecall(live, rows, k, exact);

        live.setEf(200); // escalate query-time only, same built index
        double highRecall = meanRecall(live, rows, k, exact);

        assertTrue(
                highRecall >= lowRecall,
                "Escalated ef recall (" + highRecall + ") should be >= low-ef recall (" + lowRecall + ")");
    }

    // ── Test F: deterministic (distance, id) query ordering ─────────────────

    @Test
    void queryResultsAreOrderedAscendingByDistanceThenId() {
        // Guards the primitive-array insertion sort in HnswKnnIndex.nearestIds: results must come
        // back non-decreasing by true Euclidean distance from the query row, regardless of
        // whatever (possibly nondeterministic) order the underlying index visited candidates in.
        Random rng = new Random(19);
        int n = 200;
        int d = 6;
        int k = 20;
        double[][] rows = randomCloud(rng, n, d);

        int[][] out = HnswKnnIndex.knn(rows, k, 3L, true);
        for (int i = 0; i < n; i++) {
            double prevDist = Double.NEGATIVE_INFINITY;
            int prevId = -1;
            for (int id : out[i]) {
                double dist = euclidean(rows[i], rows[id]);
                assertTrue(
                        dist > prevDist || (dist == prevDist && id > prevId),
                        "Neighbours for row " + i + " must be strictly ascending by (distance, id)");
                prevDist = dist;
                prevId = id;
            }
        }
    }

    @Test
    void queryRowBreaksExactDistanceTiesByAscendingId() {
        // Two points (id 1 and id 2) are exactly equidistant from the query vector; the
        // deterministic tie-break must always place the lower id first, independent of
        // whatever order the underlying index returns tied candidates in.
        double[][] rows = {
            {0.0}, // id 0 -- distance 0 from the query vector
            {-2.0}, // id 1 -- distance 2 (tied with id 2)
            {2.0}, // id 2 -- distance 2 (tied with id 1)
            {3.0}, // id 3 -- distance 3
        };
        HnswKnnIndex index = HnswKnnIndex.build(rows, 1L, true);
        int[] out = index.queryRow(new double[] {0.0}, 4);
        assertArrayEquals(new int[] {0, 1, 2, 3}, out, "distance ties must break by ascending id");
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    private static double euclidean(double[] a, double[] b) {
        double sumSq = 0;
        for (int j = 0; j < a.length; j++) {
            double diff = a[j] - b[j];
            sumSq += diff * diff;
        }
        return Math.sqrt(sumSq);
    }

    private static double meanRecall(HnswKnnIndex live, double[][] rows, int k, int[][] exact) {
        int n = rows.length;
        double total = 0;
        for (int i = 0; i < n; i++) {
            int[] approxRow = live.queryRow(rows[i], k + 1);
            Set<Integer> exactSet = toSet(exact[i]);
            int intersection = 0;
            int counted = 0;
            for (int v : approxRow) {
                if (v == i) {
                    continue; // self may appear since queryRow does not self-exclude
                }
                counted++;
                if (exactSet.contains(v)) {
                    intersection++;
                }
            }
            total += exactSet.isEmpty() ? 1.0 : intersection / (double) exactSet.size();
        }
        return total / n;
    }

    private static int countIfPresent(Set<Integer> set, int value) {
        return set.contains(value) ? 1 : 0;
    }

    private static double[][] randomCloud(Random rng, int n, int d) {
        double[][] rows = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                rows[i][j] = rng.nextDouble() * 100.0;
            }
        }
        return rows;
    }

    private static void fillBlob(double[][] rows, Random rng, int from, int to, int d, double center, double spread) {
        for (int i = from; i < to; i++) {
            for (int j = 0; j < d; j++) {
                rows[i][j] = center + rng.nextGaussian() * spread;
            }
        }
    }

    private static Set<Integer> toSet(int[] a) {
        Set<Integer> s = new HashSet<>();
        for (int v : a) {
            s.add(v);
        }
        return s;
    }
}
