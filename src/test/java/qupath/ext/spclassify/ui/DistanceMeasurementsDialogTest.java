package qupath.ext.spclassify.ui;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link DistanceMeasurementsDialog#sameClassNearestNeighbourDistances}.
 * <p>Locks in the contract that protects the {@code STRtree} fast-path described
 * in {@code AGENTS.md} (Known Pitfalls): self-exclusion, NaN handling, and that
 * the result agrees with a brute-force reference on small inputs.
 */
class DistanceMeasurementsDialogTest {

    private static final double EPS = 1e-9;

    @Test
    void emptyInputReturnsEmptyArray() {
        double[] out = DistanceMeasurementsDialog.sameClassNearestNeighbourDistances(new double[0], new double[0]);
        assertEquals(0, out.length);
    }

    @Test
    void singlePointReturnsNaN() {
        double[] out =
                DistanceMeasurementsDialog.sameClassNearestNeighbourDistances(new double[] {1.0}, new double[] {2.0});
        assertEquals(1, out.length);
        assertTrue(Double.isNaN(out[0]), "Single point has no neighbour");
    }

    @Test
    void twoPointsGetMutualDistance() {
        double[] xs = {0.0, 3.0};
        double[] ys = {0.0, 4.0};
        double[] out = DistanceMeasurementsDialog.sameClassNearestNeighbourDistances(xs, ys);
        assertEquals(5.0, out[0], EPS);
        assertEquals(5.0, out[1], EPS);
    }

    @Test
    void selfIsExcludedEvenWhenDuplicated() {
        // Two cells at the exact same location must each report distance 0 to the
        // OTHER, never NaN (which would mean self matched itself and was filtered).
        double[] xs = {7.0, 7.0, 100.0};
        double[] ys = {7.0, 7.0, 100.0};
        double[] out = DistanceMeasurementsDialog.sameClassNearestNeighbourDistances(xs, ys);
        assertEquals(0.0, out[0], EPS, "Duplicate at same location must match its twin, not itself");
        assertEquals(0.0, out[1], EPS, "Duplicate at same location must match its twin, not itself");
        // The lone outlier's nearest neighbour is one of the duplicates at distance sqrt(2)*93.
        double expected = Math.hypot(93.0, 93.0);
        assertEquals(expected, out[2], EPS);
    }

    @Test
    void nanCoordinatesProduceNanAndAreIgnoredByOthers() {
        double[] xs = {0.0, Double.NaN, 10.0};
        double[] ys = {0.0, Double.NaN, 0.0};
        double[] out = DistanceMeasurementsDialog.sameClassNearestNeighbourDistances(xs, ys);
        assertTrue(Double.isNaN(out[1]), "NaN-coord point gets NaN distance");
        assertEquals(10.0, out[0], EPS, "Valid point ignores the NaN neighbour");
        assertEquals(10.0, out[2], EPS, "Valid point ignores the NaN neighbour");
    }

    @Test
    void singleValidPointWithRestNanReturnsAllNaN() {
        // Only one usable point in the tree → no neighbour exists for it either.
        double[] xs = {5.0, Double.NaN, Double.NaN};
        double[] ys = {5.0, Double.NaN, Double.NaN};
        double[] out = DistanceMeasurementsDialog.sameClassNearestNeighbourDistances(xs, ys);
        for (int i = 0; i < out.length; i++) {
            assertTrue(Double.isNaN(out[i]), "Index " + i + " should be NaN");
        }
    }

    @Test
    void collinearPointsPickCorrectNeighbours() {
        // x-coords: 0, 1, 5, 12 → NN distances should be 1, 1, 4, 7
        double[] xs = {0.0, 1.0, 5.0, 12.0};
        double[] ys = {0.0, 0.0, 0.0, 0.0};
        double[] out = DistanceMeasurementsDialog.sameClassNearestNeighbourDistances(xs, ys);
        assertEquals(1.0, out[0], EPS);
        assertEquals(1.0, out[1], EPS);
        assertEquals(4.0, out[2], EPS);
        assertEquals(7.0, out[3], EPS);
    }

    @Test
    void matchesBruteForceOnRandomCloud() {
        // 500 random points: STRtree result must equal a naive O(n^2) scan
        // (this is the regression guard against the prior O(n^2) bug being
        // "fixed" by silently dropping self-exclusion or similar shortcuts).
        Random rng = new Random(42);
        int n = 500;
        double[] xs = new double[n];
        double[] ys = new double[n];
        for (int i = 0; i < n; i++) {
            xs[i] = rng.nextDouble() * 1000.0;
            ys[i] = rng.nextDouble() * 1000.0;
        }
        double[] actual = DistanceMeasurementsDialog.sameClassNearestNeighbourDistances(xs, ys);
        double[] expected = bruteForceNN(xs, ys);
        for (int i = 0; i < n; i++) {
            assertEquals(expected[i], actual[i], EPS, "Mismatch at index " + i);
        }
    }

    @Test
    void matchesBruteForceOnLargeCloudExercisingParallelism() {
        // Large enough that the parallel STRtree query loop splits across worker
        // threads. Result must still match the brute-force reference exactly,
        // guarding the parallelisation against any threading/ordering bug.
        Random rng = new Random(7);
        int n = 10_000;
        double[] xs = new double[n];
        double[] ys = new double[n];
        for (int i = 0; i < n; i++) {
            xs[i] = rng.nextDouble() * 5000.0;
            ys[i] = rng.nextDouble() * 5000.0;
        }
        double[] actual = DistanceMeasurementsDialog.sameClassNearestNeighbourDistances(xs, ys);
        double[] expected = bruteForceNN(xs, ys);
        for (int i = 0; i < n; i++) {
            assertEquals(expected[i], actual[i], EPS, "Mismatch at index " + i);
        }
    }

    @Test
    void parallelResultIsDeterministic() {
        // Computing the same input twice must yield identical results — a guard
        // against data races in the parallel query loop (shared tree/index map).
        Random rng = new Random(123);
        int n = 4_000;
        double[] xs = new double[n];
        double[] ys = new double[n];
        for (int i = 0; i < n; i++) {
            xs[i] = rng.nextDouble() * 2000.0;
            ys[i] = rng.nextDouble() * 2000.0;
        }
        double[] first = DistanceMeasurementsDialog.sameClassNearestNeighbourDistances(xs, ys);
        double[] second = DistanceMeasurementsDialog.sameClassNearestNeighbourDistances(xs, ys);
        assertArrayEquals(first, second, "Parallel computation must be reproducible");
    }

    private static double[] bruteForceNN(double[] xs, double[] ys) {
        int n = xs.length;
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            double best = Double.POSITIVE_INFINITY;
            for (int j = 0; j < n; j++) {
                if (j == i) continue;
                double dx = xs[i] - xs[j];
                double dy = ys[i] - ys[j];
                double d2 = dx * dx + dy * dy;
                if (d2 < best) best = d2;
            }
            out[i] = best == Double.POSITIVE_INFINITY ? Double.NaN : Math.sqrt(best);
        }
        return out;
    }
}
