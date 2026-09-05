package qupath.ext.spclassify.model;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;
import qupath.ext.spclassify.model.ScatterMath.PcaReduction;

/**
 * Tests the pure-Java numerical core of the scatter plot: column standardisation,
 * deterministic subsampling, and point-in-polygon hit-testing. The PCA/UMAP
 * embeddings are not exercised here — they load native math libraries (OpenBLAS /
 * ARPACK) that require {@code java.base/java.lang} to be opened, so they are
 * covered by manual QuPath QA rather than unit tests.
 */
class ScatterMathTest {

    private static final double EPS = 1e-9;

    // ── standardizeColumns ───────────────────────────────────────────────────

    @Test
    void standardizeColumnsGivesZeroMeanUnitSdPerColumn() {
        double[][] data = {{1, 10}, {2, 20}, {3, 30}};
        double[] mean = new double[2];
        double[] sd = new double[2];
        double[][] z = ScatterMath.standardizeColumns(data, mean, sd);

        // Captured per-column mean/sd (population sd, /n).
        assertEquals(2.0, mean[0], EPS);
        assertEquals(20.0, mean[1], EPS);
        assertEquals(Math.sqrt(2.0 / 3.0), sd[0], EPS);
        assertEquals(Math.sqrt(200.0 / 3.0), sd[1], EPS);

        // Each output column has ~zero mean and ~unit sd.
        for (int j = 0; j < 2; j++) {
            double m = 0;
            for (double[] row : z) {
                m += row[j];
            }
            m /= z.length;
            assertEquals(0.0, m, EPS, "column " + j + " mean");
            double var = 0;
            for (double[] row : z) {
                var += (row[j] - m) * (row[j] - m);
            }
            assertEquals(1.0, Math.sqrt(var / z.length), EPS, "column " + j + " sd");
        }
    }

    @Test
    void standardizeColumnsZeroesConstantColumns() {
        double[][] data = {{5, 1}, {5, 2}, {5, 3}};
        double[] mean = new double[2];
        double[] sd = new double[2];
        double[][] z = ScatterMath.standardizeColumns(data, mean, sd);

        assertEquals(5.0, mean[0], EPS);
        assertEquals(0.0, sd[0], EPS);
        // sd < 1e-9 → column maps to 0, never NaN/Inf.
        for (double[] row : z) {
            assertEquals(0.0, row[0], EPS);
        }
    }

    @Test
    void standardizeColumnsHandlesEmptyInput() {
        double[][] z = ScatterMath.standardizeColumns(new double[0][]);
        assertEquals(0, z.length);
    }

    @Test
    void standardizeColumnsImputesNaNToZeroAndZScoresNonNaNOverNonNaNStats() {
        // Column 0: one NaN (missing measurement) among finite values; column 1 fully finite.
        double[][] data = {{1, 10}, {Double.NaN, 20}, {3, 30}, {5, 40}};
        double[] mean = new double[2];
        double[] sd = new double[2];
        double[][] z = ScatterMath.standardizeColumns(data, mean, sd);

        // Column 0's mean/sd must be computed over the 3 non-NaN entries {1, 3, 5}, not 4 rows.
        double expectedMean0 = (1.0 + 3.0 + 5.0) / 3.0;
        double var0 = (Math.pow(1 - expectedMean0, 2) + Math.pow(3 - expectedMean0, 2) + Math.pow(5 - expectedMean0, 2))
                / 3.0;
        double expectedSd0 = Math.sqrt(var0);
        assertEquals(expectedMean0, mean[0], EPS);
        assertEquals(expectedSd0, sd[0], EPS);

        // The NaN cell itself is imputed to 0 (the column mean, in z-score terms).
        assertEquals(0.0, z[1][0], EPS, "NaN cell must impute to 0");

        // Every non-NaN cell in column 0 is z-scored using the non-NaN mean/sd.
        assertEquals((1.0 - expectedMean0) / expectedSd0, z[0][0], EPS);
        assertEquals((3.0 - expectedMean0) / expectedSd0, z[2][0], EPS);
        assertEquals((5.0 - expectedMean0) / expectedSd0, z[3][0], EPS);

        // No NaN or Infinity anywhere in the output, for any column.
        for (double[] row : z) {
            for (double v : row) {
                assertFalse(Double.isNaN(v), "output must never contain NaN");
                assertFalse(Double.isInfinite(v), "output must never contain Infinity");
            }
        }
    }

    @Test
    void standardizeColumnsAllNaNColumnBecomesAllZeros() {
        double[][] data = {{Double.NaN, 1}, {Double.NaN, 2}, {Double.NaN, 3}};
        double[] mean = new double[2];
        double[] sd = new double[2];
        double[][] z = ScatterMath.standardizeColumns(data, mean, sd);

        assertEquals(0.0, mean[0], EPS, "all-NaN column mean must default to 0");
        assertEquals(0.0, sd[0], EPS, "all-NaN column sd must default to 0");
        for (double[] row : z) {
            assertEquals(0.0, row[0], EPS, "all-NaN column must map every row to 0");
            assertFalse(Double.isNaN(row[0]));
        }
    }

    @Test
    void standardizeColumnsFullyFiniteInputMatchesPreChangeSumOverAllRows() {
        // Regression guard: for fully-finite input, the NaN-robust rewrite must produce
        // byte-identical output to the original "sum over every row" algorithm (the
        // non-NaN count equals n, so mean/sd/output must match exactly).
        Random rng = new Random(123);
        int n = 50;
        int p = 6;
        double[][] data = new double[n][p];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                data[i][j] = rng.nextGaussian() * 10;
            }
        }

        double[] actualMean = new double[p];
        double[] actualSd = new double[p];
        double[][] actual = ScatterMath.standardizeColumns(data, actualMean, actualSd);

        double[] expectedMean = new double[p];
        double[] expectedSd = new double[p];
        double[][] expected = new double[n][p];
        for (int j = 0; j < p; j++) {
            double sum = 0;
            for (double[] row : data) {
                sum += row[j];
            }
            double mean = sum / n;
            double var = 0;
            for (double[] row : data) {
                double d = row[j] - mean;
                var += d * d;
            }
            double sd = Math.sqrt(var / n);
            expectedMean[j] = mean;
            expectedSd[j] = sd;
            double inv = sd < 1e-9 ? 0.0 : 1.0 / sd;
            for (int i = 0; i < n; i++) {
                expected[i][j] = (data[i][j] - mean) * inv;
            }
        }

        assertArrayEquals(expectedMean, actualMean, 0.0);
        assertArrayEquals(expectedSd, actualSd, 0.0);
        for (int i = 0; i < n; i++) {
            assertArrayEquals(expected[i], actual[i], 0.0, "row " + i + " must be byte-identical to pre-change output");
        }
    }

    @Test
    void standardizeColumnsSingleColumnConvenienceOverloadMatches() {
        double[][] data = {{1}, {2}, {3}, {4}};
        double[][] a = ScatterMath.standardizeColumns(data);
        double[][] b = ScatterMath.standardizeColumns(data, new double[1], new double[1]);
        for (int i = 0; i < data.length; i++) {
            assertArrayEquals(b[i], a[i], EPS);
        }
    }

    // ── identity / randomSubsample ───────────────────────────────────────────

    @Test
    void identityIsAscendingRange() {
        assertArrayEquals(new int[] {0, 1, 2, 3}, ScatterMath.identity(4));
        assertArrayEquals(new int[] {}, ScatterMath.identity(0));
    }

    @Test
    void randomSubsampleReturnsDistinctSortedInRangeIndices() {
        int n = 200;
        int count = 30;
        int[] pick = ScatterMath.randomSubsample(n, count);

        assertEquals(count, pick.length);
        assertTrue(Arrays.stream(pick).allMatch(i -> i >= 0 && i < n), "in range");
        assertEquals(count, Arrays.stream(pick).distinct().count(), "distinct");

        int[] sorted = pick.clone();
        Arrays.sort(sorted);
        assertArrayEquals(sorted, pick, "returned sorted ascending");
    }

    @Test
    void randomSubsampleIsDeterministic() {
        // Fixed internal seed → same data plots the same points across runs.
        assertArrayEquals(ScatterMath.randomSubsample(500, 50), ScatterMath.randomSubsample(500, 50));
    }

    // ── pointInPolygon ───────────────────────────────────────────────────────

    @Test
    void pointInPolygonInsideAndOutsideUnitSquare() {
        List<double[]> square = List.of(
                new double[] {0, 0}, new double[] {10, 0},
                new double[] {10, 10}, new double[] {0, 10});
        assertTrue(ScatterMath.pointInPolygon(5, 5, square), "centre is inside");
        assertFalse(ScatterMath.pointInPolygon(15, 5, square), "right is outside");
        assertFalse(ScatterMath.pointInPolygon(-1, 5, square), "left is outside");
        assertFalse(ScatterMath.pointInPolygon(5, 20, square), "above is outside");
    }

    @Test
    void pointInPolygonConcaveShape() {
        // Arrow/notch polygon: a point in the notch is outside.
        List<double[]> notch = List.of(
                new double[] {0, 0}, new double[] {10, 0}, new double[] {10, 10}, new double[] {5, 5}, new double[] {
                    0, 10
                });
        assertTrue(ScatterMath.pointInPolygon(2, 2, notch), "lower-left body inside");
        assertFalse(ScatterMath.pointInPolygon(5, 9, notch), "inside the notch is outside");
    }

    // ── pcaReduce: conditional skip / disable ────────────────────────────────

    @Test
    void pcaReduceSkippedByThresholdWhenColumnsAtOrBelowThreshold() {
        Random rng = new Random(1);
        double[][] std = randomMatrix(rng, 200, 40);

        PcaReduction result = ScatterMath.pcaReduce(std, 50, 50, 100_000, 42L);

        assertFalse(result.applied(), "40 columns <= threshold 50 must skip PCA");
        assertEquals(1.0, result.cumulativeVariance(), 0.0);
        assertSame2D(std, result.reduced());
        // Identity projector: applying it to any matrix returns that matrix unchanged.
        assertSame2D(std, result.projector().apply(std));
    }

    @Test
    void pcaReduceSkippedWhenDisabledEvenWithManyColumns() {
        Random rng = new Random(2);
        double[][] std = randomMatrix(rng, 200, 200);

        PcaReduction result = ScatterMath.pcaReduce(std, 0, 50, 100_000, 42L);

        assertFalse(result.applied(), "maxComponents<=0 must disable PCA regardless of column count");
        assertEquals(200, result.nComponents(), "nComponents reports the (unreduced) column count when not applied");
        assertSame2D(std, result.reduced());
    }

    @Test
    void pcaReduceAppliedWhenColumnsExceedThreshold() {
        Random rng = new Random(3);
        int n = 300;
        int p = 120;
        double[][] std = randomMatrix(rng, n, p);

        PcaReduction result = ScatterMath.pcaReduce(std, 50, 50, 100_000, 42L);

        assertTrue(result.applied(), "120 columns > threshold 50 must apply PCA");
        assertEquals(50, result.nComponents());
        assertEquals(n, result.reduced().length, "every input row must come out reduced");
        for (double[] row : result.reduced()) {
            assertEquals(50, row.length, "each reduced row must have exactly nComponents columns");
        }
    }

    @Test
    void pcaReduceComponentCountCappedByColumnsMinusOneAndRowsMinusOne() {
        Random rng = new Random(4);
        // 10 rows, 80 columns, requesting 50 components -> capped at min(50, 79, 9) = 9.
        double[][] std = randomMatrix(rng, 10, 80);

        PcaReduction result = ScatterMath.pcaReduce(std, 50, 50, 100_000, 42L);

        assertTrue(result.applied());
        assertEquals(9, result.nComponents(), "nComponents must be capped at rows-1 when rows-1 < maxComponents");
    }

    // ── pcaReduce: determinism ────────────────────────────────────────────────

    @Test
    void pcaReduceIdenticalInputsAndSeedYieldIdenticalReducedOutput() {
        Random rng = new Random(5);
        double[][] std = randomMatrix(rng, 400, 100);

        PcaReduction a = ScatterMath.pcaReduce(std, 50, 50, 100_000, 42L);
        PcaReduction b = ScatterMath.pcaReduce(std, 50, 50, 100_000, 42L);

        assertEquals(a.nComponents(), b.nComponents());
        assertEquals(a.cumulativeVariance(), b.cumulativeVariance(), 0.0);
        for (int i = 0; i < a.reduced().length; i++) {
            assertArrayEquals(
                    a.reduced()[i], b.reduced()[i], 1e-9, "Identical input+seed must reduce to identical output");
        }
    }

    // ── pcaReduce: subsample-fit path (fitSampleCap) ─────────────────────────

    @Test
    void pcaReduceSubsampleFitPathProjectsEveryInputRowAndIsDeterministic() {
        Random rng = new Random(6);
        int n = 5_000;
        int p = 80;
        double[][] std = randomMatrix(rng, n, p);
        int fitSampleCap = 500; // n > fitSampleCap -> PCA is FIT on a seeded subsample only.

        PcaReduction a = ScatterMath.pcaReduce(std, 30, 50, fitSampleCap, 42L);
        PcaReduction b = ScatterMath.pcaReduce(std, 30, 50, fitSampleCap, 42L);

        assertTrue(a.applied());
        assertEquals(n, a.reduced().length, "every one of the n input rows must be projected, not just the fit sample");
        assertEquals(30, a.nComponents());
        for (int i = 0; i < n; i++) {
            assertArrayEquals(
                    a.reduced()[i],
                    b.reduced()[i],
                    1e-9,
                    "Subsample-fit path must be deterministic given the same seed, row " + i);
        }
    }

    @Test
    void pcaReduceReusableProjectorAppliesSameBasisToNewRows() {
        Random rng = new Random(7);
        double[][] std = randomMatrix(rng, 300, 90);

        PcaReduction fit = ScatterMath.pcaReduce(std, 20, 50, 100_000, 42L);
        assertTrue(fit.applied());

        // Projecting the ORIGINAL rows again through the returned projector must reproduce
        // the SAME reduced matrix pcaReduce itself returned (the projector is exactly what
        // pcaReduce used internally to build `reduced`).
        double[][] reprojected = fit.projector().apply(std);
        for (int i = 0; i < std.length; i++) {
            assertArrayEquals(
                    fit.reduced()[i], reprojected[i], 1e-9, "Projector must reproduce pcaReduce's own output");
        }

        // A genuinely NEW row (same column count) must also project into the nComponents-wide
        // PC space without error.
        double[][] newRows = randomMatrix(new Random(99), 5, 90);
        double[][] projectedNew = fit.projector().apply(newRows);
        assertEquals(5, projectedNew.length);
        for (double[] row : projectedNew) {
            assertEquals(fit.nComponents(), row.length);
        }
    }

    // ── pcaReduce: cumulative variance ────────────────────────────────────────

    @Test
    void pcaReduceCumulativeVarianceIsInZeroToOneRangeWhenApplied() {
        Random rng = new Random(8);
        double[][] std = randomMatrix(rng, 400, 100);

        PcaReduction result = ScatterMath.pcaReduce(std, 50, 50, 100_000, 42L);

        assertTrue(result.applied());
        assertTrue(
                result.cumulativeVariance() > 0.0 && result.cumulativeVariance() <= 1.0,
                "Cumulative variance must be in (0,1], got " + result.cumulativeVariance());
    }

    @Test
    void pcaReduceCumulativeVarianceIsOneWhenNotApplied() {
        Random rng = new Random(9);
        double[][] std = randomMatrix(rng, 200, 20);

        PcaReduction result = ScatterMath.pcaReduce(std, 50, 50, 100_000, 42L);

        assertFalse(result.applied());
        assertEquals(1.0, result.cumulativeVariance(), 0.0);
    }

    // ── pcaReduce: defaults overload ──────────────────────────────────────────

    @Test
    void pcaReduceDefaultsOverloadMatchesExplicitDefaultConstants() {
        Random rng = new Random(10);
        double[][] std = randomMatrix(rng, 500, 120);

        PcaReduction viaDefaults = ScatterMath.pcaReduce(std);
        PcaReduction viaExplicit = ScatterMath.pcaReduce(
                std,
                ScatterMath.PCA_DEFAULT_MAX_COMPONENTS,
                ScatterMath.PCA_DEFAULT_THRESHOLD,
                ScatterMath.PCA_DEFAULT_FIT_SAMPLE_CAP,
                ScatterMath.PCA_DEFAULT_SEED);

        assertEquals(viaExplicit.applied(), viaDefaults.applied());
        assertEquals(viaExplicit.nComponents(), viaDefaults.nComponents());
        assertEquals(viaExplicit.cumulativeVariance(), viaDefaults.cumulativeVariance(), 0.0);
    }

    // ── pcaReduce helpers ──────────────────────────────────────────────────────

    private static double[][] randomMatrix(Random rng, int n, int p) {
        double[][] m = new double[n][p];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                m[i][j] = rng.nextGaussian();
            }
        }
        return m;
    }

    private static void assertSame2D(double[][] expected, double[][] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], actual[i], 0.0);
        }
    }
}
