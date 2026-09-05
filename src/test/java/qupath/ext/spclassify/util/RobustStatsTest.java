package qupath.ext.spclassify.util;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

/**
 * Verifies the two robust-z variants preserve the behaviour they were extracted
 * from. The key invariant is that they diverge on a degenerate MAD: the strict
 * variant returns zeros, the fallback variant uses mean/std.
 */
class RobustStatsTest {

    private static final double EPS = 1e-9;

    @Test
    void medianOddAndEven() {
        assertEquals(2.0, RobustStats.median(new double[] {3, 1, 2}), EPS);
        assertEquals(2.5, RobustStats.median(new double[] {1, 2, 3, 4}), EPS);
        assertEquals(0.0, RobustStats.median(new double[] {}), EPS);
    }

    @Test
    void medianIgnoreNaNSkipsNaNAndIsNaNWhenAllMissing() {
        assertEquals(2.0, RobustStats.medianIgnoreNaN(new double[] {Double.NaN, 1, 2, 3, Double.NaN}), EPS);
        assertTrue(Double.isNaN(RobustStats.medianIgnoreNaN(new double[] {Double.NaN, Double.NaN})));
        assertTrue(Double.isNaN(RobustStats.medianIgnoreNaN(new double[] {})));
    }

    @Test
    void robustZStrictScoresOutlier() {
        // Spread values so the MAD is non-degenerate (median 3, MAD 1).
        double[] z = RobustStats.robustZStrict(new double[] {1, 2, 3, 4, 100});
        assertTrue(z[4] > 5.0, "outlier should score high, was " + z[4]);
        assertEquals(0.0, z[2], EPS, "value at the median should score ~0");
    }

    @Test
    void robustZStrictReturnsZerosOnDegenerateMad() {
        // All identical → MAD == 0 → strict variant returns all zeros.
        double[] z = RobustStats.robustZStrict(new double[] {5, 5, 5, 5});
        for (double v : z) assertEquals(0.0, v, EPS);
    }

    @Test
    void robustZStrictEmptyReturnsEmpty() {
        assertEquals(0, RobustStats.robustZStrict(new double[] {}).length);
    }

    @Test
    void robustZWithFallbackUsesMeanStdOnDegenerateMad() {
        // Mostly-flat baseline with one genuine outlier: MAD is 0 (median abs dev
        // of {0,0,0,0,95} is 0), so the strict variant would collapse to zeros,
        // but the fallback variant must still surface the outlier via mean/std.
        double[] values = {5, 5, 5, 5, 100};
        double[] strict = RobustStats.robustZStrict(values);
        double[] fallback = RobustStats.robustZWithFallback(values);

        assertEquals(0.0, strict[4], EPS, "strict collapses to zero on degenerate MAD");
        assertTrue(fallback[4] > 1.0, "fallback must surface the outlier, was " + fallback[4]);
    }

    @Test
    void robustZWithFallbackPreservesNaN() {
        double[] z = RobustStats.robustZWithFallback(new double[] {1, 2, Double.NaN, 3, 4});
        assertTrue(Double.isNaN(z[2]), "NaN input must map to NaN output");
        assertFalse(Double.isNaN(z[0]));
    }

    @Test
    void robustZWithFallbackAllNaNGivesAllNaN() {
        double[] z = RobustStats.robustZWithFallback(new double[] {Double.NaN, Double.NaN});
        for (double v : z) assertTrue(Double.isNaN(v));
    }
}
