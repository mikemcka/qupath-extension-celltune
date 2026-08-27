package qupath.ext.celltune.model;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link BatchNormalizerModel}: the pure UniFORM histogram / reference /
 * cross-correlation-shift / scale math. Synthetic arrays only — no QuPath/JavaFX —
 * mirroring {@code NeighborhoodModelTest} and {@code ScatterMathTest}.
 */
class BatchNormalizerModelTest {

    private static final double EPS = 1e-9;

    // ── logMinMax ────────────────────────────────────────────────────────────

    @Test
    void logMinMaxIgnoresSubThresholdAndNonFinite() {
        double[] raw = {1.0, Math.exp(2.0), 0.5, Double.NaN, Double.POSITIVE_INFINITY};
        double[] mm = BatchNormalizerModel.logMinMax(raw);
        assertEquals(0.0, mm[0], EPS, "log(1) = 0 is the min");
        assertEquals(2.0, mm[1], EPS, "log(e^2) = 2 is the max");
    }

    @Test
    void logMinMaxAllDroppedIsNaN() {
        double[] mm = BatchNormalizerModel.logMinMax(new double[] {0.2, 0.9, Double.NaN});
        assertTrue(Double.isNaN(mm[0]) && Double.isNaN(mm[1]));
    }

    // ── logHistogram ─────────────────────────────────────────────────────────

    @Test
    void logHistogramBinsByLogValueAndDropsSubThreshold() {
        // log values 0.5, 1.5, 2.5 over [0,3] with 3 bins (width 1) → one per bin.
        double[] raw = {Math.exp(0.5), Math.exp(1.5), Math.exp(2.5), 0.5, Double.NaN};
        long[] h = BatchNormalizerModel.logHistogram(raw, 0.0, 3.0, 3);
        assertArrayEquals(new long[] {1, 1, 1}, h);
    }

    @Test
    void logHistogramClampsEdgesAndHandlesDegenerateRange() {
        // Values below/above the range clamp into the end bins.
        double[] raw = {Math.exp(-5.0 + 0.0), Math.exp(100.0)}; // e^-5 < 1 dropped; e^100 clamps to top
        long[] h = BatchNormalizerModel.logHistogram(raw, 0.0, 10.0, 5);
        assertEquals(1, h[4], "huge value clamps into the top bin");
        // Degenerate range: everything qualifying lands in bin 0.
        long[] d = BatchNormalizerModel.logHistogram(new double[] {2.0, 3.0, 4.0}, 1.0, 1.0, 4);
        assertArrayEquals(new long[] {3, 0, 0, 0}, d);
    }

    // ── crossCorrelationShift (known-shift recovery) ─────────────────────────

    @Test
    void crossCorrelationRecoversKnownShift() {
        long[] ref = bump(256, 100, 12);
        for (int k : new int[] {0, 5, -7, 20, -33}) {
            long[] shifted = shiftRight(ref, k);
            assertEquals(
                    k,
                    BatchNormalizerModel.crossCorrelationShift(shifted, ref),
                    "must recover a +" + k + " bin shift (positive = brighter than reference)");
        }
    }

    @Test
    void crossCorrelationZeroForIdenticalHistograms() {
        long[] ref = bump(128, 40, 8);
        assertEquals(0, BatchNormalizerModel.crossCorrelationShift(ref, ref));
    }

    // ── autoReferenceIndex ───────────────────────────────────────────────────

    @Test
    void autoReferencePicksTheCentralDistributionNotTheOutlier() {
        // Three identical central histograms + one far outlier: reference must be one of
        // the central three (tie → lowest index 0), never the outlier.
        long[][] hists = {bump(200, 30, 6), bump(200, 30, 6), bump(200, 30, 6), bump(200, 150, 6)};
        assertEquals(0, BatchNormalizerModel.autoReferenceIndex(hists));
    }

    @Test
    void autoReferenceEmptyIsMinusOne() {
        assertEquals(-1, BatchNormalizerModel.autoReferenceIndex(new long[0][]));
    }

    // ── shiftToScale ─────────────────────────────────────────────────────────

    @Test
    void shiftToScaleRoundTripsOffsetAndNoOps() {
        // increment = (logMax-logMin)/(nBins-1) = 1023/1023 = 1 → scale = exp(-shift).
        assertEquals(Math.exp(-10), BatchNormalizerModel.shiftToScale(10, 0.0, 1023.0, 1024), 1e-12);
        assertEquals(1.0, BatchNormalizerModel.shiftToScale(0, 0.0, 1023.0, 1024), EPS, "zero shift = no-op");
        assertTrue(BatchNormalizerModel.shiftToScale(5, 0.0, 1023.0, 1024) < 1.0, "positive shift scales down");
        assertEquals(1.0, BatchNormalizerModel.shiftToScale(7, 5.0, 5.0, 1024), EPS, "degenerate range = no-op");
    }

    // ── end-to-end: a known multiplicative brightness offset is recovered ────

    @Test
    void recoversMultiplicativeBrightnessOffsetEndToEnd() {
        // ref ~ exp(N(3, 0.5)); "bright" = ref × 3 (a pure ×3 gain, i.e. +log(3) in log-space).
        // The learned scale should ≈ 1/3 to undo it (accurate to ~1 bin at 1024 bins).
        Random rng = new Random(42);
        int n = 20000;
        double[] ref = new double[n];
        double[] bright = new double[n];
        for (int i = 0; i < n; i++) {
            double v = Math.exp(3.0 + 0.5 * rng.nextGaussian());
            ref[i] = v;
            bright[i] = v * 3.0;
        }
        double[] mmR = BatchNormalizerModel.logMinMax(ref);
        double[] mmB = BatchNormalizerModel.logMinMax(bright);
        double logMin = Math.min(mmR[0], mmB[0]);
        double logMax = Math.max(mmR[1], mmB[1]);
        int nBins = BatchNormalizerModel.DEFAULT_BINS;

        long[] hRef = BatchNormalizerModel.logHistogram(ref, logMin, logMax, nBins);
        long[] hBright = BatchNormalizerModel.logHistogram(bright, logMin, logMax, nBins);
        int shift = BatchNormalizerModel.crossCorrelationShift(hBright, hRef);
        assertTrue(shift > 0, "the brighter sample must have a positive shift");
        double scale = BatchNormalizerModel.shiftToScale(shift, logMin, logMax, nBins);
        assertEquals(1.0 / 3.0, scale, 0.02, "learned scale should undo the ×3 gain");
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    /** A symmetric triangular bump of half-width {@code w} centred at {@code c}. */
    private static long[] bump(int n, int c, int w) {
        long[] h = new long[n];
        for (int i = 0; i < n; i++) {
            int d = Math.abs(i - c);
            if (d <= w) {
                h[i] = w - d + 1;
            }
        }
        return h;
    }

    /** {@code out[i] = a[i-k]} (zero-filled), i.e. content moved right by {@code k} bins. */
    private static long[] shiftRight(long[] a, int k) {
        int n = a.length;
        long[] out = new long[n];
        for (int i = 0; i < n; i++) {
            int j = i - k;
            if (j >= 0 && j < n) {
                out[i] = a[j];
            }
        }
        return out;
    }
}
