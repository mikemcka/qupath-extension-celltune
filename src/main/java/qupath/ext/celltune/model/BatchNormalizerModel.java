package qupath.ext.celltune.model;

/**
 * Pure numerical core of the <b>UniFORM</b> batch-normalization method (Wang et al.,
 * bioRxiv 2024) — per-marker distribution alignment for multiplexed imaging cohorts.
 *
 * <p>For one marker, each sample (image or batch group) has its cell-mean intensities
 * summarised as a log-intensity histogram over a shared range. One sample is chosen as
 * the reference; every other sample's histogram is aligned to it by the integer bin
 * {@link #crossCorrelationShift(long[], long[]) shift} that maximises cross-correlation.
 * That shift is a translation in log-space, so it maps back to a single multiplicative
 * {@link #shiftToScale(int, double, double, int) scale} applied to every cell:
 * {@code corrected = raw · scale}. Translation-only ⇒ the distribution's shape is
 * preserved, only its location moves — the conservative property that makes UniFORM
 * unlikely to erase real biology.
 *
 * <p>Every method is static and a pure function of primitive arrays (no JavaFX, no
 * QuPath types), so the histogram/reference/shift math is unit-testable in isolation —
 * mirroring {@link ScatterMath} and {@link NeighborhoodModel}. The streaming, memory-safe
 * cohort driver that feeds these functions lives in {@code BatchNormalizerCohort}.
 *
 * <p>Faithful to the reference package's {@code log_transform_intensities} /
 * {@code compute_correlation_shifts} / {@code calculate_shift_in_log_pixels}: values
 * {@code < 1} and non-finite values are dropped before {@code log}; the shift is an
 * integer bin lag; the applied factor is {@code exp(−shift · increment)}.
 */
public final class BatchNormalizerModel {

    private BatchNormalizerModel() {}

    /** Default number of log-intensity histogram bins (UniFORM default). */
    public static final int DEFAULT_BINS = 1024;

    /** Intensities strictly below this are dropped before the log transform (UniFORM default). */
    public static final double MIN_INTENSITY = 1.0;

    // ── Log range ────────────────────────────────────────────────────────────

    /**
     * Min and max of {@code log(v)} over the finite values {@code v ≥ }{@link #MIN_INTENSITY}
     * in {@code raw}. Returns {@code {NaN, NaN}} when no value qualifies. Callers combine
     * these per-sample ranges into a marker-global range before histogramming, so every
     * sample's histogram shares the same bin edges.
     */
    public static double[] logMinMax(double[] raw) {
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        boolean any = false;
        for (double v : raw) {
            if (v >= MIN_INTENSITY && !Double.isInfinite(v) && !Double.isNaN(v)) {
                double lv = Math.log(v);
                if (lv < min) {
                    min = lv;
                }
                if (lv > max) {
                    max = lv;
                }
                any = true;
            }
        }
        return any ? new double[] {min, max} : new double[] {Double.NaN, Double.NaN};
    }

    // ── Histogram ────────────────────────────────────────────────────────────

    /**
     * Fixed-width log-intensity histogram of {@code raw} over {@code [logMin, logMax]}
     * with {@code nBins} bins. Values {@code < }{@link #MIN_INTENSITY} and non-finite
     * values are dropped (UniFORM's {@code log_transform_intensities}); {@code log(v)}
     * is clamped into {@code [0, nBins-1]} at the edges. Mirrors the binning idiom in
     * {@code ImagePixelStats.otsuThresholdSorted}.
     */
    public static long[] logHistogram(double[] raw, double logMin, double logMax, int nBins) {
        if (nBins < 1) {
            throw new IllegalArgumentException("nBins must be >= 1");
        }
        long[] hist = new long[nBins];
        double range = logMax - logMin;
        if (!(range > 0)) {
            // Degenerate range: everything that qualifies lands in bin 0.
            for (double v : raw) {
                if (v >= MIN_INTENSITY && !Double.isInfinite(v) && !Double.isNaN(v)) {
                    hist[0]++;
                }
            }
            return hist;
        }
        double scale = nBins / range;
        for (double v : raw) {
            if (v < MIN_INTENSITY || Double.isInfinite(v) || Double.isNaN(v)) {
                continue;
            }
            int bin = (int) ((Math.log(v) - logMin) * scale);
            if (bin < 0) {
                bin = 0;
            } else if (bin >= nBins) {
                bin = nBins - 1;
            }
            hist[bin]++;
        }
        return hist;
    }

    // ── Reference selection ──────────────────────────────────────────────────

    /**
     * Index of the sample whose histogram is closest (Euclidean/L2) to the element-wise
     * mean histogram across all samples — UniFORM's automatic reference (the "most
     * central" distribution). Ties resolve to the lowest index. Returns {@code -1} for
     * an empty input. All histograms must share a length.
     */
    public static int autoReferenceIndex(long[][] perSampleHist) {
        int n = perSampleHist.length;
        if (n == 0) {
            return -1;
        }
        int bins = perSampleHist[0].length;
        double[] mean = new double[bins];
        for (long[] h : perSampleHist) {
            for (int b = 0; b < bins; b++) {
                mean[b] += h[b];
            }
        }
        for (int b = 0; b < bins; b++) {
            mean[b] /= n;
        }
        int best = 0;
        double bestDist = Double.POSITIVE_INFINITY;
        for (int s = 0; s < n; s++) {
            long[] h = perSampleHist[s];
            double d = 0;
            for (int b = 0; b < bins; b++) {
                double diff = h[b] - mean[b];
                d += diff * diff;
            }
            if (d < bestDist) {
                bestDist = d;
                best = s;
            }
        }
        return best;
    }

    // ── Cross-correlation shift ──────────────────────────────────────────────

    /**
     * Integer bin lag that best aligns {@code hist} onto {@code refHist} by full
     * cross-correlation — UniFORM's {@code compute_correlation_shifts}
     * ({@code argmax(correlate(hist, ref, 'full')) − (N−1)}). A <b>positive</b> lag means
     * {@code hist} sits at higher intensities than the reference (it must be scaled
     * <i>down</i> to align), which {@link #shiftToScale} turns into a factor {@code < 1}.
     * Ties resolve to the lag of smallest magnitude nearest zero (lowest index in the
     * full-correlation array), matching {@code numpy.argmax}. Both arrays must share a
     * length {@code N}.
     */
    public static int crossCorrelationShift(long[] hist, long[] refHist) {
        int n = hist.length;
        if (refHist.length != n) {
            throw new IllegalArgumentException("hist and refHist must have the same length");
        }
        if (n == 0) {
            return 0;
        }
        long bestCorr = Long.MIN_VALUE;
        int bestLag = 0;
        // Full correlation lags run from -(n-1) .. (n-1); numpy's argmax over the
        // full array corresponds to scanning lag ascending and keeping the first max.
        for (int lag = -(n - 1); lag <= n - 1; lag++) {
            long corr = 0;
            int iStart = Math.max(0, lag);
            int iEnd = Math.min(n - 1, n - 1 + lag);
            for (int i = iStart; i <= iEnd; i++) {
                corr += hist[i] * refHist[i - lag];
            }
            if (corr > bestCorr) {
                bestCorr = corr;
                bestLag = lag;
            }
        }
        return bestLag;
    }

    // ── Shift → scale ────────────────────────────────────────────────────────

    /**
     * Convert an integer bin {@code shift} into the multiplicative correction factor
     * applied to raw intensities: {@code exp(−shift · increment)}, where
     * {@code increment = (logMax − logMin)/(nBins − 1)} is the log-intensity width of one
     * bin (UniFORM's {@code calculate_shift_in_log_pixels} + {@code exp(negated shift)}).
     * A positive shift (sample brighter than reference) yields a factor {@code < 1}.
     * Returns {@code 1.0} (no-op) for a degenerate range or {@code nBins ≤ 1}.
     */
    public static double shiftToScale(int shift, double logMin, double logMax, int nBins) {
        if (nBins <= 1) {
            return 1.0;
        }
        double range = logMax - logMin;
        if (!(range > 0) || Double.isNaN(range)) {
            return 1.0;
        }
        double increment = range / (nBins - 1);
        return Math.exp(-shift * increment);
    }
}
