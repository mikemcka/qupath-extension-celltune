package qupath.ext.spclassify.util;

import java.util.Arrays;

/**
 * Robust summary statistics (median, MAD, robust z-scores) shared by the
 * cohort-anomaly and pixel-cohort analyzers.
 * <p>
 * Two robust-z variants are deliberately kept distinct because the analyzers
 * need different behaviour on degenerate / missing data — they are <em>not</em>
 * interchangeable:
 * <ul>
 *   <li>{@link #robustZStrict(double[])} — assumes finite input; on a degenerate
 *       MAD (≈ 0) it returns all-zeros. Used for cell-level prediction anomalies
 *       where a flat distribution legitimately means "no outliers".</li>
 *   <li>{@link #robustZWithFallback(double[])} — NaN-aware; on a degenerate MAD
 *       it falls back to a classic mean/std z so a genuine outlier above an
 *       otherwise-flat baseline (e.g. saturation fraction where most images are
 *       0%) is still surfaced rather than collapsing to zero.</li>
 * </ul>
 * The {@code 0.6745} scale factor (≈ 1/1.4826⁻¹, i.e. the 0.75 quantile of the
 * standard normal) makes the MAD a consistent estimator of σ for Gaussian data.
 */
public final class RobustStats {

    /** Scales MAD to a standard-normal-consistent estimate of σ. */
    public static final double MAD_TO_SIGMA = 0.6745;

    private static final double EPS = 1e-12;

    private RobustStats() {} // utility class

    /**
     * Median of finite values. Returns {@code 0.0} for an empty array. Does not
     * special-case NaN (callers using this variant guarantee finite input).
     */
    public static double median(double[] values) {
        if (values.length == 0) {
            return 0.0;
        }
        double[] sorted = Arrays.copyOf(values, values.length);
        Arrays.sort(sorted);
        int mid = sorted.length / 2;
        if ((sorted.length & 1) == 1) {
            return sorted[mid];
        }
        return 0.5 * (sorted[mid - 1] + sorted[mid]);
    }

    /**
     * Median ignoring NaN entries. Returns {@code Double.NaN} when there are no
     * finite values.
     */
    public static double medianIgnoreNaN(double[] values) {
        int n = 0;
        for (double v : values) {
            if (!Double.isNaN(v)) {
                n++;
            }
        }
        if (n == 0) {
            return Double.NaN;
        }
        double[] finite = new double[n];
        int k = 0;
        for (double v : values) {
            if (!Double.isNaN(v)) {
                finite[k++] = v;
            }
        }
        Arrays.sort(finite);
        int mid = n / 2;
        return (n & 1) == 1 ? finite[mid] : 0.5 * (finite[mid - 1] + finite[mid]);
    }

    /**
     * Robust z-scores assuming finite input. On a degenerate MAD ({@code < 1e-12})
     * returns an all-zero array (a flat distribution has no outliers).
     *
     * @param values finite input values
     * @return one robust z-score per input value (empty array for empty input)
     */
    public static double[] robustZStrict(double[] values) {
        if (values.length == 0) {
            return new double[0];
        }
        double median = median(values);
        double[] absDeviations = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            absDeviations[i] = Math.abs(values[i] - median);
        }
        double mad = median(absDeviations);

        double[] out = new double[values.length];
        if (mad < EPS) {
            return out;
        }
        for (int i = 0; i < values.length; i++) {
            out[i] = MAD_TO_SIGMA * (values[i] - median) / mad;
        }
        return out;
    }

    /**
     * NaN-aware robust z-scores. NaN inputs map to NaN outputs. When the MAD is
     * degenerate, falls back to a classic mean/std z (computed over finite values)
     * so an outlier above an otherwise-flat baseline is still detected; if the
     * std is also degenerate the finite entries map to {@code 0.0}.
     *
     * @param values input values (may contain NaN)
     * @return one z-score per input value (NaN preserved)
     */
    public static double[] robustZWithFallback(double[] values) {
        double[] out = new double[values.length];
        double median = medianIgnoreNaN(values);
        if (Double.isNaN(median)) {
            Arrays.fill(out, Double.NaN);
            return out;
        }
        double[] absDev = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            absDev[i] = Double.isNaN(values[i]) ? Double.NaN : Math.abs(values[i] - median);
        }
        double mad = medianIgnoreNaN(absDev);

        if (!Double.isNaN(mad) && mad >= EPS) {
            for (int i = 0; i < values.length; i++) {
                out[i] = Double.isNaN(values[i]) ? Double.NaN : MAD_TO_SIGMA * (values[i] - median) / mad;
            }
            return out;
        }

        // Degenerate MAD — fall back to mean/std so outliers above a flat
        // baseline are still surfaced.
        double sum = 0.0;
        int n = 0;
        for (double v : values) {
            if (!Double.isNaN(v)) {
                sum += v;
                n++;
            }
        }
        double mean = n > 0 ? sum / n : Double.NaN;
        double sq = 0.0;
        for (double v : values) {
            if (!Double.isNaN(v)) {
                double d = v - mean;
                sq += d * d;
            }
        }
        double std = n > 0 ? Math.sqrt(sq / n) : Double.NaN;
        for (int i = 0; i < values.length; i++) {
            if (Double.isNaN(values[i])) {
                out[i] = Double.NaN;
            } else if (Double.isNaN(std) || std < EPS) {
                out[i] = 0.0;
            } else {
                out[i] = (values[i] - mean) / std;
            }
        }
        return out;
    }
}
