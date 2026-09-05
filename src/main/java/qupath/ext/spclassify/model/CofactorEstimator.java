package qupath.ext.spclassify.model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import qupath.ext.spclassify.util.RobustStats;

/**
 * Pure numerical core of the Phase 17 cofactor-suggestion tool: turns each selected feature's raw
 * per-cell values into a per-feature suggested arcsinh cofactor and aggregates them into one
 * recommended global cofactor.
 *
 * <p>Every method is static and a pure function of primitive arrays (no JavaFX, no QuPath types), so
 * the estimator is unit-testable in isolation against synthetic per-cell distributions — mirroring
 * {@code NeighborhoodModel} and {@code LeidenModel}.
 *
 * <h2>The estimand</h2>
 * {@code arcsinh(x/c)} is near-linear for {@code x << c} and log-compressed for {@code x >> c}, so a
 * good cofactor {@code c} sits at the background/signal knee — i.e. it tracks the <em>background</em>
 * (negative) intensity level, the low end of the per-cell distribution, not a high signal percentile.
 * For a marker expressed by a minority of cells (the common multiplex case) the median of the
 * per-cell distribution falls inside the background population and therefore reports the background
 * level. The single fixed {@link #BACKGROUND_PERCENTILE} ({@code 50.0}) is applied identically to
 * every feature (D-01/D-02/D-03); it hits both the raw-fluorescence (global in the tens) and MIBI
 * (global ≈ 0.05) targets because those panels differ only in intensity scale.
 *
 * <p>The recommended global cofactor is the median of the per-feature cofactors over the features
 * that were <em>not</em> excluded (D-10/D-11): dead features (near-zero spread across cells) and
 * saturated features (background at the spinner ceiling) are flagged and dropped from the median.
 * Every returned cofactor is clamped to the arcsinh spinner range {@code [0.01, 10000]} so it is a
 * valid positive value for {@code FeatureNormalizer.setArcsinhCofactor} (which throws on {@code <= 0}).
 */
public final class CofactorEstimator {

    /** The single fixed background percentile applied to every feature (D-01/D-02/D-03). */
    public static final double BACKGROUND_PERCENTILE = 50.0;

    /** Spinner lower bound — {@code setArcsinhCofactor} throws on {@code <= 0}. */
    private static final double MIN_COFACTOR = 0.01;

    /** Spinner upper bound; a background at/above this is treated as saturated. */
    private static final double MAX_COFACTOR = 10000.0;

    /** Near-zero spread across cells ⇒ dead / all-background feature. */
    private static final double DEAD_EPS = 1e-9;

    /** Platform-default cofactor used when every feature is excluded. */
    private static final double NEUTRAL_FALLBACK = 1.0;

    private CofactorEstimator() {}

    /**
     * Per-feature diagnostic row (D-09): the value-scale summary plus the suggested cofactor and the
     * dead/saturated exclusion flag.
     *
     * @param feature    feature (marker) name
     * @param nCells     number of finite per-cell values contributing
     * @param background background estimate — the {@link #BACKGROUND_PERCENTILE} of the distribution
     * @param median     median (== {@code background}; the same p50 statistic seeds both)
     * @param p99        99th percentile — the signal-scale summary
     * @param cofactor   suggested cofactor, clamped to {@code [0.01, 10000]}
     * @param excluded   whether this feature is excluded from the global median (dead or saturated)
     * @param reason     human-readable exclusion reason, or {@code ""} when not excluded
     */
    public record FeatureCofactor(
            String feature,
            int nCells,
            double background,
            double median,
            double p99,
            double cofactor,
            boolean excluded,
            String reason) {}

    /**
     * The recommended global cofactor plus the per-feature diagnostic rows that produced it.
     *
     * @param perFeature     one {@link FeatureCofactor} per input feature, in input order
     * @param globalCofactor median of the non-excluded per-feature cofactors, clamped to range;
     *                       {@code 1.0} when every feature is excluded
     */
    public record CofactorSuggestion(List<FeatureCofactor> perFeature, double globalCofactor) {
        public CofactorSuggestion {
            perFeature = perFeature == null ? List.of() : List.copyOf(perFeature);
        }
    }

    /**
     * Estimate the per-feature cofactor from one feature's raw per-cell values. NaN values are
     * dropped (they mean "not measured", not "true zero background"). Never throws — a degenerate
     * input (empty / all-NaN) returns a neutral, excluded row.
     *
     * @param feature   feature name (echoed into the result)
     * @param rawValues raw per-cell values (may contain NaN); not modified
     * @return the per-feature diagnostic row
     */
    public static FeatureCofactor estimateFeature(String feature, double[] rawValues) {
        // Drop NaN — percentileSorted does not filter it, and NaN means "not measured".
        double[] finite = dropNaN(rawValues);
        Arrays.sort(finite); // percentileSorted requires ascending input.
        int n = finite.length;

        if (n == 0) {
            return new FeatureCofactor(
                    feature, 0, Double.NaN, Double.NaN, Double.NaN, NEUTRAL_FALLBACK, true, "no finite values");
        }

        double background = ImagePixelStats.percentileSorted(finite, BACKGROUND_PERCENTILE);
        double median = background; // same p50 statistic seeds the median column (D-09/Research A2).
        double p99 = ImagePixelStats.percentileSorted(finite, 99.0);
        double spread = finite[n - 1] - finite[0];

        boolean saturated = background >= MAX_COFACTOR;
        boolean dead = !saturated && spread < DEAD_EPS;
        boolean excluded = dead || saturated;
        String reason;
        if (saturated) {
            reason = "saturated (background at range ceiling)";
        } else if (dead) {
            reason = "dead (near-zero variance)";
        } else {
            reason = "";
        }

        double cofactor = clamp(background);
        return new FeatureCofactor(feature, n, background, median, p99, cofactor, excluded, reason);
    }

    /**
     * Estimate per-feature cofactors for a set of features and aggregate them into one recommended
     * global cofactor (COF-06/D-10).
     *
     * @param featureNames feature names, index {@code i} pairs with {@code columns[i]}
     * @param columns      raw per-cell values, one column per feature
     * @return the per-feature rows plus the recommended global cofactor
     */
    public static CofactorSuggestion estimate(String[] featureNames, double[][] columns) {
        int nFeatures = columns == null ? 0 : columns.length;
        List<FeatureCofactor> results = new ArrayList<>(nFeatures);
        for (int i = 0; i < nFeatures; i++) {
            String name = (featureNames != null && i < featureNames.length) ? featureNames[i] : ("Feature " + (i + 1));
            results.add(estimateFeature(name, columns[i]));
        }
        return new CofactorSuggestion(results, globalCofactor(results));
    }

    /**
     * Median of the per-feature cofactors over non-excluded features, clamped to the spinner range
     * (COF-06/D-10/D-11). Falls back to the neutral {@code 1.0} when every feature is excluded.
     *
     * @param results the per-feature rows
     * @return the recommended global cofactor in {@code [0.01, 10000]}
     */
    public static double globalCofactor(List<FeatureCofactor> results) {
        double[] keep = results.stream()
                .filter(r -> !r.excluded())
                .mapToDouble(FeatureCofactor::cofactor)
                .toArray();
        if (keep.length == 0) {
            return NEUTRAL_FALLBACK;
        }
        return clamp(RobustStats.median(keep));
    }

    /** Copy {@code values} without NaN entries (null → empty). */
    private static double[] dropNaN(double[] values) {
        if (values == null || values.length == 0) {
            return new double[0];
        }
        double[] out = new double[values.length];
        int k = 0;
        for (double v : values) {
            if (!Double.isNaN(v)) {
                out[k++] = v;
            }
        }
        return k == values.length ? out : Arrays.copyOf(out, k);
    }

    /** Clamp to the arcsinh spinner range {@code [0.01, 10000]}. */
    private static double clamp(double v) {
        return Math.max(MIN_COFACTOR, Math.min(MAX_COFACTOR, v));
    }
}
