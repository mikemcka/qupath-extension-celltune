package qupath.ext.spclassify.model;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;
import qupath.ext.spclassify.model.CofactorEstimator.CofactorSuggestion;
import qupath.ext.spclassify.model.CofactorEstimator.FeatureCofactor;

/**
 * Unit tests for {@link CofactorEstimator}: the pure background-percentile cofactor math behind the
 * Phase 17 cofactor-suggestion tool (COF-04, COF-05, COF-06, D-01..D-04, D-09..D-11). Synthetic
 * per-cell distributions only — no QuPath/JavaFX APIs — mirroring {@code NeighborhoodModelTest} and
 * {@code LeidenModelTest}. Determinism via {@code new Random(42)} + an {@code EPS} tolerance.
 */
class CofactorEstimatorTest {

    private static final double EPS = 1e-9;

    /** Log-normal spread of both the background and positive populations (multiplicative). */
    private static final double SIGMA = 0.3;

    // ── Synthetic generators ─────────────────────────────────────────────────

    /**
     * One feature column of {@code n} per-cell values: a Bernoulli mixture of a background log-normal
     * (median {@code bgMedian}) and a well-separated positive log-normal (median {@code posMedian}),
     * with per-cell positive probability {@code posFraction}. Deterministic under the supplied rng.
     */
    private static double[] lognormalColumn(Random rng, int n, double posFraction, double bgMedian, double posMedian) {
        double[] col = new double[n];
        for (int i = 0; i < n; i++) {
            boolean positive = rng.nextDouble() < posFraction;
            double median = positive ? posMedian : bgMedian;
            col[i] = median * Math.exp(SIGMA * rng.nextGaussian());
        }
        return col;
    }

    /** A panel of {@code fracs.length} markers at a fixed intensity scale, generated from one rng. */
    private static double[][] panel(Random rng, int nPerMarker, double[] fracs, double bgMedian, double posMedian) {
        double[][] cols = new double[fracs.length][];
        for (int m = 0; m < fracs.length; m++) {
            cols[m] = lognormalColumn(rng, nPerMarker, fracs[m], bgMedian, posMedian);
        }
        return cols;
    }

    private static String[] names(int count) {
        String[] out = new String[count];
        for (int i = 0; i < count; i++) {
            out[i] = "M" + i;
        }
        return out;
    }

    // ── COF-04: known-background recovery within factor 2 ─────────────────────

    @Test
    void recoversKnownBackgroundWithinFactorTwo() {
        Random rng = new Random(42);
        double injectedBackground = 30.0;
        // 30% positive around 800 (well separated), 70% background around 30.
        double[] col = lognormalColumn(rng, 3000, 0.30, injectedBackground, 800.0);

        double cofactor = CofactorEstimator.estimateFeature("m", col).cofactor();

        // Factor-of-2 band around the injected background: [B/2, 2B] = [15, 60].
        assertTrue(
                cofactor >= injectedBackground / 2.0 && cofactor <= injectedBackground * 2.0,
                "cofactor " + cofactor + " not within factor 2 of injected background " + injectedBackground);
    }

    // ── COF-04: fixed p50 background percentile ───────────────────────────────

    @Test
    void backgroundPercentileConstantIsFifty() {
        assertEquals(50.0, CofactorEstimator.BACKGROUND_PERCENTILE, 0.0);

        // p50 of a 5-point ramp {0,10,20,30,40} is exactly the midpoint 20.
        double cofactor = CofactorEstimator.estimateFeature("m", new double[] {0, 10, 20, 30, 40})
                .cofactor();
        assertEquals(20.0, cofactor, EPS);
    }

    // ── COF-06: global = median of per-feature cofactors ──────────────────────

    @Test
    void globalIsMedianOfPerFeatureCofactors() {
        // Three ramps whose p50s are 10, 20, 30 → global median = 20.
        String[] featureNames = {"a", "b", "c"};
        double[][] columns = {
            {0, 5, 10, 15, 20}, // p50 = 10
            {0, 10, 20, 30, 40}, // p50 = 20
            {10, 20, 30, 40, 50}, // p50 = 30
        };

        CofactorSuggestion suggestion = CofactorEstimator.estimate(featureNames, columns);
        assertEquals(20.0, suggestion.globalCofactor(), EPS);
    }

    // ── D-11: dead & saturated features flagged and excluded from the median ──

    @Test
    void excludesDeadSaturatedFromGlobalMedian() {
        String[] featureNames = {"dead", "saturated", "good10", "good30"};
        double[][] columns = {
            {5, 5, 5, 5, 5}, // dead: zero spread
            {15000, 17500, 20000, 22500, 25000}, // saturated: p50 = 20000 >= ceiling 10000
            {0, 5, 10, 15, 20}, // good: p50 = 10
            {10, 20, 30, 40, 50}, // good: p50 = 30
        };

        CofactorSuggestion suggestion = CofactorEstimator.estimate(featureNames, columns);
        List<FeatureCofactor> perFeature = suggestion.perFeature();

        assertTrue(perFeature.get(0).excluded(), "constant column should be flagged dead");
        assertTrue(perFeature.get(1).excluded(), "range-ceiling column should be flagged saturated");
        assertFalse(perFeature.get(2).excluded(), "good p50=10 column must not be excluded");
        assertFalse(perFeature.get(3).excluded(), "good p50=30 column must not be excluded");

        // Excluded rows do NOT enter the median: median{10, 30} = 20.
        assertEquals(20.0, suggestion.globalCofactor(), EPS);
    }

    // ── Acceptance: raw-fluorescence panel → global in the tens ───────────────

    @Test
    void rawFluorescenceRangeLandsInTens() {
        Random rng = new Random(42);
        double[] fracs = {0.05, 0.13, 0.21, 0.29, 0.37, 0.45};
        double[][] columns = panel(rng, 2000, fracs, 30.0, 800.0);

        CofactorSuggestion suggestion = CofactorEstimator.estimate(names(fracs.length), columns);
        double global = suggestion.globalCofactor();

        // In the tens: well above the platform default (1) and well below the no-op (~150).
        assertTrue(global > 10.0 && global < 80.0, "raw-fluorescence global cofactor out of band: " + global);
    }

    // ── Acceptance: MIBI-scale panel → global ≈ 0.05 ──────────────────────────

    @Test
    void mibiRangeLandsNearPointZeroFive() {
        Random rng = new Random(42);
        double[] fracs = {0.05, 0.13, 0.21, 0.29, 0.37, 0.45};
        double[][] columns = panel(rng, 2000, fracs, 0.04, 1.5);

        CofactorSuggestion suggestion = CofactorEstimator.estimate(names(fracs.length), columns);
        double global = suggestion.globalCofactor();

        // Order-of-magnitude ≈ 0.05 (Hartmann et al. 2021 MIBI cofactor).
        assertTrue(global > 0.02 && global < 0.15, "MIBI global cofactor out of band: " + global);
    }

    // ── COF-06 robustness: degenerate inputs never throw ──────────────────────

    @Test
    void degenerateInputReturnsNeutralWithoutThrowing() {
        // Empty column → n=0, excluded, no throw.
        FeatureCofactor empty = CofactorEstimator.estimateFeature("m", new double[0]);
        assertEquals(0, empty.nCells());
        assertTrue(empty.excluded());

        // All-NaN column → n=0 (NaN dropped), excluded, no throw.
        FeatureCofactor allNaN =
                CofactorEstimator.estimateFeature("m", new double[] {Double.NaN, Double.NaN, Double.NaN});
        assertEquals(0, allNaN.nCells());
        assertTrue(allNaN.excluded());

        // Single-value column → that value is the background estimate.
        FeatureCofactor single = CofactorEstimator.estimateFeature("m", new double[] {42.0});
        assertEquals(42.0, single.background(), EPS);

        // Every feature excluded → global falls back to the neutral positive cofactor 1.0.
        CofactorSuggestion allExcluded =
                CofactorEstimator.estimate(new String[] {"dead1", "dead2"}, new double[][] {{5, 5, 5}, {7, 7, 7}});
        assertEquals(1.0, allExcluded.globalCofactor(), EPS);
    }
}
