package qupath.ext.celltune.classifier;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Covers the pure part of LightGBM's native early stopping: choosing which of the booster's eval
 * metrics to early-stop on.
 * <p>
 * The surrounding code is native and untestable without loading {@code lib_lightgbm.so}, but this
 * is where an off-by-one silently early-stops on the wrong metric, so it is worth pinning. A
 * {@code -1} result is the signal to fall back to the manual log-loss computation.
 */
class LightGBMMetricIndexTest {

    @Test
    @DisplayName("multiclass picks multi_logloss, binary picks binary_logloss")
    void picksTheConfiguredMetric() {
        assertEquals(0, LightGBMModel.resolveMetricIndex(new String[] {"multi_logloss"}, 5));
        assertEquals(0, LightGBMModel.resolveMetricIndex(new String[] {"binary_logloss"}, 2));
        assertEquals(1, LightGBMModel.resolveMetricIndex(new String[] {"auc", "multi_logloss"}, 3));
        assertEquals(2, LightGBMModel.resolveMetricIndex(new String[] {"auc", "map", "binary_logloss"}, 2));
    }

    @Test
    @DisplayName("prefers the metric matching the class count over another logloss")
    void prefersTheMatchingMetric() {
        // A binary run must not early-stop on multi_logloss just because it is listed first.
        assertEquals(1, LightGBMModel.resolveMetricIndex(new String[] {"multi_logloss", "binary_logloss"}, 2));
        assertEquals(1, LightGBMModel.resolveMetricIndex(new String[] {"binary_logloss", "multi_logloss"}, 4));
    }

    @Test
    @DisplayName("falls back to any logloss when the exact name is absent")
    void fallsBackToAnyLogloss() {
        assertEquals(1, LightGBMModel.resolveMetricIndex(new String[] {"auc", "some_logloss"}, 3));
    }

    @Test
    @DisplayName("returns -1 when there is nothing usable, so the caller scores manually")
    void signalsFallbackWhenAbsent() {
        assertEquals(-1, LightGBMModel.resolveMetricIndex(new String[] {}, 3));
        assertEquals(-1, LightGBMModel.resolveMetricIndex(null, 3));
        assertEquals(-1, LightGBMModel.resolveMetricIndex(new String[] {"auc", "ndcg"}, 3));
    }
}
