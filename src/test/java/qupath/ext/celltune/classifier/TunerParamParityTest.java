package qupath.ext.celltune.classifier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Guards the rule that cross-validation must score the model that will actually be built.
 * <p>
 * {@link HyperparameterTuner} used to assemble its own parameter strings for both libraries. The
 * LightGBM copy omitted {@code min_gain_to_split}, so every trial was scored on an unconstrained
 * booster and the winning hyperparameters were then handed to a constrained one — the tuner was
 * optimising a model that never existed. The XGBoost copy was equivalent by luck, right up until
 * {@code max_bin} was added on the other side.
 * <p>
 * Both now call the model classes' own builders, and these tests pin the properties that make that
 * worth doing: every model-affecting parameter is present, and the thread budget is the one thing
 * a caller may legitimately vary.
 */
class TunerParamParityTest {

    // ── LightGBM ────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("LightGBM params carry min_gain_to_split — the one the tuner used to drop")
    void lightGbmParamsIncludeMinGainToSplit() {
        String p = LightGBMModel.buildParams(5, 6, 0.05f, 0.8f, 4);
        assertTrue(p.contains(" min_gain_to_split=10"), "min_gain_to_split missing: " + p);
    }

    @Test
    @DisplayName("LightGBM params are identical bar the thread budget")
    void lightGbmParamsVaryOnlyByThreads() {
        String a = LightGBMModel.buildParams(5, 6, 0.05f, 0.8f, 4);
        String b = LightGBMModel.buildParams(5, 6, 0.05f, 0.8f, 16);
        assertEquals(a.replace(" num_threads=4", ""), b.replace(" num_threads=16", ""));
    }

    @Test
    @DisplayName("LightGBM objective and metric follow the class count")
    void lightGbmObjectiveFollowsClassCount() {
        String binary = LightGBMModel.buildParams(2, 6, 0.05f, 0.8f, 4);
        assertTrue(binary.contains("objective=binary"), binary);
        assertTrue(binary.contains("metric=binary_logloss"), binary);
        assertFalse(binary.contains("num_class="), "binary must not declare num_class: " + binary);

        String multi = LightGBMModel.buildParams(7, 6, 0.05f, 0.8f, 4);
        assertTrue(multi.contains("objective=multiclass"), multi);
        assertTrue(multi.contains("metric=multi_logloss"), multi);
        assertTrue(multi.contains("num_class=7"), multi);
    }

    // ── XGBoost ─────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("XGBoost params carry the settings every measurement assumes")
    void xgboostParamsIncludeModelAffectingSettings() {
        Map<String, Object> p = XGBoostModel.buildParams(5, 6, 0.3f, 0.8f, 4);
        assertEquals("hist", p.get("tree_method"), "hist is what the benchmarks and timings assume");
        assertEquals("cpu", p.get("device"), "this build ships no CUDA kernels");
        assertEquals(4, p.get("nthread"));
        assertEquals(5, p.get("num_class"));
        assertEquals("multi:softprob", p.get("objective"));
    }

    @Test
    @DisplayName("max_bin reaches the tuner's params too, or it scores the wrong model")
    void xgboostMaxBinIsInTheSharedBuilder() {
        int original = XGBoostModel.getMaxBin();
        try {
            XGBoostModel.setMaxBin(0);
            assertFalse(
                    XGBoostModel.buildParams(5, 6, 0.3f, 0.8f, 4).containsKey("max_bin"),
                    "max_bin must be absent at the default so XGBoost applies its own 256");

            XGBoostModel.setMaxBin(64);
            assertEquals(
                    64,
                    XGBoostModel.buildParams(5, 6, 0.3f, 0.8f, 4).get("max_bin"),
                    "a configured max_bin must reach every caller of the shared builder, tuner included");
        } finally {
            XGBoostModel.setMaxBin(original);
        }
    }

    @Test
    @DisplayName("binary XGBoost drops num_class")
    void xgboostBinaryOmitsNumClass() {
        Map<String, Object> p = XGBoostModel.buildParams(2, 6, 0.3f, 0.8f, 4);
        assertEquals("binary:logistic", p.get("objective"));
        assertEquals("logloss", p.get("eval_metric"));
        assertFalse(p.containsKey("num_class"), "binary must not declare num_class");
    }
}
