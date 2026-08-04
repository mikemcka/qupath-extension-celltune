package qupath.ext.celltune.classifier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import qupath.ext.celltune.util.TrainingThreads;

/**
 * Covers the tuner's contract with its caller rather than the quality of its search.
 * <p>
 * The bug worth guarding against is structural: the tuner used to search over the round count and
 * overwrite whatever early stopping had measured, so with both options enabled the round search —
 * the single most expensive phase of a run — was performed and then discarded. Rounds are now
 * held fixed when the caller supplies them, which only means anything if every suggested trial
 * really does carry that value.
 */
class HyperparameterTunerTest {

    private static final int N_CLASSES = 4;
    private static final int N_FEATURES = 12;
    private static final int N_SAMPLES = 200;

    private static float[] data() {
        float[] d = new float[N_SAMPLES * N_FEATURES];
        for (int i = 0; i < N_SAMPLES; i++) {
            int cls = i % N_CLASSES;
            for (int f = 0; f < N_FEATURES; f++) {
                long h = ((long) i * 2654435761L) ^ ((long) f * 40503L);
                d[i * N_FEATURES + f] = ((h >>> 8) % 9973) / 9973f + (f < 3 ? cls * 0.5f : 0f);
            }
        }
        return d;
    }

    private static float[] labels() {
        float[] l = new float[N_SAMPLES];
        for (int i = 0; i < N_SAMPLES; i++) l[i] = i % N_CLASSES;
        return l;
    }

    /** Every "Trial …" line the tuner logged, so the suggestions themselves can be inspected. */
    private static List<String> trialLines(List<String> log) {
        List<String> out = new ArrayList<>();
        for (String s : log) if (s.contains("Trial ")) out.add(s.trim());
        return out;
    }

    @Test
    @DisplayName("a supplied round count is used by every trial, not searched over")
    void fixedRoundsAreHeldAcrossEveryTrial() {
        List<String> log = new ArrayList<>();
        int fixed = 137; // deliberately not a multiple of 10 — the search space snaps to those

        HyperparameterTuner.tune(data(), labels(), N_SAMPLES, N_FEATURES, N_CLASSES, 6, 3, fixed, fixed, log::add);

        List<String> trials = trialLines(log);
        assertTrue(trials.size() >= 6, "expected trial lines, got: " + trials);
        for (String line : trials) {
            assertTrue(
                    line.contains("rounds=" + fixed),
                    "a trial searched over rounds instead of holding the supplied value: " + line);
        }
    }

    @Test
    @DisplayName("with no supplied round count the search still explores rounds")
    void roundsAreSearchedWhenNotSupplied() {
        List<String> log = new ArrayList<>();

        var result = HyperparameterTuner.tune(data(), labels(), N_SAMPLES, N_FEATURES, N_CLASSES, 8, 3, log::add);

        assertNotNull(result);
        List<String> trials = trialLines(log);
        long distinct = trials.stream()
                .map(s -> s.substring(s.indexOf("rounds="), s.indexOf(",")))
                .distinct()
                .count();
        assertTrue(distinct > 1, "rounds never varied across trials — the search space collapsed: " + trials);
    }

    @Test
    @DisplayName("the fixed round count reaches the returned parameters")
    void fixedRoundsSurviveIntoTheResult() {
        int fixed = 90;
        var result = HyperparameterTuner.tune(
                data(), labels(), N_SAMPLES, N_FEATURES, N_CLASSES, 6, 3, fixed, fixed, s -> {});

        assertEquals(fixed, result.xgbParams().numRounds(), "XGBoost rounds were not held");
        assertEquals(fixed, result.lgbParams().numRounds(), "LightGBM rounds were not held");
    }

    @Test
    @DisplayName("tuning respects the training thread budget rather than the core count")
    void tuningHonoursTheThreadBudget() {
        int original = TrainingThreads.getOverride();
        try {
            TrainingThreads.setOverride(2);
            List<String> log = new ArrayList<>();
            HyperparameterTuner.tune(data(), labels(), N_SAMPLES, N_FEATURES, N_CLASSES, 3, 5, 60, 60, log::add);

            // With a budget of 2 and 5 folds there is not enough to give each fold 2 threads, so
            // the folds must run sequentially rather than announcing a parallel split.
            for (String line : log) {
                assertTrue(
                        !line.contains("Parallel CV"),
                        "folds were parallelised despite a 2-thread budget — the cap is being ignored: " + line);
            }
        } finally {
            TrainingThreads.setOverride(original);
        }
    }

    @Test
    @DisplayName("too few samples falls back to documented defaults")
    void tooFewSamplesUsesDefaults() {
        float[] d = new float[10 * N_FEATURES];
        float[] l = new float[10];
        var result = HyperparameterTuner.tune(d, l, 10, N_FEATURES, N_CLASSES, 5, 3, s -> {});
        assertEquals(200, result.xgbParams().numRounds());
        assertEquals(6, result.xgbParams().maxDepth());
    }
}
