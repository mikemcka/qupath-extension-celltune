package qupath.ext.spclassify.classifier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import qupath.ext.spclassify.util.TrainingThreads;

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

    // ── Class balancing happens inside the fold ─────────────────────────────────

    /**
     * The leak this guards against: balancing the whole dataset and splitting afterwards puts
     * SMOTE's synthetic rows — each an interpolation of two real rows — into the fold being scored
     * while their parents sit in the fold being trained on. The search then reads back its own
     * training data, and rewards whichever settings memorise hardest.
     * <p>
     * Both properties are asserted from one run because they are two halves of the same fix: the
     * resampler must see fold-sized training portions (after the split, not before), and it must
     * see them once per fold rather than once per trial per fold.
     */
    @Test
    @DisplayName("balancing is applied per fold's training rows, once, never to the whole dataset")
    void foldResamplerRunsOncePerFoldOnTrainingRowsOnly() {
        int nFolds = 4;
        int nTrials = 3;
        List<Integer> sizesSeen = Collections.synchronizedList(new ArrayList<>());

        HyperparameterTuner.FoldResampler recording = (data, labels, size) -> {
            sizesSeen.add(size);
            return new HyperparameterTuner.Resampled(data, labels, size); // identity
        };

        HyperparameterTuner.tune(
                data(), labels(), N_SAMPLES, N_FEATURES, N_CLASSES, nTrials, nFolds, 60, 60, recording, s -> {});

        assertEquals(
                nFolds,
                sizesSeen.size(),
                "balancing must run once per fold during preparation, not once per trial per fold — "
                        + "re-running it per trial repeats the most expensive phase of a run");

        for (int size : sizesSeen) {
            assertTrue(
                    size < N_SAMPLES,
                    "the resampler was handed " + size + " of " + N_SAMPLES + " rows — it must see one fold's"
                            + " training portion, which means the split already happened");
        }

        int total = sizesSeen.stream().mapToInt(Integer::intValue).sum();
        assertEquals(
                N_SAMPLES * (nFolds - 1),
                total,
                "every row must appear in exactly k-1 training folds; a different total means the"
                        + " training portions are not a clean complement of the scored folds");
    }

    @Test
    @DisplayName("with no resampler the tuner still runs")
    void absentResamplerIsAllowed() {
        var result = HyperparameterTuner.tune(
                data(), labels(), N_SAMPLES, N_FEATURES, N_CLASSES, 3, 3, 60, 60, null, s -> {});
        assertNotNull(result);
        assertEquals(60, result.xgbParams().numRounds());
    }

    // ── Leaf-bounded depth search ──────────────────────────────────────────────

    @Test
    @DisplayName("the depth ceiling is derived from the leaf cap, not hard-coded")
    void leafBoundedDepthTracksTheLeafCap() {
        // 2^5 = 32 is the first power of two to reach LightGBM's default 31 leaves, so depths
        // beyond 5 describe the same tree.
        assertEquals(5, HyperparameterTuner.leafBoundedDepth(31));
        assertEquals(2, HyperparameterTuner.leafBoundedDepth(4), "2^2 already reaches a 4-leaf cap");
        assertEquals(10, HyperparameterTuner.leafBoundedDepth(1000));
        assertEquals(
                12,
                HyperparameterTuner.leafBoundedDepth(1 << 20),
                "a cap past the search space must clamp to DEPTH_MAX rather than run away");
    }

    // ── Fold count follows the rarest class ────────────────────────────────────

    /** The balanced label vector {@link #labels()} produces, as the int[] the guard takes. */
    private static int[] balancedIntLabels() {
        int[] out = new int[N_SAMPLES];
        for (int i = 0; i < N_SAMPLES; i++) out[i] = i % N_CLASSES;
        return out;
    }

    private static int[] labelsWithRarestClassOf(int rarest) {
        // Class 0 is the rare one; the rest split what remains.
        int[] out = new int[N_SAMPLES];
        for (int i = 0; i < N_SAMPLES; i++) out[i] = 1 + (i % (N_CLASSES - 1));
        for (int i = 0; i < rarest; i++) out[i] = 0;
        return out;
    }

    @Test
    @DisplayName("folds are clamped to what the rarest class can support, and the clamp is announced")
    void foldCountFollowsTheRarestClass() {
        List<String> log = new ArrayList<>();
        int folds = HyperparameterTuner.resolveFolds(labelsWithRarestClassOf(3), N_CLASSES, 5, log::add);

        assertEquals(3, folds, "5 folds cannot each train on a 3-row class");
        assertTrue(
                log.stream().anyMatch(s -> s.contains("least populated")),
                "a silent clamp is worse than none — the run must say why it used fewer folds: " + log);
    }

    @Test
    @DisplayName("a well-populated rarest class leaves the requested fold count alone")
    void foldCountIsUntouchedWhenEveryClassIsWellPopulated() {
        List<String> log = new ArrayList<>();
        assertEquals(5, HyperparameterTuner.resolveFolds(balancedIntLabels(), N_CLASSES, 5, log::add));
        assertTrue(log.isEmpty(), "nothing to warn about when every class clears the fold count: " + log);
    }

    @Test
    @DisplayName("a single-row class cannot be scored at all, and is called out rather than clamped away")
    void singleRowClassIsReportedAsUnscoreable() {
        List<String> log = new ArrayList<>();
        int folds = HyperparameterTuner.resolveFolds(labelsWithRarestClassOf(1), N_CLASSES, 5, log::add);

        assertEquals(2, folds, "the fold count must not drop below a two-way split");
        assertTrue(
                log.stream().anyMatch(s -> s.contains("too few for cross-validation")),
                "a class no fold can train on must be named, not silently scored as a failure: " + log);
    }

    @Test
    @DisplayName("classes with no labelled rows do not drag the fold count down")
    void unlabelledClassesAreIgnoredByTheFoldGuard() {
        List<String> log = new ArrayList<>();
        // Declared class count exceeds what is actually labelled — a class the user added but has
        // not labelled yet says nothing about how finely the labelled data can be split.
        assertEquals(5, HyperparameterTuner.resolveFolds(balancedIntLabels(), N_CLASSES + 3, 5, log::add));
        assertTrue(log.isEmpty(), "an unlabelled class is not a rare class: " + log);
    }

    // ── Column sampling is searched ────────────────────────────────────────────

    @Test
    @DisplayName("colsample is searched, inside its bounds")
    void colsampleIsSearchedWithinBounds() {
        List<String> log = new ArrayList<>();
        HyperparameterTuner.tune(data(), labels(), N_SAMPLES, N_FEATURES, N_CLASSES, 8, 3, 60, 60, log::add);

        List<String> trials = trialLines(log);
        assertTrue(trials.size() >= 8, "expected trial lines, got: " + trials);

        long distinct = trials.stream()
                .map(s -> s.substring(s.indexOf("colsample=")))
                .distinct()
                .count();
        assertTrue(distinct > 1, "colsample never varied — the dimension is not being searched: " + trials);

        for (String line : trials) {
            double col = Double.parseDouble(
                    line.substring(line.indexOf("colsample=") + 10).split("[^0-9.]")[0]);
            assertTrue(
                    col >= HyperparameterTuner.COL_MIN - 1e-6 && col <= HyperparameterTuner.COL_MAX + 1e-6,
                    "colsample " + col + " escaped its bounds: " + line);
        }
    }

    @Test
    @DisplayName("LightGBM trials stay inside the depth range its leaf cap can distinguish")
    void lightGbmDepthSearchStopsAtTheLeafBound() {
        List<String> log = new ArrayList<>();
        HyperparameterTuner.tune(data(), labels(), N_SAMPLES, N_FEATURES, N_CLASSES, 8, 3, 60, 60, log::add);

        int bound = HyperparameterTuner.leafBoundedDepth(LightGBMModel.NUM_LEAVES);
        boolean inLightGbm = false;
        int lightGbmTrials = 0;
        for (String line : log) {
            if (line.contains("Tuning LightGBM")) inLightGbm = true;
            if (!inLightGbm || !line.contains("Trial ")) continue;
            lightGbmTrials++;
            int depth = Integer.parseInt(
                    line.substring(line.indexOf("depth=") + 6, line.indexOf(",", line.indexOf("depth="))));
            assertTrue(
                    depth <= bound,
                    "a LightGBM trial searched depth " + depth + " above the leaf-bound " + bound
                            + "; those trials re-score a model identical to depth " + bound + ": " + line);
        }
        assertTrue(lightGbmTrials > 0, "no LightGBM trial lines were logged: " + log);
    }
}
