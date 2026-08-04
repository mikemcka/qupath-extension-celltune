package qupath.ext.celltune.classifier;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Pins the contract that lets the train/val metrics step reuse the round search's model instead of
 * rebuilding it.
 * <p>
 * The claim under test is narrow but load-bearing: a booster serialised at the round with the best
 * validation loss is the <em>same model</em> a fresh fit of that many rounds on the same fold would
 * produce. Boosting is sequential and seeded, so the rounds that came after the best one only
 * appended trees — but if that ever stopped holding, the metrics report would silently start
 * describing a different model than the one being reported on.
 * <p>
 * These exercise the real native library. If it cannot be loaded the tests skip rather than fail,
 * so a platform without the XGBoost natives does not break the build.
 */
class XGBoostRoundSearchTest {

    private static final int N_CLASSES = 4;
    private static final int N_FEATURES = 12;
    private static final int TRAIN_ROWS = 240;
    private static final int VAL_ROWS = 80;

    private static boolean nativesAvailable = false;

    @BeforeAll
    static void checkNatives() {
        try {
            Class.forName("ml.dmlc.xgboost4j.java.XGBoost");
            new ml.dmlc.xgboost4j.java.DMatrix(new float[] {0f, 1f}, 1, 2, Float.NaN).dispose();
            nativesAvailable = true;
        } catch (Throwable t) {
            nativesAvailable = false;
        }
    }

    /** Deterministic data with a learnable class signal in the leading columns. */
    private static void fill(float[] data, float[] labels, int rows, int salt) {
        for (int i = 0; i < rows; i++) {
            int cls = i % N_CLASSES;
            labels[i] = cls;
            int off = i * N_FEATURES;
            for (int f = 0; f < N_FEATURES; f++) {
                float base = (((i + salt) * 31L + f * 7L) % 997) / 997f;
                data[off + f] = f < 4 ? base + cls * 0.6f : base;
            }
        }
    }

    private record Fixture(float[] trainData, float[] trainLabels, float[] valData, float[] valLabels) {}

    private static Fixture fixture() {
        float[] trainData = new float[TRAIN_ROWS * N_FEATURES];
        float[] trainLabels = new float[TRAIN_ROWS];
        float[] valData = new float[VAL_ROWS * N_FEATURES];
        float[] valLabels = new float[VAL_ROWS];
        fill(trainData, trainLabels, TRAIN_ROWS, 0);
        fill(valData, valLabels, VAL_ROWS, 613);
        return new Fixture(trainData, trainLabels, valData, valLabels);
    }

    private static List<String> names(String prefix, int n) {
        List<String> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) out.add(prefix + i);
        return out;
    }

    private static XGBoostModel.RoundSearch search(Fixture f, boolean snapshot) throws Exception {
        return XGBoostModel.searchRounds(
                f.trainData(),
                f.trainLabels(),
                TRAIN_ROWS,
                f.valData(),
                f.valLabels(),
                VAL_ROWS,
                N_FEATURES,
                N_CLASSES,
                30,
                4,
                0.2f,
                0.8f,
                5,
                snapshot,
                s -> {});
    }

    @Test
    @DisplayName("the snapshot predicts identically to a fresh fit of the same round count")
    void snapshotMatchesAFreshFit() throws Exception {
        assumeTrue(nativesAvailable, "XGBoost natives unavailable");
        Fixture f = fixture();

        XGBoostModel.RoundSearch result = search(f, true);
        assertNotNull(result.bestModel(), "snapshot requested but not returned");

        // Restore the snapshot.
        XGBoostModel restored = new XGBoostModel();
        restored.loadFromBytes(result.bestModel(), names("class", N_CLASSES), names("f", N_FEATURES));

        // Train from scratch to the same round count — what the metrics step does today.
        XGBoostModel fresh = new XGBoostModel();
        fresh.train(
                f.trainData(),
                f.trainLabels(),
                TRAIN_ROWS,
                N_FEATURES,
                names("class", N_CLASSES),
                names("f", N_FEATURES),
                result.bestRounds(),
                4,
                0.2f,
                0.8f);

        float[][] fromSnapshot = restored.predictProba(f.valData(), VAL_ROWS, N_FEATURES);
        float[][] fromFresh = fresh.predictProba(f.valData(), VAL_ROWS, N_FEATURES);

        assertEquals(fromFresh.length, fromSnapshot.length, "row count");
        for (int i = 0; i < fromFresh.length; i++) {
            assertArrayEquals(
                    fromFresh[i],
                    fromSnapshot[i],
                    0f,
                    "row " + i + ": restored snapshot disagrees with a fresh fit of " + result.bestRounds()
                            + " rounds");
        }
    }

    @Test
    @DisplayName("the round count is unaffected by whether a snapshot was taken")
    void snapshottingDoesNotChangeTheSearch() throws Exception {
        assumeTrue(nativesAvailable, "XGBoost natives unavailable");
        Fixture f = fixture();
        assertEquals(
                search(f, false).bestRounds(),
                search(f, true).bestRounds(),
                "taking a snapshot changed which round the search picked");
    }

    @Test
    @DisplayName("no snapshot is produced or paid for when it is not requested")
    void noSnapshotWhenNotRequested() throws Exception {
        assumeTrue(nativesAvailable, "XGBoost natives unavailable");
        XGBoostModel.RoundSearch result = search(fixture(), false);
        assertNull(result.bestModel(), "snapshot returned despite not being requested");
        assertTrue(result.bestRounds() >= 1, "best round should be 1-indexed");
    }

    @Test
    @DisplayName("findBestRounds still returns just the round count")
    void legacyEntryPointStillWorks() throws Exception {
        assumeTrue(nativesAvailable, "XGBoost natives unavailable");
        Fixture f = fixture();
        int rounds = XGBoostModel.findBestRounds(
                f.trainData(),
                f.trainLabels(),
                TRAIN_ROWS,
                f.valData(),
                f.valLabels(),
                VAL_ROWS,
                N_FEATURES,
                N_CLASSES,
                30,
                4,
                0.2f,
                0.8f,
                5,
                s -> {});
        assertEquals(search(f, false).bestRounds(), rounds);
    }

    // ── max_bin ─────────────────────────────────────────────────────────────────

    private float[][] trainAndPredict(Fixture f) throws Exception {
        XGBoostModel m = new XGBoostModel();
        m.train(
                f.trainData(),
                f.trainLabels(),
                TRAIN_ROWS,
                N_FEATURES,
                names("class", N_CLASSES),
                names("f", N_FEATURES),
                12,
                4,
                0.2f,
                0.8f);
        return m.predictProba(f.valData(), VAL_ROWS, N_FEATURES);
    }

    /**
     * The whole safety argument for shipping the {@code max_bin} preference is that leaving it
     * alone changes nothing, so that has to be pinned rather than assumed: {@code 0} must produce
     * exactly what an explicit 256 does — XGBoost's own default.
     */
    @Test
    @DisplayName("max_bin=0 is XGBoost's default, not a separate setting")
    void defaultMaxBinIsANoOp() throws Exception {
        assumeTrue(nativesAvailable, "XGBoost natives unavailable");
        int original = XGBoostModel.getMaxBin();
        try {
            Fixture f = fixture();

            XGBoostModel.setMaxBin(0);
            float[][] asDefault = trainAndPredict(f);
            XGBoostModel.setMaxBin(256);
            float[][] asExplicit256 = trainAndPredict(f);

            for (int i = 0; i < asDefault.length; i++) {
                assertArrayEquals(asDefault[i], asExplicit256[i], 0f, "row " + i + ": max_bin=0 is not 256");
            }
        } finally {
            XGBoostModel.setMaxBin(original);
        }
    }

    /**
     * The converse: the preference must actually reach XGBoost. Without this, a silently-ignored
     * setting would look identical to a working default and the speed/accuracy trade-off a user
     * thinks they made would not exist.
     */
    @Test
    @DisplayName("a lowered max_bin reaches XGBoost and changes the model")
    void loweredMaxBinChangesTheModel() throws Exception {
        assumeTrue(nativesAvailable, "XGBoost natives unavailable");
        int original = XGBoostModel.getMaxBin();
        try {
            Fixture f = fixture();

            XGBoostModel.setMaxBin(0);
            float[][] fine = trainAndPredict(f);
            XGBoostModel.setMaxBin(4); // far coarser than any sane setting, so the diff is certain
            float[][] coarse = trainAndPredict(f);

            boolean differs = false;
            for (int i = 0; i < fine.length && !differs; i++) {
                for (int c = 0; c < fine[i].length; c++) {
                    if (fine[i][c] != coarse[i][c]) {
                        differs = true;
                        break;
                    }
                }
            }
            assertTrue(differs, "max_bin=4 predicted identically to the default — the setting is being ignored");
        } finally {
            XGBoostModel.setMaxBin(original);
        }
    }

    /** Guards the clamp: XGBoost rejects fewer than 2 bins, and negatives mean "default". */
    @Test
    @DisplayName("max_bin is clamped to a value XGBoost accepts")
    void maxBinIsClamped() {
        int original = XGBoostModel.getMaxBin();
        try {
            XGBoostModel.setMaxBin(-5);
            assertEquals(0, XGBoostModel.getMaxBin(), "negative should mean default");
            XGBoostModel.setMaxBin(0);
            assertEquals(0, XGBoostModel.getMaxBin());
            XGBoostModel.setMaxBin(1);
            assertEquals(2, XGBoostModel.getMaxBin(), "1 bin is not a legal histogram");
            XGBoostModel.setMaxBin(64);
            assertEquals(64, XGBoostModel.getMaxBin());
        } finally {
            XGBoostModel.setMaxBin(original);
        }
    }
}
