package qupath.ext.celltune.classifier;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.microsoft.ml.lightgbm.PredictionType;
import io.github.metarank.lightgbm4j.LGBMBooster;
import io.github.metarank.lightgbm4j.LGBMDataset;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Oracle for the model that actually gets deployed as model 2.
 * <p>
 * {@link LightGBMModel#train} used to attempt a GPU booster first — building a full
 * {@code numRounds} loop under {@code device_type=gpu}, failing, closing that booster, and
 * rebuilding on CPU from round zero against the <em>same</em> {@code LGBMDataset}. That probe is
 * gone, because this build ships no GPU kernels so it could never succeed. The argument that
 * removing it is safe rests on the discarded GPU booster having left no trace on the shared
 * dataset — plausible, but bin construction happens against that dataset, so it is worth
 * verifying rather than asserting.
 * <p>
 * `ResamplerGoldenTest` pins the training data and `LightGBMBestRoundParityTest` pins the round
 * count; this pins the last link, the fit itself. Together they mean a default-settings run
 * produces the same model 2 as the pre-change build without anyone having to diff a cell table by
 * hand.
 * <p>
 * The legacy path is reproduced verbatim from origin/main. Do not refactor it — an oracle that
 * changes with the code it checks stops being one.
 */
class LightGBMTrainParityTest {

    private static final int N_CLASSES = 6;
    private static final int N_FEATURES = 40;
    private static final int TRAIN_ROWS = 900;
    private static final int TEST_ROWS = 200;
    private static final int ROUNDS = 45;

    private static boolean nativesAvailable = false;

    @BeforeAll
    static void checkNatives() {
        try {
            LGBMBooster.loadNative();
            nativesAvailable = LGBMBooster.isNativeLoaded();
        } catch (Throwable t) {
            nativesAvailable = false;
        }
    }

    private static void fill(float[] data, float[] labels, int rows, int salt) {
        for (int i = 0; i < rows; i++) {
            int cls = i % N_CLASSES;
            labels[i] = cls;
            int off = i * N_FEATURES;
            for (int f = 0; f < N_FEATURES; f++) {
                long h = ((long) (i + salt) * 2654435761L) ^ ((long) f * 40503L);
                float noise = ((h >>> 8) % 100003) / 100003f;
                data[off + f] = (f % 5 == 0) ? noise + (cls % 4) * 0.2f : noise;
            }
        }
    }

    private static List<String> names(String prefix, int n) {
        List<String> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) out.add(prefix + i);
        return out;
    }

    /**
     * The pre-change {@code train} body: probe GPU with a full round loop, fall back to CPU on the
     * same dataset. Copied from origin/main. Do not refactor.
     */
    private static double[] legacyTrainAndPredict(
            float[] trainData, float[] trainLabels, float[] testData, int numRounds) throws Exception {
        LGBMDataset dataset = LGBMDataset.createFromMat(trainData, TRAIN_ROWS, N_FEATURES, true, "", null);
        dataset.setField("label", trainLabels);
        String params = LightGBMModel.buildParams(
                N_CLASSES, 6, 0.05f, 0.8f, Runtime.getRuntime().availableProcessors());

        LGBMBooster booster = null;
        try {
            String gpuParams = params + " device_type=gpu";
            booster = LGBMBooster.create(dataset, gpuParams);
            for (int i = 0; i < numRounds; i++) {
                booster.updateOneIter();
            }
        } catch (Exception gpuEx) {
            if (booster != null) {
                try {
                    booster.close();
                } catch (Exception ignore) {
                    // matches the original's swallow
                }
            }
            booster = LGBMBooster.create(dataset, params);
            for (int i = 0; i < numRounds; i++) {
                booster.updateOneIter();
            }
        }
        dataset.close();
        try {
            return booster.predictForMat(testData, TEST_ROWS, N_FEATURES, true, PredictionType.C_API_PREDICT_NORMAL);
        } finally {
            booster.close();
        }
    }

    @Test
    @DisplayName("dropping the dead GPU probe leaves the fitted model identical")
    void currentTrainMatchesTheLegacyProbeAndFallback() throws Exception {
        assumeTrue(nativesAvailable, "LightGBM natives unavailable");

        float[] trainData = new float[TRAIN_ROWS * N_FEATURES];
        float[] trainLabels = new float[TRAIN_ROWS];
        float[] testData = new float[TEST_ROWS * N_FEATURES];
        float[] testLabels = new float[TEST_ROWS];
        fill(trainData, trainLabels, TRAIN_ROWS, 0);
        fill(testData, testLabels, TEST_ROWS, 613);

        double[] legacy = legacyTrainAndPredict(trainData, trainLabels, testData, ROUNDS);

        LightGBMModel current = new LightGBMModel();
        current.train(
                trainData,
                trainLabels,
                TRAIN_ROWS,
                N_FEATURES,
                names("class", N_CLASSES),
                names("f", N_FEATURES),
                ROUNDS,
                6,
                0.05f,
                0.8f,
                LightGBMModel.DEFAULT_COLSAMPLE);
        float[][] currentProba = current.predictProba(testData, TEST_ROWS, N_FEATURES);
        current.close();

        assertEquals(TEST_ROWS, currentProba.length, "row count");
        assertEquals(
                (long) TEST_ROWS * N_CLASSES,
                legacy.length,
                "legacy prediction layout changed — the oracle no longer matches what it is checking");

        for (int i = 0; i < TEST_ROWS; i++) {
            float[] expected = new float[N_CLASSES];
            for (int c = 0; c < N_CLASSES; c++) {
                expected[c] = (float) legacy[i * N_CLASSES + c];
            }
            assertArrayEquals(
                    expected,
                    currentProba[i],
                    0f,
                    "row " + i + ": the CPU-only fit differs from the old probe-then-fall-back fit."
                            + " The discarded GPU booster was affecting the shared dataset.");
        }
    }

    /**
     * The bug this file exists to keep out: {@code train} must build every round it was asked for.
     * Breaking on {@code updateOneIter()}'s {@code is_finished} truncated the model, because that
     * flag marks a barren iteration rather than convergence.
     */
    @Test
    @DisplayName("train builds every requested round, even when iterations go barren")
    void trainDoesNotStopEarly() throws Exception {
        assumeTrue(nativesAvailable, "LightGBM natives unavailable");

        float[] trainData = new float[TRAIN_ROWS * N_FEATURES];
        float[] trainLabels = new float[TRAIN_ROWS];
        float[] testData = new float[TEST_ROWS * N_FEATURES];
        float[] testLabels = new float[TEST_ROWS];
        fill(trainData, trainLabels, TRAIN_ROWS, 0);
        fill(testData, testLabels, TEST_ROWS, 613);

        LightGBMModel few = new LightGBMModel();
        few.train(
                trainData,
                trainLabels,
                TRAIN_ROWS,
                N_FEATURES,
                names("class", N_CLASSES),
                names("f", N_FEATURES),
                10,
                6,
                0.05f,
                0.8f,
                LightGBMModel.DEFAULT_COLSAMPLE);
        float[][] fewProba = few.predictProba(testData, TEST_ROWS, N_FEATURES);
        few.close();

        LightGBMModel many = new LightGBMModel();
        many.train(
                trainData,
                trainLabels,
                TRAIN_ROWS,
                N_FEATURES,
                names("class", N_CLASSES),
                names("f", N_FEATURES),
                200,
                6,
                0.05f,
                0.8f,
                LightGBMModel.DEFAULT_COLSAMPLE);
        float[][] manyProba = many.predictProba(testData, TEST_ROWS, N_FEATURES);
        many.close();

        boolean differs = false;
        for (int i = 0; i < TEST_ROWS && !differs; i++) {
            for (int c = 0; c < N_CLASSES; c++) {
                if (fewProba[i][c] != manyProba[i][c]) {
                    differs = true;
                    break;
                }
            }
        }
        org.junit.jupiter.api.Assertions.assertTrue(
                differs,
                "200 rounds predicted identically to 10 — training stopped early, which is what"
                        + " breaking on is_finished used to cause");
    }
}
