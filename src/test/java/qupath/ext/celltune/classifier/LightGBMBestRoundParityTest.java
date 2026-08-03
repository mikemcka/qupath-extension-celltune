package qupath.ext.celltune.classifier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.microsoft.ml.lightgbm.PredictionType;
import io.github.metarank.lightgbm4j.LGBMBooster;
import io.github.metarank.lightgbm4j.LGBMDataset;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Settles the one place phase 18 is not formally bit-identical.
 * <p>
 * {@link LightGBMModel#findBestRounds} now early-stops on LightGBM's own {@code multi_logloss} via
 * {@code addValidData} + {@code getEval}, instead of calling {@code predictForMat} every round and
 * scoring by hand — the change that took the search from O(rounds²) to O(rounds). Same formula,
 * same {@code /n}, so the two can only pick different rounds if consecutive losses tie to roughly
 * 15 significant figures. That is an argument, not evidence.
 * <p>
 * This is the evidence: the pre-change scoring loop is reproduced verbatim below and both are run
 * against the same fixtures. If {@code bestRound} agrees, then a default-settings run of the new
 * build trains exactly the model the old one did, and a before/after diff of predicted classes
 * should be empty rather than merely close.
 * <p>
 * Reproducing the old loop here rather than importing it is deliberate — it is an oracle, and an
 * oracle that gets refactored alongside the code it checks stops being one.
 */
class LightGBMBestRoundParityTest {

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

    /** The pre-change scoring loop, copied from origin/main. Do not refactor. */
    private static double computeLoglossLegacy(double[] preds, float[] labels, int n, int nClasses) {
        double loss = 0;
        if (nClasses == 2 && preds.length == n) {
            for (int i = 0; i < n; i++) {
                int trueClass = (int) labels[i];
                double p = trueClass == 1 ? preds[i] : 1 - preds[i];
                loss += -Math.log(Math.max(p, 1e-15));
            }
        } else {
            for (int i = 0; i < n; i++) {
                int trueClass = (int) labels[i];
                double p = preds[i * nClasses + trueClass];
                loss += -Math.log(Math.max(p, 1e-15));
            }
        }
        return loss / n;
    }

    /** The pre-change {@code findBestRounds}, copied from origin/main. Do not refactor. */
    private static int findBestRoundsLegacy(
            float[] trainData,
            float[] trainLabels,
            int trainSize,
            float[] valData,
            float[] valLabels,
            int valSize,
            int nFeatures,
            int nClasses,
            int maxRounds,
            int maxDepth,
            float learningRate,
            float subsample,
            int patience)
            throws Exception {

        LGBMDataset dataset = LGBMDataset.createFromMat(trainData, trainSize, nFeatures, true, "", null);
        dataset.setField("label", trainLabels);
        String params = LightGBMModel.buildParams(
                nClasses,
                maxDepth,
                learningRate,
                subsample,
                Runtime.getRuntime().availableProcessors());
        LGBMBooster booster = LGBMBooster.create(dataset, params);
        try {
            double bestLoss = Double.MAX_VALUE;
            int bestRound = 0;
            for (int round = 0; round < maxRounds; round++) {
                booster.updateOneIter();
                double[] preds =
                        booster.predictForMat(valData, valSize, nFeatures, true, PredictionType.C_API_PREDICT_NORMAL);
                double loss = computeLoglossLegacy(preds, valLabels, valSize, nClasses);
                if (loss < bestLoss) {
                    bestLoss = loss;
                    bestRound = round;
                }
                if (round - bestRound >= patience) break;
            }
            return bestRound + 1;
        } finally {
            booster.close();
            dataset.close();
        }
    }

    /** Deterministic, deliberately hard — an easy task converges instantly and proves nothing. */
    private static void fill(float[] data, float[] labels, int rows, int nFeatures, int nClasses, int salt) {
        for (int i = 0; i < rows; i++) {
            int cls = i % nClasses;
            labels[i] = cls;
            int off = i * nFeatures;
            for (int f = 0; f < nFeatures; f++) {
                long h = ((long) (i + salt) * 2654435761L) ^ ((long) f * 40503L);
                float noise = ((h >>> 8) % 100003) / 100003f;
                data[off + f] = (f % 5 == 0) ? noise + (cls % 4) * 0.18f : noise;
            }
        }
    }

    private void assertAgrees(String label, int nClasses, int nFeatures, int trainRows, int maxRounds)
            throws Exception {
        int valRows = Math.max(nClasses * 10, trainRows / 4);
        float[] td = new float[trainRows * nFeatures];
        float[] tl = new float[trainRows];
        float[] vd = new float[valRows * nFeatures];
        float[] vl = new float[valRows];
        fill(td, tl, trainRows, nFeatures, nClasses, 0);
        fill(vd, vl, valRows, nFeatures, nClasses, 613);

        int legacy = findBestRoundsLegacy(
                td, tl, trainRows, vd, vl, valRows, nFeatures, nClasses, maxRounds, 6, 0.05f, 0.8f, 20);
        int current = LightGBMModel.findBestRounds(
                td, tl, trainRows, vd, vl, valRows, nFeatures, nClasses, maxRounds, 6, 0.05f, 0.8f, 20, s -> {});

        assertEquals(
                legacy,
                current,
                label + ": native getEval picked a different round than the hand-rolled log-loss."
                        + " A before/after prediction diff will not be empty — investigate before shipping.");
    }

    @Test
    @DisplayName("multiclass: native scoring picks the same round as the old manual loop")
    void multiclassAgrees() throws Exception {
        assumeTrue(nativesAvailable, "LightGBM natives unavailable");
        assertAgrees("multiclass/8", 8, 60, 1200, 120);
    }

    @Test
    @DisplayName("many classes, wide panel: same round")
    void wideMulticlassAgrees() throws Exception {
        assumeTrue(nativesAvailable, "LightGBM natives unavailable");
        assertAgrees("multiclass/35", 35, 220, 2400, 120);
    }

    @Test
    @DisplayName("binary: same round")
    void binaryAgrees() throws Exception {
        assumeTrue(nativesAvailable, "LightGBM natives unavailable");
        assertAgrees("binary", 2, 60, 1200, 120);
    }
}
