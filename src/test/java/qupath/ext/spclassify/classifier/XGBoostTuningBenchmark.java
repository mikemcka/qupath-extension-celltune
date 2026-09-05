package qupath.ext.spclassify.classifier;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * Measures what the remaining XGBoost knobs cost and buy, on the shape that dominates a real run.
 * <p>
 * After the phase-18 work XGBoost is ~87% of a training run (the round search plus the final fit),
 * and a field log confirmed the time is almost entirely {@code Booster.update} — tree construction.
 * A separate probe established that the per-round {@code evalSet} on the validation fold is
 * incremental and costs ~0.4% of the search, so there is <b>no</b> free structural win left there:
 * everything below changes the model, and therefore predictions.
 * <p>
 * This exists so that trade-off is made on measured numbers rather than intuition. Each variant
 * reports wall time <em>and</em> validation accuracy, because a 2x speedup that costs 5 points of
 * accuracy is not a win for this use case.
 * <p>
 * Skipped unless {@code -Dcelltune.bench=true}. Tunables: {@code -Dcelltune.bench.xgb.rounds=N}
 * (default 40 — enough to compare per-round cost without paying for a full 127-round fit),
 * {@code -Dcelltune.bench.features=N} (default 1886, the real post-prune width),
 * {@code -Dcelltune.bench.xgb.classes=N} (default 35), {@code -Dcelltune.bench.xgb.rows=N}
 * (default 8646).
 */
@EnabledIfSystemProperty(named = "celltune.bench", matches = "true")
class XGBoostTuningBenchmark {

    private static int rounds() {
        return Integer.getInteger("celltune.bench.xgb.rounds", 40);
    }

    private static int features() {
        return Integer.getInteger("celltune.bench.features", 1886);
    }

    private static int classes() {
        return Integer.getInteger("celltune.bench.xgb.classes", 35);
    }

    private static int rows() {
        return Integer.getInteger("celltune.bench.xgb.rows", 8646);
    }

    /**
     * Deterministic data shaped like a marker panel: a minority of features carry a weak,
     * overlapping class signal and the rest are noise. Deliberately hard — data a boosted tree can
     * separate in a few rounds stops splitting early and makes every variant look equally fast.
     */
    private static void fill(float[] data, float[] labels, int rows, int nFeatures, int nClasses, int salt) {
        for (int i = 0; i < rows; i++) {
            int cls = i % nClasses;
            labels[i] = cls;
            int off = i * nFeatures;
            for (int f = 0; f < nFeatures; f++) {
                // Cheap deterministic hash, uncorrelated across (i, f).
                long h = ((long) (i + salt) * 2654435761L) ^ ((long) f * 40503L);
                float noise = ((h >>> 8) % 100003) / 100003f;
                // Only every 12th feature says anything about the class, and it says it faintly.
                data[off + f] = (f % 12 == 0) ? noise + (cls % 7) * 0.08f : noise;
            }
        }
    }

    private record Fixture(float[] trainData, float[] trainLabels, float[] valData, float[] valLabels) {}

    private static Fixture fixture(int nRows, int nFeatures, int nClasses) {
        int valRows = Math.max(nClasses * 8, nRows / 5);
        float[] td = new float[nRows * nFeatures];
        float[] tl = new float[nRows];
        float[] vd = new float[valRows * nFeatures];
        float[] vl = new float[valRows];
        fill(td, tl, nRows, nFeatures, nClasses, 0);
        fill(vd, vl, valRows, nFeatures, nClasses, 613);
        return new Fixture(td, tl, vd, vl);
    }

    private static Map<String, Object> baseParams(int nClasses) {
        // Mirrors XGBoostModel.buildParams plus the two the call sites add.
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("max_depth", 6);
        p.put("eta", 0.3);
        p.put("subsample", 0.8);
        p.put("colsample_bytree", 0.8);
        p.put("objective", "multi:softprob");
        p.put("eval_metric", "mlogloss");
        p.put("num_class", nClasses);
        p.put("nthread", Runtime.getRuntime().availableProcessors());
        p.put("seed", 42);
        p.put("device", "cpu");
        p.put("tree_method", "hist");
        p.put("verbosity", 0);
        return p;
    }

    private record Variant(String name, Map<String, Object> overrides) {}

    private record Outcome(String name, long millis, double accuracy, double logloss) {}

    private static Outcome measure(Variant v, Fixture f, int nRows, int nFeatures, int nClasses, int nRounds)
            throws Exception {
        int valRows = f.valLabels().length;
        DMatrix trainMat = new DMatrix(f.trainData(), nRows, nFeatures, Float.NaN);
        DMatrix valMat = new DMatrix(f.valData(), valRows, nFeatures, Float.NaN);
        Booster booster = null;
        try {
            trainMat.setLabel(f.trainLabels());
            valMat.setLabel(f.valLabels());

            Map<String, Object> params = baseParams(nClasses);
            params.putAll(v.overrides());

            long t0 = System.nanoTime();
            booster = XGBoost.train(trainMat, params, nRounds, new LinkedHashMap<>(), null, null);
            long millis = (System.nanoTime() - t0) / 1_000_000L;

            float[][] preds = booster.predict(valMat);
            int correct = 0;
            double loss = 0;
            for (int i = 0; i < valRows; i++) {
                int argmax = 0;
                for (int c = 1; c < preds[i].length; c++) {
                    if (preds[i][c] > preds[i][argmax]) argmax = c;
                }
                int truth = (int) f.valLabels()[i];
                if (argmax == truth) correct++;
                double p = Math.max(preds[i][truth], 1e-15);
                loss -= Math.log(p);
            }
            return new Outcome(v.name(), millis, (double) correct / valRows, loss / valRows);
        } finally {
            if (booster != null) booster.dispose();
            trainMat.dispose();
            valMat.dispose();
        }
    }

    @Test
    @DisplayName("benchmark: what the remaining XGBoost knobs cost and buy")
    void benchmarkKnobs() throws Exception {
        int nRounds = rounds();
        int nFeatures = features();
        int nClasses = classes();
        int nRows = rows();

        System.out.printf(
                "%n=== XGBoost tuning benchmark ===%n"
                        + "  rows         : %,d  (+%,d validation)%n"
                        + "  features     : %,d%n"
                        + "  classes      : %,d   -> %,d trees at %d rounds%n"
                        + "  cores        : %d%n%n"
                        + "  Every variant below changes the model. Read the accuracy column first.%n%n",
                nRows,
                Math.max(nClasses * 8, nRows / 5),
                nFeatures,
                nClasses,
                (long) nClasses * nRounds,
                nRounds,
                Runtime.getRuntime().availableProcessors());

        Fixture f = fixture(nRows, nFeatures, nClasses);

        List<Variant> variants = new ArrayList<>();
        variants.add(new Variant("baseline (max_bin=256)", Map.of()));
        variants.add(new Variant("max_bin=128", Map.of("max_bin", 128)));
        variants.add(new Variant("max_bin=64", Map.of("max_bin", 64)));
        variants.add(new Variant("colsample_bylevel=0.8", Map.of("colsample_bylevel", 0.8)));
        variants.add(new Variant("max_bin=128 + bylevel=0.8", Map.of("max_bin", 128, "colsample_bylevel", 0.8)));
        // XGBoost 2.0+: one vector-leaf tree per round instead of one tree per class per round.
        variants.add(new Variant("multi_strategy=multi_output_tree", Map.of("multi_strategy", "multi_output_tree")));

        Outcome base = null;
        for (Variant v : variants) {
            Outcome o;
            try {
                o = measure(v, f, nRows, nFeatures, nClasses, nRounds);
            } catch (Exception ex) {
                System.out.printf("  %-34s FAILED: %s%n", v.name(), ex.getMessage());
                continue;
            }
            if (base == null) base = o;
            System.out.printf(
                    "  %-34s %7.1f s  %5.2fx   acc %.4f (%+.4f)   logloss %.4f%n",
                    o.name(),
                    o.millis() / 1000.0,
                    base.millis() / (double) o.millis(),
                    o.accuracy(),
                    o.accuracy() - base.accuracy(),
                    o.logloss());
        }
        System.out.println();
    }
}
