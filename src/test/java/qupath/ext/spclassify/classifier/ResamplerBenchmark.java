package qupath.ext.spclassify.classifier;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * Wall-clock benchmark for {@link Resampler}, shaped like a real CellTune training set.
 * <p>
 * Skipped unless {@code -Dcelltune.bench=true} — the full-size case takes minutes on the
 * brute-force implementation, which is the whole point.
 * <p>
 * The default class distribution is the real pooled label distribution from a 14-image COMET
 * project (5,839 labelled cells across 19 effective classes, majority 1,348, minority 132).
 * SMOTE therefore has to synthesise 19,773 rows, taking the set to 25,612 before Tomek — and
 * Tomek is O(n²·d) on that inflated count, three times per Train click.
 * <p>
 * Tunables:
 * <ul>
 *   <li>{@code -Dcelltune.bench.features=N} — feature count (default 200, a realistic
 *       post-{@code FeaturePruner} width; the raw panel is 1,780)</li>
 *   <li>{@code -Dcelltune.bench.scale=F} — scales every class count (default 1.0); use e.g.
 *       0.25 for a quick sanity run</li>
 *   <li>{@code -Dcelltune.bench.strategies=SMOTE,TOMEK} — restrict which strategies run</li>
 * </ul>
 */
@EnabledIfSystemProperty(named = "celltune.bench", matches = "true")
class ResamplerBenchmark {

    /**
     * Real pooled per-class label counts, descending. Derived from the project's
     * {@code celltune/image-labels/*.json} with {@code X-mergedInto(Y)} collapsed to {@code Y},
     * which is what the classifier actually trains on.
     */
    private static final int[] REAL_CLASS_COUNTS = {
        1348, 1167, 284, 264, 225, 216, 215, 211, 199, 197, 189, 180, 179, 171, 170, 169, 164, 159, 132
    };

    private static int features() {
        return Integer.getInteger("celltune.bench.features", 200);
    }

    private static double scale() {
        return Double.parseDouble(System.getProperty("celltune.bench.scale", "1.0"));
    }

    private static List<ResamplingStrategy> strategies() {
        String prop = System.getProperty("celltune.bench.strategies");
        if (prop == null || prop.isBlank()) {
            return List.of(
                    ResamplingStrategy.SMOTE,
                    ResamplingStrategy.TOMEK,
                    ResamplingStrategy.SMOTE_TOMEK,
                    ResamplingStrategy.ADASYN,
                    ResamplingStrategy.ADASYN_TOMEK);
        }
        List<ResamplingStrategy> out = new ArrayList<>();
        for (String s : prop.split(",")) out.add(ResamplingStrategy.valueOf(s.trim()));
        return out;
    }

    /**
     * Builds a dataset with the given per-class counts and a deterministic, class-overlapping
     * feature layout. Values are spread over a few well-separated informative dimensions with
     * the remainder acting as correlated noise — enough structure that kNN and Tomek do real
     * work, without depending on any RNG.
     */
    private static void buildDataset(int[] counts, int nFeatures, List<float[]> rows, List<Integer> labels) {
        int i = 0;
        for (int cls = 0; cls < counts.length; cls++) {
            for (int m = 0; m < counts[cls]; m++, i++) {
                float[] row = new float[nFeatures];
                for (int f = 0; f < nFeatures; f++) {
                    float base = ((i * 31L + f * 7L) % 9973) / 9973f;
                    // First 8 dims carry a weak class signal; the rest are class-independent
                    // noise. Overlap is deliberate so Tomek links actually form.
                    row[f] = f < 8 ? base + cls * 0.12f : base;
                }
                rows.add(row);
                labels.add(cls);
            }
        }
        // Deterministic interleave so classIndices are non-contiguous, as in production.
        int n = rows.size();
        int stride = 7919;
        List<float[]> pr = new ArrayList<>(n);
        List<Integer> pl = new ArrayList<>(n);
        for (int k = 0; k < n; k++) {
            int src = (int) (((long) k * stride) % n);
            pr.add(rows.get(src));
            pl.add(labels.get(src));
        }
        rows.clear();
        rows.addAll(pr);
        labels.clear();
        labels.addAll(pl);
    }

    @Test
    @DisplayName("benchmark: resampling a real-shaped labelled set")
    void benchmarkRealShape() {
        int nFeatures = features();
        double scale = scale();

        int[] counts = new int[REAL_CLASS_COUNTS.length];
        for (int c = 0; c < counts.length; c++) {
            counts[c] = Math.max(2, (int) Math.round(REAL_CLASS_COUNTS[c] * scale));
        }

        List<float[]> rows = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        buildDataset(counts, nFeatures, rows, labels);

        int n = rows.size();
        int maxCount = 0;
        for (int c : counts) maxCount = Math.max(maxCount, c);
        int postSmote = counts.length * maxCount;

        System.out.printf(
                "%n=== Resampler benchmark ===%n"
                        + "  cells        : %,d  (%d classes, majority %,d, minority %,d)%n"
                        + "  features     : %,d%n"
                        + "  post-SMOTE n : %,d   (Tomek cost scales with this, squared)%n"
                        + "  cores        : %d%n%n",
                n,
                counts.length,
                maxCount,
                counts[counts.length - 1],
                nFeatures,
                postSmote,
                Runtime.getRuntime().availableProcessors());

        for (ResamplingStrategy strategy : strategies()) {
            List<float[]> r = new ArrayList<>(rows);
            List<Integer> l = new ArrayList<>(labels);

            long t0 = System.nanoTime();
            Resampler.Result result = Resampler.apply(r, l, counts.length, strategy, null);
            long elapsedMs = (System.nanoTime() - t0) / 1_000_000L;

            System.out.printf(
                    "  %-14s %8.2f s   -> %,7d rows%n",
                    strategy.name(), elapsedMs / 1000.0, result.rows().size());
        }
        System.out.println();
    }
}
