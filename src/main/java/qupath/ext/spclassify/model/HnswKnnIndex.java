package qupath.ext.spclassify.model;

import com.github.jelmerk.hnswlib.core.DistanceFunctions;
import com.github.jelmerk.hnswlib.core.Item;
import com.github.jelmerk.hnswlib.core.SearchResult;
import com.github.jelmerk.hnswlib.core.hnsw.HnswIndex;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Approximate-nearest-neighbour (HNSW) kNN graph builder wrapping {@code
 * com.github.jelmerk:hnswlib-core}. This is the ANN replacement for {@link
 * LeidenModel#featureKnn(double[][], int)}'s brute-force O(n&sup2;) scan,
 * used by both single-image and cohort-scale Leiden kNN graph construction
 * (LEI-07). All third-party {@code com.github.jelmerk.*} imports are confined
 * to this file so {@link LeidenModel} stays free of the ANN library surface.
 *
 * <h2>Distance</h2>
 *
 * Euclidean via {@link DistanceFunctions#DOUBLE_EUCLIDEAN_DISTANCE} — true
 * distance (not squared), ordering-equivalent to {@link
 * LeidenModel}'s {@code squaredDistance} for kNN purposes. Do not compare raw
 * distance <em>values</em> between this ANN path and the exact path (only
 * neighbour-set <em>membership</em>, e.g. for a recall gate) since one is
 * squared and the other is not.
 *
 * <h2>Reproducible vs fast build</h2>
 *
 * {@code reproducible=false} uses jelmerk's parallel {@code addAll(...)},
 * which is fast but links nodes in whichever order concurrent threads happen
 * to acquire the index's internal lock — insertion order (and therefore graph
 * topology) is not fixed run to run.
 *
 * <p>{@code reproducible=true} instead adds every row single-threaded, in a
 * fixed row-index order, removing that concurrent-insertion-order source of
 * variance. This is a <b>best-effort</b>, not a mathematically proven,
 * determinism guarantee: jelmerk's {@code HnswIndex.assignLevel(double)}
 * draws its level from {@code ThreadLocalRandom.current()} with no seed hook,
 * and {@code HnswIndex} cannot be subclassed to inject a seeded {@code
 * Random} from outside its own package — verified directly against the 1.2.1
 * bytecode: its only constructor beyond the fully {@code private} one is a
 * compiler-synthetic bridge (used internally by {@code RefinedBuilder.build()})
 * that javac does not resolve from external source, even when placed in
 * jelmerk's own package name. See
 * {@code .planning/phases/15-all-cells-leiden-clustering/15-RESEARCH.md}
 * "Common Pitfalls: Pitfall 1" for the underlying finding. In practice, with
 * the hidden defaults below (a generous {@code efConstruction} and query-time
 * {@code ef} relative to typical graph sizes), single-threaded sequential
 * insertion reliably yields the same returned neighbour sets build-to-build
 * for well-clustered marker data — this is what the reproducible-mode
 * determinism test in {@code HnswKnnIndexTest} validates — but it is not a
 * bit-for-bit guarantee of the internal graph structure itself.
 *
 * <h2>Hidden defaults (D-09)</h2>
 *
 * {@code M}, {@code efConstruction}, and the initial query-time {@code ef}
 * are tunable-but-hidden implementation defaults — no user-facing ANN knobs.
 * Query-time {@code ef} can be escalated on a live, already-built index via
 * {@link #setEf(int)} without rebuilding (D-08's recall-gate escalation
 * path).
 */
public final class HnswKnnIndex {

    /** Graph fanout — jelmerk's own {@code BuilderBase.DEFAULT_M} is 10; 16 trades a little more memory for recall. */
    private static final int DEFAULT_M = 16;

    /** Build-time accuracy/speed knob — higher gives a better-connected graph at slower build time. */
    private static final int DEFAULT_EF_CONSTRUCTION = 200;

    /** Initial query-time accuracy/speed knob — raised via {@link #setEf(int)} by the recall gate (D-08) on demand. */
    private static final int DEFAULT_EF = 64;

    private final HnswIndex<Integer, double[], RowItem, Double> index;
    private final int size;

    private HnswKnnIndex(HnswIndex<Integer, double[], RowItem, Double> index, int size) {
        this.index = index;
        this.size = size;
    }

    // ── One-shot static API ──────────────────────────────────────────────────

    /**
     * For every row, the indices of its {@code k} nearest neighbours (Euclidean),
     * excluding itself, ordered ascending by (distance, index) — the same output
     * contract as {@link LeidenModel#featureKnn(double[][], int)}, just backed by
     * an approximate (not exhaustive) search. {@code out[i].length ==
     * min(k, n-1)}; {@code n==0} or {@code n==1} return empty/neutral results
     * without throwing.
     *
     * @param rows rows to index (e.g. the z-scored active marker matrix)
     * @param k number of neighbours to keep per row
     * @param seed reproducible-build seed (only consulted when {@code reproducible} is true)
     * @param reproducible when true, builds single-threaded in fixed row order (see class javadoc)
     */
    public static int[][] knn(double[][] rows, int k, long seed, boolean reproducible) {
        int n = rows.length;
        int[][] out = new int[n][];
        if (n == 0 || k < 1) {
            fillEmpty(out);
            return out;
        }
        int keep = Math.min(k, n - 1);
        if (keep <= 0) {
            fillEmpty(out);
            return out;
        }
        HnswKnnIndex live = build(rows, seed, reproducible);
        IntStream.range(0, n).parallel().forEach(i -> out[i] = live.queryExcludingSelf(rows[i], i, keep));
        return out;
    }

    private static void fillEmpty(int[][] out) {
        for (int i = 0; i < out.length; i++) {
            out[i] = new int[0];
        }
    }

    // ── Live instance API — build once, query repeatedly at escalating ef ───

    /**
     * Builds a live HNSW index over {@code rows} that can be queried (and have
     * its {@code ef} raised) repeatedly afterwards via {@link #queryRow(double[],
     * int)} / {@link #setEf(int)} without rebuilding — required by the recall
     * gate's ef-escalation path (D-08). {@code n==0} yields a degenerate,
     * always-empty-result index rather than throwing.
     */
    public static HnswKnnIndex build(double[][] rows, long seed, boolean reproducible) {
        int n = rows.length;
        if (n == 0) {
            return new HnswKnnIndex(null, 0);
        }
        int dims = rows[0].length;
        HnswIndex<Integer, double[], RowItem, Double> index = HnswIndex.newBuilder(
                        dims, DistanceFunctions.DOUBLE_EUCLIDEAN_DISTANCE, n)
                .withM(DEFAULT_M)
                .withEfConstruction(DEFAULT_EF_CONSTRUCTION)
                .withEf(DEFAULT_EF)
                .build();

        List<RowItem> items = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            items.add(new RowItem(i, rows[i]));
        }

        if (reproducible) {
            // Single-threaded, fixed row-order sequential add -- see class javadoc
            // "Reproducible vs fast build" for why this is best-effort, not a
            // mathematically proven bit-identical guarantee. `seed` is accepted
            // for interface symmetry / forward compatibility (e.g. a future
            // jelmerk release that adds a seed hook) but jelmerk 1.2.1 has no
            // seed hook to feed it into today.
            for (RowItem item : items) {
                index.add(item);
            }
        } else {
            try {
                index.addAll(
                        items,
                        Math.max(1, Runtime.getRuntime().availableProcessors()),
                        com.github.jelmerk.hnswlib.core.NullProgressListener.INSTANCE,
                        com.github.jelmerk.hnswlib.core.Index.DEFAULT_PROGRESS_UPDATE_INTERVAL);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("HNSW parallel build interrupted", e);
            }
        }
        return new HnswKnnIndex(index, n);
    }

    /** Escalate (or lower) query-time recall on the already-built index — no rebuild (D-08). */
    public void setEf(int ef) {
        if (index != null) {
            index.setEf(ef);
        }
    }

    /**
     * Nearest {@code k} row indices to {@code vector}, ascending by (distance,
     * index), with NO self-exclusion (the caller decides — this is the
     * general-purpose query primitive the recall gate uses against arbitrary
     * query vectors, which may not themselves be rows in the index).
     */
    public int[] queryRow(double[] vector, int k) {
        if (index == null || size == 0) {
            return new int[0];
        }
        int keep = Math.max(0, Math.min(k, size));
        if (keep == 0) {
            return new int[0];
        }
        return nearestIds(vector, keep, -1);
    }

    /** Nearest {@code keep} row indices to {@code vector}, excluding row {@code selfIdx} — used by {@link #knn}. */
    private int[] queryExcludingSelf(double[] vector, int selfIdx, int keep) {
        return nearestIds(vector, keep, selfIdx);
    }

    private int[] nearestIds(double[] vector, int keep, int excludeId) {
        int fetch = excludeId >= 0 ? keep + 1 : keep;
        List<SearchResult<RowItem, Double>> results = index.findNearest(vector, fetch);
        double[] dist = new double[results.size()];
        int[] ids = new int[results.size()];
        int count = 0;
        for (SearchResult<RowItem, Double> r : results) {
            int id = r.item().id();
            if (id == excludeId) {
                continue;
            }
            dist[count] = r.distance();
            ids[count] = id;
            count++;
        }
        // Deterministic tie-break: ascending (distance, id) -- matches
        // LeidenModel.featureKnn's ordering contract, independent of whatever
        // (possibly nondeterministic) order findNearest visited candidates in.
        // Sorted with a primitive-array insertion sort (not Arrays.sort on a boxed
        // Integer[] + comparator lambda) -- this runs on every query across tens of
        // millions of per-row queries, so avoiding the boxed array/autoboxing churn
        // matters; `count` is always k-sized (small), so insertion sort's O(count^2)
        // is cheap and the ordering it produces is byte-identical to the prior sort.
        int[] order = new int[count];
        for (int i = 0; i < count; i++) {
            order[i] = i;
        }
        for (int i = 1; i < count; i++) {
            int key = order[i];
            double keyDist = dist[key];
            int keyId = ids[key];
            int j = i - 1;
            while (j >= 0 && isAfter(dist[order[j]], ids[order[j]], keyDist, keyId)) {
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = key;
        }
        int outLen = Math.min(keep, count);
        int[] out = new int[outLen];
        for (int i = 0; i < outLen; i++) {
            out[i] = ids[order[i]];
        }
        return out;
    }

    /** True when (distA, idA) sorts strictly after (distB, idB) under the ascending (distance, id) tie-break. */
    private static boolean isAfter(double distA, int idA, double distB, int idB) {
        int c = Double.compare(distA, distB);
        return c != 0 ? c > 0 : idA > idB;
    }

    /** {@code Item<Integer, double[]>} view of a pooled row: id = row index, vector = the row itself. */
    private record RowItem(Integer id, double[] vector) implements Item<Integer, double[]> {
        @Override
        public int dimensions() {
            return vector.length;
        }
    }
}
