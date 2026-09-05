package qupath.ext.spclassify.model;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.stream.IntStream;
import nl.cwts.networkanalysis.Clustering;
import nl.cwts.networkanalysis.LeidenAlgorithm;
import nl.cwts.networkanalysis.Network;
import nl.cwts.util.LargeDoubleArray;
import nl.cwts.util.LargeIntArray;

/**
 * Pure numerical core of graph-based (Leiden) phenotype clustering: feature-space
 * kNN graph construction, Jaccard/shared-nearest-neighbour edge weighting, CWTS
 * Leiden community detection, and kNN label transfer for cohort assignment
 * (scanpy {@code sc.tl.ingest} style). Every method is static and a pure function
 * of primitive arrays (no JavaFX, no QuPath types), so the graph/clustering/
 * transfer math is unit-testable against synthetic clouds — mirroring
 * {@code NeighborhoodModel} and {@code ScatterMath}.
 *
 * <p>The CWTS {@code LeidenAlgorithm} optimises the Constant Potts Model (CPM)
 * quality function (not Modularity — this library does not expose a Modularity
 * variant of Leiden; {@code LouvainAlgorithm} is the separate, also-CPM,
 * alternative). CPM {@code resolution} still behaves monotonically like the
 * scanpy/leidenalg resolution knob: larger resolution biases toward more, smaller
 * communities.
 */
public final class LeidenModel {

    private LeidenModel() {}

    /**
     * Community labels 0..nClusters-1 for every input row, and the cluster count.
     *
     * @param recall measured ANN recall (D-09) for the run that produced {@code labels} — 1.0
     *               for {@link #cluster}'s exact (brute-force featureKnn) path, since there is
     *               no approximation to measure; the recall-gate's measured passing recall for
     *               {@link #clusterViaAnn}'s ANN-routed path (also 1.0 for its degenerate
     *               n==0/n==1 inputs, which never reach the ANN gate)
     */
    public record LeidenResult(int[] labels, int nClusters, double recall) {}

    // ── Feature-space kNN ────────────────────────────────────────────────────

    /**
     * For every row, the indices of its {@code k} nearest neighbours (Euclidean
     * over all columns), excluding itself. {@code out[i].length <= k} (fewer when
     * {@code n-1 < k}). Still brute-force O(n²) in the distance scan — acceptable
     * for the bounded sampled pool (plan ceiling: ~50k rows) and revisit with an ANN
     * index only if profiling demands it (plan's deferred item) — but each row's
     * search is (a) run in parallel across rows (they are independent, each writing
     * only its own {@code out[i]}) and (b) selected with a bounded size-{@code k}
     * max-heap instead of a full sort, so per row it is O(n log k) with no boxed
     * {@code Integer} allocation rather than the old O(n log n) sort of an
     * {@code Integer[n]}. Output is byte-identical to the previous stable full-sort:
     * the {@code k} smallest by (distance asc, index asc).
     */
    public static int[][] featureKnn(double[][] rows, int k) {
        int n = rows.length;
        int[][] out = new int[n][];
        if (n == 0 || k < 1) {
            for (int i = 0; i < n; i++) {
                out[i] = new int[0];
            }
            return out;
        }
        int keep = Math.min(k, n - 1);
        if (keep <= 0) {
            for (int i = 0; i < n; i++) {
                out[i] = new int[0];
            }
            return out;
        }
        IntStream.range(0, n).parallel().forEach(i -> out[i] = nearestForRow(rows, i, keep));
        return out;
    }

    /**
     * The {@code keep} nearest rows to row {@code i} (self excluded), ordered
     * ascending by (squared distance, index). Maintains a bounded max-heap of the
     * best candidates keyed by "worseness" (distance desc, index desc) so the root
     * is always the current worst kept candidate and a new candidate is admitted
     * only when strictly better — an O(n log keep) selection over primitive arrays.
     */
    private static int[] nearestForRow(double[][] rows, int i, int keep) {
        int n = rows.length;
        double[] ri = rows[i];
        double[] heapDist = new double[keep];
        int[] heapIdx = new int[keep];
        int size = 0;
        for (int j = 0; j < n; j++) {
            if (j == i) {
                continue;
            }
            double d = squaredDistance(ri, rows[j]);
            if (size < keep) {
                heapDist[size] = d;
                heapIdx[size] = j;
                siftUp(heapDist, heapIdx, size);
                size++;
            } else if (worse(heapDist[0], heapIdx[0], d, j)) {
                // The current worst kept candidate (root) is worse than this one — evict it.
                heapDist[0] = d;
                heapIdx[0] = j;
                siftDown(heapDist, heapIdx, size);
            }
        }
        // Order the kept neighbours ascending by (distance, index) to match the old
        // stable full-sort output exactly. keep is small (graph k, ~15), so an
        // insertion sort over the heap contents is cheaper than heap extraction.
        for (int a = 1; a < size; a++) {
            double dv = heapDist[a];
            int iv = heapIdx[a];
            int b = a - 1;
            while (b >= 0 && worse(heapDist[b], heapIdx[b], dv, iv)) {
                heapDist[b + 1] = heapDist[b];
                heapIdx[b + 1] = heapIdx[b];
                b--;
            }
            heapDist[b + 1] = dv;
            heapIdx[b + 1] = iv;
        }
        int[] result = new int[size];
        System.arraycopy(heapIdx, 0, result, 0, size);
        return result;
    }

    /**
     * True when candidate {@code a} is "worse" than {@code b} — larger distance,
     * ties broken by larger index. Defines the max-heap ordering (root = worst kept)
     * and reproduces the stable full-sort's (distance asc, index asc) tie-break.
     */
    private static boolean worse(double aDist, int aIdx, double bDist, int bIdx) {
        return aDist > bDist || (aDist == bDist && aIdx > bIdx);
    }

    /** Max-heap sift-up: bubble the element at {@code c} toward the root while it is worse than its parent. */
    private static void siftUp(double[] hd, int[] hi, int c) {
        while (c > 0) {
            int parent = (c - 1) >>> 1;
            if (worse(hd[c], hi[c], hd[parent], hi[parent])) {
                swap(hd, hi, c, parent);
                c = parent;
            } else {
                break;
            }
        }
    }

    /** Max-heap sift-down from the root over the first {@code size} elements. */
    private static void siftDown(double[] hd, int[] hi, int size) {
        int c = 0;
        while (true) {
            int left = 2 * c + 1;
            int right = 2 * c + 2;
            int worst = c;
            if (left < size && worse(hd[left], hi[left], hd[worst], hi[worst])) {
                worst = left;
            }
            if (right < size && worse(hd[right], hi[right], hd[worst], hi[worst])) {
                worst = right;
            }
            if (worst == c) {
                break;
            }
            swap(hd, hi, c, worst);
            c = worst;
        }
    }

    private static void swap(double[] hd, int[] hi, int a, int b) {
        double td = hd[a];
        hd[a] = hd[b];
        hd[b] = td;
        int ti = hi[a];
        hi[a] = hi[b];
        hi[b] = ti;
    }

    private static double squaredDistance(double[] a, double[] b) {
        double sum = 0;
        int len = Math.min(a.length, b.length);
        for (int c = 0; c < len; c++) {
            double diff = a[c] - b[c];
            if (Double.isNaN(diff)) {
                continue;
            }
            sum += diff * diff;
        }
        return sum;
    }

    // ── ANN recall gate (D-07/D-08/D-09) ─────────────────────────────────────

    /** Recall threshold below which an ANN-routed run aborts (D-08) — hidden default, no user-facing knob. */
    private static final double RECALL_THRESHOLD = 0.95;

    /** Proportional recall-gate sample fraction (D-07) — hidden default. */
    private static final double RECALL_SAMPLE_FRACTION = 0.001;

    /** Cap on the recall-gate sample size (D-07) — keeps the exact reference O(sample x n), never O(n^2). */
    private static final int RECALL_SAMPLE_CAP = 10_000;

    /** Maximum number of geometric ef-doubling escalations before aborting (D-08) — hidden default. */
    private static final int MAX_ESCALATIONS = 4;

    /** Initial query-time ef the recall gate starts from before any escalation (D-09) — hidden default. */
    private static final int INITIAL_QUERY_EF = 64;

    /**
     * Recall-gate sample size for a pooled set of {@code n} rows: proportional
     * (~0.1%) and capped at {@link #RECALL_SAMPLE_CAP} (D-07) — bounds the
     * exact reference computation to O(sample x n), never the O(n^2) of a full
     * {@link #featureKnn} scan (Pitfall 3).
     */
    static int recallSampleSize(int n) {
        return Math.min(RECALL_SAMPLE_CAP, Math.max(1, (int) Math.round(n * RECALL_SAMPLE_FRACTION)));
    }

    /**
     * Exact nearest-neighbour indices (self excluded) for a small set of query
     * rows that live IN {@code rows} itself (identified by {@code sampleIdx}),
     * against the full {@code rows} array as reference — O(sampleIdx.length x n),
     * reusing the same bounded max-heap {@link #nearestForRow} selection as
     * {@link #featureKnn}, just invoked only for the sampled indices instead of
     * every row. This is the recall gate's exact reference (D-07) — never call
     * {@link #featureKnn} here (that is all-pairs O(n^2), Pitfall 3).
     */
    static int[][] exactNeighborsForQueries(double[][] rows, int[] sampleIdx, int keep) {
        int[][] out = new int[sampleIdx.length][];
        int n = rows.length;
        if (n == 0 || keep <= 0) {
            for (int s = 0; s < sampleIdx.length; s++) {
                out[s] = new int[0];
            }
            return out;
        }
        IntStream.range(0, sampleIdx.length).parallel().forEach(s -> out[s] = nearestForRow(rows, sampleIdx[s], keep));
        return out;
    }

    /**
     * Mean recall of {@code approx} against {@code exact}: for each row,
     * |exact intersect approx| / |exact|, averaged over all rows. 1.0 when
     * every approximate neighbour set matches its exact counterpart exactly as
     * a set (order-independent); 0.0 when they are always disjoint.
     */
    static double meanRecall(int[][] exact, int[][] approx) {
        int n = exact.length;
        if (n == 0) {
            return 1.0;
        }
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += rowRecall(exact[i], approx[i]);
        }
        return sum / n;
    }

    private static double rowRecall(int[] exact, int[] approx) {
        if (exact.length == 0) {
            return 1.0;
        }
        Set<Integer> approxSet = new HashSet<>(approx.length * 2 + 1);
        for (int v : approx) {
            approxSet.add(v);
        }
        int hit = 0;
        for (int v : exact) {
            if (approxSet.contains(v)) {
                hit++;
            }
        }
        return hit / (double) exact.length;
    }

    /**
     * The recall gate itself (D-07/D-08/D-09): samples {@link #recallSampleSize}
     * query rows (seeded from {@code seed}, via {@link CohortClusterModel#sampleIndices}),
     * computes their exact neighbours ({@link #exactNeighborsForQueries}), then
     * repeatedly calls {@code annQueryAtEf} — which, given an ef and the FIXED
     * sample indices, must return each sampled row's approximate neighbour
     * indices at that ef (query-time only, no index rebuild — D-08) — starting
     * at {@code initialEf} and doubling geometrically up to
     * {@link #MAX_ESCALATIONS} times whenever mean recall is below
     * {@link #RECALL_THRESHOLD}. Returns the measured (passing) recall so the
     * caller can surface it (e.g. to a status line, D-09). Throws
     * {@link AnnRecallException} — writing no labels — if recall is still
     * below threshold after the escalation cap; the exception message includes
     * the final measured recall.
     */
    static double gateAnnRecall(
            double[][] rows, int k, long seed, int initialEf, BiFunction<Integer, int[], int[][]> annQueryAtEf) {
        int n = rows.length;
        if (n <= 1) {
            return 1.0;
        }
        int keep = Math.min(k, n - 1);
        int sampleSize = Math.min(recallSampleSize(n), n);
        int[] sampleIdx = CohortClusterModel.sampleIndices(n, sampleSize, new Random(seed));
        int[][] exact = exactNeighborsForQueries(rows, sampleIdx, keep);

        int ef = initialEf;
        int escalations = 0;
        double recall = meanRecall(exact, annQueryAtEf.apply(ef, sampleIdx));
        while (recall < RECALL_THRESHOLD && escalations < MAX_ESCALATIONS) {
            ef *= 2;
            escalations++;
            recall = meanRecall(exact, annQueryAtEf.apply(ef, sampleIdx));
        }
        if (recall < RECALL_THRESHOLD) {
            throw new AnnRecallException("ANN recall " + recall + " remained below the required "
                    + RECALL_THRESHOLD + " threshold after " + escalations + " ef escalation(s) (final ef=" + ef
                    + ")");
        }
        return recall;
    }

    // ── Leiden community detection ───────────────────────────────────────────

    /** Default randomness parameter (mirrors {@link LeidenAlgorithm#DEFAULT_RANDOMNESS}). */
    private static final double DEFAULT_RANDOMNESS = LeidenAlgorithm.DEFAULT_RANDOMNESS;

    /** Default local-moving iterations per Leiden run. */
    private static final int DEFAULT_ITERATIONS = 10;

    /**
     * Build a symmetric kNN graph on {@code rows}, weight edges by Jaccard
     * similarity of each pair's (closed, self-included) neighbour sets, run the
     * CWTS Leiden algorithm at {@code resolution} keeping the best of
     * {@code randomStarts} restarts (by CPM quality), and return per-row
     * community labels relabelled to a dense {@code 0..nClusters-1} range.
     *
     * @param rows rows to cluster (e.g. the z-scored active marker matrix)
     * @param k graph kNN neighbours (default suggestion ~15, scanpy-style)
     * @param resolution CPM resolution; higher -> more, smaller communities
     * @param randomStarts number of independent Leiden restarts to keep the best
     *     of (>= 1; values < 1 are treated as 1)
     * @param seed RNG seed for reproducibility
     */
    public static LeidenResult cluster(double[][] rows, int k, double resolution, int randomStarts, long seed) {
        int n = rows.length;
        if (n == 0) {
            return new LeidenResult(new int[0], 0, 1.0);
        }
        if (n == 1) {
            return new LeidenResult(new int[] {0}, 1, 1.0);
        }

        int[][] neighbors = featureKnn(rows, k);
        // Exact brute-force neighbours -- nothing approximate to measure, so recall is 1.0.
        return clusterFromNeighbors(neighbors, resolution, randomStarts, seed, 1.0);
    }

    /**
     * Shared driver for {@link #cluster} and {@link #clusterViaAnn}: builds the Jaccard-weighted
     * kNN {@link Network} from an already-computed neighbour list, normalizes it (association-
     * strength correction — see below), runs the CWTS Leiden algorithm for {@code randomStarts}
     * restarts keeping the best by CPM quality, and relabels to a dense {@code 0..nClusters-1}
     * range via {@code removeEmptyClusters}. {@link #cluster} (exact brute-force
     * {@link #featureKnn} neighbours) and {@link #clusterViaAnn} (ANN neighbours, recall-gated)
     * differ ONLY in how {@code neighbors} was produced — this method is neighbour-source
     * agnostic, so extracting it keeps both callers byte-identical to their pre-extraction
     * behaviour.
     *
     * <p>Association-strength normalization ({@code
     * Network.createNormalizedNetworkUsingAssociationStrength}) divides each edge weight by its
     * expected weight under a random-configuration null model, exactly like modularity's
     * null-model correction. Without it, CPM's resolution has no natural "1.0" scale against raw
     * Jaccard weights (<=1) and even resolution=1.0 shatters every node into its own singleton
     * community — empirically confirmed while developing this method. With normalization,
     * resolution in the plan's UI range (0.1-3.0, default 1.0) behaves like the familiar
     * scanpy/leidenalg resolution knob.
     *
     * @param neighbors symmetric kNN neighbour list, one row per point ({@code n == neighbors.length})
     * @param recall    the measured (or exact-path constant 1.0) recall to carry through into the
     *                  returned {@link LeidenResult} — this method does not itself measure
     *                  anything ANN-related, it just threads the caller's already-known value
     */
    private static LeidenResult clusterFromNeighbors(
            int[][] neighbors, double resolution, int randomStarts, long seed, double recall) {
        int n = neighbors.length;
        Network rawNetwork = buildJaccardWeightedNetwork(n, neighbors);
        Network network = rawNetwork.createNormalizedNetworkUsingAssociationStrength();

        int starts = Math.max(1, randomStarts);
        Random random = new Random(seed);
        LeidenAlgorithm leiden = new LeidenAlgorithm(resolution, DEFAULT_ITERATIONS, DEFAULT_RANDOMNESS, random);

        Clustering best = null;
        double bestQuality = Double.NEGATIVE_INFINITY;
        for (int r = 0; r < starts; r++) {
            Clustering candidate = leiden.findClustering(network);
            double quality = leiden.calcQuality(network, candidate);
            if (best == null || quality > bestQuality) {
                best = candidate;
                bestQuality = quality;
            }
        }
        best.removeEmptyClusters();
        int[] labels = best.getClusters();
        return new LeidenResult(labels, best.getNClusters(), recall);
    }

    /**
     * Same contract as {@link #cluster}, but builds the kNN graph via the
     * approximate-NN {@link HnswKnnIndex} (LEI-07) instead of the brute-force
     * {@link #featureKnn} scan, guarded by the runtime recall gate
     * (D-07/D-08): recall is measured on a capped subsample against the exact
     * bounded-heap reference, escalating query-time {@code ef} geometrically
     * on failure, and the run aborts via {@link AnnRecallException} (writing
     * NO labels) if recall stays below 0.95 after the escalation cap. On
     * pass, the ANN-sourced neighbour list feeds the SAME unchanged
     * Jaccard/SNN weighting + association-strength normalization + CWTS
     * Leiden pipeline as {@link #cluster} -- only the neighbour source
     * differs.
     *
     * @param reproducible when true, builds the underlying HNSW index
     *     single-threaded in fixed row order (best-effort, not bit-proven,
     *     determinism -- see {@link HnswKnnIndex} class javadoc)
     */
    public static LeidenResult clusterViaAnn(
            double[][] rows, int k, double resolution, int randomStarts, long seed, boolean reproducible) {
        int n = rows.length;
        if (n == 0) {
            return new LeidenResult(new int[0], 0, 1.0);
        }
        if (n == 1) {
            return new LeidenResult(new int[] {0}, 1, 1.0);
        }

        AnnNeighborsResult annResult = annNeighborsWithGate(rows, k, seed, reproducible);
        return clusterFromNeighbors(annResult.neighbors(), resolution, randomStarts, seed, annResult.recall());
    }

    /**
     * {@link #annNeighborsWithGate}'s return value: the full kNN neighbour list AND the
     * measured (passing) recall the gate settled on (D-09) — surfaced so {@link #clusterViaAnn}
     * can thread it into its {@link LeidenResult} instead of discarding it.
     */
    record AnnNeighborsResult(int[][] neighbors, double recall) {}

    /**
     * Builds a live {@link HnswKnnIndex} over {@code rows}, runs the recall
     * gate ({@link #gateAnnRecall}) against it (escalating the SAME built
     * index's query-time {@code ef} via {@link HnswKnnIndex#setEf}, never
     * rebuilding), and -- only if the gate passes -- queries the full kNN
     * graph (every row, self excluded) at the final passing {@code ef}.
     * Package-private so the recall-gate wiring itself is directly testable
     * without going through the full {@link #clusterViaAnn} pipeline.
     */
    static AnnNeighborsResult annNeighborsWithGate(double[][] rows, int k, long seed, boolean reproducible) {
        int n = rows.length;
        int keep = Math.min(k, n - 1);
        HnswKnnIndex index = HnswKnnIndex.build(rows, seed, reproducible);

        double recall = gateAnnRecall(rows, k, seed, INITIAL_QUERY_EF, (ef, sampleIdx) -> {
            index.setEf(ef);
            int[][] out = new int[sampleIdx.length][];
            for (int s = 0; s < sampleIdx.length; s++) {
                out[s] = queryExcludingSelf(index, rows[sampleIdx[s]], sampleIdx[s], keep);
            }
            return out;
        });

        int[][] neighbors = new int[n][];
        IntStream.range(0, n).parallel().forEach(i -> neighbors[i] = queryExcludingSelf(index, rows[i], i, keep));
        return new AnnNeighborsResult(neighbors, recall);
    }

    /**
     * Nearest {@code keep} row indices to {@code vector} from the live
     * {@code index}, excluding {@code selfIdx} -- {@link HnswKnnIndex#queryRow}
     * is a general-purpose primitive with no self-exclusion of its own (the
     * caller may query a vector that is not itself a row in the index), so
     * self-exclusion is applied here by over-fetching by one and filtering.
     */
    private static int[] queryExcludingSelf(HnswKnnIndex index, double[] vector, int selfIdx, int keep) {
        int[] raw = index.queryRow(vector, keep + 1);
        int[] out = new int[keep];
        int count = 0;
        for (int id : raw) {
            if (id == selfIdx) {
                continue;
            }
            if (count < keep) {
                out[count++] = id;
            }
        }
        return count == keep ? out : Arrays.copyOf(out, count);
    }

    /**
     * Builds the undirected, Jaccard-weighted {@link Network} from a symmetric kNN
     * neighbour list. Each undirected edge {@code (i, j)} with {@code i < j} is
     * added once; the {@code Network} constructor used here (verified against the
     * CWTS {@code FileIO.readEdgeList} reference reader) symmetrizes single-
     * direction edge input internally. Jaccard weight uses the CLOSED neighbour
     * sets (each node's set includes itself) — the PhenoGraph/shared-nearest-
     * neighbour convention — so two nodes that are each other's only neighbour get
     * weight 1.0 rather than an undefined/zero weight.
     *
     * <p>Deliberately implemented with primitive sorted {@code int[]} neighbour
     * arrays and a two-pointer merge-intersection walk (no {@code HashSet<Integer>}/
     * {@code HashSet<Long>} boxing) — RESEARCH Pitfall 2 found that the previous
     * boxed per-node {@code HashSet<Integer>[]} plus a global {@code HashSet<Long>}
     * edge-dedup set allocate tens of GB of collection/boxing overhead at cohort
     * scale (~30M nodes x k~15). Do not "simplify" this back to boxed collections
     * — correctness vs. the retired boxed implementation is pinned by
     * {@code LeidenModelTest}'s boxed-vs-primitive equivalence test.
     */
    private static Network buildJaccardWeightedNetwork(int n, int[][] neighbors) {
        JaccardEdgeArrays raw = computeJaccardEdges(n, neighbors);

        // Network(nNodes, setNodeWeightsToTotalEdgeWeights, edges, edgeWeights,
        // sortedEdges, checkIntegrity) — verified against the CWTS source
        // (Network.java javadoc): sortedEdges=false means each undirected edge is
        // listed ONCE, unsorted, and the constructor itself doubles + sorts them
        // into the internal CSR adjacency. Passing sortedEdges=true here (as an
        // earlier draft did) corrupts firstNeighborIndices and crashes
        // LocalMergingAlgorithm with an ArrayIndexOutOfBoundsException.
        //
        // setNodeWeightsToTotalEdgeWeights=true so createNormalizedNetworkUsingAssociationStrength()
        // (called by the caller right after this) has a meaningful node-weight
        // basis for its null-model correction.
        return new Network(n, true, new LargeIntArray[] {raw.from(), raw.to()}, raw.weights(), false, true);
    }

    /**
     * Package-private test seam: exposes the exact same undirected edge list +
     * Jaccard weights that {@link #buildJaccardWeightedNetwork} feeds into the
     * CWTS {@link Network}, as plain arrays, so {@code LeidenModelTest}'s
     * boxed-vs-primitive equivalence test can assert byte-identical output
     * against the retired boxed reference implementation (kept only in the test
     * file). Not called from any production code path.
     */
    static JaccardEdges jaccardEdgesForTest(int n, int[][] neighbors) {
        JaccardEdgeArrays raw = computeJaccardEdges(n, neighbors);
        return new JaccardEdges(
                raw.from().toArray(), raw.to().toArray(), raw.weights().toArray());
    }

    /** Test-seam return type for {@link #jaccardEdgesForTest}. */
    record JaccardEdges(int[] from, int[] to, double[] weights) {}

    /** Internal accumulator shared by the production ({@link #buildJaccardWeightedNetwork}) and test-seam entry points. */
    private record JaccardEdgeArrays(LargeIntArray from, LargeIntArray to, LargeDoubleArray weights) {}

    /**
     * Computes the undirected Jaccard-weighted edge list over the CLOSED (self-
     * included) neighbour sets, via primitive sorted-{@code int[]} neighbour
     * arrays and a two-pointer merge-intersection walk (O(k) per pair) — no
     * boxed {@code HashSet<Integer>}/{@code HashSet<Long>} anywhere (RESEARCH
     * Pitfall 2). Edge dedup is achieved by construction rather than a global
     * seen-edges set: for an unordered pair {@code {a,b}} with {@code a < b},
     * the edge is emitted while processing the LOWER-indexed endpoint {@code a}'s
     * neighbour list whenever {@code b} appears there (covers both symmetric kNN
     * links and asymmetric links where only {@code a} lists {@code b}); it is
     * emitted while processing the HIGHER-indexed endpoint {@code b} ONLY when
     * {@code a} does not already list {@code b} (the asymmetric case where only
     * {@code b} lists {@code a}). This never double-counts because the outer loop
     * visits nodes in ascending order, so the lower endpoint's pass always
     * happens first — exactly mirroring the append order the retired boxed
     * {@code HashSet<Long>} edge-dedup produced (first-occurrence-in-iteration-
     * order wins either way), which is why the resulting edge arrays are
     * byte-identical to (not merely set-equivalent to) the boxed baseline.
     */
    private static JaccardEdgeArrays computeJaccardEdges(int n, int[][] neighbors) {
        int[][] closed = buildClosedNeighborArrays(n, neighbors);

        LargeIntArray from = new LargeIntArray(0);
        LargeIntArray to = new LargeIntArray(0);
        LargeDoubleArray weights = new LargeDoubleArray(0);

        for (int i = 0; i < n; i++) {
            for (int j : neighbors[i]) {
                if (j == i) {
                    continue;
                }
                int a = Math.min(i, j);
                int b = Math.max(i, j);
                if (i != a && containsSorted(closed[a], b)) {
                    // i is the higher-indexed endpoint and the lower endpoint's
                    // neighbour list already contains it -- this edge was (or will
                    // be) emitted while processing node a instead. Skip to avoid a
                    // duplicate.
                    continue;
                }
                double w = jaccardPrimitive(closed[a], closed[b]);
                from.append(a);
                to.append(b);
                weights.append(w);
            }
        }

        return new JaccardEdgeArrays(from, to, weights);
    }

    /**
     * Per-node CLOSED (self-included) neighbour list as a sorted, deduplicated
     * primitive {@code int[]} — replaces the boxed {@code HashSet<Integer>[]}
     * used before this plan's rewrite (RESEARCH Pitfall 2).
     */
    private static int[][] buildClosedNeighborArrays(int n, int[][] neighbors) {
        int[][] closed = new int[n][];
        for (int i = 0; i < n; i++) {
            int[] neigh = neighbors[i];
            int[] combined = new int[neigh.length + 1];
            combined[0] = i;
            System.arraycopy(neigh, 0, combined, 1, neigh.length);
            closed[i] = sortedDedup(combined);
        }
        return closed;
    }

    /** Sorts {@code arr} ascending and removes duplicate values (returns a possibly-shorter copy; input untouched). */
    private static int[] sortedDedup(int[] arr) {
        int[] sorted = arr.clone();
        Arrays.sort(sorted);
        int w = 0;
        for (int r = 0; r < sorted.length; r++) {
            if (r == 0 || sorted[r] != sorted[r - 1]) {
                sorted[w++] = sorted[r];
            }
        }
        return w == sorted.length ? sorted : Arrays.copyOf(sorted, w);
    }

    /** True if the sorted array {@code arr} contains {@code value} (binary search, no boxing). */
    private static boolean containsSorted(int[] arr, int value) {
        int lo = 0;
        int hi = arr.length - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >>> 1;
            int v = arr[mid];
            if (v == value) {
                return true;
            }
            if (v < value) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return false;
    }

    /**
     * Jaccard similarity of two sorted, deduplicated closed-neighbour arrays via a
     * two-pointer merge-intersection walk — O(|a|+|b|) per pair, no hashing or
     * boxed {@code Integer} allocation. Mathematically identical to (and, per
     * {@code LeidenModelTest}'s equivalence test, byte-identical in output to) the
     * retired boxed set-based {@code jaccard(Set,Set)} implementation: both reduce
     * to {@code intersectionCount / (double) unionCount} over the same integer
     * counts, so there is no floating-point summation-order sensitivity to worry
     * about.
     */
    private static double jaccardPrimitive(int[] a, int[] b) {
        int i = 0;
        int j = 0;
        int intersection = 0;
        while (i < a.length && j < b.length) {
            if (a[i] == b[j]) {
                intersection++;
                i++;
                j++;
            } else if (a[i] < b[j]) {
                i++;
            } else {
                j++;
            }
        }
        int union = a.length + b.length - intersection;
        return union == 0 ? 0.0 : intersection / (double) union;
    }

    // ── kNN label transfer ───────────────────────────────────────────────────

    /**
     * kNN label transfer (scanpy {@code sc.tl.ingest} style): assign each query
     * row the majority label among its {@code k} nearest reference rows (ties ->
     * lowest label id). Brute-force over the bounded reference sample. Handles a
     * degenerate/NaN query row and a single-row reference without crashing.
     *
     * @param query rows to label (e.g. every cell in a cohort image, z-scored)
     * @param reference the labelled pooled sample (z-scored)
     * @param refLabels labels for each reference row, parallel to {@code reference}
     * @param k number of nearest reference rows to vote over
     * @param nClusters number of distinct labels (for sizing the vote histogram)
     */
    public static int[] transferLabels(double[][] query, double[][] reference, int[] refLabels, int k, int nClusters) {
        int nq = query.length;
        int[] assigned = new int[nq];
        int nRef = reference.length;
        if (nRef == 0 || refLabels.length == 0) {
            // An empty reference has no labels to vote from -- default to label 0 here would
            // silently misclassify every query cell as cluster 0's phenotype (Rule 1 bug fix).
            // -1 means "unassigned"; callers must treat it as such (skip / no class), not as a
            // real cluster id.
            Arrays.fill(assigned, -1);
            return assigned;
        }
        int keep = Math.max(1, Math.min(k, nRef));
        int histSize = Math.max(1, nClusters);
        // Each query row's kNN vote is independent (writes only its own assigned[qi]), so the
        // brute-force scan parallelises across query rows. Combined with the bounded-heap
        // selection in nearestReferenceIndices (no per-query full sort or boxed Integer[]),
        // this turns the transfer from serial O(nq·nRef·log nRef) into O(nq·nRef·log keep)
        // across all cores — the same optimisation applied to featureKnn. Labels are
        // byte-identical to the previous stable full-sort implementation.
        IntStream.range(0, nq)
                .parallel()
                .forEach(qi -> assigned[qi] = voteLabel(query[qi], reference, refLabels, keep, histSize));
        return assigned;
    }

    /** Majority label among the {@code keep} nearest reference rows to {@code q} (ties -> lowest label id). */
    private static int voteLabel(double[] q, double[][] reference, int[] refLabels, int keep, int histSize) {
        int[] nearest = nearestReferenceIndices(q, reference, keep);
        int[] votes = new int[histSize];
        for (int idx : nearest) {
            int label = refLabels[idx];
            if (label >= 0) {
                if (label >= votes.length) {
                    votes = Arrays.copyOf(votes, label + 1);
                }
                votes[label]++;
            }
        }
        return argMaxLowestTie(votes);
    }

    /**
     * The {@code keep} nearest reference rows to {@code q}, ordered ascending by
     * (squared distance, index) — identical to the previous stable full-sort output,
     * so the majority vote and tie behaviour are unchanged. Uses the same bounded
     * max-heap selection as {@link #nearestForRow} (O(nRef log keep), no boxing).
     */
    private static int[] nearestReferenceIndices(double[] q, double[][] reference, int keep) {
        int n = reference.length;
        int k = Math.min(keep, n);
        double[] heapDist = new double[k];
        int[] heapIdx = new int[k];
        int size = 0;
        for (int j = 0; j < n; j++) {
            double d = squaredDistance(q, reference[j]);
            if (size < k) {
                heapDist[size] = d;
                heapIdx[size] = j;
                siftUp(heapDist, heapIdx, size);
                size++;
            } else if (worse(heapDist[0], heapIdx[0], d, j)) {
                heapDist[0] = d;
                heapIdx[0] = j;
                siftDown(heapDist, heapIdx, size);
            }
        }
        for (int a = 1; a < size; a++) {
            double dv = heapDist[a];
            int iv = heapIdx[a];
            int b = a - 1;
            while (b >= 0 && worse(heapDist[b], heapIdx[b], dv, iv)) {
                heapDist[b + 1] = heapDist[b];
                heapIdx[b + 1] = heapIdx[b];
                b--;
            }
            heapDist[b + 1] = dv;
            heapIdx[b + 1] = iv;
        }
        int[] result = new int[size];
        System.arraycopy(heapIdx, 0, result, 0, size);
        return result;
    }

    private static int argMaxLowestTie(int[] votes) {
        int best = 0;
        int bestCount = votes.length == 0 ? 0 : votes[0];
        for (int label = 1; label < votes.length; label++) {
            if (votes[label] > bestCount) {
                bestCount = votes[label];
                best = label;
            }
        }
        return best;
    }
}
