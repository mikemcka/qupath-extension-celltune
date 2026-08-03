package qupath.ext.celltune.classifier;

import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.celltune.util.BackgroundExecutors;
import qupath.ext.celltune.util.TrainingThreads;

/**
 * Resampling methods for addressing class imbalance in training data.
 * <p>
 * All methods operate on parallel lists of feature rows and integer class labels,
 * returning new resampled lists. The originals are not modified.
 * <p>
 * Supported strategies:
 * <ul>
 *   <li><b>SMOTE</b> — oversample minority classes by interpolating between
 *       k-nearest same-class neighbours</li>
 *   <li><b>ADASYN</b> — like SMOTE but concentrates synthetic samples on
 *       harder-to-learn minority examples (those with more different-class neighbours)</li>
 *   <li><b>Tomek links</b> — remove majority-class samples that form mutual
 *       nearest-neighbour pairs with minority-class samples</li>
 *   <li><b>Combinations</b> — SMOTE + Tomek, ADASYN + Tomek (oversample then clean)</li>
 * </ul>
 *
 * <h2>Implementation note — this is the training-time hot spot</h2>
 * Resampling runs on every Train click, and Tomek runs on the SMOTE-inflated row count, so its
 * O(n²·d) all-pairs search dominates training on any imbalanced dataset. The internals are
 * therefore built for speed while producing <em>bit-identical</em> output to the original
 * straightforward implementation:
 * <ul>
 *   <li>Rows and labels are held in primitive {@code float[][]}/{@code int[]} buffers rather than
 *       {@code List<float[]>}/{@code List<Integer>}, removing per-comparison unboxing from the
 *       innermost loop. The {@code List} signature is preserved at the API boundary.</li>
 *   <li>SMOTE's k-nearest-neighbour lookup is memoised. The original recomputed the identical
 *       neighbours {@code ceil(needed/m)} times per source sample, because the generation loop
 *       cycles {@code i % m} while {@code classIndices} and the rows it reads never change.</li>
 *   <li>Tomek's search iterates a per-class index partition, so the different-class filter is
 *       structural rather than a test per candidate, and abandons a distance early once it
 *       exceeds the best so far.</li>
 *   <li>The Tomek search and the neighbour precomputation are parallelised; each task writes only
 *       its own output slot, so there is nothing to merge.</li>
 * </ul>
 * Because the parallel search visits candidates out of index order, the "nearest different-class
 * neighbour" tie-break is made explicit ({@code argmin} with lowest index winning) instead of
 * relying on an ascending scan with a strict {@code <}. That is what makes the result independent
 * of thread count — see {@code ResamplerGoldenTest}, which pins the exact output of every
 * strategy and asserts it is invariant across thread counts.
 */
public final class Resampler {

    private static final Logger logger = LoggerFactory.getLogger(Resampler.class);

    /** Default number of nearest neighbours for SMOTE/ADASYN. */
    private static final int DEFAULT_K = 5;

    /**
     * Rough floating-point-operation count below which a parallel split costs more than it saves.
     * Tuned generously: thread hand-off is ~microseconds, so anything above a few million
     * operations is comfortably worth splitting.
     */
    private static final long PARALLEL_MIN_WORK = 2_000_000L;

    private Resampler() {} // utility class

    /**
     * Apply the given resampling strategy to the training data.
     *
     * @param rows     feature vectors (one per sample)
     * @param labels   integer class labels (parallel to rows)
     * @param nClasses total number of classes
     * @param strategy which resampling method to apply
     * @param log      optional progress callback
     * @return a {@link Result} containing the resampled rows and labels
     */
    public static Result apply(
            List<float[]> rows, List<Integer> labels, int nClasses, ResamplingStrategy strategy, Consumer<String> log) {
        Consumer<String> out = log != null ? log : s -> {};

        // Validate label indices up front so a corrupt label (e.g. a class index
        // left over after an out-of-sync class edit) fails with a clear message
        // rather than a bare ArrayIndexOutOfBoundsException deep in counting/kNN.
        validateLabels(labels, nClasses);

        if (strategy == ResamplingStrategy.NONE) {
            return Result.copyOf(rows, labels);
        }

        out.accept("Resampling: " + strategy + "…");

        // Count per-class
        int[] classCounts = new int[nClasses];
        for (int lbl : labels) classCounts[lbl]++;

        int maxCount = 0;
        for (int c : classCounts) if (c > maxCount) maxCount = c;

        // Exact upper bound: oversampling brings every class up to maxCount at most, and the
        // Tomek pass only ever removes. One allocation, no growth checks in the hot loop.
        int capacity = rows.size();
        for (int c : classCounts) capacity += Math.max(0, maxCount - c);

        Buffer buf = Buffer.from(rows, labels, capacity);

        try (Parallel par = new Parallel()) {
            switch (strategy) {
                case SMOTE -> {
                    smote(buf, classCounts, maxCount, nClasses, out, par);
                }
                case ADASYN -> {
                    adasyn(buf, classCounts, maxCount, nClasses, out, par);
                }
                case TOMEK -> {
                    int removed = tomekLinks(buf, classCounts, nClasses, par);
                    out.accept("Tomek links removed " + removed + " majority-class samples");
                }
                case SMOTE_TOMEK -> {
                    smote(buf, classCounts, maxCount, nClasses, out, par);
                    // Recount after SMOTE
                    int[] newCounts = buf.recount(nClasses);
                    int removed = tomekLinks(buf, newCounts, nClasses, par);
                    out.accept("Tomek links removed " + removed + " borderline samples after SMOTE");
                }
                case ADASYN_TOMEK -> {
                    adasyn(buf, classCounts, maxCount, nClasses, out, par);
                    int[] newCounts = buf.recount(nClasses);
                    int removed = tomekLinks(buf, newCounts, nClasses, par);
                    out.accept("Tomek links removed " + removed + " borderline samples after ADASYN");
                }
                default -> {
                    /* NONE — handled above */
                }
            }
        }

        // Log final distribution
        int[] finalCounts = buf.recount(nClasses);
        StringBuilder sb = new StringBuilder("Resampled distribution: ");
        for (int c = 0; c < nClasses; c++) {
            if (c > 0) sb.append(", ");
            sb.append("class ").append(c).append("=").append(finalCounts[c]);
        }
        sb.append(" (total ")
                .append(buf.size)
                .append(", was ")
                .append(rows.size())
                .append(")");
        out.accept(sb.toString());

        return buf.toResult();
    }

    // ── SMOTE ───────────────────────────────────────────────────────────────────

    private static void smote(
            Buffer buf, int[] classCounts, int targetCount, int nClasses, Consumer<String> log, Parallel par) {
        Random rng = new Random(42);
        int nFeatures = buf.rows[0].length;

        for (int cls = 0; cls < nClasses; cls++) {
            int count = classCounts[cls];
            int needed = targetCount - count;
            if (needed <= 0) continue;

            // Collect indices of this class
            int[] classIndices = buf.indicesOfClass(cls);
            int m = classIndices.length;
            if (m < 2) continue; // can't interpolate with < 2

            int k = Math.min(DEFAULT_K, m - 1);

            // Pre-compute the k nearest same-class neighbours once per source sample. The
            // generation loop below cycles i % m, so without this each sample's identical
            // neighbour set is recomputed ceil(needed/m) times.
            int[][] knn = knnSameClassBatch(buf.rows, classIndices, k, Math.min(m, needed), par);

            int generated = 0;
            for (int i = 0; generated < needed; i++) {
                int pos = i % m;
                float[] sample = buf.rows[classIndices[pos]];

                int[] neighbours = knn[pos];
                float[] neighbour = buf.rows[classIndices[neighbours[rng.nextInt(k)]]];

                // Interpolate
                float lambda = rng.nextFloat();
                float[] synthetic = new float[nFeatures];
                for (int f = 0; f < nFeatures; f++) {
                    synthetic[f] = sample[f] + lambda * (neighbour[f] - sample[f]);
                }

                buf.add(synthetic, cls);
                generated++;
            }

            log.accept("SMOTE: generated " + generated + " synthetic samples for class " + cls);
        }
    }

    // ── ADASYN ──────────────────────────────────────────────────────────────────

    private static void adasyn(
            Buffer buf, int[] classCounts, int targetCount, int nClasses, Consumer<String> log, Parallel par) {
        Random rng = new Random(42);
        int nFeatures = buf.rows[0].length;
        int totalSamples = buf.size;

        for (int cls = 0; cls < nClasses; cls++) {
            int count = classCounts[cls];
            int needed = targetCount - count;
            if (needed <= 0) continue;

            int[] classIndices = buf.indicesOfClass(cls);
            int m = classIndices.length;
            if (m < 2) continue;

            int k = Math.min(DEFAULT_K, totalSamples - 1);

            // The all-class neighbour scan sees whatever rows exist right now — which includes
            // synthetic rows generated for *earlier* classes. Snapshot that width so every
            // sample in this class is scored against the same population.
            final int scanN = buf.size;
            final int currentCls = cls;

            // For each minority sample, compute ratio of different-class neighbours
            double[] ratios = new double[m];
            par.range(m, (long) m * scanN * nFeatures, (lo, hi) -> {
                for (int i = lo; i < hi; i++) {
                    int idx = classIndices[i];
                    int[] nnIndices = knnAll(buf.rows, scanN, idx, k);
                    int differentCount = 0;
                    for (int ni : nnIndices) {
                        if (buf.labels[ni] != currentCls) differentCount++;
                    }
                    ratios[i] = (double) differentCount / k;
                }
            });
            // Summed in ascending index order, matching the original accumulation exactly —
            // double addition is not associative, so the order is part of the contract.
            double ratioSum = 0;
            for (int i = 0; i < m; i++) ratioSum += ratios[i];

            if (ratioSum < 1e-10) {
                // All neighbours are same class — fall back to uniform SMOTE
                smoteSingle(buf, classIndices, needed, nFeatures, rng, par);
                log.accept("ADASYN (uniform fallback): generated " + needed + " synthetic samples for class " + cls);
                continue;
            }

            int kLocal = Math.min(DEFAULT_K, m - 1);

            // Replay the generation schedule without doing any distance work, to learn exactly
            // which samples the loop below will reach. Only those need their neighbours computed
            // — matching the original's work, but in one parallel batch instead of serially.
            boolean[] used = new boolean[m];
            int planned = 0;
            for (int i = 0; i < m && planned < needed; i++) {
                int numForThis = (int) Math.round(needed * ratios[i] / ratioSum);
                if (numForThis <= 0) continue;
                used[i] = true;
                planned += Math.min(numForThis, needed - planned);
            }
            int[][] knn = knnSameClassBatch(buf.rows, classIndices, kLocal, used, par);

            // Normalise ratios to get per-sample generation weights
            int generated = 0;
            for (int i = 0; i < m && generated < needed; i++) {
                int numForThis = (int) Math.round(needed * ratios[i] / ratioSum);
                if (numForThis <= 0) continue;

                float[] sample = buf.rows[classIndices[i]];
                int[] neighbours = knn[i];

                for (int g = 0; g < numForThis && generated < needed; g++) {
                    float[] neighbour = buf.rows[classIndices[neighbours[rng.nextInt(kLocal)]]];

                    float lambda = rng.nextFloat();
                    float[] synthetic = new float[nFeatures];
                    for (int f = 0; f < nFeatures; f++) {
                        synthetic[f] = sample[f] + lambda * (neighbour[f] - sample[f]);
                    }
                    buf.add(synthetic, cls);
                    generated++;
                }
            }

            log.accept("ADASYN: generated " + generated + " synthetic samples for class " + cls);
        }
    }

    private static void smoteSingle(
            Buffer buf, int[] classIndices, int needed, int nFeatures, Random rng, Parallel par) {
        int m = classIndices.length;
        int k = Math.min(DEFAULT_K, m - 1);
        int cls = buf.labels[classIndices[0]];
        int[][] knn = knnSameClassBatch(buf.rows, classIndices, k, Math.min(m, needed), par);

        int generated = 0;
        for (int i = 0; generated < needed; i++) {
            int pos = i % m;
            float[] sample = buf.rows[classIndices[pos]];
            int[] neighbours = knn[pos];
            float[] neighbour = buf.rows[classIndices[neighbours[rng.nextInt(k)]]];

            float lambda = rng.nextFloat();
            float[] synthetic = new float[nFeatures];
            for (int f = 0; f < nFeatures; f++) {
                synthetic[f] = sample[f] + lambda * (neighbour[f] - sample[f]);
            }
            buf.add(synthetic, cls);
            generated++;
        }
    }

    // ── Tomek Links ─────────────────────────────────────────────────────────────

    /**
     * Remove majority-class members of Tomek link pairs.
     * A Tomek link is a pair (a, b) from different classes where each is the
     * other's nearest neighbour of the opposite class.
     *
     * @return number of samples removed
     */
    private static int tomekLinks(Buffer buf, int[] classCounts, int nClasses, Parallel par) {
        int maxCount = 0;
        for (int c : classCounts) if (c > maxCount) maxCount = c;

        // Identify majority classes (those at the max count)
        boolean[] isMajority = new boolean[nClasses];
        for (int c = 0; c < nClasses; c++) {
            if (classCounts[c] == maxCount) isMajority[c] = true;
        }

        final int n = buf.size;
        final float[][] rows = buf.rows;
        final int[] labels = buf.labels;
        final int nFeatures = n > 0 ? rows[0].length : 0;

        // Index partition by class: walking these lets the different-class filter be structural
        // (skip the whole block) instead of a per-candidate label comparison.
        final int[][] byClass = buf.partitionByClass(nClasses);

        // For each sample, find its nearest neighbour of a different class
        final int[] nnDiffClass = new int[n];
        par.range(n, (long) n * n * nFeatures, (lo, hi) -> {
            for (int i = lo; i < hi; i++) {
                float[] a = rows[i];
                int labelA = labels[i];
                float bestDist = Float.MAX_VALUE;
                int bestIdx = -1;
                for (int c = 0; c < nClasses; c++) {
                    if (c == labelA) continue;
                    int[] members = byClass[c];
                    for (int t = 0; t < members.length; t++) {
                        int j = members[t];
                        float d = distSqBounded(a, rows[j], bestDist);
                        // argmin with the lowest index winning ties — the ascending scan this
                        // replaces got that for free; stated explicitly it also holds when the
                        // candidates are visited per class rather than in index order.
                        if (d < bestDist || (d == bestDist && j < bestIdx)) {
                            bestDist = d;
                            bestIdx = j;
                        }
                    }
                }
                nnDiffClass[i] = bestIdx;
            }
        });

        // Find Tomek links — mutual nearest different-class neighbours
        boolean[] toRemove = new boolean[n];
        int removed = 0;
        for (int i = 0; i < n; i++) {
            int j = nnDiffClass[i];
            if (j < 0) continue;
            if (nnDiffClass[j] == i) {
                // Tomek link found — remove the majority-class member
                int victim = -1;
                if (isMajority[labels[i]]) {
                    victim = i;
                } else if (isMajority[labels[j]]) {
                    victim = j;
                }
                if (victim >= 0 && !toRemove[victim]) {
                    toRemove[victim] = true;
                    removed++;
                }
            }
        }

        // Single compacting pass; survivors keep their relative order, as removing in
        // descending index order did.
        buf.removeMarked(toRemove, removed);

        return removed;
    }

    // ── Helpers ──────────────────────────────────────────────────────────────────

    /**
     * Computes {@link #knnSameClass} for the first {@code count} members of {@code classIndices}
     * — the ones the generation loop will cycle through.
     */
    private static int[][] knnSameClassBatch(float[][] rows, int[] classIndices, int k, int count, Parallel par) {
        boolean[] wanted = new boolean[classIndices.length];
        for (int i = 0; i < count; i++) wanted[i] = true;
        return knnSameClassBatch(rows, classIndices, k, wanted, par);
    }

    /** Computes {@link #knnSameClass} for the selected members of {@code classIndices}. */
    private static int[][] knnSameClassBatch(
            float[][] rows, int[] classIndices, int k, boolean[] wanted, Parallel par) {
        int m = classIndices.length;
        int[][] out = new int[m][];
        int nFeatures = m > 0 ? rows[classIndices[0]].length : 0;
        int wantedCount = 0;
        for (boolean b : wanted) if (b) wantedCount++;

        par.range(m, (long) wantedCount * m * nFeatures, (lo, hi) -> {
            for (int p = lo; p < hi; p++) {
                if (wanted[p]) {
                    out[p] = knnSameClass(rows, classIndices, classIndices[p], k);
                }
            }
        });
        return out;
    }

    /** Indices within classIndices of the k nearest same-class neighbours of rows[sampleIdx]. */
    private static int[] knnSameClass(float[][] rows, int[] classIndices, int sampleIdx, int k) {
        float[] sample = rows[sampleIdx];
        int m = classIndices.length;
        float[] dists = new float[m];

        for (int i = 0; i < m; i++) {
            int ci = classIndices[i];
            if (ci == sampleIdx) {
                dists[i] = Float.MAX_VALUE;
            } else {
                dists[i] = euclideanDistSq(sample, rows[ci]);
            }
        }

        // Partial sort to find k smallest. Kept as the original k-pass selection sort: each pass
        // swaps into the unsorted tail, so which of several equal distances ends up in slot i
        // depends on that swap history. A "sort by (distance, index)" replacement would pick
        // different neighbours whenever two rows are equidistant.
        int[] indices = new int[m];
        for (int i = 0; i < m; i++) indices[i] = i;

        for (int i = 0; i < k; i++) {
            int minIdx = i;
            for (int j = i + 1; j < m; j++) {
                if (dists[indices[j]] < dists[indices[minIdx]]) {
                    minIdx = j;
                }
            }
            int tmp = indices[i];
            indices[i] = indices[minIdx];
            indices[minIdx] = tmp;
        }

        return Arrays.copyOf(indices, k);
    }

    /** Indices of the k nearest neighbours (all classes) of rows[sampleIdx], scanning rows[0, n). */
    private static int[] knnAll(float[][] rows, int n, int sampleIdx, int k) {
        float[] sample = rows[sampleIdx];
        float[] dists = new float[n];
        for (int i = 0; i < n; i++) {
            dists[i] = (i == sampleIdx) ? Float.MAX_VALUE : euclideanDistSq(sample, rows[i]);
        }

        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        for (int i = 0; i < k; i++) {
            int minIdx = i;
            for (int j = i + 1; j < n; j++) {
                if (dists[indices[j]] < dists[indices[minIdx]]) {
                    minIdx = j;
                }
            }
            int tmp = indices[i];
            indices[i] = indices[minIdx];
            indices[minIdx] = tmp;
        }

        return Arrays.copyOf(indices, k);
    }

    private static float euclideanDistSq(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return sum;
    }

    /**
     * Squared Euclidean distance that gives up once it provably cannot beat {@code bound}.
     * <p>
     * Terms are non-negative, so a partial sum already above {@code bound} can only grow. The
     * returned value is then some number strictly greater than {@code bound}, which loses both
     * the {@code <} and the {@code ==} comparison at the call site exactly as the full distance
     * would have. When the distance is <em>not</em> abandoned the accumulation is identical to
     * {@link #euclideanDistSq}, so surviving comparisons are bit-for-bit unchanged.
     * <p>
     * A NaN feature makes the sum NaN, which is never {@code >} the bound, so such a pair runs to
     * completion and then loses every comparison — matching the unbounded version.
     * <p>
     * The bound is only tested every 8th element: often enough to cut the vast majority of
     * hopeless candidates short, rarely enough not to burden the vectorisable inner loop.
     */
    private static float distSqBounded(float[] a, float[] b, float bound) {
        float sum = 0;
        int len = a.length;
        for (int i = 0; i < len; i++) {
            float d = a[i] - b[i];
            sum += d * d;
            if ((i & 7) == 7 && sum > bound) {
                return sum;
            }
        }
        return sum;
    }

    /**
     * Ensure every label is a valid class index in {@code [0, nClasses)}.
     *
     * @throws IllegalArgumentException if {@code nClasses < 1} or any label is out of range
     */
    private static void validateLabels(List<Integer> labels, int nClasses) {
        if (nClasses < 1) {
            throw new IllegalArgumentException("nClasses must be >= 1, was " + nClasses);
        }
        for (int i = 0; i < labels.size(); i++) {
            Integer lbl = labels.get(i);
            if (lbl == null || lbl < 0 || lbl >= nClasses) {
                throw new IllegalArgumentException("Label at index " + i + " is " + lbl + ", outside valid range [0, "
                        + nClasses + "). Training data is out of sync with the class list.");
            }
        }
    }

    // ── Parallel execution ───────────────────────────────────────────────────────

    /** A chunk of an index range, {@code [lo, hi)}. */
    @FunctionalInterface
    private interface RangeTask {
        void run(int lo, int hi);
    }

    /**
     * Runs index ranges across a pool sized from the training thread budget, created lazily so a
     * small dataset never pays for one. Tasks here write only to their own output slots, so there
     * is no accumulator to merge and no ordering to preserve.
     */
    private static final class Parallel implements AutoCloseable {

        private ExecutorService pool;

        /**
         * Splits {@code [0, n)} into one contiguous chunk per thread.
         *
         * @param estimatedWork rough operation count, used to skip the split when the work is too
         *                      small to be worth a hand-off
         */
        void range(int n, long estimatedWork, RangeTask task) {
            int threads = Math.min(TrainingThreads.total(), n);
            if (threads <= 1 || estimatedWork < PARALLEL_MIN_WORK) {
                task.run(0, n);
                return;
            }
            if (pool == null) {
                pool = BackgroundExecutors.newFixedPool(TrainingThreads.total(), "CellTune-Resample");
            }
            int chunk = (n + threads - 1) / threads;
            List<Future<?>> futures = new ArrayList<>(threads);
            for (int t = 0; t < threads; t++) {
                final int lo = t * chunk;
                final int hi = Math.min(n, lo + chunk);
                if (lo >= hi) break;
                futures.add(pool.submit(() -> task.run(lo, hi)));
            }
            try {
                for (Future<?> f : futures) {
                    f.get();
                }
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("Resampling was interrupted", ie);
            } catch (ExecutionException ee) {
                Throwable cause = ee.getCause();
                if (cause instanceof RuntimeException re) throw re;
                if (cause instanceof Error err) throw err;
                throw new IllegalStateException("Resampling failed", cause);
            }
        }

        @Override
        public void close() {
            if (pool != null) {
                pool.shutdown();
                pool = null;
            }
        }
    }

    // ── Primitive working buffer ─────────────────────────────────────────────────

    /**
     * Rows and labels as primitive arrays with a pre-sized capacity.
     * <p>
     * Row objects are shared by reference with the caller's list — exactly what
     * {@code new ArrayList<>(rows)} did — so only synthetic rows are ever allocated, and the
     * caller's own lists are never touched.
     */
    private static final class Buffer {

        private float[][] rows;
        private int[] labels;
        private int size;

        private Buffer(float[][] rows, int[] labels, int size) {
            this.rows = rows;
            this.labels = labels;
            this.size = size;
        }

        static Buffer from(List<float[]> srcRows, List<Integer> srcLabels, int capacity) {
            int n = srcRows.size();
            float[][] rows = new float[Math.max(capacity, n)][];
            int[] labels = new int[Math.max(capacity, n)];
            for (int i = 0; i < n; i++) {
                rows[i] = srcRows.get(i);
                labels[i] = srcLabels.get(i);
            }
            return new Buffer(rows, labels, n);
        }

        void add(float[] row, int label) {
            if (size == rows.length) {
                // Defensive only: capacity is computed to be exact for every strategy.
                int grown = Math.max(size + 1, size + (size >> 1));
                rows = Arrays.copyOf(rows, grown);
                labels = Arrays.copyOf(labels, grown);
            }
            rows[size] = row;
            labels[size] = label;
            size++;
        }

        int[] indicesOfClass(int cls) {
            int count = 0;
            for (int i = 0; i < size; i++) if (labels[i] == cls) count++;
            int[] out = new int[count];
            int w = 0;
            for (int i = 0; i < size; i++) if (labels[i] == cls) out[w++] = i;
            return out;
        }

        int[][] partitionByClass(int nClasses) {
            int[] counts = new int[nClasses];
            for (int i = 0; i < size; i++) counts[labels[i]]++;
            int[][] out = new int[nClasses][];
            for (int c = 0; c < nClasses; c++) out[c] = new int[counts[c]];
            int[] w = new int[nClasses];
            for (int i = 0; i < size; i++) {
                int c = labels[i];
                out[c][w[c]++] = i;
            }
            return out;
        }

        int[] recount(int nClasses) {
            int[] counts = new int[nClasses];
            for (int i = 0; i < size; i++) counts[labels[i]]++;
            return counts;
        }

        /** Drops every marked row in one pass, preserving the order of the survivors. */
        void removeMarked(boolean[] marked, int markedCount) {
            if (markedCount == 0) {
                return;
            }
            int w = 0;
            for (int i = 0; i < size; i++) {
                if (!marked[i]) {
                    rows[w] = rows[i];
                    labels[w] = labels[i];
                    w++;
                }
            }
            for (int i = w; i < size; i++) rows[i] = null; // release references
            size = w;
        }

        Result toResult() {
            return new Result(rows, labels, size);
        }
    }

    /** Result container for resampled data. */
    public static final class Result {

        private final float[][] rowArray;
        private final int[] labelArray;
        private final int size;

        private List<float[]> rowList;
        private List<Integer> labelList;

        Result(float[][] rowArray, int[] labelArray, int size) {
            this.rowArray = rowArray;
            this.labelArray = labelArray;
            this.size = size;
        }

        /** Builds a result that is a shallow copy of the given lists (the {@code NONE} path). */
        static Result copyOf(List<float[]> rows, List<Integer> labels) {
            int n = rows.size();
            float[][] r = new float[n][];
            int[] l = new int[n];
            for (int i = 0; i < n; i++) {
                r[i] = rows.get(i);
                l[i] = labels.get(i);
            }
            return new Result(r, l, n);
        }

        /** @return number of samples in the result */
        public int size() {
            return size;
        }

        /**
         * Feature rows as a mutable list. Materialised on first call; callers that go straight to
         * a flat training matrix should prefer {@link #rowArray()} and skip this entirely.
         */
        public List<float[]> rows() {
            if (rowList == null) {
                List<float[]> out = new ArrayList<>(size);
                for (int i = 0; i < size; i++) out.add(rowArray[i]);
                rowList = out;
            }
            return rowList;
        }

        /** Class labels as a mutable list, parallel to {@link #rows()}. */
        public List<Integer> labels() {
            if (labelList == null) {
                List<Integer> out = new ArrayList<>(size);
                for (int i = 0; i < size; i++) out.add(labelArray[i]);
                labelList = out;
            }
            return labelList;
        }

        /**
         * Feature rows as a primitive array of exactly {@link #size()} entries — avoids boxing an
         * {@code ArrayList} that the caller would immediately flatten again.
         */
        public float[][] rowArray() {
            return rowArray.length == size ? rowArray : Arrays.copyOf(rowArray, size);
        }

        /** Class labels as a primitive array of exactly {@link #size()} entries. */
        public int[] labelArray() {
            return labelArray.length == size ? labelArray : Arrays.copyOf(labelArray, size);
        }
    }
}
