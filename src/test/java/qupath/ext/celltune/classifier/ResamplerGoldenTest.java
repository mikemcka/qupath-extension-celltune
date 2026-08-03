package qupath.ext.celltune.classifier;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import qupath.ext.celltune.util.TrainingThreads;

/**
 * Bit-identity oracle for {@link Resampler}.
 * <p>
 * {@code ResamplerTest} asserts only structural invariants (parallel lists, majority class
 * untouched, Tomek never grows the set). Nothing pins the actual numeric output, so a refactor
 * could silently change which synthetic samples are generated or which rows Tomek removes and
 * every existing test would still pass.
 * <p>
 * This class closes that gap: for each strategy on each fixture it pins the exact output size,
 * the exact per-class counts, and an FNV-1a checksum folded over the raw bits of every feature
 * value and every label, in order. The constants were captured against the original brute-force
 * implementation and must not be "updated" to make a change pass — a diff here means the
 * resampled training set changed, which changes the trained model.
 * <p>
 * Fixtures are pure functions of their index (no {@link java.util.Random}, no I/O) so they are
 * stable across JDKs and machines.
 * <p>
 * <b>Not safe to run concurrently with other tests.</b> Two of these set
 * {@link TrainingThreads#setOverride(int)}, which is process-global; they restore it in a
 * {@code finally}, which is sufficient only because JUnit parallel execution is off (there is no
 * {@code junit-platform.properties}). If it is ever switched on, this class needs
 * {@code @Execution(SAME_THREAD)} or an isolated resource lock.
 */
class ResamplerGoldenTest {

    // ── Fixtures ────────────────────────────────────────────────────────────────

    /**
     * A labelled dataset held in the same parallel-list shape {@code Resampler.apply} consumes.
     */
    record Fixture(String name, List<float[]> rows, List<Integer> labels, int nClasses) {
        Fixture copy() {
            List<float[]> r = new ArrayList<>(rows.size());
            for (float[] row : rows) r.add(row.clone());
            return new Fixture(name, r, new ArrayList<>(labels), nClasses);
        }
    }

    /**
     * Builds a deterministic fixture with the given per-class counts.
     * <p>
     * Rows are generated grouped by class and then permuted by a fixed coprime stride so the
     * classes are interleaved — matching real label stores, and making
     * {@code indicesOfClass} return non-contiguous indices the way it does in production.
     * {@code classSpread} controls how far apart the class means sit: small values make the
     * classes overlap, which is what causes Tomek links to actually form.
     */
    private static Fixture build(String name, int[] counts, int nFeatures, float classSpread) {
        int n = 0;
        for (int c : counts) n += c;

        List<float[]> grouped = new ArrayList<>(n);
        List<Integer> groupedLabels = new ArrayList<>(n);
        int i = 0;
        for (int cls = 0; cls < counts.length; cls++) {
            for (int m = 0; m < counts[cls]; m++, i++) {
                float[] row = new float[nFeatures];
                for (int f = 0; f < nFeatures; f++) {
                    row[f] = ((i * 31 + f * 7) % 97) / 97f + cls * classSpread;
                }
                grouped.add(row);
                groupedLabels.add(cls);
            }
        }

        // Fixed coprime stride permutation — deterministic interleave, no RNG.
        int stride = 7919 % n == 0 ? 1 : 7919;
        List<float[]> rows = new ArrayList<>(n);
        List<Integer> labels = new ArrayList<>(n);
        for (int k = 0; k < n; k++) {
            int src = (int) (((long) k * stride) % n);
            rows.add(grouped.get(src));
            labels.add(groupedLabels.get(src));
        }
        return new Fixture(name, rows, labels, counts.length);
    }

    /** Moderate imbalance, well-separated-but-overlapping classes. */
    private static Fixture moderate() {
        return build("moderate", new int[] {200, 60, 20}, 8, 0.30f);
    }

    /**
     * Contains exact duplicate rows on both sides of a class boundary, so that
     * {@code euclideanDistSq} returns bit-equal distances for competing neighbours. This pins
     * Tomek's implicit "lowest index wins" tie-break and SMOTE's selection-sort tie-break, both
     * of which are easy to break accidentally when reordering or parallelising the search.
     */
    private static Fixture duplicates() {
        int nFeatures = 4;
        List<float[]> rows = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        // 12 distinct positions, each present once in class 0 and once in class 1 -> every
        // cross-class nearest neighbour is an exact tie at distance 0.
        for (int rep = 0; rep < 2; rep++) {
            for (int p = 0; p < 12; p++) {
                float[] row = new float[nFeatures];
                for (int f = 0; f < nFeatures; f++) row[f] = ((p * 13 + f * 5) % 17) / 17f;
                rows.add(row);
                labels.add(rep); // rep 0 -> class 0, rep 1 -> class 1
            }
        }
        // Pad class 0 so it is the strict majority (Tomek only removes majority members).
        for (int p = 0; p < 20; p++) {
            float[] row = new float[nFeatures];
            for (int f = 0; f < nFeatures; f++) row[f] = 0.5f + ((p * 3 + f) % 11) / 40f;
            rows.add(row);
            labels.add(0);
        }
        return new Fixture("duplicates", rows, labels, 2);
    }

    /**
     * Severe imbalance: {@code needed / m} is ~99 for the smallest class, so the original
     * implementation recomputes each source sample's kNN ~99 times. This is the fixture that
     * proves the memoisation returns bit-identical neighbours rather than merely similar ones.
     */
    private static Fixture severe() {
        return build("severe", new int[] {500, 5, 40}, 6, 0.25f);
    }

    /**
     * Wide enough that Tomek's bounded distance actually abandons candidates part-way.
     * <p>
     * {@code distSqBounded} only tests its bound on every 8th element, so with 8 or fewer features
     * it always runs to completion and returns the same value the unbounded version would — which
     * is exactly what the three fixtures above do. Nothing then covers the branch that returns a
     * <em>partial</em> sum, even though it is the one that fires on essentially every comparison
     * in production (a pruned panel is ~200 features wide).
     * <p>
     * 40 features with heavily overlapping classes puts a large majority of pairs above the
     * running best within the first 8 dimensions, so the early return is the common case here.
     */
    private static Fixture wide() {
        return build("wide", new int[] {160, 90, 50}, 40, 0.05f);
    }

    private static List<Fixture> fixtures() {
        return List.of(moderate(), duplicates(), severe(), wide());
    }

    private static final ResamplingStrategy[] STRATEGIES = {
        ResamplingStrategy.SMOTE,
        ResamplingStrategy.ADASYN,
        ResamplingStrategy.TOMEK,
        ResamplingStrategy.SMOTE_TOMEK,
        ResamplingStrategy.ADASYN_TOMEK
    };

    // ── Checksum ────────────────────────────────────────────────────────────────

    private static final long FNV_OFFSET = 0xcbf29ce484222325L;
    private static final long FNV_PRIME = 0x100000001b3L;

    /**
     * FNV-1a over the raw bits of every feature value followed by the label, row by row in
     * output order. Uses {@code floatToIntBits} (not {@code Float.hashCode}) so any change in
     * the last mantissa bit is caught, and so the value is stable across JVMs.
     */
    private static long checksum(List<float[]> rows, List<Integer> labels) {
        long h = FNV_OFFSET;
        for (int i = 0; i < rows.size(); i++) {
            float[] row = rows.get(i);
            for (float v : row) {
                int bits = Float.floatToIntBits(v);
                for (int b = 0; b < 4; b++) {
                    h = (h ^ ((bits >>> (b * 8)) & 0xFF)) * FNV_PRIME;
                }
            }
            int lbl = labels.get(i);
            for (int b = 0; b < 4; b++) {
                h = (h ^ ((lbl >>> (b * 8)) & 0xFF)) * FNV_PRIME;
            }
        }
        return h;
    }

    private static int[] counts(List<Integer> labels, int nClasses) {
        int[] c = new int[nClasses];
        for (int l : labels) c[l]++;
        return c;
    }

    /** Expected {@code size}, per-class counts, and content checksum for one fixture+strategy. */
    private record Golden(int size, int[] counts, long checksum) {}

    /**
     * Captured against the original brute-force implementation. A failure here means the
     * resampled output changed — investigate, do not re-baseline.
     */
    private static Golden golden(String fixture, ResamplingStrategy strategy) {
        return GOLDEN.get(fixture + "/" + strategy.name());
    }

    private static final java.util.Map<String, Golden> GOLDEN = new java.util.HashMap<>();

    static {
        put("moderate", ResamplingStrategy.SMOTE, 600, new int[] {200, 200, 200}, 0x43d184adba156998L);
        put("moderate", ResamplingStrategy.ADASYN, 596, new int[] {200, 196, 200}, 0x86bdeb62065d804bL);
        put("moderate", ResamplingStrategy.TOMEK, 262, new int[] {182, 60, 20}, 0x707b5cf0238a934aL);
        put("moderate", ResamplingStrategy.SMOTE_TOMEK, 510, new int[] {179, 155, 176}, 0x1870397ad00c8152L);
        put("moderate", ResamplingStrategy.ADASYN_TOMEK, 546, new int[] {176, 196, 174}, 0x1489c9158b0ecca1L);

        put("duplicates", ResamplingStrategy.SMOTE, 64, new int[] {32, 32}, 0x8dbcdcb3c6e8ea14L);
        put("duplicates", ResamplingStrategy.ADASYN, 64, new int[] {32, 32}, 0xa296e9f0c248b320L);
        put("duplicates", ResamplingStrategy.TOMEK, 32, new int[] {20, 12}, 0x2bb2a0ba6ea512d7L);
        put("duplicates", ResamplingStrategy.SMOTE_TOMEK, 38, new int[] {19, 19}, 0xc2ad45b8b1237dccL);
        put("duplicates", ResamplingStrategy.ADASYN_TOMEK, 38, new int[] {19, 19}, 0xe6640ed15e41563eL);

        put("severe", ResamplingStrategy.SMOTE, 1500, new int[] {500, 500, 500}, 0xb7367490fb55cb6fL);
        put("severe", ResamplingStrategy.ADASYN, 1500, new int[] {500, 500, 500}, 0x2c4fcc379f5a1c73L);
        put("severe", ResamplingStrategy.TOMEK, 537, new int[] {492, 5, 40}, 0xdcaf60fb9fea8af9L);
        put("severe", ResamplingStrategy.SMOTE_TOMEK, 1386, new int[] {471, 445, 470}, 0xedfe96f95c24f41eL);
        put("severe", ResamplingStrategy.ADASYN_TOMEK, 1358, new int[] {469, 436, 453}, 0x6073edbfc08178fL);

        put("wide", ResamplingStrategy.SMOTE, 480, new int[] {160, 160, 160}, 0x867f8bf34b60d028L);
        put("wide", ResamplingStrategy.ADASYN, 480, new int[] {160, 160, 160}, 0x69ed756f1e871b7aL);
        put("wide", ResamplingStrategy.TOMEK, 272, new int[] {132, 90, 50}, 0x35ccc88efa44ce7bL);
        put("wide", ResamplingStrategy.SMOTE_TOMEK, 358, new int[] {132, 99, 127}, 0xc386626ed1c41194L);
        put("wide", ResamplingStrategy.ADASYN_TOMEK, 358, new int[] {131, 99, 128}, 0x6609afc74a0d575bL);
    }

    private static void put(String fixture, ResamplingStrategy s, int size, int[] counts, long checksum) {
        GOLDEN.put(fixture + "/" + s.name(), new Golden(size, counts, checksum));
    }

    // ── Capture helper ──────────────────────────────────────────────────────────

    /**
     * Prints the current output signature for every fixture/strategy pair. A tool, not a test: it
     * asserts nothing and so can never fail, which is why it stays disabled rather than printing
     * on every build.
     * <p>
     * Enable it only when a <em>fixture</em> changes, and run it against an implementation known
     * to be correct — the existing constants were captured by pointing this at the pre-optimisation
     * brute-force {@code Resampler}, which is the only thing that makes them an oracle. Never run
     * it against a modified {@code Resampler} to paper over a behavioural diff.
     */
    @Disabled("capture tool — enable only to re-baseline after a fixture change; see javadoc")
    @Test
    @DisplayName("capture: print output signatures for all fixture/strategy pairs")
    void captureSignatures() {
        StringBuilder sb = new StringBuilder("\n=== Resampler golden signatures ===\n");
        for (Fixture base : fixtures()) {
            for (ResamplingStrategy strategy : STRATEGIES) {
                Fixture f = base.copy();
                Resampler.Result r = Resampler.apply(f.rows(), f.labels(), f.nClasses(), strategy, null);
                int[] c = counts(r.labels(), f.nClasses());
                sb.append(String.format(
                        "put(\"%s\", ResamplingStrategy.%s, %d, new int[] {%s}, %sL);%n",
                        f.name(),
                        strategy.name(),
                        r.rows().size(),
                        java.util.Arrays.stream(c)
                                .mapToObj(Integer::toString)
                                .reduce((a, b) -> a + ", " + b)
                                .orElse(""),
                        "0x" + Long.toHexString(checksum(r.rows(), r.labels()))));
            }
        }
        System.out.println(sb);
    }

    // ── Assertions ──────────────────────────────────────────────────────────────

    @Test
    @DisplayName("output size, class distribution, and content are bit-identical to the baseline")
    void outputIsBitIdenticalToBaseline() {
        for (Fixture base : fixtures()) {
            for (ResamplingStrategy strategy : STRATEGIES) {
                Golden expected = golden(base.name(), strategy);
                if (expected == null) continue; // not yet captured
                Fixture f = base.copy();
                Resampler.Result r = Resampler.apply(f.rows(), f.labels(), f.nClasses(), strategy, null);
                String where = base.name() + "/" + strategy;

                assertEquals(expected.size(), r.rows().size(), where + ": output row count");
                assertEquals(r.rows().size(), r.labels().size(), where + ": rows and labels parallel");
                assertArrayEquals(expected.counts(), counts(r.labels(), f.nClasses()), where + ": per-class counts");
                assertEquals(
                        expected.checksum(),
                        checksum(r.rows(), r.labels()),
                        where + ": content checksum — resampled output changed");
            }
        }
    }

    @Test
    @DisplayName("every strategy leaves the caller's lists untouched")
    void inputListsAreNeverMutated() {
        for (Fixture base : fixtures()) {
            for (ResamplingStrategy strategy : ResamplingStrategy.values()) {
                Fixture f = base.copy();
                int sizeBefore = f.rows().size();
                long contentBefore = checksum(f.rows(), f.labels());

                Resampler.apply(f.rows(), f.labels(), f.nClasses(), strategy, null);

                String where = base.name() + "/" + strategy;
                assertEquals(sizeBefore, f.rows().size(), where + ": input rows resized");
                assertEquals(sizeBefore, f.labels().size(), where + ": input labels resized");
                assertEquals(contentBefore, checksum(f.rows(), f.labels()), where + ": input content mutated");
            }
        }
    }

    /**
     * Large enough that the resampler takes its parallel path: the post-SMOTE Tomek search and
     * the class-1 neighbour precomputation both exceed the internal work threshold below which
     * splitting is not worth the hand-off. The smaller golden fixtures run single-threaded no
     * matter how many cores are present, so they cannot cover this on their own.
     */
    private static Fixture parallelSized() {
        return build("parallel", new int[] {1200, 600, 300}, 10, 0.20f);
    }

    @Test
    @DisplayName("output does not depend on the thread budget")
    void outputIsInvariantToThreadCount() {
        int original = TrainingThreads.getOverride();
        try {
            Fixture base = parallelSized();
            // setOverride clamps to availableProcessors(), so the top of this range collapses to
            // the core count on small machines — the run still covers serial vs parallel.
            int maxThreads = Math.min(8, Runtime.getRuntime().availableProcessors());

            for (ResamplingStrategy strategy : STRATEGIES) {
                Long reference = null;
                for (int threads = 1; threads <= Math.max(2, maxThreads); threads++) {
                    TrainingThreads.setOverride(threads);
                    Fixture f = base.copy();
                    Resampler.Result r = Resampler.apply(f.rows(), f.labels(), f.nClasses(), strategy, null);
                    long sum = checksum(r.rows(), r.labels());
                    if (reference == null) {
                        reference = sum;
                    } else {
                        assertEquals(
                                reference.longValue(),
                                sum,
                                strategy + ": output changed at " + threads + " thread(s) — the parallel"
                                        + " search is not order-independent");
                    }
                }
            }
        } finally {
            TrainingThreads.setOverride(original);
        }
    }

    @Test
    @DisplayName("repeated parallel runs agree (no data race in the shared search)")
    void parallelRunsAreStable() {
        int original = TrainingThreads.getOverride();
        try {
            TrainingThreads.setOverride(0); // auto — use every core
            Fixture base = parallelSized();
            for (ResamplingStrategy strategy : STRATEGIES) {
                long reference = 0;
                for (int run = 0; run < 5; run++) {
                    Fixture f = base.copy();
                    Resampler.Result r = Resampler.apply(f.rows(), f.labels(), f.nClasses(), strategy, null);
                    long sum = checksum(r.rows(), r.labels());
                    if (run == 0) {
                        reference = sum;
                    } else {
                        assertEquals(reference, sum, strategy + ": run " + run + " disagreed with run 0");
                    }
                }
            }
        } finally {
            TrainingThreads.setOverride(original);
        }
    }

    @Test
    @DisplayName("repeated calls on equal input produce identical output (seeded determinism)")
    void repeatedCallsAreDeterministic() {
        for (Fixture base : fixtures()) {
            for (ResamplingStrategy strategy : STRATEGIES) {
                Fixture a = base.copy();
                Fixture b = base.copy();
                Resampler.Result ra = Resampler.apply(a.rows(), a.labels(), a.nClasses(), strategy, null);
                Resampler.Result rb = Resampler.apply(b.rows(), b.labels(), b.nClasses(), strategy, null);
                assertEquals(
                        checksum(ra.rows(), ra.labels()),
                        checksum(rb.rows(), rb.labels()),
                        base.name() + "/" + strategy + ": two runs disagree");
            }
        }
    }
}
