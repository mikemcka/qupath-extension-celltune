package qupath.ext.celltune.util;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.Consumer;

/**
 * Records how long each stage of a long-running operation took, and how much heap was in use at
 * each stage boundary.
 * <p>
 * Training is a sequence of expensive, structurally different steps (resampling, early stopping,
 * metrics, two model fits, prediction). Without per-stage numbers "training is slow" cannot be
 * acted on — the fix for slow resampling and the fix for slow prediction are unrelated. This
 * emits a line as each stage ends plus a descending summary at the end, so the dominant cost is
 * visible in the training log rather than inferred.
 * <p>
 * The recorded {@linkplain #currentPhase() current phase} also gives an
 * {@link OutOfMemoryError} handler something specific to report, instead of "training failed".
 * <p>
 * Not thread-safe: intended to be driven from a single worker thread.
 */
public final class PhaseTimer {

    /** One completed stage. */
    public record Phase(String name, long millis, long heapUsedBytes) {}

    private static final long MIB = 1024L * 1024L;

    private final Consumer<String> out;
    private final List<Phase> completed = new ArrayList<>();
    private final long maxHeapBytes = Runtime.getRuntime().maxMemory();
    private final long startedAtNanos = System.nanoTime();

    private String current = null;
    private long currentStartNanos = 0L;
    private long peakHeapBytes = 0L;

    /**
     * @param log where per-phase lines are written; may be {@code null} for silent collection
     */
    public PhaseTimer(Consumer<String> log) {
        this.out = log != null ? log : s -> {};
        sampleHeap();
    }

    /**
     * Ends the current phase (if any) and begins a new one.
     *
     * @param name short stage label, e.g. {@code "resample (full)"}
     */
    public void start(String name) {
        stop();
        current = name;
        currentStartNanos = System.nanoTime();
    }

    /** Ends the current phase, emitting its timing line. No-op when no phase is running. */
    public void stop() {
        if (current == null) {
            return;
        }
        long millis = (System.nanoTime() - currentStartNanos) / 1_000_000L;
        long heap = sampleHeap();
        completed.add(new Phase(current, millis, heap));
        out.accept(String.format(
                "  [%-24s] %8.2fs   heap %,d/%,d MB", current, millis / 1000.0, heap / MIB, maxHeapBytes / MIB));
        current = null;
    }

    /** @return the phase currently running, or {@code null} — useful when reporting a failure */
    public String currentPhase() {
        return current;
    }

    /** @return the highest heap usage seen at any stage boundary, in bytes */
    public long peakHeapBytes() {
        return peakHeapBytes;
    }

    /** @return total elapsed time since this timer was created, in milliseconds */
    public long totalMillis() {
        return (System.nanoTime() - startedAtNanos) / 1_000_000L;
    }

    private long sampleHeap() {
        Runtime rt = Runtime.getRuntime();
        long used = rt.totalMemory() - rt.freeMemory();
        if (used > peakHeapBytes) {
            peakHeapBytes = used;
        }
        return used;
    }

    /**
     * Emits a descending-by-cost summary of every completed phase, with each one's share of the
     * total. Ends any phase still running first.
     */
    public void writeSummary() {
        stop();
        if (completed.isEmpty()) {
            return;
        }
        long sum = 0;
        for (Phase p : completed) sum += p.millis();
        final long total = sum;
        long wall = totalMillis();

        out.accept("");
        out.accept("── Where the time went ──────────────────────────────────");
        completed.stream()
                .sorted(Comparator.comparingLong(Phase::millis).reversed())
                .forEach(p -> out.accept(String.format(
                        "  %-24s %8.2fs  %5.1f%%",
                        p.name(), p.millis() / 1000.0, total == 0 ? 0.0 : 100.0 * p.millis() / total)));
        out.accept(String.format("  %-24s %8.2fs", "TOTAL (measured)", total / 1000.0));
        out.accept(String.format("  %-24s %8.2fs", "TOTAL (wall clock)", wall / 1000.0));
        out.accept(
                String.format("  peak heap                %,d MB of %,d MB", peakHeapBytes / MIB, maxHeapBytes / MIB));
        out.accept("─────────────────────────────────────────────────────────");
    }

    /** @return an unmodifiable view of the completed phases, in execution order */
    public List<Phase> phases() {
        return List.copyOf(completed);
    }
}
