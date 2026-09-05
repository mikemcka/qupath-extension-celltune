package qupath.ext.spclassify.ui;

import java.util.Collection;
import qupath.ext.spclassify.classifier.ResamplingStrategy;
import qupath.fx.dialogs.Dialogs;

/**
 * Pre-flight heap estimate for a training run.
 * <p>
 * Training allocates several large {@code float[]} matrices and then hands each to a native
 * library that makes its own copy, so an {@link OutOfMemoryError} is a real outcome on a big
 * image with a wide marker panel. Failing at that point is expensive — the run has already been
 * going for minutes, and the error surfaces as a dead progress dialog rather than an explanation.
 * This checks up front, while the user can still reduce the feature set or raise the heap.
 * <p>
 * The estimate is deliberately rough. It exists to catch "this cannot possibly fit", not to
 * predict usage precisely, and it warns rather than blocks.
 */
final class TrainingMemoryCheck {

    private static final String TITLE = "SP Classify";

    /** Bytes per {@code float}. */
    private static final long BYTES_PER_FLOAT = 4L;

    /**
     * Fallback multiplier for the resampled row count, used only when the per-class counts are not
     * available. Oversampling lifts every class to the majority count; on a real 19-class panel
     * (5,839 labelled cells, majority 1,348) that came to 25,612 rows, ~4.4x. Rounded up for
     * headroom. When the counts <em>are</em> known the exact figure is used instead — the true
     * factor swings from ~1x on a balanced set to well over 10x on a badly skewed one, which a
     * single constant cannot express.
     */
    private static final double RESAMPLING_INFLATION = 5.0;

    /**
     * Both models copy their training matrix into native memory, and the round-search fold adds
     * another. Charged against the Java heap because that is what the JVM limit governs and
     * native allocation failures surface the same way to the user.
     */
    private static final double NATIVE_COPY_FACTOR = 3.0;

    /** Fixed overhead for predictions, population sets and the object graph. */
    private static final double FIXED_OVERHEAD_GIB = 0.3;

    /** Warn once the estimate passes this fraction of the maximum heap. */
    private static final double WARN_FRACTION = 0.8;

    private TrainingMemoryCheck() {} // utility class

    /**
     * Estimates peak heap for a run and, if it looks tight, asks the user whether to continue.
     * <p>
     * Counts what is knowable at the point of asking: the labels on this image. Pooling and
     * imported rows are added later on the background thread, so a pooled run trains on more than
     * this. That makes the estimate a floor, which is the right direction for a warning that is
     * meant to catch "this cannot possibly fit".
     *
     * @param nCells      detections that will be predicted
     * @param nFeatures   feature columns after selection and pruning
     * @param classCounts labelled cells per class; may be {@code null} or empty
     * @param strategy    resampling strategy, which inflates the training matrix
     * @return {@code true} to proceed — either because there is room, or because the user chose
     *         to continue anyway
     */
    static boolean confirmEnoughHeap(
            int nCells, int nFeatures, Collection<Long> classCounts, ResamplingStrategy strategy) {
        long maxHeapBytes = Runtime.getRuntime().maxMemory();
        double maxHeapGiB = toGiB(maxHeapBytes);

        double predictGiB = toGiB((long) nCells * nFeatures * BYTES_PER_FLOAT);

        long nLabelled = 0;
        long majority = 0;
        int nClasses = 0;
        if (classCounts != null) {
            for (Long c : classCounts) {
                if (c == null) continue;
                nLabelled += c;
                majority = Math.max(majority, c);
                nClasses++;
            }
        }

        double trainRows = nLabelled;
        if (strategy != null && strategy != ResamplingStrategy.NONE) {
            // Every oversampling strategy lifts each class to the majority count, so the post-
            // resampling row count is exactly nClasses x majority. Tomek only ever removes, so
            // that is an upper bound for the combined strategies too.
            trainRows = nClasses > 0 ? (double) nClasses * majority : nLabelled * RESAMPLING_INFLATION;
        }
        double trainGiB = toGiB((long) (trainRows * nFeatures * BYTES_PER_FLOAT));

        double estimatedPeakGiB = predictGiB + trainGiB * NATIVE_COPY_FACTOR + FIXED_OVERHEAD_GIB;

        if (estimatedPeakGiB <= maxHeapGiB * WARN_FRACTION) {
            return true;
        }

        return Dialogs.showConfirmDialog(
                TITLE,
                String.format(
                        "Memory warning: %,d cells × %,d features"
                                + "%s needs an estimated %.1f GB, but the JVM heap is only %.1f GB.%n%n"
                                + "Training may fail with an OutOfMemoryError.%n%n"
                                + "To give it more memory: Edit → Preferences → 'Maximum memory',"
                                + " then restart QuPath.%n"
                                + "To need less: select fewer features, or set Balancing to None.%n%n"
                                + "Proceed anyway?",
                        nCells,
                        nFeatures,
                        (strategy != null && strategy != ResamplingStrategy.NONE)
                                ? String.format(
                                        " (%,d labelled cells → %,d after %s)", nLabelled, (long) trainRows, strategy)
                                : String.format(" (%,d labelled cells)", nLabelled),
                        estimatedPeakGiB,
                        maxHeapGiB));
    }

    private static double toGiB(long bytes) {
        return bytes / (1024.0 * 1024.0 * 1024.0);
    }
}
