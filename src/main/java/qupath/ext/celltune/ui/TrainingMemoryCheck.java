package qupath.ext.celltune.ui;

import qupath.ext.celltune.classifier.ResamplingStrategy;
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

    private static final String TITLE = "CellTune";

    /** Bytes per {@code float}. */
    private static final long BYTES_PER_FLOAT = 4L;

    /**
     * Multiplier applied to the training matrix when resampling is on. Oversampling lifts every
     * class to the majority count; on a real 19-class panel (5,839 labelled cells, majority
     * 1,348) that came to 25,612 rows, ~4.4x. Rounded up for headroom.
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
     *
     * @param nCells     detections that will be predicted
     * @param nFeatures  feature columns after selection and pruning
     * @param nLabelled  labelled cells that will form the training matrix
     * @param strategy   resampling strategy, which inflates the training matrix
     * @return {@code true} to proceed — either because there is room, or because the user chose
     *         to continue anyway
     */
    static boolean confirmEnoughHeap(int nCells, int nFeatures, int nLabelled, ResamplingStrategy strategy) {
        long maxHeapBytes = Runtime.getRuntime().maxMemory();
        double maxHeapGiB = toGiB(maxHeapBytes);

        double predictGiB = toGiB((long) nCells * nFeatures * BYTES_PER_FLOAT);

        double trainRows = nLabelled;
        if (strategy != null && strategy != ResamplingStrategy.NONE) {
            trainRows *= RESAMPLING_INFLATION;
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
                                ? String.format(" (%,d labelled cells, inflated by %s)", nLabelled, strategy)
                                : String.format(" (%,d labelled cells)", nLabelled),
                        estimatedPeakGiB,
                        maxHeapGiB));
    }

    private static double toGiB(long bytes) {
        return bytes / (1024.0 * 1024.0 * 1024.0);
    }
}
