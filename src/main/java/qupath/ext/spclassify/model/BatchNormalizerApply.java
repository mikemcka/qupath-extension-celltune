package qupath.ext.spclassify.model;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.function.DoubleConsumer;
import javafx.application.Platform;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.model.CohortClusterModel.CancellationToken;
import qupath.ext.spclassify.util.BackgroundExecutors;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.projects.ProjectImageEntry;

/**
 * Optional APPLY pass for UniFORM batch-normalization: materializes the learned
 * per-{@code (image, marker)} scales as new numeric measurements
 * {@code "<marker>: Cell: Mean (batchnorm)"} on every cell, then saves each image.
 * Non-destructive — the raw {@code "<marker>: Cell: Mean"} column is untouched, and any
 * measurement not in the fit (foundation-model embeddings, morphology, …) is left alone.
 *
 * <p>Mirrors {@code NeighborhoodCohort.assignAcrossProject}: one worker per image via
 * {@link BackgroundExecutors}, {@code entry.equals(openEntry)} identity match to reuse the
 * open image, FX-thread marshalling for the open image's hierarchy mutation, a single
 * {@code fireHierarchyChangedEvent} + {@code saveImageData} per image, {@link CancellationToken},
 * and a {@code DoubleConsumer} progress sink. Re-running overwrites the corrected columns
 * in place (idempotent by name).
 */
public final class BatchNormalizerApply {

    private static final Logger logger = LoggerFactory.getLogger(BatchNormalizerApply.class);

    /** Suffix appended to a marker feature name to form its batch-corrected column. */
    public static final String CORRECTED_SUFFIX = " (batchnorm)";

    private BatchNormalizerApply() {}

    /** Outcome of an apply run. */
    public record ApplyResult(int imagesWritten, long cellsWritten) {}

    /** The corrected-measurement name for a raw marker feature name. */
    public static String correctedName(String markerFeature) {
        return markerFeature + CORRECTED_SUFFIX;
    }

    /**
     * Write corrected columns for every selected marker across {@code entries}. Uses the
     * persisted {@link BatchShifts} (per-image scales); an image absent from the fit, or a
     * marker with no scale, gets a no-op factor of {@code 1.0}.
     */
    public static ApplyResult apply(
            List<ProjectImageEntry<BufferedImage>> entries,
            BatchShifts shifts,
            ImageData<BufferedImage> openData,
            ProjectImageEntry<BufferedImage> openEntry,
            int workers,
            Consumer<String> log,
            DoubleConsumer progress,
            CancellationToken token) {

        int nImages = entries.size();
        List<String> markers = shifts.markers;
        if (nImages == 0 || markers == null || markers.isEmpty()) {
            return new ApplyResult(0, 0);
        }
        int parallelism = Math.max(1, Math.min(workers, nImages));
        ExecutorService pool = BackgroundExecutors.newFixedPool(parallelism, "SpClassify-BatchNorm-Apply");
        AtomicInteger done = new AtomicInteger(0);
        AtomicInteger imagesWritten = new AtomicInteger(0);
        AtomicLong cellsWritten = new AtomicLong(0);
        try {
            List<Future<?>> futures = new ArrayList<>();
            for (ProjectImageEntry<BufferedImage> entry : entries) {
                futures.add(pool.submit(() -> {
                    try {
                        if (token != null && token.isCancelled()) {
                            return;
                        }
                        long n = applyOneImage(entry, shifts, markers, openData, openEntry, log);
                        if (n >= 0) {
                            imagesWritten.incrementAndGet();
                            cellsWritten.addAndGet(n);
                        }
                    } finally {
                        progress.accept(done.incrementAndGet() / (double) nImages);
                    }
                }));
            }
            for (Future<?> f : futures) {
                try {
                    f.get();
                } catch (Exception e) {
                    // per-image failures already logged in the task
                }
            }
        } finally {
            pool.shutdownNow();
        }
        return new ApplyResult(imagesWritten.get(), cellsWritten.get());
    }

    /** Write corrected columns for one image and save; returns cells written or {@code -1} on skip. */
    private static long applyOneImage(
            ProjectImageEntry<BufferedImage> entry,
            BatchShifts shifts,
            List<String> markers,
            ImageData<BufferedImage> openData,
            ProjectImageEntry<BufferedImage> openEntry,
            Consumer<String> log) {
        if (entry == null) {
            return -1;
        }
        String name = entry.getImageName();
        boolean isOpen = openEntry != null && entry.equals(openEntry) && openData != null;
        ImageData<BufferedImage> imageData;
        try {
            imageData = isOpen ? openData : entry.readImageData();
        } catch (Exception e) {
            log.accept("[" + name + "] could not load — skipped");
            return -1;
        }
        if (imageData == null) {
            return -1;
        }
        var hierarchy = imageData.getHierarchy();
        var cellCol = hierarchy.getCellObjects();
        List<PathObject> cells = new ArrayList<>(cellCol.isEmpty() ? hierarchy.getDetectionObjects() : cellCol);
        if (cells.isEmpty()) {
            log.accept("[" + name + "] no cells — skipped");
            return -1;
        }

        int nMarkers = markers.size();
        double[] scale = new double[nMarkers];
        String[] outName = new String[nMarkers];
        for (int m = 0; m < nMarkers; m++) {
            scale[m] = shifts.scaleFor(name, markers.get(m));
            outName[m] = correctedName(markers.get(m));
        }

        Runnable apply = () -> {
            for (PathObject cell : cells) {
                var ml = cell.getMeasurementList();
                for (int m = 0; m < nMarkers; m++) {
                    double raw = ml.get(markers.get(m));
                    ml.put(outName[m], raw * scale[m]);
                }
            }
            hierarchy.fireHierarchyChangedEvent(BatchNormalizerApply.class);
        };
        if (isOpen && !Platform.isFxApplicationThread()) {
            var latch = new CountDownLatch(1);
            Platform.runLater(() -> {
                try {
                    apply.run();
                } finally {
                    latch.countDown();
                }
            });
            try {
                latch.await();
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
        } else {
            apply.run();
        }

        try {
            entry.saveImageData(imageData);
        } catch (Exception e) {
            logger.error("Failed to save {}", name, e);
            log.accept("[" + name + "] save failed: " + e.getMessage());
        }
        log.accept(String.format("[%s] wrote %d corrected column(s) to %,d cells", name, nMarkers, cells.size()));
        return cells.size();
    }
}
