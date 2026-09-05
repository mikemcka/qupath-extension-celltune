package qupath.ext.spclassify.ui;

import java.awt.image.BufferedImage;
import java.util.concurrent.ExecutorService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.util.BackgroundExecutors;
import qupath.lib.images.ImageData;
import qupath.lib.projects.ProjectImageEntry;

/**
 * Background prefetcher for the cross-image review flow. Warms QuPath's caches
 * (TIFF headers, detection hierarchy, tile cache) for the next review image on a
 * daemon thread, so switching to it is fast.
 * <p>
 * A strong reference to the prefetched {@link ImageData} is held until
 * {@link #release()} so the GC can't reclaim those caches before QuPath's
 * {@code openImageEntry} consumes them. Extracted from {@code ReviewController}
 * to keep the prefetch lifecycle (executor, dedup, strong reference) in one place.
 */
final class ImagePrefetcher {

    private static final Logger logger = LoggerFactory.getLogger(ImagePrefetcher.class);

    private final ExecutorService executor = BackgroundExecutors.newSingleThread("SpClassify-ReviewPrefetch");

    /** Image name most recently submitted for prefetch, to avoid duplicate work. */
    private volatile String lastPrefetchedImageName;

    /** Strong reference to the prefetched data, pinning its warm caches until consumed. */
    private volatile ImageData<BufferedImage> prefetchedImageData;

    /**
     * Submit a background read of {@code entry}'s image data, unless that image was
     * already the most recent prefetch target. Failures are non-fatal (prefetch is
     * purely an optimisation).
     */
    void prefetch(String imageName, ProjectImageEntry<BufferedImage> entry) {
        if (executor.isShutdown()) {
            return;
        }
        if (imageName.equals(lastPrefetchedImageName)) {
            return;
        }
        lastPrefetchedImageName = imageName;
        executor.submit(() -> {
            try {
                prefetchedImageData = entry.readImageData();
                logger.debug("Prefetched ImageData for next review image '{}'", imageName);
            } catch (Throwable t) {
                logger.debug("Prefetch of '{}' failed: {}", imageName, t.getMessage());
            }
        });
    }

    /**
     * Drop the strong reference to the prefetched data once the image has been
     * opened, so two large {@code ImageData} instances aren't pinned at once.
     */
    void release() {
        prefetchedImageData = null;
    }

    boolean isShutdown() {
        return executor.isShutdown();
    }

    /** Stop the prefetch thread (called at session end). */
    void shutdown() {
        executor.shutdownNow();
    }
}
