package qupath.ext.spclassify.ui;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.projects.ProjectImageEntry;

/**
 * Cleans up the temporary project entries QuPath's ProjectBrowser auto-adds when
 * tile-mode review swaps {@code CroppedImageServer}-backed {@code ImageData} into
 * the viewer (entries shaped like {@code "crop_1.ome.tiff (x, y, w, h)"}).
 * <p>
 * A snapshot of the project's entries is taken at construction (i.e. at review-session
 * start, before any tile is shown); anything present later that is not in the snapshot
 * is a tile-derived auto-added entry and is removed, along with its on-disk data folder.
 * Extracted verbatim from {@code ReviewController}.
 */
final class TileEntryCleaner {

    private static final Logger logger = LoggerFactory.getLogger(TileEntryCleaner.class);

    private final QuPathGUI qupath;

    /**
     * Snapshot of project entry references taken when the review session started.
     * Identity-based to avoid relying on names.
     */
    private final Set<ProjectImageEntry<BufferedImage>> initialSnapshot =
            Collections.newSetFromMap(new IdentityHashMap<>());

    /**
     * Capture the baseline set of project entries. Construct at review-session start,
     * before touching the viewer, so later additions can be identified as tile artifacts.
     */
    TileEntryCleaner(QuPathGUI qupath) {
        this.qupath = qupath;
        if (qupath.getProject() != null) {
            @SuppressWarnings("unchecked")
            var allEntries = (List<ProjectImageEntry<BufferedImage>>)
                    (List<?>) qupath.getProject().getImageList();
            initialSnapshot.addAll(allEntries);
        }
    }

    /**
     * Remove any project entries that QuPath added since the snapshot. Compares against
     * the entries captured at construction; anything new must be a tile-derived
     * auto-added entry. The entry backing the currently-displayed image is never removed.
     */
    void removeAutoAddedEntries() {
        var project = qupath.getProject();
        if (project == null) return;
        try {
            // Don't try to remove the entry that backs the currently-displayed
            // ImageData — QuPath will recreate it (or worse, throw) if we yank
            // it out from under the viewer.
            ProjectImageEntry<BufferedImage> activeEntry = null;
            try {
                var activeData = qupath.getImageData();
                if (activeData != null) activeEntry = project.getEntry(activeData);
            } catch (Exception ignored) {
            }

            @SuppressWarnings("unchecked")
            var current = (List<ProjectImageEntry<BufferedImage>>) (List<?>) project.getImageList();
            List<ProjectImageEntry<BufferedImage>> toRemove = new ArrayList<>();
            for (var entry : current) {
                if (entry == null) continue;
                if (entry == activeEntry) continue;
                if (!initialSnapshot.contains(entry)) {
                    toRemove.add(entry);
                }
            }
            if (toRemove.isEmpty()) return;
            for (var entry : toRemove) {
                // Capture the on-disk path before removal so we can force-delete
                // it if QuPath's move-to-trash fallback fails (common on network
                // mounts where no system trash is available).
                java.nio.file.Path entryPath = null;
                try {
                    entryPath = entry.getEntryPath();
                } catch (Exception ignored) {
                }

                try {
                    project.removeImage(entry, true);
                    logger.info("Removed auto-added tile project entry: {}", entry.getImageName());
                } catch (Exception ex) {
                    logger.warn(
                            "Failed to remove auto-added tile entry '{}': {}", entry.getImageName(), ex.getMessage());
                }

                // Force-delete the data folder if QuPath left it behind.
                if (entryPath != null && java.nio.file.Files.exists(entryPath)) {
                    try {
                        deleteRecursively(entryPath);
                    } catch (Exception ex) {
                        logger.warn("Failed to force-delete tile entry path '{}': {}", entryPath, ex.getMessage());
                    }
                }
            }
            try {
                project.syncChanges();
            } catch (Exception ex) {
                logger.warn("Failed to sync project after removing tile entries: {}", ex.getMessage());
            }
            // Refresh the project browser so removed entries disappear from the UI.
            javafx.application.Platform.runLater(() -> {
                try {
                    qupath.refreshProject();
                } catch (Exception ignored) {
                }
            });
        } catch (Exception ex) {
            logger.warn("Failed to scan project for auto-added tile entries: {}", ex.getMessage());
        }
    }

    private static void deleteRecursively(java.nio.file.Path path) throws java.io.IOException {
        if (!java.nio.file.Files.exists(path)) return;
        try (var stream = java.nio.file.Files.walk(path)) {
            stream.sorted(java.util.Comparator.reverseOrder()).forEach(p -> {
                try {
                    java.nio.file.Files.deleteIfExists(p);
                } catch (java.io.IOException ex) {
                    logger.debug("Could not delete {}: {}", p, ex.getMessage());
                }
            });
        }
    }
}
