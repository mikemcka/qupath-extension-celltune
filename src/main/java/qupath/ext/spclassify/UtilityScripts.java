package qupath.ext.spclassify;

import com.google.gson.JsonElement;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.ResourceBundle;
import java.util.zip.GZIPInputStream;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.scene.control.ButtonType;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Dialog;
import javafx.scene.control.Hyperlink;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.tools.GuiTools;
import qupath.lib.io.GsonTools;
import qupath.lib.objects.PathAnnotationObject;
import qupath.lib.objects.PathDetectionObject;
import qupath.lib.objects.PathObject;
import qupath.lib.projects.ProjectImageEntry;

/**
 * Standalone utility scripts surfaced under the Extensions ‣ SP Classify ‣ Utility
 * Scripts menu. These are ad-hoc, image- or project-level helpers (cell filtering,
 * hierarchy resolution, annotation locking, measurement deletion, GeoJSON import)
 * that operate purely through the supplied {@link QuPathGUI} — they hold no extension
 * state. Extracted from SpClassifyExtension to keep that entry-point class focused.
 * <p>
 * Hierarchy-change events are fired with a private {@link #EVENT_SOURCE} sentinel
 * rather than the extension instance; no listener in the extension filters by event
 * source, matching the pattern already used by CohortClusterModel.
 */
final class UtilityScripts {

    private static final ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.spclassify.ui.strings");
    private static final String EXTENSION_NAME = resources.getString("name");
    private static final Logger logger = LoggerFactory.getLogger(UtilityScripts.class);
    /** Stable non-null source for hierarchy-change events fired by these scripts. */
    private static final Object EVENT_SOURCE = new Object();

    private UtilityScripts() {} // utility class

    /**
     * Remove cell detections that are likely mis-segmented or artefacts, based on
     * optional min/max bounds for area and circularity. The thresholds are entered
     * at run time (any field may be left blank for no bound); cells missing either
     * measurement are skipped (not removed).
     */
    static void filterCellsBySizeAndCircularity(QuPathGUI qupath) {
        var imageData = qupath.getImageData();
        if (imageData == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No image is currently open.");
            return;
        }
        var hierarchy = imageData.getHierarchy();

        List<PathObject> cells = new ArrayList<>();
        for (PathObject det : hierarchy.getDetectionObjects()) {
            if (det.isCell()) cells.add(det);
        }
        if (cells.isEmpty()) {
            Dialogs.showInfoNotification(EXTENSION_NAME, "No cells found on the current image.");
            return;
        }

        // Build the threshold dialog: min/max for both area and circularity.
        TextField minAreaField = new TextField();
        minAreaField.setPromptText("none");
        TextField maxAreaField = new TextField("500.0");
        TextField minCircField = new TextField("0.7");
        TextField maxCircField = new TextField();
        maxCircField.setPromptText("none");

        GridPane grid = new GridPane();
        grid.setHgap(8);
        grid.setVgap(8);
        grid.setPadding(new Insets(10));
        grid.add(new Label("Min"), 1, 0);
        grid.add(new Label("Max"), 2, 0);
        grid.add(new Label("Cell area (\u00b5m\u00b2)"), 0, 1);
        grid.add(minAreaField, 1, 1);
        grid.add(maxAreaField, 2, 1);
        grid.add(new Label("Circularity (0\u20131)"), 0, 2);
        grid.add(minCircField, 1, 2);
        grid.add(maxCircField, 2, 2);

        Dialog<ButtonType> dialog = new Dialog<>();
        dialog.setTitle(EXTENSION_NAME);
        dialog.setHeaderText("Remove cells outside the specified ranges.\nLeave a field blank for no bound.");
        dialog.getDialogPane().setContent(grid);
        dialog.getDialogPane().getButtonTypes().addAll(ButtonType.OK, ButtonType.CANCEL);
        var choice = dialog.showAndWait();
        if (choice.isEmpty() || choice.get() != ButtonType.OK) return;

        Double minArea, maxArea, minCirc, maxCirc;
        try {
            minArea = parseOptionalDouble(minAreaField.getText());
            maxArea = parseOptionalDouble(maxAreaField.getText());
            minCirc = parseOptionalDouble(minCircField.getText());
            maxCirc = parseOptionalDouble(maxCircField.getText());
        } catch (NumberFormatException e) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "Invalid number: " + e.getMessage());
            return;
        }

        if (minArea == null && maxArea == null && minCirc == null && maxCirc == null) {
            Dialogs.showInfoNotification(EXTENSION_NAME, "No thresholds specified; nothing to filter.");
            return;
        }

        List<String> criteria = new ArrayList<>();
        if (minArea != null) criteria.add(String.format("area < %.1f", minArea));
        if (maxArea != null) criteria.add(String.format("area > %.1f", maxArea));
        if (minCirc != null) criteria.add(String.format("circularity < %.2f", minCirc));
        if (maxCirc != null) criteria.add(String.format("circularity > %.2f", maxCirc));
        String criteriaText = String.join(" or ", criteria);

        int missing = 0;
        List<PathObject> toRemove = new ArrayList<>();
        for (PathObject cell : cells) {
            double area = measurementContaining(cell, "area");
            double circ = measurementContaining(cell, "circularity");
            if (Double.isNaN(area) || Double.isNaN(circ)) {
                missing++;
                continue;
            }
            boolean remove = (minArea != null && area < minArea)
                    || (maxArea != null && area > maxArea)
                    || (minCirc != null && circ < minCirc)
                    || (maxCirc != null && circ > maxCirc);
            if (remove) {
                toRemove.add(cell);
            }
        }

        if (toRemove.isEmpty()) {
            Dialogs.showInfoNotification(EXTENSION_NAME, "No cells matched the filter (" + criteriaText + ").");
            return;
        }

        boolean confirm = Dialogs.showConfirmDialog(
                EXTENSION_NAME,
                String.format(
                        "Remove %d of %d cells (%s)?%s",
                        toRemove.size(),
                        cells.size(),
                        criteriaText,
                        missing > 0
                                ? "\n\n" + missing + " cell(s) skipped (missing area/circularity measurements)."
                                : ""));
        if (!confirm) return;

        hierarchy.removeObjects(toRemove, true);
        hierarchy.fireHierarchyChangedEvent(EVENT_SOURCE);

        long remaining = hierarchy.getDetectionObjects().stream()
                .filter(PathObject::isCell)
                .count();
        Dialogs.showInfoNotification(
                EXTENSION_NAME, "Removed " + toRemove.size() + " cell(s). " + remaining + " remaining.");
        logger.info(
                "[SP Classify] Size/circularity filter removed {} of {} cells ({}, skipped={}).",
                toRemove.size(),
                cells.size(),
                criteriaText,
                missing);
    }

    /**
     * Return the value of the first measurement whose name contains {@code substring}
     * (case-insensitive), or {@link Double#NaN} if none match.
     */
    private static double measurementContaining(PathObject cell, String substring) {
        var ml = cell.getMeasurementList();
        String needle = substring.toLowerCase(java.util.Locale.ROOT);
        for (String name : ml.getMeasurementNames()) {
            if (name.toLowerCase(java.util.Locale.ROOT).contains(needle)) {
                return ml.get(name);
            }
        }
        return Double.NaN;
    }

    /**
     * Parse a numeric bound, returning {@code null} for a blank value (no bound)
     * and throwing {@link NumberFormatException} for an invalid one.
     */
    private static Double parseOptionalDouble(String s) {
        if (s == null || s.isBlank()) return null;
        return Double.parseDouble(s.strip());
    }

    /**
     * Resolve the object hierarchy (parent/child relationships from ROI containment),
     * equivalent to the {@code resolveHierarchy()} scripting call. Offers a choice
     * between the current image and every image in the project; project-wide runs on
     * a background thread and saves each entry.
     */
    static void resolveHierarchy(QuPathGUI qupath) {
        var imageData = qupath.getImageData();
        var project = qupath.getProject();
        if (imageData == null && project == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No image or project is open.");
            return;
        }

        final String currentChoice = "Current image";
        final String projectChoice = "All project images";
        String choice;
        if (project == null) {
            choice = currentChoice;
        } else if (imageData == null) {
            choice = projectChoice;
        } else {
            choice = Dialogs.showChoiceDialog(
                    EXTENSION_NAME,
                    "Resolve the object hierarchy for:",
                    List.of(currentChoice, projectChoice),
                    currentChoice);
            if (choice == null) return;
        }

        // Current image only — resolve and refresh on the FX thread.
        if (choice.equals(currentChoice)) {
            var hierarchy = imageData.getHierarchy();
            hierarchy.resolveHierarchy();
            hierarchy.fireHierarchyChangedEvent(EVENT_SOURCE);
            Dialogs.showInfoNotification(EXTENSION_NAME, "Resolved hierarchy for the current image.");
            logger.info("[SP Classify] Resolved hierarchy for current image.");
            return;
        }

        // Project-wide.
        @SuppressWarnings("unchecked")
        var entries = (List<ProjectImageEntry<BufferedImage>>) (List<?>) project.getImageList();
        if (entries.isEmpty()) {
            Dialogs.showInfoNotification(EXTENSION_NAME, "No project images found.");
            return;
        }
        if (!Dialogs.showConfirmDialog(
                EXTENSION_NAME,
                "Resolve the hierarchy for all " + entries.size() + " project image(s) and save each?")) {
            return;
        }

        var currentEntry = imageData != null ? project.getEntry(imageData) : null;

        // Resolve the open image immediately so its view updates; the rest are
        // processed off the UI thread to keep QuPath responsive.
        int done = 0;
        if (imageData != null && currentEntry != null) {
            try {
                imageData.getHierarchy().resolveHierarchy();
                imageData.getHierarchy().fireHierarchyChangedEvent(EVENT_SOURCE);
                currentEntry.saveImageData(imageData);
                done = 1;
            } catch (Exception ex) {
                logger.warn(
                        "[SP Classify] Failed to resolve hierarchy for current image {}: {}",
                        currentEntry.getImageName(),
                        ex.getMessage());
            }
        }

        final int alreadyDone = done;
        final var currentEntryF = currentEntry;
        Thread worker = new Thread(
                () -> {
                    int ok = alreadyDone;
                    int failed = 0;
                    for (var entry : entries) {
                        if (entry == null) continue;
                        if (currentEntryF != null && entry.equals(currentEntryF)) continue; // handled above
                        try {
                            var data = entry.readImageData();
                            data.getHierarchy().resolveHierarchy();
                            entry.saveImageData(data);
                            ok++;
                        } catch (Exception ex) {
                            failed++;
                            logger.warn(
                                    "[SP Classify] Failed to resolve hierarchy for {}: {}",
                                    entry.getImageName(),
                                    ex.getMessage());
                        }
                    }
                    final int okF = ok;
                    final int failedF = failed;
                    Platform.runLater(() -> Dialogs.showInfoNotification(
                            EXTENSION_NAME,
                            "Resolved hierarchy for " + okF + " image(s)."
                                    + (failedF > 0 ? " " + failedF + " failed (see log)." : "")));
                    logger.info(
                            "[SP Classify] Project-wide resolveHierarchy complete: {} ok, {} failed.", okF, failedF);
                },
                "sp-classify-resolve-hierarchy");
        worker.setDaemon(true);
        worker.start();
    }

    /**
     * Lock every annotation object so they cannot be moved, edited, or deleted in
     * the viewer. Locking does not change cell classifications or the object
     * hierarchy. Offers a choice between the current image and every image in the
     * project; project-wide runs on a background thread and saves each entry.
     * Equivalent to running
     * {@code getAnnotationObjects().each { it.setLocked(true) }} in the script
     * editor.
     */
    static void lockAllAnnotations(QuPathGUI qupath) {
        var imageData = qupath.getImageData();
        var project = qupath.getProject();
        if (imageData == null && project == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No image or project is open.");
            return;
        }

        final String currentChoice = "Current image";
        final String projectChoice = "All project images";
        String choice;
        if (project == null) {
            choice = currentChoice;
        } else if (imageData == null) {
            choice = projectChoice;
        } else {
            choice = Dialogs.showChoiceDialog(
                    EXTENSION_NAME, "Lock all annotations for:", List.of(currentChoice, projectChoice), currentChoice);
            if (choice == null) return;
        }

        // Current image only — lock and refresh on the FX thread.
        if (choice.equals(currentChoice)) {
            var hierarchy = imageData.getHierarchy();
            var annotations = hierarchy.getAnnotationObjects();
            if (annotations.isEmpty()) {
                Dialogs.showInfoNotification(EXTENSION_NAME, "No annotations found on the current image.");
                return;
            }
            int locked = lockAnnotations(annotations);
            hierarchy.fireHierarchyChangedEvent(EVENT_SOURCE);
            Dialogs.showInfoNotification(
                    EXTENSION_NAME,
                    "Locked " + locked + " annotation(s)."
                            + (locked < annotations.size()
                                    ? " " + (annotations.size() - locked) + " were already locked."
                                    : ""));
            logger.info(
                    "[SP Classify] Locked {} of {} annotation(s) on the current image.", locked, annotations.size());
            return;
        }

        // Project-wide.
        @SuppressWarnings("unchecked")
        var entries = (List<ProjectImageEntry<BufferedImage>>) (List<?>) project.getImageList();
        if (entries.isEmpty()) {
            Dialogs.showInfoNotification(EXTENSION_NAME, "No project images found.");
            return;
        }
        if (!Dialogs.showConfirmDialog(
                EXTENSION_NAME,
                "Lock all annotations for all " + entries.size() + " project image(s) and save each?")) {
            return;
        }

        var currentEntry = imageData != null ? project.getEntry(imageData) : null;

        // Lock the open image immediately so its view updates; the rest are
        // processed off the UI thread to keep QuPath responsive.
        int doneImages = 0;
        int lockedSoFar = 0;
        if (imageData != null && currentEntry != null) {
            try {
                var hierarchy = imageData.getHierarchy();
                lockedSoFar = lockAnnotations(hierarchy.getAnnotationObjects());
                hierarchy.fireHierarchyChangedEvent(EVENT_SOURCE);
                currentEntry.saveImageData(imageData);
                doneImages = 1;
            } catch (Exception ex) {
                logger.warn(
                        "[SP Classify] Failed to lock annotations for current image {}: {}",
                        currentEntry.getImageName(),
                        ex.getMessage());
            }
        }

        final int alreadyDoneImages = doneImages;
        final int alreadyLocked = lockedSoFar;
        final var currentEntryF = currentEntry;
        Thread worker = new Thread(
                () -> {
                    int okImages = alreadyDoneImages;
                    int failed = 0;
                    int totalLocked = alreadyLocked;
                    for (var entry : entries) {
                        if (entry == null) continue;
                        if (currentEntryF != null && entry.equals(currentEntryF)) continue; // handled above
                        try {
                            var data = entry.readImageData();
                            totalLocked += lockAnnotations(data.getHierarchy().getAnnotationObjects());
                            entry.saveImageData(data);
                            okImages++;
                        } catch (Exception ex) {
                            failed++;
                            logger.warn(
                                    "[SP Classify] Failed to lock annotations for {}: {}",
                                    entry.getImageName(),
                                    ex.getMessage());
                        }
                    }
                    final int okImagesF = okImages;
                    final int failedF = failed;
                    final int totalLockedF = totalLocked;
                    Platform.runLater(() -> Dialogs.showInfoNotification(
                            EXTENSION_NAME,
                            "Locked " + totalLockedF + " annotation(s) across " + okImagesF + " image(s)."
                                    + (failedF > 0 ? " " + failedF + " failed (see log)." : "")));
                    logger.info(
                            "[SP Classify] Project-wide lock complete: {} annotation(s), {} ok, {} failed.",
                            totalLockedF,
                            okImagesF,
                            failedF);
                },
                "sp-classify-lock-annotations");
        worker.setDaemon(true);
        worker.start();
    }

    /**
     * Lock any not-yet-locked annotations in the supplied collection, returning the
     * number newly locked.
     */
    private static int lockAnnotations(Collection<PathObject> annotations) {
        int locked = 0;
        for (PathObject annotation : annotations) {
            if (!annotation.isLocked()) {
                annotation.setLocked(true);
                locked++;
            }
        }
        return locked;
    }

    /**
     * Delete detection measurements whose name contains a keyword (case-insensitive
     * by default), equivalent to the delete-measurements batch script. This is
     * destructive and not undoable, so the matching columns are previewed and
     * explicitly confirmed before anything is removed.
     */
    static void deleteMeasurementsByKeyword(QuPathGUI qupath) {
        var imageData = qupath.getImageData();
        var project = qupath.getProject();
        if (imageData == null && project == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No image or project is open.");
            return;
        }

        // Options: keyword + case sensitivity.
        TextField keywordField = new TextField();
        keywordField.setPromptText("e.g. Distance");
        CheckBox caseSensitive = new CheckBox("Case sensitive");

        GridPane grid = new GridPane();
        grid.setHgap(8);
        grid.setVgap(8);
        grid.setPadding(new Insets(10));
        grid.add(new Label("Keyword:"), 0, 0);
        grid.add(keywordField, 1, 0);
        grid.add(caseSensitive, 1, 1);

        Dialog<ButtonType> dialog = new Dialog<>();
        dialog.setTitle(EXTENSION_NAME);
        dialog.setHeaderText("Delete all detection measurements whose name contains the keyword.\n"
                + "This is permanent and cannot be undone.");
        dialog.getDialogPane().setContent(grid);
        dialog.getDialogPane().getButtonTypes().addAll(ButtonType.OK, ButtonType.CANCEL);
        var choice = dialog.showAndWait();
        if (choice.isEmpty() || choice.get() != ButtonType.OK) return;

        String keyword =
                keywordField.getText() == null ? "" : keywordField.getText().strip();
        if (keyword.isEmpty()) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "Please enter a keyword.");
            return;
        }
        boolean cs = caseSensitive.isSelected();

        // Scope.
        final String currentChoice = "Current image";
        final String projectChoice = "All project images";
        boolean projectWide;
        if (project == null) {
            projectWide = false;
        } else if (imageData == null) {
            projectWide = true;
        } else {
            String scope = Dialogs.showChoiceDialog(
                    EXTENSION_NAME, "Apply to:", List.of(currentChoice, projectChoice), currentChoice);
            if (scope == null) return;
            projectWide = scope.equals(projectChoice);
        }

        // Preview the matching columns from a sample image so the user can verify
        // before anything is deleted.
        List<String> preview;
        String previewImage;
        if (imageData != null) {
            preview = matchingMeasurementNames(imageData.getHierarchy().getDetectionObjects(), keyword, cs);
            previewImage = "the current image";
        } else {
            preview = List.of();
            previewImage = null;
            @SuppressWarnings("unchecked")
            var sampleEntries = (List<ProjectImageEntry<BufferedImage>>) (List<?>) project.getImageList();
            for (var entry : sampleEntries) {
                if (entry == null) continue;
                try {
                    var data = entry.readImageData();
                    var dets = data.getHierarchy().getDetectionObjects();
                    if (!dets.isEmpty()) {
                        preview = matchingMeasurementNames(dets, keyword, cs);
                        previewImage = entry.getImageName();
                        break;
                    }
                } catch (Exception ex) {
                    logger.warn(
                            "[SP Classify] Failed to read {} for measurement preview: {}",
                            entry.getImageName(),
                            ex.getMessage());
                }
            }
        }

        if (preview.isEmpty()) {
            Dialogs.showInfoNotification(
                    EXTENSION_NAME,
                    "No measurement names contain \"" + keyword + "\""
                            + (previewImage != null ? " on " + previewImage : "") + ".");
            return;
        }

        String columnList =
                String.join("\n", preview.stream().map(s -> "  \u2022 " + s).toList());
        int imageCount = projectWide ? project.getImageList().size() : 1;
        String scopeText = projectWide ? ("all " + imageCount + " project image(s)") : "the current image";
        boolean confirmed = Dialogs.showConfirmDialog(
                EXTENSION_NAME,
                "Permanently delete these " + preview.size() + " measurement column(s) from " + scopeText + "?\n\n"
                        + columnList + "\n\nThis cannot be undone.");
        if (!confirmed) return;

        // Current image only.
        if (!projectWide) {
            int touched = removeMeasurementsByKeyword(imageData.getHierarchy().getDetectionObjects(), keyword, cs);
            imageData.getHierarchy().fireHierarchyChangedEvent(EVENT_SOURCE);
            Dialogs.showInfoNotification(
                    EXTENSION_NAME, "Removed " + preview.size() + " column(s) from " + touched + " detection(s).");
            logger.info(
                    "[SP Classify] Deleted measurements matching \"{}\" (caseSensitive={}) from {} detections on current image.",
                    keyword,
                    cs,
                    touched);
            return;
        }

        // Project-wide.
        @SuppressWarnings("unchecked")
        var entries = (List<ProjectImageEntry<BufferedImage>>) (List<?>) project.getImageList();
        var currentEntry = imageData != null ? project.getEntry(imageData) : null;

        // Handle the open image immediately on the FX thread so its table refreshes.
        int doneImages = 0;
        long doneTouched = 0;
        if (imageData != null && currentEntry != null) {
            try {
                int t = removeMeasurementsByKeyword(imageData.getHierarchy().getDetectionObjects(), keyword, cs);
                imageData.getHierarchy().fireHierarchyChangedEvent(EVENT_SOURCE);
                currentEntry.saveImageData(imageData);
                if (t > 0) doneImages = 1;
                doneTouched = t;
            } catch (Exception ex) {
                logger.warn(
                        "[SP Classify] Failed to delete measurements for current image {}: {}",
                        currentEntry.getImageName(),
                        ex.getMessage());
            }
        }

        final int alreadyImages = doneImages;
        final long alreadyTouched = doneTouched;
        final var currentEntryF = currentEntry;
        final String keywordF = keyword;
        final boolean csF = cs;
        Thread worker = new Thread(
                () -> {
                    int images = alreadyImages;
                    long touched = alreadyTouched;
                    int failed = 0;
                    for (var entry : entries) {
                        if (entry == null) continue;
                        if (currentEntryF != null && entry.equals(currentEntryF)) continue; // handled above
                        try {
                            var data = entry.readImageData();
                            int t = removeMeasurementsByKeyword(
                                    data.getHierarchy().getDetectionObjects(), keywordF, csF);
                            entry.saveImageData(data);
                            if (t > 0) images++;
                            touched += t;
                        } catch (Exception ex) {
                            failed++;
                            logger.warn(
                                    "[SP Classify] Failed to delete measurements for {}: {}",
                                    entry.getImageName(),
                                    ex.getMessage());
                        }
                    }
                    final int imagesF = images;
                    final long touchedF = touched;
                    final int failedF = failed;
                    Platform.runLater(() -> Dialogs.showInfoNotification(
                            EXTENSION_NAME,
                            "Removed matching measurements from " + touchedF + " detection(s) across "
                                    + imagesF + " image(s)."
                                    + (failedF > 0 ? " " + failedF + " image(s) failed (see log)." : "")));
                    logger.info(
                            "[SP Classify] Project-wide measurement delete matching \"{}\": {} detections across {} images, {} failed.",
                            keywordF,
                            touchedF,
                            imagesF,
                            failedF);
                },
                "sp-classify-delete-measurements");
        worker.setDaemon(true);
        worker.start();
    }

    /**
     * Return the measurement names on the first detection that has any, matching
     * {@code keyword} as a substring (case-insensitive unless {@code caseSensitive}).
     */
    private static List<String> matchingMeasurementNames(
            Collection<PathObject> detections, String keyword, boolean caseSensitive) {
        PathObject sample = null;
        for (PathObject det : detections) {
            if (!det.getMeasurementList().getNames().isEmpty()) {
                sample = det;
                break;
            }
        }
        if (sample == null) return List.of();
        String needle = caseSensitive ? keyword : keyword.toLowerCase(java.util.Locale.ROOT);
        List<String> out = new ArrayList<>();
        for (String name : sample.getMeasurementList().getNames()) {
            String hay = caseSensitive ? name : name.toLowerCase(java.util.Locale.ROOT);
            if (hay.contains(needle)) out.add(name);
        }
        return out;
    }

    /**
     * Remove every measurement matching {@code keyword} from each detection.
     * Returns the number of detections that had at least one measurement removed.
     */
    private static int removeMeasurementsByKeyword(
            Collection<PathObject> detections, String keyword, boolean caseSensitive) {
        List<String> matches = matchingMeasurementNames(detections, keyword, caseSensitive);
        if (matches.isEmpty()) return 0;
        int touched = 0;
        for (PathObject det : detections) {
            var ml = det.getMeasurementList();
            boolean changed = false;
            for (String name : matches) {
                if (ml.containsKey(name)) {
                    ml.remove(name);
                    changed = true;
                }
            }
            if (changed) touched++;
        }
        return touched;
    }

    private static final String LARGE_GEOJSON_REPO = "https://github.com/BioimageAnalysisCoreWEHI/import_large_geojson";

    /**
     * Import annotations/detections from a GeoJSON (or gzipped GeoJSON) file into the
     * current image. Intended for small-to-medium files — it loads all objects into
     * memory, so very large files should use the dedicated headless pipeline instead
     * (linked from the dialog).
     */
    static void importGeoJsonObjects(QuPathGUI qupath) {
        var imageData = qupath.getImageData();
        if (imageData == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No image is open. Open the target image first.");
            return;
        }

        FileChooser chooser = new FileChooser();
        chooser.setTitle("Import GeoJSON objects");
        chooser.getExtensionFilters()
                .addAll(
                        new FileChooser.ExtensionFilter("GeoJSON", "*.geojson", "*.json", "*.geojson.gz", "*.json.gz"),
                        new FileChooser.ExtensionFilter("All files", "*.*"));
        File file = chooser.showOpenDialog(qupath.getStage());
        if (file == null) return;

        double sizeMB = file.length() / (1024.0 * 1024.0);

        // Disclaimer + options.
        Label warning = new Label("This importer loads the entire GeoJSON into memory and is intended for "
                + "small-to-medium files. Very large files (hundreds of MB / millions of "
                + "objects) can exhaust QuPath's heap and crash the application — use the "
                + "dedicated headless pipeline for those:");
        warning.setWrapText(true);
        warning.setMaxWidth(440);

        Hyperlink link = new Hyperlink("github.com/BioimageAnalysisCoreWEHI/import_large_geojson");
        link.setOnAction(ev -> {
            try {
                GuiTools.browseURI(new URI(LARGE_GEOJSON_REPO));
            } catch (Exception ex) {
                logger.warn("[SP Classify] Could not open browser for {}: {}", LARGE_GEOJSON_REPO, ex.getMessage());
            }
        });

        Label fileLabel = new Label(String.format("File: %s (%.1f MB)", file.getName(), sizeMB));
        fileLabel.setWrapText(true);
        fileLabel.setMaxWidth(440);

        CheckBox clearCb = new CheckBox("Clear existing objects first");
        CheckBox resolveCb = new CheckBox("Resolve hierarchy after import (slower; O(n²))");

        VBox content = new VBox(10, warning, link, new javafx.scene.control.Separator(), fileLabel, clearCb, resolveCb);
        content.setPadding(new Insets(10));

        Dialog<ButtonType> dialog = new Dialog<>();
        dialog.setTitle(EXTENSION_NAME);
        dialog.setHeaderText("Import GeoJSON objects into the current image");
        dialog.getDialogPane().setContent(content);
        dialog.getDialogPane().getButtonTypes().addAll(ButtonType.OK, ButtonType.CANCEL);
        var choice = dialog.showAndWait();
        if (choice.isEmpty() || choice.get() != ButtonType.OK) return;

        final boolean clearExisting = clearCb.isSelected();
        final boolean doResolve = resolveCb.isSelected();
        final var hierarchy = imageData.getHierarchy();
        final var project = qupath.getProject();
        final var entry = project != null ? project.getEntry(imageData) : null;

        Dialogs.showInfoNotification(EXTENSION_NAME, "Importing GeoJSON — see the log for progress.");

        Thread worker = new Thread(
                () -> {
                    List<PathObject> objects;
                    try {
                        objects = parseGeoJsonObjects(file);
                    } catch (Throwable t) {
                        logger.error("[SP Classify] GeoJSON parse failed: {}", t.toString(), t);
                        Platform.runLater(() ->
                                Dialogs.showErrorMessage(EXTENSION_NAME, "Failed to read GeoJSON: " + t.getMessage()));
                        return;
                    }
                    if (objects.isEmpty()) {
                        Platform.runLater(() -> Dialogs.showInfoNotification(
                                EXTENSION_NAME, "No valid objects found in " + file.getName() + "."));
                        return;
                    }

                    List<PathObject> annotations = new ArrayList<>();
                    List<PathObject> detections = new ArrayList<>();
                    List<PathObject> others = new ArrayList<>();
                    for (PathObject o : objects) {
                        if (o instanceof PathAnnotationObject) annotations.add(o);
                        else if (o instanceof PathDetectionObject) detections.add(o);
                        else others.add(o);
                    }

                    // Hierarchy mutation happens on the FX thread (the viewer observes it).
                    Platform.runLater(() -> {
                        try {
                            if (clearExisting) hierarchy.clearAll();
                            if (!annotations.isEmpty()) {
                                annotations.forEach(a -> a.setLocked(true));
                                hierarchy.addObjects(annotations);
                            }
                            if (!detections.isEmpty()) {
                                if (detections.size() > 200_000) {
                                    int chunk = 100_000;
                                    for (int i = 0; i < detections.size(); i += chunk) {
                                        hierarchy.addObjects(
                                                detections.subList(i, Math.min(i + chunk, detections.size())));
                                    }
                                } else {
                                    hierarchy.addObjects(detections);
                                }
                            }
                            if (!others.isEmpty()) hierarchy.addObjects(others);
                            if (doResolve) hierarchy.resolveHierarchy();
                            hierarchy.fireHierarchyChangedEvent(EVENT_SOURCE);
                            if (entry != null) {
                                try {
                                    entry.saveImageData(imageData);
                                } catch (Exception ex) {
                                    logger.warn(
                                            "[SP Classify] Failed to save image data after GeoJSON import: {}",
                                            ex.getMessage());
                                }
                            }
                            Dialogs.showInfoNotification(
                                    EXTENSION_NAME,
                                    String.format(
                                            "Imported %d object(s): %d annotation(s), %d detection(s)%s.",
                                            objects.size(),
                                            annotations.size(),
                                            detections.size(),
                                            others.isEmpty() ? "" : ", " + others.size() + " other"));
                            logger.info(
                                    "[SP Classify] GeoJSON import: {} objects ({} ann, {} det, {} other) from {}.",
                                    objects.size(),
                                    annotations.size(),
                                    detections.size(),
                                    others.size(),
                                    file.getName());
                        } catch (Throwable t) {
                            logger.error("[SP Classify] GeoJSON import (add) failed: {}", t.toString(), t);
                            Dialogs.showErrorMessage(EXTENSION_NAME, "Failed to add objects: " + t.getMessage());
                        }
                    });
                },
                "sp-classify-import-geojson");
        worker.setDaemon(true);
        worker.start();
    }

    /**
     * Stream-parse a GeoJSON file (FeatureCollection object or bare feature array,
     * optionally gzipped) into a list of {@link PathObject}s, one feature at a time
     * to avoid loading the whole JSON tree.
     */
    private static List<PathObject> parseGeoJsonObjects(File file) throws IOException {
        boolean gz = file.getName().toLowerCase(java.util.Locale.ROOT).endsWith(".gz");
        var gson = GsonTools.getInstance();
        com.google.gson.TypeAdapter<JsonElement> adapter = gson.getAdapter(JsonElement.class);
        List<PathObject> out = new ArrayList<>();
        long t0 = System.currentTimeMillis();
        int[] errors = {0};

        try (var fis = new FileInputStream(file);
                var bis = new BufferedInputStream(fis, 16 * 1024 * 1024);
                java.io.InputStream raw = gz ? new GZIPInputStream(bis, 16 * 1024 * 1024) : bis;
                var isr = new InputStreamReader(raw, StandardCharsets.UTF_8);
                JsonReader reader = new JsonReader(isr)) {
            reader.setLenient(true);
            JsonToken first = reader.peek();
            if (first == JsonToken.BEGIN_OBJECT) {
                reader.beginObject();
                boolean foundFeatures = false;
                while (reader.hasNext()) {
                    if ("features".equals(reader.nextName())) {
                        foundFeatures = true;
                        readFeatureArray(reader, adapter, gson, out, errors);
                    } else {
                        reader.skipValue();
                    }
                }
                reader.endObject();
                if (!foundFeatures) {
                    throw new IOException("GeoJSON object has no 'features' array — is it a valid FeatureCollection?");
                }
            } else if (first == JsonToken.BEGIN_ARRAY) {
                readFeatureArray(reader, adapter, gson, out, errors);
            } else {
                throw new IOException("Unexpected JSON structure (expected object or array, got " + first + ").");
            }
        }
        logger.info(
                "[SP Classify] Parsed GeoJSON: {} objects ({} errors) in {} ms.",
                out.size(),
                errors[0],
                System.currentTimeMillis() - t0);
        return out;
    }

    /** Read a JSON array of GeoJSON features, converting each to a {@link PathObject}. */
    private static void readFeatureArray(
            JsonReader reader,
            com.google.gson.TypeAdapter<JsonElement> adapter,
            com.google.gson.Gson gson,
            List<PathObject> out,
            int[] errors)
            throws IOException {
        reader.beginArray();
        while (reader.hasNext()) {
            JsonElement element = adapter.read(reader);
            try {
                PathObject obj = gson.fromJson(element, PathObject.class);
                if (obj != null) out.add(obj);
            } catch (Exception fe) {
                errors[0]++;
            }
        }
        reader.endArray();
    }
}
