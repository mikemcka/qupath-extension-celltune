package qupath.ext.spclassify;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.ResourceBundle;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.TextArea;
import javafx.scene.layout.VBox;
import javafx.stage.DirectoryChooser;
import javafx.stage.FileChooser;
import javafx.stage.Modality;
import javafx.stage.Stage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.io.CellTableExporter;
import qupath.ext.spclassify.io.GroundTruthIO;
import qupath.ext.spclassify.io.MarkerTableImporter;
import qupath.ext.spclassify.io.ProjectStateManager;
import qupath.ext.spclassify.model.CellFeatureExtractor;
import qupath.ext.spclassify.model.CellTypeTable;
import qupath.ext.spclassify.model.LabelStore;
import qupath.ext.spclassify.ui.CellTableExportPane;
import qupath.ext.spclassify.ui.ImageSelectionPane;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjectFilter;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;

/**
 * Cell-table / ground-truth / marker-table import &amp; export, lifted out of
 * {@code SpClassifyExtension} (mirroring the {@code UtilityScripts}/{@code AnalysisViews} moves).
 * <p>
 * The export methods and the two CSV-append helpers are read-only; {@code exportGroundTruth}
 * takes the session state it needs (labels, imported rows, active binary marker)
 * as parameters. The two <em>import</em> methods return their result instead of mutating the
 * extension directly — the extension assigns the returned state and refreshes its panel in one
 * place — so this class stays free of session-state side effects.
 */
final class ImportExport {

    private static final ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.spclassify.ui.strings");
    private static final String EXTENSION_NAME = resources.getString("name");
    private static final Logger logger = LoggerFactory.getLogger(ImportExport.class);

    private ImportExport() {} // utility class

    /** New session state produced by a ground-truth import, applied back by the caller. */
    record GroundTruthImportResult(
            LabelStore labelStore, List<GroundTruthIO.TrainingRow> importedRows, List<String> importedFeatureNames) {}

    static void exportCellTable(QuPathGUI qupath) {
        var imageData = qupath.getImageData();
        if (imageData == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No image is open.");
            return;
        }
        var project = qupath.getProject();

        // Resolve current image name
        String currentImageName = null;
        if (project != null) {
            var currentEntry = project.getEntry(imageData);
            if (currentEntry != null) currentImageName = currentEntry.getImageName();
        }
        if (currentImageName == null || currentImageName.isBlank()) {
            currentImageName = imageData.getServer().getMetadata().getName();
        }
        final String currentImageNameFinal = currentImageName;

        // Build full list of project image names
        List<String> allImageNames;
        if (project != null) {
            @SuppressWarnings("unchecked")
            var entries = (List<ProjectImageEntry<BufferedImage>>) (List<?>) project.getImageList();
            allImageNames = entries.stream()
                    .map(ProjectImageEntry::getImageName)
                    .filter(n -> n != null && !n.isBlank())
                    .collect(java.util.stream.Collectors.toList());
        } else {
            allImageNames = List.of(currentImageNameFinal);
        }

        // Show image selection dialog when the project has more than one image
        List<String> selectedImages;
        if (allImageNames.size() > 1) {
            var pane = new ImageSelectionPane(qupath.getStage(), allImageNames, currentImageNameFinal);
            selectedImages = pane.showAndWait();
            if (selectedImages == null || selectedImages.isEmpty()) return;
        } else {
            selectedImages = new ArrayList<>(allImageNames);
        }

        // Column + polygon options — discover features from the current image so
        // the user can choose which measurement columns to export (mirrors the
        // Select Features dialog) and whether to include cell polygons.
        Collection<PathObject> currentCells = imageData.getHierarchy().getObjects(null, PathObject.class).stream()
                .filter(PathObjectFilter.DETECTIONS_ALL)
                .toList();
        List<String> allCurrentFeatures = new ArrayList<>(CellFeatureExtractor.discoverFeatureNames(currentCells));
        // Append string metadata keys (e.g. "CN Class", "… original class") so they
        // can be selected as export columns. These are string-valued and never
        // appear among the numeric measurement columns above.
        for (String metaKey : CellFeatureExtractor.discoverMetadataKeys(currentCells)) {
            if (!allCurrentFeatures.contains(metaKey)) allCurrentFeatures.add(metaKey);
        }
        if (allCurrentFeatures.isEmpty()) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No cell measurements found to export.");
            return;
        }
        // Default pre-selection: the curated whole-cell mean + distance subset.
        List<String> defaultFeatures = allCurrentFeatures.stream()
                .filter(f -> {
                    String lc = f.toLowerCase(java.util.Locale.ROOT);
                    return f.matches("^[^:]+: Cell: Mean$") || lc.contains("distance");
                })
                .collect(java.util.stream.Collectors.toList());
        if (defaultFeatures.isEmpty()) defaultFeatures = allCurrentFeatures;

        var exportPane = new CellTableExportPane(qupath.getStage(), allCurrentFeatures, defaultFeatures, true, true);
        CellTableExportPane.Result exportResult = exportPane.showAndWait();
        if (exportResult == null) return;

        final List<String> selectedFeats = exportResult.features();
        final boolean includeGeometry = exportResult.includeGeometry();
        final boolean geometryInMicrons = exportResult.geometryInMicrons();

        // Output directory
        DirectoryChooser dc = new DirectoryChooser();
        dc.setTitle("Select Output Folder for Cell Table(s)");
        if (project != null && project.getPath() != null) {
            File dir = project.getPath().getParent().toFile();
            if (dir.isDirectory()) dc.setInitialDirectory(dir);
        }
        File outDir = dc.showDialog(qupath.getStage());
        if (outDir == null) return;

        final var finalImageData = imageData;
        final var finalProject = project;
        final var finalSelected = selectedImages;
        final int total = selectedImages.size();

        // Progress dialog — built and shown on the FX thread before the worker starts
        Stage progressStage = new Stage();
        progressStage.setTitle("Exporting Cell Tables");
        progressStage.initOwner(qupath.getStage());
        progressStage.initModality(Modality.NONE);
        progressStage.setResizable(true);

        ProgressBar progressBar = new ProgressBar(0);
        progressBar.setPrefWidth(Double.MAX_VALUE);
        Label statusLabel = new Label("Starting export…");
        TextArea logArea = new TextArea();
        logArea.setEditable(false);
        logArea.setPrefHeight(130);
        logArea.setStyle("-fx-font-family: monospace; -fx-font-size: 11px;");
        Button closeBtn = new Button("Close");
        closeBtn.setDisable(true);
        closeBtn.setOnAction(e -> progressStage.close());

        VBox progressRoot = new VBox(8, statusLabel, progressBar, logArea, closeBtn);
        progressRoot.setPadding(new Insets(14));
        progressStage.setScene(new Scene(progressRoot, 440, 270));
        progressStage.show();

        Thread worker = new Thread(
                () -> {
                    AtomicInteger exported = new AtomicInteger();
                    AtomicInteger done = new AtomicInteger();
                    List<String> errors = Collections.synchronizedList(new ArrayList<>());

                    // One task per image; cap at 4 threads (I/O-bound work)
                    int nThreads = Math.min(total, 4);
                    ExecutorService pool = Executors.newFixedThreadPool(nThreads);

                    List<Callable<Void>> tasks = new ArrayList<>();
                    for (String imgName : finalSelected) {
                        tasks.add(() -> {
                            try {
                                qupath.lib.images.ImageData<BufferedImage> data;
                                if (imgName.equals(currentImageNameFinal)) {
                                    // Use the already-open image data directly (read-only)
                                    data = finalImageData;
                                } else {
                                    // Load from disk — independent per thread
                                    @SuppressWarnings("unchecked")
                                    var typedProject = (Project<BufferedImage>) (Object) finalProject;
                                    var entryOpt = typedProject.getImageList().stream()
                                            .filter(e -> imgName.equals(e.getImageName()))
                                            .findFirst();
                                    if (entryOpt.isEmpty()) {
                                        errors.add(imgName + ": not found in project");
                                        int d = done.incrementAndGet();
                                        Platform.runLater(() -> {
                                            progressBar.setProgress((double) d / total);
                                            statusLabel.setText("Processing " + d + " / " + total);
                                            logArea.appendText("✗ " + imgName + ": not found in project\n");
                                        });
                                        return null;
                                    }
                                    data = entryOpt.get().readImageData();
                                    if (data == null) {
                                        errors.add(imgName + ": could not read image data");
                                        int d = done.incrementAndGet();
                                        Platform.runLater(() -> {
                                            progressBar.setProgress((double) d / total);
                                            statusLabel.setText("Processing " + d + " / " + total);
                                            logArea.appendText("✗ " + imgName + ": could not read image data\n");
                                        });
                                        return null;
                                    }
                                }

                                Collection<PathObject> cells =
                                        data.getHierarchy().getObjects(null, PathObject.class).stream()
                                                .filter(PathObjectFilter.DETECTIONS_ALL)
                                                .toList();

                                // All annotations for geometric containment testing
                                // (captures overlapping regions the hierarchy discards).
                                Collection<PathObject> annotations =
                                        data.getHierarchy().getAnnotationObjects();

                                // Use the user-selected measurement columns. Missing
                                // measurements are written as NA by the exporter.
                                List<String> feats = selectedFeats;

                                // Pixel calibration for centroid conversion to microns
                                var cal = data.getServer().getPixelCalibration();
                                double pixelWidthUm = cal.getPixelWidthMicrons();
                                double pixelHeightUm = cal.getPixelHeightMicrons();
                                if (Double.isNaN(pixelWidthUm)) pixelWidthUm = 1.0;
                                if (Double.isNaN(pixelHeightUm)) pixelHeightUm = 1.0;

                                // Sanitise the image name to produce a safe file name
                                String safeFileName = imgName.replaceAll("[\\\\/:*?\"<>|]", "_") + ".csv";
                                Path outputPath = outDir.toPath().resolve(safeFileName);

                                CellTableExporter.export(
                                        outputPath,
                                        cells,
                                        annotations,
                                        imgName,
                                        feats,
                                        pixelWidthUm,
                                        pixelHeightUm,
                                        includeGeometry,
                                        geometryInMicrons);
                                exported.incrementAndGet();
                                int d = done.incrementAndGet();
                                Platform.runLater(() -> {
                                    progressBar.setProgress((double) d / total);
                                    statusLabel.setText("Processing " + d + " / " + total);
                                    logArea.appendText("✓ " + imgName + "\n");
                                });

                            } catch (Exception ex) {
                                String errMsg = ex.getMessage();
                                logger.error("Failed to export cell table for '{}'", imgName, ex);
                                errors.add(imgName + ": " + errMsg);
                                int d = done.incrementAndGet();
                                Platform.runLater(() -> {
                                    progressBar.setProgress((double) d / total);
                                    statusLabel.setText("Processing " + d + " / " + total);
                                    logArea.appendText("✗ " + imgName + ": " + errMsg + "\n");
                                });
                            }
                            return null;
                        });
                    }

                    try {
                        pool.invokeAll(tasks);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        pool.shutdown();
                    }

                    final int finalExported = exported.get();
                    final List<String> finalErrors = new ArrayList<>(errors);
                    Platform.runLater(() -> {
                        progressBar.setProgress(1.0);
                        closeBtn.setDisable(false);
                        if (finalErrors.isEmpty()) {
                            statusLabel.setText(
                                    "Done — exported " + finalExported + " image(s) to " + outDir.getName());
                        } else {
                            statusLabel.setText(
                                    "Done — " + finalExported + " exported, " + finalErrors.size() + " error(s).");
                        }
                    });
                },
                "SpClassify-ExportCellTable");
        worker.setDaemon(true);
        worker.start();
    }

    /**
     * Import a marker table from a user-chosen CSV, persist it to the project, and notify.
     *
     * @return the imported table (caller assigns it and refreshes the panel), or null on
     *         cancel/failure
     */
    static CellTypeTable importMarkerTable(QuPathGUI qupath) {
        FileChooser fc = new FileChooser();
        fc.setTitle("Import Marker Table");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV files", "*.csv"));
        var project = qupath.getProject();
        if (project != null && project.getPath() != null) {
            File dir = project.getPath().getParent().toFile();
            if (dir.isDirectory()) fc.setInitialDirectory(dir);
        }
        File chosen = fc.showOpenDialog(qupath.getStage());
        if (chosen == null) return null;

        try {
            CellTypeTable table = MarkerTableImporter.importFromCSV(chosen.toPath());
            if (project != null) {
                try {
                    ProjectStateManager.saveMarkerTable(project, table);
                } catch (IOException saveEx) {
                    logger.warn("Failed to persist marker table to project: {}", saveEx.getMessage());
                }
            }
            Dialogs.showInfoNotification(
                    EXTENSION_NAME, "Loaded " + table.size() + " cell types from " + chosen.getName());
            return table;
        } catch (IOException ex) {
            logger.error("Failed to import marker table", ex);
            Dialogs.showErrorMessage(EXTENSION_NAME, "Import failed: " + ex.getMessage());
            return null;
        }
    }

    // ── Ground truth export/import ─────────────────────────────────────────────

    static void exportGroundTruth(
            QuPathGUI qupath,
            LabelStore labelStore,
            String activeBinaryMarker,
            List<GroundTruthIO.TrainingRow> importedTrainingRows,
            List<String> importedTrainingFeatureNames) {
        var imageData = qupath.getImageData();
        if (imageData == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No image is open.");
            return;
        }
        boolean inBinaryMode = activeBinaryMarker != null && !activeBinaryMarker.isBlank();
        int importedRowCount = (inBinaryMode && importedTrainingRows != null) ? importedTrainingRows.size() : 0;
        int localLabelCount = (labelStore == null) ? 0 : labelStore.size();
        if (localLabelCount == 0 && importedRowCount == 0) {
            Dialogs.showErrorMessage(EXTENSION_NAME, resources.getString("gt.export.empty"));
            return;
        }

        Collection<PathObject> detections = imageData.getHierarchy().getDetectionObjects();
        List<String> featureNames = CellFeatureExtractor.discoverFeatureNames(detections);
        if (featureNames.isEmpty()) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No cell measurements found.");
            return;
        }

        FileChooser fc = new FileChooser();
        fc.setTitle("Export Ground Truth");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV files", "*.csv"));
        var project = qupath.getProject();
        if (project != null && project.getPath() != null) {
            File dir = project.getPath().getParent().toFile();
            if (dir.isDirectory()) fc.setInitialDirectory(dir);
        }
        fc.setInitialFileName(
                activeBinaryMarker != null && !activeBinaryMarker.isBlank()
                        ? activeBinaryMarker + "_ground_truth.csv"
                        : "ground_truth.csv");
        File chosen = fc.showSaveDialog(qupath.getStage());
        if (chosen == null) return;

        try {
            CellFeatureExtractor extractor = new CellFeatureExtractor(featureNames); // raw — classifier export
            String imgName = imageData.getServer().getMetadata().getName();
            GroundTruthIO.exportCSV(chosen.toPath(), detections, labelStore, extractor, imgName);

            // Pool labels from every other project image so the export reflects the
            // full training set used (matches the auto-pool behaviour during training).
            // Per-image labels are scoped by activeBinaryMarker so binary classifiers
            // don't pull labels from each other.
            int appendedFromOtherImages = 0;
            if (project != null) {
                appendedFromOtherImages = appendOtherImageLabelsToCsv(
                        chosen.toPath(), project, imageData, activeBinaryMarker, featureNames, true);
            }

            // In binary mode, also append previously-imported training rows so the exported
            // CSV is the union of (this project's labels) + (rows imported from prior projects).
            // This keeps round-trips (project1 -> project2 -> project3) lossless for a single
            // binary marker, since each export carries the full accumulated training set.
            int appendedImported = 0;
            if (inBinaryMode
                    && importedRowCount > 0
                    && importedTrainingFeatureNames != null
                    && !importedTrainingFeatureNames.isEmpty()) {
                appendedImported = appendImportedRowsToCsv(
                        chosen.toPath(), featureNames, true, importedTrainingFeatureNames, importedTrainingRows);
            }

            String msg = "Exported " + localLabelCount + " labelled cells to " + chosen.getName();
            if (appendedFromOtherImages > 0) {
                msg += " (+" + appendedFromOtherImages + " from other project images)";
            }
            if (appendedImported > 0) {
                msg += " (+" + appendedImported + " imported training rows from prior projects)";
            }
            Dialogs.showInfoNotification(EXTENSION_NAME, msg);
        } catch (IOException ex) {
            logger.error("Failed to export ground truth", ex);
            Dialogs.showErrorMessage(EXTENSION_NAME, "Export failed: " + ex.getMessage());
        }
    }

    /**
     * Append previously-imported training rows to a ground-truth CSV that was just written
     * by {@link GroundTruthIO#exportCSV}. Imported rows are aligned to the export's column
     * layout by feature name; columns absent from the imported schema are written as 0.
     *
     * @return number of rows appended
     */
    private static int appendImportedRowsToCsv(
            Path csvPath,
            List<String> featureNames,
            boolean includeRaw,
            List<String> importedFeatureNames,
            List<GroundTruthIO.TrainingRow> importedRows)
            throws IOException {
        if (importedRows == null || importedRows.isEmpty()) return 0;

        // Build the export column ordering: raw featureNames (if includeRaw).
        // Mirrors GroundTruthIO.exportCSV.
        List<String> exportCols = new ArrayList<>();
        if (includeRaw) exportCols.addAll(featureNames);
        if (exportCols.isEmpty()) return 0;

        // name -> index lookup into the imported feature vector
        Map<String, Integer> importedNameToIdx = new LinkedHashMap<>();
        for (int i = 0; i < importedFeatureNames.size(); i++) {
            importedNameToIdx.putIfAbsent(importedFeatureNames.get(i), i);
        }

        List<String> lines = new ArrayList<>();
        for (GroundTruthIO.TrainingRow row : importedRows) {
            if (row == null || row.label() == null || row.label().isBlank() || row.features() == null) continue;
            float[] src = row.features();
            StringBuilder sb = new StringBuilder();
            // Image, Label, CentroidX, CentroidY (centroid 0,0 - row originates from another project)
            sb.append("imported").append(',').append(row.label()).append(",0.00,0.00");
            for (String col : exportCols) {
                Integer idx = importedNameToIdx.get(col);
                float v = (idx != null && idx < src.length) ? src[idx] : 0f;
                sb.append(',').append(v);
            }
            lines.add(sb.toString());
        }

        if (lines.isEmpty()) return 0;
        Files.write(csvPath, lines, StandardCharsets.UTF_8, StandardOpenOption.APPEND);
        return lines.size();
    }

    /**
     * Append rows for labels saved on every project image OTHER than the currently
     * open one. Each non-current image is opened off-screen, its detections are matched
     * against its persisted labels, and a feature row is written per matched cell.
     * Schema must match {@link GroundTruthIO#exportCSV} so the resulting CSV is one
     * coherent file.
     *
     * @param scope sanitized binary marker name, or null in multi-class mode
     * @return number of rows appended (across all other images)
     */
    private static int appendOtherImageLabelsToCsv(
            Path csvPath,
            qupath.lib.projects.Project<?> project,
            qupath.lib.images.ImageData<BufferedImage> currentImageData,
            String scope,
            List<String> featureNames,
            boolean includeRaw)
            throws IOException {
        if (project == null) return 0;
        if (!includeRaw) return 0;

        @SuppressWarnings("unchecked")
        var typedProject = (qupath.lib.projects.Project<BufferedImage>) (qupath.lib.projects.Project<?>) project;
        @SuppressWarnings("unchecked")
        var allEntries = (List<ProjectImageEntry<BufferedImage>>) (List<?>) project.getImageList();
        var currentEntry = currentImageData != null ? typedProject.getEntry(currentImageData) : null;

        List<String> linesOut = new ArrayList<>();
        for (var entry : allEntries) {
            if (entry == null) continue;
            if (currentEntry != null && entry.equals(currentEntry)) continue;

            String otherImageName = entry.getImageName();
            if (otherImageName == null || otherImageName.isBlank()) continue;

            // Skip images with no saved labels — avoids the expensive readImageData() call.
            if (!ProjectStateManager.hasImageLabels(project, scope, otherImageName)) continue;

            LabelStore otherLabels;
            try {
                otherLabels = ProjectStateManager.loadImageLabels(project, scope, otherImageName);
            } catch (Exception ex) {
                logger.warn("Failed to load labels for {}: {}", otherImageName, ex.getMessage());
                continue;
            }
            if (otherLabels == null || otherLabels.size() == 0) continue;

            try {
                var otherImageData = entry.readImageData();
                var otherDetections = otherImageData.getHierarchy().getDetectionObjects();
                if (otherDetections.isEmpty()) continue;

                Map<String, PathObject> cellById = new LinkedHashMap<>();
                for (PathObject cell : otherDetections) {
                    cellById.put(cell.getID().toString(), cell);
                }

                CellFeatureExtractor extractor = new CellFeatureExtractor(featureNames); // raw — classifier export

                for (var labelEntry : otherLabels.getAllLabels().entrySet()) {
                    PathObject cell = cellById.get(labelEntry.getKey());
                    if (cell == null) continue;
                    String label = labelEntry.getValue();
                    if (label == null || label.isBlank()) continue;

                    var roi = cell.getROI();
                    double cx = roi != null ? roi.getCentroidX() : 0;
                    double cy = roi != null ? roi.getCentroidY() : 0;

                    StringBuilder sb = new StringBuilder();
                    sb.append(otherImageName).append(',').append(label);
                    sb.append(',').append(String.format("%.2f", cx));
                    sb.append(',').append(String.format("%.2f", cy));
                    if (includeRaw) {
                        float[] raw = extractor.extractRowRaw(cell);
                        for (float v : raw) sb.append(',').append(v);
                    }
                    linesOut.add(sb.toString());
                }
            } catch (Exception ex) {
                logger.warn("Failed to extract labelled features from {}: {}", otherImageName, ex.getMessage());
            }
        }

        if (linesOut.isEmpty()) return 0;
        Files.write(csvPath, linesOut, StandardCharsets.UTF_8, StandardOpenOption.APPEND);
        return linesOut.size();
    }

    /**
     * Import ground truth from a user-chosen CSV (spatial-match or training-data mode).
     *
     * @return the new session state to apply (labels and/or imported rows), or null on
     *         cancel/failure. The caller assigns the returned fields and refreshes its panel.
     */
    static GroundTruthImportResult importGroundTruth(
            QuPathGUI qupath,
            LabelStore labelStore,
            List<GroundTruthIO.TrainingRow> importedTrainingRows,
            List<String> importedTrainingFeatureNames,
            String activeBinaryMarker) {
        var imageData = qupath.getImageData();
        if (imageData == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No image is open.");
            return null;
        }

        FileChooser fc = new FileChooser();
        fc.setTitle("Import Ground Truth");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV files", "*.csv"));
        var project = qupath.getProject();
        if (project != null && project.getPath() != null) {
            File dir = project.getPath().getParent().toFile();
            if (dir.isDirectory()) fc.setInitialDirectory(dir);
        }
        File chosen = fc.showOpenDialog(qupath.getStage());
        if (chosen == null) return null;

        // Ask user: spatial match vs training data only
        var result = Dialogs.showChoiceDialog(
                resources.getString("gt.import.mode.title"),
                resources.getString("gt.import.mode.prompt"),
                List.of(resources.getString("gt.import.mode.spatial"), resources.getString("gt.import.mode.training")),
                resources.getString("gt.import.mode.spatial"));
        if (result == null) return null;

        boolean spatial = result.equals(resources.getString("gt.import.mode.spatial"));

        try {
            if (spatial) {
                String threshStr = Dialogs.showInputDialog(
                        resources.getString("gt.import.mode.title"),
                        resources.getString("gt.import.spatial.threshold"),
                        resources.getString("gt.import.spatial.default"));
                if (threshStr == null) return null;
                double maxDist;
                try {
                    maxDist = Double.parseDouble(threshStr.strip());
                } catch (NumberFormatException e) {
                    Dialogs.showErrorMessage(EXTENSION_NAME, "Invalid distance value.");
                    return null;
                }

                Collection<PathObject> detections = imageData.getHierarchy().getDetectionObjects();
                LabelStore imported = GroundTruthIO.importCSVSpatial(chosen.toPath(), detections, maxDist);

                LabelStore newStore = (labelStore == null) ? new LabelStore("CellTune") : labelStore;
                newStore.mergeFrom(imported);

                Dialogs.showInfoNotification(
                        EXTENSION_NAME,
                        "Imported " + imported.size() + " labels via spatial matching. " + "Total labels: "
                                + newStore.size());
                return new GroundTruthImportResult(newStore, importedTrainingRows, importedTrainingFeatureNames);
            } else {
                var rows = GroundTruthIO.importCSVAsTrainingData(chosen.toPath());
                var featureNames = GroundTruthIO.readFeatureNames(chosen.toPath());

                if (featureNames.isEmpty()) {
                    Dialogs.showErrorMessage(EXTENSION_NAME, "Imported file has no readable feature columns.");
                    return null;
                }

                int loaded = rows.size();
                int total;
                String schemaNote = "";

                List<GroundTruthIO.TrainingRow> newImportedRows;
                List<String> newImportedFeatureNames;
                if (importedTrainingRows == null || importedTrainingRows.isEmpty()) {
                    newImportedRows = new ArrayList<>(rows);
                    newImportedFeatureNames = new ArrayList<>(featureNames);
                } else if (importedTrainingFeatureNames != null && importedTrainingFeatureNames.equals(featureNames)) {
                    importedTrainingRows.addAll(rows);
                    newImportedRows = importedTrainingRows;
                    newImportedFeatureNames = importedTrainingFeatureNames;
                } else {
                    // Replace if schema differs; mixed schemas cannot be safely merged.
                    newImportedRows = new ArrayList<>(rows);
                    newImportedFeatureNames = new ArrayList<>(featureNames);
                    schemaNote = " Feature schema changed, so previous imported rows were replaced.";
                }

                total = newImportedRows.size();

                if (project != null) {
                    if (activeBinaryMarker != null && !activeBinaryMarker.isBlank()) {
                        ProjectStateManager.saveBinaryImportedTrainingData(
                                project, activeBinaryMarker,
                                newImportedFeatureNames, newImportedRows);
                    } else {
                        ProjectStateManager.saveImportedTrainingData(project, newImportedFeatureNames, newImportedRows);
                    }
                }

                Dialogs.showInfoNotification(
                        EXTENSION_NAME,
                        "Loaded " + loaded + " training rows ("
                                + featureNames.size() + " features). "
                                + "Imported rows available for training: " + total + "."
                                + schemaNote);
                return new GroundTruthImportResult(labelStore, newImportedRows, newImportedFeatureNames);
            }
        } catch (IOException ex) {
            logger.error("Failed to import ground truth", ex);
            Dialogs.showErrorMessage(EXTENSION_NAME, "Import failed: " + ex.getMessage());
            return null;
        }
    }

    /**
     * Clear the imported training rows for the active context — the binary marker if one is
     * active, otherwise multi-class mode — after a typed confirmation. Labels, trained models
     * and predictions are left untouched. Deletes the persisted payload
     * ({@code binary-imported/<marker>.json} in binary mode, or the imported-rows fields of
     * {@code classifier-state.json} in multi-class mode).
     *
     * @param activeBinaryMarker the active binary marker, or null/blank for multi-class
     * @return true if the user confirmed — the caller should then null its in-memory imported-rows
     *         fields and refresh the panel; false on cancel or failure
     */
    static boolean clearImportedTrainingData(QuPathGUI qupath, String activeBinaryMarker) {
        boolean binary = activeBinaryMarker != null && !activeBinaryMarker.isBlank();
        String context = binary ? "the \"" + activeBinaryMarker + "\" binary classifier" : "multi-class mode";

        boolean confirmed = Dialogs.showConfirmDialog(
                EXTENSION_NAME,
                "Remove all imported training rows for " + context + "?\n\n"
                        + "Your labels, trained models and predictions are not affected.");
        if (!confirmed) return false;

        var project = qupath.getProject();
        if (project != null) {
            try {
                if (binary) {
                    ProjectStateManager.deleteBinaryImportedTrainingData(project, activeBinaryMarker);
                } else {
                    ProjectStateManager.clearImportedTrainingData(project);
                }
            } catch (IOException ex) {
                logger.error("Failed to clear imported training data", ex);
                Dialogs.showErrorMessage(EXTENSION_NAME, "Failed to clear imported training data: " + ex.getMessage());
                return false;
            }
        }
        Dialogs.showInfoNotification(EXTENSION_NAME, "Cleared imported training rows for " + context + ".");
        return true;
    }
}
