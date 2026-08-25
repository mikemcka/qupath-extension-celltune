package qupath.ext.celltune.ui;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.RadioButton;
import javafx.scene.control.Separator;
import javafx.scene.control.Spinner;
import javafx.scene.control.SpinnerValueFactory;
import javafx.scene.control.TextArea;
import javafx.scene.control.ToggleGroup;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.stage.Modality;
import javafx.stage.Stage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.celltune.io.ProjectStateManager;
import qupath.ext.celltune.model.BatchNormalizerApply;
import qupath.ext.celltune.model.BatchNormalizerCohort;
import qupath.ext.celltune.model.BatchNormalizerCohort.NormalizerFit;
import qupath.ext.celltune.model.BatchNormalizerModel;
import qupath.ext.celltune.model.BatchShifts;
import qupath.ext.celltune.model.CohortClusterModel.CancellationToken;
import qupath.ext.celltune.model.IntensityHeatmap;
import qupath.fx.dialogs.Dialogs;
import qupath.fx.dialogs.FileChoosers;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectIO;
import qupath.lib.projects.ProjectImageEntry;

/**
 * Non-modal control dialog for <b>UniFORM batch normalization</b> — project-scope only
 * (batch correction is inherently multi-image). The user picks which intensity
 * measurements to correct (embeddings excluded), how images map to batches, and the
 * granularity; Run learns and persists per-{@code (image, marker)} scales, opens a QC
 * window, and optionally writes corrected {@code (batchnorm)} columns.
 *
 * <p>Mirrors {@code NeighborhoodAnalysisDialog}'s cohort machinery: {@link ImageSelectionPane}
 * picker, "Add project" pooling, workers spinner, background worker thread, {@code this::log}
 * + progress callback into the pure {@link BatchNormalizerCohort} backend.
 */
public class BatchNormalizationDialog {

    private static final Logger logger = LoggerFactory.getLogger(BatchNormalizationDialog.class);

    private final QuPathGUI qupath;
    private final Stage stage;

    // Measurement selection (defaults to marker means; embeddings excluded).
    private List<String> allFeatureNames = new ArrayList<>();
    private List<String> selectedMeasurements = new ArrayList<>();
    private final Label measurementsLabel = new Label();

    // Scope / images.
    private final List<String> allImageNames = new ArrayList<>();
    private List<String> selectedImages = new ArrayList<>();
    private final Label imagesCountLabel = new Label();
    private final List<Project<BufferedImage>> extraProjects = new ArrayList<>();
    private final Label extraProjectsLabel = new Label();

    // Batch grouping.
    // Manual image → batch assignment (edited via BatchAssignmentPane; auto-detected on first build).
    private Map<String, String> manualBatch = new LinkedHashMap<>();
    private final Label batchSummaryLabel = new Label();

    // Granularity.
    private final ToggleGroup granularityGroup = new ToggleGroup();
    private final RadioButton perImageRadio = new RadioButton("Per image (each image → reference)");
    private final RadioButton perBatchRadio = new RadioButton("Per batch (pool images in a batch)");

    // Fit params.
    private final Spinner<Integer> binsSpinner = new Spinner<>();
    private final Spinner<Integer> sampleSpinner = new Spinner<>();
    private final Spinner<Integer> workersSpinner = new Spinner<>();

    private final TextArea logArea = new TextArea();
    private final ProgressBar progressBar = new ProgressBar(0);
    private final Label statusLabel = new Label("Configure and click Run.");
    private final Button runBtn = new Button("Run fit");
    private final Button writeBtn = new Button("Write corrected columns");
    private final Button qcBtn = new Button("Show QC");
    private final Button cancelBtn = new Button("Cancel");
    private final Button closeBtn = new Button("Close");

    private volatile CancellationToken token;
    private NormalizerFit lastFit; // last successful fit (for QC + write)

    public BatchNormalizationDialog(QuPathGUI qupath) {
        this.qupath = qupath;
        this.stage = buildStage();
    }

    public void show() {
        stage.show();
    }

    // ── UI ─────────────────────────────────────────────────────────────────────

    private Stage buildStage() {
        // Discover measurement names + default (marker means) from the open image.
        allFeatureNames = discoverFeatureNames();
        // Default = true marker means only. Embeddings measured as channel means (e.g.
        // "kronos_emb_0: Cell: Mean") end in ": Cell: Mean" too, so filter them out here —
        // the correction is invalid for embedding dimensions.
        List<String> markerMeans = new ArrayList<>(IntensityHeatmap.discoverMarkerFeatures(allFeatureNames));
        markerMeans.removeIf(FeatureSelectionPane::isEmbedding);
        selectedMeasurements = new ArrayList<>(markerMeans);
        updateMeasurementsLabel();
        Button chooseMeasBtn = new Button("Choose measurements…");
        chooseMeasBtn.setOnAction(e -> chooseMeasurements());
        Label measHint =
                new Label("Only intensity measurements are corrected; foundation-model embeddings are excluded.");
        measHint.setWrapText(true);
        measHint.setStyle("-fx-font-size: 11px; -fx-text-fill: #666;");
        VBox measBox = new VBox(6, new HBox(8, chooseMeasBtn, measurementsLabel), measHint);

        // Images.
        var project = qupath.getProject();
        if (project != null) {
            for (var entry : project.getImageList()) {
                if (entry.getImageName() != null) {
                    allImageNames.add(entry.getImageName());
                }
            }
        }
        selectedImages = new ArrayList<>(allImageNames);
        imagesCountLabel.setText(imageCountText());
        Button imagesBtn = new Button("Choose images…");
        imagesBtn.setOnAction(e -> chooseImages());
        Button addProjectBtn = new Button("Add project…");
        addProjectBtn.setOnAction(e -> addProject());
        Button clearProjectsBtn = new Button("Clear");
        clearProjectsBtn.setOnAction(e -> {
            extraProjects.clear();
            updateExtraProjectsLabel();
        });
        updateExtraProjectsLabel();
        VBox imgBox = new VBox(
                6,
                new HBox(8, new Label("Images:"), imagesBtn, imagesCountLabel),
                new HBox(8, new Label("Also include projects:"), addProjectBtn, clearProjectsBtn),
                extraProjectsLabel);

        // Batch grouping — assign images to batches via a table editor (auto-detected as a starting point).
        autoDetectBatches();
        Button assignBtn = new Button("Assign batches…");
        assignBtn.setOnAction(e -> assignBatches());
        batchSummaryLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #666;");
        updateBatchSummary();
        HBox batchRow = new HBox(8, assignBtn, batchSummaryLabel);
        batchRow.setAlignment(Pos.CENTER_LEFT);
        VBox batchBox = new VBox(6, new Label("Batch grouping (used for per-batch mode + QC):"), batchRow);

        // Granularity.
        perImageRadio.setToggleGroup(granularityGroup);
        perBatchRadio.setToggleGroup(granularityGroup);
        perImageRadio.setSelected(true);
        perImageRadio.setTooltip(new Tooltip(
                "Align every image to an auto-chosen reference per marker (removes per-image + batch drift)."));
        perBatchRadio.setTooltip(
                new Tooltip("Pool images within a batch and align the batches (preserves within-batch biology)."));
        VBox granBox = new VBox(4, new Label("Granularity:"), perImageRadio, perBatchRadio);

        // Fit params.
        binsSpinner.setValueFactory(
                new SpinnerValueFactory.IntegerSpinnerValueFactory(64, 4096, BatchNormalizerModel.DEFAULT_BINS, 64));
        binsSpinner.setEditable(true);
        binsSpinner.setPrefWidth(90);
        binsSpinner.setTooltip(new Tooltip("Log-intensity histogram bins (UniFORM default 1024)."));
        sampleSpinner.setValueFactory(
                new SpinnerValueFactory.IntegerSpinnerValueFactory(1000, 2_000_000, 50_000, 5000));
        sampleSpinner.setEditable(true);
        sampleSpinner.setPrefWidth(110);
        sampleSpinner.setTooltip(new Tooltip("Max cells sampled per image to build the histograms (bounds memory)."));
        int cpu = Runtime.getRuntime().availableProcessors();
        workersSpinner.setValueFactory(new SpinnerValueFactory.IntegerSpinnerValueFactory(
                1, Math.max(1, cpu), Math.max(1, Math.min(8, cpu - 1)), 1));
        workersSpinner.setEditable(true);
        workersSpinner.setPrefWidth(80);
        workersSpinner.setTooltip(new Tooltip("Images processed in parallel (" + cpu + " CPUs detected)."));
        HBox paramsRow = new HBox(
                12,
                new Label("Bins:"),
                binsSpinner,
                new Label("Cells/image:"),
                sampleSpinner,
                new Label("Workers:"),
                workersSpinner);
        paramsRow.setAlignment(Pos.CENTER_LEFT);

        // Log / progress.
        logArea.setEditable(false);
        logArea.setPrefHeight(150);
        logArea.setStyle("-fx-font-family: monospace; -fx-font-size: 11px;");
        progressBar.setPrefWidth(Double.MAX_VALUE);

        // Buttons.
        runBtn.setDefaultButton(true);
        runBtn.setOnAction(e -> runFit());
        writeBtn.setDisable(true);
        writeBtn.setTooltip(new Tooltip(
                "Materialise \"<marker>: Cell: Mean (batchnorm)\" columns from the last fit (non-destructive)."));
        writeBtn.setOnAction(e -> runWrite());
        qcBtn.setDisable(true);
        qcBtn.setOnAction(e -> showQc());
        cancelBtn.setDisable(true);
        cancelBtn.setOnAction(e -> {
            if (token != null) {
                token.cancel();
                log("Cancelling…");
            }
        });
        closeBtn.setOnAction(e -> stage.close());
        HBox actions = new HBox(10, qcBtn, writeBtn, cancelBtn, runBtn, closeBtn);
        actions.setAlignment(Pos.CENTER_RIGHT);

        VBox root = new VBox(
                10,
                new Label("Correct measurements:"),
                measBox,
                new Separator(),
                imgBox,
                new Separator(),
                batchBox,
                granBox,
                paramsRow,
                new Separator(),
                statusLabel,
                progressBar,
                logArea,
                actions);
        root.setPadding(new Insets(14));

        Stage s = new Stage();
        s.setTitle("Batch Normalisation (UniFORM)");
        s.initOwner(qupath.getStage());
        s.initModality(Modality.NONE);
        s.setScene(new Scene(root, 620, 760));
        return s;
    }

    // ── Measurement selection ────────────────────────────────────────────────

    private void chooseMeasurements() {
        FeatureSelectionPane pane = new FeatureSelectionPane(stage, allFeatureNames, selectedMeasurements);
        pane.setTitle("Choose measurements to batch-correct");
        List<String> chosen = pane.showAndWait();
        if (chosen == null) {
            return;
        }
        // Guard: never correct embeddings — the log-space alignment is invalid for them.
        List<String> embeddings = new ArrayList<>();
        List<String> kept = new ArrayList<>();
        for (String f : chosen) {
            if (FeatureSelectionPane.isEmbedding(f)) {
                embeddings.add(f);
            } else {
                kept.add(f);
            }
        }
        selectedMeasurements = kept;
        updateMeasurementsLabel();
        if (!embeddings.isEmpty()) {
            Dialogs.showWarningNotification(
                    "CellTune",
                    embeddings.size() + " embedding feature(s) were excluded — batch correction only applies to "
                            + "intensity measurements.");
        }
    }

    private void updateMeasurementsLabel() {
        measurementsLabel.setText(selectedMeasurements.size() + " measurement(s) selected");
    }

    // ── Batch grouping ───────────────────────────────────────────────────────

    /** Seed the image→batch map by detecting a {@code batch\d+} token in each name (a starting point only). */
    private void autoDetectBatches() {
        Pattern p = Pattern.compile("(?i)batch\\s*\\d+");
        for (String name : allImageNames) {
            Matcher m = p.matcher(name);
            manualBatch.put(name, m.find() ? m.group() : "(unassigned)");
        }
    }

    /** Open the table editor to assign images to batches. */
    private void assignBatches() {
        List<String> names = allImageNames.isEmpty() ? new ArrayList<>(manualBatch.keySet()) : allImageNames;
        Map<String, String> result = new BatchAssignmentPane(stage, names, manualBatch).showAndWait();
        if (result != null) {
            manualBatch = result;
            updateBatchSummary();
        }
    }

    private void updateBatchSummary() {
        Map<String, Integer> counts = new TreeMap<>();
        for (String name : selectedImages.isEmpty() ? manualBatch.keySet() : selectedImages) {
            counts.merge(manualBatch.getOrDefault(name, "(unassigned)"), 1, Integer::sum);
        }
        batchSummaryLabel.setText(counts.isEmpty() ? "no images" : (counts.size() + " batch(es): " + counts));
    }

    /** Resolve image → batch for the cohort images, defaulting to {@code (unassigned)}. */
    private Map<String, String> resolveBatches(List<ProjectImageEntry<BufferedImage>> entries) {
        Map<String, String> map = new LinkedHashMap<>();
        for (ProjectImageEntry<BufferedImage> e : entries) {
            String name = e.getImageName();
            map.put(name, manualBatch.getOrDefault(name, "(unassigned)"));
        }
        return map;
    }

    // ── Run: fit ─────────────────────────────────────────────────────────────

    private void runFit() {
        var project = currentProject();
        if (project == null && extraProjects.isEmpty()) {
            log("ERROR: No project is open.");
            return;
        }
        if (selectedMeasurements.size() < 1) {
            log("ERROR: Select at least one measurement to correct.");
            return;
        }
        List<ProjectImageEntry<BufferedImage>> entries = buildCohortEntries(project);
        if (entries.size() < 2) {
            log("ERROR: Batch correction needs at least two images.");
            return;
        }
        BatchNormalizerCohort.Mode mode = perBatchRadio.isSelected()
                ? BatchNormalizerCohort.Mode.PER_BATCH
                : BatchNormalizerCohort.Mode.PER_IMAGE;
        Map<String, String> imageToBatch = resolveBatches(entries);
        if (mode == BatchNormalizerCohort.Mode.PER_BATCH) {
            long groups = imageToBatch.values().stream().distinct().count();
            if (groups < 2) {
                log("ERROR: Per-batch mode needs at least two batches — check the grouping (Preview groups).");
                return;
            }
        }
        var params = new BatchNormalizerCohort.Params(
                mode,
                binsSpinner.getValue(),
                new ArrayList<>(selectedMeasurements),
                imageToBatch,
                sampleSpinner.getValue(),
                null);

        ImageData<BufferedImage> openData = qupath.getImageData();
        ProjectImageEntry<BufferedImage> openEntry =
                (openData != null && project != null) ? project.getEntry(openData) : null;
        int workers = workersSpinner.getValue();
        token = new CancellationToken();
        setRunning(true);
        logArea.clear();
        log("Fitting batch normalization over " + entries.size() + " image(s), " + selectedMeasurements.size()
                + " measurement(s), mode=" + mode + "…");

        Thread worker = new Thread(
                () -> {
                    try {
                        NormalizerFit fit = BatchNormalizerCohort.fit(
                                entries, params, openData, openEntry, workers, this::log, this::setProgress, token);
                        if (fit == null) {
                            Platform.runLater(() -> statusLabel.setText("Fit cancelled or no data."));
                            return;
                        }
                        if (project != null) {
                            try {
                                ProjectStateManager.saveBatchShifts(project, fit);
                                log("Saved scales to celltune/batch-shifts.json");
                            } catch (Exception ex) {
                                log("WARN: could not save shifts: " + ex.getMessage());
                            }
                        }
                        String summary = BatchNormQcView.metricSummary(fit);
                        Platform.runLater(() -> {
                            lastFit = fit;
                            writeBtn.setDisable(false);
                            qcBtn.setDisable(false);
                            statusLabel.setText("Fit complete. " + summary);
                            log(summary);
                            showQc();
                        });
                    } catch (Exception ex) {
                        logger.warn("Batch-norm fit failed", ex);
                        log("ERROR: " + ex.getMessage());
                    } finally {
                        Platform.runLater(() -> setRunning(false));
                    }
                },
                "CellTune-BatchNorm-Fit");
        worker.setDaemon(true);
        worker.start();
    }

    // ── Write corrected columns ──────────────────────────────────────────────

    private void runWrite() {
        var project = currentProject();
        BatchShifts shifts = project != null ? ProjectStateManager.loadBatchShifts(project) : null;
        if (shifts == null && lastFit == null) {
            log("ERROR: Run a fit first.");
            return;
        }
        List<ProjectImageEntry<BufferedImage>> entries = buildCohortEntries(project);
        ImageData<BufferedImage> openData = qupath.getImageData();
        ProjectImageEntry<BufferedImage> openEntry =
                (openData != null && project != null) ? project.getEntry(openData) : null;
        int workers = workersSpinner.getValue();
        // Prefer freshly-persisted shifts; fall back to converting the in-memory fit.
        BatchShifts toApply = shifts != null ? shifts : fromFit(lastFit);
        token = new CancellationToken();
        setRunning(true);
        log("Writing corrected columns to " + entries.size() + " image(s)…");
        Thread worker = new Thread(
                () -> {
                    try {
                        var res = BatchNormalizerApply.apply(
                                entries, toApply, openData, openEntry, workers, this::log, this::setProgress, token);
                        Platform.runLater(() -> statusLabel.setText(String.format(
                                Locale.US,
                                "Wrote corrected columns to %,d cells across %d image(s).",
                                res.cellsWritten(),
                                res.imagesWritten())));
                    } catch (Exception ex) {
                        logger.warn("Batch-norm write failed", ex);
                        log("ERROR: " + ex.getMessage());
                    } finally {
                        Platform.runLater(() -> setRunning(false));
                    }
                },
                "CellTune-BatchNorm-Write");
        worker.setDaemon(true);
        worker.start();
    }

    private static BatchShifts fromFit(NormalizerFit fit) {
        BatchShifts s = new BatchShifts();
        s.version = 1;
        s.mode = fit.mode().name();
        s.nBins = fit.nBins();
        s.markers = new ArrayList<>(fit.markers());
        s.imageToBatch = new LinkedHashMap<>(fit.imageToBatch());
        s.scaleByImage = new LinkedHashMap<>(fit.scaleByImage());
        s.refGroupByMarker = new ArrayList<>(List.of(fit.refGroupByMarker()));
        return s;
    }

    private void showQc() {
        if (lastFit == null) {
            log("Run a fit first.");
            return;
        }
        new BatchNormQcView(stage, lastFit).show();
    }

    // ── Shared helpers (mirrors NeighborhoodAnalysisDialog) ───────────────────

    private List<String> discoverFeatureNames() {
        var imageData = qupath.getImageData();
        if (imageData == null) {
            return new ArrayList<>();
        }
        var hierarchy = imageData.getHierarchy();
        var cells = hierarchy.getCellObjects().isEmpty() ? hierarchy.getDetectionObjects() : hierarchy.getCellObjects();
        for (PathObject cell : cells) {
            return new ArrayList<>(cell.getMeasurementList().getMeasurementNames());
        }
        return new ArrayList<>();
    }

    private void chooseImages() {
        if (allImageNames.isEmpty()) {
            Dialogs.showInfoNotification("CellTune", "No project images found.");
            return;
        }
        List<String> chosen =
                new ImageSelectionPane(qupath.getStage(), allImageNames, currentImageName()).showAndWait();
        if (chosen != null) {
            selectedImages = chosen;
            imagesCountLabel.setText(imageCountText());
            updateBatchSummary();
        }
    }

    private String imageCountText() {
        int n = selectedImages.size();
        int total = allImageNames.size();
        return n == total ? ("All " + total + " images") : (n + " of " + total + " images");
    }

    private void addProject() {
        File f = FileChoosers.promptForFile(
                "Select another QuPath project (.qpproj)",
                FileChoosers.createExtensionFilter("QuPath project", "*.qpproj"));
        if (f == null) {
            return;
        }
        try {
            Project<BufferedImage> p = ProjectIO.loadProject(f, BufferedImage.class);
            extraProjects.add(p);
            updateExtraProjectsLabel();
            log("Added project (" + p.getImageList().size() + " images).");
        } catch (Exception ex) {
            Dialogs.showErrorNotification("CellTune", "Could not load project: " + ex.getMessage());
        }
    }

    private void updateExtraProjectsLabel() {
        int imgs = 0;
        for (Project<BufferedImage> p : extraProjects) {
            imgs += p.getImageList().size();
        }
        extraProjectsLabel.setText(
                extraProjects.isEmpty()
                        ? "No extra projects — this project only."
                        : (extraProjects.size() + " added project(s), +" + imgs + " images"));
        extraProjectsLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #666;");
    }

    @SuppressWarnings("unchecked")
    private Project<BufferedImage> currentProject() {
        return (Project<BufferedImage>) (Object) qupath.getProject();
    }

    private List<ProjectImageEntry<BufferedImage>> buildCohortEntries(Project<BufferedImage> current) {
        List<ProjectImageEntry<BufferedImage>> entries = new ArrayList<>();
        if (current != null) {
            Map<String, ProjectImageEntry<BufferedImage>> byName = new LinkedHashMap<>();
            for (ProjectImageEntry<BufferedImage> e : current.getImageList()) {
                byName.put(e.getImageName(), e);
            }
            for (String name : selectedImages) {
                ProjectImageEntry<BufferedImage> e = byName.get(name);
                if (e != null) {
                    entries.add(e);
                }
            }
        }
        for (Project<BufferedImage> p : extraProjects) {
            entries.addAll(p.getImageList());
        }
        return entries;
    }

    private String currentImageName() {
        var open = qupath.getImageData();
        var project = qupath.getProject();
        if (open != null && project != null) {
            var entry = project.getEntry(open);
            if (entry != null) {
                return entry.getImageName();
            }
        }
        return null;
    }

    private void setRunning(boolean running) {
        runBtn.setDisable(running);
        writeBtn.setDisable(running || lastFit == null);
        qcBtn.setDisable(running || lastFit == null);
        cancelBtn.setDisable(!running);
        progressBar.setProgress(running ? ProgressBar.INDETERMINATE_PROGRESS : 0);
    }

    private void setProgress(double frac) {
        Platform.runLater(() -> progressBar.setProgress(frac));
    }

    private void log(String msg) {
        Platform.runLater(() -> logArea.appendText(msg + "\n"));
    }
}
