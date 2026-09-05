package qupath.ext.spclassify.ui;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.function.DoubleConsumer;
import java.util.stream.Collectors;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.RadioButton;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Separator;
import javafx.scene.control.TextArea;
import javafx.scene.control.ToggleGroup;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.stage.Modality;
import javafx.stage.Stage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.model.CellFeatureExtractor;
import qupath.ext.spclassify.model.CofactorEstimator;
import qupath.ext.spclassify.model.CohortClusterModel;
import qupath.ext.spclassify.model.IntensityHeatmap;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.prefs.PathPrefs;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;

/**
 * Non-modal tool window for the Phase 17 cofactor-suggestion workflow (COF-02/03/05/06/08). It
 * composes existing machinery into a small UI: an independent calibration picker (a reused
 * {@link FeatureSelectionPane}, defaulting to per-marker whole-cell mean intensities), a scope
 * selector (open image / whole project, defaulting to whole project), a user-triggered Run that
 * computes off the JavaFX thread, a per-feature diagnostics table, one prominent recommended global
 * cofactor, and an Apply button that hands the clamped global back to the caller via a
 * {@link DoubleConsumer} — it mutates no measurements or normalizer state.
 *
 * <p><b>Ownership (critical correction #2 / D-05).</b> The window is owned by the
 * <em>ClusteringNormalizationPane's own stage</em> (passed in as {@code ownerStage}), NOT the QuPath main
 * stage. The Clustering normalisation pane is {@code APPLICATION_MODAL}; a window owned by the QuPath main stage
 * would be blocked by it, whereas a {@code Modality.NONE} child of the modal owner stays
 * interactive. Plan 17-04 wires the real owner via {@code clusteringNormalizationPane.getStage()}.
 *
 * <p>All background work marshals every scene-graph touch through {@link Platform#runLater} — the
 * worker thread never touches nodes directly (CLAUDE.md UI-thread safety).
 */
public class CofactorSuggestionDialog {

    private static final Logger logger = LoggerFactory.getLogger(CofactorSuggestionDialog.class);

    /** Arcsinh cofactor spinner lower bound — the normalizer rejects a cofactor {@code <= 0}. */
    private static final double MIN_COFACTOR = 0.01;

    /** Spinner upper bound for the arcsinh cofactor. */
    private static final double MAX_COFACTOR = 10000.0;

    private final QuPathGUI qupath;
    private final List<String> allFeatureNames;
    private final DoubleConsumer applyGlobalCofactor;
    // Non-final: read inside the calibration-picker lambda, which is defined before assignment below.
    private Stage stage;

    /** Calibration set — independent of the normalize set; defaults to per-marker means (COF-02). */
    private List<String> selectedFeatures;

    /** Last successful run's result; guards Apply and is read from the FX thread. */
    private volatile CofactorEstimator.CofactorSuggestion lastResult;

    /** The in-flight run's cancellation token (null when idle); the Cancel button flips it. */
    private volatile CohortClusterModel.CancellationToken activeToken;

    // Controls kept as fields so the Task-2 Run pipeline can drive them.
    private final ToggleGroup scopeGroup = new ToggleGroup();
    private final RadioButton openImageScope = new RadioButton("Open image");
    private final RadioButton wholeProjectScope = new RadioButton("Whole project (all cells)");
    private final Label calibCountLabel = new Label();
    private final ProgressBar progressBar = new ProgressBar(0);
    private final Button runBtn = new Button("Run");
    private final Button cancelBtn = new Button("Cancel");
    private final Button applyBtn = new Button("Apply");
    private final TextArea logArea = new TextArea();
    private final GridPane resultsGrid = new GridPane();
    private final Label globalLabel = new Label("Recommended cofactor: —");

    // Distribution plot (enhancement): raw-intensity histogram with p50/p99 markers, symlog x-axis,
    // toggling between a single chosen marker and a small-multiples grid of all calibration markers.
    private final ToggleGroup plotViewGroup = new ToggleGroup();
    private final RadioButton plotSingleView = new RadioButton("Single marker");
    private final RadioButton plotGridView = new RadioButton("All markers (grid)");
    private final ChoiceBox<String> markerChoice = new ChoiceBox<>();
    private final VBox plotContainer = new VBox(8);

    /** Compact per-feature histograms from the last run (built off-FX; read on FX to draw). */
    private volatile List<FeatureHistogram> lastHistograms = List.of();

    /**
     * Create the cofactor-suggestion window.
     *
     * @param ownerStage          the ClusteringNormalizationPane's OWN stage (correction #2) — NOT the QuPath
     *                            main stage; the window is owned by it and non-modal
     * @param qupath              QuPath handle for {@code getProject()} / {@code getImageData()}
     * @param allFeatureNames     the full measurement list from the pane (the pick-from universe)
     * @param applyGlobalCofactor callback that writes the chosen global cofactor into the normalize
     *                            spinner (COF-07 handoff); invoked only on Apply, with a clamped value
     */
    public CofactorSuggestionDialog(
            Stage ownerStage, QuPathGUI qupath, List<String> allFeatureNames, DoubleConsumer applyGlobalCofactor) {
        this.qupath = qupath;
        this.allFeatureNames = allFeatureNames == null ? List.of() : List.copyOf(allFeatureNames);
        this.applyGlobalCofactor = applyGlobalCofactor;
        // Default calibration set = per-marker whole-cell mean intensities (COF-02, D-12) — no hardcoded name.
        this.selectedFeatures = IntensityHeatmap.discoverMarkerFeatures(this.allFeatureNames);

        // 1) Calibration picker — a reused FeatureSelectionPane, independent of the normalize set (COF-02).
        Button calibBtn = new Button("Calibration features…");
        updateCalibLabel();
        calibBtn.setOnAction(e -> {
            FeatureSelectionPane picker = new FeatureSelectionPane(stage, this.allFeatureNames, selectedFeatures);
            picker.setTitle("Calibration features");
            List<String> chosen = picker.showAndWait(); // transient blocking picker, owned by this stage
            if (chosen != null) {
                selectedFeatures = chosen;
                updateCalibLabel();
            }
        });
        HBox calibRow = new HBox(8, calibBtn, calibCountLabel);
        calibRow.setAlignment(Pos.CENTER_LEFT);

        // 2) Scope selector — whole project is the default (D-13); disabled with no project open.
        openImageScope.setToggleGroup(scopeGroup);
        wholeProjectScope.setToggleGroup(scopeGroup);
        wholeProjectScope.setSelected(true);
        if (qupath == null || qupath.getProject() == null) {
            wholeProjectScope.setDisable(true);
            openImageScope.setSelected(true);
        }
        VBox scopeBox = new VBox(4, new Label("Compute over:"), openImageScope, wholeProjectScope);

        // 3) Run / Cancel / progress / log / results / global / apply.
        cancelBtn.setDisable(true);
        applyBtn.setDisable(true);
        runBtn.setOnAction(e -> runSuggestion());
        HBox runRow = new HBox(8, runBtn, cancelBtn);
        runRow.setAlignment(Pos.CENTER_LEFT);

        logArea.setEditable(false);
        logArea.setPrefRowCount(5);
        logArea.setWrapText(true);

        resultsGrid.setHgap(12);
        resultsGrid.setVgap(4);
        resultsGrid.setPadding(new Insets(4));
        ScrollPane resultsScroll = new ScrollPane(resultsGrid);
        resultsScroll.setFitToWidth(true);
        resultsScroll.setPrefViewportHeight(260);

        globalLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 15px;");

        // 4) Apply — single action; writes ONLY the clamped global back via the callback. No mutation (COF-07).
        applyBtn.setOnAction(e -> {
            CofactorEstimator.CofactorSuggestion r = lastResult;
            if (r != null && applyGlobalCofactor != null) {
                double g = Math.max(MIN_COFACTOR, Math.min(MAX_COFACTOR, r.globalCofactor()));
                applyGlobalCofactor.accept(g);
                log(String.format("Applied global cofactor %.4g to the normalize spinner.", g));
            }
        });
        HBox applyRow = new HBox(8, applyBtn);
        applyRow.setAlignment(Pos.CENTER_LEFT);

        // 4b) Distribution plot controls (enhancement): single-marker vs all-markers grid, symlog x.
        plotSingleView.setToggleGroup(plotViewGroup);
        plotGridView.setToggleGroup(plotViewGroup);
        plotSingleView.setSelected(true);
        markerChoice.setDisable(true);
        plotViewGroup.selectedToggleProperty().addListener((obs, oldT, newT) -> {
            markerChoice.setDisable(!plotSingleView.isSelected());
            refreshPlot();
        });
        markerChoice.getSelectionModel().selectedItemProperty().addListener((obs, oldV, newV) -> {
            if (plotSingleView.isSelected()) {
                refreshPlot();
            }
        });
        HBox plotControls = new HBox(8, plotSingleView, plotGridView, new Label("Marker:"), markerChoice);
        plotControls.setAlignment(Pos.CENTER_LEFT);
        plotContainer.setPadding(new Insets(4));
        plotContainer.getChildren().add(new Label("Run a suggestion to see distributions."));
        ScrollPane plotScroll = new ScrollPane(plotContainer);
        plotScroll.setFitToWidth(true);
        plotScroll.setPrefViewportHeight(320);

        // 5) Owned NON-MODAL stage (correction #2 / D-05).
        VBox root = new VBox(
                12,
                new Label("Suggest an arcsinh cofactor from the background level of the calibration features."),
                calibRow,
                new Separator(),
                scopeBox,
                new Separator(),
                runRow,
                progressBar,
                new Label("Log:"),
                logArea,
                new Separator(),
                new Label("Per-feature diagnostics:"),
                resultsScroll,
                globalLabel,
                applyRow,
                new Separator(),
                new Label("Raw intensity distributions (symlog x — p50/cofactor solid, p99 dashed):"),
                plotControls,
                plotScroll);
        root.setPadding(new Insets(14));

        ScrollPane rootScroll = new ScrollPane(root);
        rootScroll.setFitToWidth(true);

        stage = new Stage();
        stage.setTitle("Suggest cofactor");
        stage.initOwner(ownerStage);
        stage.initModality(Modality.NONE);
        stage.setScene(new Scene(rootScroll, 640, 840));
    }

    /** Show the (non-blocking) window — mirrors {@code NeighborhoodAnalysisDialog.show()}. */
    public void show() {
        stage.show();
    }

    private void updateCalibLabel() {
        calibCountLabel.setText(selectedFeatures.size() + " calibration feature(s) selected");
    }

    /**
     * User-triggered Run (D-07 — never auto-on-open): snapshot the calibration set and scope, then
     * spawn a daemon worker that computes off the FX thread. Every scene-graph touch below marshals
     * through {@link Platform#runLater}.
     */
    private void runSuggestion() {
        final List<String> features = new ArrayList<>(selectedFeatures);
        if (features.isEmpty()) {
            Dialogs.showWarningNotification("Suggest cofactor", "Select at least one calibration feature.");
            return;
        }
        final boolean wholeProject = wholeProjectScope.isSelected();

        runBtn.setDisable(true);
        cancelBtn.setDisable(false);
        applyBtn.setDisable(true);
        progressBar.setProgress(ProgressBar.INDETERMINATE_PROGRESS);
        logArea.clear();

        final CohortClusterModel.CancellationToken token = new CohortClusterModel.CancellationToken();
        activeToken = token;
        cancelBtn.setOnAction(e -> {
            token.cancel();
            log("Cancelling…");
        });

        Thread worker = new Thread(
                () -> {
                    try {
                        computeAndRender(features, wholeProject, token);
                    } catch (Exception ex) {
                        logger.warn("Cofactor suggestion run failed", ex);
                        log("ERROR: " + ex.getMessage());
                    } finally {
                        activeToken = null;
                        Platform.runLater(() -> {
                            runBtn.setDisable(false);
                            cancelBtn.setDisable(true);
                            progressBar.setProgress(0);
                        });
                    }
                },
                "SpClassify-CofactorSuggest");
        worker.setDaemon(true);
        worker.start();
    }

    /**
     * OFF the FX thread: build one raw per-cell column per calibration feature from the chosen scope,
     * run the estimator, cache the result, and marshal the table + global render back onto FX.
     */
    private void computeAndRender(
            List<String> features, boolean wholeProject, CohortClusterModel.CancellationToken token) {
        double[][] columns = wholeProject ? poolWholeProject(features, token) : poolOpenImage(features);
        if (columns == null) {
            return; // guarded / cancelled — already logged
        }
        CofactorEstimator.CofactorSuggestion result =
                CofactorEstimator.estimate(features.toArray(new String[0]), columns);
        this.lastResult = result;
        // Build compact per-feature histograms now (columns still in scope) so we retain only small
        // bin-count summaries, not the millions of raw per-cell values, after the run.
        this.lastHistograms = buildHistograms(features, columns, result);
        log("Done. Recommended cofactor: " + fmt(result.globalCofactor()) + ".");
        Platform.runLater(() -> {
            renderResults(result);
            populateMarkerChoice(features);
            refreshPlot();
            applyBtn.setDisable(false);
        });
    }

    /**
     * Open-image scope (COF-03): read every detection's RAW measurements in memory via a
     * null-normalizer {@link CellFeatureExtractor} and transpose the row-major matrix into per-feature
     * columns. No geojson/file streaming.
     */
    private double[][] poolOpenImage(List<String> features) {
        ImageData<BufferedImage> data = qupath == null ? null : qupath.getImageData();
        if (data == null) {
            log("No image is open.");
            return null;
        }
        List<PathObject> cells = new ArrayList<>(data.getHierarchy().getDetectionObjects());
        if (cells.isEmpty()) {
            log("The open image has no detections.");
            return null;
        }
        log("Reading " + String.format("%,d", cells.size()) + " cells from the open image…");
        CellFeatureExtractor extractor = new CellFeatureExtractor(features, null); // null normalizer → RAW reads
        float[] flat = extractor.extractMatrix(cells); // row-major nCells*nFeatures
        int nF = features.size();
        int nC = cells.size();
        double[][] columns = new double[nF][nC];
        for (int c = 0; c < nC; c++) {
            int offset = c * nF;
            for (int f = 0; f < nF; f++) {
                columns[f][c] = flat[offset + f];
            }
        }
        return columns;
    }

    /**
     * Whole-project scope (COF-03/COF-08/D-08): a count-only soft-ceiling pre-scan + confirm, then
     * pool every cell's RAW row via {@link CohortClusterModel#poolAllCellsRaw} and transpose into
     * per-feature columns. Cancellation-aware throughout.
     */
    private double[][] poolWholeProject(List<String> features, CohortClusterModel.CancellationToken token) {
        Project<BufferedImage> project = qupath == null ? null : qupath.getProject();
        if (project == null) {
            log("No project is open.");
            return null;
        }
        List<String> images = project.getImageList().stream()
                .map(ProjectImageEntry::getImageName)
                .collect(Collectors.toList());
        ImageData<BufferedImage> openData = qupath.getImageData();
        ProjectImageEntry<BufferedImage> openEntry = openData == null ? null : project.getEntry(openData);
        String openName = openEntry == null ? null : openEntry.getImageName();

        // Soft-ceiling confirm (D-08): count-only pre-scan (no feature extraction), reusing the SAME pref key.
        log("Estimating cell count across " + images.size() + " image(s)…");
        long estimate = estimatePooledCellCount(project, images, openData, openName, token);
        if (token.isCancelled()) {
            log("Cancelled.");
            return null;
        }
        int ceiling = PathPrefs.createPersistentPreference("celltune.allCellsSoftCeiling", 50_000_000)
                .get();
        if (estimate > ceiling) {
            boolean proceed = confirmOnFx(
                    "Large all-cells run",
                    String.format(
                            "The selected images have an estimated %,d cells, above the configured %,d-cell "
                                    + "soft ceiling. This can take a long time and use a lot of memory. "
                                    + "Continue anyway?",
                            estimate, ceiling));
            if (!proceed) {
                log("Cancelled — estimated " + String.format("%,d", estimate) + " cells exceeds the soft ceiling.");
                return null;
            }
        }

        log("Pooling raw cells across the project…");
        CohortClusterModel.PooledData pooled = CohortClusterModel.poolAllCellsRaw(
                project, images, features, null, null, openData, openName, token, this::log);
        if (pooled.cancelled()) {
            log("Cancelled.");
            return null;
        }
        double[][] rows = pooled.rows(); // [nCells][nFeatures], RAW
        int nC = rows.length;
        int nF = features.size();
        if (nC == 0) {
            log("No cells were pooled.");
            return null;
        }
        double[][] columns = new double[nF][nC];
        for (int c = 0; c < nC; c++) {
            double[] row = rows[c];
            for (int f = 0; f < nF; f++) {
                columns[f][c] = row[f];
            }
        }
        return columns;
    }

    /**
     * Cheap count-only pre-scan for the soft-ceiling estimate (D-08): sums each image's detection
     * count WITHOUT extracting features, using the live open {@link ImageData} for the open entry so
     * its unsaved edits are counted. Cancellation-aware between images.
     */
    private long estimatePooledCellCount(
            Project<BufferedImage> project,
            List<String> images,
            ImageData<BufferedImage> openData,
            String openName,
            CohortClusterModel.CancellationToken token) {
        Map<String, ProjectImageEntry<BufferedImage>> byName = new LinkedHashMap<>();
        for (ProjectImageEntry<BufferedImage> entry : project.getImageList()) {
            byName.put(entry.getImageName(), entry);
        }
        long total = 0;
        for (String name : images) {
            if (token.isCancelled()) {
                break;
            }
            if (openData != null && name.equals(openName)) {
                total += openData.getHierarchy().getDetectionObjects().size();
                continue;
            }
            ProjectImageEntry<BufferedImage> entry = byName.get(name);
            if (entry == null) {
                continue;
            }
            try {
                ImageData<BufferedImage> imageData = entry.readImageData();
                if (imageData != null) {
                    total += imageData.getHierarchy().getDetectionObjects().size();
                }
            } catch (Exception e) {
                logger.debug("Soft-ceiling pre-scan: could not read [{}] — skipped", name, e);
            }
        }
        return total;
    }

    /**
     * Show a confirm dialog from a background thread by marshalling it onto the FX thread and blocking
     * on a {@link CountDownLatch} until answered — mirrors {@code ScatterPlotView.confirmOnFx}.
     */
    private boolean confirmOnFx(String title, String message) {
        final boolean[] result = {false};
        CountDownLatch latch = new CountDownLatch(1);
        Platform.runLater(() -> {
            try {
                result[0] = Dialogs.showConfirmDialog(title, message);
            } finally {
                latch.countDown();
            }
        });
        try {
            latch.await();
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        }
        return result[0];
    }

    /**
     * Render the per-feature diagnostics table (D-09) and the prominent recommended global cofactor
     * (D-10) on the FX thread. Per-feature values are advisory/display only; excluded rows are greyed.
     */
    private void renderResults(CofactorEstimator.CofactorSuggestion result) {
        resultsGrid.getChildren().clear();
        String[] headers = {"feature", "n cells", "background", "median", "p99", "suggested cofactor", "flag"};
        for (int col = 0; col < headers.length; col++) {
            resultsGrid.add(boldLabel(headers[col]), col, 0);
        }
        int row = 1;
        for (CofactorEstimator.FeatureCofactor fc : result.perFeature()) {
            Label[] cells = {
                new Label(fc.feature()),
                new Label(String.format("%,d", fc.nCells())),
                new Label(fmt(fc.background())),
                new Label(fmt(fc.median())),
                new Label(fmt(fc.p99())),
                new Label(fmt(fc.cofactor())),
                new Label(fc.excluded() ? fc.reason() : "")
            };
            if (fc.excluded()) {
                for (Label l : cells) {
                    l.setStyle("-fx-text-fill: grey;");
                }
            }
            for (int col = 0; col < cells.length; col++) {
                resultsGrid.add(cells[col], col, row);
            }
            row++;
        }
        globalLabel.setText(String.format("Recommended cofactor: %.4g", result.globalCofactor()));
    }

    /** Compact numeric formatter — shows an em dash for NaN (a degenerate/empty feature). */
    private static String fmt(double v) {
        return Double.isNaN(v) ? "—" : String.format("%.4g", v);
    }

    /** Marshal a log line onto the FX thread. */
    private void log(String msg) {
        Platform.runLater(() -> logArea.appendText(msg + "\n"));
    }

    /** Bold header label (copied from ClusterAssignmentPane). */
    private static Label boldLabel(String text) {
        Label l = new Label(text);
        l.setStyle("-fx-font-weight: bold;");
        return l;
    }

    /** Build one compact symlog histogram per calibration feature from the raw per-cell columns. */
    private List<FeatureHistogram> buildHistograms(
            List<String> features, double[][] columns, CofactorEstimator.CofactorSuggestion result) {
        List<FeatureHistogram> out = new ArrayList<>(columns.length);
        for (int f = 0; f < columns.length; f++) {
            CofactorEstimator.FeatureCofactor fc =
                    f < result.perFeature().size() ? result.perFeature().get(f) : null;
            String name = f < features.size() ? features.get(f) : ("Feature " + (f + 1));
            out.add(FeatureHistogram.of(name, columns[f], fc));
        }
        return out;
    }

    /** Populate the single-marker selector with this run's calibration features. */
    private void populateMarkerChoice(List<String> features) {
        markerChoice.getItems().setAll(features);
        if (!features.isEmpty() && markerChoice.getSelectionModel().isEmpty()) {
            markerChoice.getSelectionModel().select(0);
        }
        markerChoice.setDisable(!plotSingleView.isSelected());
    }

    /** Redraw the distribution area: one large canvas (single marker) or a 2-col small-multiples grid. */
    private void refreshPlot() {
        plotContainer.getChildren().clear();
        List<FeatureHistogram> hs = lastHistograms;
        if (hs.isEmpty()) {
            plotContainer.getChildren().add(new Label("Run a suggestion to see distributions."));
            return;
        }
        if (plotGridView.isSelected()) {
            GridPane grid = new GridPane();
            grid.setHgap(8);
            grid.setVgap(8);
            int cols = 2;
            for (int i = 0; i < hs.size(); i++) {
                Canvas c = new Canvas(285, 150);
                drawSymlogHistogram(c.getGraphicsContext2D(), hs.get(i), 285, 150, true);
                grid.add(c, i % cols, i / cols);
            }
            plotContainer.getChildren().add(grid);
        } else {
            String sel = markerChoice.getSelectionModel().getSelectedItem();
            FeatureHistogram h =
                    hs.stream().filter(x -> x.feature().equals(sel)).findFirst().orElse(hs.get(0));
            Canvas c = new Canvas(580, 300);
            drawSymlogHistogram(c.getGraphicsContext2D(), h, 580, 300, false);
            plotContainer.getChildren().add(c);
        }
    }

    /**
     * Draw a raw-intensity histogram on a symlog x-axis: a linear window around 0 (captures the
     * zero-inflated background spike) then log for signal, with the p50 (= cofactor, solid) and p99
     * (dashed) guide lines. Mirrors the offline demo plots.
     */
    private void drawSymlogHistogram(GraphicsContext gc, FeatureHistogram h, double w, double hgt, boolean mini) {
        gc.clearRect(0, 0, w, hgt);
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, w, hgt);
        double ml = mini ? 30 : 46, mr = 10, mt = mini ? 16 : 22, mb = mini ? 18 : 28;
        double pw = w - ml - mr, ph = hgt - mt - mb;
        int[] cts = h.counts();
        int nb = cts.length;
        int maxC = 1;
        for (int c : cts) {
            if (c > maxC) {
                maxC = c;
            }
        }
        double tmax = h.tmax();

        // bars
        gc.setFill(h.excluded() ? Color.web("#AAAAAA", 0.6) : Color.web("#2E6FB0", 0.6));
        for (int i = 0; i < nb; i++) {
            double bh = (double) cts[i] / maxC * ph;
            double x = ml + (double) i / nb * pw;
            gc.fillRect(x, mt + ph - bh, pw / nb + 0.7, bh);
        }

        // x-axis + ticks (0, then decades 1,10,100,…)
        gc.setStroke(Color.GRAY);
        gc.setLineWidth(1);
        gc.setLineDashes(0);
        gc.strokeLine(ml, mt + ph, ml + pw, mt + ph);
        gc.setFill(Color.DIMGRAY);
        gc.setFont(Font.font(mini ? 8 : 10));
        gc.strokeLine(ml, mt + ph, ml, mt + ph + 3);
        gc.fillText("0", ml - 3, mt + ph + (mini ? 11 : 13));
        int dmax = h.max() <= 1 ? 0 : (int) Math.floor(Math.log10(h.max()));
        for (int d = 0; d <= dmax; d++) {
            double t = 1.0 + d;
            if (t > tmax + 1e-9) {
                break;
            }
            double px = ml + t / tmax * pw;
            gc.strokeLine(px, mt + ph, px, mt + ph + 3);
            String lab = d == 0 ? "1" : (d <= 3 ? String.valueOf((long) Math.pow(10, d)) : "1e" + d);
            gc.fillText(lab, px - (mini ? 5 : 7), mt + ph + (mini ? 11 : 13));
        }

        // p50 (= cofactor, solid) and p99 (dashed)
        gc.setStroke(Color.BLACK);
        if (!Double.isNaN(h.p50())) {
            double px = ml + h.tOf(h.p50()) / tmax * pw;
            gc.setLineWidth(mini ? 1.3 : 2.0);
            gc.setLineDashes(0);
            gc.strokeLine(px, mt, px, mt + ph);
        }
        if (!Double.isNaN(h.p99())) {
            double px = ml + h.tOf(h.p99()) / tmax * pw;
            gc.setLineWidth(1.0);
            gc.setLineDashes(3);
            gc.strokeLine(px, mt, px, mt + ph);
            gc.setLineDashes(0);
        }

        // title + y hint
        gc.setFill(h.excluded() ? Color.GRAY : Color.BLACK);
        gc.setFont(Font.font(mini ? 9 : 11));
        String title = h.feature();
        if (mini && title.length() > 22) {
            title = title.substring(0, 20) + "…";
        }
        if (!mini && !Double.isNaN(h.cofactor())) {
            title += "    cofactor = " + fmt(h.cofactor()) + (h.excluded() ? " (excluded)" : "");
        }
        gc.fillText(title, ml, mt - 5);
        if (!mini) {
            gc.setFill(Color.DIMGRAY);
            gc.setFont(Font.font(9));
            gc.fillText(String.format("%,d", maxC), 2, mt + 8);
            gc.fillText("cells", 6, mt + ph);
        }
    }

    /**
     * Compact symlog histogram summary of one feature's raw per-cell values — bin counts plus the
     * p50/p99/cofactor markers — so the dialog retains only small arrays, not the raw cells.
     */
    private record FeatureHistogram(
            String feature,
            int n,
            long nZero,
            double min,
            double max,
            double p50,
            double p99,
            double cofactor,
            boolean excluded,
            double linthresh,
            double tmax,
            int[] counts) {

        private static final int NBINS = 64;
        private static final double LINTHRESH = 1.0;

        /** symlog transform: linear in [0, linthresh] → [0,1], then +1 axis unit per decade above. */
        double tOf(double x) {
            double xx = x < 0 ? 0 : x;
            return xx <= linthresh ? xx / linthresh : 1.0 + Math.log10(xx / linthresh);
        }

        static FeatureHistogram of(String name, double[] v, CofactorEstimator.FeatureCofactor fc) {
            double mn = Double.POSITIVE_INFINITY, mx = 0.0;
            int n = 0;
            long nz = 0;
            for (double x : v) {
                if (Double.isNaN(x)) {
                    continue;
                }
                n++;
                if (x == 0.0) {
                    nz++;
                }
                if (x < mn) {
                    mn = x;
                }
                if (x > mx) {
                    mx = x;
                }
            }
            double tmax = mx <= LINTHRESH ? 1.0 : 1.0 + Math.log10(mx / LINTHRESH);
            int[] counts = new int[NBINS];
            for (double x : v) {
                if (Double.isNaN(x)) {
                    continue;
                }
                double xx = x < 0 ? 0 : x;
                double t = xx <= LINTHRESH ? xx / LINTHRESH : 1.0 + Math.log10(xx / LINTHRESH);
                int b = (int) (t / tmax * NBINS);
                if (b < 0) {
                    b = 0;
                }
                if (b >= NBINS) {
                    b = NBINS - 1;
                }
                counts[b]++;
            }
            double p50 = fc != null ? fc.background() : Double.NaN;
            double p99 = fc != null ? fc.p99() : Double.NaN;
            double cof = fc != null ? fc.cofactor() : Double.NaN;
            boolean exc = fc != null && fc.excluded();
            return new FeatureHistogram(
                    name,
                    n,
                    nz,
                    (mn == Double.POSITIVE_INFINITY ? 0 : mn),
                    mx,
                    p50,
                    p99,
                    cof,
                    exc,
                    LINTHRESH,
                    tmax,
                    counts);
        }
    }
}
