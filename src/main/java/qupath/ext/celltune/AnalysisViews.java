package qupath.ext.celltune;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.ResourceBundle;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import javafx.application.Platform;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.celltune.classifier.DualModelClassifier;
import qupath.ext.celltune.model.CellFeatureExtractor;
import qupath.ext.celltune.model.FeatureNormalizer;
import qupath.ext.celltune.model.ImagePixelStats;
import qupath.ext.celltune.model.ImagePixelStatsReader;
import qupath.ext.celltune.model.IntensityHeatmap;
import qupath.ext.celltune.model.PixelCohortAnalyzer;
import qupath.ext.celltune.model.PopulationSet;
import qupath.ext.celltune.ui.DistanceMeasurementsDialog;
import qupath.ext.celltune.ui.FeatureImportanceView;
import qupath.ext.celltune.ui.FeatureSelectionPane;
import qupath.ext.celltune.ui.IntensityHeatmapView;
import qupath.ext.celltune.ui.NeighborhoodAnalysisDialog;
import qupath.ext.celltune.ui.PixelPrescreenView;
import qupath.ext.celltune.ui.ScatterPlotView;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjectFilter;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.classes.PathClassTools;
import qupath.lib.projects.ProjectImageEntry;

/**
 * Read-only analysis/visualisation view launchers lifted out of {@code CellTuneExtension}.
 * <p>
 * Each method validates its preconditions and opens an analysis window (distance
 * measurements, intensity heatmap, scatter plot, pixel prescreen, feature importance).
 * They are grouped here because none of them write back to the extension's session state —
 * the only state they need (current predictions, the trained classifier, the feature
 * normalizer) is passed in as a parameter, and the {@code Class Control} re-launch is passed
 * as a callback. Lifted verbatim; the extension's menu handlers delegate here (mirroring the
 * {@code UtilityScripts}/{@code AnnotationRegionExporter}/{@code ProjectPredictionSummary} moves).
 */
final class AnalysisViews {

    private static final ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.celltune.ui.strings");
    private static final String EXTENSION_NAME = resources.getString("name");
    private static final Logger logger = LoggerFactory.getLogger(AnalysisViews.class);

    private AnalysisViews() {} // utility class

    // Retained scatter/clustering window, reused across menu launches for the same open image so its
    // clustering results survive a close/reopen (the user need not re-select measurements or re-cluster).
    // Dropped when a different image is open, since its cached cells/fit would be stale.
    private static ScatterPlotView openScatterPlot;

    /** Open the same-class / cross-class distance-measurement dialog. */
    static void showDistanceMeasurements(QuPathGUI qupath) {
        if (qupath.getProject() == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, resources.getString("classify.no_project"));
            return;
        }
        new DistanceMeasurementsDialog(qupath).show();
    }

    /**
     * Open the intensity-heatmap window: mean whole-cell marker intensity per
     * predicted cell class, coloured by per-marker z-score across classes. The
     * window can switch between the current image, any individual project image,
     * or a project-wide pooled heatmap, and exports to PNG/CSV.
     */
    static void showIntensityHeatmaps(QuPathGUI qupath) {
        var imageData = qupath.getImageData();
        if (imageData == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No image is open.");
            return;
        }

        Collection<PathObject> cells = imageData.getHierarchy().getObjects(null, PathObject.class).stream()
                .filter(PathObjectFilter.DETECTIONS_ALL)
                .toList();
        if (cells.isEmpty()) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No detections found. Run cell detection first.");
            return;
        }

        List<String> allFeatures = CellFeatureExtractor.discoverFeatureNames(cells);
        if (allFeatures.isEmpty()) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No cell measurements found.");
            return;
        }

        // Let the user choose which measurements to plot. Pre-select the
        // auto-discovered whole-cell marker means ("<marker>: Cell: Mean"); a
        // null/empty pre-selection makes FeatureSelectionPane default to all.
        List<String> markerDefaults = IntensityHeatmap.discoverMarkerFeatures(allFeatures);
        var selectionPane = new FeatureSelectionPane(
                qupath.getStage(), allFeatures, markerDefaults.isEmpty() ? null : markerDefaults);
        selectionPane.setTitle("Select Measurements for Intensity Heatmap");
        List<String> markerFeatures = selectionPane.showAndWait();
        if (markerFeatures == null) {
            return; // user cancelled
        }
        if (markerFeatures.isEmpty()) {
            Dialogs.showWarningNotification(EXTENSION_NAME, "No measurements selected — nothing to plot.");
            return;
        }

        String currentImageName = null;
        var project = qupath.getProject();
        if (project != null) {
            var entry = project.getEntry(imageData);
            if (entry != null) {
                currentImageName = entry.getImageName();
            }
        }
        if (currentImageName == null || currentImageName.isBlank()) {
            currentImageName = imageData.getServer().getMetadata().getName();
        }

        var acc = new IntensityHeatmap.Accumulator(markerFeatures);
        acc.add(cells);
        IntensityHeatmap.Result result = acc.build();

        new IntensityHeatmapView(qupath.getStage(), qupath, currentImageName, markerFeatures, result).show();
    }

    /**
     * Open the interactive scatter-plot window: cells projected into a 2D PCA or
     * UMAP embedding of their selected marker means, coloured by k-means cluster,
     * predicted class, or marker intensity. Box/lasso selection on the plot
     * selects the corresponding cells in the viewer, and viewer selection is
     * mirrored back onto the plot. Mirrors {@link #showIntensityHeatmaps} for the
     * cell/feature/marker discovery flow.
     *
     * @param liveCurrentPredictions the open image's in-memory predictions (may be null)
     * @param normalizer             the session feature normalizer (may be null)
     * @param openClassControl       callback to launch the Class Control dialog
     */
    static void showScatterPlot(
            QuPathGUI qupath,
            PopulationSet liveCurrentPredictions,
            FeatureNormalizer normalizer,
            Runnable openClassControl) {
        var imageData = qupath.getImageData();
        if (imageData == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No image is open.");
            return;
        }

        String currentImageName = resolveImageName(qupath, imageData);

        // Reuse the retained plot so its clustering results survive a close/reopen — skip the
        // measurement picker and re-cluster entirely. A project-scope plot is reusable regardless
        // of the active image; a current-image plot only for the same image. Otherwise its cached
        // cells/fit are stale, so drop the old window and build fresh below.
        if (openScatterPlot != null) {
            if (openScatterPlot.isReusableFor(currentImageName)) {
                openScatterPlot.show();
                return;
            }
            openScatterPlot.close();
            openScatterPlot = null;
        }

        List<PathObject> cells = imageData.getHierarchy().getObjects(null, PathObject.class).stream()
                .filter(PathObjectFilter.DETECTIONS_ALL)
                .toList();
        if (cells.isEmpty()) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No detections found. Run cell detection first.");
            return;
        }

        List<String> allFeatures = CellFeatureExtractor.discoverFeatureNames(cells);
        if (allFeatures.isEmpty()) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No cell measurements found.");
            return;
        }

        // Default to the whole-cell marker means; let the user adjust.
        List<String> markerDefaults = IntensityHeatmap.discoverMarkerFeatures(allFeatures);
        var selectionPane = new FeatureSelectionPane(
                qupath.getStage(), allFeatures, markerDefaults.isEmpty() ? null : markerDefaults);
        selectionPane.setTitle("Select Measurements for Scatter Plot");
        List<String> markerFeatures = selectionPane.showAndWait();
        if (markerFeatures == null) {
            return; // user cancelled
        }
        if (markerFeatures.isEmpty()) {
            Dialogs.showWarningNotification(EXTENSION_NAME, "No measurements selected — nothing to plot.");
            return;
        }

        // "New…" in the plot forces a fresh one: drop the retained window and re-enter this method,
        // which (with nothing retained) re-runs the measurement picker and builds a new plot.
        Runnable onNewPlot = () -> {
            if (openScatterPlot != null) {
                openScatterPlot.close();
                openScatterPlot = null;
            }
            showScatterPlot(qupath, liveCurrentPredictions, normalizer, openClassControl);
        };

        openScatterPlot = new ScatterPlotView(
                qupath.getStage(),
                qupath,
                currentImageName,
                markerFeatures,
                cells,
                liveCurrentPredictions,
                openClassControl,
                normalizer,
                onNewPlot);
        openScatterPlot.show();
    }

    /** Resolve the open image's display name (project entry name, else server metadata name). */
    private static String resolveImageName(QuPathGUI qupath, qupath.lib.images.ImageData<BufferedImage> imageData) {
        var project = qupath.getProject();
        if (project != null) {
            var entry = project.getEntry(imageData);
            if (entry != null
                    && entry.getImageName() != null
                    && !entry.getImageName().isBlank()) {
                return entry.getImageName();
            }
        }
        return imageData.getServer().getMetadata().getName();
    }

    /**
     * Open the cellular-neighborhood (CN) spatial-clustering dialog for the
     * current image: cluster each cell by the cell-type composition of its local
     * spatial window into recurring micro-environments, written back as a
     * non-destructive {@code "CN"} measurement. Validates that an image is open,
     * has cells, and carries at least two distinct non-ignored classes (the CN
     * composition is meaningless with a single type).
     */
    static void showCellularNeighborhoods(QuPathGUI qupath) {
        var imageData = qupath.getImageData();
        if (imageData == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No image is open.");
            return;
        }
        var hierarchy = imageData.getHierarchy();
        var cells = hierarchy.getCellObjects().isEmpty() ? hierarchy.getDetectionObjects() : hierarchy.getCellObjects();
        if (cells.isEmpty()) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No detections found. Run cell detection first.");
            return;
        }
        java.util.Set<PathClass> distinct = new java.util.LinkedHashSet<>();
        for (PathObject cell : cells) {
            PathClass pc = cell.getPathClass();
            if (pc != null && pc.isValid() && !PathClassTools.isIgnoredClass(pc)) {
                distinct.add(pc);
            }
        }
        if (distinct.size() < 2) {
            Dialogs.showErrorMessage(
                    EXTENSION_NAME,
                    "Cellular neighborhoods need at least two distinct cell classes. " + "Classify cells first (found "
                            + distinct.size() + ").");
            return;
        }
        new NeighborhoodAnalysisDialog(qupath).show();
    }

    static void showBatchNormalization(QuPathGUI qupath) {
        var project = qupath.getProject();
        if (project == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "Open a project — batch normalisation aligns across images.");
            return;
        }
        if (project.getImageList().size() < 2) {
            Dialogs.showErrorMessage(
                    EXTENSION_NAME,
                    "Batch normalisation needs at least two images in the project (found "
                            + project.getImageList().size() + ").");
            return;
        }
        new qupath.ext.celltune.ui.BatchNormalizationDialog(qupath).show();
    }

    /**
     * Open the cells-free <b>image pixel prescreen</b>: whole-image per-channel pixel
     * statistics read off a low-res pyramid level for every project image, contextualised
     * against the cohort. Reads run in parallel off the FX thread behind a progress dialog.
     */
    static void showImagePixelPrescreen(QuPathGUI qupath) {
        var project = qupath.getProject();
        if (project == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, resources.getString("classify.no_project"));
            return;
        }

        @SuppressWarnings("unchecked")
        var entries = (List<ProjectImageEntry<BufferedImage>>) (List<?>) project.getImageList();
        if (entries.isEmpty()) {
            Dialogs.showInfoNotification(EXTENSION_NAME, "No project images found.");
            return;
        }

        var stage = new javafx.stage.Stage();
        stage.setTitle(EXTENSION_NAME + " — Computing pixel prescreen");
        stage.initOwner(qupath.getStage());
        stage.initModality(javafx.stage.Modality.WINDOW_MODAL);
        stage.setResizable(false);
        var bar = new javafx.scene.control.ProgressBar(0);
        bar.setPrefWidth(360);
        var status = new javafx.scene.control.Label("Scanning project images…");
        status.setMaxWidth(360);
        status.setWrapText(true);
        var box = new javafx.scene.layout.VBox(8, status, bar);
        box.setPadding(new javafx.geometry.Insets(15));
        stage.setScene(new javafx.scene.Scene(box));
        stage.setOnCloseRequest(e -> e.consume());
        stage.show();

        Thread worker = new Thread(
                () -> {
                    int total = entries.size();
                    AtomicInteger done = new AtomicInteger();
                    AtomicInteger failed = new AtomicInteger();

                    // One task per image; reads are independent and I/O+decode bound,
                    // so a small fixed pool gives a near-linear speedup. Capped at 4 to
                    // bound peak memory — each task holds one decoded downsampled region.
                    int nThreads = Math.min(Math.max(total, 1), 4);
                    ExecutorService pool = Executors.newFixedThreadPool(nThreads);
                    List<Callable<ImagePixelStats.ImageStats>> tasks = new ArrayList<>(total);
                    for (var entry : entries) {
                        tasks.add(() -> {
                            ImagePixelStats.ImageStats result = null;
                            if (entry != null) {
                                String imageName = entry.getImageName();
                                try {
                                    var data = entry.readImageData();
                                    try (var server = data.getServer()) {
                                        result = ImagePixelStatsReader.read(imageName, server);
                                    }
                                } catch (Exception ex) {
                                    failed.incrementAndGet();
                                    logger.warn(
                                            "[CellTune] Pixel prescreen failed to read '{}': {}",
                                            imageName,
                                            ex.getMessage());
                                }
                            }
                            final int c = done.incrementAndGet();
                            Platform.runLater(() -> {
                                bar.setProgress((double) c / total);
                                status.setText(String.format("Reading images: %d / %d…", c, total));
                            });
                            return result;
                        });
                    }

                    // Collect in submission order (deterministic cohort input).
                    var stats = new ArrayList<ImagePixelStats.ImageStats>(total);
                    try {
                        for (var future : pool.invokeAll(tasks)) {
                            var s = future.get();
                            if (s != null) {
                                stats.add(s);
                            }
                        }
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    } catch (java.util.concurrent.ExecutionException ee) {
                        logger.warn("[CellTune] Pixel prescreen task failed: {}", ee.getMessage());
                    } finally {
                        pool.shutdown();
                    }

                    var report = PixelCohortAnalyzer.analyze(stats);
                    final int failedF = failed.get();
                    Platform.runLater(() -> {
                        stage.close();
                        if (report.images().isEmpty()) {
                            Dialogs.showErrorMessage(
                                    EXTENSION_NAME, "Could not read pixels from any project image (see log).");
                            return;
                        }
                        if (failedF > 0) {
                            Dialogs.showWarningNotification(
                                    EXTENSION_NAME, failedF + " image(s) could not be read (see log).");
                        }
                        new PixelPrescreenView(qupath, qupath.getStage(), report).show();
                    });
                },
                "celltune-pixel-prescreen");
        worker.setDaemon(true);
        worker.start();
    }

    /**
     * Open the feature-importance view for the trained classifier. Importance is computed
     * off the FX thread against the current image's detections.
     *
     * @param classifier the session classifier (must be trained)
     */
    static void showFeatureImportance(QuPathGUI qupath, DualModelClassifier classifier) {
        var imageData = qupath.getImageData();
        if (imageData == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No image is open.");
            return;
        }
        // Snapshot the classifier now (on the FX thread) so the background
        // thread uses a stable reference even if retraining is triggered later.
        final DualModelClassifier snap = classifier;
        if (snap == null || !snap.isTrained()) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No trained classifier found. Run CellTune Classification first.");
            return;
        }
        var detections = imageData.getHierarchy().getDetectionObjects();
        if (detections.isEmpty()) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No detections found.");
            return;
        }
        List<String> featureNames = snap.getFeatureNames();
        if (featureNames == null || featureNames.isEmpty()) return;

        // Raw features — importance is reported in the same space the classifier trains on.
        CellFeatureExtractor extractor = new CellFeatureExtractor(featureNames);

        Thread worker = new Thread(
                () -> {
                    try {
                        var result = snap.computeFeatureImportance(detections, extractor);
                        javafx.application.Platform.runLater(
                                () -> new FeatureImportanceView(qupath.getStage(), result).show());
                    } catch (Throwable ex) {
                        logger.error("Feature importance failed", ex);
                        javafx.application.Platform.runLater(() -> Dialogs.showErrorMessage(
                                EXTENSION_NAME, "Feature importance failed: " + ex.getMessage()));
                    }
                },
                "CellTune-FeatureImportance");
        worker.setDaemon(true);
        worker.start();
    }
}
