package qupath.ext.spclassify;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.ResourceBundle;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.io.ProjectStateManager;
import qupath.ext.spclassify.model.CohortAnomalyAnalyzer;
import qupath.ext.spclassify.model.CohortAnomalyReport;
import qupath.ext.spclassify.model.PopulationSet;
import qupath.ext.spclassify.ui.ProjectPredictionSummaryView;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.projects.ProjectImageEntry;

/**
 * The <b>Project Prediction Summary</b> dialog: scans every project image for saved
 * predictions, computes per-image agreement/disagreement counts and cohort-level anomaly
 * flags, and shows them in a {@link ProjectPredictionSummaryView}.
 * <p>
 * Loading runs in parallel off the FX thread behind a small progress dialog so large
 * projects stay responsive. Lifted verbatim out of {@code SpClassifyExtension} (mirroring the
 * {@code UtilityScripts}/{@code AnnotationRegionExporter} moves); the menu handler delegates
 * here. The only state it needs is the current image's live in-memory predictions, passed in
 * as a parameter — the caller persists them first so they are reflected in the summary.
 */
final class ProjectPredictionSummary {

    private static final ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.spclassify.ui.strings");
    private static final String EXTENSION_NAME = resources.getString("name");
    private static final Logger logger = LoggerFactory.getLogger(ProjectPredictionSummary.class);

    private ProjectPredictionSummary() {} // utility class

    /**
     * Build and show the project prediction summary.
     *
     * @param qupath                 the QuPath GUI
     * @param liveCurrentPredictions the open image's in-memory predictions (may be null); used
     *                               in place of the saved file for the current image
     */
    static void show(QuPathGUI qupath, PopulationSet liveCurrentPredictions) {
        var project = qupath.getProject();
        if (project == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, resources.getString("classify.no_project"));
            return;
        }

        String currentImageName = null;
        var currentImageData = qupath.getImageData();
        if (currentImageData != null) {
            var currentEntry = project.getEntry(currentImageData);
            if (currentEntry != null) {
                currentImageName = currentEntry.getImageName();
            }
        }
        final String currentImageNameFinal = currentImageName;
        final PopulationSet liveCurrent = liveCurrentPredictions;

        @SuppressWarnings("unchecked")
        var entries = (List<ProjectImageEntry<BufferedImage>>) (List<?>) project.getImageList();
        var imageNames = new ArrayList<String>(entries.size());
        for (var entry : entries) {
            if (entry == null || entry.getImageName() == null) continue;
            imageNames.add(entry.getImageName());
        }

        if (imageNames.isEmpty()) {
            Dialogs.showInfoNotification(EXTENSION_NAME, "No project images found.");
            return;
        }

        // ── Show a small progress dialog and load all images in parallel
        // off the FX thread so the UI stays responsive on large projects.
        var stage = new javafx.stage.Stage();
        stage.setTitle(EXTENSION_NAME + " — Loading prediction summary");
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

        Thread t = new Thread(
                () -> {
                    int total = imageNames.size();
                    java.util.concurrent.atomic.AtomicInteger done = new java.util.concurrent.atomic.AtomicInteger();
                    // Capture avgCounts per image alongside the row (rows themselves are immutable).
                    java.util.Map<String, Map<String, Long>> avgCountsByImage =
                            new java.util.concurrent.ConcurrentHashMap<>();

                    // Parallel JSON load — independent files per image, IO+CPU bound.
                    List<SummaryInputRow> sourceRows = imageNames.parallelStream()
                            .map(imageName -> {
                                PopulationSet predictions = null;
                                if (currentImageNameFinal != null
                                        && currentImageNameFinal.equals(imageName)
                                        && liveCurrent != null
                                        && liveCurrent.size() > 0) {
                                    predictions = liveCurrent;
                                } else {
                                    try {
                                        predictions = ProjectStateManager.loadImagePredictions(project, imageName);
                                    } catch (IOException ex) {
                                        logger.warn(
                                                "Failed to load predictions for {} in project summary: {}",
                                                imageName,
                                                ex.getMessage());
                                    }
                                }
                                SummaryInputRow row;
                                if (predictions == null || predictions.size() == 0) {
                                    row = new SummaryInputRow(
                                            imageName, 0L, 0L, "No saved predictions for this image.");
                                } else {
                                    long predicted = predictions.size();
                                    long disagreements = predictions.getDisagreementCount();
                                    Map<String, Long> avgCounts = predictions.getAvgCounts();
                                    avgCountsByImage.put(imageName, avgCounts);
                                    row = new SummaryInputRow(
                                            imageName, predicted, disagreements, formatClassCounts(avgCounts));
                                }
                                int c = done.incrementAndGet();
                                javafx.application.Platform.runLater(() -> {
                                    bar.setProgress((double) c / total);
                                    status.setText(String.format("Loading predictions: %d / %d images…", c, total));
                                });
                                return row;
                            })
                            .collect(java.util.stream.Collectors.toList());

                    var analyzerInputs = new ArrayList<CohortAnomalyReport.ImageInput>(sourceRows.size());
                    for (var src : sourceRows) {
                        Map<String, Long> avg = avgCountsByImage.get(src.imageName());
                        if (src.predictedCells() > 0 && avg != null) {
                            analyzerInputs.add(new CohortAnomalyReport.ImageInput(
                                    src.imageName(), src.predictedCells(), src.disagreements(), avg));
                        }
                    }
                    var anomalyReport = CohortAnomalyAnalyzer.analyze(analyzerInputs);
                    var anomalyByImage = anomalyReport.byImageName();

                    var rows = new ArrayList<ProjectPredictionSummaryView.Row>(sourceRows.size());
                    for (var source : sourceRows) {
                        rows.add(buildPredictionSummaryRow(source, anomalyByImage.get(source.imageName())));
                    }

                    javafx.application.Platform.runLater(() -> {
                        stage.close();
                        new ProjectPredictionSummaryView(qupath, qupath.getStage(), rows).show();
                    });
                },
                "sp-classify-summary-load");
        t.setDaemon(true);
        t.start();
    }

    private static ProjectPredictionSummaryView.Row buildPredictionSummaryRow(
            SummaryInputRow source, CohortAnomalyReport.ImageAnomaly anomaly) {
        if (source.predictedCells() == 0L) {
            return new ProjectPredictionSummaryView.Row(
                    source.imageName(),
                    0L,
                    0L,
                    0L,
                    "-",
                    0.0,
                    false,
                    "-",
                    "No highlighted rare classes.",
                    List.of(),
                    source.classCountsText());
        }

        long agreements = Math.max(0L, source.predictedCells() - source.disagreements());
        double agreementPct = source.predictedCells() > 0 ? (100.0 * agreements) / source.predictedCells() : 0.0;

        if (anomaly == null) {
            return new ProjectPredictionSummaryView.Row(
                    source.imageName(),
                    source.predictedCells(),
                    agreements,
                    source.disagreements(),
                    String.format("%.1f%%", agreementPct),
                    0.0,
                    false,
                    "-",
                    "No highlighted rare classes.",
                    List.of(),
                    source.classCountsText());
        }

        return new ProjectPredictionSummaryView.Row(
                source.imageName(),
                source.predictedCells(),
                agreements,
                source.disagreements(),
                String.format("%.1f%%", agreementPct),
                anomaly.anomalyScore(),
                anomaly.flagged(),
                formatFlagReasons(anomaly.flagReasons()),
                formatHighlightedRareClasses(anomaly.highlightedClasses(), anomaly.enrichmentByClass()),
                anomaly.highlightedClasses(),
                source.classCountsText());
    }

    private static String formatClassCounts(Map<String, Long> counts) {
        if (counts == null || counts.isEmpty()) {
            return "No predicted classes.";
        }

        var parts = new ArrayList<String>(counts.size());
        for (var entry : counts.entrySet()) {
            String className = entry.getKey() == null ? "(unknown)" : entry.getKey();
            long count = entry.getValue() == null ? 0L : entry.getValue();
            parts.add(className + ": " + count);
        }
        return String.join(", ", parts);
    }

    private static String formatFlagReasons(List<String> reasons) {
        if (reasons == null || reasons.isEmpty()) {
            return "-";
        }
        return String.join(", ", reasons);
    }

    private static String formatHighlightedRareClasses(
            List<String> highlightedClasses, Map<String, CohortAnomalyReport.ClassEnrichment> enrichmentByClass) {
        if (highlightedClasses == null || highlightedClasses.isEmpty()) {
            return "No highlighted rare classes.";
        }

        var parts = new ArrayList<String>(highlightedClasses.size());
        for (String className : highlightedClasses) {
            var enrichment = enrichmentByClass == null ? null : enrichmentByClass.get(className);
            if (enrichment == null) {
                parts.add(className);
                continue;
            }
            parts.add(String.format(
                    "%s (count=%d, fold=%.2fx)", className, enrichment.count(), enrichment.enrichmentFold()));
        }
        return String.join("; ", parts);
    }

    private record SummaryInputRow(String imageName, long predictedCells, long disagreements, String classCountsText) {}
}
