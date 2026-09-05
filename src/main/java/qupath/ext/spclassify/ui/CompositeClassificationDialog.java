package qupath.ext.spclassify.ui;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.Modality;
import javafx.stage.Stage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.classifier.CompositeClassifier;
import qupath.ext.spclassify.io.BinaryClassifierRegistry;
import qupath.ext.spclassify.io.ProjectStateManager;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.projects.ProjectImageEntry;

/**
 * Simple composite classification dialog.
 * Lists all trained binary markers as checkboxes and lets the user choose
 * which project images to apply to. Handles the currently-open image
 * in-memory so results are visible immediately without reloading.
 */
public class CompositeClassificationDialog {

    private static final Logger logger = LoggerFactory.getLogger(CompositeClassificationDialog.class);

    private final QuPathGUI qupath;
    private final Stage stage;

    private final Map<String, CheckBox> markerCheckBoxes = new LinkedHashMap<>();
    private final Map<String, CheckBox> imageCheckBoxes = new LinkedHashMap<>();

    private final TextArea logArea = new TextArea();
    private final ProgressBar progressBar = new ProgressBar(0);
    private final Label statusLabel = new Label("Select markers and images, then click Apply.");
    private final CheckBox mergePrimaryCheck =
            new CheckBox("Prepend current primary classification (colour follows primary)");

    public CompositeClassificationDialog(QuPathGUI qupath) {
        this.qupath = qupath;
        this.stage = buildStage();
    }

    public void showAndWait() {
        stage.showAndWait();
    }

    // ── UI ─────────────────────────────────────────────────────────────────────

    private Stage buildStage() {
        var project = qupath.getProject();

        // Determine current image name for "Current only" button
        String currentImageName = null;
        var liveImageData = qupath.getImageData();
        if (liveImageData != null && project != null) {
            var entry = project.getEntry(liveImageData);
            if (entry != null) currentImageName = entry.getImageName();
        }
        final String currentImageNameFinal = currentImageName;

        // ── Marker checkboxes ──
        VBox markerBox = new VBox(6);
        markerBox.setPadding(new Insets(4));

        if (project != null) {
            Map<String, String> registry;
            try {
                registry = BinaryClassifierRegistry.load(project);
            } catch (IOException ex) {
                logger.warn("Could not load binary classifier registry: {}", ex.getMessage());
                registry = new LinkedHashMap<>();
            }

            for (String markerName : registry.keySet()) {
                ProjectStateManager.SavedState state = null;
                try {
                    state = ProjectStateManager.loadBinaryState(project, markerName);
                } catch (Exception ignored) {
                }
                if (state == null || state.xgboostModelBase64 == null) continue;

                CheckBox cb = new CheckBox(markerName);
                cb.setSelected(true);
                markerCheckBoxes.put(markerName, cb);
                markerBox.getChildren().add(cb);
            }
        }

        if (markerCheckBoxes.isEmpty()) {
            markerBox
                    .getChildren()
                    .add(new Label(
                            "No trained binary classifiers found.\nTrain at least one binary classifier first."));
        }

        ScrollPane markerScroll = new ScrollPane(markerBox);
        markerScroll.setFitToWidth(true);
        markerScroll.setPrefHeight(150);
        markerScroll.setStyle("-fx-border-color: #ccc;");

        Button selectAllMarkers = new Button("All");
        Button selectNoneMarkers = new Button("None");
        selectAllMarkers.setOnAction(e -> markerCheckBoxes.values().forEach(cb -> cb.setSelected(true)));
        selectNoneMarkers.setOnAction(e -> markerCheckBoxes.values().forEach(cb -> cb.setSelected(false)));
        HBox markerButtons = new HBox(6, new Label("Markers:"), selectAllMarkers, selectNoneMarkers);
        markerButtons.setAlignment(javafx.geometry.Pos.CENTER_LEFT);

        // ── Image checkboxes ──
        VBox imageBox = new VBox(6);
        imageBox.setPadding(new Insets(4));

        if (project != null) {
            @SuppressWarnings("unchecked")
            var entries = (List<ProjectImageEntry<BufferedImage>>) (List<?>) project.getImageList();
            for (var entry : entries) {
                String imgName = entry.getImageName();
                CheckBox cb = new CheckBox(imgName);
                cb.setSelected(true);
                imageCheckBoxes.put(imgName, cb);
                imageBox.getChildren().add(cb);
            }
        }

        if (imageCheckBoxes.isEmpty()) {
            imageBox.getChildren().add(new Label("No images in project."));
        }

        ScrollPane imageScroll = new ScrollPane(imageBox);
        imageScroll.setFitToWidth(true);
        imageScroll.setPrefHeight(150);
        imageScroll.setStyle("-fx-border-color: #ccc;");

        Button selectAllImages = new Button("All");
        Button selectNoneImages = new Button("None");
        Button selectCurrent = new Button("Current only");
        selectAllImages.setOnAction(e -> imageCheckBoxes.values().forEach(cb -> cb.setSelected(true)));
        selectNoneImages.setOnAction(e -> imageCheckBoxes.values().forEach(cb -> cb.setSelected(false)));
        selectCurrent.setOnAction(
                e -> imageCheckBoxes.forEach((name, cb) -> cb.setSelected(name.equals(currentImageNameFinal))));
        HBox imageButtons = new HBox(6, new Label("Images:"), selectAllImages, selectNoneImages, selectCurrent);
        imageButtons.setAlignment(javafx.geometry.Pos.CENTER_LEFT);

        // ── Log / progress ──
        logArea.setEditable(false);
        logArea.setPrefHeight(130);
        logArea.setStyle("-fx-font-family: monospace; -fx-font-size: 11px;");
        progressBar.setPrefWidth(Double.MAX_VALUE);

        mergePrimaryCheck.setWrapText(true);
        mergePrimaryCheck.setTooltip(new Tooltip("When ticked, each cell's existing primary class becomes the first "
                + "segment of the composite label (e.g. Tumor:CD3+:CD8-), "
                + "and the composite class is coloured to match the primary."));

        // ── Buttons ──
        Button applyBtn = new Button("Apply");
        Button closeBtn = new Button("Close");
        applyBtn.setDefaultButton(true);
        applyBtn.setPrefWidth(80);
        closeBtn.setPrefWidth(80);
        applyBtn.setOnAction(e -> runClassification());
        closeBtn.setOnAction(e -> stage.close());
        HBox buttons = new HBox(10, applyBtn, closeBtn);
        buttons.setPadding(new Insets(4, 0, 0, 0));

        // ── Root layout ──
        VBox root = new VBox(
                8,
                markerButtons,
                markerScroll,
                new Separator(),
                imageButtons,
                imageScroll,
                new Separator(),
                mergePrimaryCheck,
                statusLabel,
                progressBar,
                logArea,
                buttons);
        root.setPadding(new Insets(14));

        Stage s = new Stage();
        s.setTitle("Composite Classification");
        s.initOwner(qupath.getStage());
        s.initModality(Modality.NONE);
        s.setResizable(true);
        s.setScene(new javafx.scene.Scene(root, 500, 660));
        return s;
    }

    // ── Classification ─────────────────────────────────────────────────────────

    private void runClassification() {
        var project = qupath.getProject();
        if (project == null) {
            Dialogs.showErrorMessage("Composite Classification", "No project is open.");
            return;
        }

        // Collect selected markers
        List<String> selectedMarkers = new ArrayList<>();
        markerCheckBoxes.forEach((name, cb) -> {
            if (cb.isSelected()) selectedMarkers.add(name);
        });
        if (selectedMarkers.isEmpty()) {
            Dialogs.showErrorMessage("Composite Classification", "Select at least one marker.");
            return;
        }

        // Collect selected images
        List<String> selectedImages = new ArrayList<>();
        imageCheckBoxes.forEach((name, cb) -> {
            if (cb.isSelected()) selectedImages.add(name);
        });
        if (selectedImages.isEmpty()) {
            Dialogs.showErrorMessage("Composite Classification", "Select at least one image.");
            return;
        }

        // Snapshot live image data so the background thread can safely use it
        final var liveImageData = qupath.getImageData();
        final String currentImageName;
        if (liveImageData != null) {
            var entry = project.getEntry(liveImageData);
            currentImageName = (entry != null) ? entry.getImageName() : null;
        } else {
            currentImageName = null;
        }

        boolean applyToCurrentImage = currentImageName != null && selectedImages.contains(currentImageName);

        // Images to handle via disk batch (everything except the open one)
        List<String> batchImages = new ArrayList<>(selectedImages);
        if (currentImageName != null) batchImages.remove(currentImageName);

        logArea.clear();
        progressBar.setProgress(ProgressBar.INDETERMINATE_PROGRESS);
        statusLabel.setText("Running…");

        final boolean mergeWithPrimary = mergePrimaryCheck.isSelected();
        CompositeClassifier compositeClassifier = new CompositeClassifier();

        Thread worker = new Thread(
                () -> {
                    try {
                        // ── 1. Current open image — classify in-memory so viewer updates immediately ──
                        if (applyToCurrentImage && liveImageData != null) {
                            log("Classifying current image: " + currentImageName);
                            int count = compositeClassifier.apply(
                                    liveImageData, selectedMarkers, project, mergeWithPrimary, this::log);
                            log("Current image done: " + count + " cells classified.");

                            // Refresh the viewer on the FX thread
                            Platform.runLater(() -> liveImageData.getHierarchy().fireHierarchyChangedEvent(this));
                        }

                        // ── 2. Remaining images — read/write from disk ──
                        if (!batchImages.isEmpty()) {
                            Map<String, String> results = compositeClassifier.batch(
                                    project, batchImages, selectedMarkers, mergeWithPrimary, this::log);
                            results.forEach((img, msg) -> log(img + ": " + msg));
                        }

                        Platform.runLater(() -> {
                            progressBar.setProgress(1.0);
                            statusLabel.setText("Done — classification complete.");
                        });

                    } catch (Exception ex) {
                        logger.error("Composite classification failed", ex);
                        Platform.runLater(() -> {
                            progressBar.setProgress(0);
                            statusLabel.setText("Error: " + ex.getMessage());
                            logArea.appendText("ERROR: " + ex.getMessage() + "\n");
                        });
                    }
                },
                "SpClassify-CompositeClassification");
        worker.setDaemon(true);
        worker.start();
    }

    private void log(String msg) {
        logger.info("[CompositeClassification] {}", msg);
        Platform.runLater(() -> logArea.appendText(msg + "\n"));
    }
}
