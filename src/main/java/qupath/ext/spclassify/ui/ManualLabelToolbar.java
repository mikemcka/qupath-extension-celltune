package qupath.ext.spclassify.ui;

import java.util.*;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.Window;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.model.CellPrediction;
import qupath.ext.spclassify.model.LabelStore;
import qupath.ext.spclassify.model.PopulationSet;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.events.PathObjectSelectionListener;

/**
 * Manual Label Mode — a floating toolbar that lets users click on cells in the
 * QuPath viewer and assign class labels directly, outside of Review Mode.
 *
 * <p>When a detection cell is selected in the viewer, a status label shows its
 * ID and current class. Clicking a class button assigns that class to the cell
 * and records it in the extension's ground-truth {@link LabelStore}.
 *
 * <p>Class buttons are populated from QuPath project classes and the
 * optional CellTypeTable entries.
 *
 * <p>Layout:
 * <pre>
 *  Selected: Cell_abc (Bcell)     ● 42 labels
 *  [CD4] [CD8] [Bcell] [DC] [Macrophage] [Treg] | [All Classes ▼]
 *  [ ] Auto-advance to next detection
 * </pre>
 */
public class ManualLabelToolbar {

    private static final Logger logger = LoggerFactory.getLogger(ManualLabelToolbar.class);

    private final QuPathGUI qupath;
    private final LabelStore labelStore;
    private final Stage stage;

    // UI elements
    private final Label selectedCellLabel = new Label("Select a cell in the viewer");
    private final Label labelCountLabel = new Label("0 labels");
    private final Circle statusDot = new Circle(6, Color.WHITE);
    private final FlowPane classButtonBox = new FlowPane(4, 4);
    private final MenuButton allClassesMenu = new MenuButton("All Classes \u25BC");
    private final CheckBox autoAdvance = new CheckBox("Auto-advance to next detection");

    // Currently selected detections (may be more than one)
    private List<PathObject> selectedCells = new ArrayList<>();

    // O(1) lookup table for auto-advance: PathObject → index in the detection list.
    // Lazily built per image so dense images (50k+ cells) don't pay an O(N) indexOf
    // scan on every click. Invalidated whenever the active image changes.
    private List<PathObject> detectionCache = null;
    private java.util.IdentityHashMap<PathObject, Integer> detectionIndex = null;
    private qupath.lib.images.ImageData<?> detectionCacheImageData = null;

    // Magenta selection ring — painted as a viewer-level overlay so updating it
    // does NOT fire hierarchy events (no detection-overlay repaint storm).
    private final SelectionHighlightOverlay highlightOverlay = new SelectionHighlightOverlay();
    private qupath.lib.gui.viewer.QuPathViewer highlightViewer = null;

    // Optional: predictions for cells (may be null)
    private final PopulationSet predictions;

    // UI for model predictions
    private final HBox predictionBox = new HBox(4);

    // Listener reference for cleanup
    private PathObjectSelectionListener selectionListener;
    // Hierarchy the selection listener is currently attached to (may differ from
    // qupath.getImageData() after an image switch). Tracked so we can detach
    // cleanly even if the active image has changed.
    private qupath.lib.objects.hierarchy.PathObjectHierarchy listenedHierarchy;
    // Listener on QuPath's active-image property so we can re-attach the
    // selection listener to the new image and clear the stale highlight ring.
    private javafx.beans.value.ChangeListener<qupath.lib.images.ImageData<java.awt.image.BufferedImage>>
            imageDataListener;

    /**
     * When non-null, class buttons and the All Classes menu are restricted to
     * exactly these classes, preventing accidental cross-marker labels during
     * a binary classifier session.
     */
    private final List<String> binaryClasses;

    /**
     * Create and show the manual label toolbar.
     *
     * @param qupath         the QuPath GUI
     * @param labelStore     the ground-truth label store to write to
     * @param extraClasses   additional class names (e.g. from CellTypeTable), may be null
     * @param owner          owner window for positioning
     * @param predictions    optional current predictions, may be null
     * @param binaryClasses  allowed class names for binary mode, or null for multi-class mode
     */
    public ManualLabelToolbar(
            QuPathGUI qupath,
            LabelStore labelStore,
            Set<String> extraClasses,
            Window owner,
            PopulationSet predictions,
            List<String> binaryClasses) {
        this.qupath = qupath;
        this.labelStore = labelStore;
        this.predictions = predictions;
        this.binaryClasses = (binaryClasses != null && !binaryClasses.isEmpty()) ? List.copyOf(binaryClasses) : null;

        stage = new Stage();
        stage.setTitle("Manual Label Mode");
        stage.initOwner(owner);
        stage.initModality(Modality.NONE);
        stage.setAlwaysOnTop(true);
        stage.setResizable(true);

        // ── Status row ──────────────────────────────────────────────────────
        selectedCellLabel.setStyle("-fx-font-size: 12px;");
        statusDot.setStroke(Color.GRAY);
        labelCountLabel.setStyle("-fx-font-family: monospace; -fx-font-size: 11px;");
        updateLabelCount();

        HBox statusRow = new HBox(8, selectedCellLabel, statusDot, new Region(), labelCountLabel);
        statusRow.setAlignment(Pos.CENTER_LEFT);
        HBox.setHgrow(statusRow.getChildren().get(2), Priority.ALWAYS);

        // ── Prediction buttons (model 1/model 2) ───────────────────────────
        predictionBox.setAlignment(Pos.CENTER_LEFT);
        updatePredictionButtons(null); // initially no cell

        // ── Class buttons ───────────────────────────────────────────────────
        classButtonBox.setPrefWrapLength(420);
        populateClassButtons(extraClasses);

        allClassesMenu.setStyle("-fx-font-size: 11px;");
        populateAllClassesMenu(extraClasses);

        VBox buttonSection = new VBox(4, classButtonBox, allClassesMenu);
        buttonSection.setAlignment(Pos.CENTER_LEFT);

        // ── Auto-advance checkbox ───────────────────────────────────────────
        autoAdvance.setSelected(false);

        // ── Done button ─────────────────────────────────────────────────────
        // Closes the panel (triggers setOnHidden, which saves labels and syncs state).
        Button doneBtn = new Button("Done");
        doneBtn.setStyle("-fx-font-weight: bold;");
        doneBtn.setOnAction(e -> stage.close());
        Region doneSpacer = new Region();
        HBox.setHgrow(doneSpacer, Priority.ALWAYS);
        HBox doneRow = new HBox(8, autoAdvance, doneSpacer, doneBtn);
        doneRow.setAlignment(Pos.CENTER_LEFT);

        // ── Layout ──────────────────────────────────────────────────────────
        VBox root = new VBox(8, statusRow, predictionBox, buttonSection, doneRow);
        root.setPadding(new Insets(10));
        root.setMinWidth(440);

        ScrollPane scroll = new ScrollPane(root);
        scroll.setFitToWidth(true);
        scroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
        scroll.setVbarPolicy(ScrollPane.ScrollBarPolicy.AS_NEEDED);
        double screenH = javafx.stage.Screen.getPrimary().getVisualBounds().getHeight();
        scroll.setMaxHeight(screenH * 0.85);

        stage.setScene(new Scene(scroll));
        stage.sizeToScene();

        // ── Listen for cell selection changes ───────────────────────────────
        installSelectionListener();

        // ── Install lightweight selection-ring overlay on the active viewer ──
        highlightViewer = qupath.getViewer();
        highlightOverlay.installOn(highlightViewer);

        // ── React to image switches so the ring & selection listener follow ──
        // Without this, the overlay keeps drawing the previous cell's slide
        // coordinates onto the newly-opened image, producing a magenta ring at
        // an apparently random location.
        imageDataListener = (obs, oldData, newData) -> {
            javafx.application.Platform.runLater(() -> {
                // Drop the stale target so we don't draw old coords on the new image.
                highlightOverlay.clear();
                selectedCells = new ArrayList<>();
                selectedCellLabel.setText("Select a detection cell in the viewer");
                statusDot.setFill(Color.WHITE);
                updatePredictionButtons(null);

                // Move the overlay onto the (possibly new) active viewer.
                var viewerNow = qupath.getViewer();
                if (viewerNow != highlightViewer) {
                    if (highlightViewer != null) highlightOverlay.uninstallFrom(highlightViewer);
                    highlightViewer = viewerNow;
                    if (highlightViewer != null) highlightOverlay.installOn(highlightViewer);
                }
                if (highlightViewer != null) highlightViewer.repaint();

                // Re-attach the selection listener to the new image's hierarchy.
                removeSelectionListener();
                installSelectionListener();
            });
        };
        qupath.imageDataProperty().addListener(imageDataListener);

        // ── Cleanup on close ─────────────────────────────────────────────
        stage.setOnHidden(e -> {
            if (imageDataListener != null) {
                qupath.imageDataProperty().removeListener(imageDataListener);
                imageDataListener = null;
            }
            removeSelectionListener();
            highlightOverlay.uninstallFrom(highlightViewer);
            if (highlightViewer != null) highlightViewer.repaint();
            highlightViewer = null;
        });

        stage.show();
    }

    /** @return the stage for external control */
    public Stage getStage() {
        return stage;
    }

    // ── Selection listener ──────────────────────────────────────────────────

    private void installSelectionListener() {
        var imageData = qupath.getImageData();
        if (imageData == null) return;
        listenedHierarchy = imageData.getHierarchy();

        selectionListener = (pathObjectSelected, previousObject, allSelected) -> {
            javafx.application.Platform.runLater(() -> {
                // Collect all selected detections
                selectedCells = new ArrayList<>();
                if (allSelected != null) {
                    for (PathObject obj : allSelected) {
                        if (obj.isDetection()) selectedCells.add(obj);
                    }
                }

                if (selectedCells.size() == 1) {
                    PathObject cell = selectedCells.get(0);
                    String id = shortId(cell);
                    String cls =
                            cell.getPathClass() != null ? cell.getPathClass().getName() : "unlabelled";
                    selectedCellLabel.setText("Selected: " + id + " (" + cls + ")");
                    statusDot.setFill(cell.getPathClass() != null ? Color.LIMEGREEN : Color.WHITE);
                    updatePredictionButtons(cell);
                    updateHighlightRing(cell);
                } else if (selectedCells.size() > 1) {
                    selectedCellLabel.setText(selectedCells.size() + " cells selected");
                    statusDot.setFill(Color.LIMEGREEN);
                    updatePredictionButtons(null);
                    updateHighlightRing(null);
                } else {
                    selectedCellLabel.setText("Select a detection cell in the viewer");
                    statusDot.setFill(Color.WHITE);
                    updatePredictionButtons(null);
                    updateHighlightRing(null);
                }
            });
        };

        listenedHierarchy.getSelectionModel().addPathObjectSelectionListener(selectionListener);
    }

    // ── Model prediction buttons (model 1/model 2) ─────────────────────────
    private void updatePredictionButtons(PathObject cell) {
        predictionBox.getChildren().clear();
        if (cell == null) return;

        if (predictions == null) {
            Label unavailable = new Label("Confidence unavailable: train/classify first");
            unavailable.setStyle("-fx-font-size: 11px; -fx-text-fill: #666666;");
            predictionBox.getChildren().add(unavailable);
            return;
        }

        String cellId = cell.getID().toString();
        CellPrediction pred = predictions.get(cellId);
        if (pred == null) {
            Label unavailable = new Label("Confidence unavailable for selected cell");
            unavailable.setStyle("-fx-font-size: 11px; -fx-text-fill: #666666;");
            predictionBox.getChildren().add(unavailable);
            return;
        }

        // Model 1 button
        Button mdl1Btn =
                new Button(String.format("Model 1: %s (%.0f%%)", pred.getModel1Label(), pred.model1Confidence() * 100));
        mdl1Btn.setStyle("-fx-background-color: #bbdefb; -fx-font-weight: bold; -fx-font-size: 11px;");
        mdl1Btn.setOnAction(e -> assignLabel(pred.getModel1Label()));

        // Model 2 button
        Button mdl2Btn =
                new Button(String.format("Model 2: %s (%.0f%%)", pred.getModel2Label(), pred.model2Confidence() * 100));
        mdl2Btn.setStyle("-fx-background-color: #f8bbd0; -fx-font-weight: bold; -fx-font-size: 11px;");
        mdl2Btn.setOnAction(e -> assignLabel(pred.getModel2Label()));

        predictionBox.getChildren().addAll(mdl1Btn, mdl2Btn);
    }

    private void removeSelectionListener() {
        if (selectionListener == null) return;
        // Detach from the hierarchy the listener was actually attached to, not
        // the currently-active image's hierarchy (which may have changed).
        if (listenedHierarchy != null) {
            listenedHierarchy.getSelectionModel().removePathObjectSelectionListener(selectionListener);
        }
        selectionListener = null;
        listenedHierarchy = null;
    }

    /**
     * Show or clear the magenta selection ring drawn by
     * {@link SelectionHighlightOverlay}. Repaints only the overlay layer
     * — no hierarchy event, no detection-overlay rebuild.
     */
    private void updateHighlightRing(PathObject cell) {
        if (highlightViewer == null) return;
        highlightOverlay.setTargetRoi(cell != null ? cell.getROI() : null);
        highlightViewer.repaint();
    }

    // NOTE: A previous version added a magenta ellipse annotation around the
    // selected cell as an extra visual hint. That added two hierarchy
    // structure-changed events per click (add + remove of the annotation),
    // each of which forced QuPath to repaint every detection in the image
    // — the dominant cost on images with tens of thousands of cells.
    // QuPath's own selection rendering already outlines the selected cell,
    // so the extra annotation was redundant and has been removed.

    // ── Class buttons ───────────────────────────────────────────────────────

    private void populateClassButtons(Set<String> extraClasses) {
        classButtonBox.getChildren().clear();
        Set<String> allNames = new LinkedHashSet<>();

        if (binaryClasses != null) {
            // Binary mode: only the resolved allowed classes
            allNames.addAll(binaryClasses);
        } else {
            // Multi-class mode: QuPath project classes, then extras
            var project = qupath.getProject();
            if (project != null) {
                for (var pc : project.getPathClasses()) {
                    if (pc != null && pc.getName() != null && !pc.getName().isEmpty()) {
                        allNames.add(pc.getName());
                    }
                }
            }
            if (extraClasses != null) {
                allNames.addAll(extraClasses);
            }
        }

        // Show up to 12 buttons inline; rest go into All Classes menu
        int count = 0;
        for (String name : allNames) {
            if (count >= 12) break;
            Button btn = new Button(name);
            btn.setStyle("-fx-font-weight: bold; -fx-font-size: 11px;");
            btn.setOnAction(e -> assignLabel(name));
            classButtonBox.getChildren().add(btn);
            count++;
        }
    }

    private void populateAllClassesMenu(Set<String> extraClasses) {
        allClassesMenu.getItems().clear();
        Set<String> allNames = new LinkedHashSet<>();

        if (binaryClasses != null) {
            // Binary mode: only the resolved allowed classes
            allNames.addAll(binaryClasses);
        } else {
            var project = qupath.getProject();
            if (project != null) {
                for (var pc : project.getPathClasses()) {
                    if (pc != null && pc.getName() != null && !pc.getName().isEmpty()) {
                        allNames.add(pc.getName());
                    }
                }
            }
            if (extraClasses != null) {
                allNames.addAll(extraClasses);
            }
        }

        for (String name : allNames) {
            MenuItem item = new MenuItem(name);
            item.setOnAction(e -> assignLabel(name));
            allClassesMenu.getItems().add(item);
        }

        if (allClassesMenu.getItems().isEmpty()) {
            allClassesMenu.setDisable(true);
        } else {
            allClassesMenu.setDisable(false);
        }
    }

    // ── Label assignment ────────────────────────────────────────────────────

    private void assignLabel(String className) {
        if (selectedCells.isEmpty()) {
            return;
        }

        PathClass pc = PathClass.fromString(className);
        for (PathObject cell : selectedCells) {
            cell.setPathClass(pc);
            labelStore.setLabel(cell.getID().toString(), className);
        }

        logger.info("Manual label: {} cell(s) → {}", selectedCells.size(), className);

        // Update status
        if (selectedCells.size() == 1) {
            selectedCellLabel.setText("Selected: " + shortId(selectedCells.get(0)) + " (" + className + ")");
        } else {
            selectedCellLabel.setText(selectedCells.size() + " cells labelled → " + className);
        }
        statusDot.setFill(Color.LIMEGREEN);
        updateLabelCount();

        // Fire hierarchy update so overlay colours refresh
        var imageData = qupath.getImageData();
        if (imageData != null) {
            imageData.getHierarchy().fireObjectClassificationsChangedEvent(this, new ArrayList<>(selectedCells));
        }

        // Auto-advance: only meaningful for single-cell labelling
        if (autoAdvance.isSelected() && selectedCells.size() == 1) {
            advanceToNextDetection();
        }
    }

    private void advanceToNextDetection() {
        var imageData = qupath.getImageData();
        if (imageData == null || selectedCells.isEmpty()) return;
        PathObject currentCell = selectedCells.get(0);

        ensureDetectionCache(imageData);
        if (detectionCache == null) return;

        Integer idxBoxed = detectionIndex.get(currentCell);
        if (idxBoxed == null) return;
        int idx = idxBoxed;
        if (idx + 1 < detectionCache.size()) {
            PathObject next = detectionCache.get(idx + 1);
            imageData.getHierarchy().getSelectionModel().setSelectedObject(next);

            // Centre viewer on the next cell
            var viewer = qupath.getViewer();
            if (viewer != null && next.getROI() != null) {
                viewer.setCenterPixelLocation(
                        next.getROI().getCentroidX(), next.getROI().getCentroidY());
            }
        }
    }

    /**
     * Lazily build the (PathObject → index) lookup for the current image's
     * detections. Cheap to rebuild only when the active image changes. The
     * cache assumes the detection list is stable during a labelling session
     * — if cells are added/removed externally, auto-advance may skip; users
     * can recover by re-clicking a cell in the viewer.
     */
    private void ensureDetectionCache(qupath.lib.images.ImageData<?> imageData) {
        if (detectionCacheImageData == imageData && detectionCache != null) {
            return;
        }
        var dets = imageData.getHierarchy().getDetectionObjects();
        detectionCache = new ArrayList<>(dets);
        detectionIndex = new java.util.IdentityHashMap<>(detectionCache.size() * 2);
        for (int i = 0; i < detectionCache.size(); i++) {
            detectionIndex.put(detectionCache.get(i), i);
        }
        detectionCacheImageData = imageData;
    }

    private void updateLabelCount() {
        labelCountLabel.setText(labelStore.size() + " labels");
    }

    private static String shortId(PathObject cell) {
        String id = cell.getID().toString();
        return id.length() > 8 ? id.substring(0, 8) + "…" : id;
    }
}
