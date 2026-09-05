package qupath.ext.spclassify.ui;

import java.util.*;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import qupath.ext.spclassify.model.CellPrediction;
import qupath.ext.spclassify.model.CellTypeTable;

/**
 * A toolbar for the Review Mode: navigation buttons, dynamic per-cell
 * prediction buttons, an "All Classes" dropdown, and a status/index indicator.
 *
 * <p>For <b>disagreement</b> cells the toolbar shows the top prediction from
 * each model (XGBoost &amp; LightGBM) as colour-coded buttons so the user can
 * quickly accept one.  For <b>agreement</b> cells a single "Both" button is
 * shown.  An "All Classes ▼" menu provides every class defined in the QuPath
 * project as a fallback.
 *
 * <p>Layout:
 * <pre>
 *  [Previous] [Next] [Skip]  | [XGB: CD4 (87%)] [LGB: Bcell (65%)] | [All Classes ▼] |  (3/50) ●
 * </pre>
 */
public class ReviewToolbar extends HBox {

    private static final ResourceBundle STRINGS = ResourceBundle.getBundle("qupath.ext.spclassify.ui.strings");

    private final ReviewController controller;
    private final ChannelSelector channelSelector;
    private final CellTypeTable cellTypeTable;

    /** Dynamically-populated prediction buttons for the current cell. */
    private final HBox predictionBox = new HBox(4);
    /** Dropdown listing every PathClass in the QuPath project. */
    private final MenuButton allClassesMenu = new MenuButton("All Classes \u25BC");
    /**
     * When non-null, the toolbar is in binary mode and only these class names
     * are offered as labelling choices (populated from saved state or labels).
     */
    private List<String> binaryClasses = null;

    // Status widgets
    private final Label indexLabel = new Label();
    private final Circle statusDot = new Circle(6);
    /** Shows the project image the current cell was sampled from — lets the
     *  user visually verify that review is pulling from multiple images. */
    private final Label imageNameLabel = new Label();
    /** Shows the names of annotations in the current image whose ROI contains
     *  the current cell's centroid. Empty when the cell isn't inside any
     *  named annotation. */
    private final Label annotationLabel = new Label();

    public ReviewToolbar(ReviewController controller, CellTypeTable cellTypeTable, ChannelSelector channelSelector) {
        super(8);
        this.controller = controller;
        this.cellTypeTable = cellTypeTable;
        this.channelSelector = channelSelector;
        setPadding(new Insets(6, 10, 6, 10));
        setAlignment(Pos.CENTER_LEFT);

        // ── Navigation buttons ──────────────────────────────────────────
        Button prevBtn = new Button(STRINGS.getString("review.previous"));
        prevBtn.setOnAction(e -> {
            controller.previous();
            refreshStatus();
            channelSelector.applyForCurrentCell(controller);
        });

        Button nextBtn = new Button(STRINGS.getString("review.next"));
        nextBtn.setOnAction(e -> {
            boolean more = controller.next();
            refreshStatus();
            channelSelector.applyForCurrentCell(controller);
            if (!more) showCompleteAlert();
        });

        Button skipBtn = new Button(STRINGS.getString("review.skip"));
        skipBtn.setOnAction(e -> {
            boolean more = controller.skip();
            refreshStatus();
            channelSelector.applyForCurrentCell(controller);
            if (!more) showCompleteAlert();
        });

        // ── Separator ──
        Separator sep1 = new Separator(javafx.geometry.Orientation.VERTICAL);

        // ── Prediction buttons — populated per-cell in refreshStatus() ──
        predictionBox.setAlignment(Pos.CENTER_LEFT);

        // ── All Classes menu — fallback for any QuPath class ──
        allClassesMenu.setStyle("-fx-font-size: 11px;");
        populateAllClassesMenu();

        // ── Separator ──
        Separator sep2 = new Separator(javafx.geometry.Orientation.VERTICAL);

        // ── Index indicator ─────────────────────────────────────────────
        HBox statusBox = new HBox(4);
        statusBox.setAlignment(Pos.CENTER);
        statusDot.setFill(Color.WHITE);
        statusDot.setStroke(Color.GRAY);
        indexLabel.setStyle("-fx-font-family: monospace;");
        imageNameLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #555;");
        imageNameLabel.setMaxWidth(260);
        imageNameLabel.setMinWidth(0);
        imageNameLabel.setEllipsisString("\u2026");
        annotationLabel.setStyle("-fx-font-size: 11px; -fx-text-fill: #0d47a1; -fx-font-weight: bold;");
        annotationLabel.setMaxWidth(Double.MAX_VALUE);
        annotationLabel.setMinWidth(Region.USE_PREF_SIZE);
        annotationLabel.setEllipsisString("\u2026");
        statusBox.getChildren().addAll(annotationLabel, imageNameLabel, indexLabel, statusDot);

        // Right-click on index → jump-to-index dialog
        indexLabel.setOnMouseClicked(event -> {
            if (event.getButton() == javafx.scene.input.MouseButton.SECONDARY) {
                showJumpDialog();
            }
        });

        // ── Spacer to push status to the right ─────────────────────────
        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        // ── Done button — closes the review stage (triggers setOnHidden, which
        //     merges reviewed labels back into the main LabelStore). ──
        Button doneBtn = new Button("Done");
        doneBtn.setStyle("-fx-font-weight: bold;");
        doneBtn.setOnAction(e -> {
            var scene = getScene();
            if (scene != null && scene.getWindow() != null) {
                scene.getWindow().hide();
            }
        });

        Separator sep3 = new Separator(javafx.geometry.Orientation.VERTICAL);

        getChildren()
                .addAll(
                        prevBtn,
                        nextBtn,
                        skipBtn,
                        sep1,
                        predictionBox,
                        allClassesMenu,
                        sep2,
                        spacer,
                        statusBox,
                        sep3,
                        doneBtn);

        // ── Auto-advance on first show ──────────────────────────────────
        controller.next();
        channelSelector.applyForCurrentCell(controller);
        // Refresh status whenever the user clicks a non-queue cell in the tile
        // viewer so the manual-selection indicator updates immediately.
        controller.setSelectionChangedCallback(this::refreshStatus);
        refreshStatus();
    }

    // ── Status refresh ──────────────────────────────────────────────────

    /** Update prediction buttons, index label, and status dot colour. */
    public void refreshStatus() {
        int idx = controller.getCurrentIndex();
        int total = controller.size();

        if (controller.isFinished()) {
            indexLabel.setText(String.format("(%d/%d)", total, total));
            statusDot.setFill(Color.WHITE);
            predictionBox.getChildren().clear();
            imageNameLabel.setText("");
            annotationLabel.setText("");
            annotationLabel.setTooltip(null);
            return;
        }

        indexLabel.setText(String.format("(%d/%d)", idx + 1, total));

        // Indicate when the user has clicked a non-queue cell that buttons
        // will now label instead of the queue cell.
        if (controller.isManualSelection()) {
            indexLabel.setText(indexLabel.getText() + "  \u2192 clicked cell");
            indexLabel.setStyle("-fx-font-family: monospace; -fx-text-fill: #d84315; -fx-font-weight: bold;");
        } else {
            indexLabel.setStyle("-fx-font-family: monospace;");
        }

        // Show which source image the current cell was sampled from.
        String src = controller.getCurrentCellImageName();
        imageNameLabel.setText(src == null || src.isBlank() ? "" : src);
        imageNameLabel.setTooltip((src == null || src.isBlank()) ? null : new Tooltip("Source image: " + src));

        // Show annotation(s) (if any) containing this cell's centroid.
        java.util.List<String> annoNames = controller.getCurrentCellAnnotationNames();
        if (annoNames.isEmpty()) {
            annotationLabel.setText("");
            annotationLabel.setTooltip(null);
        } else {
            String joined = String.join(", ", annoNames);
            annotationLabel.setText("\u25C6 " + joined);
            annotationLabel.setTooltip(new Tooltip("Inside annotation(s): " + joined));
        }

        // Rebuild the prediction buttons for the current cell
        updatePredictionButtons();

        // Determine dot colour
        CellPrediction pred = controller.getCurrentPrediction();
        String existingLabel = null;
        String currentCellId = controller.getCurrentCellId();
        if (currentCellId != null) {
            existingLabel = controller.getOutputLabels().getLabel(currentCellId);
        }

        if (existingLabel == null) {
            statusDot.setFill(Color.WHITE); // unlabelled
        } else if (pred != null && existingLabel.equals(pred.allLabel())) {
            statusDot.setFill(Color.LIMEGREEN); // matches prediction
        } else {
            statusDot.setFill(Color.TOMATO); // mismatch
        }
    }

    // ── Dynamic prediction buttons ──────────────────────────────────────

    private void updatePredictionButtons() {
        predictionBox.getChildren().clear();
        CellPrediction pred = controller.getCurrentPrediction();
        if (pred == null) return;

        if (pred.isDisagreement()) {
            // XGBoost top prediction
            Button xgbBtn =
                    new Button(String.format("XGB: %s (%.0f%%)", pred.getModel1Label(), pred.model1Confidence() * 100));
            xgbBtn.setStyle("-fx-background-color: #bbdefb; -fx-font-weight: bold; -fx-font-size: 11px;");
            xgbBtn.setOnAction(e -> assignAndAdvance(pred.getModel1Label()));

            // LightGBM top prediction
            Button lgbBtn =
                    new Button(String.format("LGB: %s (%.0f%%)", pred.getModel2Label(), pred.model2Confidence() * 100));
            lgbBtn.setStyle("-fx-background-color: #f8bbd0; -fx-font-weight: bold; -fx-font-size: 11px;");
            lgbBtn.setOnAction(e -> assignAndAdvance(pred.getModel2Label()));

            predictionBox.getChildren().addAll(xgbBtn, lgbBtn);

            // If the averaged prediction differs from both, show it too
            String avgLabel = pred.avgLabel();
            if (!avgLabel.equals(pred.getModel1Label()) && !avgLabel.equals(pred.getModel2Label())) {
                Button avgBtn = new Button(String.format("Avg: %s", avgLabel));
                avgBtn.setStyle("-fx-background-color: #c8e6c9; -fx-font-weight: bold; -fx-font-size: 11px;");
                avgBtn.setOnAction(e -> assignAndAdvance(avgLabel));
                predictionBox.getChildren().add(avgBtn);
            }
        } else {
            // Both models agree — single button
            float avgConf = (pred.model1Confidence() + pred.model2Confidence()) / 2f;
            Button agreedBtn = new Button(String.format("Both: %s (%.0f%%)", pred.getModel1Label(), avgConf * 100));
            agreedBtn.setStyle("-fx-background-color: #c8e6c9; -fx-font-weight: bold; -fx-font-size: 11px;");
            agreedBtn.setOnAction(e -> assignAndAdvance(pred.getModel1Label()));
            predictionBox.getChildren().add(agreedBtn);
        }
    }

    /** Assign a class label to the current cell and move to the next one. */
    private void assignAndAdvance(String className) {
        boolean more = controller.labelAndNext(className);
        refreshStatus();
        channelSelector.applyForCurrentCell(controller);
        if (!more) showCompleteAlert();
    }

    // ── All Classes menu ────────────────────────────────────────────────

    /**
     * Restrict class choices to the provided allowed classes when in binary mode.
     *
     * @param markerName    sanitized marker name, used as fallback to construct
     *                      "marker+"/"marker-" if allowedClasses is empty
     * @param allowedClasses the exact class names to show; if null or empty and
     *                       markerName is non-blank, falls back to marker+/marker-
     */
    public void setBinaryMarker(String markerName, List<String> allowedClasses) {
        if (allowedClasses != null && !allowedClasses.isEmpty()) {
            this.binaryClasses = List.copyOf(allowedClasses);
        } else if (markerName != null && !markerName.isBlank()) {
            this.binaryClasses = List.of(markerName + "+", markerName + "-");
        } else {
            this.binaryClasses = null;
        }
        populateAllClassesMenu();
    }

    private void populateAllClassesMenu() {
        allClassesMenu.getItems().clear();

        Set<String> seen = new LinkedHashSet<>();
        if (binaryClasses != null) {
            // Binary mode: only the resolved allowed classes
            seen.addAll(binaryClasses);
        } else {
            // Multi-class mode: QuPath project classes first, then CellTypeTable
            for (String name : controller.getQuPathClassNames()) {
                seen.add(name);
            }
            if (cellTypeTable != null) {
                for (String ct : cellTypeTable.getCellTypes()) {
                    seen.add(ct);
                }
            }
        }

        for (String name : seen) {
            MenuItem item = new MenuItem(name);
            item.setOnAction(e -> assignAndAdvance(name));
            allClassesMenu.getItems().add(item);
        }

        if (allClassesMenu.getItems().isEmpty()) {
            allClassesMenu.setDisable(true);
        } else {
            allClassesMenu.setDisable(false);
        }
    }

    // ── Dialogs ─────────────────────────────────────────────────────────

    private void showJumpDialog() {
        TextInputDialog dlg = new TextInputDialog();
        dlg.setTitle(STRINGS.getString("review.jump.title"));
        dlg.setHeaderText(null);
        dlg.setContentText(STRINGS.getString("review.jump.prompt"));
        dlg.showAndWait().ifPresent(text -> {
            try {
                int target = Integer.parseInt(text.trim()) - 1; // user enters 1-based
                if (controller.jumpTo(target)) {
                    refreshStatus();
                    channelSelector.applyForCurrentCell(controller);
                }
            } catch (NumberFormatException ignored) {
                // silently ignore bad input
            }
        });
    }

    private void showCompleteAlert() {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle(STRINGS.getString("review.complete"));
        alert.setHeaderText(null);
        alert.setContentText(String.format(
                "Review complete. %d / %d cells labelled.", controller.getLabelledCount(), controller.size()));

        // Anchor the alert to the Review Mode window so it stays on top
        // (without an owner it can be hidden behind the toolbar window).
        javafx.stage.Window owner = null;
        if (getScene() != null) {
            owner = getScene().getWindow();
        }
        if (owner != null) {
            alert.initOwner(owner);
            alert.initModality(javafx.stage.Modality.WINDOW_MODAL);
            // Place it near the top-right of the owning window, away from
            // the toolbar buttons on the left.
            double targetX = owner.getX() + Math.max(40, owner.getWidth() - 460);
            double targetY = owner.getY() + 80;
            alert.setX(targetX);
            alert.setY(targetY);
        }
        alert.setOnShown(e -> {
            javafx.stage.Stage stage =
                    (javafx.stage.Stage) alert.getDialogPane().getScene().getWindow();
            stage.setAlwaysOnTop(true);
            stage.toFront();
        });
        alert.showAndWait();
    }
}
