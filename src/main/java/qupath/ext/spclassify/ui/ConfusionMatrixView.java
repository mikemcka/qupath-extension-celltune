package qupath.ext.spclassify.ui;

import java.io.File;
import java.io.IOException;
import java.util.*;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.TextAlignment;
import javafx.stage.FileChooser;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javax.imageio.ImageIO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.io.ProjectStateManager;
import qupath.ext.spclassify.model.CellPrediction;
import qupath.ext.spclassify.model.LabelStore;
import qupath.ext.spclassify.model.PopulationSet;
import qupath.lib.gui.QuPathGUI;

/**
 * JavaFX window that renders an inter-model confusion matrix.
 * <p>
 * Rows = Model 1 (XGBoost) predictions, Columns = Model 2 (LightGBM) predictions.
 * Diagonal cells show agreement counts; off-diagonal cells show disagreement counts.
 * Cells are colour-coded by magnitude (darker = more cells). Per-class agreement
 * rates are shown in the margins.
 * <p>
 * Also computes per-class agreement rates used by {@code UncertaintySampler} for
 * weighted disagreement sampling.
 */
public class ConfusionMatrixView {

    private static final Logger logger = LoggerFactory.getLogger(ConfusionMatrixView.class);
    private static final String ALL_IMAGES_LABEL = "All Images (Combined)";

    /** Display modes for the confusion matrix. */
    public enum Mode {
        /** Rows = Model 1 (XGBoost), Columns = Model 2 (LightGBM). Inter-model agreement. */
        INTER_MODEL
    }

    private final Stage stage;
    private final Canvas canvas;
    private final Label summaryLabel;

    private final List<String> classNames;
    private final PopulationSet predALL;
    private final LabelStore labelStore;

    // ── Image-selector state (set when project context is available) ─────────
    private final QuPathGUI qupath;
    private final String currentImageName;
    private final Map<String, PopulationSet> predCache;
    private PopulationSet currentDisplayPredAll;

    // Inter-model values are kept stable for external getters (used by the
    // uncertainty sampler) regardless of currently displayed mode.
    private final int[][] interModelMatrix;
    private final double[] agreementRates;
    private final double[] f1Scores;
    private final int interModelTotal;
    private final int interModelAgreements;

    // Currently displayed view (depends on mode).
    private Mode mode = Mode.INTER_MODEL;
    private int[][] matrix;
    private double[] displayF1;
    private double[] displayRowTpRate; // per-row TP / row sum
    private double[] displayColTpRate; // per-col TP / col sum
    private int totalCells;
    private int totalAgreements;
    private String rowAxisTitle = "Model 1 (XGBoost)";
    private String colAxisTitle = "Model 2 (LightGBM)";

    // Layout constants
    private static final int CELL_SIZE = 64;
    private static final int LABEL_MARGIN = 120;
    private static final int RATE_MARGIN = 110;
    private static final int BOTTOM_MARGIN = 80; // bottom margin for Model 2 TP% row
    private static final int HEADER_HEIGHT = 40;
    private static final Font CELL_FONT = Font.font("SansSerif", 13);
    private static final Font LABEL_FONT = Font.font("SansSerif", 12);
    private static final Font HEADER_FONT =
            Font.font("SansSerif", Font.getDefault().getSize() + 1);

    /**
     * Build and display the confusion matrix in inter-model mode (no ground-truth toggle).
     */
    public ConfusionMatrixView(Stage owner, PopulationSet predALL, List<String> classNames) {
        this(owner, predALL, classNames, null, null, null);
    }

    /**
     * Build and display the confusion matrix.
     *
     * @param owner      parent stage
     * @param predALL    the Pred_ALL population set (contains all cell predictions)
     * @param classNames ordered class name list
     * @param labelStore optional ground-truth label store
     */
    public ConfusionMatrixView(Stage owner, PopulationSet predALL, List<String> classNames, LabelStore labelStore) {
        this(owner, predALL, classNames, labelStore, null, null);
    }

    /**
     * Build and display the confusion matrix with image-selector dropdown.
     * When {@code qupath} and {@code currentImageName} are supplied, a combo box
     * appears at the top so the user can switch between images or view a combined
     * matrix across all project images.
     *
     * @param owner            parent stage
     * @param predALL          predictions for the currently open image
     * @param classNames       ordered class name list
     * @param labelStore       optional ground-truth label store
     * @param qupath           QuPath GUI (for loading other images' predictions)
     * @param currentImageName image name key for the supplied predALL
     */
    public ConfusionMatrixView(
            Stage owner,
            PopulationSet predALL,
            List<String> classNames,
            LabelStore labelStore,
            QuPathGUI qupath,
            String currentImageName) {
        this.classNames = List.copyOf(classNames);
        this.predALL = predALL;
        this.labelStore = labelStore;
        this.qupath = qupath;
        this.currentImageName = currentImageName;
        this.predCache = new LinkedHashMap<>();
        this.currentDisplayPredAll = predALL;
        if (currentImageName != null && predALL != null) predCache.put(currentImageName, predALL);
        int n = classNames.size();
        stage = new Stage();

        // ── Build inter-model confusion matrix (kept as the canonical source ─
        //    of agreement rates / F1 used by the uncertainty sampler) ─────────
        int[][] imMatrix = new int[n][n];
        int imAgreements = 0;
        int imTotal = 0;
        for (CellPrediction pred : predALL.getAll().values()) {
            int row = classNames.indexOf(pred.getModel1Label());
            int col = classNames.indexOf(pred.getModel2Label());
            if (row < 0 || col < 0) continue;
            imMatrix[row][col]++;
            imTotal++;
            if (row == col) imAgreements++;
        }
        this.interModelMatrix = imMatrix;
        this.interModelTotal = imTotal;
        this.interModelAgreements = imAgreements;

        // ── Per-class agreement rates (inter-model, used by sampler) ────────
        this.agreementRates = new double[n];
        this.f1Scores = new double[n];
        for (int i = 0; i < n; i++) {
            int rowSum = 0, colSum = 0;
            for (int j = 0; j < n; j++) {
                rowSum += imMatrix[i][j];
                colSum += imMatrix[j][i];
            }
            int denom = rowSum + colSum - imMatrix[i][i];
            agreementRates[i] = denom > 0 ? (double) imMatrix[i][i] / denom : 1.0;
            int tp = imMatrix[i][i];
            int fp = colSum - tp;
            int fn = rowSum - tp;
            double precision = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0;
            double recall = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0;
            f1Scores[i] = (precision + recall) > 0 ? 2.0 * precision * recall / (precision + recall) : 0;
        }

        // ── Canvas ──────────────────────────────────────────────────────────
        int canvasWidth = LABEL_MARGIN + n * CELL_SIZE + RATE_MARGIN;
        int canvasHeight = HEADER_HEIGHT + LABEL_MARGIN + n * CELL_SIZE + BOTTOM_MARGIN + RATE_MARGIN;
        canvas = new Canvas(canvasWidth, canvasHeight);

        // ── Summary label ───────────────────────────────────────────────────
        summaryLabel = new Label("");
        summaryLabel.setPadding(new Insets(4));

        // Compute initial display state (defaults to INTER_MODEL).
        applyMode(Mode.INTER_MODEL);

        // ── Export button ───────────────────────────────────────────────────
        Button exportBtn = new Button("Export as PNG…");
        exportBtn.setOnAction(e -> exportAsPng(owner));

        Button closeBtn = new Button("Close");
        closeBtn.setOnAction(e -> stage.close());

        HBox buttons = new HBox(8, exportBtn, closeBtn);
        buttons.setAlignment(Pos.CENTER_RIGHT);
        buttons.setPadding(new Insets(8));

        VBox bottom = new VBox(4, summaryLabel, buttons);
        bottom.setAlignment(Pos.CENTER);

        BorderPane root = new BorderPane();
        // Wrap the canvas in a ScrollPane so large class counts (which produce
        // a very tall canvas) don't push the window off the bottom of the
        // screen. Cap the scene to ~85% of screen height/width so the matrix
        // remains usable on smaller displays.
        javafx.scene.control.ScrollPane scroll = new javafx.scene.control.ScrollPane(canvas);
        scroll.setFitToWidth(false);
        scroll.setFitToHeight(false);
        scroll.setPannable(true);
        scroll.setHbarPolicy(javafx.scene.control.ScrollPane.ScrollBarPolicy.AS_NEEDED);
        scroll.setVbarPolicy(javafx.scene.control.ScrollPane.ScrollBarPolicy.AS_NEEDED);
        root.setCenter(scroll);
        root.setBottom(bottom);
        if (qupath != null) {
            root.setTop(buildImageSelector());
        }
        root.setPadding(new Insets(8));

        // Inter-model is the only remaining view here. Ground-truth vs model
        // confusion matrices used to live behind a dropdown; they were removed
        // because they were computed on the full labelled set (training data
        // included) and therefore gave optimistic, training-set-contaminated
        // numbers. The honest version lives in ValidationConfusionMatrixView,
        // which is reachable from the Training Metrics dialog and uses only
        // the held-out 20% validation split.

        stage.initOwner(owner);
        stage.initModality(Modality.NONE);
        stage.setTitle("SP Classify \u2014 Inter-Model Confusion Matrix");

        var screen = javafx.stage.Screen.getPrimary().getVisualBounds();
        double maxW = screen.getWidth() * 0.85;
        double maxH = screen.getHeight() * 0.85;
        // Add a small slack for ScrollPane chrome and padding.
        double sceneW = Math.min(maxW, canvasWidth + 40);
        double sceneH = Math.min(maxH, canvasHeight + 80);
        stage.setScene(new Scene(root, sceneW, sceneH));
        stage.setResizable(true);
    }

    /** Show the confusion matrix window. */
    public void show() {
        stage.show();
        stage.toFront();
    }

    /**
     * Get the per-class agreement rate array.
     * Index i corresponds to classNames[i].
     * Values range from 0.0 (never agree) to 1.0 (always agree).
     */
    public double[] getAgreementRates() {
        return agreementRates.clone();
    }

    /**
     * Get the per-class Dice similarity coefficient array.
     * Dice (= F1) treats diagonal as TP, row off-diagonal as FN, col off-diagonal as FP.
     * This measures inter-model agreement per class, NOT ground-truth classification accuracy.
     */
    public double[] getF1Scores() {
        return f1Scores.clone();
    }

    /** @return the raw confusion matrix (rows = model 1, cols = model 2). */
    public int[][] getMatrix() {
        int n = matrix.length;
        int[][] copy = new int[n][n];
        for (int i = 0; i < n; i++) {
            copy[i] = matrix[i].clone();
        }
        return copy;
    }

    /** @return total cells used to build the matrix. */
    public int getTotalCells() {
        return totalCells;
    }

    /** @return total cells where both models agreed. */
    public int getTotalAgreements() {
        return totalAgreements;
    }

    // ── Drawing ─────────────────────────────────────────────────────────────────

    /**
     * Recompute the displayed matrix and summary for the given mode and redraw.
     * Falls back to INTER_MODEL when GT modes are requested but no labels exist.
     */
    private void applyMode(Mode requested) {
        // Only INTER_MODEL is supported now. The ground-truth modes were
        // removed because they evaluated on the full labelled set (training
        // data included) — see ValidationConfusionMatrixView for the honest,
        // held-out version.
        this.mode = Mode.INTER_MODEL;

        int n = classNames.size();
        int[][] mtx = new int[n][n];
        int total;
        int agreements;

        // Build matrix from the currently selected image (or merged set).
        total = 0;
        agreements = 0;
        PopulationSet src = currentDisplayPredAll;
        if (src != null) {
            for (CellPrediction pred : src.getAll().values()) {
                int row = classNames.indexOf(pred.getModel1Label());
                int col = classNames.indexOf(pred.getModel2Label());
                if (row < 0 || col < 0) continue;
                mtx[row][col]++;
                total++;
                if (row == col) agreements++;
            }
        } else {
            // Fallback to canonical inter-model snapshot.
            for (int i = 0; i < n; i++) System.arraycopy(interModelMatrix[i], 0, mtx[i], 0, n);
            total = interModelTotal;
            agreements = interModelAgreements;
        }
        rowAxisTitle = "Model 1 (XGBoost)";
        colAxisTitle = "Model 2 (LightGBM)";
        stage.setTitle("SP Classify \u2014 Inter-Model Confusion Matrix");

        // Per-row / per-col TP rates and per-class F1.
        double[] rowTp = new double[n];
        double[] colTp = new double[n];
        double[] f1 = new double[n];
        for (int i = 0; i < n; i++) {
            int rowSum = 0, colSum = 0;
            for (int j = 0; j < n; j++) {
                rowSum += mtx[i][j];
                colSum += mtx[j][i];
            }
            rowTp[i] = rowSum > 0 ? (double) mtx[i][i] / rowSum : 0;
            colTp[i] = colSum > 0 ? (double) mtx[i][i] / colSum : 0;
            int tp = mtx[i][i];
            int fp = colSum - tp;
            int fn = rowSum - tp;
            double precision = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0;
            double recall = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0;
            f1[i] = (precision + recall) > 0 ? 2.0 * precision * recall / (precision + recall) : 0;
        }

        this.matrix = mtx;
        this.totalCells = total;
        this.totalAgreements = agreements;
        this.displayRowTpRate = rowTp;
        this.displayColTpRate = colTp;
        this.displayF1 = f1;

        // Summary line.
        double overallRate = total > 0 ? (double) agreements / total * 100 : 0;
        double macroF1 = 0;
        for (double f : f1) macroF1 += f;
        macroF1 = n > 0 ? macroF1 / n : 0;
        String summary = String.format(
                "Total: %,d cells | Agreement: %,d (%.1f%%) | Disagreement: %,d (%.1f%%) | Macro Dice: %.3f",
                total, agreements, overallRate, total - agreements, 100 - overallRate, macroF1);
        summaryLabel.setText(summary);

        drawMatrix();
    }

    private void drawMatrix() {
        GraphicsContext gc = canvas.getGraphicsContext2D();
        int n = classNames.size();

        // Clear
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());

        // ── Separate color scales for diagonal vs off-diagonal ──────────────
        // (matching Python CellTune: diagonal has its own max, off-diagonal its own)
        int maxDiag = 1;
        int maxOffDiag = 1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    if (matrix[i][j] > maxDiag) maxDiag = matrix[i][j];
                } else {
                    if (matrix[i][j] > maxOffDiag) maxOffDiag = matrix[i][j];
                }
            }
        }

        // ── Per-model TP rates (for margins) — computed in applyMode() ─────
        double[] model1TpRate = displayRowTpRate;
        double[] model2TpRate = displayColTpRate;

        double gridLeft = LABEL_MARGIN;
        double gridTop = HEADER_HEIGHT + LABEL_MARGIN;

        // ── Axis labels ─────────────────────────────────────────────────────
        gc.setFont(HEADER_FONT);
        gc.setTextAlign(TextAlignment.CENTER);
        gc.setFill(Color.BLACK);

        // X axis title
        gc.fillText(colAxisTitle, gridLeft + n * CELL_SIZE / 2.0, HEADER_HEIGHT / 2.0 + 4);

        // Y axis title (rotated)
        gc.save();
        gc.translate(14, gridTop + n * CELL_SIZE / 2.0);
        gc.rotate(-90);
        gc.fillText(rowAxisTitle, 0, 0);
        gc.restore();

        gc.setFont(LABEL_FONT);

        // Column labels (top)
        for (int j = 0; j < n; j++) {
            double cx = gridLeft + j * CELL_SIZE + CELL_SIZE / 2.0;
            gc.save();
            gc.translate(cx, gridTop - 6);
            gc.rotate(-45);
            gc.setTextAlign(TextAlignment.LEFT);
            gc.setFill(Color.BLACK);
            gc.fillText(classNames.get(j), 0, 0);
            gc.restore();
        }

        // Row labels (left)
        gc.setTextAlign(TextAlignment.RIGHT);
        for (int i = 0; i < n; i++) {
            double cy = gridTop + i * CELL_SIZE + CELL_SIZE / 2.0 + 4;
            gc.setFill(Color.BLACK);
            gc.fillText(classNames.get(i), gridLeft - 6, cy);
        }

        // ── Cells ───────────────────────────────────────────────────────────
        gc.setFont(CELL_FONT);
        gc.setTextAlign(TextAlignment.CENTER);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double x = gridLeft + j * CELL_SIZE;
                double y = gridTop + i * CELL_SIZE;
                int val = matrix[i][j];

                Color fill;
                if (val == 0) {
                    // Blank cell for zero values (matching Python CellTune NaN approach)
                    fill = Color.WHITE;
                } else if (i == j) {
                    // Diagonal: blue scale (agreement) with separate max
                    double intensity = (double) val / maxDiag;
                    fill = Color.color(
                            1.0 - 0.7 * intensity, // R: 1.0 → 0.3
                            1.0 - 0.5 * intensity, // G: 1.0 → 0.5
                            1.0); // B: stays 1.0
                } else {
                    // Off-diagonal: orangered scale (disagreement) with separate max
                    double intensity = (double) val / maxOffDiag;
                    fill = Color.color(
                            1.0, // R: stays 1.0
                            0.98 - 0.62 * intensity, // G: ~1.0 → 0.36
                            0.96 - 0.76 * intensity); // B: ~1.0 → 0.20
                }

                gc.setFill(fill);
                gc.fillRect(x, y, CELL_SIZE, CELL_SIZE);

                // Border
                gc.setStroke(Color.gray(0.7));
                gc.setLineWidth(0.5);
                gc.strokeRect(x, y, CELL_SIZE, CELL_SIZE);

                // Value text (skip zeros — leave blank)
                if (val > 0) {
                    double threshold = (i == j) ? maxDiag * 0.65 : maxOffDiag * 0.65;
                    gc.setFill(val > threshold ? Color.WHITE : Color.BLACK);
                    gc.fillText(String.valueOf(val), x + CELL_SIZE / 2.0, y + CELL_SIZE / 2.0 + 5);
                }
            }
        }

        // ── Right margin: Model 1 TP% (per-class) ──────────────────────────
        gc.setFont(LABEL_FONT);
        gc.setTextAlign(TextAlignment.LEFT);
        double rightX = gridLeft + n * CELL_SIZE + 6;
        for (int i = 0; i < n; i++) {
            double y = gridTop + i * CELL_SIZE + CELL_SIZE / 2.0 + 4;
            // Colour the TP% cell background (blue scale like diagonal)
            double bgIntensity = model1TpRate[i];
            Color bg = Color.color(1.0 - 0.7 * bgIntensity, 1.0 - 0.5 * bgIntensity, 1.0);
            gc.setFill(bg);
            gc.fillRect(rightX - 4, gridTop + i * CELL_SIZE, 46, CELL_SIZE);
            gc.setStroke(Color.gray(0.7));
            gc.setLineWidth(0.5);
            gc.strokeRect(rightX - 4, gridTop + i * CELL_SIZE, 46, CELL_SIZE);
            // Text
            gc.setFill(bgIntensity > 0.65 ? Color.WHITE : Color.BLACK);
            gc.fillText(String.format("%.0f%%", model1TpRate[i] * 100), rightX, y);
        }

        // TP% header (right)
        gc.setFill(Color.DARKBLUE);
        gc.fillText("TP%", rightX, gridTop - 6);

        // ── Right margin: per-class F1 scores ───────────────────────────────
        double f1X = rightX + 50;
        for (int i = 0; i < n; i++) {
            double y = gridTop + i * CELL_SIZE + CELL_SIZE / 2.0 + 4;
            gc.setFill(Color.DARKRED);
            gc.fillText(String.format("%.2f", displayF1[i]), f1X, y);
        }

        // Dice column header
        gc.setFill(Color.DARKRED);
        gc.fillText("Dice", f1X, gridTop - 6);

        // ── Bottom margin: Model 2 TP% (per-class) ─────────────────────────
        double bottomY = gridTop + n * CELL_SIZE + 6;
        gc.setTextAlign(TextAlignment.CENTER);
        for (int j = 0; j < n; j++) {
            double x = gridLeft + j * CELL_SIZE;
            // Colour the TP% cell background (blue scale like diagonal)
            double bgIntensity = model2TpRate[j];
            Color bg = Color.color(1.0 - 0.7 * bgIntensity, 1.0 - 0.5 * bgIntensity, 1.0);
            gc.setFill(bg);
            gc.fillRect(x, bottomY - 4, CELL_SIZE, 30);
            gc.setStroke(Color.gray(0.7));
            gc.setLineWidth(0.5);
            gc.strokeRect(x, bottomY - 4, CELL_SIZE, 30);
            // Text
            gc.setFill(bgIntensity > 0.65 ? Color.WHITE : Color.BLACK);
            gc.fillText(String.format("%.0f%%", model2TpRate[j] * 100), x + CELL_SIZE / 2.0, bottomY + 14);
        }

        // TP% label (bottom left)
        gc.setTextAlign(TextAlignment.RIGHT);
        gc.setFill(Color.DARKBLUE);
        gc.fillText("TP%", gridLeft - 6, bottomY + 14);

        // ── Bottom: column labels (class names) ─────────────────────────────
        gc.setFont(LABEL_FONT);
        gc.setTextAlign(TextAlignment.CENTER);
        double bottomLabelY = bottomY + 44;
        for (int j = 0; j < n; j++) {
            double cx = gridLeft + j * CELL_SIZE + CELL_SIZE / 2.0;
            gc.save();
            gc.translate(cx, bottomLabelY);
            gc.rotate(-45);
            gc.setTextAlign(TextAlignment.RIGHT);
            gc.setFill(Color.BLACK);
            gc.fillText(classNames.get(j), 0, 0);
            gc.restore();
        }

        // ── Bottom key: metric definitions ────────────────────────────────
        gc.setFont(Font.font("SansSerif", 11));
        gc.setTextAlign(TextAlignment.LEFT);
        gc.setFill(Color.gray(0.25));
        double keyX = gridLeft;
        double keyY = bottomLabelY + 26;
        gc.fillText("Key: TP% = true positive rate (recall) for that model margin: TP/(TP+FN)", keyX, keyY);
        gc.fillText("      Dice = per-class overlap score: 2TP/(2TP+FP+FN)", keyX, keyY + 16);
    }

    // ── Image selector ───────────────────────────────────────────────────────

    private HBox buildImageSelector() {
        var combo = new ComboBox<String>();
        combo.setMaxWidth(Double.MAX_VALUE);
        combo.getItems().add(ALL_IMAGES_LABEL);
        if (currentImageName != null) combo.getItems().add(currentImageName);

        var project = qupath.getProject();
        if (project != null) {
            new Thread(
                            () -> {
                                try {
                                    List<String> names = ProjectStateManager.listImagesWithPredictions(project);
                                    Platform.runLater(() -> {
                                        for (String name : names) {
                                            if (!combo.getItems().contains(name))
                                                combo.getItems().add(name);
                                        }
                                        combo.setValue(currentImageName != null ? currentImageName : ALL_IMAGES_LABEL);
                                    });
                                } catch (Exception ex) {
                                    logger.warn("Failed to list images with predictions", ex);
                                }
                            },
                            "CMV-ImageListLoader")
                    .start();
        } else {
            combo.setValue(currentImageName != null ? currentImageName : ALL_IMAGES_LABEL);
        }

        combo.setOnAction(e -> {
            String selected = combo.getValue();
            if (selected == null) return;
            loadAndDisplaySelection(selected);
        });

        var label = new Label("Image:");
        label.setPadding(new Insets(0, 4, 0, 4));
        var bar = new HBox(6, label, combo);
        bar.setPadding(new Insets(6, 8, 2, 8));
        HBox.setHgrow(combo, Priority.ALWAYS);
        return bar;
    }

    private void loadAndDisplaySelection(String selected) {
        if (ALL_IMAGES_LABEL.equals(selected)) {
            var project = qupath.getProject();
            if (project == null) return;
            new Thread(
                            () -> {
                                try {
                                    List<String> names = ProjectStateManager.listImagesWithPredictions(project);
                                    for (String name : names) {
                                        if (!predCache.containsKey(name)) {
                                            PopulationSet ps = ProjectStateManager.loadImagePredictions(project, name);
                                            if (ps != null) predCache.put(name, ps);
                                        }
                                    }
                                    PopulationSet merged = mergeAllCached();
                                    Platform.runLater(() -> {
                                        currentDisplayPredAll = merged;
                                        applyMode(Mode.INTER_MODEL);
                                        stage.setTitle("SP Classify — Agreement Matrix — All Images");
                                    });
                                } catch (Exception ex) {
                                    logger.warn("Failed to merge predictions across images", ex);
                                }
                            },
                            "CMV-AllImagesLoader")
                    .start();
        } else {
            if (predCache.containsKey(selected)) {
                currentDisplayPredAll = predCache.get(selected);
                applyMode(Mode.INTER_MODEL);
                stage.setTitle("SP Classify — Agreement Matrix — " + selected);
            } else {
                var project = qupath.getProject();
                if (project == null) return;
                new Thread(
                                () -> {
                                    try {
                                        PopulationSet ps = ProjectStateManager.loadImagePredictions(project, selected);
                                        if (ps != null) {
                                            predCache.put(selected, ps);
                                            Platform.runLater(() -> {
                                                currentDisplayPredAll = ps;
                                                applyMode(Mode.INTER_MODEL);
                                                stage.setTitle("SP Classify — Agreement Matrix — " + selected);
                                            });
                                        }
                                    } catch (Exception ex) {
                                        logger.warn("Failed to load predictions for '{}'", selected, ex);
                                    }
                                },
                                "CMV-ImageLoader")
                        .start();
            }
        }
    }

    /** Merge all cached per-image predictions, prefixing cell IDs to avoid key collisions. */
    private PopulationSet mergeAllCached() {
        PopulationSet merged = new PopulationSet("Pred_ALL_merged");
        for (Map.Entry<String, PopulationSet> entry : predCache.entrySet()) {
            String imgName = entry.getKey();
            PopulationSet ps = entry.getValue();
            if (ps == null) continue;
            for (Map.Entry<String, CellPrediction> cell : ps.getAll().entrySet()) {
                merged.put(imgName + "|" + cell.getKey(), cell.getValue());
            }
        }
        return merged;
    }

    // ── PNG export ──────────────────────────────────────────────────────────────

    private void exportAsPng(Stage owner) {
        FileChooser fc = new FileChooser();
        fc.setTitle("Save Confusion Matrix as PNG");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("PNG Image", "*.png"));
        fc.setInitialFileName("confusion_matrix.png");
        File file = fc.showSaveDialog(owner);
        if (file == null) return;

        try {
            var params = new SnapshotParameters();
            params.setFill(Color.WHITE);
            var image = canvas.snapshot(params, null);
            ImageIO.write(SwingFXUtils.fromFXImage(image, null), "png", file);
            logger.info("Exported confusion matrix to {}", file);
        } catch (IOException ex) {
            logger.error("Failed to export confusion matrix", ex);
        }
    }
}
