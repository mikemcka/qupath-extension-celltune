package qupath.ext.spclassify.ui;

import java.util.*;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import javafx.scene.text.TextAlignment;
import javafx.stage.Modality;
import javafx.stage.Stage;
import qupath.ext.spclassify.classifier.DualModelClassifier.ShapResult;

/**
 * JavaFX dialog that renders per-class feature importance as a horizontal bar chart.
 * <p>
 * For XGBoost and LightGBM the values are true mean |SHAP| values computed with
 * each model's native TreeSHAP implementation. For Random Forest, normalised
 * split counts are shown (same across classes).
 * <p>
 * A class selector allows switching between cell types. The top
 * {@value #MAX_BARS} features are displayed sorted by importance. The window
 * is resizable; the chart and feature-name column adapt to the available
 * width so long feature names are not cut off.
 */
public class FeatureImportanceView {

    private static final int MAX_BARS = 10;
    private static final int BAR_HEIGHT = 22;
    private static final int BAR_GAP = 4;
    private static final int RIGHT_MARGIN = 70; // space for value labels
    private static final int TOP_MARGIN = 20;
    private static final int BOTTOM_MARGIN = 32;
    private static final int MIN_CHART_WIDTH = 200;
    private static final int MIN_LEFT_MARGIN = 80;
    private static final int LEFT_PADDING = 12; // gap between name and bar

    private static final Font BAR_FONT = Font.font("SansSerif", 11);
    private static final Font AXIS_FONT = Font.font("SansSerif", 10);

    // One colour per class (wraps if more than 8 classes)
    private static final Color[] CLASS_COLORS = {
        Color.rgb(70, 130, 180), // steel blue
        Color.rgb(210, 70, 70), // red
        Color.rgb(55, 155, 55), // green
        Color.rgb(210, 140, 25), // amber
        Color.rgb(145, 75, 195), // purple
        Color.rgb(30, 175, 175), // teal
        Color.rgb(215, 115, 45), // orange
        Color.rgb(110, 110, 110), // gray
    };

    private final Stage stage;
    private final ShapResult shapResult;
    private final Canvas canvas;
    private final Pane canvasHolder;
    private int selectedClassIdx = 0;

    /**
     * Construct and prepare the view. Call {@link #show()} to display it.
     *
     * @param owner      parent stage
     * @param shapResult computed SHAP importances from {@link qupath.ext.spclassify.classifier.DualModelClassifier}
     */
    public FeatureImportanceView(Stage owner, ShapResult shapResult) {
        this.shapResult = shapResult;

        int nBars = Math.min(MAX_BARS, shapResult.featureNames().size());
        int canvasHeight = TOP_MARGIN + nBars * (BAR_HEIGHT + BAR_GAP) + BOTTOM_MARGIN;

        canvas = new Canvas(700, canvasHeight);
        canvasHolder = new Pane(canvas);
        canvasHolder.setStyle("-fx-background-color: white;");
        // Bind canvas size to holder so the chart redraws when the window resizes.
        canvas.widthProperty().bind(canvasHolder.widthProperty());
        canvas.heightProperty().bind(canvasHolder.heightProperty());
        canvas.widthProperty().addListener((obs, o, n) -> drawChart());
        canvas.heightProperty().addListener((obs, o, n) -> drawChart());

        stage = new Stage();
        stage.initOwner(owner);
        stage.initModality(Modality.NONE);
        stage.setTitle("Feature Importance (SHAP)");
        stage.setResizable(true);
        stage.setMinWidth(420);
        stage.setMinHeight(canvasHeight + 90);

        // ── Class selector ──────────────────────────────────────────────────
        Label classLabel = new Label("Class:");
        ComboBox<String> classCombo = new ComboBox<>();
        classCombo.getItems().addAll(shapResult.classNames());
        classCombo.setValue(shapResult.classNames().get(0));
        // Conservative starting cap; refined dynamically when the popup opens
        // so the dropdown never extends past the bottom of the screen.
        classCombo.setVisibleRowCount(
                Math.min(8, Math.max(3, shapResult.classNames().size())));
        classCombo.showingProperty().addListener((obs, was, isNow) -> {
            if (!isNow) return;
            try {
                var screenBounds = javafx.stage.Screen.getPrimary().getVisualBounds();
                var comboBounds = classCombo.localToScreen(classCombo.getBoundsInLocal());
                if (comboBounds == null) return;
                double availablePx = screenBounds.getMaxY() - comboBounds.getMaxY() - 24;
                int rowPx = 26; // approx ComboBox cell height incl. padding
                int maxRows = Math.max(3, (int) (availablePx / rowPx));
                int desired = Math.min(maxRows, shapResult.classNames().size());
                if (desired != classCombo.getVisibleRowCount()) {
                    classCombo.setVisibleRowCount(desired);
                }
            } catch (Exception ignored) {
            }
        });
        classCombo.setOnAction(e -> {
            selectedClassIdx = shapResult.classNames().indexOf(classCombo.getValue());
            drawChart();
        });

        HBox topRow = new HBox(8, classLabel, classCombo);
        topRow.setAlignment(Pos.CENTER_LEFT);
        topRow.setPadding(new Insets(8, 10, 4, 10));

        // ── Subtitle ────────────────────────────────────────────────────────
        Label subtitle = new Label("Top features by mean |SHAP| value  ·  averaged across active models");
        subtitle.setStyle("-fx-font-size: 10px; -fx-text-fill: #666;");
        HBox subtitleRow = new HBox(subtitle);
        subtitleRow.setPadding(new Insets(0, 10, 6, 10));

        // ── Root layout — canvas grows to fill remaining space ──────────────
        VBox root = new VBox(topRow, subtitleRow, canvasHolder);
        root.setStyle("-fx-background-color: white;");
        VBox.setVgrow(canvasHolder, Priority.ALWAYS);

        // Pick an initial scene width wide enough for the longest feature name.
        int initialWidth = computeInitialWidth();
        Scene scene = new Scene(root, initialWidth, canvasHeight + 60);
        stage.setScene(scene);

        drawChart();
    }

    /** Display the window. */
    public void show() {
        stage.show();
    }

    // ── Rendering ───────────────────────────────────────────────────────────────

    /** Measure a string's rendered width with the bar font. */
    private static double measureText(String s, Font font) {
        Text t = new Text(s);
        t.setFont(font);
        return t.getLayoutBounds().getWidth();
    }

    /** Pick a sensible initial scene width: widest top-feature name + chart + margins. */
    private int computeInitialWidth() {
        // Take the maximum width across all classes' top features so resizing
        // between classes doesn't immediately clip.
        double maxNameWidth = 0;
        List<String> names = shapResult.featureNames();
        int nFeatures = names.size();
        for (int c = 0; c < shapResult.classNames().size(); c++) {
            double[] importance = shapResult.meanAbsShap()[c];
            Integer[] order = new Integer[nFeatures];
            for (int i = 0; i < nFeatures; i++) order[i] = i;
            Arrays.sort(order, (a, b) -> Double.compare(importance[b], importance[a]));
            int nBars = Math.min(MAX_BARS, nFeatures);
            for (int i = 0; i < nBars; i++) {
                double w = measureText(names.get(order[i]), BAR_FONT);
                if (w > maxNameWidth) maxNameWidth = w;
            }
        }
        int left = (int) Math.max(MIN_LEFT_MARGIN, maxNameWidth + LEFT_PADDING + 4);
        // Cap initial size so it isn't ridiculous on small screens; user can resize.
        int chart = 380;
        return Math.min(1400, left + chart + RIGHT_MARGIN + 24);
    }

    private void drawChart() {
        GraphicsContext gc = canvas.getGraphicsContext2D();
        double w = canvas.getWidth();
        double h = canvas.getHeight();
        if (w <= 0 || h <= 0) return;
        int c = selectedClassIdx;

        // Background
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, w, h);

        double[] importance = shapResult.meanAbsShap()[c];
        List<String> names = shapResult.featureNames();
        int nFeatures = names.size();

        // Sort by importance descending
        Integer[] order = new Integer[nFeatures];
        for (int i = 0; i < nFeatures; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Double.compare(importance[b], importance[a]));

        int nBars = Math.min(MAX_BARS, nFeatures);
        double maxVal = importance[order[0]];
        if (maxVal <= 0) maxVal = 1.0;

        // Compute dynamic left margin from the widest visible feature name.
        gc.setFont(BAR_FONT);
        double maxNameWidth = 0;
        for (int i = 0; i < nBars; i++) {
            double tw = measureText(names.get(order[i]), BAR_FONT);
            if (tw > maxNameWidth) maxNameWidth = tw;
        }
        double leftMargin = Math.max(MIN_LEFT_MARGIN, maxNameWidth + LEFT_PADDING);
        // Cap left margin so the chart still has minimum width.
        double maxLeft = Math.max(MIN_LEFT_MARGIN, w - RIGHT_MARGIN - MIN_CHART_WIDTH);
        if (leftMargin > maxLeft) leftMargin = maxLeft;
        double chartWidth = Math.max(MIN_CHART_WIDTH, w - leftMargin - RIGHT_MARGIN);

        Color barColor = CLASS_COLORS[c % CLASS_COLORS.length];

        for (int i = 0; i < nBars; i++) {
            int fIdx = order[i];
            String name = names.get(fIdx);
            double val = importance[fIdx];
            double ratio = val / maxVal;

            double y = TOP_MARGIN + i * (BAR_HEIGHT + BAR_GAP);
            double barLen = ratio * chartWidth;

            // Alternating row background
            if (i % 2 == 1) {
                gc.setFill(Color.rgb(246, 246, 250));
                gc.fillRect(0, y, w, BAR_HEIGHT);
            }

            // Bar — opacity scales with relative importance
            Color fill = barColor.deriveColor(0, 1, 1, 0.30 + 0.70 * ratio);
            gc.setFill(fill);
            gc.fillRect(leftMargin, y + 2, barLen, BAR_HEIGHT - 4);

            // Feature name (right-aligned). If the dynamic margin was capped
            // and the name is wider than available space, ellipsise as a last
            // resort so layout doesn't overlap.
            double availableForName = leftMargin - LEFT_PADDING + 6;
            String display = name;
            if (measureText(display, BAR_FONT) > availableForName) {
                display = ellipsise(display, availableForName, BAR_FONT);
            }
            gc.setFill(Color.BLACK);
            gc.setTextAlign(TextAlignment.RIGHT);
            gc.fillText(display, leftMargin - 6, y + BAR_HEIGHT - 6);

            // Value label — inside bar if wide enough, otherwise outside
            String valStr = String.format("%.4f", val);
            gc.setTextAlign(TextAlignment.LEFT);
            if (barLen > 48) {
                gc.setFill(Color.WHITE);
                gc.fillText(valStr, leftMargin + barLen - 46, y + BAR_HEIGHT - 6);
            } else {
                gc.setFill(Color.rgb(80, 80, 80));
                gc.fillText(valStr, leftMargin + barLen + 4, y + BAR_HEIGHT - 6);
            }
        }

        // Y-axis rule
        gc.setStroke(Color.rgb(180, 180, 180));
        gc.setLineWidth(1);
        double axisTop = TOP_MARGIN;
        double axisBottom = TOP_MARGIN + nBars * (BAR_HEIGHT + BAR_GAP);
        gc.strokeLine(leftMargin, axisTop, leftMargin, axisBottom);

        // X-axis label
        gc.setFill(Color.rgb(120, 120, 120));
        gc.setFont(AXIS_FONT);
        gc.setTextAlign(TextAlignment.CENTER);
        gc.fillText(
                "mean |SHAP value|  (average impact on model output)", leftMargin + chartWidth / 2.0, axisBottom + 20);
    }

    /** Truncate {@code s} with an ellipsis so it fits within {@code maxWidth}. */
    private static String ellipsise(String s, double maxWidth, Font font) {
        if (measureText(s, font) <= maxWidth) return s;
        String ellipsis = "\u2026";
        int lo = 0, hi = s.length();
        while (lo < hi) {
            int mid = (lo + hi + 1) >>> 1;
            if (measureText(s.substring(0, mid) + ellipsis, font) <= maxWidth) lo = mid;
            else hi = mid - 1;
        }
        return s.substring(0, lo) + ellipsis;
    }
}
