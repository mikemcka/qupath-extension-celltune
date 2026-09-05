package qupath.ext.spclassify.ui;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.TextAlignment;
import javafx.stage.FileChooser;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javax.imageio.ImageIO;
import qupath.ext.spclassify.classifier.TrainingMetrics;

/**
 * Held-out validation-set confusion matrix for a single model.
 *
 * <p>Two heatmaps are drawn side-by-side, mirroring the layout of the legacy
 * pipeline's {@code prepare_plot}:
 * <ul>
 *   <li><b>Absolute</b> &mdash; raw counts per (true, predicted) class pair.</li>
 *   <li><b>Normalised</b> &mdash; row-normalised so each row sums to 1.0,
 *       i.e. <code>cell[i][j] = count[i][j] / sum_j(count[i][j])</code>.
 *       This is the standard recall view: the diagonal entry of row <em>i</em>
 *       is the recall for class <em>i</em>.</li>
 * </ul>
 *
 * <p>Rows = true class (from labels). Columns = predicted class. Diagonal
 * cells show correct predictions; off-diagonal cells reveal which classes the
 * model confuses for which.
 *
 * <p>Computed on the 20% held-out validation split only &mdash; cells used to
 * train the evaluation model are <em>not</em> included, so these numbers are
 * an honest measure of out-of-sample performance.
 */
public class ValidationConfusionMatrixView {

    private final Stage stage = new Stage();
    private Canvas canvas;

    private final String modelLabel;
    private final List<String> classNames;
    private final int[][] cm; // rows = true, cols = predicted
    private final int[] rowSum;
    private final int total;
    private final int correct;
    private final double macroF1;

    // Layout constants for one heatmap
    private static final int CELL_SIZE = 56;
    private static final int LABEL_MARGIN = 110;
    private static final int HEADER_HEIGHT = 60;
    private static final int RIGHT_PAD = 30;
    private static final int BOTTOM_PAD = 90; // X-axis title + count row
    private static final int PANEL_GAP = 30;

    private static final Font CELL_FONT = Font.font("SansSerif", 12);
    private static final Font LABEL_FONT = Font.font("SansSerif", 12);
    private static final Font HEADER_FONT = Font.font("SansSerif", 14);

    public ValidationConfusionMatrixView(Stage owner, String modelLabel, TrainingMetrics validationMetrics) {
        if (validationMetrics == null) {
            throw new IllegalArgumentException("validationMetrics is null \u2014 train the classifier first");
        }
        this.modelLabel = modelLabel;
        this.classNames = validationMetrics.classNames();
        this.cm = validationMetrics.confusionMatrix();
        this.total = validationMetrics.total();

        int n = classNames.size();
        this.rowSum = new int[n];
        int corr = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                rowSum[i] += cm[i][j];
            }
            corr += cm[i][i];
        }
        this.correct = corr;

        double f1Sum = 0;
        for (double f : validationMetrics.f1()) f1Sum += f;
        this.macroF1 = n > 0 ? f1Sum / n : 0;

        int panelW = LABEL_MARGIN + n * CELL_SIZE + RIGHT_PAD;
        int panelH = HEADER_HEIGHT + LABEL_MARGIN + n * CELL_SIZE + BOTTOM_PAD;
        int canvasW = panelW * 2 + PANEL_GAP;
        int canvasH = panelH;

        this.canvas = new Canvas(canvasW, canvasH);
        drawAll(canvas.getGraphicsContext2D(), panelW);

        Label header = new Label(String.format("Validation Confusion Matrix \u2014 %s (held-out 20%%)", modelLabel));
        header.setFont(Font.font("SansSerif", 14));

        Label summary = new Label(String.format(
                "Total: %,d cells | Correct: %,d (%.1f%%) | Misclassified: %,d (%.1f%%) | Macro F1: %.3f",
                total,
                correct,
                total > 0 ? 100.0 * correct / total : 0,
                total - correct,
                total > 0 ? 100.0 * (total - correct) / total : 0,
                macroF1));

        Button exportCsv = new Button("Download CSV\u2026");
        exportCsv.setOnAction(e -> exportCsv(owner));
        Button exportPng = new Button("Download PNG…");
        exportPng.setOnAction(e -> exportPng(owner));
        Button close = new Button("Close");
        close.setOnAction(e -> stage.close());

        HBox buttons = new HBox(8, exportPng, exportCsv, close);
        buttons.setAlignment(Pos.CENTER_RIGHT);
        buttons.setPadding(new Insets(8));

        VBox bottom = new VBox(4, summary, buttons);
        bottom.setAlignment(Pos.CENTER);

        ScrollPane scroll = new ScrollPane(canvas);
        scroll.setPannable(true);
        scroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.AS_NEEDED);
        scroll.setVbarPolicy(ScrollPane.ScrollBarPolicy.AS_NEEDED);

        BorderPane root = new BorderPane();
        root.setTop(header);
        BorderPane.setMargin(header, new Insets(8, 8, 4, 8));
        root.setCenter(scroll);
        root.setBottom(bottom);
        root.setPadding(new Insets(8));

        stage.initOwner(owner);
        stage.initModality(Modality.NONE);
        stage.setTitle("SP Classify \u2014 Validation Confusion Matrix");

        var screen = javafx.stage.Screen.getPrimary().getVisualBounds();
        double sceneW = Math.min(screen.getWidth() * 0.9, canvasW + 40);
        double sceneH = Math.min(screen.getHeight() * 0.85, canvasH + 130);
        stage.setScene(new Scene(root, sceneW, sceneH));
        stage.setResizable(true);
    }

    public void show() {
        stage.show();
        stage.toFront();
    }

    // ── Drawing ─────────────────────────────────────────────────────────────

    private void drawAll(GraphicsContext gc, int panelW) {
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, gc.getCanvas().getWidth(), gc.getCanvas().getHeight());
        drawPanel(gc, 0, "Absolute counts", false);
        drawPanel(gc, panelW + PANEL_GAP, "Row-normalised (recall)", true);
    }

    private void drawPanel(GraphicsContext gc, double xOffset, String title, boolean normalised) {
        int n = classNames.size();

        double gridLeft = xOffset + LABEL_MARGIN;
        double gridTop = HEADER_HEIGHT + LABEL_MARGIN;

        // Title
        gc.setFont(HEADER_FONT);
        gc.setFill(Color.BLACK);
        gc.setTextAlign(TextAlignment.CENTER);
        gc.fillText(title, xOffset + (LABEL_MARGIN + n * CELL_SIZE) / 2.0, 24);

        // Y-axis title (rotated) — "True class"
        gc.setFont(LABEL_FONT);
        gc.save();
        gc.translate(xOffset + 14, gridTop + n * CELL_SIZE / 2.0);
        gc.rotate(-90);
        gc.fillText("True class", 0, 0);
        gc.restore();

        // X-axis title — "Predicted class" (below the grid)
        gc.setTextAlign(TextAlignment.CENTER);
        gc.fillText("Predicted class", gridLeft + n * CELL_SIZE / 2.0, gridTop + n * CELL_SIZE + 56);

        // Column labels (top, rotated -45°)
        for (int j = 0; j < n; j++) {
            double cx = gridLeft + j * CELL_SIZE + CELL_SIZE / 2.0;
            gc.save();
            gc.translate(cx, gridTop - 6);
            gc.rotate(-45);
            gc.setTextAlign(TextAlignment.LEFT);
            gc.fillText(truncate(classNames.get(j), 18), 0, 0);
            gc.restore();
        }

        // Row labels (left)
        gc.setTextAlign(TextAlignment.RIGHT);
        for (int i = 0; i < n; i++) {
            double cy = gridTop + i * CELL_SIZE + CELL_SIZE / 2.0 + 4;
            gc.fillText(truncate(classNames.get(i), 16), gridLeft - 6, cy);
        }

        // Find max for colour scaling
        int maxAbs = 0;
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if (cm[i][j] > maxAbs) maxAbs = cm[i][j];
        if (maxAbs < 1) maxAbs = 1;

        // Cells
        gc.setFont(CELL_FONT);
        gc.setTextAlign(TextAlignment.CENTER);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double x = gridLeft + j * CELL_SIZE;
                double y = gridTop + i * CELL_SIZE;

                int count = cm[i][j];
                double normValue = rowSum[i] > 0 ? (double) count / rowSum[i] : 0;

                // Colour intensity: normalised value when normalised view,
                // count / maxAbs when absolute. Both range 0..1.
                double intensity = normalised ? normValue : (double) count / maxAbs;
                gc.setFill(blues(intensity));
                gc.fillRect(x, y, CELL_SIZE, CELL_SIZE);

                // Cell border
                gc.setStroke(Color.gray(0.7));
                gc.setLineWidth(0.5);
                gc.strokeRect(x, y, CELL_SIZE, CELL_SIZE);

                // Cell text — pick black or white for legibility based on intensity.
                // Blues ramp is light at low values, dark at high values,
                // so flip to white only when the background is genuinely dark.
                gc.setFill(intensity > 0.65 ? Color.WHITE : Color.BLACK);
                String label = normalised ? String.format("%.3f", normValue) : Integer.toString(count);
                gc.fillText(label, x + CELL_SIZE / 2.0, y + CELL_SIZE / 2.0 + 4);
            }
        }
    }

    /**
     * Light-to-dark blue gradient (approximation of matplotlib's Blues).
     * Low values are near-white so black cell text stays readable; only the
     * hottest cells go dark, where we switch to white text.
     */
    private static Color blues(double t) {
        if (Double.isNaN(t)) t = 0;
        t = Math.max(0, Math.min(1, t));
        // 9-stop approximation of matplotlib's Blues colormap (RGB 0..1).
        double[][] stops = {
            {0.969, 0.984, 1.000}, // very light
            {0.871, 0.922, 0.969},
            {0.776, 0.859, 0.937},
            {0.620, 0.792, 0.882},
            {0.420, 0.682, 0.839},
            {0.259, 0.573, 0.776},
            {0.129, 0.443, 0.710},
            {0.031, 0.318, 0.612},
            {0.031, 0.188, 0.420} // dark navy
        };
        double scaled = t * (stops.length - 1);
        int idx = (int) Math.floor(scaled);
        if (idx >= stops.length - 1)
            return Color.color(stops[stops.length - 1][0], stops[stops.length - 1][1], stops[stops.length - 1][2]);
        double frac = scaled - idx;
        double r = stops[idx][0] + frac * (stops[idx + 1][0] - stops[idx][0]);
        double g = stops[idx][1] + frac * (stops[idx + 1][1] - stops[idx][1]);
        double b = stops[idx][2] + frac * (stops[idx + 1][2] - stops[idx][2]);
        return Color.color(r, g, b);
    }

    private static String truncate(String s, int max) {
        return s.length() > max ? s.substring(0, max - 1) + "\u2026" : s;
    }

    // ── CSV export ──────────────────────────────────────────────────────────

    /**
     * Export the validation confusion matrix as a long-format CSV:
     * {@code true_class,predicted_class,count,row_normalised}
     */
    private void exportCsv(Stage owner) {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Export validation confusion matrix");
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV files", "*.csv"));
        chooser.setInitialFileName(String.format(
                "sp-classify_validation_cm_%s.csv", modelLabel.toLowerCase().replaceAll("[^a-z0-9]+", "_")));
        var target = chooser.showSaveDialog(owner);
        if (target == null) return;

        Path path = target.toPath();
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(path))) {
            w.println("true_class,predicted_class,count,row_normalised");
            int n = classNames.size();
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    int count = cm[i][j];
                    double norm = rowSum[i] > 0 ? (double) count / rowSum[i] : 0;
                    w.printf("%s,%s,%d,%.6f%n", csv(classNames.get(i)), csv(classNames.get(j)), count, norm);
                }
            }
        } catch (IOException ex) {
            Alert a = new Alert(Alert.AlertType.ERROR, "Failed to write CSV: " + ex.getMessage());
            a.initOwner(stage);
            a.showAndWait();
            return;
        }

        Alert ok = new Alert(Alert.AlertType.INFORMATION, "Saved to:\n" + path);
        ok.setHeaderText("Validation confusion matrix exported");
        ok.initOwner(stage);
        ok.showAndWait();
    }

    private static String csv(String s) {
        if (s == null) return "";
        if (s.indexOf(',') < 0 && s.indexOf('"') < 0 && s.indexOf('\n') < 0) return s;
        return "\"" + s.replace("\"", "\"\"") + "\"";
    }

    // ── PNG export ────────────────────────────────────────────────────

    /** Snapshot the canvas and save it as a PNG image. */
    private void exportPng(Stage owner) {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Export validation confusion matrix as PNG");
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("PNG image", "*.png"));
        chooser.setInitialFileName(String.format(
                "sp-classify_validation_cm_%s.png", modelLabel.toLowerCase().replaceAll("[^a-z0-9]+", "_")));
        File target = chooser.showSaveDialog(owner);
        if (target == null) return;

        try {
            SnapshotParameters params = new SnapshotParameters();
            params.setFill(Color.WHITE);
            WritableImage img = new WritableImage((int) canvas.getWidth(), (int) canvas.getHeight());
            canvas.snapshot(params, img);
            ImageIO.write(SwingFXUtils.fromFXImage(img, null), "png", target);
        } catch (IOException ex) {
            Alert a = new Alert(Alert.AlertType.ERROR, "Failed to write PNG: " + ex.getMessage());
            a.initOwner(stage);
            a.showAndWait();
            return;
        }

        Alert ok = new Alert(Alert.AlertType.INFORMATION, "Saved to:\n" + target.getAbsolutePath());
        ok.setHeaderText("Validation confusion matrix exported");
        ok.initOwner(stage);
        ok.showAndWait();
    }
}
