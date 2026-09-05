package qupath.ext.spclassify.ui;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;
import javafx.stage.FileChooser;
import javafx.stage.Modality;
import javafx.stage.Stage;
import qupath.ext.spclassify.classifier.TrainingMetrics;

/**
 * Dialog showing per-class precision/recall/F1 (sklearn-style classification
 * report) for both models.
 *
 * <p>By default only the held-out <em>Validation (20%)</em> blocks are shown,
 * since those are the only honest generalisation-quality numbers. The 80%
 * train blocks remain available behind a checkbox for over-fit diagnosis
 * (large gap between train F1 and validation F1 = over-fitting).
 *
 * <p>A "Download CSV\u2026" button exports a tidy long-format CSV with one row per
 * (split, model, class) so the metrics can be loaded into Excel / pandas / R.
 */
public class TrainingMetricsView {

    private final Stage stage = new Stage();
    private final TextArea text = new TextArea();

    private final TrainingMetrics m1Train;
    private final TrainingMetrics m1Val;
    private final TrainingMetrics m2Train;
    private final TrainingMetrics m2Val;

    public TrainingMetricsView(
            Stage owner,
            TrainingMetrics m1Train,
            TrainingMetrics m1Val,
            TrainingMetrics m2Train,
            TrainingMetrics m2Val) {
        this.m1Train = m1Train;
        this.m1Val = m1Val;
        this.m2Train = m2Train;
        this.m2Val = m2Val;

        text.setEditable(false);
        text.setWrapText(false);
        text.setFont(Font.font("Monospaced", 12));
        text.setPrefRowCount(28);
        text.setPrefColumnCount(80);

        Label header = new Label("Training Metrics");
        header.setFont(Font.font("SansSerif", 14));

        // Default: validation-only (the honest generalisation numbers).
        CheckBox showTrain = new CheckBox("Show 80% training-set rows (for over-fit diagnosis)");
        showTrain.setSelected(false);
        showTrain.selectedProperty().addListener((obs, was, is) -> refresh(is));

        Button downloadBtn = new Button("Download CSV\u2026");
        downloadBtn.setOnAction(e -> exportCsv(owner));

        // Validation confusion matrix (XGBoost) \u2014 the held-out, honest view.
        // Shown side-by-side as absolute + row-normalised heatmaps.
        Button cmBtn = new Button("Validation Confusion Matrix (XGBoost)\u2026");
        cmBtn.setDisable(m1Val == null);
        cmBtn.setOnAction(e -> new ValidationConfusionMatrixView(owner, "Model 1 (XGBoost)", m1Val).show());

        Button close = new Button("Close");
        close.setOnAction(e -> stage.close());

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);
        HBox buttons = new HBox(8, showTrain, spacer, cmBtn, downloadBtn, close);
        buttons.setAlignment(Pos.CENTER_LEFT);
        buttons.setPadding(new Insets(8, 0, 0, 0));

        VBox root = new VBox(6, header, text, buttons);
        root.setPadding(new Insets(10));

        stage.initOwner(owner);
        stage.initModality(Modality.NONE);
        stage.setTitle("SP Classify \u2014 Training Metrics");
        stage.setScene(new Scene(root));
        stage.setResizable(true);

        refresh(showTrain.isSelected());
    }

    public void show() {
        stage.show();
        stage.toFront();
    }

    /** Rebuild the displayed text based on the train-rows checkbox state. */
    private void refresh(boolean includeTrain) {
        StringBuilder sb = new StringBuilder();
        int trainCount = totalSupport(m1Train);
        int valCount = totalSupport(m1Val);
        int total = trainCount + valCount;
        if (total > 0) {
            sb.append(String.format(
                    "Training data used: %d labelled cells (80%% train = %d, 20%% val = %d).%n",
                    total, trainCount, valCount));
            sb.append("If this disagrees with the sidebar label count, some labels were\n");
            sb.append("dropped \u2014 see the training log for the WARN line.\n\n");
        }
        sb.append("Held-out validation metrics from an 80/20 stratified split on labelled\n");
        sb.append("data. These are the honest, generalisation-quality F1 numbers.\n");
        if (includeTrain) {
            sb.append("\n80% training-set rows are also shown \u2014 a big gap between train F1\n");
            sb.append("and validation F1 indicates over-fitting.\n");
        }
        sb.append('\n');

        if (includeTrain) appendBlock(sb, m1Train);
        appendBlock(sb, m1Val);
        if (includeTrain) appendBlock(sb, m2Train);
        appendBlock(sb, m2Val);

        text.setText(sb.toString());
    }

    private static void appendBlock(StringBuilder sb, TrainingMetrics m) {
        if (m == null) return;
        sb.append(m.toFormattedReport());
        sb.append('\n');
    }

    /** Sum of per-class support — equals the total number of samples scored. */
    private static int totalSupport(TrainingMetrics m) {
        if (m == null || m.support() == null) return 0;
        int sum = 0;
        for (int s : m.support()) sum += s;
        return sum;
    }

    // \u2500\u2500 CSV export
    // \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    /**
     * Export all available metric blocks as a tidy long-format CSV:
     * {@code split,model,class,precision,recall,f1,support}
     * plus per-block summary rows (accuracy / macro F1 / weighted F1) whose
     * {@code class} cells use the sentinel names {@code __accuracy__},
     * {@code __macro_f1__}, {@code __weighted_f1__} so they're easy to filter
     * out in pandas/Excel.
     *
     * <p>Always exports every block we have (Train + Val for both models) so
     * the downloaded file is a complete record regardless of the on-screen
     * filter.
     */
    private void exportCsv(Stage owner) {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Export training metrics as CSV");
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV files", "*.csv"));
        chooser.setInitialFileName("sp-classify_training_metrics.csv");
        var target = chooser.showSaveDialog(owner);
        if (target == null) return;

        Path path = target.toPath();
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(path))) {
            w.println("split,model,class,precision,recall,f1,support");
            writeBlock(w, "Train", "Model 1 (XGBoost)", m1Train);
            writeBlock(w, "Val", "Model 1 (XGBoost)", m1Val);
            writeBlock(w, "Train", "Model 2 (LightGBM)", m2Train);
            writeBlock(w, "Val", "Model 2 (LightGBM)", m2Val);
        } catch (IOException ex) {
            Alert a = new Alert(Alert.AlertType.ERROR, "Failed to write CSV: " + ex.getMessage());
            a.initOwner(stage);
            a.showAndWait();
            return;
        }

        Alert ok = new Alert(Alert.AlertType.INFORMATION, "Saved to:\n" + path);
        ok.setHeaderText("Training metrics exported");
        ok.initOwner(stage);
        ok.showAndWait();
    }

    /** Append per-class rows and summary rows for one TrainingMetrics block. */
    private static void writeBlock(PrintWriter w, String split, String model, TrainingMetrics m) {
        if (m == null) return;
        var classes = m.classNames();
        double[] p = m.precision();
        double[] r = m.recall();
        double[] f1 = m.f1();
        int[] support = m.support();
        for (int i = 0; i < classes.size(); i++) {
            w.printf(
                    "%s,%s,%s,%.6f,%.6f,%.6f,%d%n",
                    csv(split), csv(model), csv(classes.get(i)), p[i], r[i], f1[i], support[i]);
        }
        // Summary rows \u2014 sentinel class names make them easy to filter out.
        w.printf("%s,%s,__accuracy__,,,%.6f,%d%n", csv(split), csv(model), m.accuracy(), m.total());
        w.printf("%s,%s,__macro_f1__,,,%.6f,%d%n", csv(split), csv(model), m.macroF1(), m.total());
        w.printf("%s,%s,__weighted_f1__,,,%.6f,%d%n", csv(split), csv(model), m.weightedF1(), m.total());
    }

    /** Minimal CSV escaping: quote and double any embedded quotes when needed. */
    private static String csv(String s) {
        if (s == null) return "";
        if (s.indexOf(',') < 0 && s.indexOf('"') < 0 && s.indexOf('\n') < 0) return s;
        return "\"" + s.replace("\"", "\"\"") + "\"";
    }
}
