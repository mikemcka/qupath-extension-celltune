package qupath.ext.celltune.ui;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.DoubleFunction;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TextField;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import javafx.scene.text.TextAlignment;
import javafx.stage.FileChooser;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javax.imageio.ImageIO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.celltune.model.IntensityHeatmap;
import qupath.ext.celltune.model.NeighborhoodModel;
import qupath.fx.dialogs.Dialogs;

/**
 * Renders the cellular-neighborhood (CN) × cell-type enrichment heatmap — the
 * paper's Fig 4B. Rows are CNs (or merged CN classes), columns are cell types,
 * each cell is the mean composition fraction of that type within the CN, coloured
 * by per-column z-score (reusing {@link IntensityHeatmap#zScoreByColumn}) so red =
 * enriched, blue = depleted. A colour-key swatch beside each row matches the
 * viewer CN colouring.
 *
 * <p>A side panel lets the user name each CN (e.g. "Tumour", "Stroma"); naming two
 * CNs the same merges them. "Apply" records the assignment back on the cells (via
 * the supplied callback), and "Merge rows by class" collapses the heatmap to one
 * count-weighted row per class. Exports the heatmap to PNG and the per-row
 * frequency + mean-composition table to CSV.
 *
 * <p>Canvas/colourbar/snapshot presentation mirrors {@link IntensityHeatmapView}.
 */
public class NeighborhoodHeatmapView {

    private static final Logger logger = LoggerFactory.getLogger(NeighborhoodHeatmapView.class);

    private final Stage stage;
    private final Canvas canvas;
    private final CheckBox showValuesBox;
    private final CheckBox mergeBox;
    private final CheckBox diversitySwatchBox;
    private final Label summaryLabel;
    private final DoubleFunction<Color> diversityColorFn; // diversity [0,1] -> colour (nullable)

    private final String title;
    private final List<String> typeNames;
    private final double[][] meanComposition; // [nCN][nTypes]
    private final long[] cnCounts;
    private final List<Color> cnColors; // viewer CN colour per CN row (may be empty)
    private final long totalCells;

    // Naming/merge state
    private final List<TextField> nameFields = new ArrayList<>();
    private final Map<Integer, String> appliedNames = new LinkedHashMap<>(); // CN id (1-based) -> name
    private final Consumer<Map<Integer, String>> onApplyNames; // null => naming disabled
    private final BiFunction<Integer, Integer, Color> classColorFn; // (code, total) -> merged-row colour

    private static final int CELL_W = 78;
    private static final int CELL_H = 34;
    private static final int LEFT_MARGIN_MIN = 150;
    private static final int LEFT_MARGIN_MAX = 380;
    private static final int LABEL_PAD = 16;
    private static final int SWATCH = 14; // CN colour key swatch
    private static final int SWATCH_GAP = 6;
    private static final int TOP_MARGIN = 64;
    private static final int BOTTOM_MARGIN = 170;
    private static final int COLORBAR_W = 110;
    private static final Font TITLE_FONT =
            Font.font("SansSerif", Font.getDefault().getSize() + 3);
    private static final Font LABEL_FONT = Font.font("SansSerif", 12);
    private static final Font VALUE_FONT = Font.font("SansSerif", 10);

    /**
     * @param owner           parent stage
     * @param title           image title shown in the heatmap header
     * @param typeNames       ordered cell-type column labels
     * @param meanComposition {@code [nCN][nTypes]} mean composition fraction per CN
     * @param cnCounts        number of cells assigned to each CN (row order)
     * @param cnColors        viewer colour per CN row (the colour key); may be null/empty
     * @param initialNames    pre-filled CN id (1-based) → name (may be null/empty)
     * @param onApplyNames    called with CN id → name when the user applies names; null disables naming
     * @param classColorFn    (class code, total classes) → colour for merged rows; may be null
     */
    public NeighborhoodHeatmapView(
            Stage owner,
            String title,
            List<String> typeNames,
            double[][] meanComposition,
            long[] cnCounts,
            List<Color> cnColors,
            Map<Integer, String> initialNames,
            Consumer<Map<Integer, String>> onApplyNames,
            BiFunction<Integer, Integer, Color> classColorFn,
            DoubleFunction<Color> diversityColorFn) {
        this.title = title != null ? title : "Current Image";
        this.typeNames = List.copyOf(typeNames);
        this.meanComposition = meanComposition;
        this.cnCounts = cnCounts.clone();
        this.cnColors = cnColors == null ? List.of() : List.copyOf(cnColors);
        this.onApplyNames = onApplyNames;
        this.classColorFn = classColorFn;
        this.diversityColorFn = diversityColorFn;
        if (initialNames != null) {
            appliedNames.putAll(initialNames);
        }
        long total = 0;
        for (long c : cnCounts) {
            total += c;
        }
        this.totalCells = total;

        stage = new Stage();
        canvas = new Canvas(10, 10);
        summaryLabel = new Label("");
        summaryLabel.setPadding(new Insets(4));
        showValuesBox = new CheckBox("Show mean fractions");
        showValuesBox.setSelected(true);
        showValuesBox.setOnAction(e -> redraw());
        mergeBox = new CheckBox("Merge rows by class");
        mergeBox.setSelected(false);
        mergeBox.setOnAction(e -> redraw());
        diversitySwatchBox = new CheckBox("Swatches by diversity");
        diversitySwatchBox.setSelected(false);
        diversitySwatchBox.setDisable(diversityColorFn == null);
        diversitySwatchBox.setTooltip(new javafx.scene.control.Tooltip(
                "Colour the row key by cell-type diversity (Shannon) instead of the distinct CN colours."));
        diversitySwatchBox.setOnAction(e -> redraw());

        Button exportPngBtn = new Button("Export as PNG…");
        exportPngBtn.setOnAction(e -> exportAsPng());
        Button exportCsvBtn = new Button("Export CN frequencies CSV…");
        exportCsvBtn.setOnAction(e -> exportCsv());
        Button closeBtn = new Button("Close");
        closeBtn.setOnAction(e -> stage.close());

        HBox buttons = new HBox(
                8, showValuesBox, mergeBox, diversitySwatchBox, new Region(), exportPngBtn, exportCsvBtn, closeBtn);
        buttons.setAlignment(Pos.CENTER_LEFT);
        buttons.setPadding(new Insets(8));
        HBox.setHgrow(buttons.getChildren().get(3), Priority.ALWAYS);

        VBox bottom = new VBox(4, summaryLabel, buttons);
        ScrollPane scroll = new ScrollPane(canvas);
        scroll.setPannable(true);

        BorderPane root = new BorderPane();
        root.setCenter(scroll);
        root.setBottom(bottom);
        if (onApplyNames != null) {
            root.setRight(buildNamingPanel());
        }
        root.setPadding(new Insets(8));

        stage.initOwner(owner);
        stage.initModality(Modality.NONE);
        stage.setTitle("CellTune — Cellular Neighborhoods");

        redraw();

        var screen = javafx.stage.Screen.getPrimary().getVisualBounds();
        double sceneW = Math.min(screen.getWidth() * 0.9, canvas.getWidth() + (onApplyNames != null ? 300 : 40));
        double sceneH = Math.min(screen.getHeight() * 0.85, canvas.getHeight() + 110);
        stage.setScene(new Scene(root, Math.max(720, sceneW), Math.max(440, sceneH)));
        stage.setResizable(true);
    }

    public void show() {
        stage.show();
        stage.toFront();
    }

    /** Close (hide) the heatmap window. The instance stays reusable via {@link #show()}. */
    public void close() {
        stage.close();
    }

    // ── Toggle state (read/restored by the owner so it survives a reopen) ────────

    /** Whether the "Show mean fractions" values overlay is on. */
    public boolean isShowValues() {
        return showValuesBox.isSelected();
    }

    /** Whether "Merge rows by class" is on. */
    public boolean isMergeRows() {
        return mergeBox.isSelected();
    }

    /** Whether the row-key swatches are coloured by diversity. */
    public boolean isDiversitySwatch() {
        return diversitySwatchBox.isSelected();
    }

    /**
     * Restore the three view toggles at once (e.g. from a previous window before a
     * new run rebuilt it), redrawing once. The diversity swatch is only applied when
     * a diversity colour function is available.
     */
    public void setToggleState(boolean showValues, boolean mergeRows, boolean diversitySwatch) {
        showValuesBox.setSelected(showValues);
        mergeBox.setSelected(mergeRows);
        if (!diversitySwatchBox.isDisabled()) {
            diversitySwatchBox.setSelected(diversitySwatch);
        }
        redraw();
    }

    // ── Naming / merge panel ─────────────────────────────────────────────────

    private VBox buildNamingPanel() {
        GridPane grid = new GridPane();
        grid.setHgap(6);
        grid.setVgap(4);
        grid.setPadding(new Insets(4));
        for (int i = 0; i < meanComposition.length; i++) {
            Rectangle swatch = new Rectangle(SWATCH, SWATCH);
            swatch.setFill(i < cnColors.size() ? cnColors.get(i) : Color.gray(0.7));
            swatch.setStroke(Color.gray(0.5));
            Label idLabel = new Label(String.format(Locale.US, "CN %d (n=%,d)", i + 1, cnCounts[i]));
            TextField field = new TextField(appliedNames.getOrDefault(i + 1, ""));
            field.setPromptText("name…");
            field.setPrefColumnCount(10);
            nameFields.add(field);
            grid.add(swatch, 0, i);
            grid.add(idLabel, 1, i);
            grid.add(field, 2, i);
        }
        ScrollPane gridScroll = new ScrollPane(grid);
        gridScroll.setFitToWidth(true);
        gridScroll.setPrefWidth(260);
        VBox.setVgrow(gridScroll, Priority.ALWAYS);

        Button applyBtn = new Button("Apply names");
        applyBtn.setMaxWidth(Double.MAX_VALUE);
        applyBtn.setOnAction(e -> applyNames());

        Label hint = new Label("Give two CNs the same name to merge them.");
        hint.setWrapText(true);
        hint.setStyle("-fx-font-size: 11px; -fx-text-fill: #666;");

        VBox panel = new VBox(6, new Label("Name / merge neighborhoods:"), gridScroll, applyBtn, hint);
        panel.setPadding(new Insets(4, 0, 4, 8));
        panel.setPrefWidth(280);
        return panel;
    }

    private void applyNames() {
        Map<Integer, String> names = new LinkedHashMap<>();
        for (int i = 0; i < nameFields.size(); i++) {
            String nm = nameFields.get(i).getText() == null
                    ? ""
                    : nameFields.get(i).getText().trim();
            names.put(i + 1, nm);
        }
        appliedNames.clear();
        appliedNames.putAll(names);
        if (onApplyNames != null) {
            onApplyNames.accept(names);
        }
        redraw();
    }

    /** The effective name for CN {@code id} (1-based): the applied name, or "CN id" when blank. */
    private String effectiveName(int id) {
        String nm = appliedNames.get(id);
        return (nm == null || nm.isBlank()) ? "CN " + id : nm;
    }

    // ── Display model (per-CN or merged) ─────────────────────────────────────

    private record Display(
            List<String> labels, List<String> keys, double[][] comp, double[][] z, long[] counts, List<Color> colors) {}

    private Display displayData() {
        int nTypes = typeNames.size();
        boolean merge = mergeBox.isSelected();
        if (!merge) {
            List<String> labels = new ArrayList<>();
            List<String> keys = new ArrayList<>();
            for (int i = 0; i < meanComposition.length; i++) {
                String nm = effectiveName(i + 1);
                String base = nm.equals("CN " + (i + 1)) ? nm : nm + " (CN " + (i + 1) + ")";
                double pct = totalCells > 0 ? 100.0 * cnCounts[i] / totalCells : 0;
                labels.add(String.format(Locale.US, "%s  (n=%,d, %.1f%%)", base, cnCounts[i], pct));
                keys.add(nm);
            }
            return new Display(
                    labels,
                    keys,
                    meanComposition,
                    IntensityHeatmap.zScoreByColumn(meanComposition),
                    cnCounts,
                    cnColors);
        }

        // Merge: group CN rows by effective name, count-weighted mean composition.
        LinkedHashMap<String, List<Integer>> groups = new LinkedHashMap<>();
        for (int i = 0; i < meanComposition.length; i++) {
            groups.computeIfAbsent(effectiveName(i + 1), k -> new ArrayList<>()).add(i);
        }
        int m = groups.size();
        double[][] comp = new double[m][nTypes];
        long[] counts = new long[m];
        List<String> labels = new ArrayList<>(m);
        List<String> keys = new ArrayList<>(m);
        List<Color> colors = new ArrayList<>(m);
        int code = 1;
        for (var e : groups.entrySet()) {
            long tot = 0;
            double[] acc = new double[nTypes];
            for (int idx : e.getValue()) {
                long c = cnCounts[idx];
                tot += c;
                for (int j = 0; j < nTypes; j++) {
                    acc[j] += meanComposition[idx][j] * c;
                }
            }
            for (int j = 0; j < nTypes; j++) {
                comp[code - 1][j] = tot > 0 ? acc[j] / tot : 0;
            }
            counts[code - 1] = tot;
            double pct = totalCells > 0 ? 100.0 * tot / totalCells : 0;
            labels.add(String.format(Locale.US, "%s  (n=%,d, %.1f%%)", e.getKey(), tot, pct));
            keys.add(e.getKey());
            colors.add(classColorFn != null ? classColorFn.apply(code, m) : Color.gray(0.7));
            code++;
        }
        return new Display(labels, keys, comp, IntensityHeatmap.zScoreByColumn(comp), counts, colors);
    }

    // ── Drawing ──────────────────────────────────────────────────────────────

    private int computeLeftMargin(List<String> labels) {
        double maxW = 0;
        Text probe = new Text();
        probe.setFont(LABEL_FONT);
        for (String label : labels) {
            probe.setText(label);
            maxW = Math.max(maxW, probe.getLayoutBounds().getWidth());
        }
        int needed = (int) Math.ceil(maxW) + SWATCH + SWATCH_GAP + LABEL_PAD + 8;
        return Math.max(LEFT_MARGIN_MIN, Math.min(LEFT_MARGIN_MAX, needed));
    }

    private void redraw() {
        Display d = displayData();
        int nRows = d.labels().size();
        int nTypes = typeNames.size();

        if (nRows == 0 || nTypes == 0) {
            canvas.setWidth(640);
            canvas.setHeight(200);
            GraphicsContext gc = canvas.getGraphicsContext2D();
            gc.setFill(Color.WHITE);
            gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
            gc.setFill(Color.gray(0.3));
            gc.setTextAlign(TextAlignment.CENTER);
            gc.fillText("No cellular neighborhoods to display.", canvas.getWidth() / 2.0, canvas.getHeight() / 2.0);
            return;
        }

        int leftMargin = computeLeftMargin(d.labels());
        double width = leftMargin + nTypes * CELL_W + COLORBAR_W;
        double height = TOP_MARGIN + nRows * CELL_H + BOTTOM_MARGIN;
        canvas.setWidth(width);
        canvas.setHeight(height);

        GraphicsContext gc = canvas.getGraphicsContext2D();
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, width, height);

        double scale = 0.0;
        for (double[] row : d.z()) {
            for (double z : row) {
                if (!Double.isNaN(z)) {
                    scale = Math.max(scale, Math.abs(z));
                }
            }
        }
        if (scale < 1e-9) {
            scale = 1.0;
        }

        double gridLeft = leftMargin;
        double gridTop = TOP_MARGIN;

        gc.setFont(TITLE_FONT);
        gc.setFill(Color.BLACK);
        gc.setTextAlign(TextAlignment.CENTER);
        gc.fillText(
                "Cellular neighborhood enrichment  (colour = z-score across rows)",
                gridLeft + nTypes * CELL_W / 2.0,
                24);
        gc.setFont(LABEL_FONT);
        gc.fillText(title, gridLeft + nTypes * CELL_W / 2.0, 44);

        boolean showValues = showValuesBox.isSelected();
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nTypes; j++) {
                double x = gridLeft + j * CELL_W;
                double y = gridTop + i * CELL_H;
                double z = d.z()[i][j];
                double frac = d.comp()[i][j];

                gc.setFill(zToColor(z, scale));
                gc.fillRect(x, y, CELL_W, CELL_H);
                gc.setStroke(Color.gray(0.8));
                gc.setLineWidth(0.5);
                gc.strokeRect(x, y, CELL_W, CELL_H);

                if (showValues && !Double.isNaN(frac)) {
                    double t = Double.isNaN(z) ? 0 : Math.min(1.0, Math.abs(z) / scale);
                    gc.setFill(t > 0.6 ? Color.WHITE : Color.gray(0.15));
                    gc.setFont(VALUE_FONT);
                    gc.setTextAlign(TextAlignment.CENTER);
                    gc.fillText(String.format(Locale.US, "%.2f", frac), x + CELL_W / 2.0, y + CELL_H / 2.0 + 4);
                }
            }
        }

        // Row labels with a colour-key swatch matching the viewer colouring.
        gc.setFont(LABEL_FONT);
        gc.setTextAlign(TextAlignment.LEFT);
        boolean divSwatch = diversitySwatchBox.isSelected() && diversityColorFn != null;
        for (int i = 0; i < nRows; i++) {
            double cy = gridTop + i * CELL_H + CELL_H / 2.0 + 4;
            double sy = gridTop + i * CELL_H + (CELL_H - SWATCH) / 2.0;
            Color swatch = divSwatch
                    ? diversityColorFn.apply(NeighborhoodModel.compositionDiversity(d.comp()[i]))
                    : (i < d.colors().size() ? d.colors().get(i) : null);
            if (swatch != null) {
                gc.setFill(swatch);
                gc.fillRect(4, sy, SWATCH, SWATCH);
                gc.setStroke(Color.gray(0.5));
                gc.setLineWidth(0.5);
                gc.strokeRect(4, sy, SWATCH, SWATCH);
            }
            gc.setFill(Color.BLACK);
            gc.fillText(d.labels().get(i), 4 + SWATCH + SWATCH_GAP, cy);
        }

        double bottomY = gridTop + nRows * CELL_H + 8;
        gc.setFont(LABEL_FONT);
        for (int j = 0; j < nTypes; j++) {
            double cx = gridLeft + j * CELL_W + CELL_W / 2.0;
            gc.save();
            gc.translate(cx, bottomY);
            gc.rotate(-45);
            gc.setTextAlign(TextAlignment.RIGHT);
            gc.setFill(Color.BLACK);
            gc.fillText(typeNames.get(j), 0, 0);
            gc.restore();
        }

        drawColorbar(gc, gridLeft + nTypes * CELL_W + 24, gridTop, nRows * CELL_H, scale);
        summaryLabel.setText(String.format(
                Locale.US,
                "%s | %d %s × %d types | %,d cells",
                title,
                nRows,
                mergeBox.isSelected() ? "classes" : "CNs",
                nTypes,
                totalCells));
    }

    private void drawColorbar(GraphicsContext gc, double x, double y, double height, double scale) {
        double barW = 22;
        int steps = 128;
        double stepH = height / steps;
        for (int s = 0; s < steps; s++) {
            double z = scale - (2.0 * scale) * (s / (double) (steps - 1));
            gc.setFill(zToColor(z, scale));
            gc.fillRect(x, y + s * stepH, barW, stepH + 1);
        }
        gc.setStroke(Color.gray(0.5));
        gc.setLineWidth(0.5);
        gc.strokeRect(x, y, barW, height);
        gc.setFill(Color.BLACK);
        gc.setFont(VALUE_FONT);
        gc.setTextAlign(TextAlignment.LEFT);
        double labelX = x + barW + 5;
        gc.fillText(String.format(Locale.US, "+%.1f", scale), labelX, y + 9);
        gc.fillText("0", labelX, y + height / 2.0 + 3);
        gc.fillText(String.format(Locale.US, "-%.1f", scale), labelX, y + height);
    }

    /** Diverging colour map: blue (low) → white (0) → red (high). */
    private static Color zToColor(double z, double scale) {
        if (Double.isNaN(z)) {
            return Color.gray(0.9);
        }
        double t = Math.max(-1.0, Math.min(1.0, z / scale));
        if (t >= 0) {
            return Color.color(1.0, 1.0 - 0.85 * t, 1.0 - 0.85 * t);
        }
        double a = -t;
        return Color.color(1.0 - 0.85 * a, 1.0 - 0.85 * a, 1.0);
    }

    // ── Export ───────────────────────────────────────────────────────────────

    private void exportAsPng() {
        FileChooser fc = new FileChooser();
        fc.setTitle("Save CN Heatmap as PNG");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("PNG Image", "*.png"));
        fc.setInitialFileName("cn_enrichment_heatmap.png");
        File file = fc.showSaveDialog(stage);
        if (file == null) {
            return;
        }
        try {
            var params = new SnapshotParameters();
            params.setFill(Color.WHITE);
            var image = canvas.snapshot(params, null);
            ImageIO.write(SwingFXUtils.fromFXImage(image, null), "png", file);
            Dialogs.showInfoNotification("CellTune", "Exported heatmap: " + file.getName());
        } catch (IOException ex) {
            logger.error("Failed to export CN heatmap", ex);
            Dialogs.showErrorMessage("CellTune", "Failed to export PNG: " + ex.getMessage());
        }
    }

    private void exportCsv() {
        Display d = displayData();
        FileChooser fc = new FileChooser();
        fc.setTitle("Export CN Frequencies CSV");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV Files", "*.csv"));
        fc.setInitialFileName(mergeBox.isSelected() ? "cn_class_frequencies.csv" : "cn_frequencies.csv");
        File file = fc.showSaveDialog(stage);
        if (file == null) {
            return;
        }
        try (BufferedWriter w = Files.newBufferedWriter(file.toPath(), StandardCharsets.UTF_8)) {
            List<String> header = new ArrayList<>();
            header.add(mergeBox.isSelected() ? "Class" : "CN");
            header.add("Name");
            header.add("Count");
            header.add("Fraction");
            header.add("Diversity");
            for (String t : typeNames) {
                header.add(csvQuote("mean_" + t));
            }
            w.write(String.join(",", header));
            w.newLine();
            for (int i = 0; i < d.labels().size(); i++) {
                double frac = totalCells > 0 ? (double) d.counts()[i] / totalCells : 0;
                StringBuilder row = new StringBuilder();
                row.append(mergeBox.isSelected() ? (i + 1) : ("CN " + (i + 1)))
                        .append(',')
                        .append(csvQuote(d.keys().get(i)))
                        .append(',')
                        .append(d.counts()[i])
                        .append(',')
                        .append(frac)
                        .append(',')
                        .append(NeighborhoodModel.compositionDiversity(d.comp()[i]));
                for (int j = 0; j < typeNames.size(); j++) {
                    row.append(',').append(d.comp()[i][j]);
                }
                w.write(row.toString());
                w.newLine();
            }
            Dialogs.showInfoNotification("CellTune", "Exported CSV: " + file.getName());
        } catch (IOException ex) {
            logger.error("Failed to export CN frequencies CSV", ex);
            Dialogs.showErrorMessage("CellTune", "Failed to export CSV: " + ex.getMessage());
        }
    }

    private static String csvQuote(String value) {
        if (value == null || value.isEmpty()) {
            return "";
        }
        if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
            return "\"" + value.replace("\"", "\"\"") + "\"";
        }
        return value;
    }
}
