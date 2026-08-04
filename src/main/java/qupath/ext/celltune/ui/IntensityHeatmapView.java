package qupath.ext.celltune.ui;

import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
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
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjectFilter;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;

/**
 * JavaFX window that renders a mean marker-intensity heatmap grouped by
 * predicted cell class.
 * <p>
 * Rows are cell classes, columns are markers (whole-cell mean intensities), and
 * the cell colour encodes the per-column z-score across classes — high (red) /
 * low (blue) relative to the other phenotypes for that marker. An image selector
 * lets the user switch between the current image, any individual project image,
 * or a project-wide pooled heatmap. The heatmap can be exported to PNG and the
 * underlying mean-intensity table to CSV.
 * <p>
 * Mirrors {@link ConfusionMatrixView}: canvas-based rendering, background
 * loading of other images via {@code readImageData()}, and a snapshot-based PNG
 * export. The open image is the one exception to that disk read — it is always
 * pooled from its live hierarchy so the combined view cannot show classes the
 * viewer has already replaced; see
 * {@link IntensityHeatmap#selectImageSource(String, String, Object, java.util.function.Supplier)}.
 */
public class IntensityHeatmapView {

    private static final Logger logger = LoggerFactory.getLogger(IntensityHeatmapView.class);
    private static final String ALL_IMAGES_LABEL = "All Images (Project Combined)";

    private final Stage stage;
    private final Canvas canvas;
    private final Label summaryLabel;
    private final CheckBox showValuesBox;

    private final QuPathGUI qupath;
    private final String currentImageName;
    private final List<String> markerFeatures;
    private final Map<String, IntensityHeatmap.Result> cache = new LinkedHashMap<>();

    private IntensityHeatmap.Result current;
    private String currentTitle;

    // Layout constants
    private static final int CELL_W = 68;
    private static final int CELL_H = 34;
    private static final int LEFT_MARGIN_MIN = 160; // minimum room for class names
    private static final int LEFT_MARGIN_MAX = 460; // cap so wide labels don't dominate
    private static final int LABEL_PAD = 16; // gap between label and grid
    private static final int TOP_MARGIN = 70; // title + spacing
    private static final int BOTTOM_MARGIN = 150; // rotated marker labels
    private static final int COLORBAR_W = 90; // colour scale on the right
    private static final Font TITLE_FONT =
            Font.font("SansSerif", Font.getDefault().getSize() + 3);
    private static final Font LABEL_FONT = Font.font("SansSerif", 12);
    private static final Font VALUE_FONT = Font.font("SansSerif", 10);

    /**
     * @param owner            parent stage
     * @param qupath           QuPath GUI (for loading other images; may be null)
     * @param currentImageName display name of the currently open image
     * @param markerFeatures   ordered full marker-mean measurement names
     * @param current          heatmap for the current image
     */
    public IntensityHeatmapView(
            Stage owner,
            QuPathGUI qupath,
            String currentImageName,
            List<String> markerFeatures,
            IntensityHeatmap.Result current) {
        this.qupath = qupath;
        this.currentImageName = currentImageName;
        this.markerFeatures = List.copyOf(markerFeatures);
        this.current = current;
        this.currentTitle = currentImageName != null ? currentImageName : "Current Image";
        if (currentImageName != null && current != null) {
            cache.put(currentImageName, current);
        }

        stage = new Stage();

        canvas = new Canvas(10, 10);

        summaryLabel = new Label("");
        summaryLabel.setPadding(new Insets(4));

        showValuesBox = new CheckBox("Show mean values");
        showValuesBox.setSelected(true);
        showValuesBox.setOnAction(e -> redraw());

        Button exportPngBtn = new Button("Export as PNG\u2026");
        exportPngBtn.setOnAction(e -> exportAsPng());

        Button exportCsvBtn = new Button("Export CSV\u2026");
        exportCsvBtn.setOnAction(e -> exportCsv());

        Button closeBtn = new Button("Close");
        closeBtn.setOnAction(e -> stage.close());

        HBox buttons =
                new HBox(8, showValuesBox, new javafx.scene.layout.Region(), exportPngBtn, exportCsvBtn, closeBtn);
        buttons.setAlignment(Pos.CENTER_LEFT);
        buttons.setPadding(new Insets(8));
        HBox.setHgrow(buttons.getChildren().get(1), Priority.ALWAYS);

        VBox bottom = new VBox(4, summaryLabel, buttons);

        ScrollPane scroll = new ScrollPane(canvas);
        scroll.setPannable(true);
        scroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.AS_NEEDED);
        scroll.setVbarPolicy(ScrollPane.ScrollBarPolicy.AS_NEEDED);

        BorderPane root = new BorderPane();
        root.setCenter(scroll);
        root.setBottom(bottom);
        if (qupath != null) {
            root.setTop(buildImageSelector());
        }
        root.setPadding(new Insets(8));

        stage.initOwner(owner);
        stage.initModality(Modality.NONE);
        stage.setTitle("CellTune \u2014 Intensity Heatmap");

        redraw();

        var screen = javafx.stage.Screen.getPrimary().getVisualBounds();
        double sceneW = Math.min(screen.getWidth() * 0.85, canvas.getWidth() + 40);
        double sceneH = Math.min(screen.getHeight() * 0.85, canvas.getHeight() + 110);
        stage.setScene(new Scene(root, Math.max(640, sceneW), Math.max(420, sceneH)));
        stage.setResizable(true);
    }

    /** Show the heatmap window. */
    public void show() {
        stage.show();
        stage.toFront();
    }

    // ── Image selector ───────────────────────────────────────────────────────

    private HBox buildImageSelector() {
        ComboBox<String> combo = new ComboBox<>();
        combo.setMaxWidth(Double.MAX_VALUE);
        combo.getItems().add(ALL_IMAGES_LABEL);
        if (currentImageName != null) {
            combo.getItems().add(currentImageName);
        }

        Project<?> project = qupath.getProject();
        if (project != null) {
            for (var entry : project.getImageList()) {
                String name = entry == null ? null : entry.getImageName();
                if (name != null && !name.isBlank() && !combo.getItems().contains(name)) {
                    combo.getItems().add(name);
                }
            }
        }
        combo.setValue(currentImageName != null ? currentImageName : ALL_IMAGES_LABEL);
        combo.setOnAction(e -> {
            String selected = combo.getValue();
            if (selected != null) {
                loadAndDisplaySelection(selected);
            }
        });

        Label label = new Label("Image:");
        label.setPadding(new Insets(0, 4, 0, 4));
        HBox bar = new HBox(6, label, combo);
        bar.setPadding(new Insets(6, 8, 2, 8));
        HBox.setHgrow(combo, Priority.ALWAYS);
        return bar;
    }

    private void loadAndDisplaySelection(String selected) {
        if (cache.containsKey(selected)) {
            current = cache.get(selected);
            currentTitle = selected;
            redraw();
            return;
        }

        var project = qupath.getProject();
        if (project == null) {
            return;
        }

        // Snapshot the live open image here, on the FX thread (this runs from the
        // ComboBox action), so the background pooling below can read it instead of
        // its .qpdata. Classifications applied to the open image live in memory
        // until QuPath saves it, so reading it from disk would pool whatever the
        // previous classifier wrote; see IntensityHeatmap#selectImageSource.
        final ImageData<BufferedImage> openData = qupath.getImageData();
        final String openName = openImageName(project, openData);

        summaryLabel.setText("Loading \u201c" + selected + "\u201d\u2026");
        new Thread(
                        () -> {
                            try {
                                IntensityHeatmap.Result result = ALL_IMAGES_LABEL.equals(selected)
                                        ? computeForAllImages(project, openData, openName)
                                        : computeForImage(project, selected, openData, openName);
                                if (result == null) {
                                    Platform.runLater(() ->
                                            summaryLabel.setText("No cells found for \u201c" + selected + "\u201d."));
                                    return;
                                }
                                Platform.runLater(() -> {
                                    cache.put(selected, result);
                                    current = result;
                                    currentTitle = selected;
                                    redraw();
                                });
                            } catch (Exception ex) {
                                logger.warn("Failed to compute intensity heatmap for '{}'", selected, ex);
                                Platform.runLater(() -> summaryLabel.setText(
                                        "Failed to load \u201c" + selected + "\u201d: " + ex.getMessage()));
                            }
                        },
                        "CellTune-IntensityHeatmap-Loader")
                .start();
    }

    private IntensityHeatmap.Result computeForImage(
            Project<BufferedImage> project, String imageName, ImageData<BufferedImage> openData, String openName)
            throws IOException {
        ProjectImageEntry<BufferedImage> entry = findEntry(project, imageName);
        if (entry == null) {
            return null;
        }
        var data = imageDataFor(entry, imageName, openData, openName);
        Collection<PathObject> cells = detections(data.getHierarchy());
        if (cells.isEmpty()) {
            return null;
        }
        var acc = new IntensityHeatmap.Accumulator(markerFeatures);
        acc.add(cells);
        return acc.build();
    }

    private IntensityHeatmap.Result computeForAllImages(
            Project<BufferedImage> project, ImageData<BufferedImage> openData, String openName) {
        var acc = new IntensityHeatmap.Accumulator(markerFeatures);
        boolean any = false;
        for (var entry : project.getImageList()) {
            if (entry == null) {
                continue;
            }
            try {
                var data = imageDataFor(entry, entry.getImageName(), openData, openName);
                Collection<PathObject> cells = detections(data.getHierarchy());
                if (!cells.isEmpty()) {
                    acc.add(cells);
                    any = true;
                }
            } catch (Exception ex) {
                logger.warn("Skipping '{}' in project heatmap: {}", entry.getImageName(), ex.getMessage());
            }
        }
        return any ? acc.build() : null;
    }

    /**
     * Resolve one image's data via {@link IntensityHeatmap#selectImageSource}: the live
     * open {@link ImageData} for the open image, its {@code .qpdata} for every other.
     * The checked {@link IOException} from {@code readImageData} is tunnelled through
     * the {@link java.util.function.Supplier} as an {@link UncheckedIOException} and
     * unwrapped here, so callers keep the original checked signature.
     */
    private static ImageData<BufferedImage> imageDataFor(
            ProjectImageEntry<BufferedImage> entry, String name, ImageData<BufferedImage> openData, String openName)
            throws IOException {
        try {
            return IntensityHeatmap.selectImageSource(name, openName, openData, () -> {
                try {
                    return entry.readImageData();
                } catch (IOException ex) {
                    throw new UncheckedIOException(ex);
                }
            });
        } catch (UncheckedIOException ex) {
            throw ex.getCause();
        }
    }

    /**
     * Display name of the open image, or null when nothing is open. Falls back to the
     * server metadata name for an image that is not a project entry, matching how
     * {@code AnalysisViews} names the current image in the selector.
     */
    private static String openImageName(Project<BufferedImage> project, ImageData<BufferedImage> openData) {
        if (openData == null) {
            return null;
        }
        var entry = project.getEntry(openData);
        if (entry != null
                && entry.getImageName() != null
                && !entry.getImageName().isBlank()) {
            return entry.getImageName();
        }
        return openData.getServer().getMetadata().getName();
    }

    private static ProjectImageEntry<BufferedImage> findEntry(Project<BufferedImage> project, String imageName) {
        for (var entry : project.getImageList()) {
            if (entry != null && imageName.equals(entry.getImageName())) {
                return entry;
            }
        }
        return null;
    }

    private static Collection<PathObject> detections(qupath.lib.objects.hierarchy.PathObjectHierarchy hierarchy) {
        return hierarchy.getObjects(null, PathObject.class).stream()
                .filter(PathObjectFilter.DETECTIONS_ALL)
                .toList();
    }

    // ── Drawing ──────────────────────────────────────────────────────────────

    /** Build the left-hand row label for a class row. */
    private static String rowLabel(IntensityHeatmap.Result r, int i) {
        return r.classNames().get(i) + "  (n=" + r.classCounts()[i] + ")";
    }

    /**
     * Compute a left margin wide enough to show the longest class label without
     * clipping, clamped to a sensible range.
     */
    private static int computeLeftMargin(IntensityHeatmap.Result r) {
        double maxW = 0;
        Text probe = new Text();
        probe.setFont(LABEL_FONT);
        for (int i = 0; i < r.classNames().size(); i++) {
            probe.setText(rowLabel(r, i));
            maxW = Math.max(maxW, probe.getLayoutBounds().getWidth());
        }
        int needed = (int) Math.ceil(maxW) + LABEL_PAD + 4;
        return Math.max(LEFT_MARGIN_MIN, Math.min(LEFT_MARGIN_MAX, needed));
    }

    private void redraw() {
        IntensityHeatmap.Result r = current;
        int nClasses = (r == null) ? 0 : r.classNames().size();
        int nMarkers = (r == null) ? 0 : r.markers().size();

        if (r == null || nClasses == 0 || nMarkers == 0) {
            canvas.setWidth(640);
            canvas.setHeight(200);
            GraphicsContext gc = canvas.getGraphicsContext2D();
            gc.setFill(Color.WHITE);
            gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
            gc.setFill(Color.gray(0.3));
            gc.setFont(LABEL_FONT);
            gc.setTextAlign(TextAlignment.CENTER);
            gc.fillText(
                    "No classified cells with marker measurements to display.",
                    canvas.getWidth() / 2.0,
                    canvas.getHeight() / 2.0);
            summaryLabel.setText("");
            return;
        }

        // Size the left margin to fit the longest "class (n=…)" label.
        int leftMargin = computeLeftMargin(r);

        double width = leftMargin + nMarkers * CELL_W + COLORBAR_W;
        double height = TOP_MARGIN + nClasses * CELL_H + BOTTOM_MARGIN;
        canvas.setWidth(width);
        canvas.setHeight(height);

        GraphicsContext gc = canvas.getGraphicsContext2D();
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, width, height);

        // Symmetric z-score colour scale.
        double scale = 0.0;
        for (double[] row : r.zScores()) {
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

        // Title
        gc.setFont(TITLE_FONT);
        gc.setFill(Color.BLACK);
        gc.setTextAlign(TextAlignment.CENTER);
        gc.fillText(
                "Mean marker expression per phenotype  (colour = z-score across phenotypes)",
                gridLeft + nMarkers * CELL_W / 2.0,
                26);
        gc.setFont(Font.font("SansSerif", 12));
        gc.fillText(currentTitle, gridLeft + nMarkers * CELL_W / 2.0, 46);

        // Cells
        boolean showValues = showValuesBox.isSelected();
        for (int i = 0; i < nClasses; i++) {
            for (int j = 0; j < nMarkers; j++) {
                double x = gridLeft + j * CELL_W;
                double y = gridTop + i * CELL_H;
                double z = r.zScores()[i][j];
                double meanVal = r.meanIntensity()[i][j];

                gc.setFill(zToColor(z, scale));
                gc.fillRect(x, y, CELL_W, CELL_H);
                gc.setStroke(Color.gray(0.8));
                gc.setLineWidth(0.5);
                gc.strokeRect(x, y, CELL_W, CELL_H);

                if (showValues && !Double.isNaN(meanVal)) {
                    double t = Double.isNaN(z) ? 0 : Math.min(1.0, Math.abs(z) / scale);
                    gc.setFill(t > 0.6 ? Color.WHITE : Color.gray(0.15));
                    gc.setFont(VALUE_FONT);
                    gc.setTextAlign(TextAlignment.CENTER);
                    gc.fillText(formatValue(meanVal), x + CELL_W / 2.0, y + CELL_H / 2.0 + 4);
                }
            }
        }

        // Row labels (class name + count)
        gc.setFont(LABEL_FONT);
        gc.setFill(Color.BLACK);
        gc.setTextAlign(TextAlignment.RIGHT);
        for (int i = 0; i < nClasses; i++) {
            double cy = gridTop + i * CELL_H + CELL_H / 2.0 + 4;
            String label = rowLabel(r, i);
            gc.fillText(label, gridLeft - LABEL_PAD, cy);
        }

        // Column labels (markers, rotated at bottom)
        double bottomY = gridTop + nClasses * CELL_H + 8;
        gc.setFont(LABEL_FONT);
        for (int j = 0; j < nMarkers; j++) {
            double cx = gridLeft + j * CELL_W + CELL_W / 2.0;
            gc.save();
            gc.translate(cx, bottomY);
            gc.rotate(-45);
            gc.setTextAlign(TextAlignment.RIGHT);
            gc.setFill(Color.BLACK);
            gc.fillText(r.markers().get(j), 0, 0);
            gc.restore();
        }

        drawColorbar(gc, gridLeft + nMarkers * CELL_W + 24, gridTop, nClasses * CELL_H, scale);

        updateSummary(r);
    }

    private void drawColorbar(GraphicsContext gc, double x, double y, double height, double scale) {
        double barW = 22;
        int steps = 128;
        double stepH = height / steps;
        for (int s = 0; s < steps; s++) {
            // Top = +scale (red), bottom = -scale (blue).
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
        gc.fillText(String.format("+%.1f", scale), labelX, y + 9);
        gc.fillText("0", labelX, y + height / 2.0 + 3);
        gc.fillText(String.format("-%.1f", scale), labelX, y + height);

        gc.save();
        gc.translate(labelX + 34, y + height / 2.0);
        gc.rotate(-90);
        gc.setTextAlign(TextAlignment.CENTER);
        gc.fillText("z-score across phenotypes", 0, 0);
        gc.restore();
    }

    /** Diverging colour map: blue (low) → white (0) → red (high). */
    private static Color zToColor(double z, double scale) {
        if (Double.isNaN(z)) {
            return Color.gray(0.9);
        }
        double t = Math.max(-1.0, Math.min(1.0, z / scale));
        if (t >= 0) {
            // white → red
            return Color.color(1.0, 1.0 - 0.85 * t, 1.0 - 0.85 * t);
        }
        // white → blue
        double a = -t;
        return Color.color(1.0 - 0.85 * a, 1.0 - 0.85 * a, 1.0);
    }

    private void updateSummary(IntensityHeatmap.Result r) {
        long totalCells = 0;
        for (long c : r.classCounts()) {
            totalCells += c;
        }
        summaryLabel.setText(String.format(
                "%s | %d classes \u00d7 %d markers | %,d cells",
                currentTitle, r.classNames().size(), r.markers().size(), totalCells));
    }

    private static String formatValue(double v) {
        double abs = Math.abs(v);
        if (abs != 0 && (abs < 0.01 || abs >= 100000)) {
            return String.format("%.1e", v);
        }
        if (abs >= 100) {
            return String.format("%.0f", v);
        }
        if (abs >= 10) {
            return String.format("%.1f", v);
        }
        return String.format("%.2f", v);
    }

    // ── Export ───────────────────────────────────────────────────────────────

    private void exportAsPng() {
        FileChooser fc = new FileChooser();
        fc.setTitle("Save Intensity Heatmap as PNG");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("PNG Image", "*.png"));
        fc.setInitialFileName("intensity_heatmap.png");
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
            logger.info("Exported intensity heatmap to {}", file);
        } catch (IOException ex) {
            logger.error("Failed to export intensity heatmap", ex);
            Dialogs.showErrorMessage("CellTune", "Failed to export PNG: " + ex.getMessage());
        }
    }

    private void exportCsv() {
        IntensityHeatmap.Result r = current;
        if (r == null || r.classNames().isEmpty()) {
            Dialogs.showInfoNotification("CellTune", "No heatmap data to export.");
            return;
        }
        FileChooser fc = new FileChooser();
        fc.setTitle("Export Intensity Heatmap CSV");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV Files", "*.csv"));
        fc.setInitialFileName("intensity_heatmap.csv");
        File file = fc.showSaveDialog(stage);
        if (file == null) {
            return;
        }
        try (BufferedWriter w = Files.newBufferedWriter(file.toPath(), StandardCharsets.UTF_8)) {
            List<String> header = new ArrayList<>();
            header.add("Class");
            header.add("CellCount");
            for (String m : r.markers()) {
                header.add(csvQuote(m));
            }
            w.write(String.join(",", header));
            w.newLine();

            for (int i = 0; i < r.classNames().size(); i++) {
                StringBuilder row = new StringBuilder();
                row.append(csvQuote(r.classNames().get(i))).append(',').append(r.classCounts()[i]);
                for (int j = 0; j < r.markers().size(); j++) {
                    double v = r.meanIntensity()[i][j];
                    row.append(',').append(Double.isNaN(v) ? "NA" : v);
                }
                w.write(row.toString());
                w.newLine();
            }
            Dialogs.showInfoNotification("CellTune", "Exported CSV: " + file.getName());
        } catch (IOException ex) {
            logger.error("Failed to export intensity heatmap CSV", ex);
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
