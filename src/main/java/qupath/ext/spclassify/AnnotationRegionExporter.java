package qupath.ext.spclassify;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.Shape;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.ResourceBundle;
import java.util.stream.Collectors;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Dialog;
import javafx.scene.control.Hyperlink;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;
import javafx.stage.DirectoryChooser;
import javafx.stage.Modality;
import javafx.stage.Stage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.tools.GuiTools;
import qupath.lib.images.servers.AbstractTileableImageServer;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.ImageServerBuilder;
import qupath.lib.images.servers.ImageServerMetadata;
import qupath.lib.images.servers.TileRequest;
import qupath.lib.objects.PathObject;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.interfaces.ROI;

/**
 * Export annotation ROIs from the current image as polygon-masked, pyramidal
 * OME-TIFF(s). This is the heaviest of the Utility-Scripts-menu helpers — it pulls
 * in the Bio-Formats {@code OMEPyramidWriter} (reflectively, since it is only on
 * QuPath's runtime classpath) and a tileable {@link RoiMaskedServer} that streams
 * the source one tile at a time and zeroes pixels outside the ROI polygon. Extracted
 * verbatim from SpClassifyExtension to keep that entry-point class focused; it holds no
 * extension state and operates purely through the supplied {@link QuPathGUI}.
 */
final class AnnotationRegionExporter {

    private static final ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.spclassify.ui.strings");
    private static final String EXTENSION_NAME = resources.getString("name");
    private static final Logger logger = LoggerFactory.getLogger(AnnotationRegionExporter.class);

    /** Linked from the export-regions dialog for headless/HPC large-region exports. */
    private static final String LARGE_EXPORT_REPO =
            "https://github.com/BioimageAnalysisCoreWEHI/export_large_annotation_regions";

    private AnnotationRegionExporter() {}

    /**
     * Export one or more annotation ROIs from the current image as polygon-masked
     * OME-TIFF(s). Intended for single-image, small-to-medium exports — for batch
     * or very large regions use the headless pipeline linked from the dialog.
     */
    static void exportAnnotationRegions(QuPathGUI qupath) {
        var imageData = qupath.getImageData();
        if (imageData == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No image is open. Open the image first.");
            return;
        }
        var server = imageData.getServer();
        var hierarchy = imageData.getHierarchy();
        List<PathObject> allAnnotations = new ArrayList<>(hierarchy.getAnnotationObjects());
        if (allAnnotations.isEmpty()) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No annotations on the current image.");
            return;
        }

        int cores = Runtime.getRuntime().availableProcessors();
        TextField namesField = new TextField();
        namesField.setPromptText("blank = all annotations");
        namesField.setPrefColumnCount(24);
        TextField downsampleField = new TextField("1.0");
        TextField tileSizeField = new TextField("512");
        TextField threadsField = new TextField(Integer.toString(Math.max(1, Math.min(32, cores))));
        ComboBox<String> compressionBox = new ComboBox<>();
        compressionBox.getItems().addAll("LZW", "UNCOMPRESSED", "ZLIB", "JPEG", "J2K", "J2K_LOSSY", "DEFAULT");
        compressionBox.setValue("LZW");
        CheckBox bigTiffCb = new CheckBox("BigTIFF (recommended for large files)");
        bigTiffCb.setSelected(true);
        CheckBox pyramidCb = new CheckBox("Build pyramid");
        pyramidCb.setSelected(true);

        Label warning = new Label("Exports the chosen annotation region(s) from the current image as polygon-masked "
                + "OME-TIFF(s). Pixels are streamed tile-by-tile, but very large or batch exports "
                + "are better run headless on HPC with the dedicated pipeline:");
        warning.setWrapText(true);
        warning.setMaxWidth(460);
        Hyperlink link = new Hyperlink("github.com/BioimageAnalysisCoreWEHI/export_large_annotation_regions");
        link.setOnAction(ev -> {
            try {
                GuiTools.browseURI(new URI(LARGE_EXPORT_REPO));
            } catch (Exception ex) {
                logger.warn("[SP Classify] Could not open browser for {}: {}", LARGE_EXPORT_REPO, ex.getMessage());
            }
        });

        GridPane grid = new GridPane();
        grid.setHgap(8);
        grid.setVgap(8);
        int row = 0;
        grid.add(new Label("Annotation names (comma-separated)"), 0, row);
        grid.add(namesField, 1, row++);
        grid.add(new Label("Downsample"), 0, row);
        grid.add(downsampleField, 1, row++);
        grid.add(new Label("Tile size (px)"), 0, row);
        grid.add(tileSizeField, 1, row++);
        grid.add(new Label("Writer threads"), 0, row);
        grid.add(threadsField, 1, row++);
        grid.add(new Label("Compression"), 0, row);
        grid.add(compressionBox, 1, row++);
        grid.add(bigTiffCb, 1, row++);
        grid.add(pyramidCb, 1, row++);

        VBox content = new VBox(10, warning, link, new javafx.scene.control.Separator(), grid);
        content.setPadding(new Insets(10));

        Dialog<ButtonType> dialog = new Dialog<>();
        dialog.setTitle(EXTENSION_NAME);
        dialog.setHeaderText("Export annotation regions as OME-TIFF");
        dialog.getDialogPane().setContent(content);
        dialog.getDialogPane().getButtonTypes().addAll(ButtonType.OK, ButtonType.CANCEL);
        var choice = dialog.showAndWait();
        if (choice.isEmpty() || choice.get() != ButtonType.OK) return;

        double downsample;
        int tileSize;
        int nThreads;
        try {
            downsample = Double.parseDouble(downsampleField.getText().strip());
            tileSize = Integer.parseInt(tileSizeField.getText().strip());
            nThreads = Integer.parseInt(threadsField.getText().strip());
        } catch (NumberFormatException e) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "Invalid number: " + e.getMessage());
            return;
        }
        if (downsample <= 0 || tileSize <= 0 || nThreads <= 0) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "Downsample, tile size and threads must be positive.");
            return;
        }
        nThreads = Math.max(1, Math.min(nThreads, cores));

        String namesRaw =
                namesField.getText() == null ? "" : namesField.getText().strip();
        List<String> targetNames = new ArrayList<>();
        if (!namesRaw.isEmpty()) {
            for (String n : namesRaw.split(",")) {
                String t = n.strip();
                if (!t.isEmpty()) targetNames.add(t);
            }
        }

        List<PathObject> toExport = new ArrayList<>();
        if (targetNames.isEmpty()) {
            toExport.addAll(allAnnotations);
        } else {
            for (PathObject a : allAnnotations) {
                String nm = a.getName();
                if (nm != null && targetNames.stream().anyMatch(t -> t.equalsIgnoreCase(nm))) toExport.add(a);
            }
        }
        if (toExport.isEmpty()) {
            String available = allAnnotations.stream()
                    .map(a -> a.getName() == null ? "(unnamed)" : a.getName())
                    .distinct()
                    .collect(Collectors.joining(", "));
            Dialogs.showErrorMessage(EXTENSION_NAME, "No annotations matched.\nAvailable: " + available);
            return;
        }
        toExport.removeIf(a -> a.getROI() == null);
        if (toExport.isEmpty()) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "Matched annotations have no ROI to export.");
            return;
        }

        DirectoryChooser dirChooser = new DirectoryChooser();
        dirChooser.setTitle("Choose output directory for exported regions");
        File outDir = dirChooser.showDialog(qupath.getStage());
        if (outDir == null) return;

        String imageName = server.getMetadata().getName();
        var project = qupath.getProject();
        var entry = project != null ? project.getEntry(imageData) : null;
        if (entry != null && entry.getImageName() != null) imageName = entry.getImageName();
        if (imageName == null || imageName.isBlank()) imageName = "image";
        String imageStem = imageName.replaceAll("(?i)\\.ome\\.tiff?$", "").replaceAll("(?i)\\.tiff?$", "");
        final String safeStem = imageStem.replaceAll("[\\\\/:*?\"<>|\\s]+", "_");

        final List<PathObject> exportList = toExport;
        final double ds = downsample;
        final int ts = tileSize;
        final int threads = nThreads;
        final boolean bigTiff = bigTiffCb.isSelected();
        final boolean pyramid = pyramidCb.isSelected();
        final String compression = compressionBox.getValue();

        final int total = exportList.size();

        // Progress dialog — built and shown on the FX thread before the worker starts.
        // Each annotation is one (potentially long) OME-TIFF write, so the bar advances
        // once per completed region; the log area mirrors the per-region INFO/ERROR logging.
        Stage progressStage = new Stage();
        progressStage.setTitle("Exporting Annotation Regions");
        progressStage.initOwner(qupath.getStage());
        progressStage.initModality(Modality.NONE);
        progressStage.setResizable(true);

        ProgressBar progressBar = new ProgressBar(0);
        progressBar.setPrefWidth(Double.MAX_VALUE);
        Label statusLabel = new Label("Starting export… (0 / " + total + ")");
        TextArea logArea = new TextArea();
        logArea.setEditable(false);
        logArea.setPrefHeight(160);
        logArea.setStyle("-fx-font-family: monospace; -fx-font-size: 11px;");
        Button closeBtn = new Button("Close");
        closeBtn.setDisable(true);
        closeBtn.setOnAction(e -> progressStage.close());

        VBox progressRoot = new VBox(8, statusLabel, progressBar, logArea, closeBtn);
        progressRoot.setPadding(new Insets(14));
        progressStage.setScene(new Scene(progressRoot, 460, 300));
        progressStage.show();

        Thread worker = new Thread(
                () -> {
                    // The OME-TIFF writer ships with the Bio-Formats extension, which is loaded at
                    // runtime but not on the compile classpath — fail fast with a clear message.
                    try {
                        Class.forName("qupath.lib.images.writers.ome.OMEPyramidWriter$Builder");
                    } catch (Throwable t) {
                        Platform.runLater(() -> {
                            progressStage.close();
                            Dialogs.showErrorMessage(
                                    EXTENSION_NAME,
                                    "OME-TIFF writer unavailable (the Bio-Formats extension is not loaded in this QuPath instance).");
                        });
                        return;
                    }

                    int ok = 0, failed = 0, done = 0;
                    HashMap<String, Integer> nameCount = new HashMap<>();
                    for (PathObject ann : exportList) {
                        String rawName =
                                (ann.getName() != null && !ann.getName().isBlank())
                                        ? ann.getName().strip()
                                        : "Unnamed";
                        final int dStart = done + 1;
                        Platform.runLater(() ->
                                statusLabel.setText("Exporting " + dStart + " / " + total + ": " + rawName + "…"));
                        String safeName = rawName.replaceAll("[\\\\/:*?\"<>|]", "_");
                        int c = nameCount.merge(safeName, 1, Integer::sum);
                        String suffix = c > 1 ? "_" + c : "";
                        File outFile = new File(outDir, safeStem + "__" + safeName + suffix + ".ome.tif");
                        RoiMaskedServer masked = null;
                        String logLine;
                        try {
                            masked = new RoiMaskedServer(server, ann.getROI(), ds);
                            writeOmePyramid(
                                    masked, compression, ts, threads, bigTiff, pyramid, outFile.getAbsolutePath());
                            ok++;
                            logger.info(
                                    "[SP Classify] Exported annotation '{}' -> {}", rawName, outFile.getAbsolutePath());
                            logLine = "✓ " + rawName + " -> " + outFile.getName() + "\n";
                        } catch (Throwable t) {
                            failed++;
                            logger.error("[SP Classify] Export failed for '{}': {}", rawName, t.toString(), t);
                            logLine = "✗ " + rawName + ": " + t + "\n";
                        } finally {
                            if (masked != null) {
                                try {
                                    masked.close();
                                } catch (Exception ignored) {
                                    /* best effort */
                                }
                            }
                        }
                        done++;
                        final int dDone = done;
                        final String fLogLine = logLine;
                        Platform.runLater(() -> {
                            progressBar.setProgress((double) dDone / total);
                            statusLabel.setText("Processed " + dDone + " / " + total);
                            logArea.appendText(fLogLine);
                        });
                    }
                    final int fOk = ok, fFailed = failed;
                    Platform.runLater(() -> {
                        statusLabel.setText(
                                fFailed == 0
                                        ? "Done — exported " + fOk + " region(s) to " + outDir.getName() + "."
                                        : "Done — exported " + fOk + " region(s); " + fFailed + " failed (see log).");
                        closeBtn.setDisable(false);
                        if (fFailed == 0)
                            Dialogs.showInfoNotification(
                                    EXTENSION_NAME, "Exported " + fOk + " region(s) to " + outDir.getName() + ".");
                        else
                            Dialogs.showWarningNotification(
                                    EXTENSION_NAME,
                                    "Exported " + fOk + " region(s); " + fFailed + " failed (see log).");
                    });
                },
                "sp-classify-export-annotations");
        worker.setDaemon(true);
        worker.start();
    }

    /**
     * Write an image server as a pyramidal OME-TIFF via the Bio-Formats {@code OMEPyramidWriter},
     * invoked reflectively because that class is only present on QuPath's runtime classpath.
     */
    private static void writeOmePyramid(
            ImageServer<BufferedImage> server,
            String compressionName,
            int tileSize,
            int nThreads,
            boolean bigTiff,
            boolean buildPyramid,
            String outputPath)
            throws Exception {
        Class<?> builderCls = Class.forName("qupath.lib.images.writers.ome.OMEPyramidWriter$Builder");
        Class<?> compCls = Class.forName("qupath.lib.images.writers.ome.OMEPyramidWriter$CompressionType");
        Class<?> serverIface = Class.forName("qupath.lib.images.servers.ImageServer");
        Object builder = builderCls.getConstructor(serverIface).newInstance(server);

        builder = applyBuilderOption(builder, "tileSize", new Class<?>[] {int.class}, new Object[] {tileSize});
        Object compression;
        try {
            compression = compCls.getMethod("valueOf", String.class)
                    .invoke(null, compressionName.toUpperCase(java.util.Locale.ROOT));
        } catch (Exception e) {
            compression = compCls.getMethod("valueOf", String.class).invoke(null, "LZW");
        }
        builder = applyBuilderOption(builder, "compression", new Class<?>[] {compCls}, new Object[] {compression});
        builder = applyBuilderOption(builder, "parallelize", new Class<?>[] {int.class}, new Object[] {nThreads});
        builder = applyBuilderOption(builder, "bigTiff", new Class<?>[] {boolean.class}, new Object[] {bigTiff});
        if (buildPyramid)
            builder = applyBuilderOption(
                    builder, "downsamples", new Class<?>[] {double[].class}, new Object[] {new double[] {1.0, 4.0, 16.0}
                    });

        Object writer = builderCls.getMethod("build").invoke(builder);
        writer.getClass().getMethod("writeSeries", String.class).invoke(writer, outputPath);
    }

    /** Best-effort reflective call of a fluent builder option; logs and skips if unavailable. */
    private static Object applyBuilderOption(Object builder, String name, Class<?>[] types, Object[] args) {
        try {
            Object result = builder.getClass().getMethod(name, types).invoke(builder, args);
            return result != null ? result : builder;
        } catch (Exception e) {
            logger.warn("[SP Classify] OME writer option '{}' unavailable in this QuPath build; using default.", name);
            return builder;
        }
    }

    /**
     * Tileable image server that crops to an annotation's bounding box and zeroes every
     * pixel outside the ROI polygon, so exported OME-TIFFs are masked to the annotation
     * shape rather than its rectangular bounds. Ported from the export_large_annotation_regions
     * pipeline; reads from the wrapped server one tile at a time to bound memory use.
     */
    private static final class RoiMaskedServer extends AbstractTileableImageServer {
        private final ImageServer<BufferedImage> wrapped;
        private final Shape roiShapeFullRes;
        private final int cropX, cropY;
        private final double baseDownsample;
        private final ImageServerMetadata metadata;
        private final String serverId;

        RoiMaskedServer(ImageServer<BufferedImage> wrapped, ROI roi, double downsample) {
            super();
            this.wrapped = wrapped;
            this.roiShapeFullRes = roi.getShape();
            this.baseDownsample = downsample;
            this.cropX = (int) roi.getBoundsX();
            this.cropY = (int) roi.getBoundsY();
            int cropW = (int) Math.ceil(roi.getBoundsWidth() / downsample);
            int cropH = (int) Math.ceil(roi.getBoundsHeight() / downsample);
            this.serverId = "RoiMaskedServer(" + wrapped.getPath()
                    + "|x=" + cropX + "|y=" + cropY + "|w=" + cropW + "|h=" + cropH
                    + "|ds=" + downsample + ")";
            this.metadata = new ImageServerMetadata.Builder(wrapped.getMetadata())
                    .width(cropW)
                    .height(cropH)
                    .levelsFromDownsamples(1.0)
                    .build();
        }

        @Override
        public ImageServerMetadata getOriginalMetadata() {
            return metadata;
        }

        @Override
        public String getServerType() {
            return "ROI Masked Server";
        }

        @Override
        protected ImageServerBuilder.ServerBuilder<BufferedImage> createServerBuilder() {
            return null;
        }

        @Override
        protected String createID() {
            return serverId;
        }

        @Override
        public Collection<URI> getURIs() {
            return wrapped.getURIs();
        }

        @Override
        protected BufferedImage readTile(TileRequest tileRequest) throws IOException {
            double sourceDownsample = baseDownsample * tileRequest.getDownsample();
            int tileX = (int) Math.floor(cropX + tileRequest.getTileX() * sourceDownsample);
            int tileY = (int) Math.floor(cropY + tileRequest.getTileY() * sourceDownsample);
            int tileW = (int) Math.ceil(tileRequest.getTileWidth() * sourceDownsample);
            int tileH = (int) Math.ceil(tileRequest.getTileHeight() * sourceDownsample);
            RegionRequest request =
                    RegionRequest.createInstance(wrapped.getPath(), sourceDownsample, tileX, tileY, tileW, tileH);
            BufferedImage tile = wrapped.readRegion(request);
            if (tile == null) return null;

            int w = tile.getWidth();
            int h = tile.getHeight();
            int nBands = tile.getRaster().getNumBands();

            AffineTransform at = new AffineTransform();
            at.scale(1.0 / sourceDownsample, 1.0 / sourceDownsample);
            at.translate(-tileX, -tileY);
            Shape scaledShape = at.createTransformedShape(roiShapeFullRes);
            Rectangle tileBounds = new Rectangle(0, 0, w, h);
            int[] zeroRow = new int[w];

            // Fast path 1: tile fully inside the ROI — nothing to mask.
            if (scaledShape.contains(tileBounds)) return tile;

            WritableRaster tileRaster = tile.getRaster();

            // Fast path 2: tile fully outside the ROI — zero everything.
            if (!scaledShape.intersects(tileBounds)) {
                for (int b = 0; b < nBands; b++)
                    for (int y = 0; y < h; y++) tileRaster.setSamples(0, y, w, 1, b, zeroRow);
                return tile;
            }

            // Boundary tile: rasterise the ROI shape and zero outside runs row-by-row.
            BufferedImage maskImg = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_BINARY);
            Graphics2D g2 = maskImg.createGraphics();
            g2.setColor(Color.WHITE);
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
            g2.fill(scaledShape);
            g2.dispose();

            Raster maskRaster = maskImg.getRaster();
            int[] maskRow = new int[w];
            for (int y = 0; y < h; y++) {
                maskRaster.getSamples(0, y, w, 1, 0, maskRow);
                int x = 0;
                while (x < w) {
                    while (x < w && maskRow[x] != 0) x++;
                    int runStart = x;
                    while (x < w && maskRow[x] == 0) x++;
                    int runLen = x - runStart;
                    if (runLen > 0)
                        for (int b = 0; b < nBands; b++) tileRaster.setSamples(runStart, y, runLen, 1, b, zeroRow);
                }
            }
            return tile;
        }
    }
}
