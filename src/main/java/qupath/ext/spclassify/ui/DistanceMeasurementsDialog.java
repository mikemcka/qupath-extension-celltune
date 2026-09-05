package qupath.ext.spclassify.ui;

import java.awt.image.BufferedImage;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.Modality;
import javafx.stage.Stage;
import org.locationtech.jts.geom.Envelope;
import org.locationtech.jts.index.strtree.ItemBoundable;
import org.locationtech.jts.index.strtree.ItemDistance;
import org.locationtech.jts.index.strtree.STRtree;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.util.BackgroundExecutors;
import qupath.lib.analysis.DistanceTools;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServerMetadata;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.classes.PathClassTools;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;

/**
 * Project-wide batch dialog that runs distance-measurement computations on the
 * selected images:
 *
 * <ul>
 *   <li>Detection-to-annotation signed distances ({@link DistanceTools#detectionToAnnotationDistancesSigned})</li>
 *   <li>Cross-class centroid distances ({@link DistanceTools#detectionCentroidDistances})</li>
 *   <li>Same-class nearest-neighbour centroid distances (computed here, excludes self)</li>
 * </ul>
 *
 * <p>If the user supplies a pixel size (µm/pixel) it temporarily overrides each
 * image's calibration so the resulting measurements are written in microns.
 * The override is reverted after the run unless "Persist…" is checked.
 *
 * <p>Images are processed in parallel via a fixed thread pool capped at
 * {@code min(selected, max(1, min(4, cores/2)))} — matching the extension's
 * existing I/O-bound parallelism pattern.
 */
public class DistanceMeasurementsDialog {

    private static final Logger logger = LoggerFactory.getLogger(DistanceMeasurementsDialog.class);

    private final QuPathGUI qupath;
    private final Stage stage;

    private final Map<String, CheckBox> imageCheckBoxes = new LinkedHashMap<>();
    private final CheckBox annotationDistCheck = new CheckBox("Detection-to-annotation signed distances");
    private final CheckBox crossClassCheck = new CheckBox("Cross-class centroid distances");
    private final CheckBox sameClassCheck = new CheckBox("Same-class nearest-neighbour distances (excludes self)");
    private final TextField pixelSizeField = new TextField();
    private final CheckBox updateCalibrationCheck =
            new CheckBox("Persist this pixel size to each image's calibration on save");
    private final CheckBox skipCompletedCheck =
            new CheckBox("Skip images where all selected measurements already exist");
    private final Spinner<Integer> workerSpinner = new Spinner<>();

    private final TextArea logArea = new TextArea();
    private final ProgressBar progressBar = new ProgressBar(0);
    private final Label statusLabel = new Label("Configure and click Apply.");
    private final Button applyBtn = new Button("Apply");
    private final Button closeBtn = new Button("Close");

    public DistanceMeasurementsDialog(QuPathGUI qupath) {
        this.qupath = qupath;
        this.stage = buildStage();
    }

    public void show() {
        stage.show();
    }

    // ── UI ─────────────────────────────────────────────────────────────────────

    private Stage buildStage() {
        var project = qupath.getProject();

        // Resolve current image + its existing pixel calibration (used as default).
        String currentImageName = null;
        Double defaultPixelSize = null;
        var liveImageData = qupath.getImageData();
        if (liveImageData != null) {
            if (project != null) {
                var entry = project.getEntry(liveImageData);
                if (entry != null) currentImageName = entry.getImageName();
            }
            try {
                var cal = liveImageData.getServer().getPixelCalibration();
                if (cal.hasPixelSizeMicrons()) {
                    defaultPixelSize = cal.getPixelWidthMicrons();
                }
            } catch (Exception ignored) {
            }
        }
        final String currentImageNameFinal = currentImageName;

        // ── Image checklist ──
        VBox imageBox = new VBox(4);
        imageBox.setPadding(new Insets(4));
        if (project != null) {
            @SuppressWarnings("unchecked")
            var entries = (List<ProjectImageEntry<BufferedImage>>) (List<?>) project.getImageList();
            for (var entry : entries) {
                String name = entry.getImageName();
                if (name == null) continue;
                CheckBox cb = new CheckBox(name);
                cb.setSelected(true);
                imageCheckBoxes.put(name, cb);
                imageBox.getChildren().add(cb);
            }
        }
        if (imageCheckBoxes.isEmpty()) {
            imageBox.getChildren().add(new Label("No project images found."));
        }

        ScrollPane imageScroll = new ScrollPane(imageBox);
        imageScroll.setFitToWidth(true);
        imageScroll.setPrefHeight(180);
        imageScroll.setStyle("-fx-border-color: #ccc;");

        Button selectAllImgs = new Button("All");
        Button selectNoneImgs = new Button("None");
        Button selectCurrentImg = new Button("Current only");
        selectAllImgs.setOnAction(e -> imageCheckBoxes.values().forEach(cb -> cb.setSelected(true)));
        selectNoneImgs.setOnAction(e -> imageCheckBoxes.values().forEach(cb -> cb.setSelected(false)));
        selectCurrentImg.setOnAction(
                e -> imageCheckBoxes.forEach((n, cb) -> cb.setSelected(n.equals(currentImageNameFinal))));
        HBox imgButtons = new HBox(6, new Label("Images:"), selectAllImgs, selectNoneImgs, selectCurrentImg);
        imgButtons.setAlignment(Pos.CENTER_LEFT);

        // ── Computation toggles ──
        annotationDistCheck.setSelected(true);
        crossClassCheck.setSelected(true);
        sameClassCheck.setSelected(true);
        annotationDistCheck.setTooltip(
                new Tooltip("Calls QuPath's DistanceTools.detectionToAnnotationDistancesSigned.\n"
                        + "Negative when the detection centroid lies inside the annotation, positive when outside."));
        crossClassCheck.setTooltip(new Tooltip("Calls QuPath's DistanceTools.detectionCentroidDistances.\n"
                + "For each cell, writes the nearest centroid-to-centroid distance to a cell of every OTHER class."));
        sameClassCheck.setTooltip(new Tooltip(
                "For each cell, finds the nearest cell of the SAME class (excluding itself) and writes the distance under\n"
                        + "'Distance to other <class> <unit>'. Parallelised across CPU cores per class."));

        // ── Pixel size override ──
        pixelSizeField.setPromptText("e.g. 0.5");
        pixelSizeField.setPrefColumnCount(8);
        if (defaultPixelSize != null) {
            pixelSizeField.setText(formatPixelSize(defaultPixelSize));
        }
        Label pxLabel = new Label("Pixel size:");
        HBox pxRow = new HBox(6, pxLabel, pixelSizeField, new Label("µm/pixel"));
        pxRow.setAlignment(Pos.CENTER_LEFT);
        String pxSourceMsg;
        if (defaultPixelSize != null) {
            String srcName = currentImageNameFinal != null ? "\"" + currentImageNameFinal + "\"" : "the current image";
            pxSourceMsg = "Pre-filled from " + srcName + " (" + formatPixelSize(defaultPixelSize)
                    + " µm/pixel) — edit if incorrect. ";
        } else if (liveImageData != null) {
            pxSourceMsg =
                    "No pixel size found in the current image's metadata — enter one manually if you want µm units. ";
        } else {
            pxSourceMsg = "";
        }
        Label pxHint = new Label(
                pxSourceMsg
                        + "Leave blank to use each image's existing calibration. If supplied, the override applies to every selected image.");
        pxHint.setWrapText(true);
        pxHint.setStyle("-fx-font-size: 11px; -fx-text-fill: #666;");

        updateCalibrationCheck.setSelected(false);
        updateCalibrationCheck.setTooltip(
                new Tooltip(
                        "When checked, the pixel size above is saved into each image's calibration metadata,\n"
                                + "so future measurements also use this scale. When unchecked, the override is reverted after the run."));

        skipCompletedCheck.setSelected(true);
        skipCompletedCheck.setTooltip(
                new Tooltip(
                        "Before computing, scans every cell in the image. If all cells already carry every\n"
                                + "measurement the selected computations would produce, the image is skipped entirely —\n"
                                + "no recompute and no re-save. Uncheck to force recomputation (e.g. after changing classes)."));

        // ── Parallel image workers ──
        int cores = Runtime.getRuntime().availableProcessors();
        int defaultWorkers = Math.max(1, Math.min(4, cores / 2));
        workerSpinner.setValueFactory(
                new SpinnerValueFactory.IntegerSpinnerValueFactory(1, Math.max(1, cores), defaultWorkers));
        workerSpinner.setEditable(true);
        workerSpinner.setPrefWidth(80);
        workerSpinner.setTooltip(new Tooltip("How many images are processed at the same time.\n"
                + "Each image's heavy distance maths already spreads across all CPU cores, so more\n"
                + "workers mainly overlaps image loading/saving (I/O) with computation."));
        HBox workerRow =
                new HBox(6, new Label("Parallel image workers:"), workerSpinner, new Label("(of " + cores + " cores)"));
        workerRow.setAlignment(Pos.CENTER_LEFT);
        Label workerHint = new Label(
                "1–2 workers is often fastest for large images, ie 500k+ cells, use more workers for smaller images. "
                        + "Default: " + defaultWorkers + ".");
        workerHint.setWrapText(true);
        workerHint.setStyle("-fx-font-size: 11px; -fx-text-fill: #666;");

        // ── Log / progress ──
        logArea.setEditable(false);
        logArea.setPrefHeight(160);
        logArea.setStyle("-fx-font-family: monospace; -fx-font-size: 11px;");
        progressBar.setPrefWidth(Double.MAX_VALUE);

        // ── Action buttons ──
        applyBtn.setDefaultButton(true);
        applyBtn.setPrefWidth(90);
        applyBtn.setOnAction(e -> runMeasurements());
        closeBtn.setPrefWidth(90);
        closeBtn.setOnAction(e -> stage.close());
        HBox buttons = new HBox(10, applyBtn, closeBtn);
        buttons.setAlignment(Pos.CENTER_RIGHT);
        buttons.setPadding(new Insets(4, 0, 0, 0));

        // ── Root layout ──
        VBox root = new VBox(
                10,
                imgButtons,
                imageScroll,
                new Separator(),
                new Label("Measurements to generate:"),
                annotationDistCheck,
                crossClassCheck,
                sameClassCheck,
                skipCompletedCheck,
                new Separator(),
                pxRow,
                pxHint,
                updateCalibrationCheck,
                new Separator(),
                workerRow,
                workerHint,
                new Separator(),
                statusLabel,
                progressBar,
                logArea,
                buttons);
        root.setPadding(new Insets(14));

        Stage s = new Stage();
        s.setTitle("Generate Distance Measurements");
        s.initOwner(qupath.getStage());
        s.initModality(Modality.NONE);
        s.setScene(new Scene(root, 580, 880));
        return s;
    }

    private static String formatPixelSize(double v) {
        String s = String.format(Locale.US, "%.6f", v);
        // Trim trailing zeros / dot
        if (s.contains(".")) {
            s = s.replaceAll("0+$", "");
            if (s.endsWith(".")) s = s.substring(0, s.length() - 1);
        }
        return s;
    }

    // ── Worker ─────────────────────────────────────────────────────────────────

    private void runMeasurements() {
        var project = qupath.getProject();
        if (project == null) {
            log("ERROR: No project open.");
            return;
        }

        var selectedImages = imageCheckBoxes.entrySet().stream()
                .filter(e -> e.getValue().isSelected())
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
        if (selectedImages.isEmpty()) {
            log("Select at least one image.");
            return;
        }

        if (!annotationDistCheck.isSelected() && !crossClassCheck.isSelected() && !sameClassCheck.isSelected()) {
            log("Select at least one measurement type.");
            return;
        }

        // Pixel size override (optional)
        Double pixelSizeOverride = null;
        String pxText =
                pixelSizeField.getText() == null ? "" : pixelSizeField.getText().trim();
        if (!pxText.isEmpty()) {
            try {
                pixelSizeOverride = Double.parseDouble(pxText);
                if (pixelSizeOverride <= 0 || Double.isNaN(pixelSizeOverride) || Double.isInfinite(pixelSizeOverride)) {
                    log("ERROR: Pixel size must be a positive number.");
                    return;
                }
            } catch (NumberFormatException ex) {
                log("ERROR: Pixel size must be a number (e.g. 0.5).");
                return;
            }
        }

        final boolean doAnn = annotationDistCheck.isSelected();
        final boolean doCross = crossClassCheck.isSelected();
        final boolean doSame = sameClassCheck.isSelected();
        final boolean persistCal = updateCalibrationCheck.isSelected();
        final boolean skipCompleted = skipCompletedCheck.isSelected();
        final Double pxSize = pixelSizeOverride;

        Integer requestedWorkers = workerSpinner.getValue();
        final int workers = requestedWorkers == null ? 1 : Math.max(1, requestedWorkers);

        applyBtn.setDisable(true);
        progressBar.setProgress(0);
        logArea.clear();
        log("Starting on " + selectedImages.size() + " image(s)…");

        @SuppressWarnings("unchecked")
        var typedProject = (Project<BufferedImage>) (Object) project;

        Thread worker = new Thread(
                () -> {
                    int cores = Runtime.getRuntime().availableProcessors();
                    int nThreads = Math.min(selectedImages.size(), workers);
                    log("Using " + nThreads + " parallel image worker(s) (cores=" + cores + ").");
                    ExecutorService pool = BackgroundExecutors.newFixedPool(nThreads, "SpClassify-Distances");
                    AtomicInteger done = new AtomicInteger();
                    int total = selectedImages.size();

                    List<Future<?>> futures = new ArrayList<>();
                    for (String imgName : selectedImages) {
                        futures.add(pool.submit(() -> {
                            try {
                                processOne(
                                        typedProject,
                                        imgName,
                                        doAnn,
                                        doCross,
                                        doSame,
                                        pxSize,
                                        persistCal,
                                        skipCompleted);
                            } catch (Exception ex) {
                                log("[" + imgName + "] ERROR: " + ex.getMessage());
                                logger.warn("Distance computation failed for {}", imgName, ex);
                            } finally {
                                int d = done.incrementAndGet();
                                Platform.runLater(() -> {
                                    progressBar.setProgress((double) d / total);
                                    statusLabel.setText("Processed " + d + " / " + total);
                                });
                            }
                        }));
                    }

                    pool.shutdown();
                    for (var f : futures) {
                        try {
                            f.get();
                        } catch (Exception ignored) {
                        }
                    }

                    Platform.runLater(() -> {
                        log("Done.");
                        applyBtn.setDisable(false);
                    });
                },
                "SpClassify-Distances-Coordinator");
        worker.setDaemon(true);
        worker.start();
    }

    private void processOne(
            Project<BufferedImage> project,
            String imgName,
            boolean doAnn,
            boolean doCross,
            boolean doSame,
            Double pxSize,
            boolean persistCal,
            boolean skipCompleted)
            throws Exception {
        var entryOpt = project.getImageList().stream()
                .filter(e -> imgName.equals(e.getImageName()))
                .findFirst();
        if (entryOpt.isEmpty()) {
            log("[" + imgName + "] not found in project");
            return;
        }
        var entry = entryOpt.get();
        log("[" + imgName + "] Loading…");
        var imageData = entry.readImageData();
        if (imageData == null) {
            log("[" + imgName + "] could not load image data");
            return;
        }

        // Fast path: if every cell already carries all the measurements the
        // selected computations would produce, skip the (expensive) recompute
        // and the re-save entirely.
        if (skipCompleted && alreadyComplete(imageData, doAnn, doCross, doSame, pxSize)) {
            log("[" + imgName + "] Skipped — all selected measurements already present.");
            return;
        }

        // Possibly override pixel calibration. Stash the original so we can
        // restore it after the run if the user didn't ask to persist.
        ImageServerMetadata originalMeta = null;
        if (pxSize != null) {
            try {
                var oldMeta = imageData.getServer().getMetadata();
                var oldCal = oldMeta.getPixelCalibration();
                boolean needsUpdate = !oldCal.hasPixelSizeMicrons()
                        || Math.abs(oldCal.getPixelWidthMicrons() - pxSize) > 1e-9
                        || Math.abs(oldCal.getPixelHeightMicrons() - pxSize) > 1e-9;
                if (needsUpdate) {
                    originalMeta = oldMeta;
                    var newMeta = new ImageServerMetadata.Builder(oldMeta)
                            .pixelSizeMicrons(pxSize, pxSize)
                            .build();
                    imageData.updateServerMetadata(newMeta);
                    log("[" + imgName + "] pixel size set to " + pxSize + " µm/pixel");
                }
            } catch (Exception ex) {
                log("[" + imgName + "] WARN: could not override pixel calibration (" + ex.getMessage()
                        + "); using existing");
            }
        }

        try {
            if (doAnn) {
                log("[" + imgName + "] Annotation signed distances…");
                DistanceTools.detectionToAnnotationDistancesSigned(imageData, false);
            }
            if (doCross) {
                log("[" + imgName + "] Cross-class centroid distances…");
                DistanceTools.detectionCentroidDistances(imageData, false);
            }
            if (doSame) {
                log("[" + imgName + "] Same-class nearest-neighbour distances…");
                computeSameClassDistances(imageData, msg -> log("[" + imgName + "] " + msg));
            }
        } finally {
            if (originalMeta != null && !persistCal) {
                try {
                    imageData.updateServerMetadata(originalMeta);
                } catch (Exception ex) {
                    log("[" + imgName + "] WARN: could not restore original calibration: " + ex.getMessage());
                }
            }
        }

        imageData.getHierarchy().fireHierarchyChangedEvent(this);
        entry.saveImageData(imageData);
        log("[" + imgName + "] Saved.");
    }

    /**
     * Returns {@code true} when every cell in the image already carries all the
     * measurements that the selected computations would produce, so the
     * recompute and re-save can be skipped entirely.
     *
     * <p>Expected measurement names are derived to match exactly what the run
     * would write:
     * <ul>
     *   <li>Annotation — {@code "Signed distance to annotation <class> <unit>"}
     *       (QuPath {@code DistanceTools}, one per valid, non-ignored annotation class).</li>
     *   <li>Cross-class — {@code "Distance to detection <class> <unit>"}
     *       (QuPath {@code DistanceTools}, one per valid, non-ignored detection class;
     *       written to every detection, so these are required of all cells).</li>
     *   <li>Same-class — {@code "Distance to other <class> <unit>"}
     *       (this dialog, only for classes with at least two members; required only
     *       of cells in those classes).</li>
     * </ul>
     *
     * <p>The unit suffix mirrors the calibration that the run would use: if a
     * pixel-size override is supplied the units become microns, otherwise the
     * image's existing calibration is used. Returns {@code false} when there is
     * nothing to compute (so the caller takes the normal, cheap path).
     */
    private static boolean alreadyComplete(
            ImageData<?> imageData, boolean doAnn, boolean doCross, boolean doSame, Double pxSize) {
        var hierarchy = imageData.getHierarchy();
        var cells = hierarchy.getCellObjects();
        var detections = cells.isEmpty() ? hierarchy.getDetectionObjects() : cells;
        if (detections.isEmpty()) return false;

        var cal = imageData.getServer().getPixelCalibration();
        boolean willBeMicrons = pxSize != null || cal.hasPixelSizeMicrons();
        // QuPath DistanceTools uses the calibration's unit string; the same-class
        // helper uses "µm"/"px". These agree for standard calibrations.
        String unitQp = pxSize != null ? "µm" : cal.getPixelWidthUnit();
        String unitSame = willBeMicrons ? "µm" : "px";

        // Names written to every detection.
        Set<String> globalExpected = new HashSet<>();
        if (doAnn) {
            for (var pc : distinctValidClasses(hierarchy.getAnnotationObjects()))
                globalExpected.add("Signed distance to annotation " + pc + " " + unitQp);
        }
        if (doCross) {
            for (var pc : distinctValidClasses(detections))
                globalExpected.add("Distance to detection " + pc + " " + unitQp);
        }

        // Same-class names — keyed on class string, only for classes with >= 2 members.
        Map<String, Integer> sameClassCounts = new HashMap<>();
        if (doSame) {
            for (var cell : cells) {
                var pc = cell.getPathClass();
                if (pc != null) sameClassCounts.merge(pc.toString(), 1, Integer::sum);
            }
        }
        boolean anySameClass = sameClassCounts.values().stream().anyMatch(c -> c >= 2);

        if (globalExpected.isEmpty() && !anySameClass) return false;

        for (var cell : detections) {
            var ml = cell.getMeasurementList();
            for (var name : globalExpected) {
                if (!ml.containsKey(name)) return false;
            }
            if (anySameClass) {
                var pc = cell.getPathClass();
                if (pc != null) {
                    Integer cnt = sameClassCounts.get(pc.toString());
                    if (cnt != null && cnt >= 2 && !ml.containsKey("Distance to other " + pc + " " + unitSame))
                        return false;
                }
            }
        }
        return true;
    }

    /** Distinct valid, non-ignored path classes across the given objects. */
    private static Set<PathClass> distinctValidClasses(Collection<PathObject> objects) {
        Set<PathClass> set = new LinkedHashSet<>();
        for (var o : objects) {
            var pc = o.getPathClass();
            if (pc != null && pc.isValid() && !PathClassTools.isIgnoredClass(pc)) set.add(pc);
        }
        return set;
    }

    /**
     * For each represented detection class, computes centroid-to-centroid
     * distance to the nearest OTHER detection in the same class. Distances are
     * scaled by the imageData's current pixel calibration (µm if calibrated,
     * pixels otherwise) — the measurement name carries the unit suffix.
     *
     * <p>Uses a JTS {@link STRtree} for O(n log n) nearest-neighbour queries.
     * Each query passes the cell's own object reference as the exclusion item,
     * so the result is guaranteed to be a different cell of the same class.
     * Comfortably handles classes with hundreds of thousands of cells.
     */
    private static void computeSameClassDistances(ImageData<?> imageData, java.util.function.Consumer<String> log) {
        var hierarchy = imageData.getHierarchy();
        var cal = imageData.getServer().getPixelCalibration();
        boolean calibrated = cal.hasPixelSizeMicrons();
        double pw = calibrated ? cal.getPixelWidthMicrons() : 1.0;
        double ph = calibrated ? cal.getPixelHeightMicrons() : 1.0;
        String unit = calibrated ? "µm" : "px";

        Map<String, List<PathObject>> byClass = new LinkedHashMap<>();
        for (var cell : hierarchy.getCellObjects()) {
            var pc = cell.getPathClass();
            if (pc == null) continue;
            byClass.computeIfAbsent(pc.toString(), k -> new ArrayList<>()).add(cell);
        }

        for (var entry : byClass.entrySet()) {
            String className = entry.getKey();
            List<PathObject> cells = entry.getValue();
            int n = cells.size();
            String mname = "Distance to other " + className + " " + unit;
            if (n < 2) {
                log.accept("Skipping '" + className + "' (n=" + n + ")");
                continue;
            }

            double[] xs = new double[n];
            double[] ys = new double[n];
            for (int i = 0; i < n; i++) {
                var roi = cells.get(i).getROI();
                if (roi == null) {
                    xs[i] = Double.NaN;
                    ys[i] = Double.NaN;
                } else {
                    xs[i] = roi.getCentroidX() * pw;
                    ys[i] = roi.getCentroidY() * ph;
                }
            }

            long t0 = System.currentTimeMillis();
            double[] distances = sameClassNearestNeighbourDistances(xs, ys);
            long elapsed = System.currentTimeMillis() - t0;

            for (int i = 0; i < n; i++) {
                cells.get(i).getMeasurementList().put(mname, distances[i]);
            }
            log.accept(String.format(Locale.US, "  %s: %d cells in %d ms → %s", className, n, elapsed, mname));
        }
    }

    /**
     * Pure helper: given parallel arrays of xs/ys (already in the desired
     * units), returns a same-length array where each entry is the distance
     * from that point to its nearest neighbour in the same set, excluding
     * itself. Entries with NaN coordinates produce NaN, as do points whose
     * only valid neighbours are themselves (single valid point).
     *
     * <p>Package-private for unit testing.
     */
    static double[] sameClassNearestNeighbourDistances(double[] xs, double[] ys) {
        if (xs.length != ys.length) {
            throw new IllegalArgumentException("xs and ys must have the same length");
        }
        int n = xs.length;
        double[] out = new double[n];
        if (n == 0) return out;

        // JTS's 3-arg nearestNeighbour does NOT auto-exclude the query item —
        // self-exclusion has to live in the ItemDistance callback. Each index
        // gets a unique Object marker; reference equality is used to skip self.
        Object[] keys = new Object[n];
        IdentityHashMap<Object, Integer> indexByKey = new IdentityHashMap<>(n * 2);
        STRtree tree = new STRtree();
        int inserted = 0;
        for (int i = 0; i < n; i++) {
            if (Double.isNaN(xs[i]) || Double.isNaN(ys[i])) continue;
            keys[i] = new Object();
            indexByKey.put(keys[i], i);
            tree.insert(new Envelope(xs[i], xs[i], ys[i], ys[i]), keys[i]);
            inserted++;
        }
        if (inserted < 2) {
            Arrays.fill(out, Double.NaN);
            return out;
        }
        tree.build();

        ItemDistance pointDistance = (ItemBoundable a, ItemBoundable b) -> {
            if (a.getItem() == b.getItem()) return Double.POSITIVE_INFINITY;
            Envelope ea = (Envelope) a.getBounds();
            Envelope eb = (Envelope) b.getBounds();
            double dx = ea.getMinX() - eb.getMinX();
            double dy = ea.getMinY() - eb.getMinY();
            return Math.sqrt(dx * dx + dy * dy);
        };

        // The STRtree is immutable after build() and its queries are read-only,
        // so the per-cell nearest-neighbour lookups parallelise safely. This is
        // the dominant cost for large classes (hundreds of thousands of cells),
        // so spreading it across cores is the main speed-up.
        java.util.stream.IntStream.range(0, n).parallel().forEach(i -> {
            double xi = xs[i], yi = ys[i];
            if (Double.isNaN(xi) || Double.isNaN(yi)) {
                out[i] = Double.NaN;
                return;
            }
            Envelope env = new Envelope(xi, xi, yi, yi);
            Object nearest = tree.nearestNeighbour(env, keys[i], pointDistance);
            Integer j = indexByKey.get(nearest);
            if (j == null || j == i) {
                out[i] = Double.NaN;
                return;
            }
            double dx = xi - xs[j];
            double dy = yi - ys[j];
            out[i] = Math.sqrt(dx * dx + dy * dy);
        });
        return out;
    }

    private void log(String msg) {
        Platform.runLater(() -> logArea.appendText(msg + "\n"));
    }
}
