package qupath.ext.spclassify.ui;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import javafx.scene.Cursor;
import javafx.scene.SnapshotParameters;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.WritableImage;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.Pane;
import javafx.scene.layout.Region;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.TextAlignment;
import qupath.ext.spclassify.model.CellPrediction;
import qupath.ext.spclassify.model.PopulationSet;
import qupath.ext.spclassify.model.ScatterMath;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;

/**
 * The visual layer of the cell scatter plot: owns the {@link Canvas}, renders the
 * embedding (axis frame, coloured dots, selection outlines, legend), and handles
 * box/lasso drag-selection plus legend click-to-select. It is deliberately free of
 * any QuPath project/hierarchy mutation — it reads plotted-row state through a
 * read-only {@link PlotModel} and reports user gestures back through callbacks, so
 * {@link ScatterPlotView} retains ownership of the data, the QuPath viewer
 * selection sync, and the status line.
 */
final class ScatterPlotCanvas {

    /** How dots are coloured. */
    enum ColorMode {
        CLUSTER,
        CLASS,
        MARKER
    }

    /**
     * Read-only view of the plotted rows the canvas draws. All index-aligned
     * arrays are length {@link #nRows()} and may be reassigned by the owner
     * between redraws, so the canvas always reads them fresh.
     */
    interface PlotModel {
        int nRows();

        double[] ex(); // embedding x (NaN = not plotted)

        double[] ey(); // embedding y

        int[] cluster(); // k-means label per row (-1 = none)

        boolean[] selected(); // plot highlight / viewer selection mirror

        double[][] raw(); // [nRows][nMarkers] raw marker values

        PathObject[] cells(); // live cells (null in project scope)

        String[] rowClass(); // class name per row (project scope; else null)

        PopulationSet predictions(); // colour-by-predicted-class source (nullable)

        List<String> markerFeatures();

        boolean projectScope();

        ColorMode colorMode();

        String embeddingName(); // axis label, e.g. "PCA" / "UMAP"

        int clusterCount(); // k

        String markerName(); // selected marker for MARKER mode (nullable)
    }

    private static final int DOT_RADIUS = 3;
    private static final double PLOT_MARGIN_LEFT = 56;
    private static final double PLOT_MARGIN_RIGHT = 150; // room for legend
    private static final double PLOT_MARGIN_TOP = 30;
    private static final double PLOT_MARGIN_BOTTOM = 48;
    private static final Font AXIS_FONT = Font.font("SansSerif", 12);
    private static final Font LEGEND_FONT = Font.font("SansSerif", 11);

    private final Canvas canvas = new Canvas(800, 600);
    private final Pane node = new Pane(canvas);

    private final PlotModel model;
    private final BooleanSupplier lassoMode;
    private final Consumer<boolean[]> onRegionGesture;
    private final IntConsumer onLegendClusterClicked;
    private final Runnable afterRedraw;

    // ── Current view geometry (recomputed each redraw) ─────────────────────────
    private double minX, maxX, minY, maxY;
    // Cached marker range for MARKER colour mode. markerLo/markerHi are percentile bounds (not raw
    // min/max) so a handful of very bright cells can't compress everyone else into one colour. The
    // percentile scan sorts, so it is cached and only redone when the marker, the plotted sample, or
    // the row count changes — never on a plain pan/zoom/selection redraw.
    private int markerColIdx = -1;
    private double markerLo, markerHi;
    private String markerRangeName;
    private double[][] markerRangeRaw;
    private int markerRangeRows = -1;

    // Percentile bounds + perceptual stretch for the marker ramp. Biological markers have a long
    // bright tail (few strong positives), so clamping to [2%, 98%] and bending the low end with a
    // gamma < 1 spreads the dim majority across the ramp instead of leaving them a flat blue.
    private static final double MARKER_LO_PCT = 0.02;
    private static final double MARKER_HI_PCT = 0.98;
    private static final double MARKER_GAMMA = 0.7;

    // ── Cluster-legend hit-testing geometry (updated each drawLegend) ──────────
    private double legendClusterX, legendClusterY, legendClusterLineH;
    private int legendClusterCount = 0;

    // ── Drag selection state ───────────────────────────────────────────────────
    private WritableImage dragCache;
    private double dragStartX, dragStartY, dragCurX, dragCurY;
    private boolean dragging = false;
    private final List<double[]> lassoPoints = new ArrayList<>();

    /**
     * @param model                  read-only plotted-row state
     * @param lassoMode              true when the lasso (vs box) gesture is active
     * @param onRegionGesture        invoked with the hit rows of a box/lasso gesture
     * @param onLegendClusterClicked invoked with the cluster index of a legend click
     * @param afterRedraw            invoked after each successful redraw (status refresh)
     */
    ScatterPlotCanvas(
            PlotModel model,
            BooleanSupplier lassoMode,
            Consumer<boolean[]> onRegionGesture,
            IntConsumer onLegendClusterClicked,
            Runnable afterRedraw) {
        this.model = model;
        this.lassoMode = lassoMode;
        this.onRegionGesture = onRegionGesture;
        this.onLegendClusterClicked = onLegendClusterClicked;
        this.afterRedraw = afterRedraw;

        canvas.widthProperty().bind(node.widthProperty());
        canvas.heightProperty().bind(node.heightProperty());
        canvas.widthProperty().addListener((o, a, b) -> redraw());
        canvas.heightProperty().addListener((o, a, b) -> redraw());

        canvas.setOnMousePressed(this::onMousePressed);
        canvas.setOnMouseDragged(this::onMouseDragged);
        canvas.setOnMouseReleased(this::onMouseReleased);
        canvas.setOnMouseMoved(this::onMouseMoved);
    }

    /** The node to place in the scene graph (a {@link Pane} wrapping the canvas). */
    Region getNode() {
        return node;
    }

    /** White-filled snapshot of the current plot, for PNG export. */
    WritableImage snapshot() {
        SnapshotParameters params = new SnapshotParameters();
        params.setFill(Color.WHITE);
        return canvas.snapshot(params, null);
    }

    // ── Drawing ──────────────────────────────────────────────────────────────

    void redraw() {
        double w = canvas.getWidth();
        double h = canvas.getHeight();
        if (w <= 0 || h <= 0) {
            return;
        }
        GraphicsContext gc = canvas.getGraphicsContext2D();
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, w, h);

        // Compute embedding bounds over plotted (non-NaN) cells.
        boolean any = computeBounds();
        if (!any) {
            gc.setFill(Color.gray(0.3));
            gc.setFont(AXIS_FONT);
            gc.setTextAlign(TextAlignment.CENTER);
            gc.fillText("No embedding to display — press Recompute.", w / 2, h / 2);
            return;
        }

        ColorMode mode = model.colorMode();
        if (mode == ColorMode.MARKER) {
            computeMarkerRange();
        }

        // Axis frame.
        double left = PLOT_MARGIN_LEFT;
        double right = w - PLOT_MARGIN_RIGHT;
        double top = PLOT_MARGIN_TOP;
        double bottom = h - PLOT_MARGIN_BOTTOM;
        gc.setStroke(Color.gray(0.75));
        gc.setLineWidth(1);
        gc.strokeRect(left, top, Math.max(1, right - left), Math.max(1, bottom - top));

        String axisName = model.embeddingName();
        gc.setFill(Color.gray(0.25));
        gc.setFont(AXIS_FONT);
        gc.setTextAlign(TextAlignment.CENTER);
        gc.fillText(axisName + "-1", (left + right) / 2, h - 14);
        gc.save();
        gc.translate(16, (top + bottom) / 2);
        gc.rotate(-90);
        gc.fillText(axisName + "-2", 0, 0);
        gc.restore();

        int n = model.nRows();
        double[] ex = model.ex();
        double[] ey = model.ey();
        boolean[] selected = model.selected();

        // Dots.
        for (int i = 0; i < n; i++) {
            if (Double.isNaN(ex[i])) {
                continue;
            }
            double px = sx(ex[i], left, right);
            double py = sy(ey[i], top, bottom);
            gc.setFill(colorFor(i, mode));
            gc.fillOval(px - DOT_RADIUS, py - DOT_RADIUS, DOT_RADIUS * 2, DOT_RADIUS * 2);
        }

        // Selection outlines (drawn on top).
        gc.setStroke(Color.WHITE);
        gc.setLineWidth(2.2);
        for (int i = 0; i < n; i++) {
            if (!selected[i] || Double.isNaN(ex[i])) {
                continue;
            }
            double px = sx(ex[i], left, right);
            double py = sy(ey[i], top, bottom);
            gc.strokeOval(px - DOT_RADIUS - 1, py - DOT_RADIUS - 1, (DOT_RADIUS + 1) * 2, (DOT_RADIUS + 1) * 2);
        }

        drawLegend(gc, right + 12, top, mode);
        afterRedraw.run();
    }

    private boolean computeBounds() {
        minX = Double.POSITIVE_INFINITY;
        maxX = Double.NEGATIVE_INFINITY;
        minY = Double.POSITIVE_INFINITY;
        maxY = Double.NEGATIVE_INFINITY;
        boolean any = false;
        int n = model.nRows();
        double[] ex = model.ex();
        double[] ey = model.ey();
        for (int i = 0; i < n; i++) {
            if (Double.isNaN(ex[i])) {
                continue;
            }
            any = true;
            minX = Math.min(minX, ex[i]);
            maxX = Math.max(maxX, ex[i]);
            minY = Math.min(minY, ey[i]);
            maxY = Math.max(maxY, ey[i]);
        }
        if (any) {
            // Avoid zero-width ranges.
            if (maxX - minX < 1e-9) {
                maxX = minX + 1;
            }
            if (maxY - minY < 1e-9) {
                maxY = minY + 1;
            }
        }
        return any;
    }

    private double sx(double dx, double left, double right) {
        return left + (dx - minX) / (maxX - minX) * (right - left);
    }

    private double sy(double dy, double top, double bottom) {
        // Invert: larger y plots higher on screen.
        return bottom - (dy - minY) / (maxY - minY) * (bottom - top);
    }

    private Color colorFor(int i, ColorMode mode) {
        switch (mode) {
            case CLASS:
                return classColor(i);
            case MARKER:
                return markerColor(i);
            case CLUSTER:
            default:
                return clusterColor(model.cluster()[i]);
        }
    }

    /** HSB-spread colour for cluster {@code c} (grey for the unclustered -1). */
    Color clusterColor(int c) {
        if (c < 0) {
            return Color.gray(0.6);
        }
        int k = Math.max(1, model.clusterCount());
        return Color.hsb(360.0 * (c % k) / k, 0.72, 0.88);
    }

    private Color classColor(int i) {
        if (model.projectScope()) {
            String[] rowClass = model.rowClass();
            return (rowClass[i] != null)
                    ? toFxColor(PathClass.fromString(rowClass[i]).getColor())
                    : Color.gray(0.6);
        }
        PopulationSet predictions = model.predictions();
        PathObject[] cells = model.cells();
        String label = null;
        if (predictions != null) {
            CellPrediction pred = predictions.get(cells[i].getID().toString());
            if (pred != null) {
                label = pred.avgLabel();
            }
        }
        PathClass pc = null;
        if (label != null) {
            pc = PathClass.fromString(label);
        } else if (cells[i].getPathClass() != null) {
            pc = cells[i].getPathClass();
        }
        if (pc == null) {
            return Color.gray(0.6);
        }
        return toFxColor(pc.getColor());
    }

    /**
     * Precompute the selected marker's percentile colour bounds over the plotted cells. Cached: the
     * sort is skipped when the marker, sample and row count are unchanged (pan/zoom/selection).
     */
    private void computeMarkerRange() {
        markerColIdx = model.markerFeatures().indexOf(model.markerName());
        if (markerColIdx < 0) {
            markerLo = 0;
            markerHi = 1;
            return;
        }
        String name = model.markerName();
        int n = model.nRows();
        double[][] raw = model.raw();
        if (java.util.Objects.equals(name, markerRangeName) && raw == markerRangeRaw && n == markerRangeRows) {
            return; // range still valid — reuse it
        }
        double[] ex = model.ex();
        double[] vals = new double[n];
        int m = 0;
        for (int r = 0; r < n; r++) {
            if (Double.isNaN(ex[r])) {
                continue;
            }
            double v = raw[r][markerColIdx];
            if (!Double.isNaN(v)) {
                vals[m++] = v;
            }
        }
        if (m == 0) {
            markerLo = 0;
            markerHi = 1;
        } else {
            double[] sorted = java.util.Arrays.copyOf(vals, m);
            java.util.Arrays.sort(sorted);
            markerLo = percentile(sorted, MARKER_LO_PCT);
            markerHi = percentile(sorted, MARKER_HI_PCT);
            if (markerHi - markerLo < 1e-9) {
                // Degenerate (nearly constant marker) — fall back to the full span so it isn't all mid-grey.
                markerLo = sorted[0];
                markerHi = sorted[m - 1];
            }
        }
        markerRangeName = name;
        markerRangeRaw = raw;
        markerRangeRows = n;
    }

    /** Linear-interpolated percentile of an ascending array; {@code p} in [0,1]. */
    private static double percentile(double[] sorted, double p) {
        if (sorted.length == 1) {
            return sorted[0];
        }
        double idx = p * (sorted.length - 1);
        int lo = (int) Math.floor(idx);
        int hi = (int) Math.ceil(idx);
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
    }

    private Color markerColor(int i) {
        if (markerColIdx < 0) {
            return Color.gray(0.6);
        }
        double[][] raw = model.raw();
        double t = (markerHi - markerLo < 1e-9) ? 0.5 : (raw[i][markerColIdx] - markerLo) / (markerHi - markerLo);
        return markerRamp(t);
    }

    /**
     * Marker colour for a normalised value {@code t} in [0,1]: clamp, apply the gamma stretch, then
     * map through {@link #gradient}. Shared by the points and the legend colourbar so they agree.
     */
    private static Color markerRamp(double t) {
        double c = Math.max(0, Math.min(1, t));
        return gradient(Math.pow(c, MARKER_GAMMA));
    }

    /** Blue (low) → red (high) gradient. */
    private static Color gradient(double t) {
        t = Math.max(0, Math.min(1, t));
        return Color.color(t, 0.15 + 0.2 * (1 - Math.abs(0.5 - t) * 2), 1 - t);
    }

    private static Color toFxColor(Integer packedRgb) {
        if (packedRgb == null) {
            return Color.gray(0.6);
        }
        int c = packedRgb;
        return Color.rgb((c >> 16) & 0xff, (c >> 8) & 0xff, c & 0xff);
    }

    private void drawLegend(GraphicsContext gc, double x, double y, ColorMode mode) {
        gc.setFont(LEGEND_FONT);
        gc.setTextAlign(TextAlignment.LEFT);
        double sw = 12;
        double lineH = 18;
        legendClusterCount = 0; // cleared unless CLUSTER legend is drawn below
        if (mode == ColorMode.CLUSTER) {
            int k = model.clusterCount();
            // Shrink the row height (and swatch) so many clusters still fit the
            // canvas and every row stays clickable for click-to-select.
            double avail = canvas.getHeight() - y - PLOT_MARGIN_BOTTOM;
            double clusterLineH = Math.min(lineH, Math.max(10, avail / Math.max(1, k)));
            double clusterSw = Math.min(sw, clusterLineH - 2);
            // Record geometry so legend rows can be hit-tested for click-to-select.
            legendClusterX = x;
            legendClusterY = y;
            legendClusterLineH = clusterLineH;
            legendClusterCount = k;
            gc.setFill(Color.BLACK);
            gc.fillText("Clusters", x, y - 6);
            for (int c = 0; c < k; c++) {
                double yy = y + c * clusterLineH;
                gc.setFill(clusterColor(c));
                gc.fillRect(x, yy, clusterSw, clusterSw);
                gc.setFill(Color.gray(0.2));
                gc.fillText("Cluster " + c, x + clusterSw + 6, yy + clusterSw - 1);
            }
        } else if (mode == ColorMode.CLASS) {
            // Distinct classes present among plotted cells (capped).
            int n = model.nRows();
            double[] ex = model.ex();
            Set<String> classes = new LinkedHashSet<>();
            for (int i = 0; i < n && classes.size() < 24; i++) {
                if (Double.isNaN(ex[i])) {
                    continue;
                }
                classes.add(classLabel(i));
            }
            gc.setFill(Color.BLACK);
            gc.fillText("Classes", x, y - 6);
            int row = 0;
            for (String cls : classes) {
                double yy = y + row * lineH;
                gc.setFill(classColorForLabel(cls));
                gc.fillRect(x, yy, sw, sw);
                gc.setFill(Color.gray(0.2));
                gc.fillText(cls, x + sw + 6, yy + sw - 2);
                row++;
            }
        } else {
            // Marker gradient colourbar.
            String marker = model.markerName();
            gc.setFill(Color.BLACK);
            gc.fillText(marker == null ? "Marker" : marker, x, y - 6);
            int steps = 80;
            double barH = 140;
            double stepH = barH / steps;
            for (int s = 0; s < steps; s++) {
                double t = 1 - s / (double) (steps - 1);
                gc.setFill(markerRamp(t));
                gc.fillRect(x, y + s * stepH, sw, stepH + 1);
            }
            gc.setFill(Color.gray(0.2));
            gc.fillText("high", x + sw + 6, y + 8);
            gc.fillText("low", x + sw + 6, y + barH);
        }
    }

    private String classLabel(int i) {
        if (model.projectScope()) {
            String[] rowClass = model.rowClass();
            return rowClass[i] != null ? rowClass[i] : "unlabelled";
        }
        PopulationSet predictions = model.predictions();
        PathObject[] cells = model.cells();
        if (predictions != null) {
            CellPrediction pred = predictions.get(cells[i].getID().toString());
            if (pred != null) {
                return pred.avgLabel();
            }
        }
        return cells[i].getPathClass() != null ? cells[i].getPathClass().getName() : "unlabelled";
    }

    private Color classColorForLabel(String label) {
        if ("unlabelled".equals(label)) {
            return Color.gray(0.6);
        }
        return toFxColor(PathClass.fromString(label).getColor());
    }

    // ── Mouse selection ──────────────────────────────────────────────────────

    private void onMouseMoved(MouseEvent e) {
        // Hint that cluster legend entries are clickable.
        boolean overLegend = clusterAtPoint(e.getX(), e.getY()) >= 0;
        canvas.setCursor(overLegend ? Cursor.HAND : Cursor.DEFAULT);
    }

    private void onMousePressed(MouseEvent e) {
        // Clicking a cluster in the legend selects that cluster's cells in the
        // viewer instead of starting a drag-selection gesture.
        int legendCluster = clusterAtPoint(e.getX(), e.getY());
        if (legendCluster >= 0) {
            dragging = false;
            onLegendClusterClicked.accept(legendCluster);
            return;
        }
        dragging = true;
        dragStartX = e.getX();
        dragStartY = e.getY();
        dragCurX = e.getX();
        dragCurY = e.getY();
        lassoPoints.clear();
        lassoPoints.add(new double[] {e.getX(), e.getY()});
        // Cache the rendered scene so drag frames are cheap.
        SnapshotParameters params = new SnapshotParameters();
        params.setFill(Color.WHITE);
        dragCache = canvas.snapshot(params, null);
    }

    private void onMouseDragged(MouseEvent e) {
        if (!dragging) {
            return;
        }
        dragCurX = e.getX();
        dragCurY = e.getY();
        GraphicsContext gc = canvas.getGraphicsContext2D();
        if (dragCache != null) {
            gc.drawImage(dragCache, 0, 0);
        }
        gc.setStroke(Color.web("#1565c0"));
        gc.setLineWidth(1.5);
        gc.setFill(Color.web("#1565c0", 0.12));
        if (lassoMode.getAsBoolean()) {
            lassoPoints.add(new double[] {e.getX(), e.getY()});
            gc.beginPath();
            gc.moveTo(lassoPoints.get(0)[0], lassoPoints.get(0)[1]);
            for (int p = 1; p < lassoPoints.size(); p++) {
                gc.lineTo(lassoPoints.get(p)[0], lassoPoints.get(p)[1]);
            }
            gc.stroke();
        } else {
            double x = Math.min(dragStartX, dragCurX);
            double y = Math.min(dragStartY, dragCurY);
            double rw = Math.abs(dragCurX - dragStartX);
            double rh = Math.abs(dragCurY - dragStartY);
            gc.fillRect(x, y, rw, rh);
            gc.strokeRect(x, y, rw, rh);
        }
    }

    private void onMouseReleased(MouseEvent e) {
        if (!dragging) {
            return;
        }
        dragging = false;
        dragCache = null;

        double left = PLOT_MARGIN_LEFT;
        double right = canvas.getWidth() - PLOT_MARGIN_RIGHT;
        double top = PLOT_MARGIN_TOP;
        double bottom = canvas.getHeight() - PLOT_MARGIN_BOTTOM;

        boolean lasso = lassoMode.getAsBoolean();
        double rx = Math.min(dragStartX, dragCurX);
        double ry = Math.min(dragStartY, dragCurY);
        double rw = Math.abs(dragCurX - dragStartX);
        double rh = Math.abs(dragCurY - dragStartY);
        boolean tinyGesture = !lasso && rw < 3 && rh < 3;

        int n = model.nRows();
        double[] ex = model.ex();
        double[] ey = model.ey();
        boolean[] hit = new boolean[n];
        if (!tinyGesture) {
            for (int i = 0; i < n; i++) {
                if (Double.isNaN(ex[i])) {
                    continue;
                }
                double px = sx(ex[i], left, right);
                double py = sy(ey[i], top, bottom);
                boolean inside = lasso
                        ? ScatterMath.pointInPolygon(px, py, lassoPoints)
                        : (px >= rx && px <= rx + rw && py >= ry && py <= ry + rh);
                if (inside) {
                    hit[i] = true;
                }
            }
        }
        onRegionGesture.accept(hit);
    }

    /**
     * Returns the cluster index whose legend row contains the canvas point, or
     * -1 if the point is not on a clickable cluster legend entry. Only valid in
     * CLUSTER colour mode (geometry is recorded by {@link #drawLegend}).
     */
    private int clusterAtPoint(double mx, double my) {
        if (model.colorMode() != ColorMode.CLUSTER || legendClusterCount <= 0) {
            return -1;
        }
        if (mx < legendClusterX - 4 || mx > canvas.getWidth()) {
            return -1;
        }
        double rel = my - (legendClusterY - 3);
        if (rel < 0) {
            return -1;
        }
        int c = (int) (rel / legendClusterLineH);
        return (c >= 0 && c < legendClusterCount) ? c : -1;
    }
}
