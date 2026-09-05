package qupath.ext.spclassify.ui;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeSet;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.Window;
import qupath.ext.spclassify.model.BatchNormalizerCohort.NormalizerFit;

/**
 * Non-modal QC window for a UniFORM batch-normalization fit: a per-marker batch-alignment
 * metric table plus before/after log-intensity distribution overlays grouped by batch.
 * "Before" histograms are the fitted per-image histograms pooled by batch; "after" are the
 * same histograms shifted by each image's integer bin shift (the correction is an exact
 * integer-bin translation in log-space), so no re-read of the data is needed.
 */
public class BatchNormQcView {

    private final Window owner;
    private final NormalizerFit fit;

    public BatchNormQcView(Window owner, NormalizerFit fit) {
        this.owner = owner;
        this.fit = fit;
    }

    // ── Metric (also used for the dialog's one-line summary) ─────────────────

    /** Row of the metric table (public getters required by the JavaFX {@code PropertyValueFactory}). */
    public static final class Row {
        private final String marker;
        private final double before;
        private final double after;
        private final double reductionPct;

        Row(String marker, double before, double after) {
            this.marker = marker;
            this.before = before;
            this.after = after;
            this.reductionPct = before > 0 ? 100.0 * (before - after) / before : 0.0;
        }

        public String getMarker() {
            return marker;
        }

        public String getBefore() {
            return String.format(Locale.US, "%.3f", before);
        }

        public String getAfter() {
            return String.format(Locale.US, "%.3f", after);
        }

        public String getReductionPct() {
            return String.format(Locale.US, "%.0f%%", reductionPct);
        }

        double beforeVal() {
            return before;
        }

        double afterVal() {
            return after;
        }
    }

    /** Per-marker batch-alignment rows: SD (across batches) of the batch median log-intensity, before/after. */
    public static List<Row> metricRows(NormalizerFit fit) {
        BatchHists bh = batchHistograms(fit);
        List<Row> rows = new ArrayList<>();
        int nMarkers = fit.markers().size();
        for (int m = 0; m < nMarkers; m++) {
            double before = spreadOfBatchMedians(bh.before, m, bh.batches, fit, m);
            double after = spreadOfBatchMedians(bh.after, m, bh.batches, fit, m);
            rows.add(new Row(fit.markers().get(m), before, after));
        }
        return rows;
    }

    /** One-line summary for the control dialog's log/status. */
    public static String metricSummary(NormalizerFit fit) {
        List<Row> rows = metricRows(fit);
        if (rows.isEmpty()) {
            return "No markers fitted.";
        }
        double b = 0;
        double a = 0;
        for (Row r : rows) {
            b += r.beforeVal();
            a += r.afterVal();
        }
        b /= rows.size();
        a /= rows.size();
        double red = b > 0 ? 100.0 * (b - a) / b : 0.0;
        return String.format(
                Locale.US, "Batch spread %.3f → %.3f (%.0f%% reduction) across %d marker(s).", b, a, red, rows.size());
    }

    // ── UI ───────────────────────────────────────────────────────────────────

    public void show() {
        List<Row> rows = metricRows(fit);
        TableView<Row> table = new TableView<>();
        table.getColumns().add(col("Marker", "marker", 240));
        table.getColumns().add(col("Before (SD)", "before", 100));
        table.getColumns().add(col("After (SD)", "after", 100));
        table.getColumns().add(col("Reduction", "reductionPct", 100));
        table.getItems().addAll(rows);
        table.setColumnResizePolicy(TableView.CONSTRAINED_RESIZE_POLICY);
        table.setPrefHeight(220);

        BatchHists bh = batchHistograms(fit);
        VBox charts = new VBox(14);
        for (int m = 0; m < fit.markers().size(); m++) {
            charts.getChildren().add(markerCharts(m, bh));
        }
        ScrollPane chartScroll = new ScrollPane(charts);
        chartScroll.setFitToWidth(true);
        chartScroll.setPrefHeight(520);

        Label header = new Label(metricSummary(fit));
        header.setStyle("-fx-font-weight: bold;");
        Label subtitle = new Label("Lower SD = batches better aligned. Curves = per-batch log-intensity densities.");
        subtitle.setStyle("-fx-font-size: 11px; -fx-text-fill: #666;");
        VBox root = new VBox(10, header, subtitle, table, new Label("Per-marker distributions:"), chartScroll);
        root.setPadding(new Insets(14));

        Stage s = new Stage();
        s.setTitle("Batch Normalisation — QC");
        if (owner != null) {
            s.initOwner(owner);
        }
        s.initModality(Modality.NONE);
        s.setScene(new Scene(root, 900, 820));
        s.show();
    }

    private static TableColumn<Row, String> col(String title, String prop, double width) {
        TableColumn<Row, String> c = new TableColumn<>(title);
        c.setCellValueFactory(new PropertyValueFactory<>(prop));
        c.setPrefWidth(width);
        return c;
    }

    /** Before | After line charts (one series per batch) for one marker. */
    private HBox markerCharts(int marker, BatchHists bh) {
        return new HBox(
                8,
                oneChart(fit.markers().get(marker) + " — before", marker, bh.before, bh),
                oneChart(fit.markers().get(marker) + " — after", marker, bh.after, bh));
    }

    private LineChart<Number, Number> oneChart(String title, int marker, Map<String, long[][]> hists, BatchHists bh) {
        NumberAxis x = new NumberAxis();
        NumberAxis y = new NumberAxis();
        x.setLabel("log intensity");
        LineChart<Number, Number> chart = new LineChart<>(x, y);
        chart.setTitle(title);
        chart.setCreateSymbols(false);
        chart.setLegendVisible(false);
        chart.setPrefSize(420, 240);
        chart.setAnimated(false);
        double logMin = fit.logMin()[marker];
        double logMax = fit.logMax()[marker];
        int nBins = fit.nBins();
        double inc = nBins > 1 && logMax > logMin ? (logMax - logMin) / (nBins - 1) : 1.0;
        int step = Math.max(1, nBins / 128); // downsample for responsiveness
        for (String batch : bh.batches) {
            long[][] perMarker = hists.get(batch);
            if (perMarker == null) {
                continue;
            }
            long[] h = perMarker[marker];
            long total = 0;
            for (long v : h) {
                total += v;
            }
            if (total == 0) {
                continue;
            }
            XYChart.Series<Number, Number> series = new XYChart.Series<>();
            series.setName(batch);
            for (int b = 0; b < nBins; b += step) {
                double density = h[b] / (double) total;
                series.getData().add(new XYChart.Data<>(logMin + b * inc, density));
            }
            chart.getData().add(series);
        }
        return chart;
    }

    // ── Batch histogram assembly ─────────────────────────────────────────────

    private record BatchHists(List<String> batches, Map<String, long[][]> before, Map<String, long[][]> after) {}

    /** Pool per-image histograms by batch (before) and by shifted histograms (after). */
    private static BatchHists batchHistograms(NormalizerFit fit) {
        int nMarkers = fit.markers().size();
        int nBins = fit.nBins();
        Map<String, long[][]> before = new LinkedHashMap<>();
        Map<String, long[][]> after = new LinkedHashMap<>();
        TreeSet<String> batchSet = new TreeSet<>();
        for (Map.Entry<String, long[][]> e : fit.histByImage().entrySet()) {
            String image = e.getKey();
            String batch = fit.imageToBatch().getOrDefault(image, "(unassigned)");
            batchSet.add(batch);
            long[][] bAcc = before.computeIfAbsent(batch, k -> new long[nMarkers][nBins]);
            long[][] aAcc = after.computeIfAbsent(batch, k -> new long[nMarkers][nBins]);
            int[] shift = fit.shiftByImage().get(image);
            long[][] ih = e.getValue();
            for (int m = 0; m < nMarkers; m++) {
                long[] h = ih[m];
                int s = shift != null ? shift[m] : 0;
                for (int b = 0; b < nBins; b++) {
                    bAcc[m][b] += h[b];
                    int nb = b - s; // correction shifts log DOWN by s bins
                    if (nb >= 0 && nb < nBins) {
                        aAcc[m][nb] += h[b];
                    }
                }
            }
        }
        return new BatchHists(new ArrayList<>(batchSet), before, after);
    }

    /** SD across batches of each batch's median log-intensity for one marker. */
    private static double spreadOfBatchMedians(
            Map<String, long[][]> hists, int marker, List<String> batches, NormalizerFit fit, int markerForRange) {
        double logMin = fit.logMin()[markerForRange];
        double logMax = fit.logMax()[markerForRange];
        int nBins = fit.nBins();
        double inc = nBins > 1 && logMax > logMin ? (logMax - logMin) / (nBins - 1) : 1.0;
        List<Double> medians = new ArrayList<>();
        for (String batch : batches) {
            long[][] perMarker = hists.get(batch);
            if (perMarker == null) {
                continue;
            }
            double medBin = medianBin(perMarker[marker]);
            if (!Double.isNaN(medBin)) {
                medians.add(logMin + medBin * inc);
            }
        }
        if (medians.size() < 2) {
            return 0.0;
        }
        double mean = 0;
        for (double v : medians) {
            mean += v;
        }
        mean /= medians.size();
        double var = 0;
        for (double v : medians) {
            var += (v - mean) * (v - mean);
        }
        return Math.sqrt(var / medians.size());
    }

    /** Interpolation-free median bin of a count histogram; {@code NaN} if empty. */
    private static double medianBin(long[] h) {
        long total = 0;
        for (long v : h) {
            total += v;
        }
        if (total == 0) {
            return Double.NaN;
        }
        long half = total / 2;
        long cum = 0;
        for (int b = 0; b < h.length; b++) {
            cum += h[b];
            if (cum >= half) {
                return b + 0.5;
            }
        }
        return h.length - 0.5;
    }
}
