package qupath.ext.spclassify.model;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;

/**
 * Computes a mean marker-intensity heatmap grouped by predicted cell class.
 * <p>
 * Rows are cell classes (the {@link PathClass} assigned to each detection, e.g.
 * the {@code Pred_ALL} prediction shown in the viewer), columns are markers
 * (whole-cell mean intensity measurements such as {@code "CD8: Cell: Mean"}),
 * and each matrix entry is the mean of that marker across all cells of that
 * class. A per-column z-score (standardised across the class rows) is also
 * produced so the heatmap colours highlight which phenotype is relatively high
 * or low for each marker — the same presentation as the reference
 * "mean marker expression per phenotype" plot.
 * <p>
 * The numeric computation is intentionally free of any JavaFX dependency so it
 * can be unit-tested in isolation. Use {@link Accumulator} to feed cells (or raw
 * rows) and call {@link Accumulator#build()} to obtain a {@link Result}.
 */
public final class IntensityHeatmap {

    /** Label used for detections that have no {@link PathClass}. */
    public static final String UNCLASSIFIED = "Unclassified";

    private static final String MARKER_MEAN_SUFFIX = ": Cell: Mean";

    private IntensityHeatmap() {}

    /**
     * Immutable heatmap result.
     *
     * @param classNames    ordered row labels (cell classes)
     * @param markers       ordered column labels (marker display names)
     * @param meanIntensity {@code [class][marker]} mean intensity ({@code NaN}
     *                      when a class has no valid measurement for a marker)
     * @param zScores       {@code [class][marker]} per-column z-scores
     * @param classCounts   number of cells contributing to each class row
     */
    public record Result(
            List<String> classNames,
            List<String> markers,
            double[][] meanIntensity,
            double[][] zScores,
            long[] classCounts) {}

    /**
     * Discover marker columns from a list of measurement names.
     * <p>
     * A marker column is any whole-cell mean of the form
     * {@code "<marker>: Cell: Mean"} whose marker name is a plain channel name
     * (no additional {@code ": "} segments). This deliberately excludes derived
     * measurements such as {@code "Neighbors: Mean: DAPI: Cell: Mean"} or other
     * compartment/aggregated variants, keeping only the raw per-channel cell
     * means. The returned list preserves first-seen order and contains the full
     * measurement names (e.g. {@code "CD8: Cell: Mean"}).
     *
     * @param featureNames available measurement names
     * @return ordered list of plain whole-cell mean measurement names
     */
    public static List<String> discoverMarkerFeatures(Collection<String> featureNames) {
        List<String> out = new ArrayList<>();
        if (featureNames == null) {
            return out;
        }
        for (String name : featureNames) {
            if (name != null && name.endsWith(MARKER_MEAN_SUFFIX) && name.length() > MARKER_MEAN_SUFFIX.length()) {
                String label = markerLabel(name);
                // Only plain channel markers — skip derived/aggregated features
                // whose marker name itself contains a ": " segment.
                if (!label.contains(":")) {
                    out.add(name);
                }
            }
        }
        return out;
    }

    /**
     * Strip the {@code ": Cell: Mean"} suffix to produce a compact marker label.
     *
     * @param markerFeatureName full measurement name
     * @return the marker display label (the part before {@code ": Cell: Mean"})
     */
    public static String markerLabel(String markerFeatureName) {
        if (markerFeatureName == null) {
            return "";
        }
        if (markerFeatureName.endsWith(MARKER_MEAN_SUFFIX)
                && markerFeatureName.length() > MARKER_MEAN_SUFFIX.length()) {
            return markerFeatureName.substring(0, markerFeatureName.length() - MARKER_MEAN_SUFFIX.length());
        }
        return markerFeatureName;
    }

    /**
     * Standardise each column of a {@code [class][marker]} mean matrix.
     * <p>
     * For each marker column the population mean and standard deviation are
     * computed across the class rows (ignoring {@code NaN} entries) and every
     * value is converted to {@code (value - mean) / std}. Columns with zero
     * variance (or fewer than two valid rows) yield {@code 0.0} for every row.
     * {@code NaN} inputs remain {@code NaN}.
     *
     * @param means {@code [class][marker]} mean intensity matrix
     * @return a new {@code [class][marker]} matrix of per-column z-scores
     */
    public static double[][] zScoreByColumn(double[][] means) {
        int nRows = means.length;
        int nCols = nRows == 0 ? 0 : means[0].length;
        double[][] z = new double[nRows][nCols];

        for (int j = 0; j < nCols; j++) {
            double sum = 0.0;
            int count = 0;
            for (double[] mean : means) {
                double v = mean[j];
                if (!Double.isNaN(v)) {
                    sum += v;
                    count++;
                }
            }
            if (count < 2) {
                for (int i = 0; i < nRows; i++) {
                    z[i][j] = Double.isNaN(means[i][j]) ? Double.NaN : 0.0;
                }
                continue;
            }
            double mean = sum / count;
            double sqSum = 0.0;
            for (double[] row : means) {
                double v = row[j];
                if (!Double.isNaN(v)) {
                    double d = v - mean;
                    sqSum += d * d;
                }
            }
            double std = Math.sqrt(sqSum / count);
            for (int i = 0; i < nRows; i++) {
                double v = means[i][j];
                if (Double.isNaN(v)) {
                    z[i][j] = Double.NaN;
                } else {
                    z[i][j] = std > 0 ? (v - mean) / std : 0.0;
                }
            }
        }
        return z;
    }

    /**
     * Resolve where ONE image's cells are read from: the LIVE open image when
     * {@code name} matches {@code openName} (disk is never touched in that case),
     * or whatever {@code diskReader} supplies otherwise.
     * <p>
     * The project-combined heatmap pools every project image, and the open one must
     * be pooled from its live hierarchy rather than re-read from its {@code .qpdata}.
     * Classifications applied to the open image stay in memory until QuPath saves it
     * — SP Classify's training and composite-classification runs both write the other
     * project images to disk but leave the open one to QuPath — so re-reading it
     * would pool STALE classes. The visible symptom is a combined heatmap carrying
     * phenotype rows from whichever classifier last wrote that file (e.g. leftover
     * {@code Marker+}/{@code Marker-} rows from an earlier composite run) while the
     * same image's own row set, built from the live hierarchy, shows the current
     * classes.
     * <p>
     * Sibling of {@code CohortClusterModel.selectImageSource}, which applies the same
     * rule to cohort pooling. This one is generic so the UI layer can pass an
     * {@code ImageData<?>}, and it additionally treats a null {@code openData} as "no
     * image open" so a name match can never resolve to a null source.
     *
     * @param name       the image being resolved
     * @param openName   name of the open image (nullable — never matches if null)
     * @param openData   the open image's live data (returned as-is when {@code name} matches)
     * @param diskReader consulted ONLY when {@code name} does not match {@code openName}
     * @param <T>        the image-data type
     * @return the live open data, or the disk reader's result
     */
    public static <T> T selectImageSource(String name, String openName, T openData, Supplier<T> diskReader) {
        if (openData != null && openName != null && openName.equals(name)) {
            return openData;
        }
        return diskReader.get();
    }

    /**
     * Running accumulator that builds an {@link IntensityHeatmap.Result} from one
     * or more batches of cells. Supports combining cells across multiple images
     * (call {@link #add(Collection)} once per image) so a project-wide heatmap
     * reflects true pooled means rather than an average of per-image averages.
     */
    public static final class Accumulator {

        private final List<String> markerFeatures;
        private final List<String> markerLabels;
        private final Map<String, double[]> sums = new LinkedHashMap<>();
        private final Map<String, long[]> counts = new LinkedHashMap<>();
        private final Map<String, Long> classCounts = new LinkedHashMap<>();

        /**
         * @param markerFeatures full mean-measurement names, in column order
         *                       (e.g. {@code "CD8: Cell: Mean"})
         */
        public Accumulator(List<String> markerFeatures) {
            this.markerFeatures = List.copyOf(markerFeatures);
            this.markerLabels = new ArrayList<>(this.markerFeatures.size());
            for (String f : this.markerFeatures) {
                this.markerLabels.add(markerLabel(f));
            }
        }

        /** @return the number of marker columns. */
        public int markerCount() {
            return markerFeatures.size();
        }

        /**
         * Add a batch of detections. Each cell's {@link PathClass} name selects
         * its row; missing or {@code NaN} measurements are skipped per marker.
         *
         * @param cells detections to accumulate
         */
        public void add(Collection<PathObject> cells) {
            if (cells == null) {
                return;
            }
            for (PathObject cell : cells) {
                PathClass pc = cell.getPathClass();
                String className = (pc != null && pc.getName() != null) ? pc.getName() : UNCLASSIFIED;
                var ml = cell.getMeasurementList();
                double[] values = new double[markerFeatures.size()];
                for (int j = 0; j < markerFeatures.size(); j++) {
                    values[j] = ml.get(markerFeatures.get(j));
                }
                addRow(className, values);
            }
        }

        /**
         * Add a single pre-extracted row. Used by tests and by callers that have
         * already read measurements. {@code NaN} values are ignored per marker.
         *
         * @param className    the cell class (row) label
         * @param markerValues one value per marker column
         */
        public void addRow(String className, double[] markerValues) {
            String key = (className == null || className.isBlank()) ? UNCLASSIFIED : className;
            double[] s = sums.computeIfAbsent(key, k -> new double[markerFeatures.size()]);
            long[] c = counts.computeIfAbsent(key, k -> new long[markerFeatures.size()]);
            for (int j = 0; j < markerFeatures.size() && j < markerValues.length; j++) {
                double v = markerValues[j];
                if (!Double.isNaN(v)) {
                    s[j] += v;
                    c[j]++;
                }
            }
            classCounts.merge(key, 1L, Long::sum);
        }

        /**
         * Build the heatmap result. Class rows are sorted alphabetically with
         * {@link #UNCLASSIFIED} forced last.
         *
         * @return the computed heatmap, or an empty result if no cells were added
         */
        public Result build() {
            List<String> classNames = new ArrayList<>(sums.keySet());
            classNames.sort(Comparator.comparing((String s) -> s.equals(UNCLASSIFIED))
                    .thenComparing(Comparator.naturalOrder()));

            int nClasses = classNames.size();
            int nMarkers = markerFeatures.size();
            double[][] means = new double[nClasses][nMarkers];
            long[] classCountArr = new long[nClasses];

            for (int i = 0; i < nClasses; i++) {
                String cls = classNames.get(i);
                double[] s = sums.get(cls);
                long[] c = counts.get(cls);
                classCountArr[i] = classCounts.getOrDefault(cls, 0L);
                for (int j = 0; j < nMarkers; j++) {
                    means[i][j] = c[j] > 0 ? s[j] / c[j] : Double.NaN;
                }
            }

            double[][] z = zScoreByColumn(means);
            return new Result(List.copyOf(classNames), List.copyOf(markerLabels), means, z, classCountArr);
        }
    }
}
