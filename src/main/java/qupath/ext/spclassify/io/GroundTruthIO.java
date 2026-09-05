package qupath.ext.spclassify.io;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.IntStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.model.CellFeatureExtractor;
import qupath.ext.spclassify.model.LabelStore;
import qupath.lib.objects.PathObject;

/**
 * Portable import/export of ground-truth labelled cell populations.
 * <p>
 * Designed for transferring training data between QuPath projects and
 * workstations. Since cell UUIDs are project-specific, this format stores
 * each labelled cell's feature vector alongside its label, so the data can
 * be used directly for training on any project with the same panel.
 * <p>
 * Two formats are supported:
 * <ul>
 *   <li><b>CSV</b> — human-readable, one row per labelled cell.
 *       Columns: {@code Image, Label, CentroidX, CentroidY, Feature1, Feature2, …}</li>
 *   <li><b>JSON</b> — machine-readable, includes metadata (source project,
 *       image name, feature names, class names, timestamp)</li>
 * </ul>
 */
public class GroundTruthIO {

    private static final Logger logger = LoggerFactory.getLogger(GroundTruthIO.class);
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private GroundTruthIO() {} // utility class

    /** Parsed CSV layout supporting both Image-first and Label-first exports. */
    private record CsvLayout(int labelCol, int centroidXCol, int centroidYCol, int firstFeatureCol) {}

    private static CsvLayout parseCsvLayout(String headerLine) {
        String[] cols = headerLine.split(",", -1);
        int labelIdx = -1;
        int xIdx = -1;
        int yIdx = -1;

        for (int i = 0; i < cols.length; i++) {
            String col = cols[i].strip();
            if (col.equalsIgnoreCase("Label")) labelIdx = i;
            else if (col.equalsIgnoreCase("CentroidX")) xIdx = i;
            else if (col.equalsIgnoreCase("CentroidY")) yIdx = i;
        }

        // Fast-path known schemas
        if (cols.length >= 4
                && cols[0].strip().equalsIgnoreCase("Image")
                && cols[1].strip().equalsIgnoreCase("Label")) {
            return new CsvLayout(1, 2, 3, 4);
        }
        if (cols.length >= 3 && cols[0].strip().equalsIgnoreCase("Label")) {
            return new CsvLayout(0, 1, 2, 3);
        }

        // Generic fallback by header names
        if (labelIdx >= 0 && xIdx >= 0 && yIdx >= 0) {
            int firstFeature = Math.max(labelIdx, Math.max(xIdx, yIdx)) + 1;
            return new CsvLayout(labelIdx, xIdx, yIdx, firstFeature);
        }

        return null;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  CSV export/import  — simple, human-editable
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Export labelled cells as a CSV file with feature vectors.
     * <p>
     * Only cells present in the {@code labelStore} are exported.
     * This file can be imported on any project with cells that have the same
     * measurement names.
     *
     * @param outputPath   where to write the CSV
     * @param cells        all detection objects in the image
     * @param labelStore   ground-truth labels
     * @param extractor    feature extractor defining column ordering
     * @param imageName    source image name (written as a header comment)
     * @throws IOException if writing fails
     */
    public static void exportCSV(
            Path outputPath,
            Collection<PathObject> cells,
            LabelStore labelStore,
            CellFeatureExtractor extractor,
            String imageName)
            throws IOException {
        exportCSV(outputPath, cells, labelStore, extractor, imageName, true);
    }

    /**
     * Export labelled cells with configurable raw feature columns.
     *
     * @param includeRaw   include raw feature columns
     */
    public static void exportCSV(
            Path outputPath,
            Collection<PathObject> cells,
            LabelStore labelStore,
            CellFeatureExtractor extractor,
            String imageName,
            boolean includeRaw)
            throws IOException {

        List<String> featureNames = extractor.getFeatureNames();

        // Build cellId → PathObject lookup
        Map<String, PathObject> cellById = new LinkedHashMap<>();
        for (PathObject cell : cells) {
            cellById.put(cell.getID().toString(), cell);
        }

        try (BufferedWriter writer = Files.newBufferedWriter(outputPath, StandardCharsets.UTF_8)) {
            // Comment header with metadata
            writer.write("# SP Classify Ground Truth Export");
            writer.newLine();
            writer.write("# Image: " + (imageName != null ? imageName : "unknown"));
            writer.newLine();
            writer.write("# Exported: " + java.time.LocalDateTime.now());
            writer.newLine();

            // Column header — raw features
            StringBuilder header = new StringBuilder("Image,Label,CentroidX,CentroidY");
            if (includeRaw) {
                for (String feat : featureNames) {
                    header.append(',').append(feat);
                }
            }
            writer.write(header.toString());
            writer.newLine();

            // Collect labelled cells in order for parallel extraction
            List<String> labelledIds = new ArrayList<>();
            List<String> labelledLabels = new ArrayList<>();
            List<PathObject> labelledCells = new ArrayList<>();
            for (var entry : labelStore.getAllLabels().entrySet()) {
                PathObject cell = cellById.get(entry.getKey());
                if (cell != null) {
                    labelledIds.add(entry.getKey());
                    labelledLabels.add(entry.getValue());
                    labelledCells.add(cell);
                }
            }
            int nLabelled = labelledCells.size();

            // Pre-extract features in parallel
            float[][] rawRows = null;
            if (includeRaw) {
                rawRows = new float[nLabelled][];
                float[][] rr = rawRows;
                IntStream.range(0, nLabelled)
                        .parallel()
                        .forEach(i -> rr[i] = extractor.extractRowRaw(labelledCells.get(i)));
            }

            int exported = 0;
            for (int idx = 0; idx < nLabelled; idx++) {
                PathObject cell = labelledCells.get(idx);
                String label = labelledLabels.get(idx);

                var roi = cell.getROI();
                double cx = roi != null ? roi.getCentroidX() : 0;
                double cy = roi != null ? roi.getCentroidY() : 0;

                String imageNameCol = null;
                if (cell.getMeasurementList().containsKey("Image")) {
                    imageNameCol = String.valueOf(cell.getMeasurementList().get("Image"));
                }
                if (imageNameCol == null) imageNameCol = imageName != null ? imageName : "image";
                StringBuilder row = new StringBuilder();
                row.append(imageNameCol);
                row.append(',').append(label);
                row.append(',').append(String.format("%.2f", cx));
                row.append(',').append(String.format("%.2f", cy));
                if (includeRaw) {
                    float[] rawFeatures = rawRows[idx];
                    for (float f : rawFeatures) {
                        row.append(',').append(f);
                    }
                }
                writer.write(row.toString());
                writer.newLine();
                exported++;
            }
            logger.info("Exported {} labelled cells to {}", exported, outputPath);
        }
    }

    /**
     * Import labelled cells from a CSV file into a LabelStore, matching cells
     * by nearest spatial proximity.
     * <p>
     * For each imported row, finds the closest detection cell (by centroid
     * distance) and assigns the label. A maximum distance threshold prevents
     * spurious matches.
     *
     * @param csvPath       path to the CSV file
     * @param cells         detection objects in the current image
     * @param maxDistPixels maximum centroid distance for a match (e.g. 20.0)
     * @return a new LabelStore with matched labels
     * @throws IOException if reading fails
     */
    public static LabelStore importCSVSpatial(Path csvPath, Collection<PathObject> cells, double maxDistPixels)
            throws IOException {

        LabelStore store = new LabelStore("Imported");

        // Pre-index cells by centroid
        List<PathObject> cellList = new ArrayList<>(cells);
        double[] cx = new double[cellList.size()];
        double[] cy = new double[cellList.size()];
        for (int i = 0; i < cellList.size(); i++) {
            var roi = cellList.get(i).getROI();
            cx[i] = roi != null ? roi.getCentroidX() : Double.NaN;
            cy[i] = roi != null ? roi.getCentroidY() : Double.NaN;
        }

        int matched = 0;
        int skipped = 0;
        double maxDist2 = maxDistPixels * maxDistPixels;

        try (BufferedReader reader = Files.newBufferedReader(csvPath, StandardCharsets.UTF_8)) {
            String line;
            CsvLayout layout = null;

            while ((line = reader.readLine()) != null) {
                if (line.startsWith("#") || line.isBlank()) continue;
                if (layout == null) {
                    layout = parseCsvLayout(line);
                    if (layout == null) {
                        throw new IOException("Unsupported ground-truth CSV header format");
                    }
                    continue;
                }

                String[] parts = line.split(",", -1);
                int requiredCol = Math.max(layout.labelCol(), Math.max(layout.centroidXCol(), layout.centroidYCol()));
                if (parts.length <= requiredCol) {
                    skipped++;
                    continue;
                }

                String label = parts[layout.labelCol()].strip();
                if (label.isEmpty()) {
                    skipped++;
                    continue;
                }
                double importX, importY;
                try {
                    importX = Double.parseDouble(parts[layout.centroidXCol()].strip());
                    importY = Double.parseDouble(parts[layout.centroidYCol()].strip());
                } catch (NumberFormatException e) {
                    skipped++;
                    continue;
                }

                // Find nearest cell
                int bestIdx = -1;
                double bestDist2 = Double.MAX_VALUE;
                for (int i = 0; i < cellList.size(); i++) {
                    double dx = cx[i] - importX;
                    double dy = cy[i] - importY;
                    double d2 = dx * dx + dy * dy;
                    if (d2 < bestDist2) {
                        bestDist2 = d2;
                        bestIdx = i;
                    }
                }

                if (bestIdx >= 0 && bestDist2 <= maxDist2) {
                    store.setLabel(cellList.get(bestIdx).getID().toString(), label);
                    matched++;
                } else {
                    skipped++;
                }
            }
        }

        logger.info("Imported {} labels ({} unmatched, maxDist={})", matched, skipped, maxDistPixels);
        return store;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Training-data-only import — no spatial matching needed
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Parsed training row from an imported ground truth file.
     * Contains the label and feature vector but no cell reference.
     */
    public record TrainingRow(String label, float[] features) {}

    /**
     * Import a CSV as raw training data (feature vectors + labels).
     * <p>
     * This is useful when the import image differs from the export image
     * (e.g. different workstation, different project). The feature vectors
     * can be used directly for model training without cell matching.
     *
     * @param csvPath path to the CSV
     * @return list of training rows with labels and feature vectors
     * @throws IOException if reading fails
     */
    public static List<TrainingRow> importCSVAsTrainingData(Path csvPath) throws IOException {
        List<TrainingRow> rows = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(csvPath, StandardCharsets.UTF_8)) {
            String line;
            CsvLayout layout = null;
            int nFeatures = -1;

            while ((line = reader.readLine()) != null) {
                if (line.startsWith("#") || line.isBlank()) continue;
                if (layout == null) {
                    layout = parseCsvLayout(line);
                    if (layout == null) {
                        throw new IOException("Unsupported ground-truth CSV header format");
                    }
                    nFeatures = Math.max(0, line.split(",", -1).length - layout.firstFeatureCol());
                    continue;
                }

                String[] parts = line.split(",", -1);
                if (parts.length <= layout.labelCol()) continue;

                String label = parts[layout.labelCol()].strip();
                if (label.isEmpty()) continue;

                // Read features from first feature column onward
                float[] features = new float[nFeatures];
                for (int i = 0; i < nFeatures && (layout.firstFeatureCol() + i) < parts.length; i++) {
                    try {
                        features[i] = Float.parseFloat(parts[layout.firstFeatureCol() + i].strip());
                    } catch (NumberFormatException e) {
                        features[i] = 0f;
                    }
                }
                rows.add(new TrainingRow(label, features));
            }
        }

        logger.info("Imported {} training rows from {}", rows.size(), csvPath);
        return rows;
    }

    /**
     * Read only the feature names (column headers) from a ground truth CSV,
     * skipping Label, CentroidX and CentroidY.
     *
     * @param csvPath the file to read
     * @return ordered list of feature names
     * @throws IOException if reading fails
     */
    public static List<String> readFeatureNames(Path csvPath) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(csvPath, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("#") || line.isBlank()) continue;
                // First non-comment line is the header
                String[] parts = line.split(",", -1);
                CsvLayout layout = parseCsvLayout(line);
                if (layout == null) return List.of();

                // Skip fixed metadata columns before features
                List<String> names = new ArrayList<>();
                for (int i = layout.firstFeatureCol(); i < parts.length; i++) {
                    names.add(parts[i].strip());
                }
                return names;
            }
        }
        return List.of();
    }
}
