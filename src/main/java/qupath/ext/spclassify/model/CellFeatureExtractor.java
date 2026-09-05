package qupath.ext.spclassify.model;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjectTools;

/**
 * Reads QuPath cell measurements into flat float arrays suitable for ML training/inference.
 * <p>
 * A single instance holds a fixed list of feature names (column ordering) so that
 * every call to {@link #extractRow(PathObject)} produces a vector with the same layout.
 * Both XGBoost and LightGBM must share the same {@code CellFeatureExtractor} to guarantee
 * identical feature ordering.
 */
public class CellFeatureExtractor {

    private final List<String> featureNames;
    private FeatureNormalizer normalizer;
    // Optional per-image UniFORM batch-correction gain, aligned to featureNames (null = none).
    // Applied to the raw value BEFORE any normalizer transform, so it corrects the measurement
    // itself. Set per image (the extractor is built per image), so streaming correction needs no
    // written columns. See BatchShifts.scaleArray.
    private double[] batchScale;

    /**
     * Create an extractor with a specific ordered list of feature (measurement) names.
     *
     * @param featureNames ordered list of measurement names to extract
     */
    public CellFeatureExtractor(List<String> featureNames) {
        this.featureNames = List.copyOf(featureNames);
    }

    /**
     * Create an extractor with feature names and a normalizer for transforms.
     *
     * @param featureNames ordered list of measurement names to extract
     * @param normalizer   feature normalizer (may be null for no transforms)
     */
    public CellFeatureExtractor(List<String> featureNames, FeatureNormalizer normalizer) {
        this.featureNames = List.copyOf(featureNames);
        this.normalizer = normalizer;
    }

    /** Set or replace the feature normalizer. */
    public void setNormalizer(FeatureNormalizer normalizer) {
        this.normalizer = normalizer;
    }

    /** @return the current normalizer, or null if none set */
    public FeatureNormalizer getNormalizer() {
        return normalizer;
    }

    /**
     * Set the per-image batch-correction gain (aligned to this extractor's feature names),
     * applied to each raw value before the normalizer. Pass {@code null} to disable. A length
     * mismatch is ignored (treated as disabled) to fail safe rather than corrupt features.
     */
    public void setBatchScale(double[] batchScale) {
        this.batchScale = (batchScale != null && batchScale.length == featureNames.size()) ? batchScale : null;
    }

    /** @return true if a per-image batch-correction gain is set. */
    public boolean hasBatchScale() {
        return batchScale != null;
    }

    /**
     * Discover all available measurement names from a collection of detection objects.
     *
     * @param detections detection objects to scan
     * @return ordered list of unique measurement names found across all detections
     */
    public static List<String> discoverFeatureNames(Collection<PathObject> detections) {
        return new ArrayList<>(PathObjectTools.getAvailableFeatures(detections));
    }

    /**
     * Discover all string metadata keys attached to a collection of detection objects.
     * <p>
     * Unlike {@link #discoverFeatureNames}, which returns only the numeric
     * measurement-list columns, this scans each object's metadata map (e.g. the
     * per-cell "CN Class", neighborhood name, or imported "… original class"
     * string values). These keys are string-valued and therefore never appear in
     * {@link PathObjectTools#getAvailableFeatures}.
     *
     * @param detections detection objects to scan
     * @return ordered list of unique metadata keys found across all detections
     */
    public static List<String> discoverMetadataKeys(Collection<PathObject> detections) {
        Set<String> keys = new LinkedHashSet<>();
        for (PathObject det : detections) {
            keys.addAll(det.getMetadata().keySet());
        }
        return new ArrayList<>(keys);
    }

    /**
     * Extract a single cell's measurements as a float array, in the order defined
     * by this extractor's feature name list.
     * <p>
     * Missing or NaN values are replaced with 0f.
     *
     * @param cell the detection object to extract features from
     * @return float array of length {@link #getNumFeatures()}
     */
    public float[] extractRow(PathObject cell) {
        var mlist = cell.getMeasurementList();
        float[] row = new float[featureNames.size()];
        for (int i = 0; i < featureNames.size(); i++) {
            double v = mlist.get(featureNames.get(i));
            row[i] = Double.isNaN(v) ? 0f : (float) v;
            if (batchScale != null) {
                row[i] *= (float) batchScale[i];
            }
            if (normalizer != null) {
                row[i] = normalizer.apply(featureNames.get(i), row[i]);
            }
        }
        return row;
    }

    /**
     * Extract a single cell's raw measurements as a float array, with no normalizer transform.
     * A per-image {@link #setBatchScale batch-correction gain}, if set, IS still applied — it
     * corrects the measurement itself and is not part of the arcsinh/z-score "normalization"
     * the classifier deliberately skips.
     *
     * @param cell the detection object to extract features from
     * @return float array of length {@link #getNumFeatures()}
     */
    public float[] extractRowRaw(PathObject cell) {
        var mlist = cell.getMeasurementList();
        float[] row = new float[featureNames.size()];
        for (int i = 0; i < featureNames.size(); i++) {
            double v = mlist.get(featureNames.get(i));
            row[i] = Double.isNaN(v) ? 0f : (float) v;
            if (batchScale != null) {
                row[i] *= (float) batchScale[i];
            }
        }
        return row;
    }

    /**
     * Extract features from multiple cells into a flat float array (row-major),
     * suitable for constructing a DMatrix or LightGBM Dataset.
     * <p>
     * Extraction is parallelised across available processors for large cell counts.
     *
     * @param cells collection of detection objects
     * @return flat float array of size {@code cells.size() * getNumFeatures()}
     */
    public float[] extractMatrix(Collection<PathObject> cells) {
        int nFeatures = featureNames.size();
        int nCells = cells.size();
        float[] matrix = new float[nCells * nFeatures];

        // Convert to indexed list for parallel random access
        List<PathObject> cellList = (cells instanceof List) ? (List<PathObject>) cells : new ArrayList<>(cells);

        java.util.stream.IntStream.range(0, nCells).parallel().forEach(i -> {
            var mlist = cellList.get(i).getMeasurementList();
            int offset = i * nFeatures;
            for (int j = 0; j < nFeatures; j++) {
                double v = mlist.get(featureNames.get(j));
                matrix[offset + j] = Double.isNaN(v) ? 0f : (float) v;
                if (batchScale != null) {
                    matrix[offset + j] *= (float) batchScale[j];
                }
                if (normalizer != null) {
                    matrix[offset + j] = normalizer.apply(featureNames.get(j), matrix[offset + j]);
                }
            }
        });

        return matrix;
    }

    /** @return the ordered list of feature names this extractor uses */
    public List<String> getFeatureNames() {
        return featureNames;
    }

    /** @return number of features (columns) in each extracted row */
    public int getNumFeatures() {
        return featureNames.size();
    }
}
