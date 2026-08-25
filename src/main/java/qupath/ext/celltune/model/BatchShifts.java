package qupath.ext.celltune.model;

import java.util.List;
import java.util.Map;

/**
 * Persisted, consumable form of a UniFORM batch-normalization fit: the per-{@code (image,
 * marker)} multiplicative scales plus the metadata needed to reproduce and interpret them.
 * Stored as {@code <project>/celltune/batch-shifts.json} (see {@code BatchNormPersistence})
 * and reloaded by the clustering "use batch-corrected values" path.
 *
 * <p>A plain field-holder (not a record) so Gson can deserialize it on any bundled Gson
 * version. Values are plain doubles — no Base64. Everything not in {@link #markers} (e.g.
 * foundation-model embeddings) is untouched and carries a scale of {@code 1.0}.
 */
public final class BatchShifts {

    /** Schema version for forward compatibility. */
    public int version;
    /** {@code "PER_IMAGE"} or {@code "PER_BATCH"}. */
    public String mode;
    /** Histogram bins used at fit time. */
    public int nBins;
    /** Corrected marker feature names, in the column order of {@link #scaleByImage} arrays. */
    public List<String> markers;
    /** image-name → batch label (as resolved at fit time). */
    public Map<String, String> imageToBatch;
    /** image-name → per-marker multiplicative scale ({@code corrected = raw · scale}). */
    public Map<String, double[]> scaleByImage;
    /** per-marker reference group label (image or batch), aligned to {@link #markers}. */
    public List<String> refGroupByMarker;

    /** Gson needs a no-arg constructor. */
    public BatchShifts() {}

    /**
     * Multiplicative gain for a given image and measurement, resolved by <b>channel</b>:
     * the gain was fitted per (image, channel), so any statistic of a corrected channel
     * (Mean, Median, percentiles, Std, erosion/expansion/environment bins) gets the same
     * factor — even stats that were not themselves in the fit's selection. Returns
     * {@code 1.0} (no-op) when the channel was not corrected or the image was not in the fit.
     */
    public double scaleFor(String image, String feature) {
        if (markers == null || scaleByImage == null) {
            return 1.0;
        }
        double[] scales = scaleByImage.get(image);
        if (scales == null) {
            return 1.0;
        }
        String channel = BatchNormalizerCohort.channelOf(feature);
        for (int i = 0; i < markers.size() && i < scales.length; i++) {
            if (BatchNormalizerCohort.channelOf(markers.get(i)).equals(channel)) {
                double s = scales[i];
                return (Double.isNaN(s) || Double.isInfinite(s) || s <= 0) ? 1.0 : s;
            }
        }
        return 1.0;
    }

    /**
     * Per-feature gain array for {@code image}, aligned to {@code featureNames} — a
     * {@code CellFeatureExtractor} multiplies each measurement by this before any transform.
     * Uncorrected channels get {@code 1.0}. Precomputes a channel→scale map once so the
     * per-feature lookup is cheap even for wide feature sets.
     */
    public double[] scaleArray(String image, List<String> featureNames) {
        double[] out = new double[featureNames.size()];
        java.util.Map<String, Double> byChannel = new java.util.HashMap<>();
        double[] scales = scaleByImage == null ? null : scaleByImage.get(image);
        if (scales != null && markers != null) {
            for (int i = 0; i < markers.size() && i < scales.length; i++) {
                double s = scales[i];
                if (!(Double.isNaN(s) || Double.isInfinite(s) || s <= 0)) {
                    byChannel.putIfAbsent(BatchNormalizerCohort.channelOf(markers.get(i)), s);
                }
            }
        }
        for (int j = 0; j < out.length; j++) {
            out[j] = byChannel.getOrDefault(BatchNormalizerCohort.channelOf(featureNames.get(j)), 1.0);
        }
        return out;
    }

    /** Whether this fit has any usable per-image scales. */
    public boolean isEmpty() {
        return markers == null || markers.isEmpty() || scaleByImage == null || scaleByImage.isEmpty();
    }
}
