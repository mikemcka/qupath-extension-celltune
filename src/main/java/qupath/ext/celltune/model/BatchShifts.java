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
     * Multiplicative scale for a given image and marker feature name; {@code 1.0} (no-op)
     * when the marker was not corrected or the image was not in the fit.
     */
    public double scaleFor(String image, String markerFeature) {
        if (markers == null || scaleByImage == null) {
            return 1.0;
        }
        int idx = markers.indexOf(markerFeature);
        if (idx < 0) {
            return 1.0;
        }
        double[] scales = scaleByImage.get(image);
        if (scales == null || idx >= scales.length) {
            return 1.0;
        }
        double s = scales[idx];
        return (Double.isNaN(s) || Double.isInfinite(s) || s <= 0) ? 1.0 : s;
    }

    /** Whether this fit has any usable per-image scales. */
    public boolean isEmpty() {
        return markers == null || markers.isEmpty() || scaleByImage == null || scaleByImage.isEmpty();
    }
}
