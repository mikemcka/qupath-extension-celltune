package qupath.ext.spclassify.model;

import java.util.*;

/**
 * Configurable per-feature normalisation applied during feature extraction.
 * <p>
 * Supported transforms:
 * <ul>
 *   <li>{@link Transform#NONE} — raw value (default)</li>
 *   <li>{@link Transform#ARCSINH} — {@code arcsinh(x / cofactor)}, common in cytometry</li>
 *   <li>{@link Transform#SQRT} — {@code sqrt(max(0, x))}</li>
 * </ul>
 * Usage: configure which features should be transformed, then pass this to
 * {@link CellFeatureExtractor} to apply transforms during row/matrix extraction.
 */
public class FeatureNormalizer {
    /** Serialize to a map: feature name → transform name. */
    public Map<String, String> toTransformMap() {
        Map<String, String> out = new LinkedHashMap<>();
        for (var e : transforms.entrySet()) {
            if (e.getValue() != null && e.getValue() != Transform.NONE)
                out.put(e.getKey(), e.getValue().name());
        }
        return out;
    }

    /** Restore transforms from a map: feature name → transform name. */
    public void fromTransformMap(Map<String, String> map) {
        transforms.clear();
        if (map == null) return;
        for (var e : map.entrySet()) {
            try {
                Transform t = Transform.valueOf(e.getValue());
                if (t != Transform.NONE) transforms.put(e.getKey(), t);
            } catch (Exception ignore) {
            }
        }
    }

    public enum Transform {
        NONE("None"),
        ARCSINH("arcsinh"),
        SQRT("sqrt");

        private final String displayName;

        Transform(String displayName) {
            this.displayName = displayName;
        }

        public String getDisplayName() {
            return displayName;
        }

        @Override
        public String toString() {
            return displayName;
        }
    }

    private final Map<String, Transform> transforms;
    private double arcsinhCofactor;

    public FeatureNormalizer() {
        this.transforms = new LinkedHashMap<>();
        this.arcsinhCofactor = 1.0; // default cofactor for arcsinh
    }

    /**
     * Set the transform for a specific feature.
     *
     * @param featureName the measurement name
     * @param transform   the transform to apply
     */
    public void setTransform(String featureName, Transform transform) {
        if (transform == Transform.NONE) {
            transforms.remove(featureName);
        } else {
            transforms.put(featureName, transform);
        }
    }

    /**
     * Set the same transform for multiple features at once.
     */
    public void setTransform(Collection<String> featureNames, Transform transform) {
        for (String name : featureNames) {
            setTransform(name, transform);
        }
    }

    /** Get the transform configured for a feature (NONE if not set). */
    public Transform getTransform(String featureName) {
        return transforms.getOrDefault(featureName, Transform.NONE);
    }

    /** @return the arcsinh cofactor (divisor applied before arcsinh) */
    public double getArcsinhCofactor() {
        return arcsinhCofactor;
    }

    /** Set the arcsinh cofactor (divisor: {@code arcsinh(x / cofactor)}). */
    public void setArcsinhCofactor(double cofactor) {
        if (cofactor <= 0) throw new IllegalArgumentException("Cofactor must be positive");
        this.arcsinhCofactor = cofactor;
    }

    /** @return true if any features have a non-NONE transform */
    public boolean hasTransforms() {
        return !transforms.isEmpty();
    }

    /** @return unmodifiable view of all configured transforms */
    public Map<String, Transform> getAllTransforms() {
        return Collections.unmodifiableMap(transforms);
    }

    /**
     * Apply the configured transform to a single feature value.
     *
     * @param featureName the feature name (to look up the transform)
     * @param value       the raw measurement value
     * @return the transformed value
     */
    public float apply(String featureName, float value) {
        Transform t = transforms.get(featureName);
        if (t == null) return value;
        return switch (t) {
            case NONE -> value;
            case ARCSINH ->
                (float) Math.log(
                        value / arcsinhCofactor + Math.sqrt(value * value / (arcsinhCofactor * arcsinhCofactor) + 1));
            case SQRT -> (float) Math.sqrt(Math.max(0, value));
        };
    }

    /** Clear all transforms. */
    public void clear() {
        transforms.clear();
    }
}
