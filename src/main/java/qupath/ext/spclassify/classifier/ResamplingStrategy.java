package qupath.ext.spclassify.classifier;

/**
 * Available resampling strategies for addressing class imbalance
 * in the training data before model fitting.
 */
public enum ResamplingStrategy {
    NONE("None"),
    SMOTE("SMOTE"),
    ADASYN("ADASYN"),
    TOMEK("Tomek links"),
    SMOTE_TOMEK("SMOTE + Tomek"),
    ADASYN_TOMEK("ADASYN + Tomek");

    private final String label;

    ResamplingStrategy(String label) {
        this.label = label;
    }

    /** Display label for UI combo boxes. */
    @Override
    public String toString() {
        return label;
    }
}
