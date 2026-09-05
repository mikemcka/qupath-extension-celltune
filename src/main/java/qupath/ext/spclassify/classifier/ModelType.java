package qupath.ext.spclassify.classifier;

/**
 * Available model types for the dual-model classifier.
 * Any combination of two different types can be used for the disagreement paradigm.
 */
public enum ModelType {
    XGBOOST("XGBoost"),
    LIGHTGBM("LightGBM"),
    RANDOM_FOREST("Random Forest");

    private final String displayName;

    ModelType(String displayName) {
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
