package qupath.ext.spclassify.classifier;

import java.util.List;

/**
 * Immutable snapshot of a trained dual-model classifier's state.
 * <p>
 * Contains serialised model bytes for XGBoost, LightGBM, and/or Random Forest,
 * plus the metadata needed to reconstruct the models and verify feature compatibility.
 * Designed to be persisted via {@link qupath.ext.spclassify.io.ProjectStateManager}.
 */
public class ClassifierState {

    private final String name;
    private final List<String> featureNames;
    private final List<String> classNames;
    private final byte[] xgboostBytes;
    private final byte[] lightgbmBytes;
    private final byte[] rfModel1Bytes;
    private final byte[] rfModel2Bytes;
    private final ModelType model1Type;
    private final ModelType model2Type;

    /**
     * Full constructor supporting all model types.
     */
    public ClassifierState(
            String name,
            List<String> featureNames,
            List<String> classNames,
            byte[] xgboostBytes,
            byte[] lightgbmBytes,
            byte[] rfModel1Bytes,
            byte[] rfModel2Bytes,
            ModelType model1Type,
            ModelType model2Type) {
        this.name = name;
        this.featureNames = List.copyOf(featureNames);
        this.classNames = List.copyOf(classNames);
        this.xgboostBytes = xgboostBytes != null ? xgboostBytes.clone() : null;
        this.lightgbmBytes = lightgbmBytes != null ? lightgbmBytes.clone() : null;
        this.rfModel1Bytes = rfModel1Bytes != null ? rfModel1Bytes.clone() : null;
        this.rfModel2Bytes = rfModel2Bytes != null ? rfModel2Bytes.clone() : null;
        this.model1Type = model1Type;
        this.model2Type = model2Type;
    }

    /**
     * Backward-compatible constructor (XGBoost + LightGBM only).
     */
    public ClassifierState(
            String name,
            List<String> featureNames,
            List<String> classNames,
            byte[] xgboostBytes,
            byte[] lightgbmBytes) {
        this(
                name,
                featureNames,
                classNames,
                xgboostBytes,
                lightgbmBytes,
                null,
                null,
                ModelType.XGBOOST,
                ModelType.LIGHTGBM);
    }

    public String getName() {
        return name;
    }

    public List<String> getFeatureNames() {
        return featureNames;
    }

    public List<String> getClassNames() {
        return classNames;
    }

    public byte[] getXgboostBytes() {
        return xgboostBytes != null ? xgboostBytes.clone() : null;
    }

    public byte[] getLightgbmBytes() {
        return lightgbmBytes != null ? lightgbmBytes.clone() : null;
    }

    public byte[] getRfModel1Bytes() {
        return rfModel1Bytes != null ? rfModel1Bytes.clone() : null;
    }

    public byte[] getRfModel2Bytes() {
        return rfModel2Bytes != null ? rfModel2Bytes.clone() : null;
    }

    public ModelType getModel1Type() {
        return model1Type;
    }

    public ModelType getModel2Type() {
        return model2Type;
    }

    /** @return true if both models have been trained and serialised */
    public boolean isComplete() {
        boolean m1 =
                switch (model1Type) {
                    case XGBOOST -> xgboostBytes != null;
                    case LIGHTGBM -> lightgbmBytes != null;
                    case RANDOM_FOREST -> rfModel1Bytes != null;
                };
        boolean m2 =
                switch (model2Type) {
                    case XGBOOST -> xgboostBytes != null;
                    case LIGHTGBM -> lightgbmBytes != null;
                    case RANDOM_FOREST -> rfModel2Bytes != null;
                };
        return m1 && m2;
    }

    /**
     * Check whether a set of feature names is compatible with this classifier.
     *
     * @param currentFeatures the feature names from the current image
     * @return true if compatible
     */
    public boolean isFeatureCompatible(List<String> currentFeatures) {
        return featureNames.equals(currentFeatures);
    }

    @Override
    public String toString() {
        return "ClassifierState[" + name
                + ", " + featureNames.size() + " features"
                + ", " + classNames.size() + " classes"
                + ", model1=" + model1Type
                + ", model2=" + model2Type
                + ", xgb=" + (xgboostBytes != null ? xgboostBytes.length + "B" : "null")
                + ", lgbm=" + (lightgbmBytes != null ? lightgbmBytes.length + "B" : "null")
                + ", rf1=" + (rfModel1Bytes != null ? rfModel1Bytes.length + "B" : "null")
                + ", rf2=" + (rfModel2Bytes != null ? rfModel2Bytes.length + "B" : "null")
                + "]";
    }
}
