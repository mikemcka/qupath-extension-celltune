package qupath.ext.spclassify.model;

import java.util.List;
import java.util.Objects;

/**
 * Holds the dual-model prediction results for a single cell.
 * <p>
 * After training, every cell gets a {@code CellPrediction} recording what each
 * model predicted and the full probability vectors. The key method
 * {@link #isDisagreement()} flags cells where the two models disagree — these
 * are candidates for human review.
 */
public class CellPrediction {

    private final String cellId;
    private final String model1Label;
    private final String model2Label;
    private final float[] model1Probs;
    private final float[] model2Probs;
    private final List<String> classNames;

    /**
     * @param cellId      unique identifier for the cell (typically PathObject ID)
     * @param model1Label predicted class name from model 1 (XGBoost)
     * @param model2Label predicted class name from model 2 (LightGBM)
     * @param model1Probs per-class probability vector from model 1
     * @param model2Probs per-class probability vector from model 2
     * @param classNames  ordered list of class names matching the probability indices
     */
    public CellPrediction(
            String cellId,
            String model1Label,
            String model2Label,
            float[] model1Probs,
            float[] model2Probs,
            List<String> classNames) {
        this.cellId = Objects.requireNonNull(cellId);
        this.model1Label = Objects.requireNonNull(model1Label);
        this.model2Label = Objects.requireNonNull(model2Label);
        this.model1Probs = model1Probs.clone();
        this.model2Probs = model2Probs.clone();
        this.classNames = List.copyOf(classNames);
    }

    /** @return true if model 1 and model 2 predicted different classes */
    public boolean isDisagreement() {
        return !model1Label.equals(model2Label);
    }

    /**
     * Returns the class with the highest average probability across both models.
     * This corresponds to the "Pred_AVG" population set in CellTune.
     *
     * @return the class name with highest averaged probability
     */
    public String avgLabel() {
        int nClasses = classNames.size();
        int bestIdx = 0;
        float bestAvg = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < nClasses; i++) {
            float avg = (model1Probs[i] + model2Probs[i]) / 2f;
            if (avg > bestAvg) {
                bestAvg = avg;
                bestIdx = i;
            }
        }
        return classNames.get(bestIdx);
    }

    /**
     * Returns the label for Pred_ALL: if both models agree, returns that label.
     * If they disagree, returns "model1Label/model2Label" to flag the disagreement.
     *
     * @return combined prediction label
     */
    public String allLabel() {
        if (!isDisagreement()) {
            return model1Label;
        }
        return model1Label + "/" + model2Label;
    }

    /** @return the maximum probability from model 1 (confidence of its prediction) */
    public float model1Confidence() {
        return model1Probs[classIndex(model1Label)];
    }

    /** @return the maximum probability from model 2 (confidence of its prediction) */
    public float model2Confidence() {
        return model2Probs[classIndex(model2Label)];
    }

    private int classIndex(String label) {
        int idx = classNames.indexOf(label);
        return idx >= 0 ? idx : 0;
    }

    public String getCellId() {
        return cellId;
    }

    public String getModel1Label() {
        return model1Label;
    }

    public String getModel2Label() {
        return model2Label;
    }

    public float[] getModel1Probs() {
        return model1Probs.clone();
    }

    public float[] getModel2Probs() {
        return model2Probs.clone();
    }

    public List<String> getClassNames() {
        return classNames;
    }
}
