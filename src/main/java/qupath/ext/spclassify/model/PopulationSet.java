package qupath.ext.spclassify.model;

import java.util.*;
import java.util.stream.Collectors;

/**
 * A named collection of {@link CellPrediction}s — one of the four prediction
 * population sets produced after training:
 * <ul>
 *   <li>{@code Pred_MDL1} — model 1 (XGBoost) predictions only</li>
 *   <li>{@code Pred_MDL2} — model 2 (LightGBM) predictions only</li>
 *   <li>{@code Pred_AVG} — averaged probabilities from both models</li>
 *   <li>{@code Pred_ALL} — both predictions when they disagree, one when they agree</li>
 * </ul>
 */
public class PopulationSet {

    private final String name;
    private final Map<String, CellPrediction> predictions; // cellId → prediction

    public PopulationSet(String name) {
        this.name = name;
        this.predictions = new LinkedHashMap<>();
    }

    /** Add or replace a prediction for a cell. */
    public void put(String cellId, CellPrediction prediction) {
        predictions.put(cellId, prediction);
    }

    /** Get the prediction for a specific cell, or null. */
    public CellPrediction get(String cellId) {
        return predictions.get(cellId);
    }

    /** @return unmodifiable view of all predictions */
    public Map<String, CellPrediction> getAll() {
        return Collections.unmodifiableMap(predictions);
    }

    /** @return total number of cells in this population set */
    public int size() {
        return predictions.size();
    }

    /** @return all cells where the two models disagreed */
    public List<CellPrediction> getDisagreements() {
        return predictions.values().stream()
                .filter(CellPrediction::isDisagreement)
                .collect(Collectors.toList());
    }

    /** @return count of disagreement cells */
    public long getDisagreementCount() {
        return predictions.values().stream()
                .filter(CellPrediction::isDisagreement)
                .count();
    }

    /**
     * Get all predictions where model 1 predicted a specific class.
     *
     * @param className the class to filter by
     * @return list of matching predictions
     */
    public List<CellPrediction> getByModel1Label(String className) {
        return predictions.values().stream()
                .filter(p -> className.equals(p.getModel1Label()))
                .collect(Collectors.toList());
    }

    /**
     * Get all predictions where model 2 predicted a specific class.
     *
     * @param className the class to filter by
     * @return list of matching predictions
     */
    public List<CellPrediction> getByModel2Label(String className) {
        return predictions.values().stream()
                .filter(p -> className.equals(p.getModel2Label()))
                .collect(Collectors.toList());
    }

    /**
     * Get all predictions where the averaged label is a specific class.
     *
     * @param className the class to filter by
     * @return list of matching predictions
     */
    public List<CellPrediction> getByAvgLabel(String className) {
        return predictions.values().stream()
                .filter(p -> className.equals(p.avgLabel()))
                .collect(Collectors.toList());
    }

    /** @return count of cells per model 1 predicted class */
    public Map<String, Long> getModel1Counts() {
        return predictions.values().stream()
                .collect(Collectors.groupingBy(
                        CellPrediction::getModel1Label, LinkedHashMap::new, Collectors.counting()));
    }

    /** @return count of cells per model 2 predicted class */
    public Map<String, Long> getModel2Counts() {
        return predictions.values().stream()
                .collect(Collectors.groupingBy(
                        CellPrediction::getModel2Label, LinkedHashMap::new, Collectors.counting()));
    }

    /** @return count of cells per averaged predicted class */
    public Map<String, Long> getAvgCounts() {
        return predictions.values().stream()
                .collect(Collectors.groupingBy(CellPrediction::avgLabel, LinkedHashMap::new, Collectors.counting()));
    }

    public String getName() {
        return name;
    }

    @Override
    public String toString() {
        return "PopulationSet[" + name + ", " + predictions.size() + " cells, " + getDisagreementCount()
                + " disagreements]";
    }
}
