package qupath.ext.spclassify.classifier;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import javafx.application.Platform;
import javafx.beans.property.DoubleProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.io.BinaryClassifierRegistry;
import qupath.ext.spclassify.io.ProjectStateManager;
import qupath.ext.spclassify.model.CellFeatureExtractor;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.projects.Project;

/**
 * Applies multiple trained binary classifiers to all cell detections and assigns
 * a composite QuPath PathClass per cell.
 */
public class CompositeClassifier {

    private static final Logger logger = LoggerFactory.getLogger(CompositeClassifier.class);

    private final DoubleProperty progress = new SimpleDoubleProperty(0.0);
    private final StringProperty status = new SimpleStringProperty("");

    public DoubleProperty progressProperty() {
        return progress;
    }

    public StringProperty statusProperty() {
        return status;
    }

    private void updateStatus(String msg, double pct) {
        Platform.runLater(() -> {
            status.set(msg);
            progress.set(pct);
        });
    }

    private static final class MarkerPrediction {
        private final String markerName;
        private final float[] positiveProbs;

        private MarkerPrediction(String markerName, float[] positiveProbs) {
            this.markerName = markerName;
            this.positiveProbs = positiveProbs;
        }
    }

    /**
     * Legacy marker-list API retained for backward compatibility.
     * Equivalent to {@link #apply(ImageData, List, Project, boolean, Consumer)}
     * with {@code mergeWithPrimary = false}.
     */
    public int apply(ImageData<?> imageData, List<String> markerNames, Project<?> project, Consumer<String> log)
            throws Exception {
        return apply(imageData, markerNames, project, false, log);
    }

    /**
     * Apply binary classifiers and assign each cell a composite class.
     *
     * @param mergeWithPrimary if true, each cell's existing PathClass is preserved as
     *                         the leading segment of the composite class name
     *                         (e.g. {@code "Tumor:CD3+:CD8-"}) and the composite
     *                         class colour is taken from that primary class so the
     *                         viewer keeps the original multiclass colouring.
     *                         Cells with no current class fall back to the
     *                         binary-only label.
     */
    public int apply(
            ImageData<?> imageData,
            List<String> markerNames,
            Project<?> project,
            boolean mergeWithPrimary,
            Consumer<String> log)
            throws Exception {

        Consumer<String> out = log != null ? log : s -> {};

        List<PathObject> detectionList =
                new ArrayList<>(imageData.getHierarchy().getDetectionObjects());
        if (detectionList.isEmpty()) {
            out.accept("No detections found in current image.");
            return 0;
        }

        // Snapshot each cell's current primary class BEFORE any reassignment.
        List<PathClass> primaryClasses = null;
        if (mergeWithPrimary) {
            primaryClasses = new ArrayList<>(detectionList.size());
            for (PathObject det : detectionList) {
                primaryClasses.add(det.getPathClass());
            }
        }

        List<String> sorted = new ArrayList<>(markerNames);
        Collections.sort(sorted);

        updateStatus("Starting composite classification...", 0.0);
        out.accept("Classifying " + detectionList.size() + " cells using " + sorted.size() + " markers"
                + (mergeWithPrimary ? " (merged with primary classification)..." : "..."));

        List<String> activeMarkers = new ArrayList<>();
        List<float[]> markerPosProbs = new ArrayList<>();

        for (int mi = 0; mi < sorted.size(); mi++) {
            String markerName = sorted.get(mi);
            MarkerPrediction prediction = predictMarker(detectionList, project, markerName, out, true);
            if (prediction != null) {
                activeMarkers.add(prediction.markerName);
                markerPosProbs.add(prediction.positiveProbs);
                out.accept("Predicted '" + markerName + "'.");
            }
            double pct = 0.2 + 0.6 * ((double) (mi + 1) / Math.max(sorted.size(), 1));
            updateStatus("Predicting " + markerName + "...", pct);
        }

        List<PathObject> classifyObjects = new ArrayList<>(detectionList.size());
        List<PathClass> classifyClasses = new ArrayList<>(detectionList.size());

        if (activeMarkers.isEmpty()) {
            for (PathObject det : detectionList) {
                classifyObjects.add(det);
                classifyClasses.add(null);
            }
        } else {
            for (int i = 0; i < detectionList.size(); i++) {
                StringBuilder sb = new StringBuilder();
                PathClass primary = mergeWithPrimary ? primaryClasses.get(i) : null;
                if (primary != null
                        && primary.getName() != null
                        && !primary.getName().isEmpty()) {
                    sb.append(primary.toString());
                }
                for (int mi = 0; mi < activeMarkers.size(); mi++) {
                    if (sb.length() > 0) {
                        sb.append(':');
                    }
                    sb.append(activeMarkers.get(mi));
                    sb.append(markerPosProbs.get(mi)[i] >= 0.5f ? '+' : '-');
                }
                PathClass composite = PathClass.fromString(sb.toString());
                if (mergeWithPrimary && primary != null && primary.getColor() != null) {
                    // Force the composite class's colour to match the primary so the
                    // viewer keeps the original multiclass colouring.
                    composite.setColor(primary.getColor());
                }
                classifyObjects.add(detectionList.get(i));
                classifyClasses.add(composite);
            }
        }

        for (int i = 0; i < classifyObjects.size(); i++) {
            classifyObjects.get(i).setPathClass(classifyClasses.get(i));
        }

        Platform.runLater(() -> imageData
                .getHierarchy()
                .fireObjectClassificationsChangedEvent(CompositeClassifier.this, classifyObjects));

        updateStatus("Complete", 1.0);
        out.accept("Composite classification complete: " + detectionList.size() + " cells.");
        return detectionList.size();
    }

    /**
     * Apply a named marker-polarity rule to the current image.
     * Only matched cells are updated unless clearUnmatched is true.
     */
    public int applyRule(
            ImageData<?> imageData, CompositeClassificationRule rule, Project<?> project, Consumer<String> log)
            throws Exception {
        return applyRule(imageData, rule, project, false, log);
    }

    /**
     * Apply a named marker-polarity rule to the current image.
     *
     * @param clearUnmatched if true, unmatched cells are assigned null PathClass
     */
    public int applyRule(
            ImageData<?> imageData,
            CompositeClassificationRule rule,
            Project<?> project,
            boolean clearUnmatched,
            Consumer<String> log)
            throws Exception {

        if (rule == null) {
            throw new IllegalArgumentException("Composite rule must not be null");
        }

        Consumer<String> out = log != null ? log : s -> {};
        List<PathObject> detectionList =
                new ArrayList<>(imageData.getHierarchy().getDetectionObjects());
        if (detectionList.isEmpty()) {
            out.accept("No detections found in current image.");
            return 0;
        }

        List<CompositeClassificationRule.MarkerCondition> conditions = rule.conditions();
        updateStatus("Starting rule application: " + rule.name(), 0.0);
        out.accept("Applying rule '" + rule.name() + "' (" + rule.expression() + ") to " + detectionList.size()
                + " cells...");

        List<MarkerPrediction> predictions = new ArrayList<>(conditions.size());
        for (int i = 0; i < conditions.size(); i++) {
            CompositeClassificationRule.MarkerCondition condition = conditions.get(i);
            MarkerPrediction prediction = predictMarker(detectionList, project, condition.markerName(), out, false);
            if (prediction == null) {
                throw new IllegalStateException("Required marker not trained: " + condition.markerName());
            }
            predictions.add(prediction);

            double pct = 0.1 + 0.6 * ((double) (i + 1) / Math.max(conditions.size(), 1));
            updateStatus("Evaluating " + condition.markerName() + "...", pct);
        }

        PathClass matchClass = PathClass.fromString(rule.expression());
        List<PathObject> changedObjects = new ArrayList<>();
        List<PathClass> changedClasses = new ArrayList<>();
        int matched = 0;

        for (int cellIndex = 0; cellIndex < detectionList.size(); cellIndex++) {
            boolean isMatch = true;

            for (int ci = 0; ci < conditions.size(); ci++) {
                CompositeClassificationRule.MarkerCondition condition = conditions.get(ci);
                float posProb = predictions.get(ci).positiveProbs[cellIndex];
                boolean isPositive = posProb >= 0.5f;
                boolean expectsPositive = condition.polarity() == CompositeClassificationRule.Polarity.POSITIVE;
                if (isPositive != expectsPositive) {
                    isMatch = false;
                    break;
                }
            }

            if (isMatch) {
                matched++;
                changedObjects.add(detectionList.get(cellIndex));
                changedClasses.add(matchClass);
            } else if (clearUnmatched) {
                changedObjects.add(detectionList.get(cellIndex));
                changedClasses.add(null);
            }
        }

        for (int i = 0; i < changedObjects.size(); i++) {
            changedObjects.get(i).setPathClass(changedClasses.get(i));
        }

        if (!changedObjects.isEmpty()) {
            Platform.runLater(() -> imageData
                    .getHierarchy()
                    .fireObjectClassificationsChangedEvent(CompositeClassifier.this, changedObjects));
        }

        updateStatus("Complete", 1.0);
        out.accept("Rule application complete: matched " + matched + " of " + detectionList.size() + " cells.");
        return matched;
    }

    /**
     * Legacy marker-list batch API retained for backward compatibility.
     * Equivalent to {@link #batch(Project, List, List, boolean, Consumer)} with
     * {@code mergeWithPrimary = false}.
     */
    public Map<String, String> batch(
            Project<?> project, List<String> imageNames, List<String> markerNames, Consumer<String> log)
            throws Exception {
        return batch(project, imageNames, markerNames, false, log);
    }

    /**
     * Apply binary classifiers to a list of project images, optionally merging
     * the result with each cell's existing primary classification.
     *
     * @see #apply(ImageData, List, Project, boolean, Consumer)
     */
    @SuppressWarnings("unchecked")
    public Map<String, String> batch(
            Project<?> project,
            List<String> imageNames,
            List<String> markerNames,
            boolean mergeWithPrimary,
            Consumer<String> log)
            throws Exception {

        Consumer<String> out = log != null ? log : s -> {};
        Map<String, String> results = new LinkedHashMap<>();
        int total = imageNames.size();

        for (int idx = 0; idx < total; idx++) {
            String imageName = imageNames.get(idx);
            updateStatus("Processing " + imageName + "...", (double) idx / Math.max(total, 1));

            var typedProject = (Project<BufferedImage>) (Project<?>) project;
            var entryOpt = typedProject.getImageList().stream()
                    .filter(e -> e.getImageName().equals(imageName))
                    .findFirst();

            if (entryOpt.isEmpty()) {
                results.put(imageName, "Skipped: image not found in project");
                out.accept("Skipped '" + imageName + "': not found in project.");
                continue;
            }

            var entry = entryOpt.get();
            try {
                var imageData = entry.readImageData();
                if (imageData == null) {
                    results.put(imageName, "Skipped: could not read image data");
                    out.accept("Skipped '" + imageName + "': null imageData.");
                    continue;
                }

                int count = apply(imageData, markerNames, project, mergeWithPrimary, out);
                entry.saveImageData(imageData);
                results.put(imageName, "Classified " + count + " cells");
                out.accept("'" + imageName + "': Classified " + count + " cells.");

            } catch (Exception ex) {
                results.put(imageName, "Error: " + ex.getMessage());
                out.accept("Error processing '" + imageName + "': " + ex.getMessage());
                logger.warn("CompositeClassifier batch error for '{}': {}", imageName, ex.getMessage(), ex);
            }
        }

        updateStatus("Batch complete", 1.0);
        return results;
    }

    /**
     * Apply a named rule to multiple project images.
     */
    @SuppressWarnings("unchecked")
    public Map<String, String> batchApplyRule(
            Project<?> project,
            List<String> imageNames,
            CompositeClassificationRule rule,
            boolean clearUnmatched,
            Consumer<String> log)
            throws Exception {

        Consumer<String> out = log != null ? log : s -> {};
        Map<String, String> results = new LinkedHashMap<>();
        int total = imageNames.size();

        for (int idx = 0; idx < total; idx++) {
            String imageName = imageNames.get(idx);
            updateStatus("Processing " + imageName + "...", (double) idx / Math.max(total, 1));

            var typedProject = (Project<BufferedImage>) (Project<?>) project;
            var entryOpt = typedProject.getImageList().stream()
                    .filter(e -> e.getImageName().equals(imageName))
                    .findFirst();

            if (entryOpt.isEmpty()) {
                results.put(imageName, "Skipped: image not found in project");
                out.accept("Skipped '" + imageName + "': not found in project.");
                continue;
            }

            var entry = entryOpt.get();
            try {
                var imageData = entry.readImageData();
                if (imageData == null) {
                    results.put(imageName, "Skipped: could not read image data");
                    out.accept("Skipped '" + imageName + "': null imageData.");
                    continue;
                }

                int matched = applyRule(imageData, rule, project, clearUnmatched, out);
                entry.saveImageData(imageData);
                results.put(imageName, "Matched " + matched + " cells");
                out.accept("'" + imageName + "': Matched " + matched + " cells.");

            } catch (Exception ex) {
                results.put(imageName, "Error: " + ex.getMessage());
                out.accept("Error processing '" + imageName + "': " + ex.getMessage());
                logger.warn("CompositeClassifier batchApplyRule error for '{}': {}", imageName, ex.getMessage(), ex);
            }
        }

        updateStatus("Batch complete", 1.0);
        return results;
    }

    private MarkerPrediction predictMarker(
            List<PathObject> detectionList,
            Project<?> project,
            String markerName,
            Consumer<String> out,
            boolean allowSkip)
            throws Exception {

        String sanitized = BinaryClassifierRegistry.sanitizeMarkerName(markerName);
        ProjectStateManager.SavedState state = ProjectStateManager.loadBinaryState(project, sanitized);

        if (state == null
                || state.xgboostModelBase64 == null
                || state.featureNames == null
                || state.featureNames.isEmpty()) {
            if (allowSkip) {
                out.accept("Skipping '" + markerName + "' - not trained or missing state.");
                return null;
            }
            throw new IllegalStateException("Marker '" + markerName + "' is not trained or missing state.");
        }

        List<String> classNames = state.classNames;
        if (classNames == null || classNames.size() < 2) {
            if (allowSkip) {
                out.accept("Skipping '" + markerName + "' - invalid class list.");
                return null;
            }
            throw new IllegalStateException("Marker '" + markerName + "' has invalid class list.");
        }

        byte[] xgbBytes = ProjectStateManager.decodeXGBoostModel(state);
        byte[] lgbBytes = ProjectStateManager.decodeLightGBMModel(state);
        if (xgbBytes == null) {
            if (allowSkip) {
                out.accept("Skipping '" + markerName + "' - no XGBoost model bytes.");
                return null;
            }
            throw new IllegalStateException("Marker '" + markerName + "' has no XGBoost model bytes.");
        }

        CellFeatureExtractor extractor = new CellFeatureExtractor(state.featureNames);
        float[] flatData = extractor.extractMatrix(detectionList);
        int nSamples = detectionList.size();
        int nFeatures = extractor.getNumFeatures();

        XGBoostModel xgbModel = new XGBoostModel();
        xgbModel.loadFromBytes(xgbBytes, classNames, state.featureNames);
        float[][] xgbProbs = xgbModel.predictProba(flatData, nSamples, nFeatures);

        float[][] lgbProbs = null;
        if (lgbBytes != null) {
            LightGBMModel lgbModel = new LightGBMModel();
            lgbModel.loadFromBytes(lgbBytes, classNames, state.featureNames);
            lgbProbs = lgbModel.predictProba(flatData, nSamples, nFeatures);
        }

        int posIdx = 0;
        for (int ci = 0; ci < classNames.size(); ci++) {
            if (classNames.get(ci).endsWith("+")) {
                posIdx = ci;
                break;
            }
        }

        float[] posProbs = new float[nSamples];
        for (int i = 0; i < nSamples; i++) {
            float xgbPos = xgbProbs[i][Math.min(posIdx, xgbProbs[i].length - 1)];
            float lgbPos = (lgbProbs != null) ? lgbProbs[i][Math.min(posIdx, lgbProbs[i].length - 1)] : xgbPos;
            posProbs[i] = (lgbProbs != null) ? (xgbPos + lgbPos) / 2.0f : xgbPos;
        }

        return new MarkerPrediction(markerName, posProbs);
    }
}
