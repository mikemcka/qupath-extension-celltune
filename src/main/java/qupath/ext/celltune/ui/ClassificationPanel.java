package qupath.ext.celltune.ui;

import java.awt.image.BufferedImage;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import qupath.ext.celltune.BatchCorrection;
import qupath.ext.celltune.CellTuneExtension;
import qupath.ext.celltune.classifier.DataPoolingService;
import qupath.ext.celltune.classifier.DualModelClassifier;
import qupath.ext.celltune.classifier.FeaturePruner;
import qupath.ext.celltune.classifier.HyperparameterTuner;
import qupath.ext.celltune.classifier.ModelType;
import qupath.ext.celltune.classifier.ResamplingStrategy;
import qupath.ext.celltune.classifier.UncertaintySampler;
import qupath.ext.celltune.classifier.XGBoostModel;
import qupath.ext.celltune.io.GroundTruthIO;
import qupath.ext.celltune.io.ProjectStateManager;
import qupath.ext.celltune.io.TrainingLogRecorder;
import qupath.ext.celltune.model.*;
import qupath.ext.celltune.model.BatchShifts;
import qupath.ext.celltune.model.FeatureNormalizer;
import qupath.ext.celltune.util.TrainingThreads;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.objects.PathObject;
import qupath.lib.projects.ProjectImageEntry;

/**
 * Main classifier sidebar panel that collects labels, trains both models,
 * and provides access to confusions, sampling, and review.
 * <p>
 * This panel is docked into QuPath's analysis pane and provides the
 * central hub for the entire CellTune workflow.
 */
public class ClassificationPanel extends VBox {

    private static final ResourceBundle STRINGS = ResourceBundle.getBundle("qupath.ext.celltune.ui.strings");

    private final QuPathGUI qupath;

    // ── State (references into extension state) ──
    private LabelStore labelStore;
    private DualModelClassifier classifier;
    private List<String> selectedFeatures;
    private CellTypeTable cellTypeTable;
    private PopulationSet predAll;
    private double[] lastAgreementRates;
    private List<String> lastSampledCellIds;
    private Map<String, String> lastSampledCellImageMap = Map.of();
    private Map<String, List<String>> lastSampledCellAnnotationsMap = Map.of();
    private PopulationSet lastSampledPredictions;
    private List<GroundTruthIO.TrainingRow> importedTrainingRows;
    private List<String> importedTrainingFeatureNames;

    // ── Callbacks ──
    private Consumer<LabelStore> onLabelStoreChanged;
    private Consumer<PopulationSet> onPredAllChanged;
    private Consumer<double[]> onAgreementRatesChanged;
    private Consumer<List<String>> onSampledCellsChanged;
    private Consumer<DualModelClassifier> onClassifierChanged;
    private Supplier<Boolean> autoClassifyCallback;
    private Supplier<List<String>> binaryTargetImagesSupplier;

    // ── UI widgets ──
    private final Label statusLabel = new Label("Ready");
    private final ProgressBar progressBar = new ProgressBar(0);
    private final Button trainButton = new Button();
    private final Button confusionsButton = new Button();
    private final Button metricsButton = new Button();
    private final Button featureImportanceButton = new Button("Feature Importance...");
    private Runnable onFeatureImportance;
    private final Button reviewButton = new Button();
    private final CheckBox currentImageOnlyCheckBox = new CheckBox("Sample current image only");
    private final TextField annotationKeywordField = new TextField();
    private final CheckBox showFeatureImportanceCheckBox =
            new CheckBox("Show top 10 feature importance after training");
    private final Label labelCountLabel = new Label("Labels: 0");
    private final Label predCountLabel = new Label("Predictions: 0");
    private final Label importedCountLabel = new Label("Imported rows: 0");
    private final Button clearImportedButton = new Button("Clear");
    private Runnable onClearImportedData;
    private final Spinner<Integer> roundsSpinner;
    private final Spinner<Integer> depthSpinner;
    private final Spinner<Integer> workersSpinner;
    private final CheckBox poolImagesCheckBox = new CheckBox("Pool labels from all images");
    private final CheckBox enableBalancingCheckBox = new CheckBox("Enable data balancing");
    private final CheckBox restrictToImportedFeaturesCheckBox =
            new CheckBox("Restrict to features shared with imported data");
    private final ComboBox<ResamplingStrategy> advancedResamplingCombo = new ComboBox<>();
    private final ComboBox<ModelType> model1Combo = new ComboBox<>();
    private final ComboBox<ModelType> model2Combo = new ComboBox<>();
    private final CheckBox autoTuneCheckBox = new CheckBox("Auto-tune hyperparameters");
    private final Spinner<Integer> tuneTrialsSpinner;
    private final Spinner<Integer> tuneFoldsSpinner;
    private final CheckBox earlyStopCheckBox = new CheckBox("Early stopping");
    private final CheckBox computeMetricsCheckBox = new CheckBox("Train/val metrics");
    private final CheckBox autoPruneCheckBox = new CheckBox("Auto-prune features (drop near-constant & redundant)");
    private final PopulationPanel populationPanel = new PopulationPanel();

    // ── Binary mode banner (hidden by default) ──
    private final Label binaryBannerLabel = new Label();
    private final Button exitBinaryButton = new Button("Exit Binary Mode");
    private Runnable onExitBinaryMode;
    private final Button applyToImagesButton = new Button("Apply to which images...");
    private Runnable onApplyToImages;
    private final Button manualLabelButton = new Button("Manual Label Mode");
    private Runnable onManualLabelMode;
    /** Sanitized name of the active binary classifier, or null in multi-class mode.
     *  Used to scope per-image label files so binary classifiers don't share labels. */
    private String activeBinaryMarker = null;
    /** Allowed class names for the active binary classifier; used to filter label
     *  collection and persistence so labels from other classifiers don't bleed in. */
    private java.util.Set<String> activeBinaryClassNames = null;

    public ClassificationPanel(QuPathGUI qupath) {
        super(10);
        this.qupath = qupath;
        setPadding(new Insets(10));

        // ── Title ──
        Label title = new Label(STRINGS.getString("name"));
        title.setStyle("-fx-font-weight: bold; -fx-font-size: 16px;");

        // ── Binary mode banner ──
        binaryBannerLabel.setStyle("-fx-font-weight: bold; -fx-text-fill: #1976D2;");
        binaryBannerLabel.setVisible(false);
        binaryBannerLabel.setManaged(false);
        exitBinaryButton.setVisible(false);
        exitBinaryButton.setManaged(false);
        exitBinaryButton.setOnAction(e -> {
            if (onExitBinaryMode != null) onExitBinaryMode.run();
        });
        applyToImagesButton.setVisible(true);
        applyToImagesButton.setManaged(true);
        applyToImagesButton.setDisable(false);
        applyToImagesButton.setOnAction(e -> {
            if (onApplyToImages != null) onApplyToImages.run();
        });
        manualLabelButton.setMaxWidth(Double.MAX_VALUE);
        manualLabelButton.setOnAction(e -> {
            if (onManualLabelMode != null) onManualLabelMode.run();
        });

        // ── Hyperparameter controls ──
        // A ceiling, not a target: with early stopping on (default) models stop when they stop
        // improving, so 500 is free for one that converges early and rescues one that doesn't.
        // Only a literal count when early stopping is off — see the tooltip.
        roundsSpinner = new Spinner<>(50, 1000, 500, 50);
        roundsSpinner.setEditable(true);
        roundsSpinner.setPrefWidth(90);
        roundsSpinner.setTooltip(new Tooltip(STRINGS.getString("param.num_rounds.help")));

        depthSpinner = new Spinner<>(2, 15, 6, 1);
        depthSpinner.setEditable(true);
        depthSpinner.setPrefWidth(70);
        depthSpinner.setTooltip(new Tooltip(STRINGS.getString("param.max_depth.help")));

        // ── The two parallelism controls ──
        // Deliberately separate, because they are bounded by different resources: one by RAM
        // (a slide at a time), one by CPU (threads inside one training run). A single combined
        // "use N of the machine" number could not express "plenty of cores, not much memory".
        // The labels carry that distinction so the tooltips are not load-bearing.

        // Images in parallel — memory-bound. Each one holds a whole slide's object hierarchy, so
        // this is capped low and defaults to 1. Only affects the batch-apply step after training.
        workersSpinner = new Spinner<>(1, 8, 1, 1);
        workersSpinner.setEditable(true);
        workersSpinner.setPrefWidth(70);
        workersSpinner.setTooltip(new Tooltip("How many images are classified at the same time by"
                + " 'Apply to which images…',\nafter training finishes.\n\n"
                + "Bounded by memory: each one loads a full slide, so raising it costs RAM"
                + " rather than CPU.\nHas no effect on training itself, or if no target images"
                + " are selected."));

        // CPU threads — compute-bound. Applies inside a single training run (both models plus
        // resampling), and is the total budget, not a per-model figure.
        Spinner<Integer> cpuThreadsSpinner = new Spinner<>(
                0,
                Runtime.getRuntime().availableProcessors(),
                CellTuneExtension.trainingThreadsProperty().get());
        cpuThreadsSpinner.setEditable(true);
        cpuThreadsSpinner.setPrefWidth(70);
        cpuThreadsSpinner.setTooltip(new Tooltip("How much CPU training may use: threads for"
                + " XGBoost/LightGBM and resampling\nwithin a single training run.\n\n"
                + "0 = automatic (all " + Runtime.getRuntime().availableProcessors() + " processors)."
                + " Lower it to keep the machine responsive\nwhile training.\n\n"
                + "Bounded by CPU, not memory — unlike 'Images at once'."));
        cpuThreadsSpinner
                .valueProperty()
                .addListener(
                        (v, o, n) -> CellTuneExtension.trainingThreadsProperty().set(n == null ? 0 : n));

        HBox paramRow = new HBox(
                8,
                new Label(STRINGS.getString("param.num_rounds.label")),
                roundsSpinner,
                new Label(STRINGS.getString("param.max_depth.label")),
                depthSpinner,
                new Label("CPU threads:"),
                cpuThreadsSpinner,
                new Label("Images at once:"),
                workersSpinner);
        paramRow.setAlignment(Pos.CENTER_LEFT);

        // ── Pool images checkbox ──
        poolImagesCheckBox.setTooltip(
                new Tooltip("Include labelled cells from all project images in the training set"));
        poolImagesCheckBox.setSelected(true);

        // ── Data balancing ──
        enableBalancingCheckBox.setSelected(true);
        enableBalancingCheckBox.setTooltip(
                new Tooltip("Apply SMOTE + Tomek resampling to address class imbalance before training.\n"
                        + "Recommended for most datasets. Use the Strategy dropdown to override."));
        advancedResamplingCombo.getItems().addAll(ResamplingStrategy.values());
        advancedResamplingCombo.setValue(ResamplingStrategy.SMOTE_TOMEK);
        advancedResamplingCombo.setTooltip(
                new Tooltip("Override the default SMOTE + Tomek strategy with another resampling method"));
        advancedResamplingCombo.visibleProperty().bind(enableBalancingCheckBox.selectedProperty());
        advancedResamplingCombo.managedProperty().bind(enableBalancingCheckBox.selectedProperty());

        HBox balancingRow = new HBox(8, enableBalancingCheckBox, new Label("Strategy:"), advancedResamplingCombo);
        balancingRow.setAlignment(Pos.CENTER_LEFT);

        model1Combo.getItems().addAll(ModelType.values());
        model1Combo.setValue(ModelType.XGBOOST);
        model1Combo.setMaxWidth(Double.MAX_VALUE);
        model1Combo.setTooltip(new Tooltip("Model 1 type"));

        model2Combo.getItems().addAll(ModelType.values());
        model2Combo.setValue(ModelType.LIGHTGBM);
        model2Combo.setMaxWidth(Double.MAX_VALUE);
        model2Combo.setTooltip(new Tooltip("Model 2 type"));

        HBox modelRow = new HBox(8, new Label("Model 1:"), model1Combo, new Label("Model 2:"), model2Combo);
        modelRow.setAlignment(Pos.CENTER_LEFT);

        autoTuneCheckBox.setSelected(false);
        autoTuneCheckBox.setTooltip(
                new Tooltip(
                        "TPE Bayesian optimisation with cross-validation to find optimal hyperparameters independently per model"));

        // Trials and folds set the cost of a search: 2 x trials x folds model fits. They were
        // fixed at 20 x 5 = 200 fits, which is hours on a wide panel with no way to ask for less.
        tuneTrialsSpinner = new Spinner<>(5, 100, HyperparameterTuner.DEFAULT_TRIALS, 5);
        tuneTrialsSpinner.setEditable(true);
        tuneTrialsSpinner.setPrefWidth(70);
        tuneTrialsSpinner.setTooltip(new Tooltip("How many hyperparameter combinations to try per model.\n\n"
                + "The first 5 are random; after that the search uses what it has\n"
                + "learned, so very low counts are close to a random guess.\n\n"
                + "Cost is 2 × trials × folds model fits in total."));

        tuneFoldsSpinner = new Spinner<>(2, 10, HyperparameterTuner.DEFAULT_FOLDS, 1);
        tuneFoldsSpinner.setEditable(true);
        tuneFoldsSpinner.setPrefWidth(70);
        tuneFoldsSpinner.setTooltip(new Tooltip("How many cross-validation folds score each combination.\n\n"
                + "More folds is a steadier score and a proportionally longer run.\n"
                + "Reduced automatically if your rarest class has fewer labelled\n"
                + "cells than the number of folds — see the training log."));

        Label tuneTrialsLabel = new Label("Trials:");
        Label tuneFoldsLabel = new Label("CV folds:");
        HBox tuneRow = new HBox(8, tuneTrialsLabel, tuneTrialsSpinner, tuneFoldsLabel, tuneFoldsSpinner);
        tuneRow.setAlignment(Pos.CENTER_LEFT);
        tuneRow.setPadding(new Insets(0, 0, 0, 20));
        // Only meaningful with auto-tune on, and it is off by default — keep the row out of the
        // way until it can do something.
        tuneRow.visibleProperty().bind(autoTuneCheckBox.selectedProperty());
        tuneRow.managedProperty().bind(autoTuneCheckBox.selectedProperty());

        computeMetricsCheckBox.setSelected(true);
        computeMetricsCheckBox.setTooltip(
                new Tooltip("Train an extra evaluation copy of each model on an 80/20 split to produce the per-class\n"
                        + "precision/recall/F1 report. These copies are discarded afterwards, so this costs\n"
                        + "roughly a third of total training time.\n\n"
                        + "Unchecked: training is faster and the Training Metrics button is unavailable."));
        earlyStopCheckBox.setSelected(true);
        earlyStopCheckBox.setTooltip(
                new Tooltip("Find optimal boosting rounds via validation loss monitoring (patience=20)"));

        showFeatureImportanceCheckBox.setSelected(true);
        showFeatureImportanceCheckBox.setTooltip(
                new Tooltip("After training, compute and display top 10 features by mean |SHAP| value per class"));

        autoPruneCheckBox.setSelected(true);
        autoPruneCheckBox.setTooltip(new Tooltip("After labels are pooled across images and normalised, examine the\n"
                + "full training matrix and drop features that are near-constant or\n"
                + "highly correlated with another feature from the same marker.\n"
                + "Pruning on the pooled set (not just the current image) means a\n"
                + "feature is only dropped when it is redundant across the whole\n"
                + "cohort. Rare-marker guardrail keeps the best feature for any marker\n"
                + "that would otherwise be dropped entirely. The full measurement set\n"
                + "on disk is never touched — only the columns used by this training\n"
                + "run are reduced."));

        // ── Status row ──
        clearImportedButton.setTooltip(
                new Tooltip("Remove imported training rows for the active context (binary marker, or multi-class). "
                        + "Labels, models and predictions are not affected."));
        clearImportedButton.setStyle("-fx-font-size: 10px; -fx-padding: 1 6 1 6;");
        clearImportedButton.setDisable(true);
        clearImportedButton.setOnAction(e -> {
            if (onClearImportedData != null) onClearImportedData.run();
        });
        HBox statsRow = new HBox(12, labelCountLabel, predCountLabel, importedCountLabel, clearImportedButton);
        statsRow.setAlignment(Pos.CENTER_LEFT);

        // ── Progress ──
        progressBar.setMaxWidth(Double.MAX_VALUE);
        progressBar.setPrefHeight(18);
        HBox.setHgrow(progressBar, Priority.ALWAYS);
        statusLabel.setStyle("-fx-font-size: 11px;");

        // ── Buttons ──
        trainButton.setText(STRINGS.getString("classify.train.button"));
        trainButton.setMaxWidth(Double.MAX_VALUE);
        trainButton.setStyle("-fx-font-weight: bold;");
        trainButton.setOnAction(e -> doTrain());

        confusionsButton.setText(STRINGS.getString("classify.plot.button"));
        confusionsButton.setMaxWidth(Double.MAX_VALUE);
        confusionsButton.setDisable(true);
        confusionsButton.setOnAction(e -> doShowConfusions());

        metricsButton.setText("Training Metrics");
        metricsButton.setMaxWidth(Double.MAX_VALUE);
        metricsButton.setDisable(true);
        metricsButton.setTooltip(new javafx.scene.control.Tooltip(
                "Per-class precision/recall/F1 from an 80/20 stratified split, computed during training."));
        metricsButton.setOnAction(e -> doShowTrainingMetrics());

        featureImportanceButton.setMaxWidth(Double.MAX_VALUE);
        featureImportanceButton.setDisable(true);
        featureImportanceButton.setOnAction(e -> {
            if (onFeatureImportance != null) onFeatureImportance.run();
        });

        // sampleButton was a separate "Sample for Review" entry point; the
        // current workflow goes straight from training through Enter Review
        // Mode, so the button is no longer wired into the panel layout.

        reviewButton.setText(STRINGS.getString("menu.review").replace("...", ""));
        reviewButton.setMaxWidth(Double.MAX_VALUE);
        reviewButton.setDisable(true);
        reviewButton.setOnAction(e -> doEnterReview());

        currentImageOnlyCheckBox.setSelected(false);
        currentImageOnlyCheckBox.setTooltip(new javafx.scene.control.Tooltip(
                "When checked, sampling and review only consider cells from the currently open image.\n"
                        + "Default (unchecked) pools predictions from every project image so review covers all FOVs."));

        annotationKeywordField.setPromptText("Filter by annotation keywords (comma-separated, e.g. Tumour)");
        annotationKeywordField.setTooltip(new javafx.scene.control.Tooltip(
                "Optional. Comma-separated keywords matched (case-insensitive substring) against\n"
                        + "annotation names. Across every project image (unless 'Sample current image only'\n"
                        + "is checked), only cells whose centroid falls inside an annotation whose name\n"
                        + "contains one of the keywords are eligible for sampling/review.\n"
                        + "Leave blank to sample all cells."));

        HBox actionRow1 = new HBox(6, confusionsButton, metricsButton);
        HBox.setHgrow(confusionsButton, Priority.ALWAYS);
        HBox.setHgrow(metricsButton, Priority.ALWAYS);

        featureImportanceButton.setMaxWidth(Double.MAX_VALUE);

        // ── Separator ──
        Separator sep = new Separator();

        // ── Layout ──
        getChildren()
                .addAll(
                        title,
                        binaryBannerLabel,
                        exitBinaryButton,
                        manualLabelButton,
                        applyToImagesButton,
                        paramRow,
                        modelRow,
                        poolImagesCheckBox,
                        balancingRow,
                        autoTuneCheckBox,
                        tuneRow,
                        earlyStopCheckBox,
                        computeMetricsCheckBox,
                        showFeatureImportanceCheckBox,
                        autoPruneCheckBox,
                        restrictToImportedFeaturesCheckBox,
                        new Separator(),
                        statsRow,
                        trainButton,
                        progressBar,
                        statusLabel,
                        new Separator(),
                        actionRow1,
                        featureImportanceButton,
                        currentImageOnlyCheckBox,
                        buildAnnotationFilterBox(),
                        reviewButton,
                        sep,
                        populationPanel);
    }

    // ── Binary mode indicator ──

    /** Show or hide the binary mode banner in the docked panel. */
    public void setActiveBinaryMarker(String markerName) {
        boolean active = (markerName != null && !markerName.isBlank());
        this.activeBinaryMarker = active ? markerName : null;
        binaryBannerLabel.setText(active ? "Active binary mode: " + markerName : "");
        binaryBannerLabel.setVisible(active);
        binaryBannerLabel.setManaged(active);
        exitBinaryButton.setVisible(active);
        exitBinaryButton.setManaged(active);
        // The "Apply to which images..." picker is useful in both modes:
        // binary-mode users want to apply per-marker classifiers to a subset,
        // multi-class users want to share one trained classifier across the
        // project so every image's PopulationSet gets persisted.
        applyToImagesButton.setVisible(true);
        applyToImagesButton.setManaged(true);
        applyToImagesButton.setDisable(false);
        // In binary mode, every label belongs to one classifier so pooling is
        // automatic — disable the manual checkbox to make that obvious.
        if (active) {
            poolImagesCheckBox.setSelected(true);
            poolImagesCheckBox.setDisable(true);
            poolImagesCheckBox.setText("Pool labels from all images (auto in binary mode)");
        } else {
            poolImagesCheckBox.setDisable(false);
            poolImagesCheckBox.setText("Pool labels from all images");
        }
    }

    public void setOnExitBinaryMode(Runnable cb) {
        this.onExitBinaryMode = cb;
    }

    /** Allowed class names for the active binary classifier (e.g. ["GrB+","GrB-"]).
     *  When non-null, label collection and persistence filter to these classes so
     *  point annotations from other classifiers don't contaminate this store. */
    public void setActiveBinaryClassNames(java.util.Collection<String> classNames) {
        if (classNames == null || classNames.isEmpty()) {
            this.activeBinaryClassNames = null;
        } else {
            this.activeBinaryClassNames = new java.util.LinkedHashSet<>(classNames);
        }
    }

    public void setOnApplyToImages(Runnable cb) {
        this.onApplyToImages = cb;
    }

    public void setOnManualLabelMode(Runnable cb) {
        this.onManualLabelMode = cb;
    }

    public void setOnFeatureImportance(Runnable cb) {
        this.onFeatureImportance = cb;
    }

    public void setOnClearImportedData(Runnable cb) {
        this.onClearImportedData = cb;
    }

    public void setApplyToImagesCount(int count) {
        applyToImagesButton.setText(
                count > 0 ? "Apply to which images... (" + count + ")" : "Apply to which images...");
    }

    // ── State setters (called by CellTuneExtension to share state) ──

    public void setLabelStore(LabelStore store) {
        this.labelStore = store;
        refreshStats();
    }

    public void setClassifier(DualModelClassifier cls) {
        this.classifier = cls;
        // Re-enable the "Training Metrics" and "Feature Importance" buttons if the restored classifier has metrics.
        metricsButton.setDisable(cls == null || !cls.hasTrainValMetrics());
        featureImportanceButton.setDisable(cls == null || !cls.isTrained());
    }

    /**
     * Apply the per-feature normaliser to a single row already aligned to
     * {@code featureNames}, mirroring how {@link CellFeatureExtractor#extractRow}
     * treats image-derived cells (non-finite values coerced to {@code 0} before the
     * transform). Used to normalise imported training rows so they share the same
     * scale as pooled cell rows. Returns a new array; the input is left untouched.
     */
    private static float[] normaliseRow(float[] row, List<String> featureNames, FeatureNormalizer normalizer) {
        float[] out = new float[row.length];
        for (int i = 0; i < row.length; i++) {
            float v = Float.isFinite(row[i]) ? row[i] : 0f;
            out[i] = normalizer == null ? v : normalizer.apply(featureNames.get(i), v);
        }
        return out;
    }

    public void setSelectedFeatures(List<String> features) {
        this.selectedFeatures = features;
    }

    public void setCellTypeTable(CellTypeTable table) {
        this.cellTypeTable = table;
    }

    public void setPredAll(PopulationSet pa) {
        this.predAll = pa;
        refreshStats();
        // Re-enable the "Agreement Confusion Matrix" button if predictions are present
        // (e.g. after restoring per-image predictions on project re-open).
        confusionsButton.setDisable(pa == null || pa.size() == 0);
    }

    public void setLastAgreementRates(double[] rates) {
        this.lastAgreementRates = rates;
    }

    public void setLastSampledCellIds(List<String> ids) {
        this.lastSampledCellIds = ids;
        if (ids == null || ids.isEmpty()) {
            this.lastSampledCellImageMap = Map.of();
            this.lastSampledCellAnnotationsMap = Map.of();
            this.lastSampledPredictions = null;
        }
        reviewButton.setDisable(ids == null || ids.isEmpty());
    }

    public void setImportedTrainingData(List<GroundTruthIO.TrainingRow> rows, List<String> featureNames) {
        this.importedTrainingRows = rows == null ? null : List.copyOf(rows);
        this.importedTrainingFeatureNames = featureNames == null ? null : List.copyOf(featureNames);
        refreshStats();
    }

    // ── Callbacks ──

    public void setOnLabelStoreChanged(Consumer<LabelStore> cb) {
        this.onLabelStoreChanged = cb;
    }

    public void setOnPredAllChanged(Consumer<PopulationSet> cb) {
        this.onPredAllChanged = cb;
    }

    public void setOnAgreementRatesChanged(Consumer<double[]> cb) {
        this.onAgreementRatesChanged = cb;
    }

    public void setOnSampledCellsChanged(Consumer<List<String>> cb) {
        this.onSampledCellsChanged = cb;
    }

    public void setOnClassifierChanged(Consumer<DualModelClassifier> cb) {
        this.onClassifierChanged = cb;
    }

    public void setAutoClassifyCallback(Supplier<Boolean> cb) {
        this.autoClassifyCallback = cb;
    }

    public void setBinaryTargetImagesSupplier(Supplier<List<String>> cb) {
        this.binaryTargetImagesSupplier = cb;
    }

    // ── Actions ──

    private ResamplingStrategy getEffectiveResamplingStrategy() {
        if (!enableBalancingCheckBox.isSelected()) return ResamplingStrategy.NONE;
        return advancedResamplingCombo.getValue() != null
                ? advancedResamplingCombo.getValue()
                : ResamplingStrategy.SMOTE_TOMEK;
    }

    private void doTrain() {
        var imageData = qupath.getImageData();
        if (imageData == null) {
            Dialogs.showErrorMessage(STRINGS.getString("name"), "No image is open.");
            return;
        }

        Collection<PathObject> detections = imageData.getHierarchy().getDetectionObjects();
        if (detections.isEmpty()) {
            Dialogs.showErrorMessage(STRINGS.getString("name"), "No detections found. Run cell detection first.");
            return;
        }

        // Collect labels from annotations
        if (labelStore == null) labelStore = new LabelStore("CellTune");
        collectLabelsFromAnnotations(labelStore);

        // Sync labels with current QuPath class list — remove labels for deleted classes.
        // In binary mode, restrict valid classes to the active binary classifier's
        // classes so labels from other classifiers (e.g. PD-1+/-) get evicted from
        // this store before training and persistence.
        var project = qupath.getProject();
        Set<String> validClasses = new LinkedHashSet<>();
        if (activeBinaryMarker != null && activeBinaryClassNames != null && !activeBinaryClassNames.isEmpty()) {
            validClasses.addAll(activeBinaryClassNames);
            labelStore.retainClasses(validClasses);
        } else if (project != null) {
            for (var pc : project.getPathClasses()) {
                if (pc != null && pc.getName() != null && !pc.getName().isEmpty()) {
                    validClasses.add(pc.getName());
                }
            }
            if (!validClasses.isEmpty()) {
                labelStore.retainClasses(validClasses);
            }
        }

        int importedEvidenceCount = (int)
                (importedTrainingRows == null
                        ? 0
                        : importedTrainingRows.stream()
                                .filter(r -> r != null
                                        && r.label() != null
                                        && !r.label().isBlank()
                                        && r.features() != null)
                                .count());

        // If not enough local labels, try to reload from saved state.
        // Imported rows are counted as training evidence and may satisfy the threshold.
        if (labelStore.size() < 10 && importedEvidenceCount == 0) {
            if (project != null && imageData != null) {
                var imgEntry = project.getEntry(imageData);
                if (imgEntry != null) {
                    try {
                        var savedLabels = ProjectStateManager.loadImageLabels(
                                project, activeBinaryMarker, imgEntry.getImageName());
                        if (savedLabels != null && savedLabels.size() > labelStore.size()) {
                            labelStore.mergeFrom(savedLabels);
                            // Re-sync after merge
                            if (!validClasses.isEmpty()) {
                                labelStore.retainClasses(validClasses);
                            }
                        }
                    } catch (Exception ex) {
                        // Ignore errors
                    }
                }
                // Also try the global classifier state labels
                if (labelStore.size() < 10) {
                    try {
                        var state = ProjectStateManager.loadState(project);
                        if (state != null && state.labels != null && state.labels.size() > labelStore.size()) {
                            labelStore.mergeFrom(new LabelStore("saved", state.labels));
                            if (!validClasses.isEmpty()) {
                                labelStore.retainClasses(validClasses);
                            }
                        }
                    } catch (Exception ex) {
                        // Ignore errors
                    }
                }
            }
        }

        int totalEvidenceCount = labelStore.size() + importedEvidenceCount;
        if (totalEvidenceCount < 10) {
            Dialogs.showErrorMessage(
                    STRINGS.getString("name"),
                    "Need at least 10 labelled cells to train. Found: " + totalEvidenceCount
                            + " (local labels=" + labelStore.size() + ", imported rows=" + importedEvidenceCount + ")"
                            + "\nUse point annotations or import training rows to label cells.");
            return;
        }

        // Discover features
        List<String> featureNames = CellFeatureExtractor.discoverFeatureNames(detections);
        if (featureNames.isEmpty()) {
            Dialogs.showErrorMessage(STRINGS.getString("name"), "No cell measurements found.");
            return;
        }

        // Apply user feature selection
        if (selectedFeatures != null && !selectedFeatures.isEmpty()) {
            featureNames =
                    featureNames.stream().filter(selectedFeatures::contains).collect(Collectors.toList());
            if (featureNames.isEmpty()) {
                Dialogs.showErrorMessage(STRINGS.getString("name"), "None of the selected features are present.");
                return;
            }
        }

        // Pre-flight heap estimate. The feature matrix plus the native copies each model makes
        // is the dominant allocation, and resampling inflates the row count before any of that,
        // so warn while the user can still act (fewer features, or a bigger -Xmx).
        if (!TrainingMemoryCheck.confirmEnoughHeap(
                detections.size(),
                featureNames.size(),
                labelStore.getClassCounts().values(),
                getEffectiveResamplingStrategy())) {
            return;
        }

        // ── Restrict to features shared with imported data ─────────────────
        // When the user has imported ground-truth rows from another project the
        // imported rows only carry values for that project's panel. Aligning to
        // the current project's full feature list zero-pads any current-only
        // marker, which biases the classifier. With this option on we drop those
        // current-only features so both data sources train on the same columns.
        String restrictPreamble = null;
        if (restrictToImportedFeaturesCheckBox.isSelected()
                && importedTrainingFeatureNames != null
                && !importedTrainingFeatureNames.isEmpty()) {
            Set<String> importedLower = importedTrainingFeatureNames.stream()
                    .filter(n -> n != null)
                    .map(n -> n.strip().toLowerCase(Locale.ROOT))
                    .collect(Collectors.toSet());
            int beforeRestrict = featureNames.size();
            List<String> intersected = featureNames.stream()
                    .filter(n -> importedLower.contains(n.strip().toLowerCase(Locale.ROOT)))
                    .collect(Collectors.toList());
            if (intersected.isEmpty()) {
                Dialogs.showErrorMessage(
                        STRINGS.getString("name"),
                        "Restrict to imported features is on, but no current-project features\n"
                                + "match any of the " + importedTrainingFeatureNames.size()
                                + " imported feature names. Disable the option or fix the panel mismatch.");
                return;
            }
            restrictPreamble = String.format(
                    "Restrict-to-imported: %d \u2192 %d features (intersection with %d imported markers)",
                    beforeRestrict, intersected.size(), importedTrainingFeatureNames.size());
            featureNames = intersected;
        }
        final String restrictPreambleFinal = restrictPreamble;

        // ── Auto-prune is deferred until AFTER label pooling + normalisation ──
        // Near-constant/redundant decisions depend on the data they are measured
        // against, so pruning here (current image only, raw measurements) would
        // judge variance/correlation on an unrepresentative subset — and worse,
        // gate which columns are even read from the pooled images. We only capture
        // the intent now; the prune runs on the background thread once the full,
        // normalised training matrix (current + pooled + imported) is assembled.
        final boolean autoPruneEnabled = autoPruneCheckBox.isSelected() && featureNames.size() > 20;

        // The feature extractor is built on the background thread once auto-prune
        // has decided the final kept-feature set against the pooled training matrix.
        if (classifier == null) classifier = new DualModelClassifier();

        // Apply model types from combos
        classifier.setModel1Type(model1Combo.getValue());
        classifier.setModel2Type(model2Combo.getValue());

        // Apply hyperparameters from spinners
        classifier.setNumRounds(roundsSpinner.getValue());
        classifier.setMaxDepth(depthSpinner.getValue());

        // Reset and bind progress (sidebar)
        progressBar.progressProperty().unbind();
        progressBar.setProgress(0);
        classifier.resetProgress();
        progressBar.progressProperty().bind(classifier.progressProperty());
        statusLabel.textProperty().bind(classifier.statusProperty());
        trainButton.setDisable(true);

        // ── Detailed progress dialog (mirrors the dropdown-menu trainer) ──
        var progressStage = new javafx.stage.Stage();
        progressStage.setTitle("CellTune \u2014 Training");
        progressStage.initOwner(qupath.getStage());
        progressStage.initModality(javafx.stage.Modality.NONE);
        progressStage.setResizable(false);
        progressStage.setAlwaysOnTop(true);

        var dlgProgressBar = new ProgressBar(0);
        dlgProgressBar.setPrefWidth(500);
        var dlgStatusLabel = new Label("Initialising\u2026");
        dlgStatusLabel.setWrapText(true);
        dlgStatusLabel.setMaxWidth(500);
        var logArea = new TextArea();
        logArea.setEditable(false);
        logArea.setPrefHeight(180);
        logArea.setPrefWidth(500);
        logArea.setStyle("-fx-font-family: monospace; -fx-font-size: 11px;");

        var progressBox = new VBox(8, dlgStatusLabel, dlgProgressBar, logArea);
        progressBox.setPadding(new Insets(15));
        progressStage.setScene(new javafx.scene.Scene(progressBox));
        progressStage.show();
        if (restrictPreambleFinal != null) {
            logArea.appendText(restrictPreambleFinal + "\n");
        }

        dlgProgressBar.progressProperty().bind(classifier.progressProperty());
        dlgStatusLabel.textProperty().bind(classifier.statusProperty());

        // Durable copy of this run's log. The on-screen text area is discarded when the progress
        // window closes, and is never written at all if the JVM dies — which is exactly the case
        // (OOM, native crash) worth being able to read afterwards.
        //
        // The settings are read here because they come from UI controls, which must be touched on
        // the FX thread. Opening the file is not: it creates a directory, lists and prunes the
        // existing logs, then writes a flushed line per header row, and on a project sitting on a
        // network share that is enough I/O to stall the click. The worker does it instead, so
        // until it has, this holds a recorder that discards everything.
        final String logTitle =
                activeBinaryMarker != null ? "Binary training — " + activeBinaryMarker : "Multi-class training";
        final List<String[]> logSettings = TrainingLogRecorder.settings(
                "Cells",
                detections.size(),
                "Labelled",
                labelStore.size(),
                "Features",
                featureNames.size(),
                // Scoped explicitly: pooling adds classes from other images, so this is
                // routinely lower than the "N classes" the classifier goes on to report,
                // which read like a contradiction sitting next to each other.
                "Classes (image)",
                labelStore.getClassNames().size(),
                "Balancing",
                getEffectiveResamplingStrategy(),
                "Early stop",
                earlyStopCheckBox.isSelected(),
                "Metrics",
                computeMetricsCheckBox.isSelected(),
                "Auto-tune",
                autoTuneCheckBox.isSelected()
                        ? "yes (" + tuneTrialsSpinner.getValue() + " trials x " + tuneFoldsSpinner.getValue()
                                + " folds)"
                        : "no",
                "Auto-prune",
                autoPruneCheckBox.isSelected(),
                "Pool images",
                poolImagesCheckBox.isSelected(),
                "Model 1",
                classifier.getModel1Type(),
                "Model 2",
                classifier.getModel2Type(),
                "CPU threads",
                TrainingThreads.total(),
                "Images at once",
                workersSpinner.getValue(),
                // Only non-zero when the user has deliberately traded accuracy for speed, and it
                // changes predictions — so a log that does not record it cannot be compared with
                // another one.
                "XGB max_bin",
                XGBoostModel.getMaxBin() == 0 ? "256 (default)" : String.valueOf(XGBoostModel.getMaxBin()));
        final java.util.concurrent.atomic.AtomicReference<TrainingLogRecorder> logRecorderRef =
                new java.util.concurrent.atomic.AtomicReference<>(TrainingLogRecorder.noOp());

        // Batch the text-area appends. Tuning and batch-apply emit lines faster than the FX
        // thread can lay out a TextArea, and one runLater per line floods the event queue enough
        // to visibly stall the UI during the run.
        final StringBuilder pendingLog = new StringBuilder();
        final java.util.concurrent.atomic.AtomicBoolean flushQueued = new java.util.concurrent.atomic.AtomicBoolean();
        Consumer<String> trainLog = msg -> {
            logRecorderRef.get().accept(msg);
            synchronized (pendingLog) {
                pendingLog.append(msg).append('\n');
            }
            if (flushQueued.compareAndSet(false, true)) {
                Platform.runLater(() -> {
                    flushQueued.set(false);
                    String chunk;
                    synchronized (pendingLog) {
                        chunk = pendingLog.toString();
                        pendingLog.setLength(0);
                    }
                    if (!chunk.isEmpty()) logArea.appendText(chunk);
                });
            }
        };

        // Train in background
        final LabelStore storeCopy = labelStore;
        // Capture the per-image label scope (sanitized binary marker, or null in
        // multi-class mode) so per-image label files don't bleed between classifiers.
        final String scope = activeBinaryMarker;
        final boolean binaryActive = (scope != null);
        // Snapshot the allowed-class set for this binary session so the background
        // thread filters out foreign labels (e.g. PD-1+/- in a GrB save) even if
        // the active marker changes mid-train.
        final java.util.Set<String> binaryClassFilter =
                (binaryActive && activeBinaryClassNames != null && !activeBinaryClassNames.isEmpty())
                        ? new java.util.LinkedHashSet<>(activeBinaryClassNames)
                        : null;
        // In binary mode every label belongs to one classifier — always pool labels
        // for that classifier, regardless of the (now-disabled) checkbox state.
        final boolean poolAllImages = binaryActive || poolImagesCheckBox.isSelected();
        final ResamplingStrategy resamplingStrategy = getEffectiveResamplingStrategy();
        final boolean autoTuneSelected = autoTuneCheckBox.isSelected();
        final int tuneTrials = tuneTrialsSpinner.getValue() == null
                ? HyperparameterTuner.DEFAULT_TRIALS
                : tuneTrialsSpinner.getValue();
        final int tuneFolds =
                tuneFoldsSpinner.getValue() == null ? HyperparameterTuner.DEFAULT_FOLDS : tuneFoldsSpinner.getValue();
        final boolean earlyStopSelected = earlyStopCheckBox.isSelected();
        final boolean computeMetricsSelected = computeMetricsCheckBox.isSelected();
        final List<String> finalFeatureNames = featureNames;
        final var projectRef = project;
        final List<GroundTruthIO.TrainingRow> importedRowsSnapshot =
                importedTrainingRows == null ? null : List.copyOf(importedTrainingRows);
        final List<String> importedFeatureNamesSnapshot =
                importedTrainingFeatureNames == null ? null : List.copyOf(importedTrainingFeatureNames);
        final boolean binaryModeActive = binaryBannerLabel.isVisible();
        List<String> suppliedTargets = binaryTargetImagesSupplier != null ? binaryTargetImagesSupplier.get() : null;
        final List<String> batchTargetImages = suppliedTargets == null ? List.of() : List.copyOf(suppliedTargets);
        final int workers = workersSpinner.getValue();
        Thread trainThread = new Thread(
                () -> {
                    // Opened here rather than on the FX thread: this touches the filesystem, and
                    // every header line is flushed individually. Set before any logging so the
                    // header is the first thing in the file.
                    final TrainingLogRecorder logRecorder = TrainingLogRecorder.open(projectRef);
                    logRecorderRef.set(logRecorder);
                    logRecorder.writeHeader(logTitle, logSettings);
                    try {
                        // Auto-backup labels
                        if (projectRef != null) {
                            var backupEntry = projectRef.getEntry(imageData);
                            String backupImageName = backupEntry != null ? backupEntry.getImageName() : null;
                            ProjectStateManager.backupLabels(projectRef, backupImageName, storeCopy);
                        }

                        // UniFORM batch correction (streamed): when the toggle is on and the
                        // project has a fit, each per-image extractor multiplies its cells'
                        // measurements by that image's per-channel gain before use. Loaded once;
                        // applied per image at every extraction seam (current, pooled, inference).
                        final BatchShifts trainBatchShifts = BatchCorrection.loadIfEnabled(projectRef);
                        final String currentImgName = (projectRef != null && projectRef.getEntry(imageData) != null)
                                ? projectRef.getEntry(imageData).getImageName()
                                : null;
                        if (trainBatchShifts != null) {
                            trainLog.accept("Batch correction ON — streaming per-image gains into training.");
                        }

                        // Pool labelled cells from other project images. These are
                        // real cells carrying the full feature panel, extracted RAW
                        // (the classifier trains on raw values — normalisation is a
                        // clustering-only concern), so they take part in the auto-prune
                        // decision below on the same raw scale.
                        List<float[]> pooledRows = null;
                        List<String> pooledLabels = null;
                        if (poolAllImages && projectRef != null) {
                            var pooled = TrainingOrchestrator.poolLabelsFromOtherImages(
                                    projectRef, imageData, scope, finalFeatureNames, null, trainBatchShifts, trainLog);
                            pooledRows = pooled.rows();
                            pooledLabels = pooled.labels();
                        }

                        // Explicitly imported training rows (if any), aligned to the
                        // full panel and kept RAW (NaN-cleaned) on the same scale as
                        // the image-derived rows — the classifier is raw-only. Imported
                        // rows may carry a different/partial panel (zero-filled where
                        // unmapped), so they are EXCLUDED from the prune decision — but
                        // they are still trained on.
                        List<float[]> importedRows = null;
                        List<String> importedLabels = null;
                        var pooledImported = DataPoolingService.poolImportedRows(
                                importedRowsSnapshot, importedFeatureNamesSnapshot, finalFeatureNames);
                        if (pooledImported.addedCount() > 0) {
                            importedRows = new ArrayList<>(pooledImported.rows().size());
                            for (float[] row : pooledImported.rows()) {
                                // null normalizer → raw values, but NaN→0 cleanup is still applied.
                                importedRows.add(normaliseRow(row, finalFeatureNames, null));
                            }
                            importedLabels = pooledImported.labels();
                            trainLog.accept("Merged " + pooledImported.addedCount() + " imported training rows ("
                                    + pooledImported.mappedFeatureCount() + "/" + finalFeatureNames.size()
                                    + " features aligned, raw; excluded from pruning).");
                        }

                        // ── Auto-prune on the pooled RAW cell matrix ──
                        // Decisions use the current image's labelled cells plus the
                        // cells pooled from other images — the data the models train
                        // on. Imported rows are deliberately left out so a partial /
                        // zero-filled imported panel can't distort the variance and
                        // correlation estimates that drive pruning.
                        List<String> effectiveFeatureNames = finalFeatureNames;
                        int[] keepIdx = null;
                        if (autoPruneEnabled) {
                            List<float[]> pruneRows = new ArrayList<>();
                            CellFeatureExtractor fullExtractor = new CellFeatureExtractor(finalFeatureNames);
                            BatchCorrection.applyTo(fullExtractor, trainBatchShifts, currentImgName);
                            var labelMap = storeCopy.getAllLabels();
                            for (PathObject cell : detections) {
                                if (labelMap.containsKey(cell.getID().toString())) {
                                    pruneRows.add(fullExtractor.extractRow(cell));
                                }
                            }
                            if (pooledRows != null) {
                                pruneRows.addAll(pooledRows);
                            }
                            if (pruneRows.size() >= 10) {
                                int before = finalFeatureNames.size();
                                FeaturePruner.PruneResult pr = FeaturePruner.prune(
                                        pruneRows.toArray(new float[0][]),
                                        finalFeatureNames,
                                        FeaturePruner.PruneOptions.defaults(),
                                        null);
                                if (!pr.keptFeatures().isEmpty()
                                        && pr.keptFeatures().size() < before) {
                                    effectiveFeatureNames = pr.keptFeatures();
                                    Map<String, Integer> fullIndex = new HashMap<>();
                                    for (int i = 0; i < finalFeatureNames.size(); i++) {
                                        fullIndex.put(finalFeatureNames.get(i), i);
                                    }
                                    keepIdx = new int[effectiveFeatureNames.size()];
                                    for (int k = 0; k < effectiveFeatureNames.size(); k++) {
                                        keepIdx[k] = fullIndex.get(effectiveFeatureNames.get(k));
                                    }
                                    trainLog.accept(String.format(
                                            "Auto-prune: %d → %d features (dropped %d near-constant, %d redundant;"
                                                    + " %d labelled cells)",
                                            before,
                                            pr.keptFeatures().size(),
                                            pr.droppedConstant(),
                                            pr.droppedWithinMarker(),
                                            pruneRows.size()));
                                }
                            }
                        }

                        // Assemble the supplementary rows (pooled + imported) the
                        // trainer appends to the current image, subsetting to the kept
                        // columns when pruning fired so every row stays aligned.
                        List<float[]> supplementaryRows = null;
                        List<String> supplementaryLabels = null;
                        if (pooledRows != null || importedRows != null) {
                            supplementaryRows = new ArrayList<>();
                            supplementaryLabels = new ArrayList<>();
                            if (pooledRows != null) {
                                supplementaryRows.addAll(pooledRows);
                                supplementaryLabels.addAll(pooledLabels);
                            }
                            if (importedRows != null) {
                                supplementaryRows.addAll(importedRows);
                                supplementaryLabels.addAll(importedLabels);
                            }
                            if (keepIdx != null) {
                                List<float[]> subset = new ArrayList<>(supplementaryRows.size());
                                for (float[] row : supplementaryRows) {
                                    float[] kept = new float[keepIdx.length];
                                    for (int k = 0; k < keepIdx.length; k++) {
                                        kept[k] = row[keepIdx[k]];
                                    }
                                    subset.add(kept);
                                }
                                supplementaryRows = subset;
                            }
                        }

                        // Classifier trains on RAW values (tree models are invariant to the
                        // arcsinh/sqrt monotone transforms; normalisation is clustering-only).
                        CellFeatureExtractor extractor = new CellFeatureExtractor(effectiveFeatureNames);
                        BatchCorrection.applyTo(extractor, trainBatchShifts, currentImgName);

                        classifier.trainAndPredict(
                                detections,
                                storeCopy,
                                extractor,
                                supplementaryRows,
                                supplementaryLabels,
                                resamplingStrategy,
                                autoTuneSelected,
                                tuneTrials,
                                tuneFolds,
                                earlyStopSelected,
                                computeMetricsSelected,
                                trainLog);

                        predAll = classifier.getPredALL();

                        // Save classifier state
                        if (projectRef != null) {
                            // In binary mode the trained binary classifier is persisted separately
                            // (the onClassifierChanged callback -> saveBinaryState), and its imported
                            // rows live in binary-imported/<marker>.json. Writing the multi-class
                            // classifier-state.json here would pollute it with the binary classifier's
                            // labels / imported rows / model (which then leak back into multi-class on
                            // the next image switch), so only save it in multi-class mode.
                            if (scope == null) {
                                var state = classifier.toClassifierState("CellTune");
                                ProjectStateManager.saveState(
                                        projectRef,
                                        state.getName(),
                                        storeCopy,
                                        state.getFeatureNames(),
                                        state.getClassNames(),
                                        state.getXgboostBytes(),
                                        state.getLightgbmBytes(),
                                        state.getRfModel1Bytes(),
                                        state.getRfModel2Bytes(),
                                        state.getModel1Type(),
                                        state.getModel2Type(),
                                        importedFeatureNamesSnapshot,
                                        importedRowsSnapshot);

                                // Persist per-model train/val metrics so the "Training Metrics" view
                                // and confusion matrix are available after reopening the project.
                                try {
                                    ProjectStateManager.saveTrainingMetrics(
                                            projectRef,
                                            classifier.getModel1TrainMetrics(),
                                            classifier.getModel1ValMetrics(),
                                            classifier.getModel2TrainMetrics(),
                                            classifier.getModel2ValMetrics());
                                } catch (Exception ex) {
                                    System.err.println(
                                            "[CellTune] Failed to persist training metrics: " + ex.getMessage());
                                }
                            }

                            // Save per-image labels for cross-image pooling
                            var imgEntry = projectRef.getEntry(imageData);
                            if (imgEntry != null) {
                                var filteredStore = filterLabelStoreToImage(storeCopy, imageData);
                                // In binary mode, drop any foreign-class labels so the per-image
                                // file for this classifier never contains the other classifier's
                                // classes (and any historical contamination gets self-healed).
                                if (binaryClassFilter != null) {
                                    filteredStore.retainClasses(binaryClassFilter);
                                }
                                ProjectStateManager.saveImageLabels(
                                        projectRef, scope, imgEntry.getImageName(), filteredStore);
                            }
                        }

                        // Batch-apply is timed into the same breakdown as training. On a project of
                        // whole slides it is routinely the largest single block of the wait, and a
                        // summary that stopped at "Done." reported a wall clock the user could see
                        // was wrong.
                        var runTimer = classifier.getPhaseTimer();
                        int batchApplied = 0;
                        if (projectRef != null && !batchTargetImages.isEmpty()) {
                            if (runTimer != null) runTimer.start("apply to other images");
                            batchApplied = TrainingOrchestrator.applyToTargetImages(
                                    projectRef,
                                    imageData,
                                    batchTargetImages,
                                    workers,
                                    classifier,
                                    classifier.getFeatureNames(),
                                    null, // raw inference — classifier is raw-only (batch gain applied separately)
                                    trainBatchShifts,
                                    this,
                                    trainLog);
                        }
                        if (runTimer != null) runTimer.writeSummary();

                        final int totalBatchApplied = batchApplied;
                        Platform.runLater(() -> {
                            progressBar.progressProperty().unbind();
                            statusLabel.textProperty().unbind();
                            progressBar.setProgress(1.0);
                            statusLabel.setText("Training complete \u2014 " + predAll.size()
                                    + " cells classified, "
                                    + predAll.getDisagreementCount() + " disagreements.");
                            dlgProgressBar.progressProperty().unbind();
                            dlgStatusLabel.textProperty().unbind();
                            dlgProgressBar.setProgress(1.0);
                            dlgStatusLabel.setText("Complete \u2014 close this window when ready.");
                            logArea.appendText("\nDone! Classified " + predAll.size() + " cells, "
                                    + predAll.getDisagreementCount() + " disagreements.\n"
                                    + (totalBatchApplied > 0
                                            ? "Applied to " + totalBatchApplied + " additional image(s).\n"
                                            : ""));
                            trainButton.setDisable(false);
                            confusionsButton.setDisable(false);
                            // Enable Review Mode now that predictions exist; clicking it
                            // will trigger uncertainty sampling on demand.
                            reviewButton.setDisable(predAll == null || predAll.size() == 0);

                            metricsButton.setDisable(!classifier.hasTrainValMetrics());

                            // Auto-show feature importance if checkbox is selected
                            if (showFeatureImportanceCheckBox.isSelected()) {
                                doShowFeatureImportance();
                            }

                            // Update spinners if auto-tuned
                            // (Tuned params are now per-model; spinners retain user defaults)

                            imageData.getHierarchy().fireHierarchyChangedEvent(this);

                            // Update population panel
                            populationPanel.update(
                                    classifier.getPredMDL1(),
                                    classifier.getPredMDL2(),
                                    classifier.getPredAVG(),
                                    classifier.getPredALL(),
                                    classifier.getClassNames());

                            refreshStats();

                            // Notify extension
                            if (onLabelStoreChanged != null) onLabelStoreChanged.accept(storeCopy);
                            if (onPredAllChanged != null) onPredAllChanged.accept(predAll);
                            if (onClassifierChanged != null) onClassifierChanged.accept(classifier);

                            String completionMessage = "Training complete. " + predAll.size() + " cells classified, "
                                    + predAll.getDisagreementCount() + " disagreements.";
                            if (totalBatchApplied > 0) {
                                completionMessage += " Applied to " + totalBatchApplied + " additional image(s).";
                            }
                            Dialogs.showInfoNotification(STRINGS.getString("name"), completionMessage);
                        });
                    } catch (Exception | OutOfMemoryError ex) {
                        // OutOfMemoryError is an Error, not an Exception: before this it escaped
                        // the handler entirely, killing the worker silently and leaving the
                        // progress dialog spinning with the Train button stuck disabled.
                        boolean oom = ex instanceof OutOfMemoryError;
                        var timer = classifier.getPhaseTimer();
                        String phase =
                                timer != null && timer.currentPhase() != null ? timer.currentPhase() : "training";
                        logRecorder.recordFailure(phase, ex);
                        if (oom) {
                            // Release the biggest retained graph before formatting anything, so
                            // the handler itself has room to run.
                            long peakMb = timer != null ? timer.peakHeapBytes() / (1024 * 1024) : 0;
                            long maxMb = Runtime.getRuntime().maxMemory() / (1024 * 1024);
                            logRecorder.accept(String.format(
                                    "!! Out of memory during '%s' — peak heap %,d MB of %,d MB", phase, peakMb, maxMb));
                        }
                        // No close() here — the finally below runs as soon as this block ends,
                        // which is before the runLater below can reach the FX thread, so the file
                        // is complete by the time the dialog quotes its path.
                        final String message = oom
                                ? String.format(
                                        "Out of memory during '%s'.%n%n"
                                                + "The JVM heap is %,d MB. To increase it: Edit → Preferences →"
                                                + " 'Maximum memory', then restart QuPath.%n"
                                                + "To need less memory: select fewer features, or set Balancing"
                                                + " to None.",
                                        phase, Runtime.getRuntime().maxMemory() / (1024 * 1024))
                                : "Training failed: " + ex.getMessage();
                        final String logPath = logRecorder.getFile() != null ? "\nLog: " + logRecorder.getFile() : "";
                        Platform.runLater(() -> {
                            progressBar.progressProperty().unbind();
                            statusLabel.textProperty().unbind();
                            progressBar.setProgress(0);
                            statusLabel.setText("Training failed");
                            dlgProgressBar.progressProperty().unbind();
                            dlgStatusLabel.textProperty().unbind();
                            dlgProgressBar.setProgress(0);
                            dlgStatusLabel.setText("Training failed!");
                            logArea.appendText("\nERROR: " + message + logPath + "\n");
                            trainButton.setDisable(false);
                            Dialogs.showErrorMessage(STRINGS.getString("name"), message + logPath);
                        });
                    } finally {
                        logRecorder.close();
                    }
                },
                "CellTune-Training");
        trainThread.setDaemon(true);
        trainThread.start();
    }

    private void doShowTrainingMetrics() {
        if (classifier == null || !classifier.hasTrainValMetrics()) {
            Dialogs.showInfoNotification(
                    STRINGS.getString("name"),
                    "No training metrics available. Train first (need \u2265 20 labelled cells).");
            return;
        }
        new TrainingMetricsView(
                        qupath.getStage(),
                        classifier.getModel1TrainMetrics(),
                        classifier.getModel1ValMetrics(),
                        classifier.getModel2TrainMetrics(),
                        classifier.getModel2ValMetrics())
                .show();
    }

    private void doShowConfusions() {
        // Auto-classify current image if a trained classifier exists
        if ((predAll == null || predAll.size() == 0) && autoClassifyCallback != null) {
            if (autoClassifyCallback.get()) {
                // predAll was updated via setPredAll callback
            }
        }
        if (predAll == null || predAll.size() == 0 || classifier == null) {
            Dialogs.showErrorMessage(STRINGS.getString("name"), "No predictions available. Train first.");
            return;
        }

        List<String> classNames = classifier.getClassNames();
        String imgName = null;
        var project = qupath.getProject();
        var imageData = qupath.getImageData();
        if (project != null && imageData != null) {
            var entry = project.getEntry(imageData);
            if (entry != null) imgName = entry.getImageName();
        }
        var view = new ConfusionMatrixView(qupath.getStage(), predAll, classNames, labelStore, qupath, imgName);
        lastAgreementRates = view.getAgreementRates();
        if (onAgreementRatesChanged != null) onAgreementRatesChanged.accept(lastAgreementRates);
        view.show();
    }

    private void doShowFeatureImportance() {
        var imageData = qupath.getImageData();
        if (imageData == null || classifier == null) return;

        Collection<PathObject> detections = imageData.getHierarchy().getDetectionObjects();
        if (detections.isEmpty()) {
            Dialogs.showErrorMessage(STRINGS.getString("name"), "No detections found.");
            return;
        }

        List<String> featureNames = classifier.getFeatureNames();
        if (featureNames == null || featureNames.isEmpty()) return;

        // Raw features — importance is reported in the same space the classifier trains on.
        CellFeatureExtractor extractor = new CellFeatureExtractor(featureNames);

        showFeatureImportanceCheckBox.setDisable(true);
        Thread worker = new Thread(
                () -> {
                    try {
                        var result = classifier.computeFeatureImportance(detections, extractor);
                        Platform.runLater(() -> {
                            showFeatureImportanceCheckBox.setDisable(false);
                            new FeatureImportanceView(qupath.getStage(), result).show();
                        });
                    } catch (Exception ex) {
                        Platform.runLater(() -> {
                            showFeatureImportanceCheckBox.setDisable(false);
                            Dialogs.showErrorMessage(
                                    STRINGS.getString("name"), "Feature importance failed: " + ex.getMessage());
                        });
                    }
                },
                "CellTune-FeatureImportance");
        worker.setDaemon(true);
        worker.start();
    }

    private void doSampleAndReview() {
        // Auto-classify current image if a trained classifier exists
        if (predAll == null && autoClassifyCallback != null) {
            autoClassifyCallback.get();
        }
        if (classifier == null) return;

        // If no confusion matrix has been shown yet, compute agreement rates now
        if (predAll != null && predAll.size() > 0 && lastAgreementRates == null) {
            var view = new ConfusionMatrixView(qupath.getStage(), predAll, classifier.getClassNames());
            lastAgreementRates = view.getAgreementRates();
            if (onAgreementRatesChanged != null) onAgreementRatesChanged.accept(lastAgreementRates);
            view.show();
        }

        loadSamplingContextAsync(samplingContext -> {
            long disagreeCount = samplingContext.predictions().getDisagreementCount();
            if (disagreeCount == 0) {
                Dialogs.showInfoNotification(
                        STRINGS.getString("name"), "No disagreement cells available across saved project predictions.");
                return;
            }

            String countStr = Dialogs.showInputDialog(
                    STRINGS.getString("sample.dialog.title"),
                    STRINGS.getString("sample.count.label") + " (" + disagreeCount + " disagreements available)",
                    "200");
            if (countStr == null) return;

            int sampleSize;
            try {
                sampleSize = Integer.parseInt(countStr.strip());
            } catch (NumberFormatException e) {
                Dialogs.showErrorMessage(STRINGS.getString("name"), "Invalid number.");
                return;
            }
            if (sampleSize <= 0) return;

            if (!sampleForReviewBatch(samplingContext, sampleSize)) {
                Dialogs.showInfoNotification(
                        STRINGS.getString("name"),
                        "No eligible disagreement cells remained after excluding reviewed cells.");
                return;
            }

            int imageCount = new LinkedHashSet<>(lastSampledCellImageMap.values()).size();
            Dialogs.showInfoNotification(
                    STRINGS.getString("name"),
                    "Sampled " + lastSampledCellIds.size() + " cells across " + imageCount
                            + " image(s). Use 'Enter Review Mode' to start.");
        });
    }

    private void doEnterReview() {
        // Auto-classify current image if a trained classifier exists
        if (predAll == null && autoClassifyCallback != null) {
            autoClassifyCallback.get();
        }
        if (classifier == null) return;

        loadSamplingContextAsync(samplingContext -> {
            long disagreeCount = samplingContext.predictions().getDisagreementCount();
            if (disagreeCount == 0) {
                Dialogs.showInfoNotification(
                        STRINGS.getString("name"), "No disagreement cells available across saved project predictions.");
                return;
            }

            // Compute agreement rates for the current image if not yet available
            if (predAll != null && predAll.size() > 0 && lastAgreementRates == null) {
                var confView = new ConfusionMatrixView(qupath.getStage(), predAll, classifier.getClassNames());
                lastAgreementRates = confView.getAgreementRates();
                if (onAgreementRatesChanged != null) onAgreementRatesChanged.accept(lastAgreementRates);
            }

            String countStr = Dialogs.showInputDialog(
                    STRINGS.getString("sample.dialog.title"),
                    "How many disagreement cells to review?" + " (" + disagreeCount + " available)",
                    "200");
            if (countStr == null) return;

            int sampleSize;
            try {
                sampleSize = Integer.parseInt(countStr.strip());
            } catch (NumberFormatException e) {
                Dialogs.showErrorMessage(STRINGS.getString("name"), "Invalid number.");
                return;
            }
            if (sampleSize <= 0) return;

            if (!sampleForReviewBatch(samplingContext, sampleSize)) {
                Dialogs.showInfoNotification(
                        STRINGS.getString("name"),
                        "No eligible disagreement cells remained after excluding reviewed cells.");
                return;
            }

            if (lastSampledCellIds == null || lastSampledCellIds.isEmpty()) {
                Dialogs.showErrorMessage(STRINGS.getString("name"), STRINGS.getString("review.no_sample"));
                return;
            }

            PopulationSet reviewPredictions = (lastSampledPredictions != null && lastSampledPredictions.size() > 0)
                    ? lastSampledPredictions
                    : predAll;
            if (reviewPredictions == null || reviewPredictions.size() == 0) return;

            launchReviewStage(reviewPredictions);
        });
    }

    /**
     * Build the sampling context off the FX thread (so the UI doesn't freeze
     * while loading 49 prediction JSONs) and run {@code onReady} on the FX
     * thread once it's available. Shows a small modal progress dialog.
     */
    private void loadSamplingContextAsync(Consumer<SamplingContext> onReady) {
        var stage = new javafx.stage.Stage();
        stage.setTitle(STRINGS.getString("name") + " \u2014 Loading predictions");
        stage.initOwner(qupath.getStage());
        stage.initModality(javafx.stage.Modality.WINDOW_MODAL);
        stage.setResizable(false);

        var bar = new ProgressBar(0);
        bar.setPrefWidth(360);
        var status = new Label("Scanning project images\u2026");
        status.setMaxWidth(360);
        status.setWrapText(true);
        var box = new VBox(8, status, bar);
        box.setPadding(new Insets(15));
        stage.setScene(new javafx.scene.Scene(box));
        stage.setOnCloseRequest(e -> e.consume()); // disable manual close
        stage.show();

        Thread t = new Thread(
                () -> {
                    SamplingContext ctx = null;
                    try {
                        ctx = buildSamplingContext((done, total) -> Platform.runLater(() -> {
                            if (total > 0) {
                                bar.setProgress((double) done / total);
                                status.setText(String.format("Loading predictions: %d / %d images\u2026", done, total));
                            }
                        }));
                    } catch (Exception e) {
                        final Exception err = e;
                        Platform.runLater(() -> {
                            stage.close();
                            Dialogs.showErrorMessage(
                                    STRINGS.getString("name"), "Failed to load predictions: " + err.getMessage());
                        });
                        return;
                    }
                    final SamplingContext result = ctx;
                    Platform.runLater(() -> {
                        stage.close();
                        if (result != null) onReady.accept(result);
                    });
                },
                "celltune-sampling-load");
        t.setDaemon(true);
        t.start();
    }

    private void launchReviewStage(PopulationSet reviewPredictions) {
        // Automated review pre-extracts a small training-image tile around each sampled
        // cell so the user labels via in-memory crops instead of switching between full
        // 20k\u00d720k project images. Cell IDs round-trip to the original images via the
        // cellId\u2192imageName map (see persistReviewedLabelsByImage).
        extractTrainingTilesAsync(extractor -> launchReviewStageWithTiles(reviewPredictions, extractor));
    }

    private void extractTrainingTilesAsync(Consumer<TrainingTileExtractor> onReady) {
        var stage = new javafx.stage.Stage();
        stage.setTitle(STRINGS.getString("name") + " \u2014 Preparing review tiles");
        stage.initOwner(qupath.getStage());
        stage.initModality(javafx.stage.Modality.WINDOW_MODAL);
        stage.setResizable(false);
        var bar = new ProgressBar(0);
        bar.setPrefWidth(360);
        var status = new Label("Extracting cell tiles\u2026");
        status.setMaxWidth(360);
        status.setWrapText(true);
        var box = new VBox(8, status, bar);
        box.setPadding(new Insets(15));
        stage.setScene(new javafx.scene.Scene(box));
        stage.setOnCloseRequest(e -> e.consume());
        stage.show();

        final int total = lastSampledCellIds == null ? 0 : lastSampledCellIds.size();
        Thread t = new Thread(
                () -> {
                    TrainingTileExtractor extractor;
                    try {
                        extractor = TrainingTileExtractor.extract(
                                qupath,
                                lastSampledCellIds,
                                lastSampledCellImageMap,
                                done -> Platform.runLater(() -> {
                                    if (total > 0) {
                                        bar.setProgress((double) done / total);
                                        status.setText(
                                                String.format("Extracting tiles: %d / %d cells\u2026", done, total));
                                    }
                                }));
                    } catch (Exception e) {
                        final Exception err = e;
                        Platform.runLater(() -> {
                            stage.close();
                            Dialogs.showErrorMessage(
                                    STRINGS.getString("name"), "Failed to prepare review tiles: " + err.getMessage());
                        });
                        return;
                    }
                    Platform.runLater(() -> {
                        stage.close();
                        onReady.accept(extractor);
                    });
                },
                "celltune-tile-extract");
        t.setDaemon(true);
        t.start();
    }

    private void launchReviewStageWithTiles(PopulationSet reviewPredictions, TrainingTileExtractor extractor) {
        var reviewController = new ReviewController(
                qupath,
                lastSampledCellIds,
                reviewPredictions,
                lastSampledCellImageMap,
                lastSampledCellAnnotationsMap,
                extractor.getPreps());
        if (reviewController.size() == 0) {
            extractor.close();
            Dialogs.showErrorMessage(STRINGS.getString("name"), "Could not resolve sampled cells in project images.");
            return;
        }

        var channelSelector = new ChannelSelector(qupath, cellTypeTable);
        var toolbar = new ReviewToolbar(reviewController, cellTypeTable, channelSelector);

        var vbox = new javafx.scene.layout.VBox(6);
        vbox.setPadding(new javafx.geometry.Insets(6));
        vbox.getChildren().addAll(toolbar, channelSelector.getCheckBox(), channelSelector.getDisplayRangeCheckBox());

        var stage = new javafx.stage.Stage();
        stage.setTitle(STRINGS.getString("review.stage.title"));
        stage.initOwner(qupath.getStage());
        stage.setScene(new javafx.scene.Scene(vbox));
        stage.setAlwaysOnTop(true);
        stage.setResizable(true);
        stage.setMinWidth(750);

        stage.setOnHidden(e -> {
            reviewController.removeHighlight();
            var outputLabels = reviewController.getOutputLabels();
            if (outputLabels.size() > 0) {
                if (labelStore == null) labelStore = new LabelStore("CellTune");

                // Pull in any cellId → imageName mappings that the review
                // controller learned while labeling (e.g. manually-clicked
                // context cells inside a tile, which weren't in the sampled
                // queue). Without this, persistReviewedLabelsByImage would
                // fall back to currentImageName and route those labels to the
                // wrong file.
                var reviewCellImageMap = reviewController.getCellImageMap();
                if (reviewCellImageMap != null && !reviewCellImageMap.isEmpty()) {
                    if (!(lastSampledCellImageMap instanceof LinkedHashMap)) {
                        lastSampledCellImageMap = new LinkedHashMap<>(lastSampledCellImageMap);
                    }
                    for (var rEntry : reviewCellImageMap.entrySet()) {
                        if (rEntry.getKey() != null && rEntry.getValue() != null) {
                            lastSampledCellImageMap.putIfAbsent(rEntry.getKey(), rEntry.getValue());
                        }
                    }
                }

                // Split reviewed labels into current-image vs other-image so the
                // in-memory store only ever holds labels for the current image's
                // detections. Other-image labels still get persisted to disk and
                // are picked up by the supplementary pooling loop on next train.
                var currentImageData = qupath.getImageData();
                java.util.Set<String> currentImageDetIds = new java.util.HashSet<>();
                if (currentImageData != null) {
                    for (var det : currentImageData.getHierarchy().getDetectionObjects()) {
                        currentImageDetIds.add(det.getID().toString());
                    }
                }

                int mergedToCurrent = 0;
                int savedToOthers = 0;
                for (var entry : outputLabels.getAllLabels().entrySet()) {
                    if (currentImageDetIds.contains(entry.getKey())) {
                        labelStore.setLabel(entry.getKey(), entry.getValue());
                        mergedToCurrent++;
                    } else {
                        savedToOthers++;
                    }
                }

                refreshStats();
                if (onLabelStoreChanged != null) onLabelStoreChanged.accept(labelStore);

                persistReviewedLabelsByImage(outputLabels);

                final int mergedFinal = mergedToCurrent;
                final int savedFinal = savedToOthers;
                Dialogs.showInfoNotification(
                        STRINGS.getString("name"),
                        String.format(
                                "Review complete: %d labels in current image, "
                                        + "%d saved to other project images (pooled on next train).",
                                mergedFinal, savedFinal));
            }
            extractor.close();
        });

        stage.show();
    }

    // ── Helpers ──

    private void refreshStats() {
        int labelCount = (labelStore != null) ? labelStore.size() : 0;
        int predCount = (predAll != null) ? predAll.size() : 0;
        int importedCount = (importedTrainingRows != null) ? importedTrainingRows.size() : 0;
        labelCountLabel.setText("Labels: " + labelCount);
        predCountLabel.setText("Predictions: " + predCount);
        importedCountLabel.setText("Imported rows: " + importedCount);
        clearImportedButton.setDisable(importedCount == 0);
    }

    private void collectLabelsFromAnnotations(LabelStore store) {
        var imageData = qupath.getImageData();
        if (imageData == null) return;
        // In binary mode, restrict to the active binary classifier's classes
        // so labels from other classifiers (e.g. PD-1+/-) don't get pulled into
        // this store via point annotations left over from a previous session.
        collectLabelsFromHierarchy(imageData.getHierarchy(), store, activeBinaryClassNames);
    }

    private static void collectLabelsFromHierarchy(
            qupath.lib.objects.hierarchy.PathObjectHierarchy hierarchy,
            LabelStore store,
            java.util.Set<String> allowedClasses) {
        // Delegate to the shared collector so this training-time label collection
        // preserves merge history (previously this copy overwrote merged labels
        // with the bare PathClass — see AnnotationLabelCollector).
        AnnotationLabelCollector.collect(hierarchy, store, allowedClasses);
    }

    /**
     * Create a copy of the label store containing only cell IDs that exist
     * as detections in the given image.
     */
    private static LabelStore filterLabelStoreToImage(
            LabelStore store, qupath.lib.images.ImageData<java.awt.image.BufferedImage> imageData) {
        var detections = imageData.getHierarchy().getDetectionObjects();
        var validIds = new java.util.HashSet<String>(detections.size());
        for (var det : detections) {
            validIds.add(det.getID().toString());
        }
        var filtered = new LabelStore(store.getName());
        for (var entry : store.getAllLabels().entrySet()) {
            if (validIds.contains(entry.getKey())) {
                filtered.setLabel(entry.getKey(), entry.getValue());
            }
        }
        return filtered;
    }

    private static final class SamplingContext {
        private final PopulationSet predictions;
        private final Map<String, String> cellToImage;
        private final Map<String, List<String>> cellToAnnotations;

        private SamplingContext(
                PopulationSet predictions,
                Map<String, String> cellToImage,
                Map<String, List<String>> cellToAnnotations) {
            this.predictions = predictions;
            this.cellToImage = cellToImage;
            this.cellToAnnotations = cellToAnnotations;
        }

        private PopulationSet predictions() {
            return predictions;
        }

        private Map<String, String> cellToImage() {
            return cellToImage;
        }

        private Map<String, List<String>> cellToAnnotations() {
            return cellToAnnotations;
        }
    }

    private SamplingContext buildSamplingContext() {
        return buildSamplingContext(null);
    }

    /**
     * Build the cross-image sampling pool. Loads each image's saved
     * PopulationSet JSON in parallel (across all available CPU cores) so that
     * entering review mode on a project with dozens of images stays responsive.
     *
     * @param progressCallback optional callback invoked from worker threads as
     *     each image finishes loading; receives (completed, total). Use
     *     Platform.runLater inside the callback if updating UI.
     */
    private SamplingContext buildSamplingContext(java.util.function.BiConsumer<Integer, Integer> progressCallback) {
        PopulationSet pooled = new PopulationSet("Pred_ALL");
        Map<String, String> cellToImage = new LinkedHashMap<>();
        // Per-cell annotation names captured at sample time, while we still
        // have the source hierarchy loaded. Lets review-mode display annotation
        // membership for cross-image cells without re-resolving hierarchies.
        Map<String, List<String>> cellToAnnotations = new LinkedHashMap<>();

        var project = qupath.getProject();
        var imageData = qupath.getImageData();
        String currentImageName = null;
        if (project != null && imageData != null) {
            var entry = project.getEntry(imageData);
            if (entry != null) {
                currentImageName = entry.getImageName();
            }
        }
        final String currentImageNameFinal = currentImageName;

        // Build optional annotation-keyword filter. When keywords are present,
        // only cells whose centroid lies inside an annotation whose name (or
        // PathClass) contains one of the keywords are eligible.
        List<String> annotationKeywords = parseAnnotationKeywords();
        boolean annotationFilterActive = !annotationKeywords.isEmpty();

        // Resolve current image's allowed-cells map (and annotation names per
        // cell). For the current image we always compute it because the
        // hierarchy is free — we just need to walk annotations once.
        Map<String, List<String>> currentImageAnnotations =
                (imageData != null) ? mapCellsToContainingAnnotations(imageData, annotationKeywords) : Map.of();
        Set<String> currentImageFilterIds = annotationFilterActive ? currentImageAnnotations.keySet() : null;

        if (predAll != null && predAll.size() > 0) {
            addPredictionsToSamplingPool(
                    pooled,
                    predAll,
                    currentImageNameFinal,
                    cellToImage,
                    currentImageFilterIds,
                    currentImageAnnotations,
                    cellToAnnotations);
        } else if (project != null && currentImageNameFinal != null) {
            try {
                var loadedCurrent = ProjectStateManager.loadImagePredictions(project, currentImageNameFinal);
                if (loadedCurrent != null && loadedCurrent.size() > 0) {
                    addPredictionsToSamplingPool(
                            pooled,
                            loadedCurrent,
                            currentImageNameFinal,
                            cellToImage,
                            currentImageFilterIds,
                            currentImageAnnotations,
                            cellToAnnotations);
                }
            } catch (Exception ignored) {
            }
        }

        if (project != null) {
            if (currentImageOnlyCheckBox.isSelected()) {
                if (progressCallback != null) progressCallback.accept(0, 0);
                return new SamplingContext(pooled, cellToImage, cellToAnnotations);
            }
            @SuppressWarnings("unchecked")
            var entries = (List<ProjectImageEntry<BufferedImage>>) (List<?>) project.getImageList();
            Map<String, ProjectImageEntry<BufferedImage>> entryByName = new LinkedHashMap<>();
            for (var e : entries) {
                if (e != null && e.getImageName() != null) entryByName.put(e.getImageName(), e);
            }
            List<String> otherNames = entries.stream()
                    .filter(e -> e != null && e.getImageName() != null)
                    .map(ProjectImageEntry::getImageName)
                    .filter(n -> currentImageNameFinal == null || !currentImageNameFinal.equals(n))
                    .collect(Collectors.toList());
            int total = otherNames.size();
            if (progressCallback != null) progressCallback.accept(0, total);
            java.util.concurrent.atomic.AtomicInteger done = new java.util.concurrent.atomic.AtomicInteger();

            // Parallel JSON load — independent files, CPU+IO bound, safe to fan out.
            List<Map.Entry<String, PopulationSet>> loaded = otherNames.parallelStream()
                    .map(name -> {
                        PopulationSet ps = null;
                        try {
                            ps = ProjectStateManager.loadImagePredictions(project, name);
                        } catch (Exception ignored) {
                        }
                        int c = done.incrementAndGet();
                        if (progressCallback != null) progressCallback.accept(c, total);
                        return new AbstractMap.SimpleEntry<>(name, ps);
                    })
                    .collect(Collectors.toList());

            // When the filter is active, load each image's hierarchy in
            // parallel (capped) and capture both filter IDs and the
            // cell→annotation-names map. readImageData() returns a fresh
            // ImageData per call so per-thread instances are independent.
            Map<String, Map<String, List<String>>> annotationsByImage;
            if (annotationFilterActive) {
                int hierarchyThreads =
                        Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors() / 2));
                java.util.concurrent.ForkJoinPool hierPool = new java.util.concurrent.ForkJoinPool(hierarchyThreads);
                try {
                    annotationsByImage = hierPool.submit(() -> loaded.parallelStream()
                                    .filter(e ->
                                            e.getValue() != null && e.getValue().size() > 0)
                                    .map(e -> {
                                        var entry = entryByName.get(e.getKey());
                                        if (entry == null) return null;
                                        try {
                                            var otherImageData = entry.readImageData();
                                            Map<String, List<String>> m =
                                                    mapCellsToContainingAnnotations(otherImageData, annotationKeywords);
                                            return m.isEmpty() ? null : new AbstractMap.SimpleEntry<>(e.getKey(), m);
                                        } catch (Exception ex) {
                                            return null;
                                        }
                                    })
                                    .filter(java.util.Objects::nonNull)
                                    .collect(Collectors.toMap(
                                            Map.Entry::getKey, Map.Entry::getValue, (a, b) -> a, LinkedHashMap::new)))
                            .get();
                } catch (Exception ex) {
                    annotationsByImage = Map.of();
                } finally {
                    hierPool.shutdown();
                }
            } else {
                annotationsByImage = null;
            }

            // Merge serially to keep dedup/order deterministic.
            for (var e : loaded) {
                PopulationSet ps = e.getValue();
                if (ps == null || ps.size() == 0) continue;
                Set<String> allowedIds = null;
                Map<String, List<String>> annoMap = null;
                if (annotationFilterActive) {
                    annoMap = annotationsByImage.get(e.getKey());
                    if (annoMap == null) continue; // no matching annotations in this image
                    allowedIds = annoMap.keySet();
                }
                addPredictionsToSamplingPool(
                        pooled, ps, e.getKey(), cellToImage, allowedIds, annoMap, cellToAnnotations);
            }
        }

        return new SamplingContext(pooled, cellToImage, cellToAnnotations);
    }

    /** Build a small titled wrapper for the annotation keyword field. */
    private VBox buildAnnotationFilterBox() {
        Label title = new Label("Specify annotations");
        title.setStyle("-fx-font-size: 11px; -fx-text-fill: #555;");
        VBox box = new VBox(2, title, annotationKeywordField);
        return box;
    }

    /** Parse the annotation-keyword text field into a non-null list of trimmed, non-empty keywords. */
    private List<String> parseAnnotationKeywords() {
        return AnnotationCellFilter.parseKeywords(annotationKeywordField.getText());
    }

    /**
     * Return the set of detection IDs in {@code imageData} whose centroid lies
     * inside any annotation whose name contains one of the keywords
     * (case-insensitive substring match). Returns an empty set if no annotation
     * matches — caller treats that as "no eligible cells".
     * /**
     * Map detection IDs in {@code imageData} to the labels of all annotations
     * (matched by keyword substring; empty keyword list means "all named
     * annotations") whose ROI geometrically contains the detection's
     * centroid. This is computed at sample time so review-mode display does
     * not depend on hierarchy parent/child resolution. Returns an empty map
     * if no eligible annotations exist.
     */
    private static Map<String, List<String>> mapCellsToContainingAnnotations(
            qupath.lib.images.ImageData<BufferedImage> imageData, List<String> keywords) {
        var hierarchy = imageData.getHierarchy();
        boolean filtering = keywords != null && !keywords.isEmpty();
        List<String> lowerKeywords = filtering
                ? keywords.stream().map(k -> k.toLowerCase(Locale.ROOT)).collect(Collectors.toList())
                : List.of();

        // (annotation, displayLabel) pairs we'll test centroids against.
        List<Map.Entry<PathObject, String>> matchingAnnotations = new ArrayList<>();
        for (PathObject anno : hierarchy.getAnnotationObjects()) {
            if (anno.getROI() == null) continue;
            String label = annotationDisplayLabel(anno);
            if (label == null) continue;
            if (filtering) {
                String lower = label.toLowerCase(Locale.ROOT);
                if (lowerKeywords.stream().noneMatch(lower::contains)) continue;
            }
            matchingAnnotations.add(new AbstractMap.SimpleEntry<>(anno, label));
        }

        Map<String, List<String>> out = new LinkedHashMap<>();
        if (matchingAnnotations.isEmpty()) return out;

        for (PathObject det : hierarchy.getDetectionObjects()) {
            var roi = det.getROI();
            if (roi == null) continue;
            double cx = roi.getCentroidX();
            double cy = roi.getCentroidY();
            List<String> names = null;
            for (var pair : matchingAnnotations) {
                if (pair.getKey().getROI().contains(cx, cy)) {
                    if (names == null) names = new ArrayList<>(2);
                    if (!names.contains(pair.getValue())) names.add(pair.getValue());
                }
            }
            if (names != null) {
                out.put(det.getID().toString(), names);
            }
        }
        return out;
    }

    /**
     * Display label for an annotation: explicit name if set, otherwise the
     * PathClass name. Returns null only when both are absent.
     */
    static String annotationDisplayLabel(PathObject anno) {
        return AnnotationCellFilter.displayLabel(anno);
    }

    private static void addPredictionsToSamplingPool(
            PopulationSet pooled,
            PopulationSet source,
            String imageName,
            Map<String, String> cellToImage,
            Set<String> allowedCellIds,
            Map<String, List<String>> sourceAnnotations,
            Map<String, List<String>> cellToAnnotations) {
        if (source == null || source.size() == 0) return;
        String safeImageName = (imageName == null || imageName.isBlank()) ? "image" : imageName;

        for (var entry : source.getAll().entrySet()) {
            String cellId = entry.getKey();
            if (cellId == null || cellId.isBlank()) continue;
            if (allowedCellIds != null && !allowedCellIds.contains(cellId)) continue;
            if (pooled.get(cellId) != null) continue;

            pooled.put(cellId, entry.getValue());
            cellToImage.put(cellId, safeImageName);
            if (sourceAnnotations != null) {
                List<String> annos = sourceAnnotations.get(cellId);
                if (annos != null && !annos.isEmpty()) {
                    cellToAnnotations.put(cellId, annos);
                }
            }
        }
    }

    private boolean sampleForReviewBatch(SamplingContext samplingContext, int sampleSize) {
        if (samplingContext == null
                || samplingContext.predictions() == null
                || samplingContext.predictions().size() == 0
                || sampleSize <= 0) {
            return false;
        }

        Set<String> reviewedCellIds =
                buildReviewedCellIdsForSampling(qupath, activeBinaryMarker, labelStore, lastSampledCellIds);
        lastSampledCellIds = UncertaintySampler.sample(
                samplingContext.predictions(),
                classifier.getClassNames(),
                lastAgreementRates,
                sampleSize,
                List.of(),
                List.of(),
                samplingContext.cellToImage(),
                reviewedCellIds,
                new Random());

        if (lastSampledCellIds == null) {
            lastSampledCellIds = List.of();
        }

        lastSampledCellImageMap = new LinkedHashMap<>();
        lastSampledCellAnnotationsMap = new LinkedHashMap<>();
        for (String id : lastSampledCellIds) {
            String imageName = samplingContext.cellToImage().get(id);
            if (imageName != null) {
                lastSampledCellImageMap.put(id, imageName);
            }
            List<String> annos = samplingContext.cellToAnnotations().get(id);
            if (annos != null && !annos.isEmpty()) {
                lastSampledCellAnnotationsMap.put(id, annos);
            }
        }
        lastSampledPredictions = samplingContext.predictions();

        reviewButton.setDisable(lastSampledCellIds.isEmpty());
        if (onSampledCellsChanged != null) {
            onSampledCellsChanged.accept(lastSampledCellIds);
        }

        persistCurrentImageSampledIds();
        return !lastSampledCellIds.isEmpty();
    }

    private void persistCurrentImageSampledIds() {
        var project = qupath.getProject();
        var imageData = qupath.getImageData();
        if (project == null || imageData == null || lastSampledCellIds == null || lastSampledCellIds.isEmpty()) {
            return;
        }

        var imgEntry = project.getEntry(imageData);
        if (imgEntry == null) return;

        String currentImageName = imgEntry.getImageName();
        List<String> currentImageOnlyIds = new ArrayList<>();
        for (String id : lastSampledCellIds) {
            String imageName = lastSampledCellImageMap.get(id);
            if (imageName == null || imageName.equals(currentImageName)) {
                currentImageOnlyIds.add(id);
            }
        }

        if (currentImageOnlyIds.isEmpty()) return;

        try {
            ProjectStateManager.saveImageSampledCells(project, currentImageName, currentImageOnlyIds);
        } catch (Exception ignored) {
        }
    }

    private void persistReviewedLabelsByImage(LabelStore reviewedLabels) {
        if (reviewedLabels == null || reviewedLabels.size() == 0) return;

        var project = qupath.getProject();
        if (project == null) return;

        String currentImageName = null;
        var currentImageData = qupath.getImageData();
        if (currentImageData != null) {
            var currentEntry = project.getEntry(currentImageData);
            if (currentEntry != null) {
                currentImageName = currentEntry.getImageName();
            }
        }

        Map<String, LabelStore> labelsByImage = new LinkedHashMap<>();
        for (var entry : reviewedLabels.getAllLabels().entrySet()) {
            String cellId = entry.getKey();
            String label = entry.getValue();
            if (cellId == null || label == null) continue;

            String imageName = lastSampledCellImageMap.get(cellId);
            if ((imageName == null || imageName.isBlank()) && currentImageName != null) {
                imageName = currentImageName;
            }
            if (imageName == null || imageName.isBlank()) continue;

            labelsByImage
                    .computeIfAbsent(imageName, ignored -> new LabelStore("CellTune"))
                    .setLabel(cellId, label);
        }

        for (var entry : labelsByImage.entrySet()) {
            String imageName = entry.getKey();
            LabelStore delta = entry.getValue();

            try {
                // In binary mode, strip foreign-class labels from the delta before
                // it touches disk, and from the merged on-disk store after merge.
                // The post-merge filter is self-healing: any historical contamination
                // (foreign labels written by older buggy code) gets cleaned up on
                // the next save.
                java.util.Set<String> binaryFilter = null;
                if (activeBinaryMarker != null && activeBinaryClassNames != null && !activeBinaryClassNames.isEmpty()) {
                    binaryFilter = new java.util.LinkedHashSet<>(activeBinaryClassNames);
                    delta.retainClasses(binaryFilter);
                    if (delta.size() == 0) continue;
                }

                LabelStore merged = ProjectStateManager.loadImageLabels(project, activeBinaryMarker, imageName);
                if (merged == null) {
                    merged = new LabelStore("CellTune");
                }
                merged.mergeFrom(delta);
                if (binaryFilter != null) {
                    merged.retainClasses(binaryFilter);
                }
                ProjectStateManager.saveImageLabels(project, activeBinaryMarker, imageName, merged);
            } catch (Exception ignored) {
            }
        }
    }

    private static Set<String> buildReviewedCellIdsForSampling(
            QuPathGUI qupath, String scope, LabelStore labels, List<String> previouslySampledIds) {
        Set<String> reviewed = new LinkedHashSet<>();
        if (labels != null) {
            reviewed.addAll(labels.getAllLabels().keySet());
        }
        if (previouslySampledIds != null) {
            reviewed.addAll(previouslySampledIds);
        }

        if (qupath != null && qupath.getProject() != null) {
            @SuppressWarnings("unchecked")
            var entries = (List<ProjectImageEntry<BufferedImage>>)
                    (List<?>) qupath.getProject().getImageList();
            for (var entry : entries) {
                if (entry == null || entry.getImageName() == null) continue;
                try {
                    LabelStore imageLabels =
                            ProjectStateManager.loadImageLabels(qupath.getProject(), scope, entry.getImageName());
                    if (imageLabels != null) {
                        reviewed.addAll(imageLabels.getAllLabels().keySet());
                    }
                } catch (Exception ignored) {
                }
            }
        }

        return reviewed;
    }
}
