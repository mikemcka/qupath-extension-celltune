package qupath.ext.celltune;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.ResourceBundle;
import java.util.Set;
import javafx.application.Platform;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.IntegerProperty;
import javafx.geometry.Insets;
import javafx.scene.control.ButtonType;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Dialog;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.VBox;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.celltune.classifier.ClassifierState;
import qupath.ext.celltune.classifier.DualModelClassifier;
import qupath.ext.celltune.classifier.UncertaintySampler;
import qupath.ext.celltune.classifier.XGBoostModel;
import qupath.ext.celltune.io.GroundTruthIO;
import qupath.ext.celltune.io.ProjectStateManager;
import qupath.ext.celltune.model.AnnotationLabelCollector;
import qupath.ext.celltune.model.CellFeatureExtractor;
import qupath.ext.celltune.model.CellTypeTable;
import qupath.ext.celltune.model.FeatureNormalizer;
import qupath.ext.celltune.model.LabelStore;
import qupath.ext.celltune.model.PopulationSet;
import qupath.ext.celltune.ui.ChannelSelector;
import qupath.ext.celltune.ui.ClassControlDialog;
import qupath.ext.celltune.ui.ClassificationPanel;
import qupath.ext.celltune.ui.ClusteringNormalizationPane;
import qupath.ext.celltune.ui.CompositeClassificationDialog;
import qupath.ext.celltune.ui.ConfusionMatrixView;
import qupath.ext.celltune.ui.FeatureSelectionPane;
import qupath.ext.celltune.ui.ImageSelectionPane;
import qupath.ext.celltune.ui.ManualLabelToolbar;
import qupath.ext.celltune.ui.ReviewController;
import qupath.ext.celltune.ui.ReviewToolbar;
import qupath.ext.celltune.ui.TrainingTileExtractor;
import qupath.ext.celltune.util.JvmModuleOpener;
import qupath.ext.celltune.util.TrainingThreads;
import qupath.fx.dialogs.Dialogs;
import qupath.fx.prefs.controlsfx.PropertyItemBuilder;
import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.prefs.PathPrefs;
import qupath.lib.objects.PathObject;
import qupath.lib.projects.ProjectImageEntry;

/**
 * QuPath extension that provides CellTune-style active learning cell classification.
 * <p>
 * Uses dual gradient-boosted models (XGBoost + LightGBM) to identify disagreement
 * cells, then presents them for human review in an iterative loop that progressively
 * improves classification accuracy.
 * <p>
 * Workflow: Landmark → Train → Confusions → Sample → Review → Retrain
 */
public class CellTuneExtension implements QuPathExtension, BinaryClassifierManager.Host {

    private static final ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.celltune.ui.strings");
    private static final Logger logger = LoggerFactory.getLogger(CellTuneExtension.class);

    private static final String EXTENSION_NAME = resources.getString("name");
    private static final String EXTENSION_DESCRIPTION = resources.getString("description");
    private static final Version EXTENSION_QUPATH_VERSION = Version.parse("v0.7.0");

    /** Persistent preference — lets users disable the extension from the Preferences pane. */
    private static final BooleanProperty enableExtensionProperty =
            PathPrefs.createPersistentPreference("celltune.enabled", true);

    /**
     * Persistent preference — caps how many threads the ML training path uses. 0 means "auto"
     * (every available processor), matching the behaviour before this preference existed. Useful
     * on a shared workstation, where letting two native libraries each grab every core makes
     * everything else unusable while a model trains.
     */
    private static final IntegerProperty trainingThreadsProperty =
            PathPrefs.createPersistentPreference("celltune.trainingThreads", 0);

    /**
     * Persistent preference — histogram bins per feature for XGBoost. 0 means "XGBoost's default"
     * (256), which is what this shipped with, so leaving it alone changes nothing.
     * <p>
     * Deliberately a preference rather than a control on the training panel: it is the one knob
     * left that meaningfully changes training time (~2x at 128 bins, ~2.5x at 64 — see
     * {@link XGBoostModel#setMaxBin}), but it <b>changes predictions</b>, so it should be a
     * considered decision made once and then diffed, not something adjusted between runs.
     */
    private static final IntegerProperty xgbMaxBinProperty =
            PathPrefs.createPersistentPreference("celltune.xgbMaxBin", 0);

    /**
     * When on, clustering and classifier feature extraction stream the persisted UniFORM
     * batch-correction gains (from {@code <project>/celltune/batch-shifts.json}) — each cell's
     * measurements are multiplied by its image's per-channel scale before use, so no corrected
     * columns need be written. Off by default; fitting scales does nothing until this is enabled.
     */
    private static final BooleanProperty useBatchCorrectionProperty =
            PathPrefs.createPersistentPreference("celltune.useBatchCorrection", false);

    /** @return the persistent "stream batch-corrected values into clustering + ML" toggle. */
    public static BooleanProperty useBatchCorrectionProperty() {
        return useBatchCorrectionProperty;
    }

    /** @return whether batch-corrected values should be streamed into feature extraction. */
    public static boolean useBatchCorrection() {
        return useBatchCorrectionProperty.get();
    }

    static {
        TrainingThreads.setOverride(trainingThreadsProperty.get());
        trainingThreadsProperty.addListener((v, o, n) -> TrainingThreads.setOverride(n == null ? 0 : n.intValue()));
        XGBoostModel.setMaxBin(xgbMaxBinProperty.get());
        xgbMaxBinProperty.addListener((v, o, n) -> XGBoostModel.setMaxBin(n == null ? 0 : n.intValue()));
    }

    /** @return the persistent training-thread cap (0 = auto) */
    public static IntegerProperty trainingThreadsProperty() {
        return trainingThreadsProperty;
    }

    /** @return the persistent XGBoost histogram bin count (0 = XGBoost's default of 256) */
    public static IntegerProperty xgbMaxBinProperty() {
        return xgbMaxBinProperty;
    }

    private boolean isInstalled = false;

    /** Listener for image changes — stored so it can be removed if needed. */
    private javafx.beans.value.ChangeListener<qupath.lib.images.ImageData<java.awt.image.BufferedImage>>
            imageDataListener;

    /** Shared extension state — populated as the user works. */
    private CellTypeTable cellTypeTable;

    private LabelStore labelStore;
    private PopulationSet predAll;
    private DualModelClassifier classifier;
    /** User-selected feature subset; null means use all features. */
    private List<String> selectedFeatures;
    /** Per-feature normalization config; null means no transforms. */
    private FeatureNormalizer clusteringNormalizer;
    /** Per-class agreement rates from the last confusion matrix. */
    private double[] lastAgreementRates;
    /** Cell IDs sampled for review. */
    private List<String> lastSampledCellIds;
    /** Sampled cell ID to image name map for cross-image review. */
    private Map<String, String> lastSampledCellImageMap = Map.of();
    /** Prediction pool used to generate lastSampledCellIds. */
    private PopulationSet lastSampledPredictions;
    /** Training rows imported from CSV (feature vectors + labels). */
    private List<GroundTruthIO.TrainingRow> importedTrainingRows;
    /** Feature names associated with importedTrainingRows. */
    private List<String> importedTrainingFeatureNames;
    /** Docked classification panel (Phase 7). */
    private ClassificationPanel classificationPanel;
    /** TitledPane wrapping the docked panel — stored so binary mode can expand it. */
    private javafx.scene.control.TitledPane dockPane;
    /** Tab holding the docked panel — stored so binary mode can select it. */
    private javafx.scene.control.Tab dockTab;

    // ── Binary classifier state ────────────────────────────────────────────────
    /** Owns binary-mode transitions + the pre-binary snapshot and marker registry. */
    private final BinaryClassifierManager binaryManager = new BinaryClassifierManager(this);
    /** Name of the currently active binary classifier, or null if in multi-class mode. */
    private String activeBinaryMarker = null;
    /** Class names resolved from saved state or label store for the active binary classifier. */
    private List<String> activeBinaryClassNames = null;
    /** Images pre-selected via "Apply to Images..." button; used in next training run. */
    private List<String> binaryTargetImages = new ArrayList<>();

    // ── QuPathExtension API ────────────────────────────────────────────────────

    @Override
    public void installExtension(QuPathGUI qupath) {
        if (isInstalled) {
            logger.debug("{} is already installed", getName());
            return;
        }
        isInstalled = true;
        // Open java.base/java.lang so Smile's native PCA/UMAP (loaded via JavaCPP)
        // work without the user setting --add-opens. Must run before the cell
        // scatter plot first touches a Smile native class.
        JvmModuleOpener.ensureJavaLangOpen();
        suppressImageTypePromptOnce();
        addPreferenceToPane(qupath);
        addMenuItems(qupath);
        dockClassificationPanel(qupath);
        checkHeapMemory();
    }

    @Override
    public String getName() {
        return EXTENSION_NAME;
    }

    @Override
    public String getDescription() {
        return EXTENSION_DESCRIPTION;
    }

    @Override
    public Version getQuPathVersion() {
        return EXTENSION_QUPATH_VERSION;
    }

    @Override
    public Version getVersion() {
        var v = getClass().getPackage().getImplementationVersion();
        return (v == null) ? QuPathExtension.super.getVersion() : Version.parse(v);
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    private void dockClassificationPanel(QuPathGUI qupath) {
        classificationPanel = new ClassificationPanel(qupath);
        classificationPanel.setLabelStore(labelStore);
        classificationPanel.setClassifier(classifier);
        classificationPanel.setSelectedFeatures(selectedFeatures);
        classificationPanel.setCellTypeTable(cellTypeTable);
        classificationPanel.setPredAll(predAll);
        classificationPanel.setImportedTrainingData(importedTrainingRows, importedTrainingFeatureNames);

        // Sync state back from panel to extension
        classificationPanel.setOnLabelStoreChanged(ls -> this.labelStore = ls);
        classificationPanel.setOnPredAllChanged(pa -> {
            this.predAll = pa;
            persistCurrentImagePredictions(qupath);
        });
        classificationPanel.setOnAgreementRatesChanged(ar -> this.lastAgreementRates = ar);
        classificationPanel.setOnSampledCellsChanged(ids -> {
            this.lastSampledCellIds = ids;
            this.lastSampledCellImageMap = Map.of();
            this.lastSampledPredictions = null;
        });
        classificationPanel.setOnClassifierChanged(cls -> {
            this.classifier = cls;
            // Auto-save binary classifier state after training completes in binary mode
            if (activeBinaryMarker != null && cls != null && cls.isTrained()) {
                var proj = qupath.getProject();
                if (proj != null) {
                    try {
                        ClassifierState state = cls.toClassifierState(activeBinaryMarker);
                        // Defence-in-depth: drop labels for classes outside this binary
                        // classifier before writing the canonical state file.
                        LabelStore safeStore = labelStore != null ? labelStore : new LabelStore(activeBinaryMarker);
                        if (activeBinaryClassNames != null && !activeBinaryClassNames.isEmpty()) {
                            safeStore.retainClasses(new java.util.LinkedHashSet<>(activeBinaryClassNames));
                        }
                        ProjectStateManager.saveBinaryState(
                                proj,
                                activeBinaryMarker,
                                safeStore,
                                state.getFeatureNames(),
                                state.getClassNames(),
                                state.getXgboostBytes(),
                                state.getLightgbmBytes(),
                                state.getRfModel1Bytes(),
                                state.getRfModel2Bytes(),
                                state.getModel1Type(),
                                state.getModel2Type());
                        logger.info("[CellTune] Auto-saved binary classifier state for '{}'", activeBinaryMarker);
                    } catch (Exception ex) {
                        logger.warn(
                                "Failed to auto-save binary state for '{}': {}", activeBinaryMarker, ex.getMessage());
                    }
                }
            }
        });
        classificationPanel.setAutoClassifyCallback(() -> autoClassifyCurrentImage(qupath));
        classificationPanel.setOnExitBinaryMode(() -> exitBinaryMode(qupath));
        classificationPanel.setOnApplyToImages(() -> applyBinaryClassifierToImages(qupath));
        classificationPanel.setOnManualLabelMode(() -> showManualLabelMode(qupath));
        classificationPanel.setOnFeatureImportance(() -> showFeatureImportance(qupath));
        classificationPanel.setOnClearImportedData(() -> clearImportedTrainingData(qupath));
        classificationPanel.setBinaryTargetImagesSupplier(
                () -> binaryTargetImages == null ? List.of() : new ArrayList<>(binaryTargetImages));

        // Listen for image changes so we can save/reset/load state per image.
        // Store the listener reference so it can be removed if the extension is ever reloaded.
        imageDataListener = (obs, oldData, newData) -> handleImageChange(qupath, oldData, newData);
        qupath.imageDataProperty().addListener(imageDataListener);

        // Dock into QuPath's analysis pane. Wrap the panel in a ScrollPane so
        // the user can reach widgets that exceed the tab's visible height.
        javafx.scene.control.ScrollPane scrollPane = new javafx.scene.control.ScrollPane(classificationPanel);
        scrollPane.setFitToWidth(true);
        scrollPane.setHbarPolicy(javafx.scene.control.ScrollPane.ScrollBarPolicy.NEVER);
        scrollPane.setVbarPolicy(javafx.scene.control.ScrollPane.ScrollBarPolicy.AS_NEEDED);
        scrollPane.setPannable(false);
        scrollPane.getStyleClass().add("celltune-dock-scroll");

        dockPane = new javafx.scene.control.TitledPane(EXTENSION_NAME, scrollPane);
        dockPane.setCollapsible(true);
        dockPane.setExpanded(false);
        dockTab = new javafx.scene.control.Tab(EXTENSION_NAME, dockPane);
        dockTab.setClosable(false);
        qupath.getAnalysisTabPane().getTabs().add(dockTab);
    }

    /**
     * Called when the user switches images. Saves labels for the old image
     * (filtered to its detections), resets transient prediction state, and
     * loads any persisted labels for the new image.
     */
    private void handleImageChange(
            QuPathGUI qupath,
            qupath.lib.images.ImageData<BufferedImage> oldData,
            qupath.lib.images.ImageData<BufferedImage> newData) {
        if (ReviewController.isReviewSessionActive()) {
            logger.debug("Skipping image-change state sync during active review navigation");
            return;
        }

        var project = qupath.getProject();

        // ── Save labels for the OLD image (filtered) ──
        if (oldData != null && project != null) {
            var oldEntry = project.getEntry(oldData);
            if (oldEntry != null) {
                // Save labels
                if (labelStore != null && labelStore.size() > 0) {
                    var filteredStore = filterLabelStoreToImage(labelStore, oldData);
                    // In binary mode, drop any labels for classes that don't belong
                    // to this binary classifier so foreign labels (e.g. PD-1+/- in a
                    // GrB store) don't get persisted to per-image label files.
                    if (activeBinaryMarker != null
                            && activeBinaryClassNames != null
                            && !activeBinaryClassNames.isEmpty()) {
                        filteredStore.retainClasses(new java.util.LinkedHashSet<>(activeBinaryClassNames));
                    }
                    if (filteredStore.size() > 0) {
                        try {
                            ProjectStateManager.saveImageLabels(
                                    project, activeBinaryMarker, oldEntry.getImageName(), filteredStore);
                        } catch (IOException ex) {
                            logger.warn(
                                    "Failed to save labels for {} on image switch: {}",
                                    oldEntry.getImageName(),
                                    ex.getMessage());
                        }
                    }
                }
                // Save sampled cell IDs (independent of labels)
                if (lastSampledCellIds != null && !lastSampledCellIds.isEmpty()) {
                    List<String> oldImageOnlyIds = new ArrayList<>();
                    for (String id : lastSampledCellIds) {
                        String imageName = lastSampledCellImageMap.get(id);
                        if (imageName == null || imageName.equals(oldEntry.getImageName())) {
                            oldImageOnlyIds.add(id);
                        }
                    }
                    if (!oldImageOnlyIds.isEmpty()) {
                        try {
                            ProjectStateManager.saveImageSampledCells(
                                    project, oldEntry.getImageName(), oldImageOnlyIds);
                        } catch (IOException ex) {
                            logger.warn(
                                    "Failed to save sampled cells for {} on image switch: {}",
                                    oldEntry.getImageName(),
                                    ex.getMessage());
                        }
                    }
                }

                // Save predictions for the old image so confidence is available
                // immediately when returning to it.
                if (predAll != null && predAll.size() > 0) {
                    try {
                        ProjectStateManager.saveImagePredictions(project, oldEntry.getImageName(), predAll);
                    } catch (IOException ex) {
                        logger.warn(
                                "Failed to save predictions for {} on image switch: {}",
                                oldEntry.getImageName(),
                                ex.getMessage());
                    }
                }
            }
        }

        // ── Reset transient state ──
        this.predAll = null;
        this.lastAgreementRates = null;
        this.lastSampledCellIds = null;
        this.lastSampledCellImageMap = Map.of();
        this.lastSampledPredictions = null;

        // ── Load labels for the NEW image (if any) ──
        if (newData != null && project != null) {
            var newEntry = project.getEntry(newData);
            if (newEntry != null) {
                LabelStore loaded = null;
                try {
                    loaded = ProjectStateManager.loadImageLabels(project, activeBinaryMarker, newEntry.getImageName());
                } catch (IOException ex) {
                    logger.warn("Failed to load labels for {}: {}", newEntry.getImageName(), ex.getMessage());
                }
                this.labelStore = (loaded != null) ? loaded : new LabelStore("CellTune");
                // Also pick up any annotation-based labels on the new image
                collectLabelsFromAnnotations(qupath, this.labelStore);
                // In binary mode, drop any labels loaded from disk that belong to
                // other classifiers (self-healing for previously-contaminated files).
                if (activeBinaryMarker != null && activeBinaryClassNames != null && !activeBinaryClassNames.isEmpty()) {
                    this.labelStore.retainClasses(new java.util.LinkedHashSet<>(activeBinaryClassNames));
                }

                // Load sampled cell IDs for new image
                try {
                    List<String> sampledIds =
                            ProjectStateManager.loadImageSampledCells(project, newEntry.getImageName());
                    this.lastSampledCellIds = sampledIds;
                    this.lastSampledCellImageMap = Map.of();
                    this.lastSampledPredictions = null;
                } catch (IOException ex) {
                    logger.warn("Failed to load sampled cell IDs for {}: {}", newEntry.getImageName(), ex.getMessage());
                }

                // Load per-image predictions if available.
                try {
                    this.predAll = ProjectStateManager.loadImagePredictions(project, newEntry.getImageName());
                } catch (IOException ex) {
                    logger.warn("Failed to load predictions for {}: {}", newEntry.getImageName(), ex.getMessage());
                }
            } else {
                this.labelStore = new LabelStore("CellTune");
                this.lastSampledCellIds = null;
                this.predAll = null;
            }
            logger.info(
                    "Switched to image '{}' — {} labels loaded",
                    newEntry != null ? newEntry.getImageName() : "unknown",
                    labelStore.size());
        } else {
            this.labelStore = new LabelStore("CellTune");
            this.lastSampledCellIds = null;
            this.predAll = null;
        }

        // ── Try to restore feature selection and normalization from project state ──
        if (project != null) {
            try {
                // Restore the persisted marker table (auto channel switching) so it
                // survives QuPath restarts without re-importing the CSV.
                CellTypeTable savedTable = ProjectStateManager.loadMarkerTable(project);
                if (savedTable != null) {
                    this.cellTypeTable = savedTable;
                }
                // Only restore the multi-class session (feature selection, imported training
                // rows, normalizer, classifier) when NOT in binary mode. In binary mode that
                // state was loaded by enterBinaryMode for the active marker and is project/
                // marker-level (not per-image), so reloading it from the multi-class
                // classifier-state.json on an image switch would clobber the binary session
                // and surface multi-class imported rows under the binary classifier.
                var state = (activeBinaryMarker == null) ? ProjectStateManager.loadState(project) : null;
                if (state != null) {
                    if (state.selectedFeatures != null) this.selectedFeatures = new ArrayList<>(state.selectedFeatures);
                    this.importedTrainingFeatureNames = ProjectStateManager.getImportedTrainingFeatureNames(state);
                    this.importedTrainingRows = ProjectStateManager.decodeImportedTrainingRows(state);
                    if (state.featureTransforms != null || state.arcsinhCofactor != null) {
                        FeatureNormalizer norm = new FeatureNormalizer();
                        if (state.featureTransforms != null) norm.fromTransformMap(state.featureTransforms);
                        if (state.arcsinhCofactor != null) norm.setArcsinhCofactor(state.arcsinhCofactor);
                        this.clusteringNormalizer = norm;
                    }
                    // Restore trained classifier from saved model bytes
                    if (classifier == null || !classifier.isTrained()) {
                        byte[] xgbBytes = ProjectStateManager.decodeXGBoostModel(state);
                        byte[] lgbBytes = ProjectStateManager.decodeLightGBMModel(state);
                        byte[] rf1Bytes = ProjectStateManager.decodeRFModel1(state);
                        byte[] rf2Bytes = ProjectStateManager.decodeRFModel2(state);
                        if (xgbBytes != null || lgbBytes != null || rf1Bytes != null || rf2Bytes != null) {
                            var classifierState = new ClassifierState(
                                    state.name != null ? state.name : "CellTune",
                                    state.featureNames,
                                    state.classNames,
                                    xgbBytes,
                                    lgbBytes,
                                    rf1Bytes,
                                    rf2Bytes,
                                    ProjectStateManager.getModel1Type(state),
                                    ProjectStateManager.getModel2Type(state));
                            classifier = new DualModelClassifier();
                            classifier.loadFromState(classifierState);
                            classifier.setTrainingMetrics(
                                    state.model1TrainMetrics, state.model1ValMetrics,
                                    state.model2TrainMetrics, state.model2ValMetrics);
                            logger.info("[CellTune] Restored trained classifier from saved state.");
                        }
                    }
                }
            } catch (Exception ex) {
                logger.warn("Failed to restore feature selection/normalization: {}", ex.getMessage());
            }
        }
        // ── Sync labels with current QuPath class list ──
        syncLabelsToCurrentClasses(qupath);
        // ── Push reset state into panel ──
        syncPanelState();
    }

    /** Push current extension state into the docked panel. */
    @Override
    public void syncPanelState() {
        if (classificationPanel == null) return;
        classificationPanel.setLabelStore(labelStore);
        classificationPanel.setClassifier(classifier);
        classificationPanel.setSelectedFeatures(selectedFeatures);
        classificationPanel.setCellTypeTable(cellTypeTable);
        classificationPanel.setPredAll(predAll);
        classificationPanel.setLastAgreementRates(lastAgreementRates);
        classificationPanel.setLastSampledCellIds(lastSampledCellIds);
        classificationPanel.setImportedTrainingData(importedTrainingRows, importedTrainingFeatureNames);
        classificationPanel.setActiveBinaryMarker(activeBinaryMarker);
        classificationPanel.setActiveBinaryClassNames(activeBinaryClassNames);
    }

    // ── BinaryClassifierManager.Host ─────────────────────────────────────────────
    // Plain accessors over the shared session fields, so binary mode can read/write them
    // while they stay where the rest of the extension uses them.

    @Override
    public LabelStore getLabelStore() {
        return labelStore;
    }

    @Override
    public void setLabelStore(LabelStore labelStore) {
        this.labelStore = labelStore;
    }

    @Override
    public DualModelClassifier getClassifier() {
        return classifier;
    }

    @Override
    public void setClassifier(DualModelClassifier classifier) {
        this.classifier = classifier;
    }

    @Override
    public List<GroundTruthIO.TrainingRow> getImportedTrainingRows() {
        return importedTrainingRows;
    }

    @Override
    public void setImportedTrainingRows(List<GroundTruthIO.TrainingRow> rows) {
        this.importedTrainingRows = rows;
    }

    @Override
    public List<String> getImportedTrainingFeatureNames() {
        return importedTrainingFeatureNames;
    }

    @Override
    public void setImportedTrainingFeatureNames(List<String> featureNames) {
        this.importedTrainingFeatureNames = featureNames;
    }

    @Override
    public String getActiveBinaryMarker() {
        return activeBinaryMarker;
    }

    @Override
    public void setActiveBinaryMarker(String marker) {
        this.activeBinaryMarker = marker;
    }

    @Override
    public List<String> getActiveBinaryClassNames() {
        return activeBinaryClassNames;
    }

    @Override
    public void setActiveBinaryClassNames(List<String> classNames) {
        this.activeBinaryClassNames = classNames;
    }

    @Override
    public void selectAndExpandDockPanel(QuPathGUI qupath) {
        javafx.application.Platform.runLater(() -> {
            if (dockTab != null) {
                qupath.getAnalysisTabPane().getSelectionModel().select(dockTab);
            }
            if (dockPane != null) {
                dockPane.setExpanded(true);
            }
        });
    }

    /** Persist Pred_ALL for the current image so manual mode can show confidence after reload. */
    private void persistCurrentImagePredictions(QuPathGUI qupath) {
        if (qupath == null
                || qupath.getProject() == null
                || qupath.getImageData() == null
                || predAll == null
                || predAll.size() == 0) {
            return;
        }

        var project = qupath.getProject();
        var entry = project.getEntry(qupath.getImageData());
        if (entry == null) return;

        try {
            ProjectStateManager.saveImagePredictions(project, entry.getImageName(), predAll);
        } catch (IOException ex) {
            logger.warn("Failed to save predictions for {}: {}", entry.getImageName(), ex.getMessage());
        }
    }

    /** Load Pred_ALL for the current image if it was previously saved. */
    private boolean loadCurrentImagePredictions(QuPathGUI qupath) {
        if (qupath == null || qupath.getProject() == null || qupath.getImageData() == null) {
            return false;
        }

        var project = qupath.getProject();
        var entry = project.getEntry(qupath.getImageData());
        if (entry == null) return false;

        try {
            var loaded = ProjectStateManager.loadImagePredictions(project, entry.getImageName());
            if (loaded != null && loaded.size() > 0) {
                this.predAll = loaded;
                syncPanelState();
                return true;
            }
        } catch (IOException ex) {
            logger.warn("Failed to load predictions for {}: {}", entry.getImageName(), ex.getMessage());
        }
        return false;
    }

    /**
     * Synchronise the label store with the current QuPath class list.
     * Removes any labels whose class name no longer exists in the project's class list.
     * This ensures removed/renamed classes don't persist as stale labels.
     */
    private void syncLabelsToCurrentClasses(QuPathGUI qupath) {
        if (labelStore == null || labelStore.size() == 0) return;

        var project = qupath.getProject();
        if (project == null) return;

        Set<String> validClasses = new java.util.LinkedHashSet<>();
        for (var pc : project.getPathClasses()) {
            if (pc != null && pc.getName() != null && !pc.getName().isEmpty()) {
                validClasses.add(pc.getName());
            }
        }

        if (validClasses.isEmpty()) return; // No classes defined — don't purge everything

        int removed = labelStore.retainClasses(validClasses);
        if (removed > 0) {
            logger.info("Removed {} labels for classes no longer in QuPath class list", removed);
        }
    }

    /**
     * Check JVM heap at startup and warn if it looks too low for large datasets.
     * QuPath defaults to about 1/4 of system RAM which may be insufficient for
     * 500K+ cells with 2000+ features.
     */
    private void checkHeapMemory() {
        long maxHeapBytes = Runtime.getRuntime().maxMemory();
        double maxHeapGiB = maxHeapBytes / (1024.0 * 1024.0 * 1024.0);
        logger.info("CellTune: JVM max heap = {} GiB", String.format("%.1f", maxHeapGiB));

        if (maxHeapGiB < 8.0) {
            javafx.application.Platform.runLater(() -> Dialogs.showWarningNotification(
                    EXTENSION_NAME,
                    String.format(
                            "JVM heap is only %.1f GiB. For large datasets "
                                    + "(100K+ cells, 1000+ features) increase memory via "
                                    + "Edit \u2192 Preferences \u2192 'Max memory' or "
                                    + "Help \u2192 Show setup options. "
                                    + "Recommended: at least 16 GiB for large panels.",
                            maxHeapGiB)));
        }
    }

    private void addPreferenceToPane(QuPathGUI qupath) {
        var item = new PropertyItemBuilder<>(enableExtensionProperty, Boolean.class)
                .name(resources.getString("menu.enable"))
                .category(EXTENSION_NAME)
                .description(EXTENSION_DESCRIPTION)
                .build();
        var maxBinItem = new PropertyItemBuilder<>(xgbMaxBinProperty, Integer.class)
                .name("XGBoost histogram bins")
                .category(EXTENSION_NAME)
                .description("Bins per feature for XGBoost's histogram split search. 0 = XGBoost's default (256).\n\n"
                        + "XGBoost dominates training time and this is the main lever on it: 128 bins is roughly "
                        + "twice as fast, 64 roughly 2.5x, measured on a 35-class, 1,886-feature panel.\n\n"
                        + "It CHANGES PREDICTIONS. Coarser bins also regularise, so accuracy does not simply drop — "
                        + "but verify on your own data (export the cell table before and after and diff the "
                        + "predicted class column) before settling on a value.")
                .build();
        var batchCorrItem = new PropertyItemBuilder<>(useBatchCorrectionProperty, Boolean.class)
                .name("Use batch-corrected values")
                .category(EXTENSION_NAME)
                .description("Stream persisted UniFORM batch-correction gains into clustering and classifier "
                        + "feature extraction. Each cell's measurements are multiplied by its image's per-channel "
                        + "scale from <project>/celltune/batch-shifts.json before use — no corrected columns are "
                        + "written. Run Batch Normalisation first to create the scales; off by default.")
                .build();
        qupath.getPreferencePane().getPropertySheet().getItems().addAll(item, maxBinItem, batchCorrItem);
    }

    /**
     * Stop QuPath from popping the "Set image type" dialog the first time each image
     * is opened. CellTune's workflow doesn't depend on the image type being set, and
     * the prompt is disruptive for users with large multi-image projects. We switch
     * the global preference to AUTO_ESTIMATE so QuPath quietly guesses (RGB →
     * brightfield, multichannel → fluorescence). Users can change this back via
     * Edit → Preferences → Set image type if they want.
     */
    private void suppressImageTypePromptOnce() {
        try {
            var prop = PathPrefs.imageTypeSettingProperty();
            var current = prop.getValue();
            if (current == null || "PROMPT".equals(current.name())) {
                // Resolve AUTO_ESTIMATE reflectively so we don't have to import the
                // nested enum type at compile time.
                @SuppressWarnings({"unchecked", "rawtypes"})
                Class<? extends Enum> enumCls =
                        (Class<? extends Enum>) Class.forName("qupath.lib.gui.prefs.PathPrefs$ImageTypeSetting");
                @SuppressWarnings({"unchecked", "rawtypes"})
                Enum auto = Enum.valueOf(enumCls, "AUTO_ESTIMATE");
                @SuppressWarnings("unchecked")
                javafx.beans.property.ObjectProperty<Enum> typed =
                        (javafx.beans.property.ObjectProperty<Enum>) (Object) prop;
                typed.setValue(auto);
                logger.debug("Set QuPath imageTypeSetting to AUTO_ESTIMATE to suppress first-open prompt");
            }
        } catch (Throwable t) {
            logger.debug("Could not adjust imageTypeSetting preference: {}", t.toString());
        }
    }

    private void addMenuItems(QuPathGUI qupath) {
        MenuItemFactory.addMenuItems(qupath, this, enableExtensionProperty);
    }

    void showDistanceMeasurements(QuPathGUI qupath) {
        AnalysisViews.showDistanceMeasurements(qupath);
    }

    void showIntensityHeatmaps(QuPathGUI qupath) {
        AnalysisViews.showIntensityHeatmaps(qupath);
    }

    void showScatterPlot(QuPathGUI qupath) {
        AnalysisViews.showScatterPlot(qupath, predAll, clusteringNormalizer, () -> showClassControl(qupath));
    }

    void showCellularNeighborhoods(QuPathGUI qupath) {
        AnalysisViews.showCellularNeighborhoods(qupath);
    }

    void showBatchNormalization(QuPathGUI qupath) {
        AnalysisViews.showBatchNormalization(qupath);
    }

    // ── Placeholder actions (wired in later phases) ────────────────────────────

    /**
     * Completely reset CellTune's saved state for the current project — a clean
     * slate for trying different ML options on a copied project.
     * <p>
     * Deletes the whole {@code <project>/celltune/} folder (labels, trained
     * models, predictions, feature selection, normalization, marker table, binary
     * classifiers, review state) and resets the in-memory session. A timestamped
     * {@code celltune_backup_*.zip} is written to the project folder first as a
     * safety net. Images and detections are untouched unless the user opts in to
     * also clearing CellTune label points and cell classifications from every
     * image. Guarded by a typed "RESET" confirmation.
     */
    void showResetProjectState(QuPathGUI qupath) {
        var project = qupath.getProject();
        if (project == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, resources.getString("classify.no_project"));
            return;
        }

        CheckBox stripBox =
                new CheckBox("Also clear CellTune label points and all cell classifications from every image");
        stripBox.setWrapText(true);

        TextField confirmField = new TextField();
        confirmField.setPromptText("Type RESET to confirm");

        Label info = new Label("This permanently deletes this project's celltune/ folder:\n"
                + "  • all labels and per-image label files\n"
                + "  • trained classifiers (multi-class + binary) and predictions\n"
                + "  • feature selection, normalisation, marker table, composite rules\n"
                + "  • sampling/review state\n\n"
                + "A timestamped backup (celltune_backup_*.zip) is written to the "
                + "project folder first, so this can be undone by unzipping it.\n\n"
                + "Images and detections are kept. Tick the box below to also strip "
                + "CellTune label points and cell classifications from every image.");
        info.setWrapText(true);

        VBox content = new VBox(10, info, stripBox, new Label("Confirm:"), confirmField);
        content.setPadding(new Insets(10));
        content.setPrefWidth(480);

        Dialog<ButtonType> dialog = new Dialog<>();
        dialog.initOwner(qupath.getStage());
        dialog.setTitle(EXTENSION_NAME);
        dialog.setHeaderText("Reset CellTune state for this project");
        dialog.getDialogPane().setContent(content);
        dialog.getDialogPane().getButtonTypes().addAll(ButtonType.OK, ButtonType.CANCEL);

        javafx.scene.Node okBtn = dialog.getDialogPane().lookupButton(ButtonType.OK);
        okBtn.setDisable(true);
        confirmField
                .textProperty()
                .addListener((o, a, b) -> okBtn.setDisable(!"RESET".equals(b == null ? "" : b.strip())));

        var choice = dialog.showAndWait();
        if (choice.isEmpty() || choice.get() != ButtonType.OK) {
            return;
        }
        final boolean strip = stripBox.isSelected();

        // ── Back up then delete the celltune/ state directory (usually small). ──
        Path backup;
        try {
            backup = ProjectStateManager.backupProjectState(project);
        } catch (IOException ex) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "Backup failed; nothing was deleted.\n" + ex.getMessage());
            return;
        }
        try {
            ProjectStateManager.deleteProjectState(project);
        } catch (IOException ex) {
            Dialogs.showErrorMessage(
                    EXTENSION_NAME,
                    "Failed to delete CellTune state: " + ex.getMessage()
                            + (backup != null ? "\nBackup is at: " + backup : ""));
            return;
        }

        // ── Reset the running session so nothing re-saves the old state. ──
        resetInMemoryState(qupath);

        final String backupMsg = backup != null ? "Backup: " + backup.getFileName() : "(no previous state on disk)";

        if (!strip) {
            Dialogs.showInfoNotification(EXTENSION_NAME, "CellTune project state reset. " + backupMsg);
            return;
        }

        // ── Optional: strip CellTune artifacts from every image's data. ──
        @SuppressWarnings("unchecked")
        var entries = (List<ProjectImageEntry<BufferedImage>>) (List<?>) project.getImageList();
        var currentImageData = qupath.getImageData();
        var currentEntry = currentImageData != null ? project.getEntry(currentImageData) : null;

        // Handle the open image inline so the viewer refreshes immediately.
        if (currentImageData != null && currentEntry != null) {
            try {
                stripCellTuneArtifactsFromHierarchy(currentImageData.getHierarchy());
                currentImageData.getHierarchy().fireHierarchyChangedEvent(this);
                currentEntry.saveImageData(currentImageData);
            } catch (Exception ex) {
                logger.warn("[CellTune] Failed to strip artifacts from current image: {}", ex.getMessage());
            }
        }

        var stage = new javafx.stage.Stage();
        stage.setTitle(EXTENSION_NAME + " — Clearing image artifacts");
        stage.initOwner(qupath.getStage());
        stage.initModality(javafx.stage.Modality.WINDOW_MODAL);
        stage.setResizable(false);
        var bar = new javafx.scene.control.ProgressBar(0);
        bar.setPrefWidth(360);
        var status = new javafx.scene.control.Label("Clearing image artifacts…");
        var box = new javafx.scene.layout.VBox(8, status, bar);
        box.setPadding(new javafx.geometry.Insets(15));
        stage.setScene(new javafx.scene.Scene(box));
        stage.setOnCloseRequest(e -> e.consume());
        stage.show();

        final var currentEntryF = currentEntry;
        Thread worker = new Thread(
                () -> {
                    int total = entries.size();
                    int done = 0;
                    int failed = 0;
                    for (var entry : entries) {
                        if (entry != null && (currentEntryF == null || !entry.equals(currentEntryF))) {
                            try {
                                var data = entry.readImageData();
                                stripCellTuneArtifactsFromHierarchy(data.getHierarchy());
                                entry.saveImageData(data);
                            } catch (Exception ex) {
                                failed++;
                                logger.warn(
                                        "[CellTune] Failed to strip artifacts from '{}': {}",
                                        entry.getImageName(),
                                        ex.getMessage());
                            }
                        }
                        final int c = ++done;
                        Platform.runLater(() -> {
                            bar.setProgress((double) c / total);
                            status.setText(String.format("Clearing image artifacts: %d / %d…", c, total));
                        });
                    }
                    final int failedF = failed;
                    Platform.runLater(() -> {
                        stage.close();
                        Dialogs.showInfoNotification(
                                EXTENSION_NAME,
                                "CellTune project state reset and image artifacts cleared. " + backupMsg
                                        + (failedF > 0 ? " (" + failedF + " image(s) failed — see log.)" : ""));
                    });
                },
                "celltune-reset-strip");
        worker.setDaemon(true);
        worker.start();
    }

    /**
     * Remove CellTune ground-truth label points (classified point annotations)
     * and clear all cell classifications (predictions) from a hierarchy.
     *
     * @return {@code [annotationsRemoved, cellsCleared]}
     */
    private static int[] stripCellTuneArtifactsFromHierarchy(
            qupath.lib.objects.hierarchy.PathObjectHierarchy hierarchy) {
        if (hierarchy == null) {
            return new int[] {0, 0};
        }
        List<PathObject> labelPoints = new ArrayList<>();
        for (PathObject anno : hierarchy.getAnnotationObjects()) {
            if (anno.getROI() != null && anno.getROI().isPoint() && anno.getPathClass() != null) {
                labelPoints.add(anno);
            }
        }
        if (!labelPoints.isEmpty()) {
            hierarchy.removeObjects(labelPoints, false);
        }
        int cellsCleared = 0;
        for (PathObject det : hierarchy.getDetectionObjects()) {
            if (det.getPathClass() != null) {
                det.setPathClass(null);
                cellsCleared++;
            }
        }
        return new int[] {labelPoints.size(), cellsCleared};
    }

    /** Reset all in-memory CellTune session state to defaults and refresh the panel. */
    private void resetInMemoryState(QuPathGUI qupath) {
        this.labelStore = new LabelStore("CellTune");
        this.classifier = null;
        this.predAll = null;
        this.selectedFeatures = null;
        this.clusteringNormalizer = null;
        this.lastAgreementRates = null;
        this.lastSampledCellIds = null;
        this.lastSampledCellImageMap = Map.of();
        this.lastSampledPredictions = null;
        this.importedTrainingRows = null;
        this.importedTrainingFeatureNames = null;
        this.cellTypeTable = null;

        // Binary-classifier session state.
        this.binaryManager.reset();
        this.activeBinaryMarker = null;
        this.activeBinaryClassNames = null;
        this.binaryTargetImages = new ArrayList<>();

        syncPanelState();
        logger.info("[CellTune] In-memory session state reset.");
    }

    /**
     * Whole-image pixel prescreen: read a low-resolution version of every project
     * image straight off the pyramid, compute per-channel pixel statistics, and
     * rank/flag images against the cohort (background-heavy, saturated, weak
     * signal, intensity outlier). Needs no cells — intended as a first-pass QC
     * step at the start of a project. Reads run sequentially on a background
     * thread to keep memory bounded and the UI responsive.
     */
    void showImagePixelPrescreen(QuPathGUI qupath) {
        AnalysisViews.showImagePixelPrescreen(qupath);
    }

    void showProjectPredictionSummary(QuPathGUI qupath) {
        // Ensure latest in-memory predictions for current image are included in the summary.
        persistCurrentImagePredictions(qupath);
        ProjectPredictionSummary.show(qupath, predAll);
    }

    private void showFeatureImportance(QuPathGUI qupath) {
        AnalysisViews.showFeatureImportance(qupath, classifier);
    }

    /**
     * Collect ground-truth labels from classified point annotations overlapping detections.
     */
    private void collectLabelsFromAnnotations(QuPathGUI qupath, LabelStore store) {
        var imageData = qupath.getImageData();
        if (imageData == null) return;
        collectLabelsFromHierarchy(imageData.getHierarchy(), store, activeBinaryClassFilter());
    }

    /** Returns the allowed binary class names when in binary mode, or null otherwise. */
    private java.util.Set<String> activeBinaryClassFilter() {
        if (activeBinaryMarker == null) return null;
        if (activeBinaryClassNames == null || activeBinaryClassNames.isEmpty()) return null;
        return new java.util.LinkedHashSet<>(activeBinaryClassNames);
    }

    /**
     * Save the current image's label store as a per-image JSON file so that
     * labels from review and manual labelling can be pooled during training
     * on other images.
     */
    private void saveCurrentImageLabels(QuPathGUI qupath) {
        if (labelStore == null || labelStore.size() == 0) return;
        var project = qupath.getProject();
        var imageData = qupath.getImageData();
        if (project == null || imageData == null) return;

        var entry = project.getEntry(imageData);
        if (entry == null) return;

        // Filter to only cell IDs that belong to this image's detections
        var filteredStore = filterLabelStoreToImage(labelStore, imageData);
        // In binary mode, also drop labels for classes outside this binary classifier
        // (defence-in-depth — collection paths already filter, but stale memory could
        // otherwise reintroduce foreign labels into the per-image file on save).
        if (activeBinaryMarker != null && activeBinaryClassNames != null && !activeBinaryClassNames.isEmpty()) {
            filteredStore.retainClasses(new java.util.LinkedHashSet<>(activeBinaryClassNames));
        }
        if (filteredStore.size() == 0) return;

        try {
            ProjectStateManager.saveImageLabels(project, activeBinaryMarker, entry.getImageName(), filteredStore);
        } catch (IOException ex) {
            logger.warn("Failed to save per-image labels for {}: {}", entry.getImageName(), ex.getMessage());
        }
    }

    /**
     * Create a copy of the label store containing only cell IDs that exist
     * as detections in the given image. This prevents stale IDs from other
     * images leaking into per-image label files.
     */
    private static LabelStore filterLabelStoreToImage(
            LabelStore store, qupath.lib.images.ImageData<BufferedImage> imageData) {
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

    private boolean sampleForReviewBatch(
            QuPathGUI qupath, ReviewSampling.SamplingContext samplingContext, int sampleSize) {
        if (samplingContext == null
                || samplingContext.predictions() == null
                || samplingContext.predictions().size() == 0
                || sampleSize <= 0) {
            return false;
        }

        Set<String> reviewedCellIds = ReviewSampling.buildReviewedCellIdsForSampling(
                qupath, activeBinaryMarker, labelStore, lastSampledCellIds);
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
        for (String id : lastSampledCellIds) {
            String imageName = samplingContext.cellToImage().get(id);
            if (imageName != null) {
                lastSampledCellImageMap.put(id, imageName);
            }
        }
        lastSampledPredictions = samplingContext.predictions();

        syncPanelState();
        persistCurrentImageSampledIds(qupath);
        return !lastSampledCellIds.isEmpty();
    }

    private void persistCurrentImageSampledIds(QuPathGUI qupath) {
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
        } catch (Exception ex) {
            logger.warn("Failed to save sampled cells: {}", ex.getMessage());
        }
    }

    private void persistReviewedLabelsByImage(QuPathGUI qupath, LabelStore reviewedLabels) {
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

        java.util.Set<String> binaryFilter = activeBinaryClassFilter();

        for (var entry : labelsByImage.entrySet()) {
            String imageName = entry.getKey();
            LabelStore delta = entry.getValue();

            try {
                // In binary mode, strip foreign-class labels from the delta before
                // it touches disk, and from the merged on-disk store after merge.
                // The post-merge filter is self-healing: any historical contamination
                // (foreign labels written by older buggy code) gets cleaned up on
                // the next save.
                if (binaryFilter != null) {
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
            } catch (Exception ex) {
                logger.warn("Failed to save reviewed labels for {}: {}", imageName, ex.getMessage());
            }
        }
    }

    /**
     * Collect ground-truth labels from classified point annotations in a hierarchy.
     * Works with any image's hierarchy — used for both the current image and other
     * project images when pooling training data. Delegates to the shared
     * {@link AnnotationLabelCollector} so the merge-history-preservation behaviour
     * is identical here and in {@code ClassificationPanel}.
     */
    private static void collectLabelsFromHierarchy(
            qupath.lib.objects.hierarchy.PathObjectHierarchy hierarchy,
            LabelStore store,
            java.util.Set<String> allowedClasses) {
        AnnotationLabelCollector.collect(hierarchy, store, allowedClasses);
    }

    private void showManualLabelMode(QuPathGUI qupath) {
        if (qupath.getImageData() == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No image is open.");
            return;
        }
        if (qupath.getImageData().getHierarchy().getDetectionObjects().isEmpty()) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No detections found. Run cell detection first.");
            return;
        }

        if (labelStore == null) {
            labelStore = new LabelStore("CellTune");
        }

        // First try to restore saved predictions for this image.
        if (predAll == null || predAll.size() == 0) {
            loadCurrentImagePredictions(qupath);
        }

        // If a trained classifier exists, ensure predictions are available for
        // the current image so manual mode can show per-cell confidence scores.
        if (predAll == null || predAll.size() == 0) {
            autoClassifyCurrentImage(qupath);
        }

        // Gather extra class names from CellTypeTable if loaded
        java.util.Set<String> extraClasses = null;
        if (cellTypeTable != null) {
            extraClasses = cellTypeTable.getCellTypes();
        }

        var toolbar = new ManualLabelToolbar(
                qupath, labelStore, extraClasses, qupath.getStage(), predAll, activeBinaryClassNames);

        // When the manual label window closes, sync state
        toolbar.getStage().setOnHidden(e -> {
            syncPanelState();
            logger.info("Manual label mode ended — {} total labels", labelStore.size());

            // Persist per-image labels so they can be pooled from other images
            saveCurrentImageLabels(qupath);

            Dialogs.showInfoNotification(
                    EXTENSION_NAME, "Manual labelling complete: " + labelStore.size() + " total labels.");
        });
    }

    /**
     * If a trained classifier exists but predictions have not been run on the
     * current image (e.g. after an image switch), automatically classify all
     * cells so that review mode, confusion matrices and sampling work
     * immediately without retraining.
     *
     * @return true if predictions are now available (existing or freshly generated)
     */
    private boolean autoClassifyCurrentImage(QuPathGUI qupath) {
        if (predAll != null && predAll.size() > 0) {
            return true; // already have predictions
        }
        if (classifier == null || !classifier.isTrained()) {
            return false; // no trained classifier
        }
        var imageData = qupath.getImageData();
        if (imageData == null) return false;

        var detections = imageData.getHierarchy().getDetectionObjects();
        if (detections.isEmpty()) return false;

        // Use the feature columns from the trained classifier
        List<String> featureNames = classifier.getFeatureNames();
        if (featureNames == null || featureNames.isEmpty()) return false;

        try {
            // Raw inference — the classifier is trained on raw values (normalisation is
            // a clustering-only concern; tree models are invariant to it anyway).
            var extractor = new CellFeatureExtractor(featureNames);
            classifier.predictOnly(
                    detections, extractor, true, msg -> logger.info("[CellTune] Auto-classify: {}", msg));
            predAll = classifier.getPredALL();
            persistCurrentImagePredictions(qupath);
            imageData.getHierarchy().fireHierarchyChangedEvent(this);
            syncPanelState();
            logger.info("[CellTune] Auto-classified {} cells on current image.", predAll != null ? predAll.size() : 0);
            return predAll != null && predAll.size() > 0;
        } catch (Exception e) {
            logger.error("[CellTune] Auto-classify failed: {}", e.getMessage(), e);
            return false;
        }
    }

    private void showReviewMode(QuPathGUI qupath) {
        if (qupath.getProject() == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, resources.getString("classify.no_project"));
            return;
        }

        // Try to load persisted sampled cell IDs if not in memory
        if ((lastSampledCellIds == null || lastSampledCellIds.isEmpty()) && qupath.getProject() != null) {
            var imgData = qupath.getImageData();
            if (imgData != null) {
                var entry = qupath.getProject().getEntry(imgData);
                if (entry != null) {
                    try {
                        var loaded =
                                ProjectStateManager.loadImageSampledCells(qupath.getProject(), entry.getImageName());
                        if (loaded != null && !loaded.isEmpty()) {
                            this.lastSampledCellIds = loaded;
                            this.lastSampledCellImageMap = Map.of();
                            this.lastSampledPredictions = null;
                            syncPanelState();
                        }
                    } catch (IOException ex) {
                        logger.warn("Failed to load sampled cells: {}", ex.getMessage());
                    }
                }
            }
        }

        // Auto-classify if we have a trained classifier but no predictions yet
        autoClassifyCurrentImage(qupath);

        if (classifier == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No trained classifier available. Train first.");
            return;
        }

        // Ask the user whether to pool predictions across the project or to
        // restrict sampling to the currently open image.
        boolean currentImageOnly = false;
        {
            var imgData = qupath.getImageData();
            String currentName = null;
            if (imgData != null && qupath.getProject() != null) {
                var entry = qupath.getProject().getEntry(imgData);
                if (entry != null) currentName = entry.getImageName();
            }
            if (currentName != null) {
                var allBtn = new javafx.scene.control.ButtonType(
                        "All project images", javafx.scene.control.ButtonBar.ButtonData.OK_DONE);
                var currentBtn = new javafx.scene.control.ButtonType(
                        "Current image only", javafx.scene.control.ButtonBar.ButtonData.OTHER);
                var cancelBtn = new javafx.scene.control.ButtonType(
                        "Cancel", javafx.scene.control.ButtonBar.ButtonData.CANCEL_CLOSE);
                var alert = new javafx.scene.control.Alert(
                        javafx.scene.control.Alert.AlertType.CONFIRMATION,
                        "Sample disagreement cells from which images?",
                        allBtn,
                        currentBtn,
                        cancelBtn);
                alert.setTitle("Enter Review Mode");
                alert.setHeaderText("Current image: " + currentName);
                alert.initOwner(qupath.getStage());
                var choice = alert.showAndWait();
                if (choice.isEmpty() || choice.get() == cancelBtn) return;
                currentImageOnly = (choice.get() == currentBtn);
            }
        }

        ReviewSampling.SamplingContext samplingContext =
                ReviewSampling.buildSamplingContext(qupath, predAll, currentImageOnly);
        long disagreeCount = samplingContext.predictions().getDisagreementCount();
        if (disagreeCount == 0) {
            Dialogs.showInfoNotification(
                    EXTENSION_NAME, "No disagreement cells available across saved project predictions.");
            return;
        }

        // Compute agreement rates if not yet available (for current image)
        if (predAll != null && predAll.size() > 0 && lastAgreementRates == null) {
            var confView = new ConfusionMatrixView(qupath.getStage(), predAll, classifier.getClassNames());
            lastAgreementRates = confView.getAgreementRates();
        }

        String countStr = Dialogs.showInputDialog(
                resources.getString("sample.dialog.title"),
                "How many disagreement cells to review?" + " (" + disagreeCount + " available)",
                "200");
        if (countStr == null) return;

        int sampleSize;
        try {
            sampleSize = Integer.parseInt(countStr.strip());
        } catch (NumberFormatException e) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "Invalid number.");
            return;
        }
        if (sampleSize <= 0) return;

        if (!sampleForReviewBatch(qupath, samplingContext, sampleSize)) {
            Dialogs.showInfoNotification(
                    EXTENSION_NAME, "No eligible disagreement cells remained after excluding reviewed cells.");
            return;
        }

        if (lastSampledCellIds == null || lastSampledCellIds.isEmpty()) {
            Dialogs.showErrorMessage(EXTENSION_NAME, resources.getString("review.no_sample"));
            return;
        }

        PopulationSet reviewPredictions = (lastSampledPredictions != null && lastSampledPredictions.size() > 0)
                ? lastSampledPredictions
                : predAll;
        if (reviewPredictions == null || reviewPredictions.size() == 0) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No predictions available. Train and sample first.");
            return;
        }

        // Build controller and UI components
        // Pre-extract training tiles around each sampled cell so review avoids
        // costly switches between large project images. Cell IDs round-trip via
        // lastSampledCellImageMap (see persistReviewedLabelsByImage).
        final PopulationSet finalReviewPredictions = reviewPredictions;
        launchTrainingTileReview(qupath, finalReviewPredictions);
    }

    private void launchTrainingTileReview(QuPathGUI qupath, PopulationSet reviewPredictions) {
        var progressStage = new javafx.stage.Stage();
        progressStage.setTitle(EXTENSION_NAME + " \u2014 Preparing review tiles");
        progressStage.initOwner(qupath.getStage());
        progressStage.initModality(javafx.stage.Modality.WINDOW_MODAL);
        progressStage.setResizable(false);
        var bar = new javafx.scene.control.ProgressBar(0);
        bar.setPrefWidth(360);
        var status = new javafx.scene.control.Label("Extracting cell tiles\u2026");
        status.setMaxWidth(360);
        status.setWrapText(true);
        var box = new javafx.scene.layout.VBox(8, status, bar);
        box.setPadding(new javafx.geometry.Insets(15));
        progressStage.setScene(new javafx.scene.Scene(box));
        progressStage.setOnCloseRequest(ev -> ev.consume());
        progressStage.show();

        final int total = lastSampledCellIds == null ? 0 : lastSampledCellIds.size();
        final List<String> cellIds = lastSampledCellIds;
        final Map<String, String> imageMap = lastSampledCellImageMap;
        Thread t = new Thread(
                () -> {
                    TrainingTileExtractor extractor;
                    try {
                        extractor = TrainingTileExtractor.extract(
                                qupath,
                                cellIds,
                                imageMap,
                                done -> javafx.application.Platform.runLater(() -> {
                                    if (total > 0) {
                                        bar.setProgress((double) done / total);
                                        status.setText(
                                                String.format("Extracting tiles: %d / %d cells\u2026", done, total));
                                    }
                                }));
                    } catch (Exception e) {
                        final Exception err = e;
                        javafx.application.Platform.runLater(() -> {
                            progressStage.close();
                            Dialogs.showErrorMessage(
                                    EXTENSION_NAME, "Failed to prepare review tiles: " + err.getMessage());
                        });
                        return;
                    }
                    final TrainingTileExtractor finalExtractor = extractor;
                    javafx.application.Platform.runLater(() -> {
                        progressStage.close();
                        showTileReviewStage(qupath, reviewPredictions, finalExtractor);
                    });
                },
                "celltune-tile-extract");
        t.setDaemon(true);
        t.start();
    }

    private void showTileReviewStage(
            QuPathGUI qupath, PopulationSet reviewPredictions, TrainingTileExtractor extractor) {
        var reviewController = new ReviewController(
                qupath, lastSampledCellIds, reviewPredictions, lastSampledCellImageMap, Map.of(), extractor.getPreps());
        if (reviewController.size() == 0) {
            extractor.close();
            Dialogs.showErrorMessage(EXTENSION_NAME, "Could not resolve sampled cells in project images.");
            return;
        }

        var channelSelector = new ChannelSelector(qupath, cellTypeTable);
        var toolbar = new ReviewToolbar(reviewController, cellTypeTable, channelSelector);
        toolbar.setBinaryMarker(activeBinaryMarker, activeBinaryClassNames);

        // Build the review stage
        var vbox = new javafx.scene.layout.VBox(6);
        vbox.setPadding(new javafx.geometry.Insets(6));
        vbox.getChildren().addAll(toolbar, channelSelector.getCheckBox(), channelSelector.getDisplayRangeCheckBox());

        double reviewScreenH =
                javafx.stage.Screen.getPrimary().getVisualBounds().getHeight();
        var reviewScroll = new javafx.scene.control.ScrollPane(vbox);
        reviewScroll.setFitToWidth(true);
        reviewScroll.setHbarPolicy(javafx.scene.control.ScrollPane.ScrollBarPolicy.AS_NEEDED);
        reviewScroll.setVbarPolicy(javafx.scene.control.ScrollPane.ScrollBarPolicy.AS_NEEDED);
        reviewScroll.setMaxHeight(reviewScreenH * 0.9);

        var stage = new javafx.stage.Stage();
        stage.setTitle(resources.getString("review.stage.title"));
        stage.initOwner(qupath.getStage());
        stage.setScene(new javafx.scene.Scene(reviewScroll));
        stage.setAlwaysOnTop(true);
        stage.setResizable(true);
        stage.setMinWidth(750);

        // When the review window is closed, merge labels back
        stage.setOnHidden(e -> {
            reviewController.removeHighlight();
            var outputLabels = reviewController.getOutputLabels();
            if (outputLabels.size() > 0) {
                if (labelStore == null) {
                    labelStore = new LabelStore("CellTune");
                }
                labelStore.mergeFrom(outputLabels);
                syncPanelState();
                logger.info("Review complete - merged {} labels into main label store", outputLabels.size());

                persistReviewedLabelsByImage(qupath, outputLabels);

                Dialogs.showInfoNotification(
                        EXTENSION_NAME, String.format("Review complete: %d labels merged.", outputLabels.size()));
            }
            extractor.close();
        });

        stage.show();
    }

    private void showConfusions(QuPathGUI qupath) {
        // Auto-classify if we have a trained classifier but no predictions yet
        autoClassifyCurrentImage(qupath);

        if (predAll == null || predAll.size() == 0) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No predictions available. Train a classifier first.");
            return;
        }
        if (classifier == null || classifier.getClassNames() == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "Classifier not available. Train first.");
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
        view.show();
    }

    void exportCellTable(QuPathGUI qupath) {
        ImportExport.exportCellTable(qupath);
    }

    void importMarkerTable(QuPathGUI qupath) {
        CellTypeTable imported = ImportExport.importMarkerTable(qupath);
        if (imported != null) {
            cellTypeTable = imported;
            syncPanelState();
        }
    }

    // ── Ground truth export/import ─────────────────────────────

    void exportGroundTruth(QuPathGUI qupath) {
        ImportExport.exportGroundTruth(
                qupath, labelStore, activeBinaryMarker, importedTrainingRows, importedTrainingFeatureNames);
    }

    void importGroundTruth(QuPathGUI qupath) {
        var result = ImportExport.importGroundTruth(
                qupath, labelStore, importedTrainingRows, importedTrainingFeatureNames, activeBinaryMarker);
        if (result != null) {
            labelStore = result.labelStore();
            importedTrainingRows = result.importedRows();
            importedTrainingFeatureNames = result.importedFeatureNames();
            syncPanelState();
        }
    }

    /** Clear imported training rows for the active context (binary marker, or multi-class). */
    private void clearImportedTrainingData(QuPathGUI qupath) {
        if (ImportExport.clearImportedTrainingData(qupath, activeBinaryMarker)) {
            importedTrainingRows = null;
            importedTrainingFeatureNames = null;
            syncPanelState();
        }
    }

    private boolean ensureActiveBinaryMarker() {
        if (activeBinaryMarker == null || activeBinaryMarker.isBlank()) {
            Dialogs.showErrorMessage(EXTENSION_NAME, resources.getString("binary.marker.required"));
            return false;
        }
        return true;
    }

    void exportActiveBinaryGroundTruth(QuPathGUI qupath) {
        if (!ensureActiveBinaryMarker()) return;
        exportGroundTruth(qupath);
    }

    void importActiveBinaryGroundTruth(QuPathGUI qupath) {
        if (!ensureActiveBinaryMarker()) return;
        importGroundTruth(qupath);
    }

    // ── Feature selection ──────────────────────────────────────────────────────

    void showFeatureSelection(QuPathGUI qupath) {
        var imageData = qupath.getImageData();
        if (imageData == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No image is open.");
            return;
        }

        Collection<PathObject> detections = imageData.getHierarchy().getDetectionObjects();
        if (detections.isEmpty()) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No detections found. Run cell detection first.");
            return;
        }

        List<String> allFeatures = CellFeatureExtractor.discoverFeatureNames(detections);
        if (allFeatures.isEmpty()) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No cell measurements found.");
            return;
        }

        var pane = new FeatureSelectionPane(qupath.getStage(), allFeatures, selectedFeatures);
        List<String> chosen = pane.showAndWait();
        if (chosen != null) {
            if (chosen.isEmpty()) {
                Dialogs.showWarningNotification(
                        EXTENSION_NAME, "No features selected — using all features for training.");
                selectedFeatures = null;
            } else if (chosen.size() == allFeatures.size()) {
                selectedFeatures = null;
                Dialogs.showInfoNotification(EXTENSION_NAME, "All " + allFeatures.size() + " features selected.");
            } else {
                selectedFeatures = chosen;
                Dialogs.showInfoNotification(
                        EXTENSION_NAME,
                        chosen.size() + " of " + allFeatures.size() + " features selected for training.");
            }
            persistSelectedFeatures(qupath);
            syncPanelState();
        }
    }

    /**
     * Save the current {@link #selectedFeatures} list to the active project so
     * it survives QuPath restarts. No-op when no project is open.
     */
    private void persistSelectedFeatures(QuPathGUI qupath) {
        var project = qupath.getProject();
        if (project == null) {
            return;
        }
        try {
            ProjectStateManager.saveSelectedFeatures(project, selectedFeatures);
        } catch (Exception ex) {
            logger.warn("Failed to persist selected features: {}", ex.getMessage());
        }
    }

    // ── Normalization ──────────────────────────────────────────────────────────

    void showClusteringNormalisation(QuPathGUI qupath) {
        var imageData = qupath.getImageData();
        if (imageData == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No image is open.");
            return;
        }

        Collection<PathObject> detections = imageData.getHierarchy().getDetectionObjects();
        if (detections.isEmpty()) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No detections found. Run cell detection first.");
            return;
        }

        // Use selected features if available, otherwise discover all
        List<String> featureNames;
        if (selectedFeatures != null && !selectedFeatures.isEmpty()) {
            featureNames = selectedFeatures;
        } else {
            featureNames = CellFeatureExtractor.discoverFeatureNames(detections);
        }

        if (featureNames.isEmpty()) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "No cell measurements found.");
            return;
        }

        var pane = new ClusteringNormalizationPane(qupath, qupath.getStage(), featureNames, clusteringNormalizer);
        FeatureNormalizer result = pane.showAndWait();
        if (result != null) {
            if (result.hasTransforms()) {
                clusteringNormalizer = result;
                long count = result.getAllTransforms().size();
                Dialogs.showInfoNotification(
                        EXTENSION_NAME,
                        count + " feature(s) will be normalised for clustering (cofactor=" + result.getArcsinhCofactor()
                                + "). The classifier uses raw values.");
            } else {
                clusteringNormalizer = null;
                Dialogs.showInfoNotification(EXTENSION_NAME, "Feature normalization cleared.");
            }
        }
    }

    // ── Binary classifier management ───────────────────────────────────────────

    /**
     * Show the Binary Classifiers management dialog.
     * Loads the registry from disk, wires callbacks, and shows the panel in a modal stage.
     */
    void showBinaryClassifiers(QuPathGUI qupath) {
        binaryManager.showBinaryClassifiers(qupath);
    }

    private void applyBinaryClassifierToImages(QuPathGUI qupath) {
        var project = qupath.getProject();
        if (project == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "Open a QuPath project first.");
            return;
        }

        @SuppressWarnings("unchecked")
        var entries = (java.util.List<ProjectImageEntry<java.awt.image.BufferedImage>>)
                (java.util.List<?>) project.getImageList();
        java.util.List<String> allImageNames =
                entries.stream().map(ProjectImageEntry::getImageName).collect(java.util.stream.Collectors.toList());
        if (allImageNames.isEmpty()) return;

        String currentImageName = null;
        var liveImageData = qupath.getImageData();
        if (liveImageData != null) {
            var curEntry = project.getEntry(liveImageData);
            if (curEntry != null) currentImageName = curEntry.getImageName();
        }

        // Pre-populate picker with any previously stored selection
        var imageSelector = new ImageSelectionPane(qupath.getStage(), allImageNames, currentImageName);
        java.util.List<String> selected = imageSelector.showAndWait();
        if (selected == null) return; // cancelled

        // Store selection — applied on next training run, no classification now
        binaryTargetImages = new java.util.ArrayList<>(selected);
        classificationPanel.setApplyToImagesCount(binaryTargetImages.size());
    }

    void showCompositeClassification(QuPathGUI qupath) {
        new CompositeClassificationDialog(qupath).showAndWait();
    }

    /** Singleton dialog instance — reuse across menu invocations. */
    private ClassControlDialog classControlDialog;

    void showClassControl(QuPathGUI qupath) {
        if (classControlDialog == null) {
            classControlDialog = new ClassControlDialog(
                    qupath,
                    () -> labelStore,
                    ls -> this.labelStore = ls,
                    // Pre-op: flush active image's in-memory labels to disk so a class
                    // operation (delete/merge/undo) doesn't get clobbered later by a
                    // save that reflects stale memory. Only runs in multi-class mode.
                    () -> {
                        if (activeBinaryMarker != null) return;
                        var project = qupath.getProject();
                        var imageData = qupath.getImageData();
                        if (project == null || imageData == null || labelStore == null || labelStore.size() == 0)
                            return;
                        var entry = project.getEntry(imageData);
                        if (entry == null) return;
                        try {
                            var filtered = filterLabelStoreToImage(labelStore, imageData);
                            if (filtered.size() > 0) {
                                ProjectStateManager.saveImageLabels(project, entry.getImageName(), filtered);
                            }
                        } catch (IOException ex) {
                            logger.warn("Pre-op label flush failed for {}: {}", entry.getImageName(), ex.getMessage());
                        }
                    },
                    // Post-op: reload active image's labels from disk so memory matches
                    // the freshly-rewritten on-disk state. Disk is the source of truth.
                    () -> {
                        if (activeBinaryMarker != null) return;
                        var project = qupath.getProject();
                        var imageData = qupath.getImageData();
                        if (project == null || imageData == null) return;
                        var entry = project.getEntry(imageData);
                        if (entry == null) return;
                        try {
                            LabelStore reloaded = ProjectStateManager.loadImageLabels(project, entry.getImageName());
                            this.labelStore = (reloaded != null) ? reloaded : new LabelStore("CellTune");
                            collectLabelsFromAnnotations(qupath, this.labelStore);
                        } catch (IOException ex) {
                            logger.warn(
                                    "Post-op label reload failed for {}: {}", entry.getImageName(), ex.getMessage());
                        }
                    });
        }
        classControlDialog.show();
    }

    /**
     * Enter binary mode for the given marker.
     * Saves the current multi-class state, loads the binary marker's labels and classifier,
     * swaps the panel state, and expands the docked panel.
     *
     * @param qupath     the QuPath GUI
     * @param markerName the sanitized marker name to activate
     */
    private void enterBinaryMode(QuPathGUI qupath, String markerName) {
        binaryManager.enterBinaryMode(qupath, markerName);
    }

    /**
     * Exit binary mode and restore the multi-class classifier state.
     * Saves the binary marker's current labels before restoring.
     *
     * @param qupath the QuPath GUI
     */
    private void exitBinaryMode(QuPathGUI qupath) {
        binaryManager.exitBinaryMode(qupath);
    }
}
