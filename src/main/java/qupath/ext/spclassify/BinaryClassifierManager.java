package qupath.ext.spclassify;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.classifier.ClassifierState;
import qupath.ext.spclassify.classifier.DualModelClassifier;
import qupath.ext.spclassify.io.BinaryClassifierRegistry;
import qupath.ext.spclassify.io.GroundTruthIO;
import qupath.ext.spclassify.io.ProjectStateManager;
import qupath.ext.spclassify.model.LabelStore;
import qupath.ext.spclassify.ui.BinaryClassifierPanel;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.projects.Project;

/**
 * Owns the <b>binary-classifier mode</b> session state and transitions, lifted out of
 * {@code SpClassifyExtension}.
 * <p>
 * Binary mode temporarily swaps the multi-class session (labels, classifier, imported rows)
 * for a single marker's two-class state. The pre-binary snapshot and the per-project marker
 * registry are private to this manager; the <em>shared</em> session fields it has to read and
 * write (the active label store, classifier, imported training data, and the active
 * marker/class names) are reached through a {@link Host} the extension implements, so those
 * fields can stay where the rest of the extension reads them.
 * <p>
 * Lifted verbatim (behaviour preserved 1:1); the extension's menu/panel handlers delegate here.
 */
final class BinaryClassifierManager {

    private static final Logger logger = LoggerFactory.getLogger(BinaryClassifierManager.class);
    private static final String EXTENSION_NAME = java.util.ResourceBundle.getBundle("qupath.ext.spclassify.ui.strings")
            .getString("name");

    /**
     * The shared session state binary mode mutates, owned by the extension. All getters/setters
     * are plain field accessors; {@link #syncPanelState()} pushes the current state to the docked
     * panel and {@link #selectAndExpandDockPanel(QuPathGUI)} focuses it after entering binary mode.
     */
    interface Host {
        LabelStore getLabelStore();

        void setLabelStore(LabelStore labelStore);

        DualModelClassifier getClassifier();

        void setClassifier(DualModelClassifier classifier);

        List<GroundTruthIO.TrainingRow> getImportedTrainingRows();

        void setImportedTrainingRows(List<GroundTruthIO.TrainingRow> rows);

        List<String> getImportedTrainingFeatureNames();

        void setImportedTrainingFeatureNames(List<String> featureNames);

        String getActiveBinaryMarker();

        void setActiveBinaryMarker(String marker);

        List<String> getActiveBinaryClassNames();

        void setActiveBinaryClassNames(List<String> classNames);

        void syncPanelState();

        void selectAndExpandDockPanel(QuPathGUI qupath);
    }

    private final Host host;

    // Marker registry for this project (loaded on demand by showBinaryClassifiers).
    private Map<String, String> binaryRegistry = new LinkedHashMap<>();

    // Pre-binary multi-class snapshot, restored on exit.
    private LabelStore preBinaryLabelStore = null;
    private DualModelClassifier preBinaryClassifier = null;
    private List<GroundTruthIO.TrainingRow> preBinaryImportedTrainingRows = null;
    private List<String> preBinaryImportedTrainingFeatureNames = null;

    BinaryClassifierManager(Host host) {
        this.host = host;
    }

    /** Clear the registry + pre-binary snapshot (called by the extension's in-memory reset). */
    void reset() {
        this.binaryRegistry = new LinkedHashMap<>();
        this.preBinaryLabelStore = null;
        this.preBinaryClassifier = null;
        this.preBinaryImportedTrainingRows = null;
        this.preBinaryImportedTrainingFeatureNames = null;
    }

    void showBinaryClassifiers(QuPathGUI qupath) {
        var project = qupath.getProject();
        if (project == null) {
            Dialogs.showErrorMessage(EXTENSION_NAME, "Open a QuPath project first.");
            return;
        }
        try {
            binaryRegistry = BinaryClassifierRegistry.load(project);
        } catch (IOException ex) {
            logger.warn("Failed to load binary classifier registry: {}", ex.getMessage());
            binaryRegistry = new LinkedHashMap<>();
        }

        var panel = new BinaryClassifierPanel();
        panel.setMarkerNames(new ArrayList<>(binaryRegistry.keySet()));
        panel.setActiveBinaryMarker(host.getActiveBinaryMarker());

        panel.setOnRegisterMarker(markerName -> {
            try {
                BinaryClassifierRegistry.register(project, binaryRegistry, markerName);
            } catch (IOException ex) {
                logger.warn("Failed to register binary classifier '{}': {}", markerName, ex.getMessage());
            }
        });

        panel.setOnDeleteMarker(markerName -> {
            try {
                BinaryClassifierRegistry.remove(project, binaryRegistry, markerName);
                if (markerName.equals(host.getActiveBinaryMarker())) exitBinaryMode(qupath);
            } catch (IOException ex) {
                logger.warn("Failed to remove binary classifier '{}': {}", markerName, ex.getMessage());
            }
        });

        panel.setOnOpenMarker(markerName -> enterBinaryMode(qupath, markerName));
        panel.setOnExitBinaryMode(() -> exitBinaryMode(qupath));

        var stage = new javafx.stage.Stage();
        stage.initModality(javafx.stage.Modality.APPLICATION_MODAL);
        stage.setTitle("Binary Classifiers — " + EXTENSION_NAME);
        stage.setScene(new javafx.scene.Scene(panel, 420, 340));
        stage.showAndWait();
    }

    /**
     * Enter binary mode for the given marker.
     * Saves the current multi-class state, loads the binary marker's labels and classifier,
     * swaps the panel state, and expands the docked panel.
     *
     * @param qupath     the QuPath GUI
     * @param markerName the sanitized marker name to activate
     */
    void enterBinaryMode(QuPathGUI qupath, String markerName) {
        var project = qupath.getProject();
        if (project == null) return;

        // Preserve current multi-class state — but only when we're actually transitioning
        // INTO binary mode. If we're already in binary mode (user switched markers without
        // exiting first), the existing pre-binary snapshot must be left alone, otherwise we
        // would clobber it with the previous marker's binary state and lose the multi-class
        // labels/classifier on exit.
        if (host.getActiveBinaryMarker() == null) {
            preBinaryLabelStore = host.getLabelStore();
            preBinaryClassifier = host.getClassifier();
            preBinaryImportedTrainingRows =
                    host.getImportedTrainingRows() == null ? null : new ArrayList<>(host.getImportedTrainingRows());
            preBinaryImportedTrainingFeatureNames = host.getImportedTrainingFeatureNames() == null
                    ? null
                    : new ArrayList<>(host.getImportedTrainingFeatureNames());
        }

        // Load binary marker's labels
        try {
            host.setLabelStore(ProjectStateManager.loadBinaryLabels(project, markerName));
        } catch (IOException ex) {
            logger.warn("Failed to load binary labels for '{}': {}", markerName, ex.getMessage());
            host.setLabelStore(new LabelStore(markerName));
        }

        // Load trained binary classifier if one exists
        host.setClassifier(null);
        try {
            ProjectStateManager.SavedState savedState = ProjectStateManager.loadBinaryState(project, markerName);
            if (savedState != null
                    && (savedState.xgboostModelBase64 != null
                            || savedState.lightgbmModelBase64 != null
                            || savedState.rfModel1Base64 != null)) {
                var cs = new ClassifierState(
                        savedState.name,
                        savedState.featureNames,
                        savedState.classNames,
                        ProjectStateManager.decodeXGBoostModel(savedState),
                        ProjectStateManager.decodeLightGBMModel(savedState),
                        ProjectStateManager.decodeRFModel1(savedState),
                        ProjectStateManager.decodeRFModel2(savedState),
                        ProjectStateManager.getModel1Type(savedState),
                        ProjectStateManager.getModel2Type(savedState));
                var classifier = new DualModelClassifier();
                classifier.loadFromState(cs);
                classifier.setTrainingMetrics(
                        savedState.model1TrainMetrics, savedState.model1ValMetrics,
                        savedState.model2TrainMetrics, savedState.model2ValMetrics);
                host.setClassifier(classifier);
            }
        } catch (Exception ex) {
            logger.warn("Failed to load binary classifier state for '{}': {}", markerName, ex.getMessage());
            host.setClassifier(null);
        }

        try {
            var imported = ProjectStateManager.loadBinaryImportedTrainingData(project, markerName);
            if (imported != null) {
                host.setImportedTrainingFeatureNames(new ArrayList<>(imported.featureNames()));
                host.setImportedTrainingRows(new ArrayList<>(imported.rows()));
                logger.info(
                        "[SP Classify] Loaded binary imported training rows for '{}' ({} rows)",
                        markerName,
                        imported.rows().size());
            } else {
                host.setImportedTrainingFeatureNames(null);
                host.setImportedTrainingRows(null);
            }
        } catch (Exception ex) {
            logger.warn("Failed to load binary imported rows for '{}': {}", markerName, ex.getMessage());
            host.setImportedTrainingFeatureNames(null);
            host.setImportedTrainingRows(null);
        }

        host.setActiveBinaryMarker(markerName);

        // Resolve the allowed class names for UI restriction.
        //
        // CRITICAL: A binary classifier for marker X is ALWAYS exactly two classes
        // {X+, X-}. We must NEVER derive this set from the on-disk label store,
        // because if that file is contaminated with foreign classes (e.g. PD-1+/-
        // labels in a GrB store) we would then "validate" those foreign classes
        // and every subsequent retainClasses() filter would become a no-op,
        // permanently preserving the contamination on every save.
        //
        // The trained classifier is allowed to override the canonical pair only
        // when its class list is a subset of {markerName+, markerName-} — this
        // covers degenerate single-class classifiers but never widens the set.
        LinkedHashSet<String> canonical = new LinkedHashSet<>();
        canonical.add(markerName + "+");
        canonical.add(markerName + "-");
        DualModelClassifier classifier = host.getClassifier();
        if (classifier != null
                && classifier.getClassNames() != null
                && !classifier.getClassNames().isEmpty()
                && canonical.containsAll(classifier.getClassNames())) {
            host.setActiveBinaryClassNames(List.copyOf(classifier.getClassNames()));
        } else {
            host.setActiveBinaryClassNames(List.copyOf(canonical));
        }

        // Self-heal: unconditionally drop any labels whose class is not in the
        // canonical set for this marker. This evicts cross-contamination from
        // earlier sessions where labels from other classifiers bled into this
        // marker's store (e.g. via collectLabelsFromAnnotations or older buggy
        // save paths). Runs every time the user enters binary mode so a single
        // open of the panel is enough to clean a contaminated file.
        LabelStore labelStore = host.getLabelStore();
        if (labelStore != null && labelStore.size() > 0) {
            int removed = labelStore.retainClasses(canonical);
            if (removed > 0) {
                logger.info(
                        "[SP Classify] Pruned {} foreign-class labels from binary store '{}' on entry",
                        removed,
                        markerName);
                try {
                    ProjectStateManager.saveBinaryLabels(project, markerName, labelStore);
                } catch (IOException ex) {
                    logger.warn(
                            "Failed to persist self-healed binary labels for '{}': {}", markerName, ex.getMessage());
                }
                // Also scrub every per-image label file for this marker so that
                // disk state matches the now-cleaned in-memory state. Without
                // this, a stale per-image file would be re-loaded on the next
                // image switch and reintroduce foreign labels.
                scrubBinaryPerImageLabels(project, markerName, canonical);
            }
        }

        host.syncPanelState();
        logger.info(
                "[SP Classify] Entered binary mode for marker '{}' (classes: {})",
                markerName,
                host.getActiveBinaryClassNames());

        // Select and expand the docked classification panel for immediate use
        host.selectAndExpandDockPanel(qupath);
    }

    /**
     * Exit binary mode and restore the multi-class classifier state.
     * Saves the binary marker's current labels before restoring.
     *
     * @param qupath the QuPath GUI
     */
    void exitBinaryMode(QuPathGUI qupath) {
        String activeBinaryMarker = host.getActiveBinaryMarker();
        if (activeBinaryMarker == null) return;
        var project = qupath.getProject();

        // Save binary labels before exiting
        LabelStore labelStore = host.getLabelStore();
        if (project != null && labelStore != null) {
            // Defence-in-depth: filter out any labels that don't belong to this binary
            // classifier's classes before persisting the canonical state.
            List<String> activeBinaryClassNames = host.getActiveBinaryClassNames();
            if (activeBinaryClassNames != null && !activeBinaryClassNames.isEmpty()) {
                labelStore.retainClasses(new LinkedHashSet<>(activeBinaryClassNames));
            }
            try {
                ProjectStateManager.saveBinaryLabels(project, activeBinaryMarker, labelStore);
            } catch (IOException ex) {
                logger.warn("Failed to save binary labels for '{}' on exit: {}", activeBinaryMarker, ex.getMessage());
            }
        }

        // Restore multi-class state
        host.setLabelStore((preBinaryLabelStore != null) ? preBinaryLabelStore : new LabelStore("CellTune"));
        host.setClassifier(preBinaryClassifier);
        host.setImportedTrainingRows(preBinaryImportedTrainingRows);
        host.setImportedTrainingFeatureNames(preBinaryImportedTrainingFeatureNames);
        host.setActiveBinaryMarker(null);
        host.setActiveBinaryClassNames(null);
        this.preBinaryLabelStore = null;
        this.preBinaryClassifier = null;
        this.preBinaryImportedTrainingRows = null;
        this.preBinaryImportedTrainingFeatureNames = null;

        host.syncPanelState();
        logger.info("[SP Classify] Exited binary mode — restored multi-class state");
    }

    /**
     * Rewrite every per-image label file under the given binary marker scope,
     * stripping any entries whose class is not in {@code allowedClasses}. Files
     * that become empty are deleted. Errors on individual files are logged and
     * do not abort the scrub.
     */
    private static void scrubBinaryPerImageLabels(Project<?> project, String markerName, Set<String> allowedClasses) {
        if (project == null || markerName == null || allowedClasses == null || allowedClasses.isEmpty()) return;
        try {
            var files = ProjectStateManager.listImageLabelFiles(project, markerName);
            int filesScrubbed = 0;
            int totalRemoved = 0;
            for (var file : files) {
                try {
                    var labels = ProjectStateManager.readImageLabelsRaw(file);
                    if (labels.isEmpty()) continue;
                    int before = labels.size();
                    labels.entrySet().removeIf(e -> !allowedClasses.contains(e.getValue()));
                    int removed = before - labels.size();
                    if (removed == 0) continue;
                    totalRemoved += removed;
                    filesScrubbed++;
                    if (labels.isEmpty()) {
                        java.nio.file.Files.deleteIfExists(file);
                    } else {
                        ProjectStateManager.writeImageLabelsRaw(file, labels);
                    }
                } catch (Exception ex) {
                    logger.warn("Failed to scrub per-image labels file {}: {}", file, ex.getMessage());
                }
            }
            if (totalRemoved > 0) {
                logger.info(
                        "[SP Classify] Scrubbed {} foreign-class labels from {} per-image file(s) for binary marker '{}'",
                        totalRemoved,
                        filesScrubbed,
                        markerName);
            }
        } catch (Exception ex) {
            logger.warn(
                    "Failed to enumerate per-image label files for binary marker '{}': {}",
                    markerName,
                    ex.getMessage());
        }
    }
}
