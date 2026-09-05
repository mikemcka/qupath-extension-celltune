package qupath.ext.spclassify.ui;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.geometry.Insets;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.Modality;
import javafx.stage.Stage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.io.ClassManager;
import qupath.ext.spclassify.model.LabelStore;
import qupath.ext.spclassify.util.BackgroundExecutors;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.objects.classes.PathClass;

/**
 * Dialog for adding, deleting, and merging cell classes across the QuPath
 * class panel and all persisted {@code celltune/image-labels/*.json} files.
 *
 * <h3>Merge history</h3>
 * When classes are merged, the original class name is preserved in the label
 * value as {@code "originalClass-mergedInto(targetClass)"}.  This means:
 * <ul>
 *   <li>Training sees only the effective (post-merge) class name.</li>
 *   <li>The JSON files carry a full audit trail of every merge.</li>
 *   <li>Any merge can be undone from the <em>Undo Merge</em> section.</li>
 * </ul>
 */
public class ClassControlDialog {

    private static final Logger logger = LoggerFactory.getLogger(ClassControlDialog.class);

    private static final String EXTENSION_NAME = "SP Classify";

    private final QuPathGUI qupath;
    private final Supplier<LabelStore> labelStoreSupplier;
    private final Consumer<LabelStore> labelStoreUpdater;
    /** Optional: flush the active image's in-memory labels to disk before a class op runs. */
    private final Runnable preOpDiskSync;
    /** Optional: reload the active image's labels from disk after a class op finishes. */
    private final Runnable postOpDiskReload;

    private final Stage stage;
    private ExecutorService bgExecutor = newBgExecutor();

    private static ExecutorService newBgExecutor() {
        return BackgroundExecutors.newSingleThread("ClassControl-BG");
    }

    /** Returns a live executor, recreating it if a previous close shut it down. */
    private synchronized ExecutorService executor() {
        if (bgExecutor.isShutdown() || bgExecutor.isTerminated()) {
            bgExecutor = newBgExecutor();
        }
        return bgExecutor;
    }

    // ── Shared status bar ────────────────────────────────────────────────────
    private final Label statusLabel = new Label("Ready.");

    /**
     * @param qupath              the QuPath GUI instance
     * @param labelStoreSupplier  provides the current in-memory LabelStore (may return null)
     * @param labelStoreUpdater   called after an operation that changes the in-memory store
     */
    public ClassControlDialog(
            QuPathGUI qupath, Supplier<LabelStore> labelStoreSupplier, Consumer<LabelStore> labelStoreUpdater) {
        this(qupath, labelStoreSupplier, labelStoreUpdater, null, null);
    }

    /**
     * @param qupath              the QuPath GUI instance
     * @param labelStoreSupplier  provides the current in-memory LabelStore (may return null)
     * @param labelStoreUpdater   called after an operation that changes the in-memory store
     * @param preOpDiskSync       optional: invoked on the FX thread before each class
     *                            operation runs, to flush the active image's in-memory
     *                            labels to disk. Prevents stale memory from clobbering
     *                            freshly-merged on-disk entries when the active image
     *                            is later re-saved.
     * @param postOpDiskReload    optional: invoked on the FX thread after each class
     *                            operation finishes, to reload the active image's
     *                            labels from disk so memory matches the new disk state.
     */
    public ClassControlDialog(
            QuPathGUI qupath,
            Supplier<LabelStore> labelStoreSupplier,
            Consumer<LabelStore> labelStoreUpdater,
            Runnable preOpDiskSync,
            Runnable postOpDiskReload) {
        this.qupath = qupath;
        this.labelStoreSupplier = labelStoreSupplier;
        this.labelStoreUpdater = labelStoreUpdater;
        this.preOpDiskSync = preOpDiskSync;
        this.postOpDiskReload = postOpDiskReload;
        this.stage = buildStage();
    }

    public void show() {
        if (!stage.isShowing()) stage.show();
        else stage.toFront();
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Stage construction
    // ─────────────────────────────────────────────────────────────────────────

    private Stage buildStage() {
        var root = new VBox(0);

        var tabPane = new TabPane();
        tabPane.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE);
        tabPane.getTabs().addAll(buildAddTab(), buildDeleteTab(), buildMergeTab(), buildUndoMergeTab());

        statusLabel.setStyle("-fx-font-size: 11; -fx-text-fill: #555;");
        var statusBar = new HBox(statusLabel);
        statusBar.setPadding(new Insets(4, 10, 4, 10));
        statusBar.setStyle("-fx-border-color: #ccc; -fx-border-width: 1 0 0 0;");

        VBox.setVgrow(tabPane, Priority.ALWAYS);
        root.getChildren().addAll(tabPane, statusBar);

        var s = new Stage();
        s.setTitle(EXTENSION_NAME + " — Class Control");
        s.initOwner(qupath.getStage());
        s.initModality(Modality.NONE);
        s.setScene(new javafx.scene.Scene(root, 440, 480));
        // NB: do not shut down bgExecutor here — the dialog instance is cached and
        // reused across menu invocations. The executor uses daemon threads so it
        // will die with the JVM; executor() recreates it if it ever does shut down.
        return s;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Add tab
    // ─────────────────────────────────────────────────────────────────────────

    private Tab buildAddTab() {
        var nameField = new TextField();
        nameField.setPromptText("New class name…");

        var addBtn = new Button("Add Class");
        addBtn.setMaxWidth(Double.MAX_VALUE);
        addBtn.setDefaultButton(true);
        addBtn.setOnAction(e -> {
            String name = nameField.getText().trim();
            if (name.isEmpty()) {
                setStatus("Enter a class name first.");
                return;
            }
            ClassManager.addPathClass(qupath, name);
            nameField.clear();
            setStatus("Added class: " + name);
        });

        var box = new VBox(
                10,
                label("Enter the name of a new classification class to add to the QuPath class panel."),
                nameField,
                addBtn);
        box.setPadding(new Insets(16));

        return new Tab("Add", box);
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Delete tab
    // ─────────────────────────────────────────────────────────────────────────

    private Tab buildDeleteTab() {
        var classListView = new ListView<String>();
        classListView.setItems(FXCollections.observableArrayList(currentEffectiveClassNames()));
        classListView.getSelectionModel().setSelectionMode(SelectionMode.SINGLE);
        VBox.setVgrow(classListView, Priority.ALWAYS);

        var purgeCheck = new CheckBox("Also remove labels with this class from all image-label files");
        purgeCheck.setWrapText(true);

        var refreshBtn = new Button("Refresh list");
        refreshBtn.setOnAction(
                e -> classListView.setItems(FXCollections.observableArrayList(currentEffectiveClassNames())));

        var deleteBtn = new Button("Delete Selected Class");
        deleteBtn.setMaxWidth(Double.MAX_VALUE);
        deleteBtn.setStyle("-fx-base: #d9534f;");
        deleteBtn.setOnAction(e -> {
            String selected = classListView.getSelectionModel().getSelectedItem();
            if (selected == null) {
                setStatus("Select a class to delete.");
                return;
            }
            boolean purge = purgeCheck.isSelected();
            String msg = purge
                    ? "Delete class \"" + selected
                            + "\" and remove all its labels from every image file? This cannot be undone."
                    : "Remove class \"" + selected
                            + "\" from the QuPath class panel? Labels already stored will remain.";
            if (!Dialogs.showConfirmDialog("Delete Class", msg)) return;

            deleteBtn.setDisable(true);
            setStatus("Deleting…");
            runPreOpSync();
            try {
                executor().submit(() -> {
                    try {
                        int removed = ClassManager.deletePathClass(qupath, selected, purge);
                        Platform.runLater(() -> {
                            classListView.setItems(FXCollections.observableArrayList(currentEffectiveClassNames()));
                            runPostOpReload();
                            setStatus("Deleted \"" + selected + "\""
                                    + (purge ? " (" + removed + " label entries purged)." : "."));
                        });
                    } catch (Exception ex) {
                        logger.error("Delete class failed", ex);
                        Platform.runLater(() -> {
                            Dialogs.showErrorMessage(EXTENSION_NAME, "Delete failed: " + ex.getMessage());
                            setStatus("Error: " + ex.getMessage());
                        });
                    } finally {
                        Platform.runLater(() -> deleteBtn.setDisable(false));
                    }
                });
            } catch (RejectedExecutionException rex) {
                logger.error("Delete task could not be submitted", rex);
                deleteBtn.setDisable(false);
                Dialogs.showErrorMessage(EXTENSION_NAME, "Could not start delete task: " + rex.getMessage());
                setStatus("Error: could not start delete task.");
            }
        });

        var box = new VBox(
                10,
                label("Select a class to remove from the QuPath class panel.\n"
                        + "Tick the checkbox to also purge its labels from disk."),
                classListView,
                purgeCheck,
                new HBox(8, refreshBtn, hSpacer(), deleteBtn));
        box.setPadding(new Insets(16));

        return new Tab("Delete", box);
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Merge tab
    // ─────────────────────────────────────────────────────────────────────────

    private Tab buildMergeTab() {
        // Source class multi-select
        var sourceListView = new ListView<String>();
        sourceListView.setItems(FXCollections.observableArrayList(currentEffectiveClassNames()));
        sourceListView.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE);
        VBox.setVgrow(sourceListView, Priority.ALWAYS);

        var refreshMergeBtn = new Button("Refresh");
        refreshMergeBtn.setOnAction(
                e -> sourceListView.setItems(FXCollections.observableArrayList(currentEffectiveClassNames())));

        // Target class — either pick an existing class or type a new one
        var targetField = new TextField();
        targetField.setPromptText("Target class name (new or existing)…");

        var existingCombo = new ComboBox<String>();
        existingCombo.setPromptText("…or pick existing");
        existingCombo.setMaxWidth(Double.MAX_VALUE);
        existingCombo.setItems(FXCollections.observableArrayList(currentEffectiveClassNames()));
        existingCombo.setOnAction(e -> {
            if (existingCombo.getValue() != null) targetField.setText(existingCombo.getValue());
        });

        var infoLabel = label("Original class names are encoded in the label file for undo:\n"
                + "\"test1\" merged into \"myType\" → stored as \"test1-mergedInto(myType)\".\n"
                + "Training always sees only the effective class name.");
        infoLabel.setStyle("-fx-font-size: 11; -fx-text-fill: #555;");

        var mergeBtn = new Button("Merge Selected → Target");
        mergeBtn.setMaxWidth(Double.MAX_VALUE);
        mergeBtn.setStyle("-fx-base: #337ab7;");
        mergeBtn.setOnAction(e -> {
            List<String> sources =
                    new ArrayList<>(sourceListView.getSelectionModel().getSelectedItems());
            String target = targetField.getText().trim();
            if (sources.isEmpty()) {
                setStatus("Select at least one source class.");
                return;
            }
            if (target.isEmpty()) {
                setStatus("Enter or select a target class name.");
                return;
            }
            if (sources.size() == 1 && sources.get(0).equals(target)) {
                setStatus("Source and target are the same — nothing to merge.");
                return;
            }

            String sourceSummary = sources.stream().collect(Collectors.joining(", "));
            if (!Dialogs.showConfirmDialog(
                    "Merge Classes",
                    "Merge " + sources.size() + " class(es) [" + sourceSummary + "] into \"" + target
                            + "\"?\n\nLabel files will be updated. Use Undo Merge to reverse.")) return;

            mergeBtn.setDisable(true);
            setStatus("Merging…");
            runPreOpSync();
            LabelStore inMemory = labelStoreSupplier.get();
            try {
                executor().submit(() -> {
                    try {
                        int count = ClassManager.mergeClasses(qupath, sources, target, inMemory);
                        Platform.runLater(() -> {
                            if (inMemory != null) labelStoreUpdater.accept(inMemory);
                            runPostOpReload();
                            // Refresh both source lists
                            List<String> refreshed = currentEffectiveClassNames();
                            sourceListView.setItems(FXCollections.observableArrayList(refreshed));
                            existingCombo.setItems(FXCollections.observableArrayList(refreshed));
                            targetField.clear();
                            setStatus("Merged " + sources.size() + " class(es) → \"" + target + "\" (" + count
                                    + " labels updated).");
                        });
                    } catch (Exception ex) {
                        logger.error("Merge classes failed", ex);
                        Platform.runLater(() -> {
                            Dialogs.showErrorMessage(EXTENSION_NAME, "Merge failed: " + ex.getMessage());
                            setStatus("Error: " + ex.getMessage());
                        });
                    } finally {
                        Platform.runLater(() -> mergeBtn.setDisable(false));
                    }
                });
            } catch (RejectedExecutionException rex) {
                logger.error("Merge task could not be submitted", rex);
                mergeBtn.setDisable(false);
                Dialogs.showErrorMessage(EXTENSION_NAME, "Could not start merge task: " + rex.getMessage());
                setStatus("Error: could not start merge task.");
            }
        });

        var box = new VBox(
                10,
                label("Select one or more source classes (hold Ctrl/Cmd to multi-select),\n"
                        + "then enter or choose the target class name."),
                sourceListView,
                new HBox(8, refreshMergeBtn),
                new HBox(8, new Label("Target:"), targetField),
                new HBox(8, new Label("Existing:"), existingCombo),
                infoLabel,
                mergeBtn);
        box.setPadding(new Insets(16));

        return new Tab("Merge", box);
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Undo Merge tab
    // ─────────────────────────────────────────────────────────────────────────

    private Tab buildUndoMergeTab() {
        var targetCombo = new ComboBox<String>();
        targetCombo.setPromptText("Merged-into class to undo…");
        targetCombo.setMaxWidth(Double.MAX_VALUE);

        var refreshUndoBtn = new Button("Refresh");
        refreshUndoBtn.setOnAction(e -> refreshUndoCombo(targetCombo));
        refreshUndoCombo(targetCombo);

        var infoLabel = label("Select the class that was the merge TARGET to restore all source\n"
                + "classes to their original names. Only labels carrying\n"
                + "merge-history annotations are changed.\n\n"
                + "Note: the source PathClasses are re-added to the QuPath\n"
                + "class panel automatically.");
        infoLabel.setStyle("-fx-font-size: 11; -fx-text-fill: #555;");

        var undoBtn = new Button("Undo Merge for Selected Class");
        undoBtn.setMaxWidth(Double.MAX_VALUE);
        undoBtn.setOnAction(e -> {
            String target = targetCombo.getValue();
            if (target == null) {
                setStatus("Select a merged-into class first.");
                return;
            }

            if (!Dialogs.showConfirmDialog(
                    "Undo Merge",
                    "Restore all labels that were merged into \"" + target + "\" back to their original class names?\n"
                            + "The source PathClasses will be re-added to the QuPath class panel.")) return;

            undoBtn.setDisable(true);
            setStatus("Undoing merge…");
            runPreOpSync();
            LabelStore inMemory = labelStoreSupplier.get();
            try {
                executor().submit(() -> {
                    try {
                        int count = ClassManager.undoMerge(qupath, target, inMemory);
                        Platform.runLater(() -> {
                            if (inMemory != null) labelStoreUpdater.accept(inMemory);
                            runPostOpReload();
                            refreshUndoCombo(targetCombo);
                            setStatus("Restored " + count + " labels merged into \"" + target + "\".");
                        });
                    } catch (Exception ex) {
                        logger.error("Undo merge failed", ex);
                        Platform.runLater(() -> {
                            Dialogs.showErrorMessage(EXTENSION_NAME, "Undo failed: " + ex.getMessage());
                            setStatus("Error: " + ex.getMessage());
                        });
                    } finally {
                        Platform.runLater(() -> undoBtn.setDisable(false));
                    }
                });
            } catch (RejectedExecutionException rex) {
                logger.error("Undo task could not be submitted", rex);
                undoBtn.setDisable(false);
                Dialogs.showErrorMessage(EXTENSION_NAME, "Could not start undo task: " + rex.getMessage());
                setStatus("Error: could not start undo task.");
            }
        });

        var box = new VBox(
                10,
                label("Undo a previous merge operation."),
                new HBox(8, targetCombo, refreshUndoBtn),
                infoLabel,
                undoBtn);
        box.setPadding(new Insets(16));

        return new Tab("Undo Merge", box);
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Helpers
    // ─────────────────────────────────────────────────────────────────────────

    /** Current effective class names from QuPath's available PathClasses (FX thread). */
    private List<String> currentEffectiveClassNames() {
        return qupath.getAvailablePathClasses().stream()
                .filter(pc ->
                        pc != null && pc.getName() != null && !pc.getName().isEmpty())
                .map(PathClass::getName)
                .collect(Collectors.toList());
    }

    /**
     * Populate the undo-merge combo with class names that appear as merge targets
     * in the current QuPath available classes AND exist as merge targets in memory.
     * As a simpler proxy, just offer all current class names (the user knows which
     * ones were produced by merges).
     */
    private void refreshUndoCombo(ComboBox<String> combo) {
        var project = qupath.getProject();
        if (project == null) {
            combo.setItems(FXCollections.observableArrayList(currentEffectiveClassNames()));
            return;
        }
        // Collect unique effective merge-target names from label files in background
        List<String> targets = new ArrayList<>();
        try {
            for (var path : ClassManager.listImageLabelFiles(project)) {
                String json = java.nio.file.Files.readString(path, java.nio.charset.StandardCharsets.UTF_8);
                @SuppressWarnings("unchecked")
                var map = new com.google.gson.Gson()
                        .<java.util.Map<String, String>>fromJson(
                                json,
                                new com.google.gson.reflect.TypeToken<java.util.Map<String, String>>() {}.getType());
                if (map != null) {
                    for (String raw : map.values()) {
                        int start = raw.indexOf("-mergedInto(");
                        if (start >= 0 && raw.endsWith(")")) {
                            String t = raw.substring(start + "-mergedInto(".length(), raw.length() - 1);
                            if (!targets.contains(t)) targets.add(t);
                        }
                    }
                }
            }
        } catch (IOException e) {
            logger.warn("Could not scan label files for merge targets: {}", e.getMessage());
        }
        if (targets.isEmpty()) targets.addAll(currentEffectiveClassNames());
        combo.setItems(FXCollections.observableArrayList(targets));
    }

    /** Flush the active image's in-memory labels to disk before a class op. */
    private void runPreOpSync() {
        if (preOpDiskSync == null) return;
        try {
            preOpDiskSync.run();
        } catch (Exception ex) {
            logger.warn("Pre-op disk sync failed: {}", ex.getMessage());
        }
    }

    /** Reload the active image's labels from disk after a class op. */
    private void runPostOpReload() {
        if (postOpDiskReload == null) return;
        try {
            postOpDiskReload.run();
        } catch (Exception ex) {
            logger.warn("Post-op disk reload failed: {}", ex.getMessage());
        }
    }

    private static Label label(String text) {
        var l = new Label(text);
        l.setWrapText(true);
        return l;
    }

    private static Region hSpacer() {
        var r = new Region();
        HBox.setHgrow(r, Priority.ALWAYS);
        return r;
    }

    private void setStatus(String msg) {
        if (Platform.isFxApplicationThread()) {
            statusLabel.setText(msg);
        } else {
            Platform.runLater(() -> statusLabel.setText(msg));
        }
    }
}
