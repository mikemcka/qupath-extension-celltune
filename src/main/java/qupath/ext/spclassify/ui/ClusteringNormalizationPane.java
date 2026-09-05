package qupath.ext.spclassify.ui;

import java.util.*;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.collections.transformation.FilteredList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.Modality;
import javafx.stage.Stage;
import qupath.ext.spclassify.model.FeatureNormalizer;
import qupath.ext.spclassify.model.FeatureNormalizer.Transform;
import qupath.lib.gui.QuPathGUI;

/**
 * Checkbox-based dialog for selecting which features to normalise.
 * <p>
 * The user picks a single transform type (arcsinh or sqrt) at the top,
 * then ticks the features to apply it to — matching the familiar
 * {@link FeatureSelectionPane} pattern.
 * <p>
 * Recommended arcsinh cofactors (scale-dependent — tune to your data):
 * <ul>
 *   <li><b>~25–50</b> — raw fluorescence panels (COMET, CODEX, IF), values in the hundreds–thousands</li>
 *   <li><b>0.05</b> — MIBI mass-spectrometry mean intensities (Hartmann et al. 2021; squidpy MIBI-TOF tutorial)</li>
 * </ul>
 * The ideal value tracks the measurement intensity scale — see the User Guide (§4.2).
 */
public class ClusteringNormalizationPane {

    private final Stage stage;
    private final QuPathGUI qupath;
    private final List<String> featureNames;
    private final ObservableList<FeatureItem> allItems = FXCollections.observableArrayList();
    private final FilteredList<FeatureItem> filteredItems;
    private final ListView<FeatureItem> listView;
    private final TextField searchField;
    private final ComboBox<String> prefixCombo;
    private final ComboBox<Transform> transformCombo;
    private final Spinner<Double> cofactorSpinner;
    private final Label cofactorLabel;
    private final Label cofactorHint;
    private final Button suggestBtn;
    private final Label countLabel;

    private boolean confirmed = false;

    /**
     * Create the normalization dialog.
     *
     * @param qupath       QuPath handle, forwarded to the Suggest cofactor tool (Phase 17)
     * @param owner        parent stage
     * @param featureNames all available feature names
     * @param existing     existing normalizer to pre-populate from (may be null)
     */
    public ClusteringNormalizationPane(
            QuPathGUI qupath, Stage owner, List<String> featureNames, FeatureNormalizer existing) {
        stage = new Stage();
        this.qupath = qupath;
        this.featureNames = featureNames;

        // Pre-populate: tick features that already have a transform
        for (String name : featureNames) {
            boolean selected = existing != null && existing.getTransform(name) != Transform.NONE;
            allItems.add(new FeatureItem(name, selected));
        }

        // Infer the transform type from existing config (all features share one type)
        Transform existingTransform = Transform.ARCSINH;
        if (existing != null && existing.hasTransforms()) {
            existingTransform = existing.getAllTransforms().values().iterator().next();
        }

        // ── Transform type selector ──
        transformCombo = new ComboBox<>();
        transformCombo.getItems().addAll(Transform.ARCSINH, Transform.SQRT);
        transformCombo.setValue(existingTransform);
        transformCombo.setOnAction(e -> updateCofactorVisibility());

        // ── Cofactor ──
        cofactorLabel = new Label("Cofactor:");
        cofactorSpinner = new Spinner<>(0.01, 10000.0, existing != null ? existing.getArcsinhCofactor() : 1.0, 1.0);
        cofactorSpinner.setEditable(true);
        cofactorSpinner.setPrefWidth(100);

        cofactorHint = new Label("(fluor ~25–50, MIBI 0.05 — scale-dependent)");
        cofactorHint.setStyle("-fx-text-fill: #888; -fx-font-size: 11;");

        // ── Suggest cofactor tool (arcsinh-only; Phase 17 COF-01 / D-06) ──
        suggestBtn = new Button("Suggest…");
        suggestBtn.setOnAction(e -> openSuggestTool());

        HBox transformRow = new HBox(
                8, new Label("Transform:"), transformCombo, cofactorLabel, cofactorSpinner, suggestBtn, cofactorHint);
        transformRow.setAlignment(Pos.CENTER_LEFT);

        // ── Search / filter ──
        searchField = new TextField();
        searchField.setPromptText("Search features…");
        HBox.setHgrow(searchField, Priority.ALWAYS);

        prefixCombo = new ComboBox<>();
        prefixCombo.getItems().add("All prefixes");
        prefixCombo.getItems().addAll(FeatureSelectionPane.discoverPrefixes(featureNames));
        prefixCombo.getSelectionModel().selectFirst();
        prefixCombo.setPrefWidth(180);

        filteredItems = new FilteredList<>(allItems, p -> true);
        searchField.textProperty().addListener((o, a, b) -> updateFilter());
        prefixCombo.valueProperty().addListener((o, a, b) -> updateFilter());

        HBox filterRow = new HBox(6, new Label("Filter:"), searchField, prefixCombo);
        filterRow.setAlignment(Pos.CENTER_LEFT);

        // ── Buttons ──
        Button selectAllBtn = new Button("Select All");
        selectAllBtn.setOnAction(e -> setAllVisible(true));

        Button clearAllBtn = new Button("Clear All");
        clearAllBtn.setOnAction(e -> setAllVisible(false));

        Button selectPrefixBtn = new Button("Select Prefix");
        selectPrefixBtn.setOnAction(e -> selectCurrentPrefix());

        Button clearPrefixBtn = new Button("Clear Prefix");
        clearPrefixBtn.setOnAction(e -> clearCurrentPrefix());

        HBox btnRow = new HBox(
                6,
                selectAllBtn,
                clearAllBtn,
                new Separator(javafx.geometry.Orientation.VERTICAL),
                selectPrefixBtn,
                clearPrefixBtn);
        btnRow.setAlignment(Pos.CENTER_LEFT);

        // ── List view with checkboxes ──
        listView = new ListView<>(filteredItems);
        listView.setCellFactory(lv -> new CheckBoxListCell());
        listView.setPrefHeight(400);
        listView.setPrefWidth(500);
        VBox.setVgrow(listView, Priority.ALWAYS);

        // ── Count label ──
        countLabel = new Label();
        updateCount();

        // ── OK / Cancel ──
        Button okBtn = new Button("OK");
        okBtn.setDefaultButton(true);
        okBtn.setPrefWidth(80);
        okBtn.setOnAction(e -> {
            confirmed = true;
            stage.close();
        });

        Button cancelBtn = new Button("Cancel");
        cancelBtn.setCancelButton(true);
        cancelBtn.setPrefWidth(80);
        cancelBtn.setOnAction(e -> stage.close());

        HBox okCancelRow = new HBox(
                8,
                countLabel,
                new Region() {
                    {
                        HBox.setHgrow(this, Priority.ALWAYS);
                    }
                },
                okBtn,
                cancelBtn);
        okCancelRow.setAlignment(Pos.CENTER);

        // Clustering-only scope note. The classifier trains/predicts on raw values (tree models
        // are invariant to these monotone transforms), so normalisation here affects ONLY the
        // scale-dependent workflows: the scatter-plot clustering (k-means/Leiden), PCA, and gating.
        Label scopeNote = new Label("Note: normalisation applies to clustering / scatter-plot / gating "
                + "only — the classifier always uses raw values.");
        scopeNote.setWrapText(true);
        scopeNote.setStyle("-fx-text-fill: -fx-accent; -fx-font-size: 11px;");

        VBox root = new VBox(8, scopeNote, transformRow, filterRow, btnRow, listView, okCancelRow);
        root.setPadding(new Insets(10));

        stage.initOwner(owner);
        stage.initModality(Modality.APPLICATION_MODAL);
        stage.setTitle("Clustering normalisation");
        stage.setScene(new Scene(root, 600, 550));
        stage.setMinWidth(400);
        stage.setMinHeight(350);

        updateCofactorVisibility();
    }

    /** @return this pane's own Stage — used as the owner for the non-modal Suggest cofactor window. */
    public Stage getStage() {
        return stage;
    }

    /**
     * Show the dialog and block until the user confirms or cancels.
     *
     * @return a configured {@link FeatureNormalizer}, or null if cancelled
     */
    public FeatureNormalizer showAndWait() {
        confirmed = false;
        stage.showAndWait();
        if (!confirmed) return null;

        Transform transform = transformCombo.getValue();
        FeatureNormalizer normalizer = new FeatureNormalizer();
        if (transform == Transform.ARCSINH) {
            normalizer.setArcsinhCofactor(cofactorSpinner.getValue());
        }

        for (FeatureItem item : allItems) {
            if (item.isSelected()) {
                normalizer.setTransform(item.getName(), transform);
            }
        }

        return normalizer;
    }

    // ── UI helpers ───────────────────────────────────────────────────────────────

    private void updateCofactorVisibility() {
        boolean isArcsinh = transformCombo.getValue() == Transform.ARCSINH;
        cofactorLabel.setVisible(isArcsinh);
        cofactorLabel.setManaged(isArcsinh);
        cofactorSpinner.setVisible(isArcsinh);
        cofactorSpinner.setManaged(isArcsinh);
        suggestBtn.setVisible(isArcsinh);
        suggestBtn.setManaged(isArcsinh);
        cofactorHint.setVisible(isArcsinh);
        cofactorHint.setManaged(isArcsinh);
    }

    /**
     * Open the non-modal Suggest cofactor tool (COF-01). The window is owned by this pane's OWN stage
     * (correction #2) — a {@code Modality.NONE} child of the {@code APPLICATION_MODAL} pane stays
     * interactive, whereas a window owned by the QuPath main stage would be blocked. The apply callback
     * writes the clamped recommended global into the existing cofactor spinner in one action, with no
     * measurement/normalizer mutation and no normalization run (COF-07); the user still confirms
     * normalization via OK, exactly as today.
     */
    private void openSuggestTool() {
        CofactorSuggestionDialog dialog = new CofactorSuggestionDialog(
                stage, // OWN stage — child of the APPLICATION_MODAL pane (correction #2)
                qupath,
                featureNames,
                global -> { // COF-07: single action, no mutation, no normalize run
                    double v = Math.max(0.01, Math.min(10000.0, global));
                    cofactorSpinner.getValueFactory().setValue(v);
                });
        dialog.show(); // non-blocking; the Clustering normalisation pane stays open (D-05)
    }

    private void updateFilter() {
        String search = searchField.getText() == null
                ? ""
                : searchField.getText().toLowerCase().strip();
        String prefix = prefixCombo.getValue();
        boolean allPrefixes = "All prefixes".equals(prefix);

        filteredItems.setPredicate(item -> {
            boolean matchesSearch =
                    search.isEmpty() || item.getName().toLowerCase().contains(search);
            boolean matchesPrefix = allPrefixes || item.getName().startsWith(prefix);
            return matchesSearch && matchesPrefix;
        });
    }

    private void setAllVisible(boolean selected) {
        for (FeatureItem item : filteredItems) {
            item.setSelected(selected);
        }
        listView.refresh();
        updateCount();
    }

    private void selectCurrentPrefix() {
        String prefix = prefixCombo.getValue();
        if ("All prefixes".equals(prefix)) {
            setAllVisible(true);
            return;
        }
        for (FeatureItem item : allItems) {
            if (item.getName().startsWith(prefix)) item.setSelected(true);
        }
        listView.refresh();
        updateCount();
    }

    private void clearCurrentPrefix() {
        String prefix = prefixCombo.getValue();
        if ("All prefixes".equals(prefix)) {
            setAllVisible(false);
            return;
        }
        for (FeatureItem item : allItems) {
            if (item.getName().startsWith(prefix)) item.setSelected(false);
        }
        listView.refresh();
        updateCount();
    }

    private void updateCount() {
        long selected = allItems.stream().filter(FeatureItem::isSelected).count();
        countLabel.setText(selected + " / " + allItems.size() + " selected for normalisation");
    }

    // ── Inner classes ───────────────────────────────────────────────────────────

    static class FeatureItem {
        private final String name;
        private boolean selected;

        FeatureItem(String name, boolean selected) {
            this.name = name;
            this.selected = selected;
        }

        String getName() {
            return name;
        }

        boolean isSelected() {
            return selected;
        }

        void setSelected(boolean s) {
            this.selected = s;
        }

        @Override
        public String toString() {
            return name;
        }
    }

    /** ListView cell with a CheckBox — same pattern as FeatureSelectionPane. */
    private class CheckBoxListCell extends ListCell<FeatureItem> {
        private final CheckBox checkBox = new CheckBox();

        CheckBoxListCell() {
            checkBox.setOnAction(e -> {
                FeatureItem item = getItem();
                if (item != null) {
                    item.setSelected(checkBox.isSelected());
                    updateCount();
                }
            });
        }

        @Override
        protected void updateItem(FeatureItem item, boolean empty) {
            super.updateItem(item, empty);
            if (empty || item == null) {
                setGraphic(null);
                setText(null);
            } else {
                checkBox.setSelected(item.isSelected());
                checkBox.setText(item.getName());
                setGraphic(checkBox);
                setText(null);
            }
        }
    }
}
