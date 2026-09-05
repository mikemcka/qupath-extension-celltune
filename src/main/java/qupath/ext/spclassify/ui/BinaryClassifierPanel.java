package qupath.ext.spclassify.ui;

import java.util.List;
import java.util.function.Consumer;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;
import javafx.scene.control.Separator;
import javafx.scene.control.TextInputDialog;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import qupath.ext.spclassify.io.BinaryClassifierRegistry;
import qupath.fx.dialogs.Dialogs;

/**
 * A {@link VBox} dialog panel for managing per-marker binary classifiers.
 * <p>
 * Shown inside a modal {@code Stage} by {@code SpClassifyExtension.showBinaryClassifiers()}.
 * The panel lists all registered marker names and provides Create / Open / Delete actions.
 * When binary mode is active it also shows a banner and an "Exit Binary Mode" button.
 * <p>
 * All user actions are delegated to callbacks set by the owning {@code SpClassifyExtension}
 * so this panel has no direct dependency on the extension or the project.
 */
public class BinaryClassifierPanel extends VBox {

    private final ListView<String> markerList = new ListView<>();
    private final ObservableList<String> markerNames = FXCollections.observableArrayList();

    private Consumer<String> onOpenMarker;
    private Consumer<String> onRegisterMarker;
    private Consumer<String> onDeleteMarker;
    private Runnable onExitBinaryMode;

    private final Label activeBannerLabel;
    private final Button exitButton;

    public BinaryClassifierPanel() {
        super(8);
        setPadding(new Insets(12));

        // ── Title ──────────────────────────────────────────────────────────────
        Label title = new Label("Binary Classifiers");
        title.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");

        Label subtitle = new Label("Train a separate pos/neg classifier per marker.");
        subtitle.setStyle("-fx-text-fill: #666666; -fx-font-size: 11px;");

        // ── Marker list ────────────────────────────────────────────────────────
        markerList.setItems(markerNames);
        markerList.setPrefHeight(200);
        VBox.setVgrow(markerList, Priority.ALWAYS);

        // ── Action buttons ─────────────────────────────────────────────────────
        Button createButton = new Button("Create...");
        Button openButton = new Button("Open");
        Button deleteButton = new Button("Delete");

        openButton
                .disableProperty()
                .bind(markerList.getSelectionModel().selectedItemProperty().isNull());
        deleteButton
                .disableProperty()
                .bind(markerList.getSelectionModel().selectedItemProperty().isNull());

        createButton.setOnAction(e -> handleCreate());
        openButton.setOnAction(e -> handleOpen());
        deleteButton.setOnAction(e -> handleDelete());

        HBox buttonBar = new HBox(6, createButton, openButton, deleteButton);

        // ── Active-mode banner (hidden by default) ─────────────────────────────
        activeBannerLabel = new Label();
        activeBannerLabel.setStyle("-fx-font-weight: bold; -fx-text-fill: #1976D2;");
        activeBannerLabel.setVisible(false);
        activeBannerLabel.setManaged(false);

        exitButton = new Button("Exit Binary Mode");
        exitButton.setVisible(false);
        exitButton.setManaged(false);
        exitButton.setOnAction(e -> {
            if (onExitBinaryMode != null) onExitBinaryMode.run();
        });

        getChildren().addAll(title, subtitle, markerList, buttonBar, new Separator(), activeBannerLabel, exitButton);
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    /** Replace the displayed marker names with the supplied list. */
    public void setMarkerNames(List<String> names) {
        markerNames.setAll(names);
    }

    /** Called with the sanitized marker name when the user clicks Open. */
    public void setOnOpenMarker(Consumer<String> cb) {
        this.onOpenMarker = cb;
    }

    /** Called with the sanitized marker name when the user successfully creates a new classifier. */
    public void setOnRegisterMarker(Consumer<String> cb) {
        this.onRegisterMarker = cb;
    }

    /** Called with the sanitized marker name when the user confirms deletion. */
    public void setOnDeleteMarker(Consumer<String> cb) {
        this.onDeleteMarker = cb;
    }

    /** Called when the user clicks "Exit Binary Mode". */
    public void setOnExitBinaryMode(Runnable cb) {
        this.onExitBinaryMode = cb;
    }

    /**
     * Show or hide the "binary mode active" banner.
     *
     * @param markerName the active marker name, or null / blank to hide the banner
     */
    public void setActiveBinaryMarker(String markerName) {
        boolean active = (markerName != null && !markerName.isBlank());
        activeBannerLabel.setText(active ? "Active binary mode: " + markerName : "");
        activeBannerLabel.setVisible(active);
        activeBannerLabel.setManaged(active);
        exitButton.setVisible(active);
        exitButton.setManaged(active);
    }

    // ── Private handlers ───────────────────────────────────────────────────────

    private void handleCreate() {
        var dialog = new TextInputDialog();
        dialog.setTitle("New Binary Classifier");
        dialog.setHeaderText("Enter marker name (e.g. CD4, CD3, CD20):");
        dialog.setContentText("Marker name:");
        dialog.showAndWait().ifPresent(raw -> {
            try {
                String safe = BinaryClassifierRegistry.sanitizeMarkerName(raw);
                if (!markerNames.contains(safe)) {
                    markerNames.add(safe);
                    markerList.getSelectionModel().select(safe);
                }
                if (onRegisterMarker != null) onRegisterMarker.accept(safe);
            } catch (IllegalArgumentException ex) {
                Dialogs.showErrorMessage("Binary Classifiers", "Invalid marker name: " + ex.getMessage());
            }
        });
    }

    private void handleOpen() {
        String selected = markerList.getSelectionModel().getSelectedItem();
        if (selected == null) {
            Dialogs.showWarningNotification("Binary Classifiers", "Select a marker first.");
            return;
        }
        if (onOpenMarker != null) onOpenMarker.accept(selected);
        // Close the dialog so the user can see the panel switch into binary mode
        if (getScene() != null && getScene().getWindow() != null) {
            getScene().getWindow().hide();
        }
    }

    private void handleDelete() {
        String selected = markerList.getSelectionModel().getSelectedItem();
        if (selected == null) return;
        boolean confirm = Dialogs.showConfirmDialog(
                "Delete Binary Classifier", "Delete binary classifier '" + selected + "'? This cannot be undone.");
        if (confirm) {
            markerNames.remove(selected);
            if (onDeleteMarker != null) onDeleteMarker.accept(selected);
        }
    }
}
