package qupath.ext.spclassify.ui;

import java.util.ArrayList;
import java.util.List;
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
import javafx.stage.Window;

/**
 * A dual-list image selector dialog, allowing the user to choose which
 * project images to include (apply predictions to) or exclude.
 *
 * <p>Layout mirrors the standard "Exclude Images" pattern:
 * <pre>
 * ┌─────────────────────────────────────────────────┐
 * │  Included Images List      Excluded Images List  │
 * │  [Search...]               [Search...]           │
 * │  ┌──────────────┐  [>>]   ┌──────────────┐      │
 * │  │ Image_01     │  [ >]   │              │      │
 * │  │ Image_02     │  [ <]   │              │      │
 * │  │ Image_03     │  [<<]   │              │      │
 * │  └──────────────┘         └──────────────┘      │
 * │       [Cancel]                [OK]               │
 * └─────────────────────────────────────────────────┘
 * </pre>
 */
public class ImageSelectionPane {

    private final Stage dialog;
    private List<String> result = null;

    // Source data backing the two lists
    private final ObservableList<String> includedItems = FXCollections.observableArrayList();
    private final ObservableList<String> excludedItems = FXCollections.observableArrayList();

    /**
     * Create the image selection dialog.
     *
     * @param owner           owner window for modality
     * @param allImageNames   all image names from the project
     * @param currentImageName name of the currently-open image (always included, non-removable)
     */
    public ImageSelectionPane(Window owner, List<String> allImageNames, String currentImageName) {
        dialog = new Stage();
        dialog.setTitle("Select Images for Classification");
        dialog.initModality(Modality.APPLICATION_MODAL);
        dialog.initOwner(owner);
        dialog.setResizable(true);

        // Start with all images included
        includedItems.addAll(allImageNames);

        // ── Left list: Included ─────────────────────────────────────────────
        Label includedLabel = new Label("Included Images");
        includedLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 13px;");

        TextField includedSearch = new TextField();
        includedSearch.setPromptText("Search...");

        FilteredList<String> filteredIncluded = new FilteredList<>(includedItems, s -> true);
        includedSearch.textProperty().addListener((obs, oldVal, newVal) -> {
            String filter = newVal == null ? "" : newVal.toLowerCase();
            filteredIncluded.setPredicate(name -> name.toLowerCase().contains(filter));
        });

        ListView<String> includedList = new ListView<>(filteredIncluded);
        includedList.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE);
        includedList.setPrefHeight(400);
        includedList.setPrefWidth(250);

        VBox leftBox = new VBox(4, includedLabel, includedSearch, includedList);

        // ── Right list: Excluded ────────────────────────────────────────────
        Label excludedLabel = new Label("Excluded Images");
        excludedLabel.setStyle("-fx-font-weight: bold; -fx-font-size: 13px;");

        TextField excludedSearch = new TextField();
        excludedSearch.setPromptText("Search...");

        FilteredList<String> filteredExcluded = new FilteredList<>(excludedItems, s -> true);
        excludedSearch.textProperty().addListener((obs, oldVal, newVal) -> {
            String filter = newVal == null ? "" : newVal.toLowerCase();
            filteredExcluded.setPredicate(name -> name.toLowerCase().contains(filter));
        });

        ListView<String> excludedList = new ListView<>(filteredExcluded);
        excludedList.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE);
        excludedList.setPrefHeight(400);
        excludedList.setPrefWidth(250);

        VBox rightBox = new VBox(4, excludedLabel, excludedSearch, excludedList);

        // ── Centre transfer buttons ─────────────────────────────────────────
        Button moveAllRight = new Button(">>");
        moveAllRight.setTooltip(new Tooltip("Exclude all images"));
        moveAllRight.setPrefWidth(40);
        moveAllRight.setOnAction(e -> {
            List<String> toMove = new ArrayList<>(includedItems);
            // Keep current image in included
            if (currentImageName != null) toMove.remove(currentImageName);
            includedItems.removeAll(toMove);
            excludedItems.addAll(toMove);
            FXCollections.sort(excludedItems);
        });

        Button moveRight = new Button(">");
        moveRight.setTooltip(new Tooltip("Exclude selected images"));
        moveRight.setPrefWidth(40);
        moveRight.setOnAction(e -> {
            List<String> selected =
                    new ArrayList<>(includedList.getSelectionModel().getSelectedItems());
            // Don't allow moving the current image
            if (currentImageName != null) selected.remove(currentImageName);
            includedItems.removeAll(selected);
            excludedItems.addAll(selected);
            FXCollections.sort(excludedItems);
        });

        Button moveLeft = new Button("<");
        moveLeft.setTooltip(new Tooltip("Include selected images"));
        moveLeft.setPrefWidth(40);
        moveLeft.setOnAction(e -> {
            List<String> selected =
                    new ArrayList<>(excludedList.getSelectionModel().getSelectedItems());
            excludedItems.removeAll(selected);
            includedItems.addAll(selected);
            FXCollections.sort(includedItems);
        });

        Button moveAllLeft = new Button("<<");
        moveAllLeft.setTooltip(new Tooltip("Include all images"));
        moveAllLeft.setPrefWidth(40);
        moveAllLeft.setOnAction(e -> {
            includedItems.addAll(excludedItems);
            excludedItems.clear();
            FXCollections.sort(includedItems);
        });

        VBox buttonBox = new VBox(8, moveAllRight, moveRight, moveLeft, moveAllLeft);
        buttonBox.setAlignment(Pos.CENTER);
        buttonBox.setPadding(new Insets(0, 8, 0, 8));

        // ── Bottom buttons ──────────────────────────────────────────────────
        Button cancelBtn = new Button("Cancel");
        cancelBtn.setPrefWidth(100);
        cancelBtn.setOnAction(e -> {
            result = null;
            dialog.close();
        });

        Button okBtn = new Button("OK");
        okBtn.setPrefWidth(100);
        okBtn.setDefaultButton(true);
        okBtn.setOnAction(e -> {
            result = new ArrayList<>(includedItems);
            dialog.close();
        });

        HBox bottomBar = new HBox(20, cancelBtn, okBtn);
        bottomBar.setAlignment(Pos.CENTER);
        bottomBar.setPadding(new Insets(10, 0, 0, 0));

        // ── Assemble ────────────────────────────────────────────────────────
        HBox content = new HBox(leftBox, buttonBox, rightBox);
        content.setAlignment(Pos.CENTER);

        VBox root = new VBox(10, content, bottomBar);
        root.setPadding(new Insets(15));

        dialog.setScene(new Scene(root));
        dialog.sizeToScene();
    }

    /**
     * Show the dialog and wait for the user's selection.
     *
     * @return list of included image names, or null if cancelled
     */
    public List<String> showAndWait() {
        dialog.showAndWait();
        return result;
    }
}
