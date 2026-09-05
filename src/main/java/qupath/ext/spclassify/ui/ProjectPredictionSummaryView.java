package qupath.ext.spclassify.ui;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import javafx.beans.property.SimpleObjectProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.collections.transformation.FilteredList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.TableCell;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextArea;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import javafx.stage.Modality;
import javafx.stage.Stage;
import qupath.ext.spclassify.io.ProjectSummaryCsvExporter;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.projects.ProjectImageEntry;

/**
 * Dialog that summarizes per-image prediction agreement/disagreement counts
 * and anomaly analytics from saved Pred_ALL results.
 */
public class ProjectPredictionSummaryView {

    /** Immutable table row model. */
    public record Row(
            String imageName,
            long predictedCells,
            long agreements,
            long disagreements,
            String agreementRate,
            double anomalyScore,
            boolean flagged,
            String flagReasons,
            String rareEnrichmentText,
            List<String> highlightedRareClasses,
            String classCountsText) {

        public Row {
            imageName = imageName == null ? "" : imageName;
            agreementRate = agreementRate == null ? "-" : agreementRate;
            flagReasons = flagReasons == null ? "-" : flagReasons;
            rareEnrichmentText = rareEnrichmentText == null ? "No highlighted rare classes." : rareEnrichmentText;
            highlightedRareClasses = highlightedRareClasses == null ? List.of() : List.copyOf(highlightedRareClasses);
            classCountsText = classCountsText == null ? "No predicted classes." : classCountsText;
        }
    }

    private static final String TITLE = "Project Prediction Summary";
    private static final String PRESET_STRICT = "strict";
    private static final String PRESET_BALANCED = "balanced";
    private static final String PRESET_SENSITIVE = "sensitive";
    private static final String ALL_CLASSES = "All classes";

    private final Stage stage;
    private final QuPathGUI qupath;
    private final List<Row> rows;
    private final TableView<Row> table;
    private final FilteredList<Row> filteredRows;
    private final Label summaryLabel;
    private final TextArea detailsArea;
    private final CheckBox flaggedOnlyBox;
    private final ComboBox<String> targetClassBox;
    private final ComboBox<String> presetBox;

    public ProjectPredictionSummaryView(QuPathGUI qupath, Stage owner, List<Row> rows) {
        this.qupath = qupath;
        this.rows = rows == null ? List.of() : List.copyOf(rows);

        ObservableList<Row> backingRows = FXCollections.observableArrayList(this.rows);
        filteredRows = new FilteredList<>(backingRows, row -> true);
        table = new TableView<>(filteredRows);

        TableColumn<Row, String> imageCol = new TableColumn<>("Image");
        imageCol.setCellValueFactory(
                cd -> new SimpleStringProperty(cd.getValue().imageName()));
        imageCol.setMinWidth(240);

        TableColumn<Row, Number> predictedCol = new TableColumn<>("Predicted");
        predictedCol.setCellValueFactory(
                cd -> new SimpleObjectProperty<>(cd.getValue().predictedCells()));
        predictedCol.setStyle("-fx-alignment: CENTER-RIGHT;");

        TableColumn<Row, Number> agreementsCol = new TableColumn<>("Agreements");
        agreementsCol.setCellValueFactory(
                cd -> new SimpleObjectProperty<>(cd.getValue().agreements()));
        agreementsCol.setStyle("-fx-alignment: CENTER-RIGHT;");

        TableColumn<Row, Number> disagreementsCol = new TableColumn<>("Disagreements");
        disagreementsCol.setCellValueFactory(
                cd -> new SimpleObjectProperty<>(cd.getValue().disagreements()));
        disagreementsCol.setStyle("-fx-alignment: CENTER-RIGHT;");

        TableColumn<Row, String> agreementRateCol = new TableColumn<>("Agreement %");
        agreementRateCol.setCellValueFactory(
                cd -> new SimpleStringProperty(cd.getValue().agreementRate()));
        agreementRateCol.setStyle("-fx-alignment: CENTER-RIGHT;");

        TableColumn<Row, Number> anomalyCol = new TableColumn<>("Anomaly");
        anomalyCol.setCellValueFactory(
                cd -> new SimpleObjectProperty<>(cd.getValue().anomalyScore()));
        anomalyCol.setStyle("-fx-alignment: CENTER-RIGHT;");
        // Value factory keeps a raw Number so sorting stays numeric; format only the display.
        anomalyCol.setCellFactory(col -> new TableCell<>() {
            @Override
            protected void updateItem(Number value, boolean empty) {
                super.updateItem(value, empty);
                setText(empty || value == null ? null : String.format(Locale.ROOT, "%.3f", value.doubleValue()));
            }
        });

        TableColumn<Row, String> flaggedCol = new TableColumn<>("Flagged");
        flaggedCol.setCellValueFactory(
                cd -> new SimpleStringProperty(cd.getValue().flagged() ? "Yes" : "No"));
        flaggedCol.setStyle("-fx-alignment: CENTER;");

        table.getColumns().add(imageCol);
        table.getColumns().add(predictedCol);
        table.getColumns().add(agreementsCol);
        table.getColumns().add(disagreementsCol);
        table.getColumns().add(agreementRateCol);
        table.getColumns().add(anomalyCol);
        table.getColumns().add(flaggedCol);

        anomalyCol.setSortType(TableColumn.SortType.DESCENDING);
        disagreementsCol.setSortType(TableColumn.SortType.DESCENDING);
        table.getSortOrder().add(anomalyCol);
        table.getSortOrder().add(disagreementsCol);
        table.sort();

        summaryLabel = new Label();
        summaryLabel.setStyle("-fx-font-weight: bold;");

        detailsArea = new TextArea();
        detailsArea.setEditable(false);
        detailsArea.setWrapText(true);
        detailsArea.setPrefRowCount(7);

        table.getSelectionModel().selectedItemProperty().addListener((obs, oldRow, newRow) -> showDetailsFor(newRow));

        flaggedOnlyBox = new CheckBox("Flagged only");

        targetClassBox = new ComboBox<>();
        targetClassBox.getItems().add(ALL_CLASSES);
        targetClassBox.getItems().addAll(buildTargetClassOptions(this.rows));
        targetClassBox.getSelectionModel().select(ALL_CLASSES);

        presetBox = new ComboBox<>();
        presetBox.getItems().addAll(PRESET_STRICT, PRESET_BALANCED, PRESET_SENSITIVE);
        presetBox.getSelectionModel().select(PRESET_SENSITIVE);

        flaggedOnlyBox.selectedProperty().addListener((obs, oldVal, newVal) -> applyFilters());
        targetClassBox.valueProperty().addListener((obs, oldVal, newVal) -> applyFilters());
        presetBox.valueProperty().addListener((obs, oldVal, newVal) -> applyFilters());

        Label detailsLabel = new Label("Selected image details:");
        Label targetClassLabel = new Label("Target class:");
        Label presetLabel = new Label("Threshold preset:");

        HBox filtersRow = new HBox(10, flaggedOnlyBox, targetClassLabel, targetClassBox, presetLabel, presetBox);
        filtersRow.setAlignment(Pos.CENTER_LEFT);

        Button openImageButton = new Button("Open Selected Image");
        openImageButton.setOnAction(e -> openSelectedImage());

        Button exportCsvButton = new Button("Export CSV");
        exportCsvButton.setOnAction(e -> exportCsv());

        Button closeButton = new Button("Close");
        closeButton.setOnAction(e -> closeButton.getScene().getWindow().hide());

        HBox buttonRow = new HBox(8, openImageButton, exportCsvButton, closeButton);
        buttonRow.setAlignment(Pos.CENTER_LEFT);

        VBox root = new VBox(8, summaryLabel, filtersRow, table, detailsLabel, detailsArea, buttonRow);
        root.setPadding(new Insets(10));
        VBox.setVgrow(table, Priority.ALWAYS);

        stage = new Stage();
        stage.initOwner(owner);
        stage.initModality(Modality.NONE);
        stage.setTitle(TITLE);
        stage.setScene(new Scene(root, 1120, 620));

        applyFilters();
    }

    public void show() {
        stage.show();
    }

    private void applyFilters() {
        boolean flaggedOnly = flaggedOnlyBox.isSelected();
        String targetClass = targetClassBox.getValue();
        String preset = presetBox.getValue();

        filteredRows.setPredicate(row -> {
            if (row == null) {
                return false;
            }

            if (flaggedOnly && !row.flagged()) {
                return false;
            }

            if (targetClass != null
                    && !ALL_CLASSES.equals(targetClass)
                    && !row.highlightedRareClasses().contains(targetClass)) {
                return false;
            }

            double minScore = minScoreForPreset(preset);
            return row.flagged() || row.anomalyScore() >= minScore || row.predictedCells() == 0L;
        });

        if (filteredRows.isEmpty()) {
            table.getSelectionModel().clearSelection();
            detailsArea.setText("No rows match the current filters.");
        } else if (!filteredRows.contains(table.getSelectionModel().getSelectedItem())) {
            table.getSelectionModel().selectFirst();
        }

        updateSummaryLabel();
        table.sort();
    }

    private double minScoreForPreset(String preset) {
        if (PRESET_STRICT.equalsIgnoreCase(preset)) {
            return 1.5;
        }
        if (PRESET_SENSITIVE.equalsIgnoreCase(preset)) {
            return 0.0;
        }
        return 0.5;
    }

    private void updateSummaryLabel() {
        long withPredictions = 0L;
        long flaggedCount = 0L;
        for (Row row : filteredRows) {
            if (row.predictedCells() > 0) {
                withPredictions++;
            }
            if (row.flagged()) {
                flaggedCount++;
            }
        }

        summaryLabel.setText(String.format(
                "Displayed: %d / %d images | with predictions: %d | flagged: %d",
                filteredRows.size(), rows.size(), withPredictions, flaggedCount));
    }

    private List<String> buildTargetClassOptions(List<Row> allRows) {
        var classes = new LinkedHashSet<String>();
        for (Row row : allRows) {
            classes.addAll(row.highlightedRareClasses());
        }
        var out = new ArrayList<>(classes);
        out.sort(Comparator.naturalOrder());
        return out;
    }

    private void showDetailsFor(Row row) {
        if (row == null) {
            detailsArea.setText("Select an image to view details.");
            return;
        }

        detailsArea.setText(String.format(
                "Anomaly score: %.3f (higher means more unusual vs the project baseline)\n"
                        + "Flagged: %s\n"
                        + "Reasons: %s\n"
                        + "Rare enrichment: %s\n"
                        + "  - Rare enrichment highlights classes that are rare in the project overall\n"
                        + "    but enriched in this image above count/fold thresholds.\n\n"
                        + "Class counts:\n%s",
                row.anomalyScore(),
                row.flagged() ? "Yes" : "No",
                row.flagReasons(),
                row.rareEnrichmentText(),
                row.classCountsText()));
    }

    private void openSelectedImage() {
        Row selected = table.getSelectionModel().getSelectedItem();
        if (selected == null) {
            Dialogs.showWarningNotification(TITLE, "Select an image row first.");
            return;
        }
        if (qupath == null || qupath.getProject() == null) {
            Dialogs.showErrorMessage(TITLE, "Open a QuPath project first.");
            return;
        }

        @SuppressWarnings("unchecked")
        var entries = (List<ProjectImageEntry<BufferedImage>>)
                (List<?>) qupath.getProject().getImageList();
        var entryOpt = entries.stream()
                .filter(e -> e != null && selected.imageName().equals(e.getImageName()))
                .findFirst();

        if (entryOpt.isEmpty()) {
            Dialogs.showErrorMessage(TITLE, "Could not find image in project: " + selected.imageName());
            return;
        }

        // Suppress QuPath's "save changes?" prompt without actually writing the
        // .qpdata file — saving is slow on big images and unnecessary here, since
        // the user is just navigating between images in the summary view.
        // Must be the last hierarchy-touching op before openImageEntry, because
        // any hierarchy event re-flips ImageData.changed back to true.
        var currentImageData = qupath.getImageData();
        if (currentImageData != null) {
            try {
                currentImageData.setChanged(false);
            } catch (Exception ex) {
                // Non-fatal: if this fails, QuPath's own prompt will catch unsaved work.
            }
        }

        boolean opened = qupath.openImageEntry(entryOpt.get());
        if (!opened) {
            Dialogs.showErrorMessage(TITLE, "QuPath could not open image: " + selected.imageName());
        }
    }

    private void exportCsv() {
        if (filteredRows.isEmpty()) {
            Dialogs.showInfoNotification(TITLE, "No rows available to export.");
            return;
        }

        FileChooser chooser = new FileChooser();
        chooser.setTitle("Export Project Prediction Summary CSV");
        chooser.setInitialFileName("sp-classify_project_prediction_summary.csv");
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV Files", "*.csv"));

        File out = chooser.showSaveDialog(stage);
        if (out == null) {
            return;
        }

        var exportRows = new ArrayList<ProjectSummaryCsvExporter.Row>(filteredRows.size());
        for (Row row : filteredRows) {
            exportRows.add(new ProjectSummaryCsvExporter.Row(
                    row.imageName(),
                    row.predictedCells(),
                    row.agreements(),
                    row.disagreements(),
                    row.agreementRate(),
                    row.anomalyScore(),
                    row.flagged(),
                    row.flagReasons(),
                    row.rareEnrichmentText(),
                    row.classCountsText()));
        }

        try {
            ProjectSummaryCsvExporter.writeCsv(out.toPath(), exportRows);
            Dialogs.showInfoNotification(TITLE, "Exported CSV: " + out.getAbsolutePath());
        } catch (IOException ex) {
            Dialogs.showErrorMessage(TITLE, "Failed to export CSV: " + ex.getMessage());
        }
    }
}
