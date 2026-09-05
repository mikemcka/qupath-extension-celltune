package qupath.ext.spclassify.ui;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
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
import javafx.scene.control.Label;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextArea;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import javafx.stage.Modality;
import javafx.stage.Stage;
import qupath.ext.spclassify.io.PixelStatsCsvExporter;
import qupath.ext.spclassify.model.PixelCohortReport;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.projects.ProjectImageEntry;

/**
 * Dialog summarising whole-image pixel statistics across a project: a verdict,
 * an anomaly score, and the headline per-channel summary for each image, with a
 * plain-English review of how the selected image compares to the cohort.
 * <p>
 * This is the pixel-level twin of {@link ProjectPredictionSummaryView}: it works
 * straight off the low-resolution pyramid and needs no cells, so it can be run as
 * a prescreen at the very start of a project to spot background-heavy,
 * over-exposed, or weakly-stained images.
 */
public class PixelPrescreenView {

    private static final String TITLE = "Image Pixel Prescreen";

    private final Stage stage;
    private final QuPathGUI qupath;
    private final PixelCohortReport report;
    private final TableView<PixelCohortReport.ImageReport> table;
    private final FilteredList<PixelCohortReport.ImageReport> filteredRows;
    private final Label summaryLabel;
    private final TextArea detailsArea;
    private final CheckBox flaggedOnlyBox;

    public PixelPrescreenView(QuPathGUI qupath, Stage owner, PixelCohortReport report) {
        this.qupath = qupath;
        this.report = report == null ? new PixelCohortReport(List.of()) : report;

        ObservableList<PixelCohortReport.ImageReport> backing = FXCollections.observableArrayList(this.report.images());
        filteredRows = new FilteredList<>(backing, row -> true);
        table = new TableView<>(filteredRows);

        TableColumn<PixelCohortReport.ImageReport, String> imageCol = new TableColumn<>("Image");
        imageCol.setCellValueFactory(
                cd -> new SimpleStringProperty(cd.getValue().imageName()));
        imageCol.setMinWidth(240);

        TableColumn<PixelCohortReport.ImageReport, String> verdictCol = new TableColumn<>("Verdict");
        verdictCol.setCellValueFactory(
                cd -> new SimpleStringProperty(cd.getValue().verdict()));
        verdictCol.setMinWidth(120);

        TableColumn<PixelCohortReport.ImageReport, Number> scoreCol = new TableColumn<>("Score");
        scoreCol.setCellValueFactory(
                cd -> new SimpleObjectProperty<>(cd.getValue().score()));
        scoreCol.setStyle("-fx-alignment: CENTER-RIGHT;");

        TableColumn<PixelCohortReport.ImageReport, String> fgCol = new TableColumn<>("Foreground %");
        fgCol.setCellValueFactory(
                cd -> new SimpleStringProperty(pct(cd.getValue().meanForegroundCoverage())));
        fgCol.setStyle("-fx-alignment: CENTER-RIGHT;");

        TableColumn<PixelCohortReport.ImageReport, String> emptyCol = new TableColumn<>("Empty %");
        emptyCol.setCellValueFactory(
                cd -> new SimpleStringProperty(pct(cd.getValue().emptyFraction())));
        emptyCol.setStyle("-fx-alignment: CENTER-RIGHT;");

        TableColumn<PixelCohortReport.ImageReport, String> satCol = new TableColumn<>("Max sat %");
        satCol.setCellValueFactory(
                cd -> new SimpleStringProperty(pct(cd.getValue().maxSaturationFraction())));
        satCol.setStyle("-fx-alignment: CENTER-RIGHT;");

        TableColumn<PixelCohortReport.ImageReport, String> dynCol = new TableColumn<>("Dyn. range");
        dynCol.setCellValueFactory(
                cd -> new SimpleStringProperty(num(cd.getValue().medianDynamicRange())));
        dynCol.setStyle("-fx-alignment: CENTER-RIGHT;");

        TableColumn<PixelCohortReport.ImageReport, String> focusCol = new TableColumn<>("Focus");
        focusCol.setCellValueFactory(
                cd -> new SimpleStringProperty(num(cd.getValue().maxFocus())));
        focusCol.setStyle("-fx-alignment: CENTER-RIGHT;");

        TableColumn<PixelCohortReport.ImageReport, String> intensityCol = new TableColumn<>("Intensity z");
        intensityCol.setCellValueFactory(
                cd -> new SimpleStringProperty(num(cd.getValue().maxIntensityZ())));
        intensityCol.setStyle("-fx-alignment: CENTER-RIGHT;");

        TableColumn<PixelCohortReport.ImageReport, String> flaggedCol = new TableColumn<>("Flagged");
        flaggedCol.setCellValueFactory(
                cd -> new SimpleStringProperty(cd.getValue().flagged() ? "Yes" : "No"));
        flaggedCol.setStyle("-fx-alignment: CENTER;");

        table.getColumns().add(imageCol);
        table.getColumns().add(verdictCol);
        table.getColumns().add(scoreCol);
        table.getColumns().add(fgCol);
        table.getColumns().add(emptyCol);
        table.getColumns().add(satCol);
        table.getColumns().add(dynCol);
        table.getColumns().add(focusCol);
        table.getColumns().add(intensityCol);
        table.getColumns().add(flaggedCol);

        scoreCol.setSortType(TableColumn.SortType.DESCENDING);
        table.getSortOrder().add(scoreCol);

        summaryLabel = new Label();
        summaryLabel.setStyle("-fx-font-weight: bold;");

        detailsArea = new TextArea();
        detailsArea.setEditable(false);
        detailsArea.setWrapText(true);
        detailsArea.setPrefRowCount(10);

        table.getSelectionModel().selectedItemProperty().addListener((obs, oldRow, newRow) -> showDetailsFor(newRow));

        flaggedOnlyBox = new CheckBox("Flagged only");
        flaggedOnlyBox.selectedProperty().addListener((obs, oldVal, newVal) -> applyFilters());

        HBox filtersRow = new HBox(10, flaggedOnlyBox);
        filtersRow.setAlignment(Pos.CENTER_LEFT);

        Button openImageButton = new Button("Open Selected Image");
        openImageButton.setOnAction(e -> openSelectedImage());

        Button exportCsvButton = new Button("Export CSV");
        exportCsvButton.setOnAction(e -> exportCsv());

        Button closeButton = new Button("Close");
        closeButton.setOnAction(e -> closeButton.getScene().getWindow().hide());

        HBox buttonRow = new HBox(8, openImageButton, exportCsvButton, closeButton);
        buttonRow.setAlignment(Pos.CENTER_LEFT);

        Label detailsLabel = new Label("Selected image review:");

        VBox root = new VBox(8, summaryLabel, filtersRow, table, detailsLabel, detailsArea, buttonRow);
        root.setPadding(new Insets(10));
        VBox.setVgrow(table, Priority.ALWAYS);

        stage = new Stage();
        stage.initOwner(owner);
        stage.initModality(Modality.NONE);
        stage.setTitle(TITLE);
        stage.setScene(new Scene(root, 1040, 640));

        applyFilters();
    }

    public void show() {
        stage.show();
    }

    private void applyFilters() {
        boolean flaggedOnly = flaggedOnlyBox.isSelected();
        filteredRows.setPredicate(row -> row != null && (!flaggedOnly || row.flagged()));

        if (filteredRows.isEmpty()) {
            table.getSelectionModel().clearSelection();
            detailsArea.setText("No images match the current filter.");
        } else if (!filteredRows.contains(table.getSelectionModel().getSelectedItem())) {
            table.getSelectionModel().selectFirst();
        }
        updateSummaryLabel();
        table.sort();
    }

    private void updateSummaryLabel() {
        long flaggedCount = report.images().stream()
                .filter(PixelCohortReport.ImageReport::flagged)
                .count();
        summaryLabel.setText(String.format(
                Locale.ROOT,
                "Displayed: %d / %d images | flagged: %d",
                filteredRows.size(),
                report.images().size(),
                flaggedCount));
    }

    private void showDetailsFor(PixelCohortReport.ImageReport row) {
        if (row == null) {
            detailsArea.setText("Select an image to view its review.");
            return;
        }
        StringBuilder sb = new StringBuilder();
        sb.append(row.narrative()).append("\n\n");
        sb.append("Per-channel (median | p99 | foreground% | dyn.range | sat% | focus):\n");
        for (var cc : row.channels()) {
            var s = cc.stats();
            sb.append(String.format(
                    Locale.ROOT,
                    "  %-16s %8s | %8s | %7s | %8s | %6s | %8s%n",
                    truncate(cc.channel(), 16),
                    num(s.median()),
                    num(s.p99()),
                    pct(s.foregroundCoverage()),
                    num(s.dynamicRange()),
                    pct(s.saturationFraction()),
                    num(s.laplacianVariance())));
        }
        detailsArea.setText(sb.toString());
    }

    private void openSelectedImage() {
        PixelCohortReport.ImageReport selected = table.getSelectionModel().getSelectedItem();
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

        // Suppress QuPath's "save changes?" prompt without writing — the user is
        // just navigating between images (mirrors ProjectPredictionSummaryView).
        var currentImageData = qupath.getImageData();
        if (currentImageData != null) {
            try {
                currentImageData.setChanged(false);
            } catch (Exception ex) {
                // Non-fatal.
            }
        }
        if (!qupath.openImageEntry(entryOpt.get())) {
            Dialogs.showErrorMessage(TITLE, "QuPath could not open image: " + selected.imageName());
        }
    }

    private void exportCsv() {
        if (report.images().isEmpty()) {
            Dialogs.showInfoNotification(TITLE, "No rows available to export.");
            return;
        }
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Export Image Pixel Prescreen CSV");
        chooser.setInitialFileName("sp-classify_image_pixel_prescreen.csv");
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV Files", "*.csv"));

        File out = chooser.showSaveDialog(stage);
        if (out == null) {
            return;
        }
        try {
            PixelStatsCsvExporter.writeCsv(out.toPath(), report);
            Dialogs.showInfoNotification(TITLE, "Exported CSV: " + out.getAbsolutePath());
        } catch (IOException ex) {
            Dialogs.showErrorMessage(TITLE, "Failed to export CSV: " + ex.getMessage());
        }
    }

    private static String pct(double frac) {
        return Double.isNaN(frac) ? "-" : String.format(Locale.ROOT, "%.1f%%", frac * 100.0);
    }

    private static String num(double v) {
        if (Double.isNaN(v)) {
            return "-";
        }
        return Math.abs(v) >= 1000.0 ? String.format(Locale.ROOT, "%.0f", v) : String.format(Locale.ROOT, "%.2f", v);
    }

    private static String truncate(String s, int max) {
        if (s == null) {
            return "";
        }
        return s.length() <= max ? s : s.substring(0, max - 1) + "…";
    }
}
