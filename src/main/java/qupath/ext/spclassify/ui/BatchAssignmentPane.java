package qupath.ext.spclassify.ui;

import java.io.File;
import java.nio.file.Files;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import javafx.collections.FXCollections;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.SelectionMode;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextInputDialog;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.control.cell.TextFieldTableCell;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.Window;
import qupath.fx.dialogs.FileChoosers;

/**
 * Modal editor for assigning images to batches — a friendlier alternative to a regex.
 * Every image is a row with an editable <b>Batch</b> cell; the user can double-click to
 * type a label, multi-select rows and "Assign selected → batch…", or use the two helpers
 * ("Auto-detect from name" and "Load CSV…") to fill labels in bulk and then tweak by hand.
 * Returns the {@code image → batch} map, or {@code null} if cancelled.
 */
public class BatchAssignmentPane {

    /** Default pattern used by "Auto-detect from name" (hidden from the user). */
    private static final Pattern AUTO_PATTERN = Pattern.compile("(?i)batch\\s*\\d+");

    private final Window owner;
    private final List<Row> rows;

    /** A table row: an image and its (editable) batch label. */
    public static final class Row {
        private final String image;
        private final StringProperty batch;

        Row(String image, String batch) {
            this.image = image;
            this.batch = new SimpleStringProperty(batch == null ? "" : batch);
        }

        public String getImage() {
            return image;
        }

        public String getBatch() {
            return batch.get();
        }

        public void setBatch(String b) {
            batch.set(b == null ? "" : b);
        }

        public StringProperty batchProperty() {
            return batch;
        }
    }

    /**
     * @param owner   dialog owner window
     * @param images  the cohort image names, in display order
     * @param initial current image→batch assignments (may be empty); missing images start blank
     */
    public BatchAssignmentPane(Window owner, List<String> images, Map<String, String> initial) {
        this.owner = owner;
        this.rows = new java.util.ArrayList<>();
        for (String img : images) {
            rows.add(new Row(img, initial != null ? initial.get(img) : null));
        }
    }

    /** Show modally; returns the resolved image→batch map, or {@code null} if cancelled. */
    public Map<String, String> showAndWait() {
        TableView<Row> table = new TableView<>(FXCollections.observableArrayList(rows));
        table.setEditable(true);
        table.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE);
        table.setColumnResizePolicy(TableView.CONSTRAINED_RESIZE_POLICY);

        TableColumn<Row, String> imgCol = new TableColumn<>("Image");
        imgCol.setCellValueFactory(new PropertyValueFactory<>("image"));
        imgCol.setEditable(false);
        imgCol.setPrefWidth(360);

        TableColumn<Row, String> batchCol = new TableColumn<>("Batch");
        batchCol.setCellValueFactory(c -> c.getValue().batchProperty());
        batchCol.setCellFactory(TextFieldTableCell.forTableColumn());
        batchCol.setOnEditCommit(e -> e.getRowValue().setBatch(e.getNewValue()));
        batchCol.setEditable(true);
        batchCol.setPrefWidth(180);

        table.getColumns().add(imgCol);
        table.getColumns().add(batchCol);
        VBox.setVgrow(table, Priority.ALWAYS);

        Button assignBtn = new Button("Assign selected → batch…");
        assignBtn.setOnAction(e -> {
            var selected = table.getSelectionModel().getSelectedItems();
            if (selected.isEmpty()) {
                return;
            }
            TextInputDialog dlg = new TextInputDialog(firstNonEmptyBatch(selected));
            dlg.initOwner(owner);
            dlg.setHeaderText("Batch label for " + selected.size() + " selected image(s):");
            dlg.setTitle("Assign to batch");
            dlg.showAndWait().ifPresent(name -> {
                for (Row r : selected) {
                    r.setBatch(name.trim());
                }
                table.refresh();
            });
        });
        Button autoBtn = new Button("Auto-detect from name");
        autoBtn.setOnAction(e -> {
            for (Row r : rows) {
                Matcher m = AUTO_PATTERN.matcher(r.getImage());
                if (m.find()) {
                    r.setBatch(m.group());
                }
            }
            table.refresh();
        });
        Button csvBtn = new Button("Load CSV…");
        csvBtn.setOnAction(e -> loadCsv(table));

        Label hint = new Label("Double-click a Batch cell to edit, or select rows (Shift/Ctrl-click) and "
                + "\"Assign selected → batch…\". Images left blank are grouped as \"(unassigned)\".");
        hint.setWrapText(true);
        hint.setStyle("-fx-font-size: 11px; -fx-text-fill: #666;");

        final Map<String, String> result = new LinkedHashMap<>();
        final boolean[] ok = {false};
        Button okBtn = new Button("OK");
        okBtn.setDefaultButton(true);
        Button cancelBtn = new Button("Cancel");
        Stage stage = new Stage();
        okBtn.setOnAction(e -> {
            for (Row r : rows) {
                String b = r.getBatch() == null || r.getBatch().isBlank()
                        ? "(unassigned)"
                        : r.getBatch().trim();
                result.put(r.getImage(), b);
            }
            ok[0] = true;
            stage.close();
        });
        cancelBtn.setOnAction(e -> stage.close());

        HBox helpers = new HBox(8, assignBtn, autoBtn, csvBtn);
        helpers.setAlignment(Pos.CENTER_LEFT);
        HBox actions = new HBox(10, okBtn, cancelBtn);
        actions.setAlignment(Pos.CENTER_RIGHT);
        VBox root = new VBox(10, hint, helpers, table, actions);
        root.setPadding(new Insets(14));

        stage.setTitle("Assign Images to Batches");
        if (owner != null) {
            stage.initOwner(owner);
        }
        stage.initModality(Modality.APPLICATION_MODAL);
        stage.setScene(new Scene(root, 620, 560));
        stage.showAndWait();
        return ok[0] ? result : null;
    }

    private void loadCsv(TableView<Row> table) {
        File f = FileChoosers.promptForFile(
                "Select an image,batch CSV", FileChoosers.createExtensionFilter("CSV", "*.csv"));
        if (f == null) {
            return;
        }
        try {
            Map<String, String> map = parseBatchCsv(Files.readAllLines(f.toPath()));
            for (Row r : rows) {
                String b = map.get(r.getImage());
                if (b == null) {
                    b = map.get(stripExt(r.getImage()));
                }
                if (b != null) {
                    r.setBatch(b);
                }
            }
            table.refresh();
        } catch (Exception ex) {
            qupath.fx.dialogs.Dialogs.showErrorNotification("SP Classify", "Could not read CSV: " + ex.getMessage());
        }
    }

    private static String firstNonEmptyBatch(List<Row> rows) {
        for (Row r : rows) {
            if (r.getBatch() != null && !r.getBatch().isBlank()) {
                return r.getBatch();
            }
        }
        return "";
    }

    /** Parse an {@code image,batch} CSV: columns named image/batch (case-insensitive) or the first two columns. */
    static Map<String, String> parseBatchCsv(List<String> lines) {
        Map<String, String> map = new LinkedHashMap<>();
        if (lines.isEmpty()) {
            return map;
        }
        String[] header = lines.get(0).split(",", -1);
        int imgCol = 0;
        int batchCol = 1;
        for (int i = 0; i < header.length; i++) {
            String h = header[i].trim().toLowerCase(Locale.US);
            if (h.equals("image") || h.equals("image_name") || h.equals("imagename")) {
                imgCol = i;
            } else if (h.equals("batch") || h.equals("group")) {
                batchCol = i;
            }
        }
        for (int r = 1; r < lines.size(); r++) {
            String[] c = lines.get(r).split(",", -1);
            if (c.length > Math.max(imgCol, batchCol)) {
                String img = c[imgCol].trim();
                String batch = c[batchCol].trim();
                if (!img.isEmpty()) {
                    map.put(img, batch);
                }
            }
        }
        return map;
    }

    private static String stripExt(String name) {
        int dot = name.indexOf('.');
        return dot > 0 ? name.substring(0, dot) : name;
    }
}
