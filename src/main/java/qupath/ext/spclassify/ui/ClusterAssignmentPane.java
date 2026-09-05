package qupath.ext.spclassify.ui;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.IntFunction;
import java.util.function.Supplier;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Dialog;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Separator;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.stage.Modality;
import javafx.stage.Window;
import qupath.lib.objects.classes.PathClass;

/**
 * Shared "assign clusters to classes" dialog used by both single-image and
 * project-wide clustering. Each k-means cluster gets a row showing a colour
 * swatch, its cell count, an optional per-marker centroid heatmap (mean z-scored
 * intensity: red = high, blue = low — the cluster's phenotype fingerprint), and
 * an editable class dropdown. Returns the chosen cluster → {@link PathClass}
 * mapping, or {@code null} if cancelled.
 */
final class ClusterAssignmentPane {

    private static final String SKIP_CLASS = "— skip —";

    private ClusterAssignmentPane() {}

    /**
     * @param owner        owner window
     * @param title        dialog title
     * @param k            number of clusters
     * @param counts       cells per cluster (length {@code k})
     * @param centroids    [k][nMarkers] z-scored centroid intensities, or null to
     *                     omit the heatmap (e.g. when no fit is available)
     * @param markers           marker names (columns of {@code centroids})
     * @param classNamesSupplier supplies the current class names to seed/refresh the
     *                          editable dropdowns (re-queried by "Refresh classes")
     * @param clusterColor      cluster index → swatch colour
     * @param openClassControl  opens the Class Control dialog so users can add/delete
     *                          classes without leaving this dialog (nullable → no button)
     * @return cluster → class mapping (clusters left as "skip" are absent), or null
     */
    static Map<Integer, PathClass> show(
            Window owner,
            String title,
            int k,
            int[] counts,
            double[][] centroids,
            List<String> markers,
            Supplier<List<String>> classNamesSupplier,
            IntFunction<Color> clusterColor,
            Runnable openClassControl) {
        int nMarkers = markers.size();
        boolean heatmap = centroids != null && nMarkers > 0;
        List<String> classNames = classNamesSupplier.get();

        double maxAbs = 1e-9;
        if (heatmap) {
            for (int c = 0; c < k; c++) {
                for (int j = 0; j < nMarkers; j++) {
                    maxAbs = Math.max(maxAbs, Math.abs(centroids[c][j]));
                }
            }
        }

        GridPane grid = new GridPane();
        grid.setHgap(heatmap ? 3 : 10);
        grid.setVgap(heatmap ? 4 : 6);
        grid.setPadding(new Insets(4));

        grid.add(boldLabel("Cluster"), 0, 0);
        grid.add(boldLabel("Cells"), 1, 0);
        int classCol = 2;
        if (heatmap) {
            for (int j = 0; j < nMarkers; j++) {
                Label h = new Label(markers.get(j));
                h.setStyle("-fx-font-size: 9px;");
                h.setRotate(-90);
                var box = new HBox(h);
                box.setAlignment(Pos.BOTTOM_CENTER);
                box.setMinHeight(64);
                grid.add(box, 2 + j, 0);
            }
            classCol = 2 + nMarkers;
        }
        grid.add(boldLabel("Assign to class"), classCol, 0);

        Map<Integer, ComboBox<String>> selectors = new LinkedHashMap<>();
        int row = 0;
        for (int c = 0; c < k; c++) {
            if (counts[c] == 0) {
                continue;
            }
            row++;
            Rectangle swatch = new Rectangle(12, 12, clusterColor.apply(c));
            swatch.setStroke(Color.gray(0.4));
            HBox label = new HBox(5, swatch, new Label("Cluster " + c));
            label.setAlignment(Pos.CENTER_LEFT);
            grid.add(label, 0, row);
            grid.add(new Label(String.format("%,d", counts[c])), 1, row);

            if (heatmap) {
                for (int j = 0; j < nMarkers; j++) {
                    Rectangle cell = new Rectangle(16, 16, heatColor(centroids[c][j], maxAbs));
                    cell.setStroke(Color.gray(0.85));
                    Tooltip.install(
                            cell,
                            new Tooltip(String.format("Cluster %d · %s: z=%.2f", c, markers.get(j), centroids[c][j])));
                    grid.add(cell, 2 + j, row);
                }
            }

            ComboBox<String> combo = new ComboBox<>();
            combo.setEditable(true);
            combo.getItems().add(SKIP_CLASS);
            combo.getItems().addAll(classNames);
            combo.setValue(SKIP_CLASS);
            combo.setPrefWidth(180);
            selectors.put(c, combo);
            grid.add(combo, classCol, row);
        }

        Label legend = new Label(
                heatmap
                        ? "Heatmap = per-cluster mean marker intensity (z-scored): "
                                + "red = high, blue = low. Name each cluster from its high "
                                + "markers, or leave “" + SKIP_CLASS + "”."
                        : "Name each cluster, or leave “" + SKIP_CLASS + "” to skip it.");
        legend.setWrapText(true);
        legend.setMaxWidth(880);

        ScrollPane scroll = new ScrollPane(grid);
        scroll.setFitToWidth(false);
        scroll.setPrefViewportHeight(Math.min(420, 90 + k * 26));
        scroll.setPrefViewportWidth(heatmap ? 880 : 420);
        scroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.AS_NEEDED);

        // ── Class-management toolbar: re-queries class names into every dropdown,
        // and links to Class Control so users can add/delete classes in place. ──
        Button refreshBtn = new Button("Refresh classes");
        refreshBtn.setTooltip(new Tooltip("Re-read the QuPath class list into every dropdown — click after "
                + "adding or deleting classes in Class Control."));
        refreshBtn.setOnAction(e -> {
            List<String> latest = classNamesSupplier.get();
            for (ComboBox<String> combo : selectors.values()) {
                String cur = combo.getValue();
                combo.getItems().setAll(SKIP_CLASS);
                combo.getItems().addAll(latest);
                combo.setValue(cur); // editable combo: preserves typed/selected value
            }
        });

        HBox toolbar = new HBox(8);
        toolbar.setAlignment(Pos.CENTER_LEFT);
        if (openClassControl != null) {
            Button manageBtn = new Button("Manage Classes…");
            manageBtn.setTooltip(
                    new Tooltip("Open Class Control to add or delete classes, then click " + "“Refresh classes”."));
            manageBtn.setOnAction(e -> openClassControl.run());
            toolbar.getChildren().add(manageBtn);
        }
        Region toolbarSpacer = new Region();
        HBox.setHgrow(toolbarSpacer, Priority.ALWAYS);
        toolbar.getChildren().addAll(toolbarSpacer, refreshBtn);

        VBox content = new VBox(10, toolbar, scroll, new Separator(), legend);
        content.setPadding(new Insets(6));

        Dialog<ButtonType> dlg = new Dialog<>();
        dlg.initOwner(owner);
        // Non-modal so the (non-modal) Class Control dialog stays interactive while
        // this dialog is open; showAndWait still blocks the caller as usual.
        dlg.initModality(Modality.NONE);
        dlg.setTitle(title);
        dlg.setResizable(true);
        dlg.getDialogPane().setPrefWidth(heatmap ? 940 : 480);
        dlg.getDialogPane().setContent(content);
        dlg.getDialogPane().getButtonTypes().addAll(ButtonType.APPLY, ButtonType.CANCEL);

        Optional<ButtonType> res = dlg.showAndWait();
        if (res.isEmpty() || res.get() != ButtonType.APPLY) {
            return null;
        }

        Map<Integer, PathClass> mapping = new LinkedHashMap<>();
        for (var e : selectors.entrySet()) {
            String v = e.getValue().getValue();
            if (v == null) {
                continue;
            }
            v = v.trim();
            if (v.isEmpty() || v.equals(SKIP_CLASS)) {
                continue;
            }
            mapping.put(e.getKey(), PathClass.fromString(v));
        }
        return mapping;
    }

    private static Label boldLabel(String text) {
        Label l = new Label(text);
        l.setStyle("-fx-font-weight: bold;");
        return l;
    }

    // Perceptual stretch for the diverging ramp. maxAbs is pinned to the single largest |z| in the
    // grid — usually one strong positive — so low/moderate z-scores (especially the blue side) map
    // near white and wash out. Raising |t| to this exponent (<1) pushes those mid/low values away
    // from white toward saturation while leaving 0 and the ±maxAbs endpoints fixed.
    private static final double HEATMAP_GAMMA = 0.55;

    /** Diverging colour: blue (low) → white (0) → red (high), scaled by maxAbs with a gamma stretch. */
    private static Color heatColor(double v, double maxAbs) {
        double t = maxAbs < 1e-9 ? 0 : Math.max(-1, Math.min(1, v / maxAbs));
        double s = Math.signum(t) * Math.pow(Math.abs(t), HEATMAP_GAMMA);
        if (s >= 0) {
            return Color.color(1, 1 - s, 1 - s);
        }
        return Color.color(1 + s, 1 + s, 1);
    }
}
