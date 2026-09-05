package qupath.ext.spclassify.ui;

import java.util.*;
import javafx.geometry.Insets;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import qupath.ext.spclassify.model.PopulationSet;
import qupath.lib.objects.classes.PathClass;

/**
 * A panel that lists all four {@link PopulationSet}s with per-class cell
 * counts and colour swatches matching QuPath overlay colours.
 * <p>
 * Each population set is displayed in a titled pane with a table of
 * class names, counts, and disagreement info.
 */
public class PopulationPanel extends VBox {

    private final List<PopulationSet> populationSets = new ArrayList<>();
    private final VBox content = new VBox(6);

    public PopulationPanel() {
        super(8);
        setPadding(new Insets(8));

        Label header = new Label("Population Sets");
        header.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");
        getChildren().addAll(header, content);
    }

    /**
     * Update the panel with new population sets from the classifier.
     *
     * @param predMDL1   XGBoost predictions
     * @param predMDL2   LightGBM predictions
     * @param predAVG    averaged predictions
     * @param predALL    combined predictions
     * @param classNames ordered class names
     */
    public void update(
            PopulationSet predMDL1,
            PopulationSet predMDL2,
            PopulationSet predAVG,
            PopulationSet predALL,
            List<String> classNames) {
        populationSets.clear();
        content.getChildren().clear();

        if (predMDL1 != null) populationSets.add(predMDL1);
        if (predMDL2 != null) populationSets.add(predMDL2);
        if (predAVG != null) populationSets.add(predAVG);
        if (predALL != null) populationSets.add(predALL);

        for (PopulationSet ps : populationSets) {
            content.getChildren().add(buildPopulationPane(ps, classNames));
        }
    }

    /** Clear the panel. */
    public void clear() {
        populationSets.clear();
        content.getChildren().clear();
    }

    private TitledPane buildPopulationPane(PopulationSet ps, List<String> classNames) {
        // Get counts based on the population set type
        Map<String, Long> counts = ps.getAvgCounts();

        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(4);
        grid.setPadding(new Insets(4));

        // Header row
        grid.add(new Label(""), 0, 0);
        Label nameHeader = new Label("Class");
        nameHeader.setStyle("-fx-font-weight: bold;");
        grid.add(nameHeader, 1, 0);
        Label countHeader = new Label("Count");
        countHeader.setStyle("-fx-font-weight: bold;");
        grid.add(countHeader, 2, 0);

        int row = 1;
        long total = 0;
        for (String cls : classNames) {
            // Colour swatch from QuPath PathClass
            Rectangle swatch = new Rectangle(12, 12);
            try {
                PathClass pc = PathClass.fromString(cls);
                Integer rgb = pc.getColor();
                if (rgb != null) {
                    int r = (rgb >> 16) & 0xFF;
                    int g = (rgb >> 8) & 0xFF;
                    int b = rgb & 0xFF;
                    swatch.setFill(Color.rgb(r, g, b));
                } else {
                    swatch.setFill(Color.GRAY);
                }
            } catch (Exception e) {
                swatch.setFill(Color.GRAY);
            }
            swatch.setStroke(Color.DARKGRAY);
            swatch.setStrokeWidth(0.5);

            long count = counts.getOrDefault(cls, 0L);
            total += count;

            grid.add(swatch, 0, row);
            grid.add(new Label(cls), 1, row);
            grid.add(new Label(String.valueOf(count)), 2, row);
            row++;
        }

        // Summary row
        Separator sep = new Separator();
        grid.add(sep, 0, row, 3, 1);
        row++;

        grid.add(new Label(""), 0, row);
        Label totalLabel = new Label("Total");
        totalLabel.setStyle("-fx-font-weight: bold;");
        grid.add(totalLabel, 1, row);
        grid.add(new Label(String.valueOf(total)), 2, row);
        row++;

        long disagree = ps.getDisagreementCount();
        grid.add(new Label(""), 0, row);
        Label disagreeLabel = new Label("Disagreements");
        disagreeLabel.setStyle("-fx-text-fill: #c0392b;");
        grid.add(disagreeLabel, 1, row);
        Label disagreeCount = new Label(String.valueOf(disagree));
        disagreeCount.setStyle("-fx-text-fill: #c0392b;");
        grid.add(disagreeCount, 2, row);

        String title = ps.getName() + " (" + total + " cells)";
        TitledPane tp = new TitledPane(title, grid);
        // Only expand Pred_ALL by default
        tp.setExpanded("Pred_ALL".equals(ps.getName()));
        return tp;
    }
}
