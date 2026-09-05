package qupath.ext.spclassify;

import java.util.ResourceBundle;
import javafx.beans.property.BooleanProperty;
import javafx.scene.control.Menu;
import javafx.scene.control.MenuItem;
import javafx.scene.control.SeparatorMenuItem;
import qupath.ext.spclassify.ui.CitationDialog;
import qupath.lib.gui.QuPathGUI;

/**
 * Builds the <b>Extensions &gt; SP Classify</b> menu, lifted out of {@code SpClassifyExtension}.
 * <p>
 * Pure construction: every item is wired to a handler on the extension and disabled while the
 * extension is off (bound to the shared {@code enabled} property). The item/group/separator
 * layout, labels and ordering are preserved 1:1 from the original {@code addMenuItems}.
 */
final class MenuItemFactory {

    private static final ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.spclassify.ui.strings");
    private static final String EXTENSION_NAME = resources.getString("name");

    private MenuItemFactory() {} // utility class

    /**
     * Populate the SP Classify submenu under Extensions.
     *
     * @param qupath  the QuPath GUI
     * @param ext     the extension supplying the menu action handlers
     * @param enabled the extension-enabled property; every item is disabled while it is false
     */
    static void addMenuItems(QuPathGUI qupath, SpClassifyExtension ext, BooleanProperty enabled) {
        var menu = qupath.getMenu("Extensions>" + EXTENSION_NAME, true);

        // Group the export/import actions into two submenus to keep the
        // top-level dropdown compact.
        Menu exportMenu = group(
                "menu.group.export",
                enabled,
                item(resources.getString("menu.export.short"), enabled, () -> ext.exportCellTable(qupath)),
                item(
                        resources.getString("menu.export.groundtruth.short"),
                        enabled,
                        () -> ext.exportGroundTruth(qupath)),
                item(
                        resources.getString("menu.export.binary.groundtruth.short"),
                        enabled,
                        () -> ext.exportActiveBinaryGroundTruth(qupath)));

        Menu importMenu = group(
                "menu.group.import",
                enabled,
                item(resources.getString("menu.import.markers.short"), enabled, () -> ext.importMarkerTable(qupath)),
                item(
                        resources.getString("menu.import.groundtruth.short"),
                        enabled,
                        () -> ext.importGroundTruth(qupath)),
                item(
                        resources.getString("menu.import.binary.groundtruth.short"),
                        enabled,
                        () -> ext.importActiveBinaryGroundTruth(qupath)));

        // Utility scripts: ad-hoc helpers commonly reused across projects.
        Menu utilityScriptsMenu = group(
                "menu.group.utilities",
                enabled,
                item(
                        resources.getString("menu.utility.filter.cells"),
                        enabled,
                        () -> UtilityScripts.filterCellsBySizeAndCircularity(qupath)),
                item(
                        resources.getString("menu.utility.resolve.hierarchy"),
                        enabled,
                        () -> UtilityScripts.resolveHierarchy(qupath)),
                item(
                        resources.getString("menu.utility.lock.annotations"),
                        enabled,
                        () -> UtilityScripts.lockAllAnnotations(qupath)),
                item(
                        resources.getString("menu.utility.import.geojson"),
                        enabled,
                        () -> UtilityScripts.importGeoJsonObjects(qupath)),
                item(
                        resources.getString("menu.utility.export.regions"),
                        enabled,
                        () -> AnnotationRegionExporter.exportAnnotationRegions(qupath)),
                item(
                        resources.getString("menu.utility.delete.measurements"),
                        enabled,
                        () -> UtilityScripts.deleteMeasurementsByKeyword(qupath)),
                new SeparatorMenuItem(),
                item(
                        resources.getString("menu.utility.reset.project"),
                        enabled,
                        () -> ext.showResetProjectState(qupath)));

        menu.getItems()
                .addAll(
                        item("Binary Classifiers...", enabled, () -> ext.showBinaryClassifiers(qupath)),
                        item("Composite Classification...", enabled, () -> ext.showCompositeClassification(qupath)),
                        item("Class Control...", enabled, () -> ext.showClassControl(qupath)),
                        item(resources.getString("menu.features"), enabled, () -> ext.showFeatureSelection(qupath)),
                        item("Clustering Normalisation", enabled, () -> ext.showClusteringNormalisation(qupath)),
                        new SeparatorMenuItem(),
                        item(
                                resources.getString("menu.prediction.summary"),
                                enabled,
                                () -> ext.showProjectPredictionSummary(qupath)),
                        item(
                                resources.getString("menu.intensity.heatmaps"),
                                enabled,
                                () -> ext.showIntensityHeatmaps(qupath)),
                        item(
                                resources.getString("menu.pixel.prescreen"),
                                enabled,
                                () -> ext.showImagePixelPrescreen(qupath)),
                        item("Scatter Plots and Clustering...", enabled, () -> ext.showScatterPlot(qupath)),
                        item(
                                resources.getString("menu.neighborhoods"),
                                enabled,
                                () -> ext.showCellularNeighborhoods(qupath)),
                        item(resources.getString("menu.batchnorm"), enabled, () -> ext.showBatchNormalization(qupath)),
                        item("Generate Distance Measurements...", enabled, () -> ext.showDistanceMeasurements(qupath)),
                        new SeparatorMenuItem(),
                        exportMenu,
                        importMenu,
                        new SeparatorMenuItem(),
                        utilityScriptsMenu,
                        new SeparatorMenuItem(),
                        // Always enabled: citation info stays available even when the extension is off.
                        alwaysItem("How to Cite...", () -> CitationDialog.show(qupath)));
    }

    /** A menu item that runs {@code action} and is always enabled. */
    private static MenuItem alwaysItem(String label, Runnable action) {
        MenuItem mi = new MenuItem(label);
        mi.setOnAction(e -> action.run());
        return mi;
    }

    /** A menu item that runs {@code action} and is disabled while the extension is off. */
    private static MenuItem item(String label, BooleanProperty enabled, Runnable action) {
        MenuItem mi = new MenuItem(label);
        mi.setOnAction(e -> action.run());
        mi.disableProperty().bind(enabled.not());
        return mi;
    }

    /** A submenu (disabled while the extension is off) containing the given items. */
    private static Menu group(String labelKey, BooleanProperty enabled, MenuItem... items) {
        Menu m = new Menu(resources.getString(labelKey));
        m.disableProperty().bind(enabled.not());
        m.getItems().addAll(items);
        return m;
    }
}
