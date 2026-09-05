package qupath.ext.spclassify.io;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.interfaces.ROI;

class CellTableExporterTest {

    /** A single square cell with two marker-mean measurements. */
    private static PathObject squareCell() {
        // Square polygon: (10,20) (30,20) (30,40) (10,40) in pixels.
        ROI roi = ROIs.createPolygonROI(
                new double[] {10, 30, 30, 10}, new double[] {20, 20, 40, 40}, ImagePlane.getDefaultPlane());
        PathObject cell = PathObjects.createDetectionObject(roi);
        cell.getMeasurementList().put("CD8: Cell: Mean", 1.5);
        cell.getMeasurementList().put("CD4: Cell: Mean", 2.5);
        return cell;
    }

    private static String[] exportAndRead(
            Path file, List<String> features, boolean includeGeometry, boolean geometryInMicrons) throws IOException {
        CellTableExporter.export(
                file,
                List.of(squareCell()),
                List.of(),
                "img-a",
                features,
                2.0, // pixelWidthMicrons
                2.0, // pixelHeightMicrons
                includeGeometry,
                geometryInMicrons);
        return Files.readString(file, StandardCharsets.UTF_8).split("\\R");
    }

    @Test
    void exportsGeometryInMicronsWhenRequested(@TempDir Path dir) throws IOException {
        String[] lines = exportAndRead(dir.resolve("um.csv"), List.of("CD8: Cell: Mean"), true, true);

        String header = lines[0];
        assertTrue(header.contains("Geometry_um"), header);
        assertFalse(header.contains("Geometry_px"), header);

        // Micron vertices are pixel × 2.0 (e.g. first vertex 10,20 → 20.00 40.00).
        assertTrue(lines[1].contains("POLYGON ((20.00 40.00"), lines[1]);
    }

    @Test
    void exportsGeometryInPixelsWhenRequested(@TempDir Path dir) throws IOException {
        String[] lines = exportAndRead(dir.resolve("px.csv"), List.of("CD8: Cell: Mean"), true, false);

        String header = lines[0];
        assertTrue(header.contains("Geometry_px"), header);
        assertFalse(header.contains("Geometry_um"), header);

        // Pixel vertices are written unscaled (first vertex 10,20 → 10.00 20.00).
        assertTrue(lines[1].contains("POLYGON ((10.00 20.00"), lines[1]);
    }

    @Test
    void omitsGeometryColumnWhenDisabled(@TempDir Path dir) throws IOException {
        String[] lines = exportAndRead(dir.resolve("nogeom.csv"), List.of("CD8: Cell: Mean"), false, true);

        String header = lines[0];
        assertFalse(header.contains("Geometry"), header);
        assertFalse(lines[1].contains("POLYGON"), lines[1]);

        // 8 fixed columns + 1 feature column = 9 fields.
        assertEquals(9, header.split(",", -1).length, header);
    }

    @Test
    void exportsOnlySelectedFeaturesAndWritesNaForMissing(@TempDir Path dir) throws IOException {
        String[] lines =
                exportAndRead(dir.resolve("feats.csv"), List.of("CD8: Cell: Mean", "Missing: Feature"), false, true);

        String header = lines[0];
        assertTrue(header.endsWith("CD8: Cell: Mean,Missing: Feature"), header);
        // CD4 was measured but not selected — it must not appear.
        assertFalse(header.contains("CD4: Cell: Mean"), header);

        String[] fields = lines[1].split(",", -1);
        // Last two fields: selected CD8 value, then NA for the missing feature.
        assertEquals("1.5", fields[fields.length - 2]);
        assertEquals("NA", fields[fields.length - 1]);
    }

    @Test
    void exportsFullCompositeClassificationNotJustLeaf(@TempDir Path dir) throws IOException {
        ROI roi = ROIs.createPolygonROI(
                new double[] {10, 30, 30, 10}, new double[] {20, 20, 40, 40}, ImagePlane.getDefaultPlane());
        PathObject cell = PathObjects.createDetectionObject(roi);
        // Composite/derived class as produced by CompositeClassifier.
        cell.setPathClass(PathClass.fromString("CD8-:GrB-:HLA-DR-:PD-1-:TCR-"));

        Path file = dir.resolve("composite.csv");
        CellTableExporter.export(file, List.of(cell), List.of(), "img-a", List.of(), 1.0, 1.0, false, false);
        String[] lines = Files.readString(file, StandardCharsets.UTF_8).split("\\R");

        // Classification is column index 5 (Image, CellID, CentroidX_um,
        // CentroidY_um, Area_um2, Classification, ...). The full composite is
        // written, not just the leaf segment "TCR-".
        String classification = lines[1].split(",", -1)[5];
        assertEquals("CD8-: GrB-: HLA-DR-: PD-1-: TCR-", classification);
    }
}
