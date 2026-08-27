package qupath.ext.celltune.model;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.interfaces.ROI;

/** Unit tests for the shared annotation-name → cell filter used by both the ML and clustering workflows. */
class AnnotationCellFilterTest {

    private static final ImagePlane PLANE = ImagePlane.getDefaultPlane();

    private static PathObject detectionAt(double x, double y) {
        return PathObjects.createDetectionObject(ROIs.createRectangleROI(x, y, 2, 2, PLANE));
    }

    private static PathObject namedAnnotation(String name, double x, double y, double w, double h) {
        PathObject anno = PathObjects.createAnnotationObject(ROIs.createRectangleROI(x, y, w, h, PLANE));
        anno.setName(name);
        return anno;
    }

    // ── parseKeywords ────────────────────────────────────────────────────────────

    @Test
    void parseKeywords_blankOrNull_isEmpty() {
        assertTrue(AnnotationCellFilter.parseKeywords(null).isEmpty());
        assertTrue(AnnotationCellFilter.parseKeywords("").isEmpty());
        assertTrue(AnnotationCellFilter.parseKeywords("   ").isEmpty());
    }

    @Test
    void parseKeywords_trimsSplitsAndDropsEmpties() {
        assertEquals(List.of("Tumour"), AnnotationCellFilter.parseKeywords("  Tumour "));
        assertEquals(List.of("Tumour", "Stroma", "x"), AnnotationCellFilter.parseKeywords("Tumour, Stroma ,, x"));
    }

    // ── displayLabel ─────────────────────────────────────────────────────────────

    @Test
    void displayLabel_prefersNameThenClassThenNull() {
        assertEquals("Tumour", AnnotationCellFilter.displayLabel(namedAnnotation("Tumour", 0, 0, 5, 5)));

        PathObject classified = PathObjects.createAnnotationObject(
                ROIs.createRectangleROI(0, 0, 5, 5, PLANE), PathClass.fromString("Stroma"));
        assertEquals("Stroma", AnnotationCellFilter.displayLabel(classified));

        PathObject bare = PathObjects.createAnnotationObject(ROIs.createRectangleROI(0, 0, 5, 5, PLANE));
        assertEquals(null, AnnotationCellFilter.displayLabel(bare));
    }

    // ── matchingAnnotationRois ───────────────────────────────────────────────────

    @Test
    void matchingAnnotationRois_matchesSubstringCaseInsensitive() {
        PathObjectHierarchy h = new PathObjectHierarchy();
        h.addObject(namedAnnotation("Tumour core", 0, 0, 10, 10));
        h.addObject(namedAnnotation("Stroma", 100, 100, 10, 10));

        // Case-insensitive substring on one keyword.
        assertEquals(
                1,
                AnnotationCellFilter.matchingAnnotationRois(h, List.of("tumour"))
                        .size());
        // Multiple keywords match multiple annotations.
        assertEquals(
                2,
                AnnotationCellFilter.matchingAnnotationRois(h, List.of("tumour", "strom"))
                        .size());
        // No match.
        assertTrue(AnnotationCellFilter.matchingAnnotationRois(h, List.of("immune"))
                .isEmpty());
        // Empty keyword list = no filter target (empty result).
        assertTrue(AnnotationCellFilter.matchingAnnotationRois(h, List.of()).isEmpty());
    }

    // ── centroidInAny ────────────────────────────────────────────────────────────

    @Test
    void centroidInAny_testsCentroidInsideRoi() {
        ROI box = ROIs.createRectangleROI(0, 0, 10, 10, PLANE);
        assertTrue(AnnotationCellFilter.centroidInAny(detectionAt(4, 4), List.of(box)));
        assertFalse(AnnotationCellFilter.centroidInAny(detectionAt(50, 50), List.of(box)));
        assertFalse(AnnotationCellFilter.centroidInAny(detectionAt(4, 4), List.of()));
    }

    // ── filterByAnnotations ──────────────────────────────────────────────────────

    @Test
    void filterByAnnotations_emptyKeywords_returnsAll() {
        List<PathObject> cells = List.of(detectionAt(1, 1), detectionAt(99, 99));
        assertEquals(cells, AnnotationCellFilter.filterByAnnotations(cells, new PathObjectHierarchy(), List.of()));
    }

    @Test
    void filterByAnnotations_keepsOnlyCellsInsideMatchingAnnotations() {
        PathObjectHierarchy h = new PathObjectHierarchy();
        h.addObject(namedAnnotation("Tumour", 0, 0, 10, 10));
        PathObject inside = detectionAt(5, 5);
        PathObject outside = detectionAt(500, 500);

        List<PathObject> kept =
                AnnotationCellFilter.filterByAnnotations(List.of(inside, outside), h, List.of("Tumour"));
        assertEquals(List.of(inside), kept);
    }

    @Test
    void filterByAnnotations_noMatchingAnnotation_returnsEmpty() {
        PathObjectHierarchy h = new PathObjectHierarchy();
        h.addObject(namedAnnotation("Tumour", 0, 0, 10, 10));
        List<PathObject> kept =
                AnnotationCellFilter.filterByAnnotations(List.of(detectionAt(5, 5)), h, List.of("Immune"));
        assertTrue(kept.isEmpty());
    }
}
