package qupath.ext.spclassify.model;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Set;
import org.junit.jupiter.api.Test;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.interfaces.ROI;

/**
 * Regression tests for {@link AnnotationLabelCollector}, the unified
 * point-annotation → label collector. The critical case is merge-history
 * preservation (finding #13): a previously-merged label must NOT be overwritten
 * by a bare annotation class — the bug that existed in the old ClassificationPanel
 * copy.
 */
class AnnotationLabelCollectorTest {

    private static final ImagePlane PLANE = ImagePlane.getDefaultPlane();

    /** A detection covering the unit-ish square around (15,15). */
    private static PathObject detectionAt() {
        ROI roi = ROIs.createRectangleROI(10, 10, 20, 20, PLANE); // covers (15,15)
        return PathObjects.createDetectionObject(roi);
    }

    /** A point annotation at (15,15) with the given class. */
    private static PathObject pointAnnotation(String className) {
        ROI pt = ROIs.createPointsROI(15, 15, PLANE);
        return PathObjects.createAnnotationObject(pt, PathClass.fromString(className));
    }

    private static PathObjectHierarchy hierarchyWith(PathObject... objects) {
        PathObjectHierarchy h = new PathObjectHierarchy();
        for (PathObject o : objects) {
            h.addObject(o);
        }
        return h;
    }

    @Test
    void labelsDetectionUnderPointAnnotation() {
        PathObject det = detectionAt();
        PathObjectHierarchy h = hierarchyWith(det, pointAnnotation("Tumor"));
        LabelStore store = new LabelStore("test");

        AnnotationLabelCollector.collect(h, store, null);

        assertEquals("Tumor", store.getLabel(det.getID().toString()));
    }

    @Test
    void preservesMergeHistoryWhenAnnotationMatchesOriginalClass() {
        PathObject det = detectionAt();
        String id = det.getID().toString();
        PathObjectHierarchy h = hierarchyWith(det, pointAnnotation("Tumor"));

        LabelStore store = new LabelStore("test");
        // This cell was previously merged: Tumor folded into Immune.
        store.setLabel(id, "Tumor-mergedInto(Immune)");

        AnnotationLabelCollector.collect(h, store, null);

        // The merge-encoded value must survive — NOT be clobbered back to "Tumor".
        assertEquals("Tumor-mergedInto(Immune)", store.getLabel(id), "merge history must be preserved");
    }

    @Test
    void preservesChainedMergeHistory() {
        PathObject det = detectionAt();
        String id = det.getID().toString();
        PathObjectHierarchy h = hierarchyWith(det, pointAnnotation("Tumor"));

        LabelStore store = new LabelStore("test");
        store.setLabel(id, "Tumor-mergedInto(Immune-mergedInto(Other))");

        AnnotationLabelCollector.collect(h, store, null);

        assertEquals("Tumor-mergedInto(Immune-mergedInto(Other))", store.getLabel(id));
    }

    @Test
    void overwritesWhenExistingClassDiffers() {
        PathObject det = detectionAt();
        String id = det.getID().toString();
        PathObjectHierarchy h = hierarchyWith(det, pointAnnotation("Tumor"));

        LabelStore store = new LabelStore("test");
        store.setLabel(id, "Stroma"); // unrelated prior class → should be updated

        AnnotationLabelCollector.collect(h, store, null);

        assertEquals("Tumor", store.getLabel(id));
    }

    @Test
    void allowedClassesFilterExcludesOtherClasses() {
        PathObject det = detectionAt();
        PathObjectHierarchy h = hierarchyWith(det, pointAnnotation("Tumor"));
        LabelStore store = new LabelStore("test");

        AnnotationLabelCollector.collect(h, store, Set.of("Immune")); // Tumor not allowed

        assertNull(store.getLabel(det.getID().toString()), "annotation outside the allowed-class set must be ignored");
    }

    @Test
    void areaAnnotationsAreIgnored() {
        PathObject det = detectionAt();
        ROI area = ROIs.createRectangleROI(0, 0, 100, 100, PLANE);
        PathObject areaAnno = PathObjects.createAnnotationObject(area, PathClass.fromString("Tumor"));
        PathObjectHierarchy h = hierarchyWith(det, areaAnno);
        LabelStore store = new LabelStore("test");

        AnnotationLabelCollector.collect(h, store, null);

        assertNull(store.getLabel(det.getID().toString()), "only point annotations count as ground truth");
    }
}
