package qupath.ext.spclassify.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjectTools;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;

/**
 * Collects ground-truth labels from point annotations in a hierarchy into a
 * {@link LabelStore}.
 * <p>
 * Only <em>point</em> annotations count as ground truth — area/region annotations
 * describe tissue regions, not individual cell labels. Each point is resolved to
 * the detection(s) it falls within, which are labelled with the annotation's
 * {@code PathClass}.
 * <p>
 * <b>Merge-history preservation.</b> If a detection already carries a label whose
 * innermost original class equals the annotation's class — i.e. the stored value
 * is {@code "<cls>-mergedInto(target)"} (or a chained merge) — the existing value
 * is left untouched. The annotation has no knowledge of the merge, so overwriting
 * it with the bare {@code PathClass} would silently destroy the merge result.
 * This single shared implementation replaces two previously-divergent copies (one
 * in {@code SpClassifyExtension}, one in {@code ClassificationPanel}); the panel
 * copy lacked this guard, which could overwrite merged labels during training-time
 * label collection.
 */
public final class AnnotationLabelCollector {

    private AnnotationLabelCollector() {} // utility class

    /**
     * Collect point-annotation labels into {@code store}.
     *
     * @param hierarchy      the image hierarchy to scan
     * @param store          the label store to populate
     * @param allowedClasses if non-null, only annotations whose class is in this
     *                       set are collected (used to restrict to a binary
     *                       classifier's classes); pass {@code null} for no filter
     */
    public static void collect(PathObjectHierarchy hierarchy, LabelStore store, Set<String> allowedClasses) {
        for (PathObject anno : hierarchy.getAnnotationObjects()) {
            if (anno.getPathClass() == null || anno.getROI() == null) continue;
            // Only point annotations count as ground truth — area/region annotations
            // describe tissue regions, not individual cell labels.
            if (!anno.getROI().isPoint()) continue;
            String cls = anno.getPathClass().toString();
            if (allowedClasses != null && !allowedClasses.contains(cls)) continue;

            List<PathObject> hits = new ArrayList<>();
            for (var pt : anno.getROI().getAllPoints()) {
                hits.addAll(PathObjectTools.getObjectsForLocation(
                        hierarchy,
                        pt.getX(),
                        pt.getY(),
                        anno.getROI().getZ(),
                        anno.getROI().getT(),
                        -1));
            }

            for (PathObject det : hits) {
                if (det.isDetection()) {
                    String id = det.getID().toString();
                    String existing = store.getLabel(id);
                    // Preserve merge-history encoding: if the existing stored value is
                    // "<cls>-mergedInto(target)" (or a chained merge whose innermost
                    // original equals cls), don't overwrite it with the bare PathClass.
                    if (existing != null && cls.equals(LabelStore.innermostOriginal(existing))) {
                        continue;
                    }
                    store.setLabel(id, cls);
                }
            }
        }
    }
}
