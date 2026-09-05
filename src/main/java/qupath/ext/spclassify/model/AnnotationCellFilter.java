package qupath.ext.spclassify.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.roi.interfaces.ROI;

/**
 * Shared, JavaFX-free helper for restricting a cell population to the cells whose
 * centroid falls inside one or more <em>named</em> annotations.
 * <p>
 * The matching semantics are the single source of truth for both the ML/training
 * workflow ({@code ui.ClassificationPanel}) and the clustering workflow
 * ({@code ui.ScatterPlotView} + {@link CohortClusterModel}):
 * <ul>
 *   <li>An annotation matches a keyword when its <em>display label</em> (explicit
 *       name, else its {@code PathClass} name) contains the keyword as a
 *       case-insensitive substring.</li>
 *   <li>A cell matches when its ROI centroid lies inside any matching
 *       annotation's ROI (a geometric test — not hierarchy parent/child).</li>
 * </ul>
 * An empty keyword list means "no filter" — callers keep every cell.
 */
public final class AnnotationCellFilter {

    private AnnotationCellFilter() {}

    /** Split a comma-separated keyword string into trimmed, non-empty keywords (never null). */
    public static List<String> parseKeywords(String csv) {
        List<String> out = new ArrayList<>();
        if (csv == null || csv.isBlank()) {
            return out;
        }
        for (String part : csv.split(",")) {
            String k = part.trim();
            if (!k.isEmpty()) {
                out.add(k);
            }
        }
        return out;
    }

    /** Display label for an annotation: explicit name, else PathClass name, else null. */
    public static String displayLabel(PathObject anno) {
        String name = anno.getName();
        if (name != null && !name.isBlank()) {
            return name;
        }
        PathClass pc = anno.getPathClass();
        if (pc != null) {
            String pcName = pc.getName();
            if (pcName != null && !pcName.isBlank()) {
                return pcName;
            }
        }
        return null;
    }

    /**
     * ROIs of all annotations whose display label contains any of the keywords
     * (case-insensitive substring). Returns an empty list for a null/empty
     * hierarchy or keyword list — an <em>active</em> filter with no matching ROIs
     * means "no eligible cells", which callers handle explicitly.
     */
    public static List<ROI> matchingAnnotationRois(PathObjectHierarchy hierarchy, List<String> keywords) {
        List<ROI> rois = new ArrayList<>();
        if (hierarchy == null || keywords == null || keywords.isEmpty()) {
            return rois;
        }
        List<String> lower = new ArrayList<>(keywords.size());
        for (String k : keywords) {
            lower.add(k.toLowerCase(Locale.ROOT));
        }
        for (PathObject anno : hierarchy.getAnnotationObjects()) {
            ROI roi = anno.getROI();
            if (roi == null) {
                continue;
            }
            String label = displayLabel(anno);
            if (label == null) {
                continue;
            }
            String ll = label.toLowerCase(Locale.ROOT);
            for (String kw : lower) {
                if (ll.contains(kw)) {
                    rois.add(roi);
                    break;
                }
            }
        }
        return rois;
    }

    /** True when the cell's ROI centroid lies inside any of the given ROIs. */
    public static boolean centroidInAny(PathObject cell, List<ROI> rois) {
        ROI cr = cell.getROI();
        if (cr == null) {
            return false;
        }
        double cx = cr.getCentroidX();
        double cy = cr.getCentroidY();
        for (ROI r : rois) {
            if (r.contains(cx, cy)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Keep only the cells whose centroid lies inside an annotation matching one of
     * the keywords. An empty keyword list returns the input list unchanged
     * (no filter); an active filter that matches no annotation returns an empty
     * list (no eligible cells).
     */
    public static List<PathObject> filterByAnnotations(
            List<PathObject> cells, PathObjectHierarchy hierarchy, List<String> keywords) {
        if (keywords == null || keywords.isEmpty()) {
            return cells;
        }
        List<ROI> rois = matchingAnnotationRois(hierarchy, keywords);
        if (rois.isEmpty()) {
            return List.of();
        }
        List<PathObject> out = new ArrayList<>();
        for (PathObject cell : cells) {
            if (centroidInAny(cell, rois)) {
                out.add(cell);
            }
        }
        return out;
    }
}
