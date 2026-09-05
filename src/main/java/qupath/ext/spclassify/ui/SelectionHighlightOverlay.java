package qupath.ext.spclassify.ui;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.lib.gui.viewer.overlays.AbstractOverlay;
import qupath.lib.images.ImageData;
import qupath.lib.regions.ImageRegion;
import qupath.lib.roi.interfaces.ROI;

/**
 * A lightweight QuPath custom overlay that paints a single magenta ring around
 * a target ROI. Used by Manual Label mode and Review mode to make the currently
 * selected cell easy to spot when the image is crowded with yellow detection
 * outlines.
 *
 * <p>Unlike adding a highlight {@code PathObject} to the hierarchy, updating
 * this overlay does <em>not</em> fire hierarchy-changed events, so it does not
 * trigger a full repaint of every detection in the image — only the overlay
 * layer itself is redrawn. That keeps per-click latency low on dense images
 * (50k+ detections).
 *
 * <p>Lifecycle:
 * <pre>
 *   var overlay = new SelectionHighlightOverlay();
 *   viewer.getCustomOverlayLayers().add(overlay);
 *   // … on every selection change:
 *   overlay.setTargetRoi(roi);
 *   viewer.repaint();
 *   // … on close:
 *   viewer.getCustomOverlayLayers().remove(overlay);
 * </pre>
 *
 * <p>Thread-safety: methods are intended to be called on the JavaFX
 * Application Thread. The {@code volatile} target ROI tolerates a read from
 * the QuPath repaint thread while a write happens on the FX thread.
 */
public class SelectionHighlightOverlay extends AbstractOverlay {

    /** Ring stroke width in screen pixels (scaled by downsample at paint time). */
    private static final float STROKE_WIDTH_SCREEN_PX = 2.5f;

    /** Minimum ring radius in image pixels — avoids invisibly small rings on tiny ROIs. */
    private static final double MIN_RADIUS_IMAGE_PX = 15.0;

    /** Padding multiplier — ring radius = max(width, height) * PADDING_FACTOR. */
    private static final double PADDING_FACTOR = 0.8;

    private volatile ROI targetRoi;

    public SelectionHighlightOverlay() {
        super(null); // no OverlayOptions required for a single ring
    }

    /** Update the ROI to highlight. Pass {@code null} to hide the ring. */
    public void setTargetRoi(ROI roi) {
        this.targetRoi = roi;
    }

    /** Clear the ring. */
    public void clear() {
        this.targetRoi = null;
    }

    @Override
    public void paintOverlay(
            Graphics2D g2d,
            ImageRegion imageRegion,
            double downsampleFactor,
            ImageData<BufferedImage> imageData,
            boolean paintCompletely) {
        ROI roi = targetRoi;
        if (roi == null) return;

        double cx = roi.getCentroidX();
        double cy = roi.getCentroidY();
        double r = Math.max(roi.getBoundsWidth(), roi.getBoundsHeight()) * PADDING_FACTOR;
        if (r < MIN_RADIUS_IMAGE_PX) r = MIN_RADIUS_IMAGE_PX;

        // Quick viewport cull — skip painting entirely if the ring's bounding box
        // doesn't intersect the visible region.
        if (imageRegion != null) {
            double rx = cx - r, ry = cy - r, rw = 2 * r, rh = 2 * r;
            if (rx + rw < imageRegion.getMinX()
                    || rx > imageRegion.getMaxX()
                    || ry + rh < imageRegion.getMinY()
                    || ry > imageRegion.getMaxY()) {
                return;
            }
        }

        // The Graphics2D supplied by QuPath is already in image-space coordinates,
        // so a fixed-screen-width stroke must be scaled by the downsample factor.
        Graphics2D g = (Graphics2D) g2d.create();
        try {
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            float strokeWidth = (float) (STROKE_WIDTH_SCREEN_PX * downsampleFactor);
            g.setStroke(new BasicStroke(strokeWidth));
            g.setColor(Color.MAGENTA);
            g.drawOval((int) Math.round(cx - r), (int) Math.round(cy - r), (int) Math.round(2 * r), (int)
                    Math.round(2 * r));
        } finally {
            g.dispose();
        }
    }

    /**
     * Convenience: install this overlay on a viewer if it isn't already present.
     * Returns {@code true} if newly added.
     */
    public boolean installOn(QuPathViewer viewer) {
        if (viewer == null) return false;
        var layers = viewer.getCustomOverlayLayers();
        if (!layers.contains(this)) {
            layers.add(this);
            return true;
        }
        return false;
    }

    /** Convenience: remove this overlay from a viewer (if present) and clear the ROI. */
    public void uninstallFrom(QuPathViewer viewer) {
        if (viewer == null) return;
        viewer.getCustomOverlayLayers().remove(this);
        clear();
    }
}
