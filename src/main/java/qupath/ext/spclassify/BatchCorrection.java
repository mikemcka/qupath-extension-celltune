package qupath.ext.spclassify;

import qupath.ext.spclassify.io.ProjectStateManager;
import qupath.ext.spclassify.model.BatchShifts;
import qupath.ext.spclassify.model.CellFeatureExtractor;
import qupath.lib.projects.Project;

/**
 * Small facade for streaming UniFORM batch correction into feature extraction. When the
 * {@link SpClassifyExtension#useBatchCorrection() toggle} is on and a project has persisted
 * scales ({@code batch-shifts.json}), callers load the {@link BatchShifts} once and apply a
 * per-image gain to each {@link CellFeatureExtractor} they build inside their per-image loops
 * — the extractor multiplies each measurement by its image's per-channel scale before any
 * transform. No corrected columns are written; correction is entirely in-memory and re-fittable.
 */
public final class BatchCorrection {

    private BatchCorrection() {}

    /** The project's persisted scales if the toggle is on and a fit exists, else {@code null}. */
    public static BatchShifts loadIfEnabled(Project<?> project) {
        if (project == null || !SpClassifyExtension.useBatchCorrection()) {
            return null;
        }
        BatchShifts shifts = ProjectStateManager.loadBatchShifts(project);
        return (shifts == null || shifts.isEmpty()) ? null : shifts;
    }

    /**
     * Set the per-image gain on {@code extractor} from {@code shifts} for {@code imageName}.
     * No-op when {@code shifts} is {@code null} (toggle off / no fit) or arguments are missing,
     * so callers can wire it unconditionally.
     */
    public static void applyTo(CellFeatureExtractor extractor, BatchShifts shifts, String imageName) {
        if (extractor == null || shifts == null || imageName == null) {
            return;
        }
        extractor.setBatchScale(shifts.scaleArray(imageName, extractor.getFeatureNames()));
    }
}
