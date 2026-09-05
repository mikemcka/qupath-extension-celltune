package qupath.ext.spclassify;

import java.awt.image.BufferedImage;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import qupath.ext.spclassify.io.ProjectStateManager;
import qupath.ext.spclassify.model.LabelStore;
import qupath.ext.spclassify.model.PopulationSet;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.projects.ProjectImageEntry;

/**
 * Pure-ish review-mode sampling helpers lifted out of {@code SpClassifyExtension}.
 * <p>
 * Builds the pool of disagreement predictions to sample from (the current image's
 * predictions plus, optionally, every other project image's saved predictions) and the
 * set of already-reviewed cell IDs to exclude. The FX review orchestration and the
 * {@code lastSampled*} session state stay in the extension because they are woven into the
 * image-change listener and the docked panel; everything here takes its inputs as
 * parameters and holds no session state.
 */
final class ReviewSampling {

    private ReviewSampling() {} // utility class

    /** Pooled predictions to sample from, plus the cell-id → image-name map. */
    record SamplingContext(PopulationSet predictions, Map<String, String> cellToImage) {}

    /**
     * Build the sampling pool: the current image's predictions (in memory if present, else the
     * saved file) and — unless {@code currentImageOnly} — every other project image's saved
     * predictions, de-duplicated by cell id.
     *
     * @param currentPredAll   the open image's in-memory predictions (may be null)
     * @param currentImageOnly restrict pooling to the current image
     */
    static SamplingContext buildSamplingContext(
            QuPathGUI qupath, PopulationSet currentPredAll, boolean currentImageOnly) {
        PopulationSet pooled = new PopulationSet("Pred_ALL");
        Map<String, String> cellToImage = new LinkedHashMap<>();

        var project = qupath.getProject();
        var imageData = qupath.getImageData();
        String currentImageName = null;
        if (project != null && imageData != null) {
            var entry = project.getEntry(imageData);
            if (entry != null) {
                currentImageName = entry.getImageName();
            }
        }

        if (currentPredAll != null && currentPredAll.size() > 0) {
            addPredictionsToSamplingPool(pooled, currentPredAll, currentImageName, cellToImage);
        } else if (project != null && currentImageName != null) {
            try {
                var loadedCurrent = ProjectStateManager.loadImagePredictions(project, currentImageName);
                if (loadedCurrent != null && loadedCurrent.size() > 0) {
                    addPredictionsToSamplingPool(pooled, loadedCurrent, currentImageName, cellToImage);
                }
            } catch (Exception ignored) {
            }
        }

        if (project != null && !currentImageOnly) {
            @SuppressWarnings("unchecked")
            var entries = (List<ProjectImageEntry<BufferedImage>>) (List<?>) project.getImageList();
            for (var entry : entries) {
                if (entry == null || entry.getImageName() == null) continue;
                if (currentImageName != null && currentImageName.equals(entry.getImageName())) continue;

                try {
                    var loaded = ProjectStateManager.loadImagePredictions(project, entry.getImageName());
                    if (loaded != null && loaded.size() > 0) {
                        addPredictionsToSamplingPool(pooled, loaded, entry.getImageName(), cellToImage);
                    }
                } catch (Exception ignored) {
                }
            }
        }

        return new SamplingContext(pooled, cellToImage);
    }

    /**
     * Add a source image's predictions into {@code pooled}, skipping blank ids and any cell
     * already pooled (first image wins), and record each cell's originating image name.
     */
    static void addPredictionsToSamplingPool(
            PopulationSet pooled, PopulationSet source, String imageName, Map<String, String> cellToImage) {
        if (source == null || source.size() == 0) return;
        String safeImageName = (imageName == null || imageName.isBlank()) ? "image" : imageName;

        for (var entry : source.getAll().entrySet()) {
            String cellId = entry.getKey();
            if (cellId == null || cellId.isBlank()) continue;
            if (pooled.get(cellId) != null) continue;

            pooled.put(cellId, entry.getValue());
            cellToImage.put(cellId, safeImageName);
        }
    }

    /**
     * The set of cell IDs already labelled/reviewed (and therefore excluded from new samples):
     * the current in-memory labels, the previously-sampled ids, and every project image's saved
     * labels for the given scope (binary marker, or null for multi-class).
     */
    static Set<String> buildReviewedCellIdsForSampling(
            QuPathGUI qupath, String scope, LabelStore labels, List<String> previouslySampledIds) {
        Set<String> reviewed = new LinkedHashSet<>();
        if (labels != null) {
            reviewed.addAll(labels.getAllLabels().keySet());
        }
        if (previouslySampledIds != null) {
            reviewed.addAll(previouslySampledIds);
        }

        if (qupath != null && qupath.getProject() != null) {
            @SuppressWarnings("unchecked")
            var entries = (List<ProjectImageEntry<BufferedImage>>)
                    (List<?>) qupath.getProject().getImageList();
            for (var entry : entries) {
                if (entry == null || entry.getImageName() == null) continue;
                try {
                    LabelStore imageLabels =
                            ProjectStateManager.loadImageLabels(qupath.getProject(), scope, entry.getImageName());
                    if (imageLabels != null) {
                        reviewed.addAll(imageLabels.getAllLabels().keySet());
                    }
                } catch (Exception ignored) {
                }
            }
        }

        return reviewed;
    }
}
