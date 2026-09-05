package qupath.ext.spclassify.io;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.model.CellPrediction;
import qupath.ext.spclassify.model.PopulationSet;
import qupath.lib.projects.Project;

/**
 * Per-image {@code Pred_ALL} prediction persistence.
 * <p>
 * Predictions are stored as {@code <project>/celltune/image-predictions/<image>.json}.
 * Each file records the class-name ordering plus one entry per cell
 * (both model labels and their probability vectors).
 * <p>
 * Extracted verbatim from {@link ProjectStateManager}; the public API there delegates
 * here so existing call sites are unaffected. Shares
 * {@link ProjectStateManager#getCellTuneDir(Project)}, {@link ProjectStateManager#sanitiseFileName(String)}
 * and {@link ProjectStateManager#GSON} with its siblings in this package.
 */
final class PredictionPersistence {

    private static final Logger logger = LoggerFactory.getLogger(PredictionPersistence.class);
    private static final String IMAGE_PREDICTIONS_DIR = "image-predictions";

    private PredictionPersistence() {} // utility class

    /** Internal JSON shape for persisted per-image predictions. */
    private static class SavedPredictions {
        List<String> classNames;
        List<SavedPredictionEntry> predictions;
    }

    /** Internal JSON shape for a single persisted cell prediction. */
    private static class SavedPredictionEntry {
        String cellId;
        String model1Label;
        String model2Label;
        float[] model1Probs;
        float[] model2Probs;
    }

    /**
     * Save Pred_ALL predictions for a specific image.
     */
    static Path saveImagePredictions(Project<?> project, String imageName, PopulationSet predAll) throws IOException {
        if (predAll == null || predAll.size() == 0) {
            return null;
        }

        Path dir = ProjectStateManager.getCellTuneDir(project).resolve(IMAGE_PREDICTIONS_DIR);
        Files.createDirectories(dir);

        String safeFileName = ProjectStateManager.sanitiseFileName(imageName) + ".json";
        Path outPath = dir.resolve(safeFileName);

        SavedPredictions state = new SavedPredictions();
        state.predictions = new ArrayList<>(predAll.size());

        List<String> classNames = null;
        for (CellPrediction pred : predAll.getAll().values()) {
            if (classNames == null || classNames.isEmpty()) {
                classNames = pred.getClassNames();
            }

            SavedPredictionEntry row = new SavedPredictionEntry();
            row.cellId = pred.getCellId();
            row.model1Label = pred.getModel1Label();
            row.model2Label = pred.getModel2Label();
            row.model1Probs = pred.getModel1Probs();
            row.model2Probs = pred.getModel2Probs();
            state.predictions.add(row);
        }

        if (classNames == null || classNames.isEmpty() || state.predictions.isEmpty()) {
            return null;
        }

        state.classNames = List.copyOf(classNames);
        Files.writeString(outPath, ProjectStateManager.GSON.toJson(state), StandardCharsets.UTF_8);
        logger.info("Saved {} predictions for image '{}' to {}", state.predictions.size(), imageName, outPath);
        return outPath;
    }

    /**
     * Load Pred_ALL predictions previously saved for a specific image.
     */
    static PopulationSet loadImagePredictions(Project<?> project, String imageName) throws IOException {
        Path dir = ProjectStateManager.getCellTuneDir(project).resolve(IMAGE_PREDICTIONS_DIR);
        String safeFileName = ProjectStateManager.sanitiseFileName(imageName) + ".json";
        Path filePath = dir.resolve(safeFileName);

        if (!Files.exists(filePath)) {
            return null;
        }

        SavedPredictions saved = ProjectStateManager.GSON.fromJson(
                Files.readString(filePath, StandardCharsets.UTF_8), SavedPredictions.class);
        if (saved == null
                || saved.classNames == null
                || saved.classNames.isEmpty()
                || saved.predictions == null
                || saved.predictions.isEmpty()) {
            return null;
        }

        PopulationSet predAll = new PopulationSet("Pred_ALL");
        for (SavedPredictionEntry row : saved.predictions) {
            if (row == null
                    || row.cellId == null
                    || row.cellId.isBlank()
                    || row.model1Label == null
                    || row.model2Label == null
                    || row.model1Probs == null
                    || row.model2Probs == null) {
                continue;
            }

            if (row.model1Probs.length != saved.classNames.size()
                    || row.model2Probs.length != saved.classNames.size()) {
                continue;
            }

            CellPrediction pred = new CellPrediction(
                    row.cellId, row.model1Label, row.model2Label, row.model1Probs, row.model2Probs, saved.classNames);
            predAll.put(row.cellId, pred);
        }

        if (predAll.size() == 0) {
            return null;
        }

        logger.info("Loaded {} predictions for image '{}' from {}", predAll.size(), imageName, filePath);
        return predAll;
    }

    /**
     * List the names of all images in the project that have saved prediction files.
     */
    static List<String> listImagesWithPredictions(Project<?> project) throws IOException {
        Path dir = ProjectStateManager.getCellTuneDir(project).resolve(IMAGE_PREDICTIONS_DIR);
        if (!Files.exists(dir)) return List.of();
        List<String> result = new ArrayList<>();
        for (var entry : project.getImageList()) {
            String name = entry.getImageName();
            if (Files.exists(dir.resolve(ProjectStateManager.sanitiseFileName(name) + ".json"))) {
                result.add(name);
            }
        }
        return result;
    }
}
