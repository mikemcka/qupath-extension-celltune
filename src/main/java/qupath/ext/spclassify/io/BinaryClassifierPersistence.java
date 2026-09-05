package qupath.ext.spclassify.io;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.classifier.ModelType;
import qupath.ext.spclassify.io.ProjectStateManager.BinaryImportedTrainingData;
import qupath.ext.spclassify.io.ProjectStateManager.SavedState;
import qupath.ext.spclassify.model.LabelStore;
import qupath.lib.projects.Project;

/**
 * Per-marker binary-classifier persistence.
 * <p>
 * State lives under {@code <project>/celltune/binary/<sanitizedMarker>.json}
 * (reusing the {@link SavedState} structure with exactly two class names) and
 * imported training payloads under {@code <project>/celltune/binary-imported/<marker>.json}.
 * <p>
 * Extracted verbatim from {@link ProjectStateManager}; the public API there delegates
 * here so existing call sites are unaffected. Shares the package-private helpers
 * ({@link ProjectStateManager#getCellTuneDir(Project)}, {@code writeState},
 * {@code toSavedTrainingRows}, {@link ProjectStateManager#toLabelStore(SavedState)},
 * {@code TIMESTAMP_FMT}, {@code GSON}) with its siblings in this package.
 */
final class BinaryClassifierPersistence {

    private static final Logger logger = LoggerFactory.getLogger(BinaryClassifierPersistence.class);

    private BinaryClassifierPersistence() {} // utility class

    /**
     * Resolve the binary/ subdirectory inside the celltune dir, creating it if needed.
     */
    private static Path getBinaryDir(Project<?> project) throws IOException {
        Path ctDir = ProjectStateManager.getCellTuneDir(project);
        Path binaryDir = ctDir.resolve("binary");
        Files.createDirectories(binaryDir);
        return binaryDir;
    }

    static Path saveBinaryState(
            Project<?> project,
            String sanitizedMarkerName,
            LabelStore labelStore,
            List<String> featureNames,
            List<String> classNames,
            byte[] xgboostBytes,
            byte[] lightgbmBytes,
            byte[] rfModel1Bytes,
            byte[] rfModel2Bytes,
            ModelType model1Type,
            ModelType model2Type)
            throws IOException {
        Path binaryDir = getBinaryDir(project);
        Path outPath = binaryDir.resolve(sanitizedMarkerName + ".json");

        SavedState state = new SavedState();
        state.name = sanitizedMarkerName;
        state.timestamp = LocalDateTime.now().format(ProjectStateManager.TIMESTAMP_FMT);
        state.featureNames = List.copyOf(featureNames);
        state.classNames = List.copyOf(classNames);
        state.labels = labelStore.getAllLabels();
        if (xgboostBytes != null) state.xgboostModelBase64 = Base64.getEncoder().encodeToString(xgboostBytes);
        if (lightgbmBytes != null)
            state.lightgbmModelBase64 = Base64.getEncoder().encodeToString(lightgbmBytes);
        if (rfModel1Bytes != null) state.rfModel1Base64 = Base64.getEncoder().encodeToString(rfModel1Bytes);
        if (rfModel2Bytes != null) state.rfModel2Base64 = Base64.getEncoder().encodeToString(rfModel2Bytes);
        state.model1Type = model1Type != null ? model1Type.name() : ModelType.XGBOOST.name();
        state.model2Type = model2Type != null ? model2Type.name() : ModelType.LIGHTGBM.name();

        ProjectStateManager.writeState(outPath, state);
        logger.info("Saved binary classifier state for '{}' to {}", sanitizedMarkerName, outPath);
        return outPath;
    }

    static SavedState loadBinaryState(Project<?> project, String sanitizedMarkerName) throws IOException {
        Path binaryDir = getBinaryDir(project);
        Path statePath = binaryDir.resolve(sanitizedMarkerName + ".json");
        if (!Files.exists(statePath)) return null;
        String json = Files.readString(statePath, StandardCharsets.UTF_8);
        return ProjectStateManager.GSON.fromJson(json, SavedState.class);
    }

    static void saveBinaryLabels(Project<?> project, String sanitizedMarkerName, LabelStore labelStore)
            throws IOException {
        Path binaryDir = getBinaryDir(project);
        Path statePath = binaryDir.resolve(sanitizedMarkerName + ".json");

        SavedState state;
        if (Files.exists(statePath)) {
            String existing = Files.readString(statePath, StandardCharsets.UTF_8);
            state = ProjectStateManager.GSON.fromJson(existing, SavedState.class);
            if (state == null) state = new SavedState();
        } else {
            state = new SavedState();
        }
        state.name = sanitizedMarkerName;
        state.timestamp = LocalDateTime.now().format(ProjectStateManager.TIMESTAMP_FMT);
        state.labels = labelStore.getAllLabels();

        ProjectStateManager.writeState(statePath, state);
        logger.info("Saved binary labels for '{}' ({} labels)", sanitizedMarkerName, labelStore.size());
    }

    static LabelStore loadBinaryLabels(Project<?> project, String sanitizedMarkerName) throws IOException {
        SavedState state = loadBinaryState(project, sanitizedMarkerName);
        if (state == null || state.labels == null) return new LabelStore(sanitizedMarkerName);
        return ProjectStateManager.toLabelStore(state);
    }

    private static class SavedBinaryImportedTrainingData {
        public String timestamp;
        public List<String> featureNames;
        public List<SavedState.SavedTrainingRow> rows;
    }

    /**
     * Resolve the binary-imported/ subdirectory inside the celltune dir, creating it if needed.
     */
    private static Path getBinaryImportedDir(Project<?> project) throws IOException {
        Path ctDir = ProjectStateManager.getCellTuneDir(project);
        Path importedDir = ctDir.resolve("binary-imported");
        Files.createDirectories(importedDir);
        return importedDir;
    }

    static Path saveBinaryImportedTrainingData(
            Project<?> project,
            String sanitizedMarkerName,
            List<String> featureNames,
            List<GroundTruthIO.TrainingRow> rows)
            throws IOException {
        String safeMarker = BinaryClassifierRegistry.sanitizeMarkerName(sanitizedMarkerName);
        Path importedDir = getBinaryImportedDir(project);
        Path outPath = importedDir.resolve(safeMarker + ".json");

        SavedBinaryImportedTrainingData saved = new SavedBinaryImportedTrainingData();
        saved.timestamp = LocalDateTime.now().format(ProjectStateManager.TIMESTAMP_FMT);
        saved.featureNames = featureNames == null ? List.of() : List.copyOf(featureNames);
        saved.rows = ProjectStateManager.toSavedTrainingRows(rows);

        Files.writeString(outPath, ProjectStateManager.GSON.toJson(saved), StandardCharsets.UTF_8);
        int rowCount = saved.rows == null ? 0 : saved.rows.size();
        logger.info("Saved binary imported training rows for '{}' ({} rows)", safeMarker, rowCount);
        return outPath;
    }

    /**
     * Delete the imported training payload for a binary marker, if present.
     *
     * @return true if a payload file existed and was deleted
     */
    static boolean deleteBinaryImportedTrainingData(Project<?> project, String sanitizedMarkerName) throws IOException {
        String safeMarker = BinaryClassifierRegistry.sanitizeMarkerName(sanitizedMarkerName);
        Path importedDir = getBinaryImportedDir(project);
        Path inPath = importedDir.resolve(safeMarker + ".json");
        boolean deleted = Files.deleteIfExists(inPath);
        if (deleted) {
            logger.info("Deleted binary imported training rows for '{}'", safeMarker);
        }
        return deleted;
    }

    static BinaryImportedTrainingData loadBinaryImportedTrainingData(Project<?> project, String sanitizedMarkerName)
            throws IOException {
        String safeMarker = BinaryClassifierRegistry.sanitizeMarkerName(sanitizedMarkerName);
        Path importedDir = getBinaryImportedDir(project);
        Path inPath = importedDir.resolve(safeMarker + ".json");
        if (!Files.exists(inPath)) {
            return null;
        }

        SavedBinaryImportedTrainingData saved = ProjectStateManager.GSON.fromJson(
                Files.readString(inPath, StandardCharsets.UTF_8), SavedBinaryImportedTrainingData.class);
        if (saved == null) {
            return null;
        }

        List<String> featureNames = saved.featureNames == null ? List.of() : List.copyOf(saved.featureNames);
        List<GroundTruthIO.TrainingRow> rows = decodeSavedTrainingRows(saved.rows);
        return new BinaryImportedTrainingData(featureNames, rows);
    }

    private static List<GroundTruthIO.TrainingRow> decodeSavedTrainingRows(
            List<SavedState.SavedTrainingRow> savedRows) {
        if (savedRows == null || savedRows.isEmpty()) {
            return List.of();
        }
        List<GroundTruthIO.TrainingRow> rows = new ArrayList<>();
        for (SavedState.SavedTrainingRow saved : savedRows) {
            if (saved == null || saved.label == null || saved.label.isBlank() || saved.features == null) {
                continue;
            }
            rows.add(new GroundTruthIO.TrainingRow(saved.label, saved.features.clone()));
        }
        return rows;
    }
}
