package qupath.ext.spclassify.io;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.classifier.ModelType;
import qupath.ext.spclassify.classifier.TrainingMetrics;
import qupath.ext.spclassify.model.CellTypeTable;
import qupath.ext.spclassify.model.LabelStore;
import qupath.ext.spclassify.model.PopulationSet;
import qupath.lib.projects.Project;

/**
 * Saves and loads the full classifier state to/from a JSON file in the QuPath project folder.
 * <p>
 * The state includes:
 * <ul>
 *   <li>Serialised model bytes (XGBoost + LightGBM) as Base64</li>
 *   <li>The current {@link LabelStore}</li>
 *   <li>Feature column names used during training</li>
 *   <li>Class name ordering</li>
 * </ul>
 * Files are stored under {@code <project>/celltune/} with a descriptive name.
 */
public class ProjectStateManager {

    private static final Logger logger = LoggerFactory.getLogger(ProjectStateManager.class);
    private static final String CELLTUNE_DIR = "celltune";
    private static final String STATE_FILENAME = "classifier-state.json";
    // Package-private so the focused persistence helpers in this package
    // (PredictionPersistence, …) share one identically-configured Gson instance.
    static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    // Package-private so the focused persistence helpers in this package share the format.
    static final DateTimeFormatter TIMESTAMP_FMT = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");

    private ProjectStateManager() {} // utility class

    /**
     * Internal JSON-serialisable structure for the full classifier state.
     */
    public static class SavedState {
        public static class SavedTrainingRow {
            public String label;
            public float[] features;
        }

        public String name;
        public String timestamp;
        public List<String> featureNames;
        public List<String> classNames;
        public Map<String, String> labels; // cellId â†’ class name
        public String xgboostModelBase64; // null if not yet trained
        public String lightgbmModelBase64; // null if not yet trained
        public String rfModel1Base64; // null if not yet trained
        public String rfModel2Base64; // null if not yet trained
        public String model1Type; // ModelType name, null defaults to XGBOOST
        public String model2Type; // ModelType name, null defaults to LIGHTGBM
        // New fields for feature selection and normalization
        public List<String> selectedFeatures;
        public Map<String, String> featureTransforms; // feature name â†’ transform name
        public Double arcsinhCofactor;
        public List<String> importedTrainingFeatureNames;
        public List<SavedTrainingRow> importedTrainingRows;
        // Persisted per-model training/validation metrics (nullable until first training)
        public TrainingMetrics model1TrainMetrics;
        public TrainingMetrics model1ValMetrics;
        public TrainingMetrics model2TrainMetrics;
        public TrainingMetrics model2ValMetrics;
    }

    /**
     * Resolve the celltune directory inside the project folder, creating it if needed.
     *
     * @param project the QuPath project
     * @return path to the celltune subdirectory
     * @throws IOException if the directory cannot be created
     */
    public static Path getCellTuneDir(Project<?> project) throws IOException {
        Path projectDir = project.getPath().getParent();
        Path ctDir = projectDir.resolve(CELLTUNE_DIR);
        Files.createDirectories(ctDir);
        return ctDir;
    }

    /**
     * Resolve the celltune directory <em>without</em> creating it.
     *
     * @param project the QuPath project
     * @return the path to {@code <project>/celltune}, or {@code null} if the
     *         project has no path on disk
     */
    public static Path cellTuneDirPath(Project<?> project) {
        if (project == null || project.getPath() == null) {
            return null;
        }
        return project.getPath().getParent().resolve(CELLTUNE_DIR);
    }

    /** @return true if the project has a {@code celltune/} state directory on disk. */
    public static boolean hasProjectState(Project<?> project) {
        Path dir = cellTuneDirPath(project);
        return dir != null && Files.isDirectory(dir);
    }

    /**
     * Zip the entire {@code celltune/} state directory to a timestamped archive
     * ({@code celltune_backup_<timestamp>.zip}) beside it in the project folder,
     * as a safety net before a destructive reset.
     *
     * @param project the QuPath project
     * @return the path to the written archive, or {@code null} if there was no
     *         state directory (or it was empty)
     * @throws IOException if writing the archive fails
     */
    public static Path backupProjectState(Project<?> project) throws IOException {
        Path ctDir = cellTuneDirPath(project);
        if (ctDir == null || !Files.isDirectory(ctDir)) {
            return null;
        }
        Path zipTarget = ctDir.getParent()
                .resolve("celltune_backup_" + LocalDateTime.now().format(TIMESTAMP_FMT) + ".zip");
        Path written = zipDirectory(ctDir, zipTarget);
        if (written != null) {
            logger.info("Backed up SP Classify state to {}", written);
        }
        return written;
    }

    /**
     * Delete the entire {@code celltune/} state directory. A no-op if it does not
     * exist. Callers should normally {@link #backupProjectState(Project) back up}
     * first.
     *
     * @param project the QuPath project
     * @throws IOException if deletion fails
     */
    public static void deleteProjectState(Project<?> project) throws IOException {
        Path ctDir = cellTuneDirPath(project);
        if (ctDir == null) {
            return;
        }
        deleteDirectoryRecursively(ctDir);
        logger.info("Deleted SP Classify state directory {}", ctDir);
    }

    /**
     * Zip every file under {@code dir} into {@code zipTarget}, using paths
     * relative to {@code dir} as entry names.
     *
     * @return {@code zipTarget} if at least one file was written, or {@code null}
     *         if {@code dir} is not a directory or is empty (no archive is left
     *         behind in the empty case)
     */
    static Path zipDirectory(Path dir, Path zipTarget) throws IOException {
        return FileSystemUtilities.zipDirectory(dir, zipTarget);
    }

    /** Recursively delete a directory tree. A no-op if {@code dir} does not exist. */
    static void deleteDirectoryRecursively(Path dir) throws IOException {
        FileSystemUtilities.deleteDirectoryRecursively(dir);
    }

    /**
     * Save the classifier state to the project's celltune directory.
     *
     * @param project       the QuPath project
     * @param classifierName user-given name for this classifier
     * @param labelStore    current ground-truth labels
     * @param featureNames  feature column ordering used during training
     * @param classNames    class name ordering used during training
     * @param xgboostBytes  serialised XGBoost model bytes (may be null)
     * @param lightgbmBytes serialised LightGBM model bytes (may be null)
     * @return path to the saved JSON file
     * @throws IOException if writing fails
     */
    public static Path saveState(
            Project<?> project,
            String classifierName,
            LabelStore labelStore,
            List<String> featureNames,
            List<String> classNames,
            byte[] xgboostBytes,
            byte[] lightgbmBytes,
            byte[] rfModel1Bytes,
            byte[] rfModel2Bytes,
            ModelType model1Type,
            ModelType model2Type,
            List<String> importedTrainingFeatureNames,
            List<GroundTruthIO.TrainingRow> importedTrainingRows)
            throws IOException {
        Path dir = getCellTuneDir(project);
        Path outPath = dir.resolve(STATE_FILENAME);

        SavedState existing = null;
        if (Files.exists(outPath)) {
            try {
                String existingJson = Files.readString(outPath, StandardCharsets.UTF_8);
                existing = GSON.fromJson(existingJson, SavedState.class);
            } catch (Exception ex) {
                logger.warn("Failed to read existing state for imported training preservation: {}", ex.getMessage());
            }
        }

        SavedState state = new SavedState();
        state.name = classifierName;
        state.timestamp = LocalDateTime.now().format(TIMESTAMP_FMT);
        state.featureNames = List.copyOf(featureNames);
        state.classNames = List.copyOf(classNames);
        state.labels = labelStore.getAllLabels();
        if (xgboostBytes != null) {
            state.xgboostModelBase64 = Base64.getEncoder().encodeToString(xgboostBytes);
        }
        if (lightgbmBytes != null) {
            state.lightgbmModelBase64 = Base64.getEncoder().encodeToString(lightgbmBytes);
        }
        if (rfModel1Bytes != null) {
            state.rfModel1Base64 = Base64.getEncoder().encodeToString(rfModel1Bytes);
        }
        if (rfModel2Bytes != null) {
            state.rfModel2Base64 = Base64.getEncoder().encodeToString(rfModel2Bytes);
        }
        state.model1Type = model1Type != null ? model1Type.name() : ModelType.XGBOOST.name();
        state.model2Type = model2Type != null ? model2Type.name() : ModelType.LIGHTGBM.name();
        // Save feature selection and normalization
        state.selectedFeatures = null; // to be set by caller
        state.featureTransforms = null; // to be set by caller
        state.arcsinhCofactor = null; // to be set by caller

        if (importedTrainingFeatureNames != null || importedTrainingRows != null) {
            state.importedTrainingFeatureNames =
                    importedTrainingFeatureNames == null || importedTrainingFeatureNames.isEmpty()
                            ? null
                            : List.copyOf(importedTrainingFeatureNames);
            state.importedTrainingRows = toSavedTrainingRows(importedTrainingRows);
        } else if (existing != null) {
            // Preserve previously saved imported training data if caller didn't provide it.
            state.importedTrainingFeatureNames = existing.importedTrainingFeatureNames;
            state.importedTrainingRows = existing.importedTrainingRows;
        }

        writeState(outPath, state);
        logger.info(
                "Saved classifier state to {} ({} labels, {} features)",
                outPath,
                labelStore.size(),
                featureNames.size());
        return outPath;
    }

    /**
     * Backward-compatible overload without imported training data.
     */
    public static Path saveState(
            Project<?> project,
            String classifierName,
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
        return saveState(
                project,
                classifierName,
                labelStore,
                featureNames,
                classNames,
                xgboostBytes,
                lightgbmBytes,
                rfModel1Bytes,
                rfModel2Bytes,
                model1Type,
                model2Type,
                null,
                null);
    }

    /**
     * Load a previously saved classifier state from the project's celltune directory.
     *
     * @param project the QuPath project
     * @return the loaded state, or null if no saved state exists
     * @throws IOException if reading or parsing fails
     */
    public static SavedState loadState(Project<?> project) throws IOException {
        Path dir = getCellTuneDir(project);
        Path statePath = dir.resolve(STATE_FILENAME);

        if (!Files.exists(statePath)) {
            logger.info("No saved classifier state found at {}", statePath);
            return null;
        }

        String json = Files.readString(statePath, StandardCharsets.UTF_8);
        SavedState state = GSON.fromJson(json, SavedState.class);
        logger.info(
                "Loaded classifier state '{}' from {} ({} labels)",
                state.name,
                statePath,
                state.labels != null ? state.labels.size() : 0);
        return state;
    }

    /**
     * Persist the user's selected feature list so it survives QuPath restarts.
     * Merges into the existing state file (creating one if needed).
     *
     * @param project           target QuPath project
     * @param selectedFeatures  feature names to persist, or {@code null}/empty to clear the selection
     * @return path to the updated state file
     * @throws IOException if reading/writing the state file fails
     */
    public static Path saveSelectedFeatures(Project<?> project, List<String> selectedFeatures) throws IOException {
        Path dir = getCellTuneDir(project);
        Path statePath = dir.resolve(STATE_FILENAME);

        SavedState state;
        if (Files.exists(statePath)) {
            try {
                String json = Files.readString(statePath, StandardCharsets.UTF_8);
                state = GSON.fromJson(json, SavedState.class);
            } catch (Exception ex) {
                logger.warn("Failed to read existing state for feature selection merge: {}", ex.getMessage());
                state = new SavedState();
            }
            if (state == null) state = new SavedState();
        } else {
            state = new SavedState();
        }

        if (state.name == null || state.name.isBlank()) state.name = "CellTune";
        state.timestamp = LocalDateTime.now().format(TIMESTAMP_FMT);
        state.selectedFeatures =
                (selectedFeatures == null || selectedFeatures.isEmpty()) ? null : List.copyOf(selectedFeatures);

        writeState(statePath, state);
        logger.info(
                "Saved selected features to {} ({} features)",
                statePath,
                state.selectedFeatures == null ? 0 : state.selectedFeatures.size());
        return statePath;
    }

    /**
     * Merge per-model train/validation metrics into an existing saved state.
     * Called after {@link #saveState} so the metrics survive QuPath restarts.
     *
     * @return true if the file existed and was updated, false otherwise
     */
    public static boolean saveTrainingMetrics(
            Project<?> project,
            TrainingMetrics m1Train,
            TrainingMetrics m1Val,
            TrainingMetrics m2Train,
            TrainingMetrics m2Val)
            throws IOException {
        Path outPath = getCellTuneDir(project).resolve(STATE_FILENAME);
        if (!Files.exists(outPath)) {
            logger.warn("Cannot save training metrics: no classifier state at {}", outPath);
            return false;
        }
        SavedState state;
        try {
            String json = Files.readString(outPath, StandardCharsets.UTF_8);
            state = GSON.fromJson(json, SavedState.class);
        } catch (Exception ex) {
            logger.warn("Failed to read existing state for metrics merge: {}", ex.getMessage());
            return false;
        }
        if (state == null) return false;
        state.model1TrainMetrics = m1Train;
        state.model1ValMetrics = m1Val;
        state.model2TrainMetrics = m2Train;
        state.model2ValMetrics = m2Val;
        writeState(outPath, state);
        return true;
    }

    /**
     * Persist imported training vectors/labels immediately, even before model training.
     */
    public static Path saveImportedTrainingData(
            Project<?> project, List<String> featureNames, List<GroundTruthIO.TrainingRow> rows) throws IOException {
        Path dir = getCellTuneDir(project);
        Path statePath = dir.resolve(STATE_FILENAME);

        SavedState state;
        if (Files.exists(statePath)) {
            String json = Files.readString(statePath, StandardCharsets.UTF_8);
            state = GSON.fromJson(json, SavedState.class);
            if (state == null) state = new SavedState();
        } else {
            state = new SavedState();
        }

        if (state.name == null || state.name.isBlank()) state.name = "CellTune";
        state.timestamp = LocalDateTime.now().format(TIMESTAMP_FMT);
        state.importedTrainingFeatureNames =
                featureNames == null || featureNames.isEmpty() ? null : List.copyOf(featureNames);
        state.importedTrainingRows = toSavedTrainingRows(rows);

        writeState(statePath, state);
        logger.info("Saved imported training data to {} ({} rows)", statePath, rows == null ? 0 : rows.size());
        return statePath;
    }

    /**
     * Strip the imported training data (rows + feature names) from the saved multi-class
     * classifier state, leaving the trained models, labels, feature selection and everything
     * else intact. A no-op if there is no saved state file.
     *
     * @return true if a state file existed and was rewritten without the imported rows
     * @throws IOException if writing fails
     */
    public static boolean clearImportedTrainingData(Project<?> project) throws IOException {
        Path statePath = getCellTuneDir(project).resolve(STATE_FILENAME);
        if (!Files.exists(statePath)) {
            return false;
        }
        SavedState state;
        try {
            String json = Files.readString(statePath, StandardCharsets.UTF_8);
            state = GSON.fromJson(json, SavedState.class);
        } catch (Exception ex) {
            logger.warn("Failed to read state for imported-data clear: {}", ex.getMessage());
            return false;
        }
        if (state == null) {
            return false;
        }
        state.importedTrainingFeatureNames = null;
        state.importedTrainingRows = null;
        writeState(statePath, state);
        logger.info("Cleared imported training data from {}", statePath);
        return true;
    }

    /**
     * Reconstruct a {@link LabelStore} from a saved state.
     *
     * @param state the loaded state
     * @return a LabelStore populated with the saved labels
     */
    public static LabelStore toLabelStore(SavedState state) {
        String storeName = state.name != null ? state.name : "Loaded";
        if (state.labels == null || state.labels.isEmpty()) {
            return new LabelStore(storeName);
        }
        return new LabelStore(storeName, state.labels);
    }

    /**
     * Decode the XGBoost model bytes from a saved state.
     *
     * @param state the loaded state
     * @return the raw model bytes, or null if not present
     */
    public static byte[] decodeXGBoostModel(SavedState state) {
        if (state.xgboostModelBase64 == null) return null;
        return Base64.getDecoder().decode(state.xgboostModelBase64);
    }

    /**
     * Decode the LightGBM model bytes from a saved state.
     */
    public static byte[] decodeLightGBMModel(SavedState state) {
        if (state.lightgbmModelBase64 == null) return null;
        return Base64.getDecoder().decode(state.lightgbmModelBase64);
    }

    /**
     * Decode the Random Forest model 1 bytes from a saved state.
     */
    public static byte[] decodeRFModel1(SavedState state) {
        if (state.rfModel1Base64 == null) return null;
        return Base64.getDecoder().decode(state.rfModel1Base64);
    }

    /**
     * Decode the Random Forest model 2 bytes from a saved state.
     */
    public static byte[] decodeRFModel2(SavedState state) {
        if (state.rfModel2Base64 == null) return null;
        return Base64.getDecoder().decode(state.rfModel2Base64);
    }

    /**
     * Get imported training feature names from a saved state.
     */
    public static List<String> getImportedTrainingFeatureNames(SavedState state) {
        if (state == null
                || state.importedTrainingFeatureNames == null
                || state.importedTrainingFeatureNames.isEmpty()) {
            return null;
        }
        return new ArrayList<>(state.importedTrainingFeatureNames);
    }

    /**
     * Decode imported training rows from a saved state.
     */
    public static List<GroundTruthIO.TrainingRow> decodeImportedTrainingRows(SavedState state) {
        if (state == null || state.importedTrainingRows == null || state.importedTrainingRows.isEmpty()) {
            return null;
        }

        List<GroundTruthIO.TrainingRow> rows = new ArrayList<>();
        for (var saved : state.importedTrainingRows) {
            if (saved == null || saved.label == null || saved.label.isBlank() || saved.features == null) {
                continue;
            }
            rows.add(new GroundTruthIO.TrainingRow(saved.label, saved.features.clone()));
        }
        return rows.isEmpty() ? null : rows;
    }

    /**
     * Get the model 1 type from a saved state, defaulting to XGBOOST.
     */
    public static ModelType getModel1Type(SavedState state) {
        if (state.model1Type == null) return ModelType.XGBOOST;
        try {
            return ModelType.valueOf(state.model1Type);
        } catch (IllegalArgumentException e) {
            return ModelType.XGBOOST;
        }
    }

    /**
     * Get the model 2 type from a saved state, defaulting to LIGHTGBM.
     */
    public static ModelType getModel2Type(SavedState state) {
        if (state.model2Type == null) return ModelType.LIGHTGBM;
        try {
            return ModelType.valueOf(state.model2Type);
        } catch (IllegalArgumentException e) {
            return ModelType.LIGHTGBM;
        }
    }

    /**
     * Backward-compatible overload for XGBoost + LightGBM only.
     */
    public static Path saveState(
            Project<?> project,
            String classifierName,
            LabelStore labelStore,
            List<String> featureNames,
            List<String> classNames,
            byte[] xgboostBytes,
            byte[] lightgbmBytes)
            throws IOException {
        return saveState(
                project,
                classifierName,
                labelStore,
                featureNames,
                classNames,
                xgboostBytes,
                lightgbmBytes,
                null,
                null,
                ModelType.XGBOOST,
                ModelType.LIGHTGBM,
                null,
                null);
    }

    static List<SavedState.SavedTrainingRow> toSavedTrainingRows(List<GroundTruthIO.TrainingRow> rows) {
        if (rows == null || rows.isEmpty()) return null;
        List<SavedState.SavedTrainingRow> out = new ArrayList<>(rows.size());
        for (var row : rows) {
            if (row == null || row.label() == null || row.label().isBlank() || row.features() == null) {
                continue;
            }
            SavedState.SavedTrainingRow saved = new SavedState.SavedTrainingRow();
            saved.label = row.label();
            saved.features = row.features().clone();
            out.add(saved);
        }
        return out.isEmpty() ? null : out;
    }

    static void writeState(Path statePath, SavedState state) throws IOException {
        String json = GSON.toJson(state);
        Files.writeString(statePath, json, StandardCharsets.UTF_8);
    }

    /**
     * Save a timestamped backup of the current labels.
     * Useful for auto-backup before each training cycle.
     *
     * <p>Backwards-compatible overload — image name is unknown so it is recorded
     * as {@code null}. Prefer {@link #backupLabels(Project, String, LabelStore)}.
     */
    public static Path backupLabels(Project<?> project, LabelStore labelStore) throws IOException {
        return backupLabels(project, null, labelStore);
    }

    /**
     * Save a timestamped backup of labels across the whole project. The current
     * image's in-memory labels are merged with any per-image label files on disk
     * so the backup captures every labelled cell with its originating image.
     *
     * <p>Format: a JSON list of {@code {cellId, className, imageName}} records.
     *
     * @param project          the QuPath project
     * @param currentImageName image name for the in-memory labels (nullable)
     * @param currentLabels    in-memory labels for the current image
     * @return path to the backup file
     * @throws IOException if writing fails
     */
    public static Path backupLabels(Project<?> project, String currentImageName, LabelStore currentLabels)
            throws IOException {
        return LabelPersistence.backupLabels(project, currentImageName, currentLabels);
    }

    // ── Per-image label persistence ──────────────────────────────────────────────

    /**
     * Save labels for a specific image. Used to persist review and manual labels
     * so they can be pooled into training from other images.
     */
    public static Path saveImageLabels(Project<?> project, String imageName, LabelStore labelStore) throws IOException {
        return LabelPersistence.saveImageLabels(project, imageName, labelStore);
    }

    /**
     * Load labels previously saved for a specific image.
     */
    public static LabelStore loadImageLabels(Project<?> project, String imageName) throws IOException {
        return LabelPersistence.loadImageLabels(project, imageName);
    }

    /**
     * Return true if a saved label file exists for the given image.
     * This is a cheap file-existence check — no JSON parsing.
     */
    public static boolean hasImageLabels(Project<?> project, String imageName) {
        return LabelPersistence.hasImageLabels(project, imageName);
    }

    /** Save labels for an image within an optional scope (null/blank = multi-class). */
    public static Path saveImageLabels(Project<?> project, String scope, String imageName, LabelStore labelStore)
            throws IOException {
        return LabelPersistence.saveImageLabels(project, scope, imageName, labelStore);
    }

    /** Load labels for an image within an optional scope (null/blank = multi-class). */
    public static LabelStore loadImageLabels(Project<?> project, String scope, String imageName) throws IOException {
        return LabelPersistence.loadImageLabels(project, scope, imageName);
    }

    /** True if labels exist for the image within the given scope. */
    public static boolean hasImageLabels(Project<?> project, String scope, String imageName) {
        return LabelPersistence.hasImageLabels(project, scope, imageName);
    }

    /**
     * Replace characters that are unsafe in file names with underscores.
     * Package-private so the focused persistence helpers in this package share it.
     */
    static String sanitiseFileName(String name) {
        return name.replaceAll("[^a-zA-Z0-9._\\-]", "_");
    }

    /**
     * List all per-image label files for the given scope (binary marker name,
     * or null for multi-class). Returns an empty list if the directory does
     * not exist. The returned paths are absolute and point at the JSON files
     * themselves, not the image names — call {@link #loadImageLabels(Project, String, String)}
     * to read them by name.
     */
    public static List<Path> listImageLabelFiles(Project<?> project, String scope) throws IOException {
        return LabelPersistence.listImageLabelFiles(project, scope);
    }

    /**
     * Overwrite a per-image labels file with the raw map of cellId -> class.
     * Used by self-heal flows that have already filtered the map and need to
     * persist the cleaned version directly (without going through LabelStore).
     */
    public static void writeImageLabelsRaw(Path filePath, Map<String, String> labels) throws IOException {
        LabelPersistence.writeImageLabelsRaw(filePath, labels);
    }

    /**
     * Read a per-image labels file as a raw cellId -> class map. Returns an
     * empty map if the file does not exist or is empty.
     */
    public static Map<String, String> readImageLabelsRaw(Path filePath) throws IOException {
        return LabelPersistence.readImageLabelsRaw(filePath);
    }

    // â”€â”€ Per-image sampled cell state
    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    private static final String IMAGE_SAMPLED_DIR = "image-sampled";

    /**
     * Save sampled cell IDs for a specific image.
     */
    public static Path saveImageSampledCells(Project<?> project, String imageName, List<String> sampledCellIds)
            throws IOException {
        Path dir = getCellTuneDir(project).resolve(IMAGE_SAMPLED_DIR);
        Files.createDirectories(dir);
        String safeFileName = sanitiseFileName(imageName) + ".json";
        Path outPath = dir.resolve(safeFileName);
        String json = GSON.toJson(sampledCellIds);
        Files.writeString(outPath, json, StandardCharsets.UTF_8);
        logger.info("Saved {} sampled cell IDs for image '{}' to {}", sampledCellIds.size(), imageName, outPath);
        return outPath;
    }

    /**
     * Load sampled cell IDs previously saved for a specific image.
     */
    @SuppressWarnings("unchecked")
    public static List<String> loadImageSampledCells(Project<?> project, String imageName) throws IOException {
        Path dir = getCellTuneDir(project).resolve(IMAGE_SAMPLED_DIR);
        String safeFileName = sanitiseFileName(imageName) + ".json";
        Path filePath = dir.resolve(safeFileName);
        if (!Files.exists(filePath)) {
            return null;
        }
        String json = Files.readString(filePath, StandardCharsets.UTF_8);
        List<String> ids = GSON.fromJson(json, List.class);
        if (ids == null || ids.isEmpty()) return null;
        return ids;
    }

    // â”€â”€ Per-image prediction persistence
    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /**
     * Save Pred_ALL predictions for a specific image.
     */
    public static Path saveImagePredictions(Project<?> project, String imageName, PopulationSet predAll)
            throws IOException {
        return PredictionPersistence.saveImagePredictions(project, imageName, predAll);
    }

    /**
     * Load Pred_ALL predictions previously saved for a specific image.
     */
    public static PopulationSet loadImagePredictions(Project<?> project, String imageName) throws IOException {
        return PredictionPersistence.loadImagePredictions(project, imageName);
    }

    /**
     * List the names of all images in the project that have saved prediction files.
     */
    public static List<String> listImagesWithPredictions(Project<?> project) throws IOException {
        return PredictionPersistence.listImagesWithPredictions(project);
    }
    // â”€â”€ Binary classifier persistence
    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /**
     * Save a binary classifier's trained state to
     * {@code <project>/celltune/binary/<sanitizedMarker>.json}.
     * <p>
     * Reuses the existing {@link SavedState} structure. The classNames list must
     * have exactly 2 entries (e.g. {@code ["CD4_pos", "CD4_neg"]}).
     *
     * @param project              the QuPath project
     * @param sanitizedMarkerName  the sanitized marker name (from {@code BinaryClassifierRegistry.sanitizeMarkerName})
     * @param labelStore           current ground-truth labels for this marker
     * @param featureNames         feature column ordering used during training
     * @param classNames           class name ordering (exactly 2 classes)
     * @param xgboostBytes         serialised XGBoost model bytes (may be null)
     * @param lightgbmBytes        serialised LightGBM model bytes (may be null)
     * @param rfModel1Bytes        serialised RF model 1 bytes (may be null)
     * @param rfModel2Bytes        serialised RF model 2 bytes (may be null)
     * @param model1Type           type of model 1
     * @param model2Type           type of model 2
     * @return path to the saved JSON file
     * @throws IOException if writing fails
     */
    public static Path saveBinaryState(
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
        return BinaryClassifierPersistence.saveBinaryState(
                project,
                sanitizedMarkerName,
                labelStore,
                featureNames,
                classNames,
                xgboostBytes,
                lightgbmBytes,
                rfModel1Bytes,
                rfModel2Bytes,
                model1Type,
                model2Type);
    }

    /**
     * Load a binary classifier's state from
     * {@code <project>/celltune/binary/<sanitizedMarker>.json}.
     *
     * @param project             the QuPath project
     * @param sanitizedMarkerName the sanitized marker name
     * @return the loaded state, or null if no state file exists for this marker
     * @throws IOException if reading or parsing fails
     */
    public static SavedState loadBinaryState(Project<?> project, String sanitizedMarkerName) throws IOException {
        return BinaryClassifierPersistence.loadBinaryState(project, sanitizedMarkerName);
    }

    /**
     * Save only the labels for a binary classifier without touching model bytes.
     * <p>
     * Used during active labelling before a model has been trained. If a state file
     * already exists, model bytes and other fields are preserved.
     *
     * @param project             the QuPath project
     * @param sanitizedMarkerName the sanitized marker name
     * @param labelStore          labels to persist
     * @throws IOException if reading or writing fails
     */
    public static void saveBinaryLabels(Project<?> project, String sanitizedMarkerName, LabelStore labelStore)
            throws IOException {
        BinaryClassifierPersistence.saveBinaryLabels(project, sanitizedMarkerName, labelStore);
    }

    /**
     * Load only the LabelStore for a binary classifier.
     * <p>
     * Returns an empty LabelStore named after the marker if no state file exists.
     *
     * @param project             the QuPath project
     * @param sanitizedMarkerName the sanitized marker name
     * @return a populated LabelStore, never null
     * @throws IOException if reading fails
     */
    public static LabelStore loadBinaryLabels(Project<?> project, String sanitizedMarkerName) throws IOException {
        return BinaryClassifierPersistence.loadBinaryLabels(project, sanitizedMarkerName);
    }

    /**
     * Marker-specific imported training payload used by binary transfer workflows.
     */
    public record BinaryImportedTrainingData(List<String> featureNames, List<GroundTruthIO.TrainingRow> rows) {}

    /**
     * Save imported training rows for a specific binary marker.
     *
     * @param project             the QuPath project
     * @param sanitizedMarkerName sanitized marker name (or raw marker name that can be sanitized)
     * @param featureNames        ordered feature names for the training rows
     * @param rows                imported labelled feature rows
     * @return path to the persisted marker payload
     * @throws IOException if writing fails
     */
    public static Path saveBinaryImportedTrainingData(
            Project<?> project,
            String sanitizedMarkerName,
            List<String> featureNames,
            List<GroundTruthIO.TrainingRow> rows)
            throws IOException {
        return BinaryClassifierPersistence.saveBinaryImportedTrainingData(
                project, sanitizedMarkerName, featureNames, rows);
    }

    /**
     * Load imported training rows for a specific binary marker.
     *
     * @param project             the QuPath project
     * @param sanitizedMarkerName sanitized marker name (or raw marker name that can be sanitized)
     * @return marker payload, or null when no payload exists
     * @throws IOException if reading fails
     */
    public static BinaryImportedTrainingData loadBinaryImportedTrainingData(
            Project<?> project, String sanitizedMarkerName) throws IOException {
        return BinaryClassifierPersistence.loadBinaryImportedTrainingData(project, sanitizedMarkerName);
    }

    /**
     * Delete the imported training payload for a binary marker, if present.
     *
     * @return true if a payload file existed and was deleted
     * @throws IOException if deletion fails
     */
    public static boolean deleteBinaryImportedTrainingData(Project<?> project, String sanitizedMarkerName)
            throws IOException {
        return BinaryClassifierPersistence.deleteBinaryImportedTrainingData(project, sanitizedMarkerName);
    }

    // -- Marker table persistence -----------------------------------------------

    /**
     * Save the imported marker table to {@code <project>/celltune/marker-table.json}
     * so it survives QuPath restarts and no longer has to be re-imported.
     */
    public static void saveMarkerTable(Project<?> project, CellTypeTable table) throws IOException {
        MarkerTablePersistence.saveMarkerTable(project, table);
    }

    /**
     * Load the persisted marker table from {@code <project>/celltune/marker-table.json}.
     */
    public static CellTypeTable loadMarkerTable(Project<?> project) {
        return MarkerTablePersistence.loadMarkerTable(project);
    }

    /**
     * Save UniFORM batch-normalization scales to {@code <project>/celltune/batch-shifts.json}
     * so the clustering "use batch-corrected values" path and future runs can reuse them.
     */
    public static void saveBatchShifts(
            Project<?> project, qupath.ext.spclassify.model.BatchNormalizerCohort.NormalizerFit fit)
            throws IOException {
        BatchNormPersistence.saveFit(project, fit);
    }

    /**
     * Load persisted batch-normalization scales, or {@code null} if none exist / unreadable.
     */
    public static qupath.ext.spclassify.model.BatchShifts loadBatchShifts(Project<?> project) {
        return BatchNormPersistence.load(project);
    }
}
