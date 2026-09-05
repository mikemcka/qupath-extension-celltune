package qupath.ext.spclassify.io;

import com.google.gson.JsonSyntaxException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.model.LabelStore;
import qupath.lib.projects.Project;

/**
 * Per-image ground-truth label persistence (multi-class and binary-scoped) plus
 * the project-wide timestamped label backup.
 * <p>
 * Multi-class labels live in {@code <project>/celltune/image-labels/<image>.json};
 * binary-scoped labels live in {@code <project>/celltune/binary-image-labels/<scope>/<image>.json}
 * so each binary classifier owns its own per-image labels and they don't bleed between
 * classifiers.
 * <p>
 * Extracted verbatim from {@link ProjectStateManager}; the public API there delegates
 * here so existing call sites are unaffected. Shares
 * {@link ProjectStateManager#getCellTuneDir(Project)}, {@link ProjectStateManager#sanitiseFileName(String)},
 * {@code GSON} and {@code TIMESTAMP_FMT} with its siblings in this package.
 */
final class LabelPersistence {

    private static final Logger logger = LoggerFactory.getLogger(LabelPersistence.class);

    private static final String IMAGE_LABELS_DIR = "image-labels";
    private static final String BINARY_IMAGE_LABELS_DIR = "binary-image-labels";

    private LabelPersistence() {} // utility class

    /**
     * Save a timestamped backup of labels across the whole project. The current
     * image's in-memory labels are merged with any per-image label files on disk
     * so the backup captures every labelled cell with its originating image.
     */
    static Path backupLabels(Project<?> project, String currentImageName, LabelStore currentLabels) throws IOException {
        Path dir = ProjectStateManager.getCellTuneDir(project);
        String filename = "labels_backup_" + LocalDateTime.now().format(ProjectStateManager.TIMESTAMP_FMT) + ".json";
        Path outPath = dir.resolve(filename);

        // Collect per-image labels: image -> (cellId -> className).
        Map<String, Map<String, String>> byImage = new LinkedHashMap<>();

        // Disk-resident per-image labels (other images that aren't currently open).
        Path labelsDir = dir.resolve(IMAGE_LABELS_DIR);
        if (Files.isDirectory(labelsDir)) {
            try (var stream = Files.list(labelsDir)) {
                for (Path file : (Iterable<Path>) stream::iterator) {
                    String fname = file.getFileName().toString();
                    if (!fname.endsWith(".json")) continue;
                    String imageName = fname.substring(0, fname.length() - ".json".length());
                    try {
                        String json = Files.readString(file, StandardCharsets.UTF_8);
                        @SuppressWarnings("unchecked")
                        Map<String, String> labels = ProjectStateManager.GSON.fromJson(json, Map.class);
                        if (labels != null && !labels.isEmpty()) {
                            byImage.put(imageName, new LinkedHashMap<>(labels));
                        }
                    } catch (IOException | JsonSyntaxException e) {
                        logger.warn("Skipping unreadable label file {}: {}", file, e.getMessage());
                    }
                }
            }
        }

        // Overlay the current in-memory labels (most up-to-date for the open image).
        if (currentLabels != null && !currentLabels.getAllLabels().isEmpty()) {
            String key = currentImageName != null ? currentImageName : "";
            byImage.put(key, new LinkedHashMap<>(currentLabels.getAllLabels()));
        }

        // Flatten to a list of records so each entry carries its image name.
        List<Map<String, String>> records = new ArrayList<>();
        for (var imgEntry : byImage.entrySet()) {
            String imageName = imgEntry.getKey().isEmpty() ? null : imgEntry.getKey();
            for (var lbl : imgEntry.getValue().entrySet()) {
                Map<String, String> rec = new LinkedHashMap<>();
                rec.put("cellId", lbl.getKey());
                rec.put("className", lbl.getValue());
                rec.put("imageName", imageName);
                records.add(rec);
            }
        }

        Files.writeString(outPath, ProjectStateManager.GSON.toJson(records), StandardCharsets.UTF_8);
        logger.info("Backed up {} labels across {} image(s) to {}", records.size(), byImage.size(), outPath);
        return outPath;
    }

    // ── Multi-class per-image labels (no scope) ──────────────────────────────────

    static Path saveImageLabels(Project<?> project, String imageName, LabelStore labelStore) throws IOException {
        Path dir = ProjectStateManager.getCellTuneDir(project).resolve(IMAGE_LABELS_DIR);
        Files.createDirectories(dir);

        String safeFileName = ProjectStateManager.sanitiseFileName(imageName) + ".json";
        Path outPath = dir.resolve(safeFileName);

        String json = ProjectStateManager.GSON.toJson(labelStore.getAllLabels());
        Files.writeString(outPath, json, StandardCharsets.UTF_8);
        logger.info("Saved {} labels for image '{}' to {}", labelStore.size(), imageName, outPath);
        return outPath;
    }

    static LabelStore loadImageLabels(Project<?> project, String imageName) throws IOException {
        Path dir = ProjectStateManager.getCellTuneDir(project).resolve(IMAGE_LABELS_DIR);
        String safeFileName = ProjectStateManager.sanitiseFileName(imageName) + ".json";
        Path filePath = dir.resolve(safeFileName);

        if (!Files.exists(filePath)) {
            return null;
        }

        String json = Files.readString(filePath, StandardCharsets.UTF_8);
        @SuppressWarnings("unchecked")
        Map<String, String> labels = ProjectStateManager.GSON.fromJson(json, Map.class);
        if (labels == null || labels.isEmpty()) {
            return null;
        }

        LabelStore store = new LabelStore(imageName, labels);
        logger.info("Loaded {} labels for image '{}' from {}", store.size(), imageName, filePath);
        return store;
    }

    static boolean hasImageLabels(Project<?> project, String imageName) {
        try {
            Path dir = ProjectStateManager.getCellTuneDir(project).resolve(IMAGE_LABELS_DIR);
            Path filePath = dir.resolve(ProjectStateManager.sanitiseFileName(imageName) + ".json");
            return Files.exists(filePath);
        } catch (IOException e) {
            logger.debug("hasImageLabels: cannot resolve label path for '{}': {}", imageName, e.getMessage());
            return false;
        }
    }

    // ── Scope-aware overloads (binary classifier vs multi-class) ─────────────────
    //
    // When {@code scope} is null/blank, labels live in the shared
    // {@code image-labels/<image>.json} (multi-class).
    // When {@code scope} is a sanitized binary-marker name, labels live in
    // {@code binary-image-labels/<scope>/<image>.json} so each binary classifier
    // owns its own per-image labels and they don't bleed between classifiers.

    private static Path resolveImageLabelsDir(Project<?> project, String scope) throws IOException {
        Path ctDir = ProjectStateManager.getCellTuneDir(project);
        if (scope == null || scope.isBlank()) {
            return ctDir.resolve(IMAGE_LABELS_DIR);
        }
        String safeScope = ProjectStateManager.sanitiseFileName(scope);
        return ctDir.resolve(BINARY_IMAGE_LABELS_DIR).resolve(safeScope);
    }

    static Path saveImageLabels(Project<?> project, String scope, String imageName, LabelStore labelStore)
            throws IOException {
        Path dir = resolveImageLabelsDir(project, scope);
        Files.createDirectories(dir);
        String safeFileName = ProjectStateManager.sanitiseFileName(imageName) + ".json";
        Path outPath = dir.resolve(safeFileName);
        String json = ProjectStateManager.GSON.toJson(labelStore.getAllLabels());
        Files.writeString(outPath, json, StandardCharsets.UTF_8);
        logger.info(
                "Saved {} labels for image '{}' (scope='{}') to {}",
                labelStore.size(),
                imageName,
                scope == null ? "" : scope,
                outPath);
        return outPath;
    }

    static LabelStore loadImageLabels(Project<?> project, String scope, String imageName) throws IOException {
        Path dir = resolveImageLabelsDir(project, scope);
        Path filePath = dir.resolve(ProjectStateManager.sanitiseFileName(imageName) + ".json");
        if (!Files.exists(filePath)) return null;
        String json = Files.readString(filePath, StandardCharsets.UTF_8);
        @SuppressWarnings("unchecked")
        Map<String, String> labels = ProjectStateManager.GSON.fromJson(json, Map.class);
        if (labels == null || labels.isEmpty()) return null;
        return new LabelStore(imageName, labels);
    }

    static boolean hasImageLabels(Project<?> project, String scope, String imageName) {
        try {
            Path dir = resolveImageLabelsDir(project, scope);
            Path filePath = dir.resolve(ProjectStateManager.sanitiseFileName(imageName) + ".json");
            return Files.exists(filePath);
        } catch (IOException e) {
            logger.debug(
                    "hasImageLabels: cannot resolve label path for '{}' (scope '{}'): {}",
                    imageName,
                    scope,
                    e.getMessage());
            return false;
        }
    }

    static List<Path> listImageLabelFiles(Project<?> project, String scope) throws IOException {
        Path dir = resolveImageLabelsDir(project, scope);
        if (!Files.isDirectory(dir)) return List.of();
        try (var stream = Files.list(dir)) {
            return stream.filter(p -> p.getFileName().toString().endsWith(".json"))
                    .filter(Files::isRegularFile)
                    .sorted()
                    .toList();
        }
    }

    static void writeImageLabelsRaw(Path filePath, Map<String, String> labels) throws IOException {
        Files.createDirectories(filePath.getParent());
        String json = ProjectStateManager.GSON.toJson(labels);
        Files.writeString(filePath, json, StandardCharsets.UTF_8);
    }

    @SuppressWarnings("unchecked")
    static Map<String, String> readImageLabelsRaw(Path filePath) throws IOException {
        if (!Files.exists(filePath)) return new LinkedHashMap<>();
        String json = Files.readString(filePath, StandardCharsets.UTF_8);
        Map<String, String> labels = ProjectStateManager.GSON.fromJson(json, Map.class);
        return labels == null ? new LinkedHashMap<>() : labels;
    }
}
