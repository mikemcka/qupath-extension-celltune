package qupath.ext.spclassify.io;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.projects.Project;

/**
 * Tracks all named binary classifiers in a QuPath project.
 * <p>
 * Persists a {@code markerName → relativeStatePath} mapping (relative to the celltune dir,
 * e.g. {@code "binary/CD4.json"}) to {@code <project>/celltune/binary-registry.json}.
 * <p>
 * All methods that accept a marker name call {@link #sanitizeMarkerName(String)} internally
 * to prevent path traversal and ensure safe filesystem use.
 */
public class BinaryClassifierRegistry {

    private static final Logger logger = LoggerFactory.getLogger(BinaryClassifierRegistry.class);
    private static final String REGISTRY_FILENAME = "binary-registry.json";
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private BinaryClassifierRegistry() {} // utility class

    /**
     * JSON-serialisable container for the registry data.
     */
    private static class RegistryData {
        public Map<String, String> markers = new LinkedHashMap<>(); // markerName → relativeStatePath
    }

    /**
     * Sanitize a marker name for safe use as a filename.
     * <p>
     * Allows alphanumerics, hyphens, underscores, and dots only. All other characters
     * are replaced with {@code _}. Blank names and names starting with {@code .} or
     * {@code -} are rejected to prevent path traversal.
     *
     * @param markerName the raw marker name entered by the user
     * @return the sanitized marker name (safe for use as a filename)
     * @throws IllegalArgumentException if name is blank or starts with '.' or '-'
     */
    public static String sanitizeMarkerName(String markerName) {
        if (markerName == null || markerName.isBlank())
            throw new IllegalArgumentException("Marker name must not be blank");
        String sanitized = markerName.trim().replaceAll("[^A-Za-z0-9._-]", "_");
        if (sanitized.startsWith(".") || sanitized.startsWith("-"))
            throw new IllegalArgumentException("Marker name must not start with '.' or '-': " + markerName);
        return sanitized;
    }

    /**
     * Load the registry from the project's celltune directory.
     * Returns an empty map if no registry file exists yet.
     *
     * @param project the QuPath project
     * @return mutable map of sanitized marker name → relative state path
     * @throws IOException if reading fails
     */
    public static Map<String, String> load(Project<?> project) throws IOException {
        Path dir = ProjectStateManager.getCellTuneDir(project);
        Path regPath = dir.resolve(REGISTRY_FILENAME);
        if (!Files.exists(regPath)) return new LinkedHashMap<>();
        String json = Files.readString(regPath, StandardCharsets.UTF_8);
        RegistryData data = GSON.fromJson(json, RegistryData.class);
        return (data != null && data.markers != null) ? new LinkedHashMap<>(data.markers) : new LinkedHashMap<>();
    }

    /**
     * Save the registry map back to disk.
     *
     * @param project the QuPath project
     * @param markers the current marker name → relative path mapping
     * @throws IOException if writing fails
     */
    public static void save(Project<?> project, Map<String, String> markers) throws IOException {
        Path dir = ProjectStateManager.getCellTuneDir(project);
        Path regPath = dir.resolve(REGISTRY_FILENAME);
        RegistryData data = new RegistryData();
        data.markers = new LinkedHashMap<>(markers);
        Files.writeString(regPath, GSON.toJson(data), StandardCharsets.UTF_8);
        logger.debug("Saved binary classifier registry ({} entries)", markers.size());
    }

    /**
     * Register a new marker in the registry and immediately persist it to disk.
     * <p>
     * The marker name is sanitized before use. If a sanitized form already exists,
     * the existing entry is preserved (idempotent).
     *
     * @param project    the QuPath project
     * @param registry   the in-memory registry map to update
     * @param markerName the raw marker name to register
     * @return the sanitized marker name that was actually stored
     * @throws IllegalArgumentException if the marker name is invalid
     * @throws IOException              if saving fails
     */
    public static String register(Project<?> project, Map<String, String> registry, String markerName)
            throws IOException {
        String safe = sanitizeMarkerName(markerName);
        String relPath = "binary/" + safe + ".json";
        registry.put(safe, relPath);
        save(project, registry);
        logger.info("Registered binary classifier '{}'", safe);
        return safe;
    }

    /**
     * Remove a marker from the registry and delete its state file from disk.
     * <p>
     * Silently does nothing if the marker is not present in the registry.
     *
     * @param project    the QuPath project
     * @param registry   the in-memory registry map to update
     * @param markerName the sanitized marker name to remove
     * @throws IOException if saving or deleting fails
     */
    public static void remove(Project<?> project, Map<String, String> registry, String markerName) throws IOException {
        String safe = sanitizeMarkerName(markerName);
        String relPath = registry.remove(safe);
        save(project, registry);
        if (relPath != null) {
            Path dir = ProjectStateManager.getCellTuneDir(project);
            Files.deleteIfExists(dir.resolve(relPath));
            logger.info("Removed binary classifier '{}' and deleted state file", safe);
        }
    }
}
