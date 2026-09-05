package qupath.ext.spclassify.io;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonSyntaxException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.model.CellTypeTable;
import qupath.lib.projects.Project;

/**
 * Imported marker-table persistence ({@code <project>/celltune/marker-table.json}).
 * <p>
 * Both CSV formats are preserved losslessly: simple tables store their display markers,
 * rule tables store their primary/secondary/tertiary gating expressions.
 * <p>
 * Extracted verbatim from {@link ProjectStateManager}; the public API there delegates
 * here so existing call sites are unaffected. Shares
 * {@link ProjectStateManager#getCellTuneDir(Project)} and {@code GSON} with its siblings
 * in this package.
 */
final class MarkerTablePersistence {

    private static final Logger logger = LoggerFactory.getLogger(MarkerTablePersistence.class);
    private static final String MARKER_TABLE_FILENAME = "marker-table.json";
    private static final int MARKER_TABLE_SCHEMA_VERSION = 1;

    private MarkerTablePersistence() {} // utility class

    static void saveMarkerTable(Project<?> project, CellTypeTable table) throws IOException {
        if (project == null) {
            logger.warn("saveMarkerTable: project is null - skipping save");
            return;
        }

        Path dir = ProjectStateManager.getCellTuneDir(project);
        Path path = dir.resolve(MARKER_TABLE_FILENAME);

        if (table == null || table.isEmpty()) {
            Files.deleteIfExists(path);
            logger.info("Cleared marker table at {}", path);
            return;
        }

        JsonObject root = new JsonObject();
        root.addProperty("version", MARKER_TABLE_SCHEMA_VERSION);
        boolean hasRules = table.hasGatingRules();
        root.addProperty("hasRules", hasRules);

        JsonArray entries = new JsonArray();
        for (String cellType : table.getCellTypes()) {
            JsonObject entry = new JsonObject();
            entry.addProperty("cellType", cellType);
            if (hasRules) {
                String primary = table.getPrimaryExpression(cellType);
                String secondary = table.getSecondaryMarkers(cellType);
                String tertiary = table.getTertiaryMarkers(cellType);
                if (primary != null) entry.addProperty("primary", primary);
                if (secondary != null) entry.addProperty("secondary", secondary);
                if (tertiary != null) entry.addProperty("tertiary", tertiary);
            } else {
                JsonArray markers = new JsonArray();
                for (String marker : table.getMarkers(cellType)) {
                    markers.add(marker);
                }
                entry.add("markers", markers);
            }
            entries.add(entry);
        }
        root.add("entries", entries);

        Files.writeString(path, ProjectStateManager.GSON.toJson(root), StandardCharsets.UTF_8);
        logger.info(
                "Saved marker table ({} cell types, {} format) to {}",
                entries.size(),
                hasRules ? "rule" : "simple",
                path);
    }

    static CellTypeTable loadMarkerTable(Project<?> project) {
        if (project == null) {
            return null;
        }

        Path path;
        try {
            path = ProjectStateManager.getCellTuneDir(project).resolve(MARKER_TABLE_FILENAME);
        } catch (IOException ex) {
            logger.warn("loadMarkerTable: cannot resolve celltune dir: {}", ex.getMessage());
            return null;
        }
        if (!Files.exists(path)) {
            return null;
        }

        JsonObject root;
        try {
            String json = Files.readString(path, StandardCharsets.UTF_8);
            JsonElement parsed = JsonParser.parseString(json);
            if (!parsed.isJsonObject()) {
                logger.warn("loadMarkerTable: unexpected JSON shape in {}", path);
                return null;
            }
            root = parsed.getAsJsonObject();
        } catch (IOException | JsonSyntaxException ex) {
            logger.warn("loadMarkerTable: failed to read {}: {}", path, ex.getMessage());
            return null;
        }

        JsonElement rawVersion = root.get("version");
        if (rawVersion != null
                && rawVersion.isJsonPrimitive()
                && rawVersion.getAsJsonPrimitive().isNumber()
                && rawVersion.getAsInt() != MARKER_TABLE_SCHEMA_VERSION) {
            logger.warn(
                    "loadMarkerTable: schema version {} in {} (expected {})",
                    rawVersion.getAsInt(),
                    path,
                    MARKER_TABLE_SCHEMA_VERSION);
        }

        boolean hasRules = root.has("hasRules")
                && root.get("hasRules").isJsonPrimitive()
                && root.get("hasRules").getAsBoolean();

        JsonElement rawEntries = root.get("entries");
        if (rawEntries == null || !rawEntries.isJsonArray()) {
            return null;
        }

        CellTypeTable table = new CellTypeTable();
        for (JsonElement rawEntry : rawEntries.getAsJsonArray()) {
            if (!rawEntry.isJsonObject()) continue;
            JsonObject entry = rawEntry.getAsJsonObject();
            String cellType = getOptionalString(entry, "cellType");
            if (cellType == null || cellType.isBlank()) {
                logger.warn("loadMarkerTable: skipping entry with missing cellType");
                continue;
            }
            if (hasRules) {
                table.putRule(
                        cellType,
                        getOptionalString(entry, "primary"),
                        getOptionalString(entry, "secondary"),
                        getOptionalString(entry, "tertiary"));
            } else {
                List<String> markers = new ArrayList<>();
                JsonElement rawMarkers = entry.get("markers");
                if (rawMarkers != null && rawMarkers.isJsonArray()) {
                    for (JsonElement m : rawMarkers.getAsJsonArray()) {
                        if (m.isJsonPrimitive() && m.getAsJsonPrimitive().isString()) {
                            markers.add(m.getAsString());
                        }
                    }
                }
                table.put(cellType, markers);
            }
        }

        if (table.isEmpty()) {
            return null;
        }
        logger.info(
                "Loaded marker table ({} cell types, {} format) from {}",
                table.size(),
                hasRules ? "rule" : "simple",
                path);
        return table;
    }

    private static String getOptionalString(JsonObject obj, String key) {
        JsonElement raw = obj.get(key);
        if (raw == null
                || raw.isJsonNull()
                || !raw.isJsonPrimitive()
                || !raw.getAsJsonPrimitive().isString()) {
            return null;
        }
        return raw.getAsString();
    }
}
