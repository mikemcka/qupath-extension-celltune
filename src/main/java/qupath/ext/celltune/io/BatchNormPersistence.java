package qupath.ext.celltune.io;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.celltune.model.BatchNormalizerCohort.NormalizerFit;
import qupath.ext.celltune.model.BatchShifts;
import qupath.lib.projects.Project;

/**
 * Persistence for UniFORM batch-normalization scales
 * ({@code <project>/celltune/batch-shifts.json}). Mirrors {@link MarkerTablePersistence}:
 * package-private, shares {@link ProjectStateManager#getCellTuneDir(Project)} and its
 * {@code GSON}, and the public API on {@link ProjectStateManager} delegates here. Plain
 * doubles — no Base64.
 */
final class BatchNormPersistence {

    private static final Logger logger = LoggerFactory.getLogger(BatchNormPersistence.class);
    private static final String FILENAME = "batch-shifts.json";
    private static final int SCHEMA_VERSION = 1;

    private BatchNormPersistence() {} // utility class

    /** Serialise a fit's per-(image, marker) scales + metadata. */
    static void saveFit(Project<?> project, NormalizerFit fit) throws IOException {
        if (project == null) {
            logger.warn("saveBatchShifts: project is null — skipping save");
            return;
        }
        Path dir = ProjectStateManager.getCellTuneDir(project);
        Path path = dir.resolve(FILENAME);
        if (fit == null || fit.scaleByImage().isEmpty()) {
            Files.deleteIfExists(path);
            logger.info("Cleared batch shifts at {}", path);
            return;
        }
        BatchShifts data = new BatchShifts();
        data.version = SCHEMA_VERSION;
        data.mode = fit.mode().name();
        data.nBins = fit.nBins();
        data.markers = new ArrayList<>(fit.markers());
        data.imageToBatch = new LinkedHashMap<>(fit.imageToBatch());
        data.scaleByImage = new LinkedHashMap<>(fit.scaleByImage());
        data.refGroupByMarker = new ArrayList<>(List.of(fit.refGroupByMarker()));
        String json = ProjectStateManager.GSON.toJson(data);
        Files.writeString(path, json, StandardCharsets.UTF_8);
        logger.info(
                "Saved batch shifts ({} markers, {} images) to {}",
                data.markers.size(),
                data.scaleByImage.size(),
                path);
    }

    /** Load the persisted scales, or {@code null} if none / unreadable. */
    static BatchShifts load(Project<?> project) {
        if (project == null) {
            return null;
        }
        Path dir = ProjectStateManager.cellTuneDirPath(project);
        if (dir == null) {
            return null;
        }
        Path path = dir.resolve(FILENAME);
        if (!Files.exists(path)) {
            return null;
        }
        try {
            String json = Files.readString(path, StandardCharsets.UTF_8);
            BatchShifts data = ProjectStateManager.GSON.fromJson(json, BatchShifts.class);
            if (data == null || data.isEmpty()) {
                return null;
            }
            // Gson leaves nulls for absent maps; normalise so callers can rely on them.
            if (data.imageToBatch == null) {
                data.imageToBatch = Map.of();
            }
            return data;
        } catch (Exception e) {
            logger.warn("Failed to load batch shifts from {}: {}", path, e.getMessage());
            return null;
        }
    }
}
