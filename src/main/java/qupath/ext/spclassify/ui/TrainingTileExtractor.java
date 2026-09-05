package qupath.ext.spclassify.ui;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntConsumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.WrappedBufferedImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;
import qupath.lib.regions.ImagePlane;
import qupath.lib.regions.ImageRegion;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.interfaces.ROI;

/**
 * Pre-extracts small "training tiles" around sampled cells so the review UI
 * can display them without paying the cost of switching between full project
 * images for every cell.
 *
 * <p>For each unique source image we open the {@link ImageData} once and reuse it
 * across every cell that came from that image. For each individual cell we build
 * a {@link CroppedImageServer} covering the cell's bounding box plus padding,
 * along with a small {@link qupath.lib.objects.hierarchy.PathObjectHierarchy}
 * containing translated copies of detections that fall inside the crop (so the
 * user sees spatial context). The displayed PathObject does NOT preserve the
 * original UUID — round-trip to the original cell is handled via
 * {@code ReviewController.getCurrentCellId()} which returns the cached sampler
 * cellId rather than reading it from the displayed object.
 *
 * <p>The extractor holds strong references to the source {@link ImageData}
 * instances; call {@link #close()} when the review session ends to release them.
 */
public final class TrainingTileExtractor implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(TrainingTileExtractor.class);

    /** Total region size = bounding box * (1 + 2 * PADDING_FACTOR) → 8x bbox at PADDING_FACTOR=3.5. */
    private static final double PADDING_FACTOR = 3.5;
    /** Floor on tile size in pixels so tiny cells still get a usable view. */
    private static final int MIN_TILE_PX = 96;

    /** A single pre-extracted tile description, ready to display. */
    public static final class TilePrep {
        final String cellId;
        final String imageName;
        final ImageData<BufferedImage> sourceImageData;
        final ImageRegion cropRegion;
        final List<PathObject> contextObjects; // already translated to tile coords
        final PathObject highlightCell; // member of contextObjects, in tile coords
        final ImageData.ImageType imageType;
        /**
         * Tile pixels pre-rendered at full resolution during extraction. Wrapping
         * this with a {@link WrappedBufferedImageServer} (see
         * {@link TrainingTileExtractor#buildTileImageData}) avoids paying the
         * per-tile disk read cost when the user navigates between cells in the
         * review UI. May be {@code null} if pre-render failed for this tile, in
         * which case display will be skipped.
         */
        final BufferedImage tilePixels;

        /**
         * Maps each context detection copy (including the {@link #highlightCell})
         * back to the original cell's UUID string in the source image. Enables the
         * review UI to identify cells the user manually clicks on inside the tile
         * and label them via {@link qupath.ext.spclassify.model.LabelStore}. Uses
         * identity-based lookup since the copies have fresh UUIDs.
         */
        final Map<PathObject, String> contextOriginalCellIds;

        TilePrep(
                String cellId,
                String imageName,
                ImageData<BufferedImage> sourceImageData,
                ImageRegion cropRegion,
                List<PathObject> contextObjects,
                PathObject highlightCell,
                ImageData.ImageType imageType,
                BufferedImage tilePixels,
                Map<PathObject, String> contextOriginalCellIds) {
            this.cellId = cellId;
            this.imageName = imageName;
            this.sourceImageData = sourceImageData;
            this.cropRegion = cropRegion;
            this.contextObjects = contextObjects;
            this.highlightCell = highlightCell;
            this.imageType = imageType;
            this.tilePixels = tilePixels;
            this.contextOriginalCellIds = contextOriginalCellIds;
        }

        public String cellId() {
            return cellId;
        }

        public String imageName() {
            return imageName;
        }

        public ImageRegion cropRegion() {
            return cropRegion;
        }

        public PathObject highlightCell() {
            return highlightCell;
        }

        public List<PathObject> contextObjects() {
            return contextObjects;
        }

        public ImageData<BufferedImage> sourceImageData() {
            return sourceImageData;
        }

        public ImageData.ImageType imageType() {
            return imageType;
        }

        public BufferedImage tilePixels() {
            return tilePixels;
        }

        public Map<PathObject, String> contextOriginalCellIds() {
            return contextOriginalCellIds;
        }
    }

    private final Map<String, TilePrep> preps = new LinkedHashMap<>();
    private final List<ImageData<BufferedImage>> openImageData = new ArrayList<>();

    private TrainingTileExtractor() {}

    /**
     * Build tiles for the given cell IDs.
     *
     * @param qupath          QuPath GUI (used to access the project)
     * @param cellIds         ordered list of cell IDs to extract
     * @param cellImageMap    cellId → image name mapping (cells without a mapping are skipped)
     * @param progress        optional progress callback (0..total); may be null
     * @return populated extractor; caller must close() when done
     */
    public static TrainingTileExtractor extract(
            QuPathGUI qupath, List<String> cellIds, Map<String, String> cellImageMap, IntConsumer progress) {
        var extractor = new TrainingTileExtractor();
        try {
            extractor.run(qupath, cellIds, cellImageMap, progress);
        } catch (Throwable t) {
            extractor.close();
            throw t;
        }
        return extractor;
    }

    private void run(QuPathGUI qupath, List<String> cellIds, Map<String, String> cellImageMap, IntConsumer progress) {
        Project<BufferedImage> project = qupath.getProject();
        if (project == null) {
            logger.warn("No project open; cannot extract training tiles");
            return;
        }

        // Index project entries by image name.
        Map<String, ProjectImageEntry<BufferedImage>> entryByName = new LinkedHashMap<>();
        for (var entry : project.getImageList()) {
            if (entry != null && entry.getImageName() != null) {
                entryByName.put(entry.getImageName(), entry);
            }
        }

        // Reuse the ImageData that's already open in QuPath for the currently
        // selected image — opening a fresh server for a 20k\u00d720k pyramidal TIFF
        // is one of the biggest costs in the whole extraction pipeline, so we
        // skip it when we can. The viewer's hierarchy is the same one the user
        // sees, so reads stay consistent with what's on screen.
        ImageData<BufferedImage> liveImageData = qupath.getImageData();
        String liveImageName = null;
        if (liveImageData != null) {
            var entry = project.getEntry(liveImageData);
            if (entry != null) liveImageName = entry.getImageName();
        }
        final String liveImageNameFinal = liveImageName;

        // Group cells by image so each ImageData is opened at most once.
        Map<String, List<String>> idsByImage = new LinkedHashMap<>();
        for (String id : cellIds) {
            if (id == null || id.isBlank()) continue;
            String img = cellImageMap == null ? null : cellImageMap.get(id);
            if (img == null || img.isBlank()) continue;
            idsByImage.computeIfAbsent(img, k -> new ArrayList<>()).add(id);
        }

        final int total = cellIds.size();
        final AtomicInteger processed = new AtomicInteger();

        // One worker thread per CPU, capped at 8 so we don't thrash a slow disk
        // or blow through OS file-handle limits on huge projects.
        int workers = Math.max(1, Math.min(8, Runtime.getRuntime().availableProcessors()));
        ExecutorService pool = Executors.newFixedThreadPool(workers, r -> {
            Thread t = new Thread(r, "sp-classify-tile-prep");
            t.setDaemon(true);
            return t;
        });

        // Collected preps appended in source order. We populate a parallel array
        // by index and then drain into the LinkedHashMap to preserve the
        // original cellId ordering chosen by the sampler.
        TilePrep[] flatPreps = new TilePrep[total];

        try {
            for (var groupEntry : idsByImage.entrySet()) {
                String imageName = groupEntry.getKey();
                List<String> ids = groupEntry.getValue();
                ProjectImageEntry<BufferedImage> entry = entryByName.get(imageName);
                if (entry == null) {
                    logger.warn("Project entry not found for image '{}'; skipping {} cell(s)", imageName, ids.size());
                    processed.addAndGet(ids.size());
                    if (progress != null) progress.accept(Math.min(processed.get(), total));
                    continue;
                }

                ImageData<BufferedImage> imageData;
                boolean isLive = liveImageNameFinal != null && liveImageNameFinal.equals(imageName);
                if (isLive) {
                    imageData = liveImageData;
                } else {
                    try {
                        imageData = entry.readImageData();
                    } catch (IOException ex) {
                        logger.warn("Could not read ImageData for '{}': {}", imageName, ex.getMessage());
                        processed.addAndGet(ids.size());
                        if (progress != null) progress.accept(Math.min(processed.get(), total));
                        continue;
                    }
                }

                final ImageServer<BufferedImage> server;
                final int serverW;
                final int serverH;
                final ImageData.ImageType imageType;
                try {
                    server = imageData.getServer();
                    imageType = imageData.getImageType();
                    serverW = server.getWidth();
                    serverH = server.getHeight();
                } catch (Exception ex) {
                    logger.warn(
                            "Could not open ImageServer for '{}' (file moved or deleted?): {}",
                            imageName,
                            ex.getMessage());
                    if (!isLive) {
                        try {
                            imageData.close();
                        } catch (Exception ignored) {
                        }
                    }
                    processed.addAndGet(ids.size());
                    if (progress != null) progress.accept(Math.min(processed.get(), total));
                    continue;
                }
                if (!isLive) {
                    // We only register non-live ImageData with the close list \u2014 the
                    // viewer owns the live one.
                    openImageData.add(imageData);
                }

                // Index detections by ID for fast cell lookup.
                final Map<String, PathObject> detById = new LinkedHashMap<>();
                for (PathObject det : imageData.getHierarchy().getDetectionObjects()) {
                    detById.put(det.getID().toString(), det);
                }
                final var hierarchy = imageData.getHierarchy();

                // Submit per-cell tasks for parallel pixel reads. We use the
                // ordinal position in the global cellIds list to write into
                // flatPreps so insertion order is preserved.
                List<Future<?>> futures = new ArrayList<>(ids.size());
                for (final String cellId : ids) {
                    final int outputIndex = cellIds.indexOf(cellId);
                    futures.add(pool.submit(() -> {
                        try {
                            TilePrep prep = buildPrepForCell(
                                    cellId, imageName, imageData, hierarchy, detById, server, serverW, serverH,
                                    imageType);
                            if (prep != null && outputIndex >= 0 && outputIndex < flatPreps.length) {
                                flatPreps[outputIndex] = prep;
                            }
                        } catch (Throwable t) {
                            logger.warn("Tile prep failed for cell {} in '{}': {}", cellId, imageName, t.toString());
                        } finally {
                            int done = processed.incrementAndGet();
                            if (progress != null) progress.accept(Math.min(done, total));
                        }
                    }));
                }
                // Wait for this image's cells to finish before moving on — keeps
                // the active hierarchy/server scoped per image.
                for (Future<?> f : futures) {
                    try {
                        f.get();
                    } catch (Exception ignored) {
                    }
                }
            }
        } finally {
            pool.shutdown();
            try {
                pool.awaitTermination(30, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        // Drain in source order into the ordered map.
        for (int i = 0; i < flatPreps.length; i++) {
            TilePrep p = flatPreps[i];
            if (p != null) preps.put(p.cellId, p);
        }

        logger.info(
                "TrainingTileExtractor: built {} tile(s) from {} source image(s)", preps.size(), openImageData.size());
    }

    /**
     * Build a single {@link TilePrep} for one cell. Safe to call from worker
     * threads as long as {@link qupath.lib.objects.hierarchy.PathObjectHierarchy}
     * reads are guarded — we synchronize on the hierarchy when calling
     * {@code getAllDetectionsForRegion}. ImageServer reads are thread-safe in
     * QuPath 0.7 (the tile cache is concurrent), so pixel reads run truly in
     * parallel.
     */
    private static TilePrep buildPrepForCell(
            String cellId,
            String imageName,
            ImageData<BufferedImage> imageData,
            qupath.lib.objects.hierarchy.PathObjectHierarchy hierarchy,
            Map<String, PathObject> detById,
            ImageServer<BufferedImage> server,
            int serverW,
            int serverH,
            ImageData.ImageType imageType) {
        PathObject originalCell = detById.get(cellId);
        if (originalCell == null) {
            logger.warn("Cell {} not found in image '{}'", cellId, imageName);
            return null;
        }
        ROI cellRoi = originalCell.getROI();
        if (cellRoi == null) return null;

        ImageRegion region = computeCropRegion(cellRoi, serverW, serverH);

        // Collect context detections inside the crop and translate to tile coords.
        // Hierarchy reads are synchronized so we don't race with another
        // worker on the same image.
        @SuppressWarnings("unused")
        ImagePlane tilePlane = ImagePlane.getPlaneWithChannel(0, 0, 0);
        List<PathObject> ctx = new ArrayList<>();
        Map<PathObject, String> ctxToOrig = new java.util.IdentityHashMap<>();
        PathObject highlight = null;
        double dx = -region.getX();
        double dy = -region.getY();

        java.util.Collection<PathObject> detectionsInRegion;
        synchronized (hierarchy) {
            detectionsInRegion = new ArrayList<>(hierarchy.getAllDetectionsForRegion(region));
        }

        for (PathObject det : detectionsInRegion) {
            ROI r = det.getROI();
            if (r == null) continue;
            ROI translated;
            try {
                translated = r.translate(dx, dy);
            } catch (Exception ex) {
                continue;
            }
            PathClass pc = det.getPathClass();
            PathObject copy = PathObjects.createDetectionObject(translated, pc);
            ctx.add(copy);
            try {
                ctxToOrig.put(copy, det.getID().toString());
            } catch (Exception ignored) {
                // Skip mapping if ID is unavailable; lookup will simply return null.
            }
            if (det == originalCell) {
                highlight = copy;
            }
        }

        if (highlight == null) {
            // Defensive: ensure the sampled cell is always in the tile, even if
            // getAllDetectionsForRegion missed it.
            ROI translatedCellRoi = cellRoi.translate(dx, dy);
            highlight = PathObjects.createDetectionObject(translatedCellRoi, originalCell.getPathClass());
            ctx.add(highlight);
            try {
                ctxToOrig.put(highlight, originalCell.getID().toString());
            } catch (Exception ignored) {
            }
        }

        // Pre-render the tile pixels at full resolution. This is the single
        // most expensive step per cell on a big pyramidal TIFF — doing it now
        // (in parallel across cells) means the review UI never blocks on disk
        // when the user clicks Next.
        BufferedImage tilePixels = null;
        try {
            RegionRequest req = RegionRequest.createInstance(server.getPath(), 1.0, region);
            tilePixels = server.readRegion(req);
        } catch (Exception ex) {
            logger.warn("Could not pre-render tile pixels for cell {} in '{}': {}", cellId, imageName, ex.getMessage());
        }

        return new TilePrep(cellId, imageName, imageData, region, ctx, highlight, imageType, tilePixels, ctxToOrig);
    }

    private static ImageRegion computeCropRegion(ROI cellRoi, int serverW, int serverH) {
        double w = cellRoi.getBoundsWidth();
        double h = cellRoi.getBoundsHeight();
        double size = Math.max(w, h);
        if (size <= 0) size = MIN_TILE_PX;
        double tileSize = Math.max(MIN_TILE_PX, size * (1 + 2 * PADDING_FACTOR));

        double cx = cellRoi.getCentroidX();
        double cy = cellRoi.getCentroidY();
        double half = tileSize / 2.0;

        int x = (int) Math.floor(Math.max(0, cx - half));
        int y = (int) Math.floor(Math.max(0, cy - half));
        int regionW = (int) Math.ceil(Math.min(serverW - x, tileSize));
        int regionH = (int) Math.ceil(Math.min(serverH - y, tileSize));
        regionW = Math.max(1, regionW);
        regionH = Math.max(1, regionH);

        int z = cellRoi.getZ();
        int t = cellRoi.getT();
        return ImageRegion.createInstance(x, y, regionW, regionH, z, t);
    }

    /** @return ordered map of cellId → TilePrep (insertion order matches the order extracted). */
    public Map<String, TilePrep> getPreps() {
        return preps;
    }

    /**
     * Build a fresh ImageData for the given prep, ready for the review viewer.
     *
     * <p>Uses the pre-rendered {@link TilePrep#tilePixels} (built during
     * extraction) wrapped in a {@link WrappedBufferedImageServer}. The result
     * is an in-memory image server — opening it is essentially free, so
     * switching between cells in the review UI does not touch disk.
     *
     * <p>Falls back to a cropped-from-source view if pre-rendered pixels are
     * unavailable (e.g. the pre-render step failed earlier).
     */
    public static ImageData<BufferedImage> buildTileImageData(TilePrep prep) {
        ImageServer<BufferedImage> tileServer;
        if (prep.tilePixels != null) {
            // Cheap path: pixels already in memory.
            String name = String.format(
                    "%s @ (%d,%d,%dx%d)",
                    prep.imageName,
                    prep.cropRegion.getX(),
                    prep.cropRegion.getY(),
                    prep.cropRegion.getWidth(),
                    prep.cropRegion.getHeight());
            // Preserve the source server's channel metadata so the channel
            // selector shows real biomarker names (DAPI, CD3, ...) instead of
            // generic "Channel 1..N" labels.
            var sourceServer = prep.sourceImageData.getServer();
            var channels = sourceServer.getMetadata().getChannels();
            tileServer = new WrappedBufferedImageServer(name, prep.tilePixels, channels);
        } else {
            // Slow fallback: read region on demand from the underlying server.
            ImageServer<BufferedImage> source = prep.sourceImageData.getServer();
            tileServer = new qupath.lib.images.servers.CroppedImageServer(source, prep.cropRegion);
        }

        var hierarchy = new qupath.lib.objects.hierarchy.PathObjectHierarchy();
        hierarchy.addObjects(prep.contextObjects);

        ImageData.ImageType type = prep.imageType == null ? ImageData.ImageType.UNSET : prep.imageType;
        return new ImageData<>(tileServer, hierarchy, type);
    }

    @Override
    public void close() {
        for (var data : openImageData) {
            try {
                data.close();
            } catch (Exception ex) {
                logger.debug("Closing source ImageData failed: {}", ex.getMessage());
            }
        }
        openImageData.clear();
        preps.clear();
    }
}
