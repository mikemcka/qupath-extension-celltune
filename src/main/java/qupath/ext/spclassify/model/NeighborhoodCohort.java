package qupath.ext.spclassify.model;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.function.DoubleConsumer;
import javafx.application.Platform;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.util.BackgroundExecutors;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.classes.PathClassTools;
import qupath.lib.projects.ProjectImageEntry;

/**
 * Headless, memory-safe backend for <b>project-wide</b> cellular-neighborhood (CN)
 * clustering — the cohort analogue of the single-image pipeline in
 * {@code NeighborhoodAnalysisDialog}, mirroring {@link CohortClusterModel}'s two
 * streaming passes (and reusing its {@code sampleIndices}/{@code nearestCentroid}):
 *
 * <ol>
 *   <li>{@link #sample} — pool a bounded random sample of per-cell neighborhood
 *       composition vectors across the selected images (drawn evenly per image).
 *       The caller standardises (optionally) and fits k-means on this sample.</li>
 *   <li>{@link #assignAcrossProject} — stream image-by-image, recompute every
 *       cell's composition, assign it to its nearest cohort centroid, write the CN
 *       id as a numeric {@code "CN"} measurement, and save each image.</li>
 * </ol>
 *
 * Because one model is fit once and applied everywhere, CN identity is consistent
 * across the cohort (CN 3 = the same micro-environment in every image), exactly as
 * Schürch et al. cluster both patient groups together. The cell phenotype
 * ({@code getPathClass()}) is never modified.
 */
public final class NeighborhoodCohort {

    private static final Logger logger = LoggerFactory.getLogger(NeighborhoodCohort.class);

    /** Numeric measurement written per cell (matches {@code NeighborhoodAnalysisDialog.CN_MEASUREMENT}). */
    public static final String CN_MEASUREMENT = "CN";

    private NeighborhoodCohort() {}

    /**
     * Shared neighborhood-window parameters for both passes.
     *
     * @param mode            how the window is defined (kNN / radius / Delaunay)
     * @param k               window size for {@link NeighborhoodModel.WindowMode#KNN}
     * @param radius          window radius for {@link NeighborhoodModel.WindowMode#RADIUS} (coordinate units)
     * @param delaunayMaxEdge Delaunay edge-length cutoff (coordinate units); {@code NaN} = data-driven auto
     *                        (Tukey upper whisker), {@code <= 0} = no pruning
     * @param includeCenter   count the centre cell's own type in its window
     * @param pixelOverride   µm/pixel override (nullable → use image calibration)
     */
    public record Params(
            NeighborhoodModel.WindowMode mode,
            int k,
            double radius,
            double delaunayMaxEdge,
            boolean includeCenter,
            Double pixelOverride) {}

    /** A bounded pool of raw composition rows for fitting. */
    public record SampleResult(double[][] rows, int sampled, int total, int imageCount) {}

    /**
     * Cohort assignment outcome.
     *
     * @param assigned  cells given a CN (non-empty window) across all images
     * @param empty     cells with an empty window (CN = -1)
     * @param cnCounts  per-CN cell counts (pooled across images)
     * @param cnMean    per-CN mean composition fraction (pooled, raw) for the enrichment heatmap
     */
    public record AssignResult(long assigned, long empty, long[] cnCounts, double[][] cnMean) {}

    private record ImageComp(List<PathObject> cells, double[][] comp) {}

    /** One image's contribution to the fit pool: its sampled rows and its non-empty-window count. */
    private record ImageSample(List<double[]> rows, int activeTotal) {}

    /** One image's assignment tally, merged into the cohort totals after all workers finish. */
    private record ImageAssign(long assigned, long empty, long[] cnCounts, double[][] cnMeanSum) {}

    // ── Pass 1: sample ──────────────────────────────────────────────────────────

    /**
     * Pool a bounded random sample of non-empty composition rows across {@code images},
     * drawn roughly evenly per image (seeded, deterministic). Returns raw fraction
     * rows; the caller standardises/fits.
     */
    public static SampleResult sample(
            List<ProjectImageEntry<BufferedImage>> entries,
            List<String> typeNames,
            Params params,
            int sampleCap,
            int workers,
            Consumer<String> log) {
        Map<String, Integer> typeIndex = typeIndex(typeNames);
        int nTypes = typeNames.size();
        int perImage = Math.max(1, sampleCap / Math.max(1, entries.size()));

        // One task per image (each an independent read + composition build). Each image
        // draws its own seeded sub-sample so the pooled fit set is deterministic no matter
        // the order tasks finish in — reproducibility the single shared RNG used to give.
        // Entries may span more than one project; each reads (and later saves) its own project.
        int parallelism = Math.max(1, Math.min(workers, Math.max(1, entries.size())));
        ExecutorService pool = BackgroundExecutors.newFixedPool(parallelism, "SpClassify-CN-Sample");
        try {
            List<Future<ImageSample>> futures = new ArrayList<>();
            for (int idx = 0; idx < entries.size(); idx++) {
                ProjectImageEntry<BufferedImage> entry = entries.get(idx);
                String name = entry.getImageName();
                long seed = 42L + idx;
                futures.add(
                        pool.submit(() -> sampleOneImage(entry, name, typeIndex, nTypes, params, perImage, seed, log)));
            }

            // Merge in submission order (== image order) so the pooled rows are stable.
            List<double[]> pooled = new ArrayList<>();
            int total = 0;
            int imageCount = 0;
            for (Future<ImageSample> f : futures) {
                ImageSample s;
                try {
                    s = f.get();
                } catch (Exception e) {
                    continue;
                }
                if (s == null) {
                    continue;
                }
                total += s.activeTotal();
                imageCount++;
                pooled.addAll(s.rows());
            }
            double[][] rows = pooled.toArray(new double[0][]);
            return new SampleResult(rows, rows.length, total, imageCount);
        } finally {
            pool.shutdownNow();
        }
    }

    /** Load one image, build its compositions, and draw a seeded sub-sample of its non-empty windows. */
    private static ImageSample sampleOneImage(
            ProjectImageEntry<BufferedImage> entry,
            String name,
            Map<String, Integer> typeIndex,
            int nTypes,
            Params params,
            int perImage,
            long seed,
            Consumer<String> log) {
        if (entry == null) {
            log.accept("[" + name + "] not in project — skipped");
            return null;
        }
        ImageData<BufferedImage> imageData;
        try {
            imageData = entry.readImageData();
        } catch (Exception e) {
            log.accept("[" + name + "] could not load — skipped");
            return null;
        }
        if (imageData == null) {
            return null;
        }
        ImageComp ic = build(imageData, typeIndex, nTypes, params);
        List<Integer> active = activeRows(ic.comp());
        if (active.isEmpty()) {
            log.accept("[" + name + "] no non-empty windows — skipped");
            return null;
        }
        int take = Math.min(perImage, active.size());
        int[] pick = CohortClusterModel.sampleIndices(active.size(), take, new Random(seed));
        List<double[]> rows = new ArrayList<>(pick.length);
        for (int p : pick) {
            rows.add(ic.comp()[active.get(p)]);
        }
        log.accept(String.format("[%s] sampled %,d of %,d windows", name, take, active.size()));
        return new ImageSample(rows, active.size());
    }

    // ── Pass 2: assign ──────────────────────────────────────────────────────────

    /**
     * Stream over {@code images}, assigning every cell's neighborhood composition to
     * its nearest cohort centroid and writing the CN id (1-based; empty windows = -1)
     * as a numeric {@code "CN"} measurement, then saving each image. When
     * {@code mean}/{@code sd} are non-null the same standardisation used at fit time
     * is replayed before the nearest-centroid lookup.
     *
     * @param centroids fitted centroids in the same space as the (optionally standardised) sample
     * @param openData  open image's live data (nullable); its hierarchy is mutated on the FX thread
     * @param openEntry project entry of the open image (nullable); matched by identity to reuse
     *                  {@code openData} instead of reloading, and to write on the FX thread
     */
    public static AssignResult assignAcrossProject(
            List<ProjectImageEntry<BufferedImage>> entries,
            List<String> typeNames,
            Params params,
            double[] mean,
            double[] sd,
            double[][] centroids,
            ImageData<BufferedImage> openData,
            ProjectImageEntry<BufferedImage> openEntry,
            int workers,
            Consumer<String> log,
            DoubleConsumer progress) {

        Map<String, Integer> typeIndex = typeIndex(typeNames);
        int nTypes = typeNames.size();
        int kc = centroids.length;
        boolean standardize = mean != null && sd != null;
        int nImages = entries.size();

        // Per-image classify+write+save, one task each. Each image mutates only its own
        // ImageData (and saves to its own project), so the only shared state is the merged tally
        // (done after join) and the progress counter (atomic). The open image still writes on the FX
        // thread inside applyMeasurements; closed images write on their worker thread.
        int parallelism = Math.max(1, Math.min(workers, Math.max(1, nImages)));
        ExecutorService pool = BackgroundExecutors.newFixedPool(parallelism, "SpClassify-CN-Assign");
        AtomicInteger done = new AtomicInteger(0);
        try {
            List<Future<ImageAssign>> futures = new ArrayList<>();
            for (ProjectImageEntry<BufferedImage> entry : entries) {
                futures.add(pool.submit(() -> {
                    try {
                        return assignOneImage(
                                entry,
                                typeIndex,
                                nTypes,
                                params,
                                mean,
                                sd,
                                standardize,
                                centroids,
                                kc,
                                openData,
                                openEntry,
                                log);
                    } finally {
                        progress.accept(done.incrementAndGet() / (double) nImages);
                    }
                }));
            }

            long[] cnCounts = new long[kc];
            double[][] cnMeanSum = new double[kc][nTypes];
            long assigned = 0;
            long empty = 0;
            for (Future<ImageAssign> f : futures) {
                ImageAssign r;
                try {
                    r = f.get();
                } catch (Exception e) {
                    continue;
                }
                if (r == null) {
                    continue;
                }
                assigned += r.assigned();
                empty += r.empty();
                for (int c = 0; c < kc; c++) {
                    cnCounts[c] += r.cnCounts()[c];
                    for (int j = 0; j < nTypes; j++) {
                        cnMeanSum[c][j] += r.cnMeanSum()[c][j];
                    }
                }
            }

            double[][] cnMean = new double[kc][nTypes];
            for (int c = 0; c < kc; c++) {
                if (cnCounts[c] > 0) {
                    for (int j = 0; j < nTypes; j++) {
                        cnMean[c][j] = cnMeanSum[c][j] / cnCounts[c];
                    }
                }
            }
            return new AssignResult(assigned, empty, cnCounts, cnMean);
        } finally {
            pool.shutdownNow();
        }
    }

    /** Classify every cell in one image to its nearest centroid, write the CN measurement, and save. */
    private static ImageAssign assignOneImage(
            ProjectImageEntry<BufferedImage> entry,
            Map<String, Integer> typeIndex,
            int nTypes,
            Params params,
            double[] mean,
            double[] sd,
            boolean standardize,
            double[][] centroids,
            int kc,
            ImageData<BufferedImage> openData,
            ProjectImageEntry<BufferedImage> openEntry,
            Consumer<String> log) {
        long[] cnCounts = new long[kc];
        double[][] cnMeanSum = new double[kc][nTypes];
        if (entry == null) {
            return new ImageAssign(0, 0, cnCounts, cnMeanSum);
        }
        String name = entry.getImageName();
        boolean isOpen = openEntry != null && entry.equals(openEntry) && openData != null;
        long tRead0 = System.nanoTime();
        ImageData<BufferedImage> imageData;
        try {
            imageData = isOpen ? openData : entry.readImageData();
        } catch (Exception e) {
            log.accept("[" + name + "] could not load — skipped");
            return new ImageAssign(0, 0, cnCounts, cnMeanSum);
        }
        if (imageData == null) {
            return new ImageAssign(0, 0, cnCounts, cnMeanSum);
        }
        long tRead = System.nanoTime() - tRead0;

        long tBuild0 = System.nanoTime();
        ImageComp ic = build(imageData, typeIndex, nTypes, params);
        long tBuild = System.nanoTime() - tBuild0;
        int n = ic.cells().size();
        double[] cnValue = new double[n];
        double[] z = new double[nTypes];
        long empty = 0;
        int imgAssigned = 0;
        for (int i = 0; i < n; i++) {
            double[] row = ic.comp()[i];
            double sum = 0;
            for (double v : row) {
                sum += v;
            }
            if (sum <= 0) {
                cnValue[i] = -1;
                empty++;
                continue;
            }
            if (standardize) {
                for (int j = 0; j < nTypes; j++) {
                    z[j] = sd[j] < 1e-9 ? 0.0 : (row[j] - mean[j]) / sd[j];
                }
            } else {
                System.arraycopy(row, 0, z, 0, nTypes);
            }
            int c = CohortClusterModel.nearestCentroid(z, centroids);
            cnValue[i] = c + 1.0;
            cnCounts[c]++;
            for (int j = 0; j < nTypes; j++) {
                cnMeanSum[c][j] += row[j];
            }
            imgAssigned++;
        }

        long tWrite0 = System.nanoTime();
        applyMeasurements(imageData, ic.cells(), cnValue, isOpen);
        long tWrite = System.nanoTime() - tWrite0;
        long tSave0 = System.nanoTime();
        try {
            entry.saveImageData(imageData);
        } catch (Exception e) {
            logger.error("Failed to save {}", name, e);
            log.accept("[" + name + "] save failed: " + e.getMessage());
        }
        long tSave = System.nanoTime() - tSave0;
        log.accept(String.format("[%s] CN assigned to %,d of %,d cells", name, imgAssigned, n));
        log.accept(String.format(
                "[%s] timing: read %.1fs · knn/build %.1fs · write %.1fs · save %.1fs",
                name, tRead / 1e9, tBuild / 1e9, tWrite / 1e9, tSave / 1e9));
        return new ImageAssign(imgAssigned, empty, cnCounts, cnMeanSum);
    }

    // ── Pass 3: apply names (cohort-wide) ─────────────────────────────────────────

    /**
     * Write the user's CN naming/merge to every cell of the given {@code images}, in parallel.
     * Each cell's already-saved {@code CN} id (1-based; {@code -1} = empty window) is mapped through
     * {@code nameByCn} / {@code codeByCn} to a name string (written to metadata key {@code nameKey}) and
     * a numeric code (written to measurement {@code codeKey}); empty/unknown cells get
     * {@code unassignedName} / {@code -1}. Each image is read, updated, and saved. Non-destructive —
     * the cell phenotype ({@code getPathClass()}) is untouched.
     *
     * @return the total number of cells updated across all images
     */
    public static long applyNamesAcrossProject(
            List<ProjectImageEntry<BufferedImage>> entries,
            Map<Integer, String> nameByCn,
            Map<Integer, Integer> codeByCn,
            String unassignedName,
            String codeKey,
            String nameKey,
            int workers,
            Consumer<String> log,
            DoubleConsumer progress) {
        int nImages = entries.size();
        int parallelism = Math.max(1, Math.min(workers, Math.max(1, nImages)));
        ExecutorService pool = BackgroundExecutors.newFixedPool(parallelism, "SpClassify-CN-Naming");
        AtomicInteger done = new AtomicInteger(0);
        AtomicLong updated = new AtomicLong(0);
        try {
            List<Future<?>> futures = new ArrayList<>();
            for (ProjectImageEntry<BufferedImage> entry : entries) {
                futures.add(pool.submit(() -> {
                    try {
                        updated.addAndGet(applyNamesOneImage(
                                entry,
                                entry.getImageName(),
                                nameByCn,
                                codeByCn,
                                unassignedName,
                                codeKey,
                                nameKey,
                                log));
                    } finally {
                        progress.accept(nImages == 0 ? 1.0 : done.incrementAndGet() / (double) nImages);
                    }
                }));
            }
            for (Future<?> f : futures) {
                try {
                    f.get();
                } catch (Exception e) {
                    // per-image failures are already logged inside the task
                }
            }
            return updated.get();
        } finally {
            pool.shutdownNow();
        }
    }

    /** Map every cell's {@code CN} id to a name/code and write both, then save the image. */
    private static long applyNamesOneImage(
            ProjectImageEntry<BufferedImage> entry,
            String name,
            Map<Integer, String> nameByCn,
            Map<Integer, Integer> codeByCn,
            String unassignedName,
            String codeKey,
            String nameKey,
            Consumer<String> log) {
        if (entry == null) {
            log.accept("[" + name + "] not in project — skipped");
            return 0;
        }
        ImageData<BufferedImage> imageData;
        try {
            imageData = entry.readImageData();
        } catch (Exception e) {
            log.accept("[" + name + "] could not load — skipped");
            return 0;
        }
        if (imageData == null) {
            return 0;
        }
        var hierarchy = imageData.getHierarchy();
        var cellsCol = hierarchy.getCellObjects();
        List<PathObject> cells = new ArrayList<>(cellsCol.isEmpty() ? hierarchy.getDetectionObjects() : cellsCol);

        long updated = 0;
        for (PathObject cell : cells) {
            double cnVal = cell.getMeasurementList().get(CN_MEASUREMENT);
            int code = -1;
            String nm = unassignedName;
            if (!Double.isNaN(cnVal) && cnVal >= 1) {
                int id = (int) Math.round(cnVal);
                String n = nameByCn.get(id);
                if (n != null) {
                    nm = n;
                }
                Integer c = codeByCn.get(id);
                code = c != null ? c : -1;
            }
            cell.getMeasurementList().put(codeKey, code);
            cell.getMetadata().put(nameKey, nm);
            updated++;
        }

        try {
            entry.saveImageData(imageData);
        } catch (Exception e) {
            logger.error("Failed to save {}", name, e);
            log.accept("[" + name + "] save failed: " + e.getMessage());
        }
        log.accept(String.format("[%s] CN Class written to %,d cells", name, updated));
        return updated;
    }

    // ── Shared helpers ──────────────────────────────────────────────────────────

    /** Build the [n][nTypes] raw composition matrix and aligned cell list for one image. */
    private static ImageComp build(ImageData<?> imageData, Map<String, Integer> typeIndex, int nTypes, Params params) {
        var hierarchy = imageData.getHierarchy();
        var cellsCol = hierarchy.getCellObjects();
        List<PathObject> cells = new ArrayList<>(cellsCol.isEmpty() ? hierarchy.getDetectionObjects() : cellsCol);
        int n = cells.size();

        var cal = imageData.getServer().getPixelCalibration();
        boolean calibrated = cal.hasPixelSizeMicrons();
        double pw = params.pixelOverride() != null
                ? params.pixelOverride()
                : (calibrated ? cal.getPixelWidthMicrons() : 1.0);
        double ph = params.pixelOverride() != null
                ? params.pixelOverride()
                : (calibrated ? cal.getPixelHeightMicrons() : 1.0);

        double[] xs = new double[n];
        double[] ys = new double[n];
        int[] typeId = new int[n];
        for (int i = 0; i < n; i++) {
            var roi = cells.get(i).getROI();
            if (roi == null) {
                xs[i] = Double.NaN;
                ys[i] = Double.NaN;
            } else {
                xs[i] = roi.getCentroidX() * pw;
                ys[i] = roi.getCentroidY() * ph;
            }
            PathClass pc = cells.get(i).getPathClass();
            Integer idx = (pc != null && pc.isValid() && !PathClassTools.isIgnoredClass(pc))
                    ? typeIndex.get(pc.toString())
                    : null;
            typeId[i] = idx != null ? idx : -1;
        }

        int[][] neighbors =
                switch (params.mode()) {
                    case KNN -> NeighborhoodModel.kNearestNeighborIndices(xs, ys, params.k());
                    case RADIUS -> NeighborhoodModel.radiusNeighborIndices(xs, ys, params.radius());
                    case DELAUNAY ->
                        Double.isNaN(params.delaunayMaxEdge())
                                ? NeighborhoodModel.delaunayNeighborIndicesAuto(xs, ys)
                                : NeighborhoodModel.delaunayNeighborIndices(xs, ys, params.delaunayMaxEdge());
                };
        double[][] comp = NeighborhoodModel.compositionMatrix(neighbors, typeId, nTypes, params.includeCenter());
        return new ImageComp(cells, comp);
    }

    private static List<Integer> activeRows(double[][] comp) {
        List<Integer> active = new ArrayList<>();
        for (int i = 0; i < comp.length; i++) {
            double sum = 0;
            for (double v : comp[i]) {
                sum += v;
            }
            if (sum > 0) {
                active.add(i);
            }
        }
        return active;
    }

    /** Writes the "CN" measurement; for the open image this marshals to the FX thread. */
    private static void applyMeasurements(
            ImageData<?> imageData, List<PathObject> cells, double[] values, boolean isOpen) {
        Runnable apply = () -> {
            for (int i = 0; i < cells.size(); i++) {
                cells.get(i).getMeasurementList().put(CN_MEASUREMENT, values[i]);
            }
            imageData.getHierarchy().fireHierarchyChangedEvent(NeighborhoodCohort.class);
        };
        if (isOpen && !Platform.isFxApplicationThread()) {
            var latch = new java.util.concurrent.CountDownLatch(1);
            Platform.runLater(() -> {
                try {
                    apply.run();
                } finally {
                    latch.countDown();
                }
            });
            try {
                latch.await();
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
        } else {
            apply.run();
        }
    }

    private static Map<String, Integer> typeIndex(List<String> typeNames) {
        Map<String, Integer> idx = new LinkedHashMap<>();
        for (String t : typeNames) {
            idx.put(t, idx.size());
        }
        return idx;
    }
}
