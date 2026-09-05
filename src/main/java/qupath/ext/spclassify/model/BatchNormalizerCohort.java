package qupath.ext.spclassify.model;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.DoubleConsumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.model.CohortClusterModel.CancellationToken;
import qupath.ext.spclassify.util.BackgroundExecutors;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.projects.ProjectImageEntry;

/**
 * Headless, memory-safe backend for the project-wide <b>UniFORM batch-normalization</b>
 * FIT — the cohort analogue of the pure {@link BatchNormalizerModel} math. It streams
 * every selected image once (one worker per image), sub-samples cells, reads the selected
 * intensity measurements, and learns one multiplicative <b>per-{@code (image, channel)}
 * gain</b> — fit on a robust anchor stat (Cell: Mean) and applied to every selected stat of
 * that channel — aligning each image's (or batch's) intensity distribution to an auto-chosen
 * reference. Every intensity statistic of a channel (Mean/Median/percentiles/Std/erosion/
 * expansion/environment) scales by the same gain, so this keeps them mutually consistent.
 *
 * <p>Mirrors {@code NeighborhoodCohort}'s streaming/parallel idioms (multi-project
 * {@code List<ProjectImageEntry>}, per-image workers via {@link BackgroundExecutors},
 * {@code entry.equals(openEntry)} identity match to reuse the open image, {@link CancellationToken},
 * {@code DoubleConsumer} progress, deterministic post-join merge). Only intensity
 * measurements the caller selects are touched — foundation-model embeddings are never
 * passed in, because the log-space correction is invalid for them.
 *
 * <p>The returned {@link NormalizerFit} retains per-image histograms and integer bin
 * shifts (≈{@code nImages·nMarkers·nBins} longs, ~16&nbsp;MB at 60×34×1024) so the QC view
 * can render before/after distributions by shifting histograms — no raw values are kept.
 * The applying pass (write corrected measurements) lives in {@code BatchNormalizerApply}.
 */
public final class BatchNormalizerCohort {

    private static final Logger logger = LoggerFactory.getLogger(BatchNormalizerCohort.class);

    private BatchNormalizerCohort() {}

    /** Whether each image is aligned individually, or images are pooled and aligned per batch. */
    public enum Mode {
        PER_IMAGE,
        PER_BATCH
    }

    /**
     * Fit configuration shared across the streaming pass.
     *
     * @param mode            per-image (each image aligned) or per-batch (batch-pooled)
     * @param nBins           log-intensity histogram bins ({@link BatchNormalizerModel#DEFAULT_BINS})
     * @param markerFeatures  the intensity measurement names to correct (embeddings excluded upstream)
     * @param imageToBatch    image-name → batch label (used for PER_BATCH grouping and QC)
     * @param perImageCap     max cells sampled per image for the fit (bounds memory; histograms are stable well below the total)
     * @param referenceOverride optional group label (image or batch) to force as reference; {@code null} = auto (L2-closest-to-mean)
     */
    public record Params(
            Mode mode,
            int nBins,
            List<String> markerFeatures,
            Map<String, String> imageToBatch,
            int perImageCap,
            String referenceOverride) {}

    /**
     * Learned normalization.
     *
     * @param markers        selected marker feature names, in column order
     * @param scaleByImage   image-name → per-marker multiplicative scale ({@code corrected = raw·scale})
     * @param shiftByImage   image-name → per-marker integer bin shift (for QC "after" histograms)
     * @param histByImage    image-name → per-marker BEFORE histogram ({@code [marker][bin]})
     * @param logMin/logMax  per-marker global log-intensity range (histogram edges; also drives shift→scale)
     * @param refGroupByMarker per-marker reference group label (image or batch)
     * @param imageToBatch   the resolved image→batch map (echoed for QC/persistence)
     * @param mode/nBins     echoed configuration
     */
    public record NormalizerFit(
            List<String> markers,
            Map<String, double[]> scaleByImage,
            Map<String, int[]> shiftByImage,
            Map<String, long[][]> histByImage,
            double[] logMin,
            double[] logMax,
            String[] refGroupByMarker,
            Map<String, String> imageToBatch,
            Mode mode,
            int nBins) {}

    /** One image's sub-sampled raw marker values: {@code values[marker][cell]}. */
    private record ImageValues(String name, String batch, double[][] values, int nCells) {}

    /**
     * Learn per-(image, marker) correction scales across {@code entries}. Streams each image
     * once in parallel, then computes histograms/shifts/scales on the pooled per-image samples.
     * Returns {@code null} if cancelled or no usable data.
     */
    public static NormalizerFit fit(
            List<ProjectImageEntry<BufferedImage>> entries,
            Params params,
            ImageData<BufferedImage> openData,
            ProjectImageEntry<BufferedImage> openEntry,
            int workers,
            Consumer<String> log,
            DoubleConsumer progress,
            CancellationToken token) {

        List<String> markers = params.markerFeatures();
        int nMarkers = markers.size();
        int nBins = params.nBins();
        int nImages = entries.size();
        if (nMarkers == 0 || nImages == 0) {
            log.accept("Nothing to fit (no markers or no images).");
            return null;
        }

        // ── Pass 1: parallel per-image read + subsample + extract marker values ──
        int parallelism = Math.max(1, Math.min(workers, nImages));
        ExecutorService pool = BackgroundExecutors.newFixedPool(parallelism, "SpClassify-BatchNorm-Fit");
        AtomicInteger done = new AtomicInteger(0);
        Map<String, ImageValues> byImage = new LinkedHashMap<>();
        try {
            List<Future<ImageValues>> futures = new ArrayList<>();
            for (int idx = 0; idx < nImages; idx++) {
                ProjectImageEntry<BufferedImage> entry = entries.get(idx);
                long seed = 42L + idx;
                futures.add(pool.submit(() -> {
                    try {
                        if (token != null && token.isCancelled()) {
                            return null;
                        }
                        return readOneImage(entry, params, openData, openEntry, seed, log);
                    } finally {
                        progress.accept(done.incrementAndGet() / (double) nImages);
                    }
                }));
            }
            for (Future<ImageValues> f : futures) {
                ImageValues v;
                try {
                    v = f.get();
                } catch (Exception e) {
                    continue;
                }
                if (v != null) {
                    byImage.put(v.name(), v);
                }
            }
        } finally {
            pool.shutdownNow();
        }
        if (token != null && token.isCancelled()) {
            return null;
        }
        if (byImage.isEmpty()) {
            log.accept("No image yielded usable cells — nothing to fit.");
            return null;
        }

        // ── Per-marker global log range across all images ──
        double[] logMin = new double[nMarkers];
        double[] logMax = new double[nMarkers];
        for (int m = 0; m < nMarkers; m++) {
            double lo = Double.POSITIVE_INFINITY;
            double hi = Double.NEGATIVE_INFINITY;
            for (ImageValues v : byImage.values()) {
                double[] mm = BatchNormalizerModel.logMinMax(v.values()[m]);
                if (!Double.isNaN(mm[0])) {
                    lo = Math.min(lo, mm[0]);
                    hi = Math.max(hi, mm[1]);
                }
            }
            logMin[m] = Double.isInfinite(lo) ? 0.0 : lo;
            logMax[m] = Double.isInfinite(hi) ? 0.0 : hi;
        }

        // ── Per-image histograms (retained for QC), then free the raw values ──
        Map<String, long[][]> histByImage = new LinkedHashMap<>();
        for (ImageValues v : byImage.values()) {
            long[][] h = new long[nMarkers][];
            for (int m = 0; m < nMarkers; m++) {
                h[m] = BatchNormalizerModel.logHistogram(v.values()[m], logMin[m], logMax[m], nBins);
            }
            histByImage.put(v.name(), h);
        }
        // Group label per image (batch for PER_BATCH, image name for PER_IMAGE).
        Map<String, String> groupOf = new LinkedHashMap<>();
        for (ImageValues v : byImage.values()) {
            groupOf.put(v.name(), params.mode() == Mode.PER_BATCH ? v.batch() : v.name());
        }
        byImage.clear(); // release raw values

        // ── Per-channel gain: ONE scale per (image, channel), fit on an anchor stat ──
        // A batch effect is a multiplicative gain per channel; every intensity stat of that
        // channel (Mean/Median/percentiles/Std/erosion/expansion/environment) scales by the
        // same factor. So we group selected measurements by channel (token before the first
        // ": "), fit the gain on a robust anchor (Cell: Mean if selected, else Cell: Median,
        // else the first stat), and apply that one scale to every selected stat of the channel.
        String[] channelOf = new String[nMarkers];
        Map<String, List<Integer>> byChannel = new LinkedHashMap<>();
        for (int m = 0; m < nMarkers; m++) {
            channelOf[m] = channelOf(markers.get(m));
            byChannel.computeIfAbsent(channelOf[m], k -> new ArrayList<>()).add(m);
        }
        List<String> groups = new ArrayList<>(new TreeSet<>(groupOf.values()));
        Map<String, Integer> groupIndex = new LinkedHashMap<>();
        for (int i = 0; i < groups.size(); i++) {
            groupIndex.put(groups.get(i), i);
        }

        Map<String, String> refGroupByChannel = new LinkedHashMap<>();
        Map<String, double[]> channelGroupScale = new LinkedHashMap<>(); // channel → per-group scale
        for (Map.Entry<String, List<Integer>> e : byChannel.entrySet()) {
            int a = pickAnchor(e.getValue(), markers);
            long[][] anchorGroupHist = new long[groups.size()][nBins];
            for (Map.Entry<String, long[][]> he : histByImage.entrySet()) {
                long[] ah = he.getValue()[a];
                long[] acc = anchorGroupHist[groupIndex.get(groupOf.get(he.getKey()))];
                for (int b = 0; b < nBins; b++) {
                    acc[b] += ah[b];
                }
            }
            int refIdx = referenceIndex(groups, anchorGroupHist, params.referenceOverride());
            refGroupByChannel.put(
                    e.getKey(), refIdx >= 0 ? groups.get(refIdx) : (groups.isEmpty() ? null : groups.get(0)));
            long[] ref = anchorGroupHist[Math.max(0, refIdx)];
            double[] gScale = new double[groups.size()];
            for (int i = 0; i < groups.size(); i++) {
                int shift = BatchNormalizerModel.crossCorrelationShift(anchorGroupHist[i], ref);
                gScale[i] = BatchNormalizerModel.shiftToScale(shift, logMin[a], logMax[a], nBins);
            }
            channelGroupScale.put(e.getKey(), gScale);
        }

        // ── Map each image → per-measurement scale (its channel's) + a QC bin shift ──
        String[] refGroupByMarker = new String[nMarkers];
        for (int m = 0; m < nMarkers; m++) {
            refGroupByMarker[m] = refGroupByChannel.get(channelOf[m]);
        }
        Map<String, int[]> shiftByImage = new LinkedHashMap<>();
        Map<String, double[]> scaleByImage = new LinkedHashMap<>();
        for (String image : histByImage.keySet()) {
            int gi = groupIndex.get(groupOf.get(image));
            int[] shift = new int[nMarkers];
            double[] scale = new double[nMarkers];
            for (int m = 0; m < nMarkers; m++) {
                double s = channelGroupScale.get(channelOf[m])[gi];
                scale[m] = s;
                // QC "after" shift, expressed in THIS measurement's own histogram bins so a
                // channel gain renders correctly on stats with different intensity ranges.
                double binWidth = nBins > 1 && logMax[m] > logMin[m] ? (logMax[m] - logMin[m]) / (nBins - 1) : 0.0;
                shift[m] = (binWidth > 0 && s > 0) ? (int) Math.round(-Math.log(s) / binWidth) : 0;
            }
            shiftByImage.put(image, shift);
            scaleByImage.put(image, scale);
        }

        log.accept(String.format(
                "Fitted %d channel gain(s) over %d measurement(s), %d image(s), %d group(s) (%s).",
                byChannel.size(), nMarkers, histByImage.size(), groups.size(), params.mode()));
        return new NormalizerFit(
                new ArrayList<>(markers),
                scaleByImage,
                shiftByImage,
                histByImage,
                logMin,
                logMax,
                refGroupByMarker,
                new LinkedHashMap<>(params.imageToBatch()),
                params.mode(),
                nBins);
    }

    /** Read + subsample one image, extracting the selected marker measurements as raw values. */
    private static ImageValues readOneImage(
            ProjectImageEntry<BufferedImage> entry,
            Params params,
            ImageData<BufferedImage> openData,
            ProjectImageEntry<BufferedImage> openEntry,
            long seed,
            Consumer<String> log) {
        if (entry == null) {
            return null;
        }
        String name = entry.getImageName();
        boolean isOpen = openEntry != null && entry.equals(openEntry) && openData != null;
        ImageData<BufferedImage> imageData;
        try {
            imageData = isOpen ? openData : entry.readImageData();
        } catch (Exception e) {
            log.accept("[" + name + "] could not load — skipped");
            return null;
        }
        if (imageData == null) {
            return null;
        }
        var hierarchy = imageData.getHierarchy();
        var cellCol = hierarchy.getCellObjects();
        List<PathObject> cells = new ArrayList<>(cellCol.isEmpty() ? hierarchy.getDetectionObjects() : cellCol);
        if (cells.isEmpty()) {
            log.accept("[" + name + "] no cells — skipped");
            return null;
        }
        int take = Math.min(params.perImageCap(), cells.size());
        int[] pick = CohortClusterModel.sampleIndices(cells.size(), take, new Random(seed));

        List<String> markers = params.markerFeatures();
        int nMarkers = markers.size();
        double[][] values = new double[nMarkers][take];
        for (int t = 0; t < take; t++) {
            var ml = cells.get(pick[t]).getMeasurementList();
            for (int m = 0; m < nMarkers; m++) {
                values[m][t] = ml.get(markers.get(m));
            }
        }
        String batch = params.imageToBatch().getOrDefault(name, "(unassigned)");
        log.accept(String.format("[%s] sampled %,d of %,d cells [batch=%s]", name, take, cells.size(), batch));
        return new ImageValues(name, batch, values, cells.size());
    }

    /** Channel token of a measurement name: the part before the first {@code ": "} (or the whole name). */
    static String channelOf(String feature) {
        int i = feature.indexOf(": ");
        return i > 0 ? feature.substring(0, i) : feature;
    }

    /** Anchor stat for a channel's gain fit: prefer {@code Cell: Mean}, then {@code Cell: Median}, else the first. */
    private static int pickAnchor(List<Integer> indices, List<String> markers) {
        int median = -1;
        for (int i : indices) {
            String n = markers.get(i);
            if (n.endsWith(": Cell: Mean")) {
                return i;
            }
            if (median < 0 && n.endsWith(": Cell: Median")) {
                median = i;
            }
        }
        return median >= 0 ? median : indices.get(0);
    }

    /** Reference group index: the override's index if supplied and present, else auto (L2-closest-to-mean). */
    private static int referenceIndex(List<String> groups, long[][] perGroup, String override) {
        if (override != null) {
            int i = groups.indexOf(override);
            if (i >= 0) {
                return i;
            }
        }
        return BatchNormalizerModel.autoReferenceIndex(perGroup);
    }
}
