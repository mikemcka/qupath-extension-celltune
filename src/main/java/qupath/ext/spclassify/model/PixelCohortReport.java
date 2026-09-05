package qupath.ext.spclassify.model;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Result of contextualising per-image {@link ImagePixelStats.ImageStats} against
 * the whole project: robust z-scores, percentile ranks, outlier flags, and a
 * plain-English review per image. Produced by {@link PixelCohortAnalyzer}.
 * <p>
 * This is the pixel-level analogue of {@link CohortAnomalyReport}: the same
 * MAD-based robust-z machinery, applied to pixel-intensity statistics instead of
 * cell-class composition.
 *
 * @param images per-image reports, sorted worst-first by {@link ImageReport#score()}
 */
public record PixelCohortReport(List<ImageReport> images) {

    // ── Reason codes (stable identifiers used in flags, CSV, and tests) ──────────
    public static final String BACKGROUND_HEAVY = "BACKGROUND_HEAVY";
    public static final String SATURATED = "SATURATED";
    public static final String WEAK_SIGNAL = "WEAK_SIGNAL";
    public static final String INTENSITY_OUTLIER = "INTENSITY_OUTLIER";

    /** Human-readable verdict shown when an image trips no flags. */
    public static final String VERDICT_OK = "OK";

    public PixelCohortReport {
        images = images == null ? List.of() : List.copyOf(images);
    }

    /** @return reports keyed by image name (first occurrence wins on duplicates). */
    public Map<String, ImageReport> byImageName() {
        var map = new LinkedHashMap<String, ImageReport>(images.size());
        for (var r : images) {
            map.putIfAbsent(r.imageName(), r);
        }
        return map;
    }

    /**
     * Tunable cutoffs for the verdict rules. All z-score cutoffs are in robust
     * (MAD-scaled) standard deviations.
     *
     * @param backgroundForegroundZ   flag BACKGROUND_HEAVY when mean foreground
     *                                coverage z ≤ −this (much less signal than peers)
     * @param backgroundEmptyZ        flag BACKGROUND_HEAVY when empty-fraction z ≥ this
     * @param saturationZ             flag SATURATED when max saturation-fraction z ≥ this
     *                                (cohort-relative path) …
     * @param saturationMinFraction   …and the max saturation fraction is ≥ this absolute value
     * @param saturationHardFraction  always flag SATURATED when the max saturation
     *                                fraction is ≥ this, regardless of the cohort
     *                                (clipping this severe is a defect on its own)
     * @param weakSignalZ             flag WEAK_SIGNAL when median dynamic-range z ≤ −this
     * @param intensityOutlierZ       flag INTENSITY_OUTLIER when a signal-bearing
     *                                channel's p99 (brightness) z magnitude ≥ this
     */
    public record Thresholds(
            double backgroundForegroundZ,
            double backgroundEmptyZ,
            double saturationZ,
            double saturationMinFraction,
            double saturationHardFraction,
            double weakSignalZ,
            double intensityOutlierZ) {}

    /** Default thresholds (the values agreed for the first cut). */
    public static final Thresholds DEFAULT_THRESHOLDS = new Thresholds(
            2.5, // backgroundForegroundZ
            2.5, // backgroundEmptyZ
            3.0, // saturationZ
            0.01, // saturationMinFraction (1% of pixels clipped)
            0.05, // saturationHardFraction (5% clipped → always flag)
            2.5, // weakSignalZ
            2.5 // intensityOutlierZ
            );

    /**
     * Per-channel context for one image: the raw per-channel statistics, carried
     * through for display and CSV export.
     *
     * @param channel channel name
     * @param stats   the raw per-channel statistics
     */
    public record ChannelContext(String channel, ImagePixelStats.ChannelStats stats) {}

    /**
     * Whole-image report with verdict, flags, the cohort-relative narrative, and
     * image-level summary scalars plus their z-scores.
     *
     * @param imageName                 image name
     * @param verdict                   short human verdict (e.g. "OK", "Background-heavy")
     * @param flags                     reason codes that fired (may be empty)
     * @param score                     sort key; higher = more anomalous
     * @param narrative                 multi-line plain-English review
     * @param emptyFraction             fraction background in all channels
     * @param emptyFractionZ            robust z of empty fraction vs cohort
     * @param meanForegroundCoverage    mean foreground coverage across channels
     * @param meanForegroundCoverageZ   robust z vs cohort
     * @param maxSaturationFraction     max saturation fraction across channels
     * @param maxSaturationChannel      channel carrying that max (may be null)
     * @param maxSaturationFractionZ    robust z vs cohort
     * @param medianDynamicRange        median dynamic range across channels
     * @param medianDynamicRangeZ       robust z vs cohort
     * @param maxFocus                  image focus: max per-channel Laplacian variance
     *                                  (informational sharpness proxy; not flagged —
     *                                  it tracks brightness as much as focus)
     * @param maxFocusZ                 robust z of {@link #maxFocus} vs cohort
     * @param maxIntensityZ             largest signal-bearing channel p99 z magnitude
     *                                  vs cohort (drives the intensity-outlier flag)
     * @param maxIntensityChannel       channel carrying that largest p99 z (may be null)
     * @param channels                  per-channel context, channel order
     */
    public record ImageReport(
            String imageName,
            String verdict,
            List<String> flags,
            double score,
            String narrative,
            double emptyFraction,
            double emptyFractionZ,
            double meanForegroundCoverage,
            double meanForegroundCoverageZ,
            double maxSaturationFraction,
            String maxSaturationChannel,
            double maxSaturationFractionZ,
            double medianDynamicRange,
            double medianDynamicRangeZ,
            double maxFocus,
            double maxFocusZ,
            double maxIntensityZ,
            String maxIntensityChannel,
            List<ChannelContext> channels) {

        public ImageReport {
            imageName = imageName == null ? "" : imageName;
            verdict = verdict == null ? VERDICT_OK : verdict;
            flags = flags == null ? List.of() : List.copyOf(flags);
            channels = channels == null ? List.of() : List.copyOf(channels);
        }

        /** @return {@code true} if any flag fired. */
        public boolean flagged() {
            return !flags.isEmpty();
        }
    }
}
