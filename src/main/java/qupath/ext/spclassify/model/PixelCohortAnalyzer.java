package qupath.ext.spclassify.model;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import qupath.ext.spclassify.util.RobustStats;

/**
 * Contextualises whole-image pixel statistics against the whole project.
 * <p>
 * Part of the image pixel prescreen, whose statistics methodology was adapted from
 * <a href="https://pypi.org/project/pixel-patrol/">pixel-patrol</a> (MIT) — see
 * {@link ImagePixelStats} for the attribution note.
 * <p>
 * Given one {@link ImagePixelStats.ImageStats} per project image, this computes a
 * handful of robust (MAD-scaled) z-scores for image-level summary scalars
 * (empty/background fraction, mean foreground coverage, peak saturation, median
 * dynamic range), then applies deterministic threshold rules to assign each image
 * a verdict, a set of flag reason codes, and a plain-English review.
 * <p>
 * These four image-level scalars have genuine cohort spread, so the robust z is
 * well-behaved. Per-channel intensity-outlier detection was intentionally dropped:
 * a MAD-scaled z on a near-dead channel (whose medians all hover at the noise
 * floor) explodes tiny, meaningless differences into huge z-scores and false
 * "intensity outlier" verdicts. The raw per-channel statistics are still carried
 * through for display and CSV export, but no longer drive any verdict.
 * <p>
 * The robust-z definition is identical to {@link CohortAnomalyAnalyzer}:
 * {@code z = 0.6745 · (x − median) / MAD}. Like that class, this one is free of
 * any QuPath/JavaFX dependency and is unit-tested in isolation.
 */
public final class PixelCohortAnalyzer {

    /**
     * A channel takes part in intensity-outlier detection only if its cohort-median
     * foreground coverage clears this floor. Near-dead / mostly-background markers
     * (whose p99 hovers at the noise floor) are excluded, so their meaningless
     * relative jitter cannot manufacture spurious z-scores — the failure mode that
     * sank the original per-channel median z.
     */
    private static final double SIGNAL_FOREGROUND_FLOOR = 0.05;

    private PixelCohortAnalyzer() {}

    /** Analyse with {@link PixelCohortReport#DEFAULT_THRESHOLDS}. */
    public static PixelCohortReport analyze(List<ImagePixelStats.ImageStats> inputs) {
        return analyze(inputs, PixelCohortReport.DEFAULT_THRESHOLDS);
    }

    /**
     * Analyse a cohort of per-image pixel statistics.
     *
     * @param inputs     per-image statistics (nulls are skipped)
     * @param thresholds verdict cutoffs; {@code null} uses the defaults
     * @return the cohort report, images sorted worst-first
     */
    public static PixelCohortReport analyze(
            List<ImagePixelStats.ImageStats> inputs, PixelCohortReport.Thresholds thresholds) {

        if (thresholds == null) {
            thresholds = PixelCohortReport.DEFAULT_THRESHOLDS;
        }

        var safe = new ArrayList<ImagePixelStats.ImageStats>();
        if (inputs != null) {
            for (var in : inputs) {
                if (in != null) {
                    safe.add(in);
                }
            }
        }
        if (safe.isEmpty()) {
            return new PixelCohortReport(List.of());
        }

        int nImages = safe.size();

        // ── Image-level summary scalars ─────────────────────────────────────────
        double[] emptyFraction = new double[nImages];
        double[] meanForeground = new double[nImages];
        double[] maxSaturation = new double[nImages];
        String[] maxSaturationChannel = new String[nImages];
        double[] medianDynamicRange = new double[nImages];
        double[] maxFocus = new double[nImages];

        for (int i = 0; i < nImages; i++) {
            var img = safe.get(i);
            emptyFraction[i] = img.emptyFraction();

            var fgList = new ArrayList<Double>();
            var dynList = new ArrayList<Double>();
            double bestSat = Double.NaN;
            String bestSatCh = null;
            double bestFocus = Double.NaN;
            for (var cs : img.channels()) {
                if (!Double.isNaN(cs.foregroundCoverage())) {
                    fgList.add(cs.foregroundCoverage());
                }
                if (!Double.isNaN(cs.dynamicRange())) {
                    dynList.add(cs.dynamicRange());
                }
                double sat = cs.saturationFraction();
                if (!Double.isNaN(sat) && (Double.isNaN(bestSat) || sat > bestSat)) {
                    bestSat = sat;
                    bestSatCh = cs.channel();
                }
                // Image focus = the sharpest channel. The brightest/most-structured
                // channel dominates; near-dead channels have tiny Laplacian variance
                // and never win, so a blurred image's max still drops cohort-wide.
                double focus = cs.laplacianVariance();
                if (!Double.isNaN(focus) && (Double.isNaN(bestFocus) || focus > bestFocus)) {
                    bestFocus = focus;
                }
            }
            meanForeground[i] = mean(fgList);
            medianDynamicRange[i] = medianOf(dynList);
            maxSaturation[i] = bestSat;
            maxSaturationChannel[i] = bestSatCh;
            maxFocus[i] = bestFocus;
        }

        double[] emptyFractionZ = robustZ(emptyFraction);
        double[] meanForegroundZ = robustZ(meanForeground);
        double[] maxSaturationZ = robustZ(maxSaturation);
        double[] medianDynamicRangeZ = robustZ(medianDynamicRange);
        double[] maxFocusZ = robustZ(maxFocus);

        double medEmptyFraction = medianIgnoreNaN(emptyFraction);
        double medMeanForeground = medianIgnoreNaN(meanForeground);
        double medMaxSaturation = medianIgnoreNaN(maxSaturation);
        double medMedianDynamicRange = medianIgnoreNaN(medianDynamicRange);
        double medMaxFocus = medianIgnoreNaN(maxFocus);

        // ── Per-channel p99 (brightness) z, on signal-bearing channels only ──────
        // This is the intensity-outlier signal: a slide whose brightness profile
        // diverges from the cohort is a likely ML challenge. Restricting to
        // signal-bearing channels keeps near-dead markers from exploding the z.
        var channelOrder = new LinkedHashSet<String>();
        for (var img : safe) {
            for (var cs : img.channels()) {
                channelOrder.add(cs.channel());
            }
        }
        var p99ZByChannel = new LinkedHashMap<String, double[]>();
        var medP99ByChannel = new LinkedHashMap<String, Double>();
        for (String ch : channelOrder) {
            double[] p99 = new double[nImages];
            double[] fg = new double[nImages];
            for (int i = 0; i < nImages; i++) {
                var cs = channelOf(safe.get(i), ch);
                p99[i] = cs == null ? Double.NaN : cs.p99();
                fg[i] = cs == null ? Double.NaN : cs.foregroundCoverage();
            }
            double medFg = medianIgnoreNaN(fg);
            if (!Double.isNaN(medFg) && medFg >= SIGNAL_FOREGROUND_FLOOR) {
                p99ZByChannel.put(ch, robustZ(p99));
                medP99ByChannel.put(ch, medianIgnoreNaN(p99));
            }
        }

        // ── Build per-image reports ─────────────────────────────────────────────
        var reports = new ArrayList<PixelCohortReport.ImageReport>(nImages);
        for (int i = 0; i < nImages; i++) {
            var img = safe.get(i);

            var channelContexts = new ArrayList<PixelCohortReport.ChannelContext>(
                    img.channels().size());
            for (var cs : img.channels()) {
                channelContexts.add(new PixelCohortReport.ChannelContext(cs.channel(), cs));
            }

            // ── Flags ──
            var flags = new ArrayList<String>(3);
            boolean backgroundHeavy = ge(emptyFractionZ[i], thresholds.backgroundEmptyZ())
                    || le(meanForegroundZ[i], -thresholds.backgroundForegroundZ());
            if (backgroundHeavy) {
                flags.add(PixelCohortReport.BACKGROUND_HEAVY);
            }
            boolean saturated = !Double.isNaN(maxSaturation[i])
                    && ((maxSaturation[i] >= thresholds.saturationMinFraction()
                                    && ge(maxSaturationZ[i], thresholds.saturationZ()))
                            || maxSaturation[i] >= thresholds.saturationHardFraction());
            if (saturated) {
                flags.add(PixelCohortReport.SATURATED);
            }
            boolean weakSignal = le(medianDynamicRangeZ[i], -thresholds.weakSignalZ());
            if (weakSignal) {
                flags.add(PixelCohortReport.WEAK_SIGNAL);
            }
            // Focus (maxFocus) is computed and surfaced for inspection but does NOT
            // drive a verdict: raw Laplacian variance tracks brightness as much as
            // sharpness, so flagging it would mislabel dim-but-fine slides.

            // Intensity outlier: the signal-bearing channel whose p99 (brightness)
            // diverges most from the cohort. These slides are the likely ML
            // challenges the prescreen is meant to surface.
            double maxIntensityAbsZ = 0.0;
            double maxIntensitySignedZ = 0.0;
            String maxIntensityChannel = null;
            for (var cs : img.channels()) {
                double[] zs = p99ZByChannel.get(cs.channel());
                if (zs == null) {
                    continue;
                }
                double z = zs[i];
                if (!Double.isNaN(z) && Math.abs(z) > maxIntensityAbsZ) {
                    maxIntensityAbsZ = Math.abs(z);
                    maxIntensitySignedZ = z;
                    maxIntensityChannel = cs.channel();
                }
            }
            boolean intensityOutlier = maxIntensityAbsZ >= thresholds.intensityOutlierZ();
            if (intensityOutlier) {
                flags.add(PixelCohortReport.INTENSITY_OUTLIER);
            }

            String verdict = verdict(flags);

            double score = relu(emptyFractionZ[i])
                    + relu(-meanForegroundZ[i])
                    + (saturated ? relu(maxSaturationZ[i]) : 0.0)
                    + relu(-medianDynamicRangeZ[i])
                    + relu(maxIntensityAbsZ - 2.0);

            double medP99 = maxIntensityChannel == null
                    ? Double.NaN
                    : medP99ByChannel.getOrDefault(maxIntensityChannel, Double.NaN);
            String narrative = buildNarrative(
                    img,
                    verdict,
                    flags,
                    emptyFraction[i],
                    emptyFractionZ[i],
                    medEmptyFraction,
                    meanForeground[i],
                    meanForegroundZ[i],
                    medMeanForeground,
                    maxSaturation[i],
                    maxSaturationChannel[i],
                    maxSaturationZ[i],
                    medMaxSaturation,
                    medianDynamicRange[i],
                    medianDynamicRangeZ[i],
                    medMedianDynamicRange,
                    maxFocus[i],
                    maxFocusZ[i],
                    medMaxFocus,
                    maxIntensityChannel,
                    maxIntensitySignedZ,
                    medP99);

            reports.add(new PixelCohortReport.ImageReport(
                    img.imageName(),
                    verdict,
                    flags,
                    score,
                    narrative,
                    emptyFraction[i],
                    emptyFractionZ[i],
                    meanForeground[i],
                    meanForegroundZ[i],
                    maxSaturation[i],
                    maxSaturationChannel[i],
                    maxSaturationZ[i],
                    medianDynamicRange[i],
                    medianDynamicRangeZ[i],
                    maxFocus[i],
                    maxFocusZ[i],
                    maxIntensitySignedZ,
                    maxIntensityChannel,
                    channelContexts));
        }

        reports.sort(Comparator.comparingDouble(PixelCohortReport.ImageReport::score)
                .reversed()
                .thenComparing(PixelCohortReport.ImageReport::imageName));

        return new PixelCohortReport(reports);
    }

    // ── Narrative ───────────────────────────────────────────────────────────────

    private static String buildNarrative(
            ImagePixelStats.ImageStats img,
            String verdict,
            List<String> flags,
            double emptyFraction,
            double emptyFractionZ,
            double medEmptyFraction,
            double meanForeground,
            double meanForegroundZ,
            double medMeanForeground,
            double maxSaturation,
            String maxSaturationChannel,
            double maxSaturationZ,
            double medMaxSaturation,
            double medianDynamicRange,
            double medianDynamicRangeZ,
            double medMedianDynamicRange,
            double maxFocus,
            double maxFocusZ,
            double medMaxFocus,
            String maxIntensityChannel,
            double maxIntensitySignedZ,
            double medP99) {

        var sb = new StringBuilder();
        sb.append(img.imageName()).append(" — ").append(verdict).append('\n');

        // Foreground / background context.
        if (flags.contains(PixelCohortReport.BACKGROUND_HEAVY) || notable(emptyFractionZ) || notable(meanForegroundZ)) {
            sb.append("• Foreground coverage ")
                    .append(fmtPct(meanForeground))
                    .append(" (cohort median ")
                    .append(fmtPct(medMeanForeground))
                    .append(", ")
                    .append(fmtZ(meanForegroundZ))
                    .append(" MAD).");
            if (!Double.isNaN(emptyFraction)) {
                sb.append(" Empty/glass ")
                        .append(fmtPct(emptyFraction))
                        .append(" (median ")
                        .append(fmtPct(medEmptyFraction))
                        .append(", ")
                        .append(fmtZ(emptyFractionZ))
                        .append(" MAD).");
            }
            sb.append('\n');
        }

        // Saturation context.
        if (flags.contains(PixelCohortReport.SATURATED) || notable(maxSaturationZ)) {
            sb.append("• Peak saturation ").append(fmtPct(maxSaturation));
            if (maxSaturationChannel != null) {
                sb.append(" in ").append(maxSaturationChannel);
            }
            sb.append(" (cohort median ")
                    .append(fmtPct(medMaxSaturation))
                    .append(", ")
                    .append(fmtZ(maxSaturationZ))
                    .append(" MAD).\n");
        }

        // Dynamic range / weak signal context.
        if (flags.contains(PixelCohortReport.WEAK_SIGNAL) || notable(medianDynamicRangeZ)) {
            sb.append("• Median dynamic range ")
                    .append(fmtNum(medianDynamicRange))
                    .append(" (cohort median ")
                    .append(fmtNum(medMedianDynamicRange))
                    .append(", ")
                    .append(fmtZ(medianDynamicRangeZ))
                    .append(" MAD).\n");
        }

        // Focus / sharpness context (informational only — not a flag).
        if (notable(maxFocusZ)) {
            sb.append("• Focus (sharpest-channel Laplacian variance) ")
                    .append(fmtNum(maxFocus))
                    .append(" (cohort median ")
                    .append(fmtNum(medMaxFocus))
                    .append(", ")
                    .append(fmtZ(maxFocusZ))
                    .append(" MAD).\n");
        }

        // Intensity-outlier context: name the divergent channel and its brightness.
        if ((flags.contains(PixelCohortReport.INTENSITY_OUTLIER) || notable(maxIntensitySignedZ))
                && maxIntensityChannel != null) {
            double imgP99 = Double.NaN;
            for (var cs : img.channels()) {
                if (cs.channel().equals(maxIntensityChannel)) {
                    imgP99 = cs.p99();
                    break;
                }
            }
            String direction = maxIntensitySignedZ >= 0 ? "brighter" : "dimmer";
            sb.append("• ")
                    .append(maxIntensityChannel)
                    .append(" brightness (p99) ")
                    .append(fmtNum(imgP99))
                    .append(" is ")
                    .append(direction)
                    .append(" than the cohort (median ")
                    .append(fmtNum(medP99))
                    .append(", ")
                    .append(fmtZ(maxIntensitySignedZ))
                    .append(" MAD).\n");
        }

        if (flags.isEmpty()) {
            sb.append("• Pixel statistics are within the normal range for this project.\n");
        }

        sb.append("Suggested action: ").append(suggestedAction(verdict, flags));
        return sb.toString();
    }

    private static String suggestedAction(String verdict, List<String> flags) {
        if (flags.isEmpty()) {
            return "include — looks normal.";
        }
        if (flags.contains(PixelCohortReport.BACKGROUND_HEAVY)) {
            return "review / consider removing — mostly background.";
        }
        if (flags.contains(PixelCohortReport.SATURATED)) {
            return "review acquisition — channel(s) clipped/over-exposed.";
        }
        if (flags.contains(PixelCohortReport.WEAK_SIGNAL)) {
            return "review staining/exposure — unusually flat signal.";
        }
        if (flags.contains(PixelCohortReport.INTENSITY_OUTLIER)) {
            return "review / normalize — intensity differs from the cohort (may challenge ML).";
        }
        return "review — pixel statistics differ from the cohort.";
    }

    private static String verdict(List<String> flags) {
        if (flags.isEmpty()) {
            return PixelCohortReport.VERDICT_OK;
        }
        if (flags.contains(PixelCohortReport.BACKGROUND_HEAVY)) {
            return "Background-heavy";
        }
        if (flags.contains(PixelCohortReport.SATURATED)) {
            return "Saturated";
        }
        if (flags.contains(PixelCohortReport.WEAK_SIGNAL)) {
            return "Weak signal";
        }
        if (flags.contains(PixelCohortReport.INTENSITY_OUTLIER)) {
            return "Intensity outlier";
        }
        return PixelCohortReport.VERDICT_OK;
    }

    // ── Robust statistics helpers ────────────────────────────────────────────────

    /** Find a channel's stats by name within one image, or {@code null}. */
    private static ImagePixelStats.ChannelStats channelOf(ImagePixelStats.ImageStats img, String channel) {
        for (var cs : img.channels()) {
            if (channel.equals(cs.channel())) {
                return cs;
            }
        }
        return null;
    }

    /**
     * Robust z-scores, NaN-preserving, MAD-scaled, with a mean/std fallback for a
     * degenerate MAD. Delegates to {@link RobustStats#robustZWithFallback(double[])};
     * kept here as a package-private seam so existing call sites and tests are
     * unaffected. See that method for the full rationale.
     */
    static double[] robustZ(double[] values) {
        return RobustStats.robustZWithFallback(values);
    }

    private static double medianIgnoreNaN(double[] values) {
        return RobustStats.medianIgnoreNaN(values);
    }

    private static double medianOf(List<Double> values) {
        if (values == null || values.isEmpty()) {
            return Double.NaN;
        }
        double[] arr = new double[values.size()];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = values.get(i);
        }
        return medianIgnoreNaN(arr);
    }

    private static double mean(List<Double> values) {
        if (values == null || values.isEmpty()) {
            return Double.NaN;
        }
        double sum = 0.0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.size();
    }

    private static double relu(double x) {
        return (Double.isNaN(x) || x < 0.0) ? 0.0 : x;
    }

    private static boolean ge(double x, double threshold) {
        return !Double.isNaN(x) && x >= threshold;
    }

    private static boolean le(double x, double threshold) {
        return !Double.isNaN(x) && x <= threshold;
    }

    private static boolean notable(double z) {
        return !Double.isNaN(z) && Math.abs(z) >= 2.0;
    }

    // ── Formatting ────────────────────────────────────────────────────────────────

    private static String fmtPct(double frac) {
        if (Double.isNaN(frac)) {
            return "n/a";
        }
        return String.format(Locale.ROOT, "%.1f%%", frac * 100.0);
    }

    private static String fmtZ(double z) {
        if (Double.isNaN(z)) {
            return "n/a";
        }
        return String.format(Locale.ROOT, "%+.1f", z);
    }

    private static String fmtNum(double v) {
        if (Double.isNaN(v)) {
            return "n/a";
        }
        if (Math.abs(v) >= 1000.0) {
            return String.format(Locale.ROOT, "%.0f", v);
        }
        return String.format(Locale.ROOT, "%.2f", v);
    }
}
