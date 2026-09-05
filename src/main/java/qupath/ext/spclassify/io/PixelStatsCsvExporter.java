package qupath.ext.spclassify.io;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import qupath.ext.spclassify.model.PixelCohortReport;

/**
 * CSV writer for the whole-image pixel-prescreen report.
 * <p>
 * Produces one row per image in a wide layout: image-level verdict/score/summary
 * columns followed by a fixed block of per-channel statistic columns for every
 * channel seen across the cohort (blank where an image lacks a channel). This
 * mirrors the per-marker spreadsheet layout used for manual cell-level review.
 */
public final class PixelStatsCsvExporter {

    /** Per-channel statistic columns, in output order. */
    private static final List<String> CHANNEL_METRICS = List.of(
            "Median",
            "Mean",
            "Std",
            "P1",
            "P99",
            "Max",
            "ForegroundCoverage",
            "BackgroundFraction",
            "DynamicRange",
            "SaturationFraction",
            "LaplacianVariance");

    private PixelStatsCsvExporter() {}

    /** Serialise the report to CSV text. */
    public static String toCsv(PixelCohortReport report) {
        var images = report == null ? List.<PixelCohortReport.ImageReport>of() : report.images();

        // Channel union in first-seen order across all images.
        var channels = new LinkedHashSet<String>();
        for (var img : images) {
            for (var ch : img.channels()) {
                channels.add(ch.channel());
            }
        }
        var orderedChannels = new ArrayList<>(channels);

        StringBuilder sb = new StringBuilder();

        // Header.
        sb.append("Image,Verdict,Flags,Score,EmptyFraction,EmptyFractionZ,")
                .append("MeanForegroundCoverage,MeanForegroundCoverageZ,")
                .append("MaxSaturationFraction,MaxSaturationChannel,MaxSaturationZ,")
                .append("MedianDynamicRange,MedianDynamicRangeZ,")
                .append("MaxFocus,MaxFocusZ,")
                .append("MaxIntensityZ,MaxIntensityChannel");
        for (String ch : orderedChannels) {
            for (String metric : CHANNEL_METRICS) {
                sb.append(',').append(csvEscape(ch + ": " + metric));
            }
        }
        sb.append('\n');

        // Rows.
        for (var img : images) {
            sb.append(csvEscape(img.imageName()))
                    .append(',')
                    .append(csvEscape(img.verdict()))
                    .append(',')
                    .append(csvEscape(String.join("; ", img.flags())))
                    .append(',')
                    .append(num(img.score()))
                    .append(',')
                    .append(num(img.emptyFraction()))
                    .append(',')
                    .append(num(img.emptyFractionZ()))
                    .append(',')
                    .append(num(img.meanForegroundCoverage()))
                    .append(',')
                    .append(num(img.meanForegroundCoverageZ()))
                    .append(',')
                    .append(num(img.maxSaturationFraction()))
                    .append(',')
                    .append(csvEscape(img.maxSaturationChannel()))
                    .append(',')
                    .append(num(img.maxSaturationFractionZ()))
                    .append(',')
                    .append(num(img.medianDynamicRange()))
                    .append(',')
                    .append(num(img.medianDynamicRangeZ()))
                    .append(',')
                    .append(num(img.maxFocus()))
                    .append(',')
                    .append(num(img.maxFocusZ()))
                    .append(',')
                    .append(num(img.maxIntensityZ()))
                    .append(',')
                    .append(csvEscape(img.maxIntensityChannel()));

            for (String ch : orderedChannels) {
                PixelCohortReport.ChannelContext cc = findChannel(img, ch);
                if (cc == null) {
                    for (int m = 0; m < CHANNEL_METRICS.size(); m++) {
                        sb.append(',');
                    }
                    continue;
                }
                var s = cc.stats();
                sb.append(',')
                        .append(num(s.median()))
                        .append(',')
                        .append(num(s.mean()))
                        .append(',')
                        .append(num(s.std()))
                        .append(',')
                        .append(num(s.p1()))
                        .append(',')
                        .append(num(s.p99()))
                        .append(',')
                        .append(num(s.max()))
                        .append(',')
                        .append(num(s.foregroundCoverage()))
                        .append(',')
                        .append(num(s.backgroundFraction()))
                        .append(',')
                        .append(num(s.dynamicRange()))
                        .append(',')
                        .append(num(s.saturationFraction()))
                        .append(',')
                        .append(num(s.laplacianVariance()));
            }
            sb.append('\n');
        }

        return sb.toString();
    }

    /** Write the report to {@code outputPath} as UTF-8 CSV. */
    public static void writeCsv(Path outputPath, PixelCohortReport report) throws IOException {
        Files.writeString(outputPath, toCsv(report), StandardCharsets.UTF_8);
    }

    private static PixelCohortReport.ChannelContext findChannel(PixelCohortReport.ImageReport img, String channel) {
        for (var cc : img.channels()) {
            if (channel.equals(cc.channel())) {
                return cc;
            }
        }
        return null;
    }

    /** Format a number, leaving blank cells for NaN. */
    private static String num(double v) {
        if (Double.isNaN(v)) {
            return "";
        }
        return String.format(Locale.ROOT, "%.4f", v);
    }

    private static String csvEscape(String value) {
        return CsvUtils.quoteAlways(value);
    }
}
