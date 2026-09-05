package qupath.ext.spclassify.model;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class PixelCohortAnalyzerTest {

    /** Build a 2-channel image whose channels share the given summary values. */
    private static ImagePixelStats.ImageStats image(
            String name,
            double fgCoverage,
            double dynamicRange,
            double saturationFraction,
            double emptyFraction,
            double median) {
        return image(name, fgCoverage, dynamicRange, saturationFraction, emptyFraction, median, 100.0);
    }

    /** As {@link #image}, with an explicit focus (Laplacian variance) value. */
    private static ImagePixelStats.ImageStats image(
            String name,
            double fgCoverage,
            double dynamicRange,
            double saturationFraction,
            double emptyFraction,
            double median,
            double focus) {
        double bg = 1.0 - fgCoverage;
        double p1 = 10.0;
        double p99 = p1 + dynamicRange;
        var channels = new ArrayList<ImagePixelStats.ChannelStats>();
        for (String ch : List.of("DAPI", "CD8")) {
            channels.add(new ImagePixelStats.ChannelStats(
                    ch,
                    1_000_000L,
                    median, // mean (≈ median for the test)
                    dynamicRange / 4, // std
                    0.0, // min
                    median, // median
                    p99 + 1, // max
                    p1,
                    p99,
                    saturationFraction,
                    50.0, // otsu threshold
                    bg,
                    fgCoverage,
                    dynamicRange,
                    focus));
        }
        return new ImagePixelStats.ImageStats(name, 4.0, 1024, 1024, channels, emptyFraction);
    }

    /** Six tightly-clustered normal images. */
    private static List<ImagePixelStats.ImageStats> normalCohort() {
        var list = new ArrayList<ImagePixelStats.ImageStats>();
        for (int i = 0; i < 6; i++) {
            double jitter = i * 0.3;
            list.add(image(
                    "normal-" + i,
                    0.60 + i * 0.004, // fg coverage ~0.60
                    190.0 + jitter, // dynamic range ~190
                    0.0, // no saturation
                    0.30 + i * 0.004, // empty fraction ~0.30
                    100.0 + jitter)); // median ~100
        }
        return list;
    }

    @Test
    void flagsBackgroundHeavyImage() {
        var inputs = normalCohort();
        inputs.add(image(
                "mostly-background",
                0.08, // very low foreground coverage
                190.0, // normal dynamic range
                0.0,
                0.90, // very high empty fraction
                100.0));

        var report = PixelCohortAnalyzer.analyze(inputs);
        var r = report.byImageName().get("mostly-background");

        assertNotNull(r);
        assertTrue(r.flags().contains(PixelCohortReport.BACKGROUND_HEAVY), "flags=" + r.flags());
        assertEquals("Background-heavy", r.verdict());
        assertTrue(r.narrative().toLowerCase().contains("background"));
        // Worst image sorts to the top.
        assertEquals("mostly-background", report.images().get(0).imageName());
    }

    @Test
    void flagsSaturatedImageEvenWhenBaselineIsZero() {
        var inputs = normalCohort(); // all 0% saturation
        inputs.add(image(
                "over-exposed",
                0.60,
                190.0,
                0.25, // 25% of pixels clipped
                0.30,
                100.0));

        var report = PixelCohortAnalyzer.analyze(inputs);
        var r = report.byImageName().get("over-exposed");

        assertNotNull(r);
        assertTrue(r.flags().contains(PixelCohortReport.SATURATED), "flags=" + r.flags());
        assertTrue(List.of("DAPI", "CD8").contains(r.maxSaturationChannel()));
        assertTrue(r.maxSaturationFraction() >= 0.01);
    }

    @Test
    void flagsWeakSignalImage() {
        var inputs = normalCohort();
        inputs.add(image(
                "washed-out",
                0.60,
                4.0, // almost no dynamic range
                0.0,
                0.30,
                100.0));

        var report = PixelCohortAnalyzer.analyze(inputs);
        var r = report.byImageName().get("washed-out");

        assertNotNull(r);
        assertTrue(r.flags().contains(PixelCohortReport.WEAK_SIGNAL), "flags=" + r.flags());
    }

    @Test
    void leavesNormalImagesUnflagged() {
        var report = PixelCohortAnalyzer.analyze(normalCohort());
        for (var r : report.images()) {
            assertFalse(r.flagged(), r.imageName() + " should not be flagged: " + r.flags());
            assertEquals(PixelCohortReport.VERDICT_OK, r.verdict());
        }
    }

    @Test
    void emptyInputYieldsEmptyReport() {
        assertTrue(PixelCohortAnalyzer.analyze(List.of()).images().isEmpty());
        assertTrue(PixelCohortAnalyzer.analyze(null).images().isEmpty());
    }

    @Test
    void deadChannelNoiseDoesNotFlagAnyImage() {
        // A near-dead channel: every image's median hovers at the noise floor
        // (~0) with one slightly-larger blip. The MAD-scaled z used to explode
        // this into a huge "intensity outlier" score; that detection has been
        // removed, so a cohort that is otherwise healthy must stay unflagged.
        var inputs = new ArrayList<ImagePixelStats.ImageStats>();
        for (int i = 0; i < 8; i++) {
            double deadMedian = (i == 0) ? 0.005 : 1e-7 * i;
            inputs.add(deadChannelImage("img-" + i, deadMedian));
        }

        var report = PixelCohortAnalyzer.analyze(inputs);
        for (var r : report.images()) {
            assertFalse(r.flagged(), r.imageName() + " unexpectedly flagged: " + r.flags());
            assertEquals(PixelCohortReport.VERDICT_OK, r.verdict());
        }
    }

    /** Healthy bright channel + a near-dead channel sitting at the noise floor. */
    private static ImagePixelStats.ImageStats deadChannelImage(String name, double deadMedian) {
        return deadChannelImage(name, deadMedian, 0.4);
    }

    /** As above, with an explicit (noise-floor) p99 for the near-dead TCR channel. */
    private static ImagePixelStats.ImageStats deadChannelImage(String name, double deadMedian, double tcrP99) {
        var channels = new ArrayList<ImagePixelStats.ChannelStats>();
        channels.add(new ImagePixelStats.ChannelStats(
                "DAPI", 1_000_000L, 100.0, 47.5, 0.0, 100.0, 201.0, 10.0, 200.0, 0.0, 50.0, 0.40, 0.60, 190.0, 100.0));
        // TCR foreground 0.02 is below the signal floor → excluded from intensity.
        channels.add(new ImagePixelStats.ChannelStats(
                "TCR",
                1_000_000L,
                deadMedian,
                0.01,
                0.0,
                deadMedian,
                tcrP99 + 0.1,
                0.0,
                tcrP99,
                0.0,
                0.05,
                0.98,
                0.02,
                tcrP99,
                1e-7));
        return new ImagePixelStats.ImageStats(name, 4.0, 1024, 1024, channels, 0.30);
    }

    @Test
    void flagsBrightIntensityOutlierOnSignalChannel() {
        // Cohort of normal-brightness slides + one much brighter slide. The bright
        // slide's signal-channel p99 diverges sharply from the cohort.
        var inputs = new ArrayList<ImagePixelStats.ImageStats>();
        for (int i = 0; i < 8; i++) {
            inputs.add(image("normal-" + i, 0.60, 190.0 + i * 0.5, 0.0, 0.30, 100.0));
        }
        inputs.add(image("too-bright", 0.60, 600.0, 0.0, 0.30, 100.0));

        var report = PixelCohortAnalyzer.analyze(inputs);
        var r = report.byImageName().get("too-bright");

        assertNotNull(r);
        assertTrue(r.flags().contains(PixelCohortReport.INTENSITY_OUTLIER), "flags=" + r.flags());
        assertEquals("Intensity outlier", r.verdict());
        assertNotNull(r.maxIntensityChannel());
        // Worst image sorts to the top.
        assertEquals("too-bright", report.images().get(0).imageName());
    }

    @Test
    void deadChannelP99SpikeDoesNotTriggerIntensityOutlier() {
        // A 45× relative p99 spike on a sub-floor (near-dead) channel must NOT
        // flag: dead channels are excluded from intensity-outlier detection.
        var inputs = new ArrayList<ImagePixelStats.ImageStats>();
        for (int i = 0; i < 8; i++) {
            double tcrP99 = (i == 0) ? 0.9 : 0.02;
            inputs.add(deadChannelImage("img-" + i, 1e-7, tcrP99));
        }

        var report = PixelCohortAnalyzer.analyze(inputs);
        for (var r : report.images()) {
            assertFalse(
                    r.flags().contains(PixelCohortReport.INTENSITY_OUTLIER), r.imageName() + " flagged: " + r.flags());
        }
    }

    @Test
    void lowFocusImageIsSurfacedButNotFlagged() {
        // Focus tracks brightness as much as sharpness, so a low-focus (e.g. dim)
        // image must NOT be flagged on focus alone — the metric is informational.
        var inputs = new ArrayList<ImagePixelStats.ImageStats>();
        for (int i = 0; i < 6; i++) {
            inputs.add(image("sharp-" + i, 0.60, 190.0, 0.0, 0.30, 100.0, 100.0 + i * 0.5));
        }
        inputs.add(image("low-focus", 0.60, 190.0, 0.0, 0.30, 100.0, 1.0));

        var report = PixelCohortAnalyzer.analyze(inputs);
        var r = report.byImageName().get("low-focus");

        assertNotNull(r);
        // maxFocus is computed and surfaced …
        assertEquals(1.0, r.maxFocus(), 1e-9);
        // … but it does not produce a verdict.
        assertFalse(r.flagged(), "should not be flagged on focus: " + r.flags());
        assertEquals(PixelCohortReport.VERDICT_OK, r.verdict());
    }

    @Test
    void robustZFallsBackToStdWhenMadIsZero() {
        // Seven zeros and one clear outlier: MAD is 0, so the std fallback must
        // still produce a large positive z for the outlier.
        double[] vals = {0, 0, 0, 0, 0, 0, 0, 0.25};
        double[] z = PixelCohortAnalyzer.robustZ(vals);
        assertTrue(z[7] > 2.5, "outlier z=" + z[7]);
    }
}
