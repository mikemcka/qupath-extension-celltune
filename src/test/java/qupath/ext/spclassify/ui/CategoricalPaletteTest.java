package qupath.ext.spclassify.ui;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * Guards the CN categorical palette against black-on-black: every entry the viewer overlay can paint
 * must be bright enough to see on QuPath's black background. Luminance is recomputed independently
 * here (the WCAG sRGB formula, from the {@code 0xRRGGBB} ints the mapper actually paints) rather than
 * reusing the production helper, so a bug in that helper cannot make this test pass falsely. The
 * palette is read as ints because the test classpath has no {@code javafx.scene.paint.Color}.
 */
class CategoricalPaletteTest {

    private static double linearize(double channel) {
        return channel <= 0.03928 ? channel / 12.92 : Math.pow((channel + 0.055) / 1.055, 2.4);
    }

    /** WCAG relative luminance of a {@code 0xRRGGBB} colour, in {@code [0, 1]}. */
    private static double luminance(int rgb) {
        double r = ((rgb >> 16) & 0xff) / 255.0;
        double g = ((rgb >> 8) & 0xff) / 255.0;
        double b = (rgb & 0xff) / 255.0;
        return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b);
    }

    @Test
    void everyDisplayedColourIsVisibleOnBlack() {
        for (int rgb : NeighborhoodAnalysisDialog.categoricalRgb()) {
            assertTrue(
                    luminance(rgb) >= NeighborhoodAnalysisDialog.MIN_LUMINANCE_ON_BLACK - 1e-9,
                    () -> String.format(
                            "Palette colour #%06x (luminance %.4f) is below the black-viewer floor %.2f",
                            rgb, luminance(rgb), NeighborhoodAnalysisDialog.MIN_LUMINANCE_ON_BLACK));
        }
    }

    @Test
    void liftPreservesPaletteSize() {
        assertEquals(
                NeighborhoodAnalysisDialog.categoricalRawRgb().length,
                NeighborhoodAnalysisDialog.categoricalRgb().length);
    }

    @Test
    void rawPaletteActuallyHadTooDarkEntries() {
        // If this fails the raw palette changed; the lift (and this guard) would then be pointless.
        long tooDark = java.util.Arrays.stream(NeighborhoodAnalysisDialog.categoricalRawRgb())
                .filter(rgb -> luminance(rgb) < NeighborhoodAnalysisDialog.MIN_LUMINANCE_ON_BLACK)
                .count();
        assertTrue(tooDark > 0, "Expected the raw Glasbey palette to contain near-black entries");
    }

    @Test
    void alreadyVisibleColoursAreLeftUntouched() {
        // Entries already above the floor must be passed through unchanged (no gratuitous lift).
        int[] raw = NeighborhoodAnalysisDialog.categoricalRawRgb();
        int[] lifted = NeighborhoodAnalysisDialog.categoricalRgb();
        for (int i = 0; i < raw.length; i++) {
            int rawRgb = raw[i];
            if (luminance(rawRgb) >= NeighborhoodAnalysisDialog.MIN_LUMINANCE_ON_BLACK) {
                assertEquals(
                        rawRgb,
                        lifted[i],
                        () -> String.format("Visible colour #%06x should not be altered by the lift", rawRgb));
            }
        }
    }

    @Test
    void liftIsDeterministic() {
        // Same input palette must always yield the same displayed palette (static, no randomness).
        assertArrayEquals(NeighborhoodAnalysisDialog.categoricalRgb(), NeighborhoodAnalysisDialog.categoricalRgb());
    }
}
