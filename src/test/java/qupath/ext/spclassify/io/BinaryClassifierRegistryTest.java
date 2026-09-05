package qupath.ext.spclassify.io;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

/**
 * Covers {@link BinaryClassifierRegistry#sanitizeMarkerName(String)} — the
 * single chokepoint that keeps user-entered marker names safe to use as
 * filesystem paths.
 */
class BinaryClassifierRegistryTest {

    @Test
    void plainNamePassesThrough() {
        assertEquals("CD3", BinaryClassifierRegistry.sanitizeMarkerName("CD3"));
        assertEquals("PD-1", BinaryClassifierRegistry.sanitizeMarkerName("PD-1"));
        assertEquals("HLA_DR", BinaryClassifierRegistry.sanitizeMarkerName("HLA_DR"));
        assertEquals("Ki67.alt", BinaryClassifierRegistry.sanitizeMarkerName("Ki67.alt"));
    }

    @Test
    void disallowedCharactersBecomeUnderscore() {
        assertEquals("CD3_CD8", BinaryClassifierRegistry.sanitizeMarkerName("CD3/CD8"));
        assertEquals("a_b_c", BinaryClassifierRegistry.sanitizeMarkerName("a b c"));
        // Greek / unicode marker glyphs collapse to underscores.
        assertEquals("TCR_", BinaryClassifierRegistry.sanitizeMarkerName("TCRγ"));
    }

    @Test
    void pathTraversalSeparatorsAreNeutralised() {
        // Slashes/backslashes are replaced; the result cannot escape its directory.
        String s = BinaryClassifierRegistry.sanitizeMarkerName("foo/bar");
        assertFalse(s.contains("/"));
        assertFalse(s.contains("\\"));
        assertEquals("foo_bar", s);
    }

    @Test
    void leadingDotOrHyphenIsRejected() {
        // "../" style or hidden-file names must not be accepted.
        assertThrows(
                IllegalArgumentException.class,
                () -> BinaryClassifierRegistry.sanitizeMarkerName("..")); // ".." stays "..", starts with '.'
        assertThrows(IllegalArgumentException.class, () -> BinaryClassifierRegistry.sanitizeMarkerName(".hidden"));
        assertThrows(IllegalArgumentException.class, () -> BinaryClassifierRegistry.sanitizeMarkerName("-flag"));
    }

    @Test
    void blankOrNullIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> BinaryClassifierRegistry.sanitizeMarkerName(null));
        assertThrows(IllegalArgumentException.class, () -> BinaryClassifierRegistry.sanitizeMarkerName("   "));
    }

    @Test
    void leadingWhitespaceIsTrimmedBeforeChecks() {
        assertEquals("CD3", BinaryClassifierRegistry.sanitizeMarkerName("  CD3  "));
    }
}
