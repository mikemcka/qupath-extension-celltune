package qupath.ext.spclassify.gating;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

class GatingExpressionTest {

    // ── helpers ─────────────────────────────────────────────────────────────

    /** Build a single-length boolean mask for one marker. */
    private static boolean[] mask(boolean value) {
        return new boolean[] {value};
    }

    private static Map<String, boolean[]> masks(Object... pairs) {
        var map = new java.util.LinkedHashMap<String, boolean[]>();
        for (int i = 0; i < pairs.length; i += 2) {
            map.put((String) pairs[i], new boolean[] {(Boolean) pairs[i + 1]});
        }
        return map;
    }

    // ── basic evaluation ────────────────────────────────────────────────────

    @Test
    void singleMarkerPassesWhenPositive() {
        var expr = new GatingExpression("CD3");
        boolean[] result = expr.evaluate(Map.of("CD3", mask(true)));
        assertNotNull(result);
        assertTrue(result[0]);
    }

    @Test
    void singleMarkerFailsWhenNegative() {
        var expr = new GatingExpression("CD3");
        boolean[] result = expr.evaluate(Map.of("CD3", mask(false)));
        assertNotNull(result);
        assertFalse(result[0]);
    }

    @Test
    void andExpressionRequiresBothTrue() {
        var expr = new GatingExpression("CD4&CD3");
        assertTrue(expr.evaluate(masks("CD4", true, "CD3", true))[0]);
        assertFalse(expr.evaluate(masks("CD4", true, "CD3", false))[0]);
        assertFalse(expr.evaluate(masks("CD4", false, "CD3", true))[0]);
        assertFalse(expr.evaluate(masks("CD4", false, "CD3", false))[0]);
    }

    @Test
    void orExpressionPassesWhenEitherTrue() {
        var expr = new GatingExpression("CD68|CD163");
        assertTrue(expr.evaluate(masks("CD68", true, "CD163", false))[0]);
        assertTrue(expr.evaluate(masks("CD68", false, "CD163", true))[0]);
        assertTrue(expr.evaluate(masks("CD68", true, "CD163", true))[0]);
        assertFalse(expr.evaluate(masks("CD68", false, "CD163", false))[0]);
    }

    @Test
    void notExpressionNegatesMarker() {
        var expr = new GatingExpression("CD38&!IgA");
        assertTrue(expr.evaluate(masks("CD38", true, "IgA", false))[0]);
        assertFalse(expr.evaluate(masks("CD38", true, "IgA", true))[0]);
    }

    @Test
    void groupedOrInsideAndEvaluatesCorrectly() {
        // A & (B | C)  →  A must be true, and at least one of B or C
        var expr = new GatingExpression("A&(B|C)");
        assertTrue(expr.evaluate(masks("A", true, "B", true, "C", false))[0]);
        assertTrue(expr.evaluate(masks("A", true, "B", false, "C", true))[0]);
        assertFalse(expr.evaluate(masks("A", false, "B", true, "C", true))[0]);
        assertFalse(expr.evaluate(masks("A", true, "B", false, "C", false))[0]);
    }

    @Test
    void evaluateReturnsNullForMissingMarker() {
        var expr = new GatingExpression("CD3");
        assertNull(expr.evaluate(Map.of())); // no CD3 mask provided
    }

    // ── categorize ──────────────────────────────────────────────────────────

    @Test
    void categorizesSingleMarkerAsMustHave() {
        var cats = new GatingExpression("CD3").categorize();
        assertTrue(cats.mustHave().contains("CD3"));
        assertTrue(cats.orExpression().isEmpty());
        assertTrue(cats.not().isEmpty());
    }

    @Test
    void categorizesAndOperandsAsMustHave() {
        var cats = new GatingExpression("CD4&CD3").categorize();
        assertTrue(cats.mustHave().contains("CD4"));
        assertTrue(cats.mustHave().contains("CD3"));
        assertTrue(cats.orExpression().isEmpty());
    }

    @Test
    void categorizesOrOperandsAsOrExpression() {
        var cats = new GatingExpression("CD68|CD163").categorize();
        assertTrue(cats.orExpression().contains("CD68"));
        assertTrue(cats.orExpression().contains("CD163"));
        assertTrue(cats.mustHave().isEmpty());
    }

    @Test
    void categorizesNotOperandAsNot() {
        var cats = new GatingExpression("CD38&!IgA").categorize();
        assertTrue(cats.mustHave().contains("CD38"));
        assertTrue(cats.not().contains("IgA"));
        assertFalse(cats.not().contains("CD38"));
    }

    // ── getAllMarkers ────────────────────────────────────────────────────────

    @Test
    void getAllMarkersReturnsAllReferencedMarkers() {
        var expr = new GatingExpression("CD4&(CD68|CD163)&!IgA");
        var markers = expr.getAllMarkers();
        assertEquals(Set.of("CD4", "CD68", "CD163", "IgA"), markers);
    }

    // ── isEmpty / blank ──────────────────────────────────────────────────────

    @Test
    void emptyExpressionIsEmpty() {
        assertTrue(new GatingExpression("").isEmpty());
        assertTrue(new GatingExpression("   ").isEmpty());
        assertNull(new GatingExpression("").evaluate(Map.of()));
    }

    // ── validation ──────────────────────────────────────────────────────────

    @Test
    void unknownMarkerThrowsWhenValidMarkersProvided() {
        assertThrows(IllegalArgumentException.class, () -> new GatingExpression("CD3", Set.of("CD4")));
    }

    @Test
    void knownMarkerDoesNotThrowWithValidMarkerSet() {
        assertDoesNotThrow(() -> new GatingExpression("CD4&CD8", Set.of("CD4", "CD8")));
    }

    @Test
    void unclosedParenthesisThrows() {
        assertThrows(IllegalArgumentException.class, () -> new GatingExpression("CD4&(CD3"));
    }

    @Test
    void unexpectedTokenAfterExpressionThrows() {
        assertThrows(IllegalArgumentException.class, () -> new GatingExpression("CD4 CD3"));
    }

    // ── multi-cell evaluation ─────────────────────────────────────────────────

    @Test
    void andEvaluatesCorrectlyAcrossMultipleCells() {
        // 3 cells: [T,T], [T,F], [F,T]
        var expr = new GatingExpression("CD4&CD3");
        boolean[] cd4 = {true, true, false};
        boolean[] cd3 = {true, false, true};
        boolean[] result = expr.evaluate(Map.of("CD4", cd4, "CD3", cd3));
        assertNotNull(result);
        assertEquals(3, result.length);
        assertTrue(result[0]); // T&T → true
        assertFalse(result[1]); // T&F → false
        assertFalse(result[2]); // F&T → false
    }

    @Test
    void orEvaluatesCorrectlyAcrossMultipleCells() {
        var expr = new GatingExpression("CD68|CD163");
        boolean[] cd68 = {true, false, false};
        boolean[] cd163 = {false, true, false};
        boolean[] result = expr.evaluate(Map.of("CD68", cd68, "CD163", cd163));
        assertTrue(result[0]); // T|F → true
        assertTrue(result[1]); // F|T → true
        assertFalse(result[2]); // F|F → false
    }

    // ── nested NOT ───────────────────────────────────────────────────────────

    @Test
    void nestedNotInsideOr_AAndBOrNotC() {
        // A & (B | !C)  →  A must be true, and either B is true OR C is false
        var expr = new GatingExpression("A&(B|!C)");
        // cell: A=T, B=F, C=F  →  A=T, (B|!C) = (F|T) = T  →  result=T
        assertTrue(expr.evaluate(masks("A", true, "B", false, "C", false))[0]);
        // cell: A=T, B=F, C=T  →  A=T, (B|!C) = (F|F) = F  →  result=F
        assertFalse(expr.evaluate(masks("A", true, "B", false, "C", true))[0]);
        // cell: A=F, B=T, C=F  →  A=F  →  result=F
        assertFalse(expr.evaluate(masks("A", false, "B", true, "C", false))[0]);
    }

    // ── getExpression ─────────────────────────────────────────────────────────

    @Test
    void getExpressionReturnsOriginalString() {
        String raw = "CD4&(CD3|CD8)";
        assertEquals(raw, new GatingExpression(raw).getExpression());
    }

    // ── toString ─────────────────────────────────────────────────────────────

    @Test
    void toStringRendersExpressionText() {
        var expr = new GatingExpression("CD4&CD3");
        String text = expr.toString();
        assertFalse(text.isBlank());
        assertTrue(text.contains("CD4"));
        assertTrue(text.contains("CD3"));
    }
}
