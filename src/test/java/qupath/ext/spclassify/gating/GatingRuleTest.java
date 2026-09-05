package qupath.ext.spclassify.gating;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import org.junit.jupiter.api.Test;

class GatingRuleTest {

    // ── primary expression encoding ──────────────────────────────────────────

    @Test
    void mustHaveMarkersGetCode1() {
        // "CD4&CD3" → both are AND operands → code 1
        var rule = new GatingRule("CD4T", "CD4&CD3", null, null, List.of("CD4", "CD3", "CD8", "IgA"));
        assertEquals(1, rule.getEncoding("CD4"));
        assertEquals(1, rule.getEncoding("CD3"));
    }

    @Test
    void orExpressionMarkersGetCode2() {
        // "CD68|CD163" → both inside OR → code 2
        var rule = new GatingRule("Macro", "CD68|CD163", null, null, List.of("CD68", "CD163", "CD3"));
        assertEquals(2, rule.getEncoding("CD68"));
        assertEquals(2, rule.getEncoding("CD163"));
    }

    @Test
    void explicitNotMarkersGetCodeMinus3() {
        // "CD38&!IgA" → IgA is negated → code -3
        var rule = new GatingRule("Plasma", "CD38&!IgA", null, null, List.of("CD38", "IgA", "CD3"));
        assertEquals(-3, rule.getEncoding("IgA"));
        assertEquals(1, rule.getEncoding("CD38"));
    }

    @Test
    void unlistedChannelsGetSoftNotMinus1() {
        var rule = new GatingRule("CD4T", "CD4", null, null, List.of("CD4", "CD8", "VIM"));
        assertEquals(-1, rule.getEncoding("CD8"));
        assertEquals(-1, rule.getEncoding("VIM"));
    }

    @Test
    void channelNotInAllChannelsListDefaultsMinus1() {
        var rule = new GatingRule("CD4T", "CD4", null, null, List.of("CD4"));
        // "unknown" was never in allChannels
        assertEquals(-1, rule.getEncoding("unknown"));
    }

    // ── secondary / tertiary → code 0 ───────────────────────────────────────

    @Test
    void secondaryMarkersGetCode0() {
        var rule = new GatingRule("CD8T", "CD8", "CD45|VIM", null, List.of("CD8", "CD45", "VIM", "CD3"));
        assertEquals(1, rule.getEncoding("CD8")); // primary stays 1
        assertEquals(0, rule.getEncoding("CD45")); // secondary → 0
        assertEquals(0, rule.getEncoding("VIM")); // secondary → 0
        assertEquals(-1, rule.getEncoding("CD3")); // unlisted
    }

    @Test
    void tertiaryMarkersGetCode0() {
        var rule = new GatingRule("CD4T", "CD4", null, "CD103|CD45RA", List.of("CD4", "CD103", "CD45RA", "CD8"));
        assertEquals(0, rule.getEncoding("CD103"));
        assertEquals(0, rule.getEncoding("CD45RA"));
        assertEquals(-1, rule.getEncoding("CD8"));
    }

    @Test
    void secondaryDoesNotDemotePrimaryMarkers() {
        // CD4 appears in both primary (→1) and secondary — secondary must not overwrite
        var rule = new GatingRule("CD4T", "CD4", "CD4|VIM", null, List.of("CD4", "VIM"));
        assertEquals(1, rule.getEncoding("CD4")); // stays 1, not demoted to 0
        assertEquals(0, rule.getEncoding("VIM"));
    }

    @Test
    void secondaryDoesNotDemoteStrictNot() {
        // IgA is -3 from primary "!IgA"; also in secondary — secondary must not overwrite -3
        var rule = new GatingRule("Plasma", "CD38&!IgA", "IgA", null, List.of("CD38", "IgA"));
        assertEquals(-3, rule.getEncoding("IgA")); // strict NOT, not demoted to 0
    }

    // ── getChannelsWithEncoding ──────────────────────────────────────────────

    @Test
    void getChannelsWithEncodingFiltersCorrectly() {
        var rule = new GatingRule("T", "CD4&CD3&!IgA", "CD45", null, List.of("CD4", "CD3", "IgA", "CD45", "CD8"));
        assertTrue(rule.getChannelsWithEncoding(1).containsAll(List.of("CD4", "CD3")));
        assertTrue(rule.getChannelsWithEncoding(-3).contains("IgA"));
        assertTrue(rule.getChannelsWithEncoding(0).contains("CD45"));
        assertTrue(rule.getChannelsWithEncoding(-1).contains("CD8"));
    }

    @Test
    void getChannelsWithEncodingReturnsEmptyForUnusedCode() {
        var rule = new GatingRule("T", "CD4", null, null, List.of("CD4", "CD8"));
        // No or-expression markers, so code 2 should be empty
        assertTrue(rule.getChannelsWithEncoding(2).isEmpty());
    }

    // ── promoteOverlappingSoftNots ───────────────────────────────────────────

    @Test
    void promotesSharedPrimaryContextSoftNotToStrictNot() {
        // Rule1: CD4 primary → CD4=1, CD8=-1
        var rule1 = new GatingRule("CD4T", "CD4", null, null, List.of("CD4", "CD8"));
        // Rule2: CD4&CD8 primary → CD4=1, CD8=1
        var rule2 = new GatingRule("CD4CD8T", "CD4&CD8", null, null, List.of("CD4", "CD8"));

        rule1.promoteOverlappingSoftNots(rule2);

        // CD8 is must-have in rule2 and soft-NOT in rule1 → promoted to strict-NOT
        assertEquals(-3, rule1.getEncoding("CD8"));
    }

    @Test
    void doesNotPromoteWhenNoSharedPrimaryMarkers() {
        // Rule1: CD4 only; Rule2: CD8 only — no shared primary, no promotion
        var rule1 = new GatingRule("CD4T", "CD4", null, null, List.of("CD4", "CD8"));
        var rule2 = new GatingRule("CD8T", "CD8", null, null, List.of("CD4", "CD8"));

        rule1.promoteOverlappingSoftNots(rule2);

        // CD8 should remain -1 in rule1 (no shared primary context)
        assertEquals(-1, rule1.getEncoding("CD8"));
    }

    // ── getters ──────────────────────────────────────────────────────────────

    @Test
    void getCellTypeAndExpressionReturnConstructorValues() {
        var rule = new GatingRule("Macrophage", "CD68|CD163", "CD14", null, List.of("CD68", "CD163", "CD14"));
        assertEquals("Macrophage", rule.getCellType());
        assertEquals("CD68|CD163", rule.getPrimaryExpression());
    }

    @Test
    void getMarkerEncodingIsUnmodifiable() {
        var rule = new GatingRule("T", "CD4", null, null, List.of("CD4", "CD8"));
        assertThrows(
                UnsupportedOperationException.class,
                () -> rule.getMarkerEncoding().put("CD8", 99));
    }

    // ── integration: GatingExpression feeds GatingRule ───────────────────────

    @Test
    void complexExpressionEncodesCorrectly() {
        // "CD4&(CD8|CD3)&!IgA"
        // mustHave: CD4  |  orExpression: CD8, CD3  |  not: IgA
        var rule =
                new GatingRule("Complex", "CD4&(CD8|CD3)&!IgA", null, null, List.of("CD4", "CD8", "CD3", "IgA", "VIM"));
        assertEquals(1, rule.getEncoding("CD4"));
        assertEquals(2, rule.getEncoding("CD8"));
        assertEquals(2, rule.getEncoding("CD3"));
        assertEquals(-3, rule.getEncoding("IgA"));
        assertEquals(-1, rule.getEncoding("VIM"));
    }
}
