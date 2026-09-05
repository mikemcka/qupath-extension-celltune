package qupath.ext.spclassify.classifier;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import org.junit.jupiter.api.Test;

class CompositeClassificationRuleTest {

    // ── parse roundtrip ──────────────────────────────────────────────────────

    @Test
    void parseRoundTripPreservesConditions() {
        var rule = CompositeClassificationRule.parse("CD4 T-cell", "CD4+:CD3+:IgA-");
        assertEquals("CD4 T-cell", rule.name());
        assertEquals("CD4+:CD3+:IgA-", rule.expression());
        assertEquals(3, rule.conditions().size());
    }

    @Test
    void parseConditionsHaveCorrectPolarities() {
        var rule = CompositeClassificationRule.parse("Rule", "CD4+:CD3+:IgA-");
        var conditions = rule.conditions();
        assertEquals("CD4", conditions.get(0).markerName());
        assertEquals(
                CompositeClassificationRule.Polarity.POSITIVE, conditions.get(0).polarity());
        assertEquals("CD3", conditions.get(1).markerName());
        assertEquals(
                CompositeClassificationRule.Polarity.POSITIVE, conditions.get(1).polarity());
        assertEquals("IgA", conditions.get(2).markerName());
        assertEquals(
                CompositeClassificationRule.Polarity.NEGATIVE, conditions.get(2).polarity());
    }

    @Test
    void ofFactoryMatchesParseResult() {
        var conditions = List.of(
                CompositeClassificationRule.MarkerCondition.of("CD4", CompositeClassificationRule.Polarity.POSITIVE),
                CompositeClassificationRule.MarkerCondition.of("IgA", CompositeClassificationRule.Polarity.NEGATIVE));
        var rule = CompositeClassificationRule.of("Plasma cell", conditions);
        assertEquals("CD4+:IgA-", rule.expression());
    }

    @Test
    void formatExpressionRoundTrip() {
        String expr = "CD4+:CD3+:IgA-";
        var conditions = CompositeClassificationRule.parseExpression(expr);
        assertEquals(expr, CompositeClassificationRule.formatExpression(conditions));
    }

    // ── validation ──────────────────────────────────────────────────────────

    @Test
    void blankNameThrows() {
        assertThrows(IllegalArgumentException.class, () -> CompositeClassificationRule.parse("", "CD4+"));
        assertThrows(IllegalArgumentException.class, () -> CompositeClassificationRule.parse("  ", "CD4+"));
    }

    @Test
    void blankExpressionThrows() {
        assertThrows(IllegalArgumentException.class, () -> CompositeClassificationRule.parse("Rule", ""));
        assertThrows(IllegalArgumentException.class, () -> CompositeClassificationRule.parse("Rule", "  "));
    }

    @Test
    void duplicateMarkerInRuleThrows() {
        assertThrows(IllegalArgumentException.class, () -> CompositeClassificationRule.parse("Rule", "CD4+:CD4-"));
    }

    @Test
    void duplicateMarkerCaseInsensitiveThrows() {
        assertThrows(IllegalArgumentException.class, () -> CompositeClassificationRule.parse("Rule", "CD4+:cd4-"));
    }

    @Test
    void markerWithUnsupportedCharactersThrows() {
        // Space in marker name triggers sanitize mismatch
        assertThrows(IllegalArgumentException.class, () -> CompositeClassificationRule.parse("Rule", "CD 4+"));
    }

    @Test
    void emptyConditionsListThrows() {
        assertThrows(IllegalArgumentException.class, () -> CompositeClassificationRule.of("Rule", List.of()));
    }

    @Test
    void nullConditionInListThrows() {
        List<CompositeClassificationRule.MarkerCondition> conditions = new java.util.ArrayList<>();
        conditions.add(null);
        assertThrows(IllegalArgumentException.class, () -> CompositeClassificationRule.of("Rule", conditions));
    }

    // ── Polarity fromToken ───────────────────────────────────────────────────

    @Test
    void polarityFromTokenRecognizesAllVariants() {
        assertEquals(
                CompositeClassificationRule.Polarity.POSITIVE, CompositeClassificationRule.Polarity.fromToken("+"));
        assertEquals(
                CompositeClassificationRule.Polarity.POSITIVE, CompositeClassificationRule.Polarity.fromToken("pos"));
        assertEquals(
                CompositeClassificationRule.Polarity.POSITIVE,
                CompositeClassificationRule.Polarity.fromToken("positive"));
        assertEquals(
                CompositeClassificationRule.Polarity.NEGATIVE, CompositeClassificationRule.Polarity.fromToken("-"));
        assertEquals(
                CompositeClassificationRule.Polarity.NEGATIVE, CompositeClassificationRule.Polarity.fromToken("neg"));
        assertEquals(
                CompositeClassificationRule.Polarity.NEGATIVE,
                CompositeClassificationRule.Polarity.fromToken("negative"));
    }

    @Test
    void polarityFromTokenThrowsOnInvalidInput() {
        assertThrows(IllegalArgumentException.class, () -> CompositeClassificationRule.Polarity.fromToken("unknown"));
        assertThrows(IllegalArgumentException.class, () -> CompositeClassificationRule.Polarity.fromToken(""));
        assertThrows(IllegalArgumentException.class, () -> CompositeClassificationRule.Polarity.fromToken(null));
    }

    @Test
    void polarityFromSymbolWorks() {
        assertEquals(
                CompositeClassificationRule.Polarity.POSITIVE, CompositeClassificationRule.Polarity.fromSymbol('+'));
        assertEquals(
                CompositeClassificationRule.Polarity.NEGATIVE, CompositeClassificationRule.Polarity.fromSymbol('-'));
    }

    @Test
    void polarityFromSymbolThrowsOnInvalidChar() {
        assertThrows(IllegalArgumentException.class, () -> CompositeClassificationRule.Polarity.fromSymbol('x'));
    }

    // ── toString ─────────────────────────────────────────────────────────────

    @Test
    void toStringIncludesNameAndExpression() {
        var rule = CompositeClassificationRule.parse("THelper", "CD4+:CD3+");
        String s = rule.toString();
        assertTrue(s.contains("THelper"));
        assertTrue(s.contains("CD4+:CD3+"));
    }

    // ── single-condition rule ─────────────────────────────────────────────────

    @Test
    void singleConditionRuleIsValid() {
        var rule = CompositeClassificationRule.parse("Positive Only", "CD4+");
        assertEquals(1, rule.conditions().size());
        assertEquals("CD4+", rule.expression());
    }
}
