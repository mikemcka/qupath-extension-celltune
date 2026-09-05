package qupath.ext.spclassify.model;

import static org.junit.jupiter.api.Assertions.*;

import java.util.LinkedHashMap;
import java.util.List;
import org.junit.jupiter.api.Test;

class CohortAnomalyAnalyzerTest {

    @Test
    void detectsRareTypeEnrichmentUsingBaselineAndCountGates() {
        var inputs = List.of(
                input("img-rare", 1000, 120, counts("NK", 60, "T", 940)),
                input("img-b", 5000, 40, counts("T", 5000)),
                input("img-c", 5000, 35, counts("T", 5000)));

        var report = CohortAnomalyAnalyzer.analyze(inputs);
        var byImage = report.byImageName();
        var rare = byImage.get("img-rare");

        assertNotNull(rare);
        assertTrue(rare.flagReasons().contains(CohortAnomalyReport.RARE_ENRICHMENT));
        assertTrue(rare.highlightedClasses().contains("NK"));

        var nk = rare.enrichmentByClass().get("NK");
        assertNotNull(nk);
        assertTrue(nk.rareCandidate());
        assertTrue(nk.highlighted());
        assertTrue(nk.enrichmentFold() >= CohortAnomalyReport.DEFAULT_THRESHOLDS.rareEnrichmentFoldMin());
        assertTrue(nk.count() >= CohortAnomalyReport.DEFAULT_THRESHOLDS.rareMinCount());
    }

    @Test
    void ranksDisagreementAndCompositionOutlierAtTop() {
        var inputs = List.of(
                input("img-1", 1000, 40, counts("A", 500, "B", 500)),
                input("img-2", 1000, 45, counts("A", 510, "B", 490)),
                input("img-3", 1000, 48, counts("A", 495, "B", 505)),
                input("img-4", 1000, 42, counts("A", 505, "B", 495)),
                input("img-5", 1000, 39, counts("A", 498, "B", 502)),
                input("img-6", 1000, 41, counts("A", 503, "B", 497)),
                input("img-outlier", 1000, 700, counts("A", 980, "B", 20)));

        var report = CohortAnomalyAnalyzer.analyze(inputs);
        assertFalse(report.images().isEmpty());

        var top = report.images().get(0);
        assertEquals("img-outlier", top.imageName());
        assertTrue(top.flagged());
        assertTrue(top.anomalyScore() > 0.0);
        assertTrue(top.flagReasons().contains(CohortAnomalyReport.OUTLIER_DISAGREEMENT)
                || top.flagReasons().contains(CohortAnomalyReport.OUTLIER_COMPOSITION));
    }

    @Test
    void keepsBalancedCohortUnflagged() {
        var inputs = List.of(
                input("img-a", 1000, 50, counts("A", 500, "B", 500)),
                input("img-b", 1000, 50, counts("A", 500, "B", 500)),
                input("img-c", 1000, 50, counts("A", 500, "B", 500)),
                input("img-d", 1000, 50, counts("A", 500, "B", 500)));

        var report = CohortAnomalyAnalyzer.analyze(inputs);
        assertEquals(4, report.images().size());

        for (var row : report.images()) {
            assertFalse(row.flagged());
            assertEquals(0.0, row.anomalyScore(), 1e-9);
            assertTrue(row.flagReasons().isEmpty());
        }
    }

    private static CohortAnomalyReport.ImageInput input(
            String imageName, long predicted, long disagreements, LinkedHashMap<String, Long> classCounts) {
        return new CohortAnomalyReport.ImageInput(imageName, predicted, disagreements, classCounts);
    }

    private static LinkedHashMap<String, Long> counts(Object... kv) {
        var out = new LinkedHashMap<String, Long>();
        for (int i = 0; i < kv.length; i += 2) {
            out.put(String.valueOf(kv[i]), ((Number) kv[i + 1]).longValue());
        }
        return out;
    }
}
