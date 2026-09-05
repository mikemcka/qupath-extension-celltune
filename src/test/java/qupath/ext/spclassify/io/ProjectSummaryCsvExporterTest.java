package qupath.ext.spclassify.io;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class ProjectSummaryCsvExporterTest {

    @Test
    void exportsExpectedHeaderAndColumnOrder() {
        var rows = List.of(new ProjectSummaryCsvExporter.Row(
                "img-a",
                100,
                90,
                10,
                "90.0%",
                1.25,
                true,
                "HIGH_DISAGREEMENT",
                "NK (count=25, fold=5.10x)",
                "NK: 25, T: 75"));

        String csv = ProjectSummaryCsvExporter.toCsv(rows);
        String[] lines = csv.split("\\n");

        assertTrue(
                lines[0].startsWith(
                        "Image,Predicted,Agreements,Disagreements,AgreementPercent,AnomalyScore,Flagged,FlagReasons,RareEnrichment,ClassCounts"));
        assertTrue(lines[1].contains("\"img-a\""));
        assertTrue(lines[1].contains("1.2500"));
    }

    @Test
    void escapesQuotesAndCommasAndPreservesInputOrder() {
        var row1 = new ProjectSummaryCsvExporter.Row(
                "img,one", 10, 9, 1, "90.0%", 0.5, false, "-", "No highlighted rare classes.", "A: 5, B: 5");
        var row2 = new ProjectSummaryCsvExporter.Row(
                "img-two",
                20,
                10,
                10,
                "50.0%",
                2.75,
                true,
                "RARE_ENRICHMENT, COMPOSITION_OUTLIER",
                "Treg \"hot\" region",
                "Treg: 20");

        String csv = ProjectSummaryCsvExporter.toCsv(List.of(row1, row2));
        String[] lines = csv.split("\\n");

        assertTrue(lines[1].startsWith("\"img,one\""));
        assertTrue(lines[2].startsWith("\"img-two\""));
        assertTrue(lines[2].contains("\"Treg \"\"hot\"\" region\""));
        assertEquals(3, lines.length);
    }
}
