package qupath.ext.spclassify.io;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import qupath.ext.spclassify.model.ImagePixelStats;
import qupath.ext.spclassify.model.PixelCohortAnalyzer;
import qupath.ext.spclassify.model.PixelCohortReport;

class PixelStatsCsvExporterTest {

    private static ImagePixelStats.ImageStats image(String name, double median, double fg) {
        var channels = List.of(
                new ImagePixelStats.ChannelStats(
                        "DAPI",
                        100L,
                        median,
                        5.0,
                        0.0,
                        median,
                        median + 50,
                        median - 5,
                        median + 40,
                        0.0,
                        20.0,
                        1.0 - fg,
                        fg,
                        45.0,
                        100.0),
                new ImagePixelStats.ChannelStats(
                        "CD8",
                        100L,
                        median,
                        5.0,
                        0.0,
                        median,
                        median + 50,
                        median - 5,
                        median + 40,
                        0.0,
                        20.0,
                        1.0 - fg,
                        fg,
                        45.0,
                        100.0));
        return new ImagePixelStats.ImageStats(name, 4.0, 64, 64, channels, 1.0 - fg);
    }

    @Test
    void writesHeaderWithImageAndPerChannelColumns() {
        var inputs = new ArrayList<ImagePixelStats.ImageStats>();
        for (int i = 0; i < 5; i++) {
            inputs.add(image("img-" + i, 100.0 + i, 0.6));
        }
        var report = PixelCohortAnalyzer.analyze(inputs);

        String csv = PixelStatsCsvExporter.toCsv(report);
        String[] lines = csv.split("\\n");

        String header = lines[0];
        assertTrue(header.startsWith("Image,Verdict,Flags,Score,EmptyFraction"), header);
        assertTrue(header.contains("\"DAPI: Median\""), header);
        assertTrue(header.contains("\"CD8: SaturationFraction\""), header);
        // One header + five image rows.
        assertEquals(6, lines.length);
    }

    @Test
    void emptyReportProducesHeaderOnly() {
        String csv = PixelStatsCsvExporter.toCsv(new PixelCohortReport(List.of()));
        String[] lines = csv.split("\\n");
        assertEquals(1, lines.length);
        assertTrue(lines[0].startsWith("Image,Verdict,Flags,Score"));
    }

    @Test
    void escapesImageNamesWithCommas() {
        var report = PixelCohortAnalyzer.analyze(List.of(image("img,with,commas", 100.0, 0.6)));
        String csv = PixelStatsCsvExporter.toCsv(report);
        assertTrue(csv.contains("\"img,with,commas\""), csv);
    }
}
