package qupath.ext.spclassify.io;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/**
 * CSV writer for project prediction summary rows.
 */
public final class ProjectSummaryCsvExporter {

    private ProjectSummaryCsvExporter() {}

    public record Row(
            String imageName,
            long predictedCells,
            long agreements,
            long disagreements,
            String agreementRate,
            double anomalyScore,
            boolean flagged,
            String flagReasons,
            String rareEnrichment,
            String classCounts) {}

    public static String toCsv(List<Row> rows) {
        var safeRows = rows == null ? List.<Row>of() : List.copyOf(rows);

        StringBuilder sb = new StringBuilder();
        sb.append(
                "Image,Predicted,Agreements,Disagreements,AgreementPercent,AnomalyScore,Flagged,FlagReasons,RareEnrichment,ClassCounts\n");

        for (var row : safeRows) {
            sb.append(csvEscape(row.imageName()))
                    .append(',')
                    .append(row.predictedCells())
                    .append(',')
                    .append(row.agreements())
                    .append(',')
                    .append(row.disagreements())
                    .append(',')
                    .append(csvEscape(row.agreementRate()))
                    .append(',')
                    .append(String.format(Locale.ROOT, "%.4f", row.anomalyScore()))
                    .append(',')
                    .append(row.flagged())
                    .append(',')
                    .append(csvEscape(row.flagReasons()))
                    .append(',')
                    .append(csvEscape(row.rareEnrichment()))
                    .append(',')
                    .append(csvEscape(row.classCounts()))
                    .append('\n');
        }

        return sb.toString();
    }

    public static void writeCsv(Path outputPath, List<Row> rows) throws IOException {
        Files.writeString(outputPath, toCsv(rows), StandardCharsets.UTF_8);
    }

    private static String csvEscape(String value) {
        return CsvUtils.quoteAlways(value);
    }
}
