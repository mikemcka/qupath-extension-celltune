package qupath.ext.spclassify.io;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests the file-parsing side of {@link GroundTruthIO} (no QuPath objects
 * required): header-layout detection, feature parsing with the non-numeric→0
 * fallback, and the skip rules for comments, blanks, and unlabelled rows.
 */
class GroundTruthIOTest {

    private static Path writeCsv(Path dir, String content) throws Exception {
        Path p = dir.resolve("gt.csv");
        Files.writeString(p, content, StandardCharsets.UTF_8);
        return p;
    }

    @Test
    void importsTrainingRowsWithImageLabelHeader(@TempDir Path tmp) throws Exception {
        Path csv = writeCsv(tmp, """
                Image,Label,CentroidX,CentroidY,F1,F2
                img1,T,10,20,1.5,2.5
                img1,NK,11,21,3.0,4.0
                """);
        List<GroundTruthIO.TrainingRow> rows = GroundTruthIO.importCSVAsTrainingData(csv);

        assertEquals(2, rows.size());
        assertEquals("T", rows.get(0).label());
        assertArrayEquals(new float[] {1.5f, 2.5f}, rows.get(0).features(), 1e-6f);
        assertEquals("NK", rows.get(1).label());
        assertArrayEquals(new float[] {3.0f, 4.0f}, rows.get(1).features(), 1e-6f);
    }

    @Test
    void nonNumericFeatureBecomesZero(@TempDir Path tmp) throws Exception {
        Path csv = writeCsv(tmp, """
                Image,Label,CentroidX,CentroidY,F1,F2
                img1,T,10,20,NA,7.0
                """);
        List<GroundTruthIO.TrainingRow> rows = GroundTruthIO.importCSVAsTrainingData(csv);
        assertEquals(1, rows.size());
        assertArrayEquals(new float[] {0f, 7.0f}, rows.get(0).features(), 1e-6f);
    }

    @Test
    void skipsCommentsBlanksAndUnlabelledRows(@TempDir Path tmp) throws Exception {
        Path csv = writeCsv(tmp, """
                # exported by CellTune
                Image,Label,CentroidX,CentroidY,F1

                img1,T,10,20,1.0
                img1,,11,21,2.0
                img1,NK,12,22,3.0
                """);
        List<GroundTruthIO.TrainingRow> rows = GroundTruthIO.importCSVAsTrainingData(csv);
        assertEquals(2, rows.size(), "the empty-label row must be skipped");
        assertEquals("T", rows.get(0).label());
        assertEquals("NK", rows.get(1).label());
    }

    @Test
    void readFeatureNamesReturnsHeaderTail(@TempDir Path tmp) throws Exception {
        Path csv = writeCsv(tmp, """
                Image,Label,CentroidX,CentroidY,CD3,CD8,Ki67
                img1,T,10,20,1,2,3
                """);
        List<String> names = GroundTruthIO.readFeatureNames(csv);
        assertEquals(List.of("CD3", "CD8", "Ki67"), names);
    }

    @Test
    void unsupportedHeaderThrows(@TempDir Path tmp) throws Exception {
        Path csv = writeCsv(tmp, """
                completely,unrelated,columns
                a,b,c
                """);
        assertThrows(Exception.class, () -> GroundTruthIO.importCSVAsTrainingData(csv));
    }
}
