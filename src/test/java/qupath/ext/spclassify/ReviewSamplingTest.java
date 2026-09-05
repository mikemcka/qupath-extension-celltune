package qupath.ext.spclassify;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import qupath.ext.spclassify.model.CellPrediction;
import qupath.ext.spclassify.model.PopulationSet;

/**
 * Unit coverage for the pure sampling-pool dedup logic extracted into {@link ReviewSampling}.
 */
class ReviewSamplingTest {

    private static final List<String> CLASSES = List.of("A", "B");

    private static CellPrediction pred(String id) {
        return new CellPrediction(id, "A", "B", new float[] {0.6f, 0.4f}, new float[] {0.3f, 0.7f}, CLASSES);
    }

    private static PopulationSet set(String name, String... ids) {
        PopulationSet ps = new PopulationSet(name);
        for (String id : ids) ps.put(id, pred(id));
        return ps;
    }

    @Test
    void addsAllCellsAndMapsThemToImage() {
        PopulationSet pooled = new PopulationSet("Pred_ALL");
        Map<String, String> cellToImage = new LinkedHashMap<>();

        ReviewSampling.addPredictionsToSamplingPool(pooled, set("src", "c1", "c2"), "slide1.tif", cellToImage);

        assertEquals(2, pooled.size());
        assertEquals("slide1.tif", cellToImage.get("c1"));
        assertEquals("slide1.tif", cellToImage.get("c2"));
    }

    @Test
    void firstImageWinsForDuplicateCellIds() {
        PopulationSet pooled = new PopulationSet("Pred_ALL");
        Map<String, String> cellToImage = new LinkedHashMap<>();

        ReviewSampling.addPredictionsToSamplingPool(pooled, set("a", "c1"), "imgA", cellToImage);
        ReviewSampling.addPredictionsToSamplingPool(pooled, set("b", "c1", "c2"), "imgB", cellToImage);

        assertEquals(2, pooled.size());
        // c1 was already pooled from imgA — second occurrence is skipped, mapping unchanged.
        assertEquals("imgA", cellToImage.get("c1"));
        assertEquals("imgB", cellToImage.get("c2"));
    }

    @Test
    void blankImageNameBecomesPlaceholder() {
        PopulationSet pooled = new PopulationSet("Pred_ALL");
        Map<String, String> cellToImage = new LinkedHashMap<>();
        ReviewSampling.addPredictionsToSamplingPool(pooled, set("s", "c1"), "  ", cellToImage);
        assertEquals("image", cellToImage.get("c1"));
    }

    @Test
    void nullOrEmptySourceIsNoOp() {
        PopulationSet pooled = new PopulationSet("Pred_ALL");
        Map<String, String> cellToImage = new LinkedHashMap<>();
        ReviewSampling.addPredictionsToSamplingPool(pooled, null, "img", cellToImage);
        ReviewSampling.addPredictionsToSamplingPool(pooled, new PopulationSet("empty"), "img", cellToImage);
        assertEquals(0, pooled.size());
        assertNull(cellToImage.get("c1"));
    }
}
