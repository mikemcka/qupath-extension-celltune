package qupath.ext.spclassify.model;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import org.junit.jupiter.api.Test;

class PopulationSetTest {

    private static final List<String> CLASSES = List.of("T", "NK", "B");

    private static CellPrediction agreed(String cellId, String cls, float conf) {
        float[] probs = new float[3];
        probs[CLASSES.indexOf(cls)] = conf;
        return new CellPrediction(cellId, cls, cls, probs, probs.clone(), CLASSES);
    }

    private static CellPrediction disagreed(String cellId, String m1, String m2) {
        float[] p1 = new float[3];
        p1[CLASSES.indexOf(m1)] = 0.8f;
        float[] p2 = new float[3];
        p2[CLASSES.indexOf(m2)] = 0.8f;
        return new CellPrediction(cellId, m1, m2, p1, p2, CLASSES);
    }

    // ── basic CRUD ───────────────────────────────────────────────────────────

    @Test
    void putAndGetRoundTrip() {
        var ps = new PopulationSet("Pred_ALL");
        var pred = agreed("cell-1", "T", 0.9f);
        ps.put("cell-1", pred);
        assertSame(pred, ps.get("cell-1"));
    }

    @Test
    void getReturnsNullForMissingCell() {
        var ps = new PopulationSet("Pred_ALL");
        assertNull(ps.get("missing"));
    }

    @Test
    void sizeReflectsNumberOfEntries() {
        var ps = new PopulationSet("Pred_ALL");
        assertEquals(0, ps.size());
        ps.put("cell-1", agreed("cell-1", "T", 0.9f));
        ps.put("cell-2", agreed("cell-2", "NK", 0.8f));
        assertEquals(2, ps.size());
    }

    @Test
    void putOverwritesExistingEntry() {
        var ps = new PopulationSet("Pred_ALL");
        ps.put("cell-1", agreed("cell-1", "T", 0.9f));
        var updated = agreed("cell-1", "NK", 0.7f);
        ps.put("cell-1", updated);
        assertEquals(1, ps.size());
        assertSame(updated, ps.get("cell-1"));
    }

    @Test
    void getAllReturnsUnmodifiableView() {
        var ps = new PopulationSet("test");
        ps.put("cell-1", agreed("cell-1", "T", 0.9f));
        var all = ps.getAll();
        assertThrows(UnsupportedOperationException.class, () -> all.put("cell-2", agreed("cell-2", "NK", 0.5f)));
    }

    // ── disagreement filtering ───────────────────────────────────────────────

    @Test
    void getDisagreementsReturnsOnlyDisagreementCells() {
        var ps = new PopulationSet("Pred_ALL");
        ps.put("cell-1", agreed("cell-1", "T", 0.9f));
        ps.put("cell-2", disagreed("cell-2", "T", "NK"));
        ps.put("cell-3", disagreed("cell-3", "NK", "B"));
        ps.put("cell-4", agreed("cell-4", "B", 0.8f));

        var disc = ps.getDisagreements();
        assertEquals(2, disc.size());
        assertTrue(disc.stream().allMatch(CellPrediction::isDisagreement));
    }

    @Test
    void getDisagreementCountMatchesListSize() {
        var ps = new PopulationSet("Pred_ALL");
        ps.put("cell-1", agreed("cell-1", "T", 0.9f));
        ps.put("cell-2", disagreed("cell-2", "T", "NK"));
        ps.put("cell-3", disagreed("cell-3", "B", "NK"));

        assertEquals(2, ps.getDisagreementCount());
        assertEquals(ps.getDisagreements().size(), ps.getDisagreementCount());
    }

    @Test
    void getDisagreementsIsEmptyWhenAllAgree() {
        var ps = new PopulationSet("Pred_ALL");
        ps.put("cell-1", agreed("cell-1", "T", 0.9f));
        ps.put("cell-2", agreed("cell-2", "NK", 0.8f));
        assertTrue(ps.getDisagreements().isEmpty());
        assertEquals(0, ps.getDisagreementCount());
    }

    // ── label-based filtering ────────────────────────────────────────────────

    @Test
    void getByModel1LabelFiltersCorrectly() {
        var ps = new PopulationSet("Pred_ALL");
        ps.put("cell-1", agreed("cell-1", "T", 0.9f));
        ps.put("cell-2", disagreed("cell-2", "T", "NK")); // model1 = T
        ps.put("cell-3", agreed("cell-3", "NK", 0.8f));

        var tPreds = ps.getByModel1Label("T");
        assertEquals(2, tPreds.size());
        assertTrue(tPreds.stream().allMatch(p -> "T".equals(p.getModel1Label())));
    }

    @Test
    void getByModel2LabelFiltersCorrectly() {
        var ps = new PopulationSet("Pred_ALL");
        ps.put("cell-1", disagreed("cell-1", "T", "NK")); // model2 = NK
        ps.put("cell-2", agreed("cell-2", "NK", 0.8f)); // model2 = NK
        ps.put("cell-3", agreed("cell-3", "T", 0.9f)); // model2 = T

        var nkPreds = ps.getByModel2Label("NK");
        assertEquals(2, nkPreds.size());
    }

    // ── count maps ───────────────────────────────────────────────────────────

    @Test
    void getModel1CountsAggregatesCorrectly() {
        var ps = new PopulationSet("Pred_ALL");
        ps.put("c1", agreed("c1", "T", 0.9f));
        ps.put("c2", agreed("c2", "T", 0.8f));
        ps.put("c3", agreed("c3", "NK", 0.7f));

        var counts = ps.getModel1Counts();
        assertEquals(2L, counts.get("T"));
        assertEquals(1L, counts.get("NK"));
        assertNull(counts.get("B"));
    }

    @Test
    void getAvgCountsAggregatesCorrectly() {
        var ps = new PopulationSet("Pred_ALL");
        // Both probs identical → avgLabel = that class
        ps.put("c1", agreed("c1", "T", 0.9f));
        ps.put("c2", agreed("c2", "B", 0.9f));
        ps.put("c3", agreed("c3", "B", 0.8f));

        var counts = ps.getAvgCounts();
        assertEquals(1L, counts.get("T"));
        assertEquals(2L, counts.get("B"));
    }
}
