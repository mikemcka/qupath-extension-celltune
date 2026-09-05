package qupath.ext.spclassify.model;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CellTypeTableTest {

    @TempDir
    Path tempDir;

    // ── simple format (in-memory API) ────────────────────────────────────────

    @Test
    void putAndGetMarkersRoundTrip() {
        var tbl = new CellTypeTable();
        tbl.put("T-Cell", List.of("CD3", "CD4"));
        assertEquals(List.of("CD3", "CD4"), tbl.getMarkers("T-Cell"));
    }

    @Test
    void getMarkersReturnsEmptyForUnknownType() {
        var tbl = new CellTypeTable();
        assertTrue(tbl.getMarkers("Unknown").isEmpty());
    }

    @Test
    void getMarkersIsCaseInsensitive() {
        var tbl = new CellTypeTable();
        tbl.put("CD4T", List.of("CD4", "CD3"));
        // The predicted class name may differ only in case from the CSV CellType.
        assertEquals(List.of("CD4", "CD3"), tbl.getMarkers("cd4t"));
        assertEquals(List.of("CD4", "CD3"), tbl.getMarkers("Cd4T"));
    }

    @Test
    void getMarkersToleratesSpacingAndPunctuation() {
        var tbl = new CellTypeTable();
        tbl.put("CD4 T", List.of("CD4", "CD3"));
        assertEquals(List.of("CD4", "CD3"), tbl.getMarkers("CD4T"));
        assertEquals(List.of("CD4", "CD3"), tbl.getMarkers("CD4-T"));
        assertEquals(List.of("CD4", "CD3"), tbl.getMarkers("  cd4_t "));
    }

    @Test
    void getMarkersExactMatchWinsOverNormalized() {
        var tbl = new CellTypeTable();
        tbl.put("CD4 T", List.of("CD4"));
        tbl.put("CD4T", List.of("CD4", "CD3"));
        // Both normalize to "cd4t"; an exact key must still return its own markers.
        assertEquals(List.of("CD4"), tbl.getMarkers("CD4 T"));
        assertEquals(List.of("CD4", "CD3"), tbl.getMarkers("CD4T"));
    }

    @Test
    void getMarkersNullTypeReturnsEmpty() {
        var tbl = new CellTypeTable();
        tbl.put("CD4T", List.of("CD4"));
        assertTrue(tbl.getMarkers(null).isEmpty());
    }

    @Test
    void getMarkersNormalizedFallbackWorksForRuleFormat() {
        var tbl = new CellTypeTable();
        tbl.putRule("CD8 T", "CD8&CD3", null, null);
        var markers = tbl.getMarkers("cd8t");
        assertTrue(markers.contains("CD8"));
        assertTrue(markers.contains("CD3"));
    }

    @Test
    void getCellTypesReturnsAllInsertedTypes() {
        var tbl = new CellTypeTable();
        tbl.put("T-Cell", List.of("CD3"));
        tbl.put("Macrophage", List.of("CD68"));
        var types = tbl.getCellTypes();
        assertEquals(2, types.size());
        assertTrue(types.contains("T-Cell"));
        assertTrue(types.contains("Macrophage"));
    }

    @Test
    void sizeAndIsEmptyReflectState() {
        var tbl = new CellTypeTable();
        assertTrue(tbl.isEmpty());
        assertEquals(0, tbl.size());
        tbl.put("T-Cell", List.of("CD3"));
        assertFalse(tbl.isEmpty());
        assertEquals(1, tbl.size());
    }

    @Test
    void blankMarkersAreSkipped() {
        var tbl = new CellTypeTable();
        tbl.put("T-Cell", List.of("CD3", "", "  ", "CD4"));
        // blank entries should be excluded
        assertEquals(List.of("CD3", "CD4"), tbl.getMarkers("T-Cell"));
    }

    @Test
    void markersAreCappedAtMaxMarkers() {
        var tbl = new CellTypeTable();
        tbl.put("BigType", List.of("M1", "M2", "M3", "M4", "M5", "M6", "M7"));
        assertTrue(tbl.getMarkers("BigType").size() <= CellTypeTable.MAX_MARKERS);
    }

    // ── rule format (in-memory API) ───────────────────────────────────────────

    @Test
    void putRuleEnablesHasGatingRules() {
        var tbl = new CellTypeTable();
        assertFalse(tbl.hasGatingRules());
        tbl.putRule("CD4T", "CD4&CD3", null, null);
        assertTrue(tbl.hasGatingRules());
    }

    @Test
    void putRuleDerivesDisplayMarkersFromPrimaryExpression() {
        var tbl = new CellTypeTable();
        tbl.putRule("CD4T", "CD4&CD3", null, null);
        var markers = tbl.getMarkers("CD4T");
        assertTrue(markers.contains("CD4"));
        assertTrue(markers.contains("CD3"));
    }

    @Test
    void getPrimaryExpressionReturnsCorrectValue() {
        var tbl = new CellTypeTable();
        tbl.putRule("Macro", "CD68|CD163", "CD14", "VIM");
        assertEquals("CD68|CD163", tbl.getPrimaryExpression("Macro"));
        assertEquals("CD14", tbl.getSecondaryMarkers("Macro"));
        assertEquals("VIM", tbl.getTertiaryMarkers("Macro"));
    }

    @Test
    void getPrimaryExpressionReturnsNullForSimpleType() {
        var tbl = new CellTypeTable();
        tbl.put("T-Cell", List.of("CD3"));
        assertNull(tbl.getPrimaryExpression("T-Cell"));
    }

    // ── getAllRuleChannels ────────────────────────────────────────────────────

    @Test
    void getAllRuleChannelsCollectsFromAllRuleExpressions() {
        var tbl = new CellTypeTable();
        tbl.putRule("CD4T", "CD4&CD3", "CD45", "CD103");
        tbl.putRule("Macro", "CD68|CD163", null, "VIM");

        var channels = tbl.getAllRuleChannels();
        assertTrue(channels.contains("CD4"));
        assertTrue(channels.contains("CD3"));
        assertTrue(channels.contains("CD45"));
        assertTrue(channels.contains("CD103"));
        assertTrue(channels.contains("CD68"));
        assertTrue(channels.contains("CD163"));
        assertTrue(channels.contains("VIM"));
    }

    // ── CSV simple format roundtrip ───────────────────────────────────────────

    @Test
    void simpleFormatCsvRoundTrip() throws IOException {
        Path csv = tempDir.resolve("simple.csv");
        String content = """
                CellType,Marker1,Marker2,Marker3
                T-Cell,CD3,CD4,
                Macrophage,CD68,CD163,CD206
                """;
        Files.writeString(csv, content);

        CellTypeTable tbl = CellTypeTable.loadFromCSV(csv);

        assertEquals(2, tbl.size());
        assertFalse(tbl.hasGatingRules());
        assertTrue(tbl.getMarkers("T-Cell").contains("CD3"));
        assertTrue(tbl.getMarkers("T-Cell").contains("CD4"));
        assertTrue(tbl.getMarkers("Macrophage").containsAll(List.of("CD68", "CD163", "CD206")));
    }

    @Test
    void simpleFormatEmptyMarkerColumnsAreSkipped() throws IOException {
        Path csv = tempDir.resolve("sparse.csv");
        Files.writeString(csv, "CellType,Marker1,Marker2,Marker3\nT-Cell,CD3,,\n");

        var tbl = CellTypeTable.loadFromCSV(csv);
        assertEquals(List.of("CD3"), tbl.getMarkers("T-Cell"));
    }

    @Test
    void emptyFileReturnsEmptyTable() throws IOException {
        Path csv = tempDir.resolve("empty.csv");
        Files.writeString(csv, "");

        var tbl = CellTypeTable.loadFromCSV(csv);
        assertTrue(tbl.isEmpty());
    }

    // ── CSV rule format roundtrip ─────────────────────────────────────────────

    @Test
    void ruleFormatCsvRoundTrip() throws IOException {
        Path csv = tempDir.resolve("rules.csv");
        String content = """
                CellType,PrimaryMarker,SecondaryMarker,TertiaryMarker
                CD8T,CD8&CD3,CD45,CD103|CD45RA
                Plasma,CD38&!IgA,,CD45|VIM
                """;
        Files.writeString(csv, content);

        CellTypeTable tbl = CellTypeTable.loadFromCSV(csv);

        assertEquals(2, tbl.size());
        assertTrue(tbl.hasGatingRules());
        assertEquals("CD8&CD3", tbl.getPrimaryExpression("CD8T"));
        assertEquals("CD45", tbl.getSecondaryMarkers("CD8T"));
        assertEquals("CD103|CD45RA", tbl.getTertiaryMarkers("CD8T"));
        assertEquals("CD38&!IgA", tbl.getPrimaryExpression("Plasma"));
        assertNull(tbl.getSecondaryMarkers("Plasma"));
    }

    @Test
    void saveAndReloadSimpleFormatIsConsistent() throws IOException {
        var original = new CellTypeTable();
        original.put("T-Cell", List.of("CD3", "CD4"));
        original.put("NK", List.of("CD56", "CD16"));

        Path csv = tempDir.resolve("save_simple.csv");
        original.saveToCSV(csv);

        var loaded = CellTypeTable.loadFromCSV(csv);

        assertEquals(original.size(), loaded.size());
        assertEquals(original.getMarkers("T-Cell"), loaded.getMarkers("T-Cell"));
        assertEquals(original.getMarkers("NK"), loaded.getMarkers("NK"));
    }

    @Test
    void saveAndReloadRuleFormatIsConsistent() throws IOException {
        var original = new CellTypeTable();
        original.putRule("CD4T", "CD4&CD3", "CD45", null);
        original.putRule("Macro", "CD68|CD163", null, "VIM");

        Path csv = tempDir.resolve("save_rules.csv");
        original.saveToCSV(csv);

        var loaded = CellTypeTable.loadFromCSV(csv);

        assertTrue(loaded.hasGatingRules());
        assertEquals("CD4&CD3", loaded.getPrimaryExpression("CD4T"));
        assertEquals("CD45", loaded.getSecondaryMarkers("CD4T"));
        assertEquals("CD68|CD163", loaded.getPrimaryExpression("Macro"));
        assertEquals("VIM", loaded.getTertiaryMarkers("Macro"));
    }
}
