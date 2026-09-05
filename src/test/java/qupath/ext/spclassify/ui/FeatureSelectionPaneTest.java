package qupath.ext.spclassify.ui;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class FeatureSelectionPaneTest {

    @Test
    void markerIntensityFeaturesGroupUnderTheirMarker() {
        assertEquals("DAPI_AF", FeatureSelectionPane.groupOf("DAPI_AF: Cell: Mean"));
        assertEquals("DAPI_AF", FeatureSelectionPane.groupOf("DAPI_AF: Cytoplasm: Std.Dev."));
        assertEquals("DAPI_AF", FeatureSelectionPane.groupOf("DAPI_AF: Cell: Percentile: 70.0"));
        assertEquals("DAPI_AF", FeatureSelectionPane.groupOf("DAPI_AF: Cell: ErosionBin_1: Mean"));
        assertEquals("mCherry_S1 - TRITC_AF", FeatureSelectionPane.groupOf("mCherry_S1 - TRITC_AF: Membrane: Median"));
        assertEquals("PD-L1_S13 - TRITC_AF", FeatureSelectionPane.groupOf("PD-L1_S13 - TRITC_AF: Cell: Max"));
    }

    @Test
    void compartmentOnlyFeaturesAreMorphology() {
        assertEquals(FeatureSelectionPane.GROUP_MORPHOLOGY, FeatureSelectionPane.groupOf("Cell: Area µm^2"));
        assertEquals(FeatureSelectionPane.GROUP_MORPHOLOGY, FeatureSelectionPane.groupOf("Cell: Circularity"));
        assertEquals(FeatureSelectionPane.GROUP_MORPHOLOGY, FeatureSelectionPane.groupOf("Cell: Solidity"));
        assertEquals(
                FeatureSelectionPane.GROUP_MORPHOLOGY, FeatureSelectionPane.groupOf("Cell: ErosionBin_1: Area_px"));
        assertEquals(
                FeatureSelectionPane.GROUP_MORPHOLOGY,
                FeatureSelectionPane.groupOf("Cell: ErosionBin_2: Area_Fraction"));
    }

    @Test
    void embeddingFeaturesAreDetectedAcrossNamingStyles() {
        assertTrue(FeatureSelectionPane.isEmbedding("UMAP 1"));
        assertTrue(FeatureSelectionPane.isEmbedding("Embedding_0"));
        assertTrue(FeatureSelectionPane.isEmbedding("PCA_2"));
        assertTrue(FeatureSelectionPane.isEmbedding("tSNE 3"));
        assertTrue(FeatureSelectionPane.isEmbedding("kronos_emb_0"));
        assertTrue(FeatureSelectionPane.isEmbedding("kronos_emb_10"));
        assertEquals(FeatureSelectionPane.GROUP_EMBEDDINGS, FeatureSelectionPane.groupOf("UMAP 1"));
        assertEquals(FeatureSelectionPane.GROUP_EMBEDDINGS, FeatureSelectionPane.groupOf("kronos_emb_5"));
        // Markers that merely contain the letters must not be misclassified.
        assertFalse(FeatureSelectionPane.isEmbedding("PECAM_S1 - Cy5_AF: Cell: Mean"));
        assertFalse(FeatureSelectionPane.isEmbedding("DAPI_AF: Cell: Mean"));
    }

    @Test
    void neighborFeaturesGroupUnderNeighborsAndKeepFullLabel() {
        assertEquals(
                FeatureSelectionPane.GROUP_NEIGHBORS, FeatureSelectionPane.groupOf("Neighbors: Mean: Cell: Area µm^2"));
        assertEquals(
                FeatureSelectionPane.GROUP_NEIGHBORS,
                FeatureSelectionPane.groupOf("Neighbors: Mean: DAPI: Cell: Mean"));
        // The "Neighbors:" prefix must be retained in the displayed label.
        assertEquals(
                "Neighbors: Mean: Cell: Area µm^2",
                FeatureSelectionPane.leafLabel(
                        "Neighbors: Mean: Cell: Area µm^2", FeatureSelectionPane.GROUP_NEIGHBORS));
    }

    @Test
    void unrecognizedNamesFallIntoOtherNotMorphology() {
        // Names with no ": " separator and no embedding token are uncategorized.
        assertEquals(FeatureSelectionPane.GROUP_OTHER, FeatureSelectionPane.groupOf("my_custom_score"));
        assertEquals(FeatureSelectionPane.GROUP_OTHER, FeatureSelectionPane.groupOf("SomeRandomFeature"));
        assertEquals(FeatureSelectionPane.GROUP_OTHER, FeatureSelectionPane.groupOf(null));
        // The full name is kept as the label for the Other group.
        assertEquals(
                "my_custom_score", FeatureSelectionPane.leafLabel("my_custom_score", FeatureSelectionPane.GROUP_OTHER));
    }

    @Test
    void leafLabelStripsMarkerPrefixOnlyForMarkerGroups() {
        assertEquals("Cell: Mean", FeatureSelectionPane.leafLabel("DAPI_AF: Cell: Mean", "DAPI_AF"));
        assertEquals(
                "Membrane: Std.Dev.",
                FeatureSelectionPane.leafLabel("mCherry_S1 - TRITC_AF: Membrane: Std.Dev.", "mCherry_S1 - TRITC_AF"));
        // Morphology / embedding labels are left intact.
        assertEquals(
                "Cell: Area µm^2",
                FeatureSelectionPane.leafLabel("Cell: Area µm^2", FeatureSelectionPane.GROUP_MORPHOLOGY));
    }
}
