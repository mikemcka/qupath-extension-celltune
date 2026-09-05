package qupath.ext.spclassify.ui;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

/**
 * Tests for {@link ChannelSelector#selectChannelIndices(List, List)} — the pure channel-matching
 * logic behind auto channel-switching, exercised without a viewer.
 */
class ChannelSelectorTest {

    @Test
    void exactMatchSelectsSingleChannel() {
        var channels = List.of("DAPI", "CD3", "CD8", "CD20");
        assertEquals(Set.of(1), ChannelSelector.selectChannelIndices(channels, List.of("CD3")));
    }

    @Test
    void exactMatchDoesNotAlsoSelectSuperstringChannels() {
        // The original substring matcher lit up CD31/CD34 when asked for CD3.
        var channels = List.of("CD3", "CD31", "CD34");
        assertEquals(Set.of(0), ChannelSelector.selectChannelIndices(channels, List.of("CD3")));
    }

    @Test
    void fuzzyFallbackMatchesFluorophoreTaggedChannel() {
        // No bare "CD3" channel, so the tagged one is matched via substring fallback.
        var channels = List.of("DAPI", "CD3 (Opal 570)", "CD8");
        assertEquals(Set.of(1), ChannelSelector.selectChannelIndices(channels, List.of("CD3")));
    }

    @Test
    void matchingIsCaseAndPunctuationInsensitive() {
        var channels = List.of("cd3_s2 - cy5", "CD8");
        assertEquals(Set.of(0), ChannelSelector.selectChannelIndices(channels, List.of("CD3_S2-Cy5")));
    }

    @Test
    void multipleMarkersUnionTheirChannels() {
        var channels = List.of("DAPI", "CD3", "CD8", "CD20");
        assertEquals(Set.of(1, 2), ChannelSelector.selectChannelIndices(channels, List.of("CD3", "CD8")));
    }

    @Test
    void noMatchReturnsEmptySoDisplayIsLeftUntouched() {
        var channels = List.of("DAPI", "CD3", "CD8");
        assertTrue(
                ChannelSelector.selectChannelIndices(channels, List.of("FOXP3")).isEmpty());
    }

    @Test
    void blankMarkersAreIgnored() {
        var channels = List.of("DAPI", "CD3");
        assertTrue(ChannelSelector.selectChannelIndices(channels, List.of("", "   "))
                .isEmpty());
    }

    @Test
    void nullArgumentsReturnEmpty() {
        assertTrue(ChannelSelector.selectChannelIndices(null, List.of("CD3")).isEmpty());
        assertTrue(ChannelSelector.selectChannelIndices(List.of("CD3"), null).isEmpty());
    }
}
