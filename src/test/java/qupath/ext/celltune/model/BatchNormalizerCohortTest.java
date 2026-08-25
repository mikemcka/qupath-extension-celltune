package qupath.ext.celltune.model;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

/**
 * Tests for the pure, package-visible helpers of {@link BatchNormalizerCohort} (the
 * streaming fit itself needs QuPath images and is exercised via UAT). Focused on
 * {@code channelOf}, which groups a channel's stats so they share one gain.
 */
class BatchNormalizerCohortTest {

    @Test
    void channelOfParsesTheTokenBeforeFirstColonSpace() {
        assertEquals("CD3", BatchNormalizerCohort.channelOf("CD3: Cell: Mean"));
        assertEquals("CD3", BatchNormalizerCohort.channelOf("CD3: Nucleus: Median"));
        assertEquals("CD3", BatchNormalizerCohort.channelOf("CD3: Cell: Percentile: 99"));
        assertEquals("CD3", BatchNormalizerCohort.channelOf("CD3: Cell: ErosionBin_2: Mean"));
        assertEquals("CSL_aSMA - TRITC_AF", BatchNormalizerCohort.channelOf("CSL_aSMA - TRITC_AF: Cell: Mean"));
    }

    @Test
    void channelOfReturnsWholeNameWhenNoColonSpace() {
        assertEquals("Area", BatchNormalizerCohort.channelOf("Area"));
    }
}
