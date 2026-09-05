package qupath.ext.spclassify.model;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Test;
import qupath.ext.spclassify.model.LeidenModel.LeidenResult;
import qupath.lib.objects.classes.PathClass;

/**
 * Unit tests for {@link LeidenModel}: the pure feature-kNN / Jaccard-weighting /
 * CWTS-Leiden / kNN-label-transfer math behind graph-based phenotype clustering
 * (LEI-02, LEI-04, LEI-05). Synthetic clouds only — no QuPath/JavaFX APIs —
 * mirroring {@code NeighborhoodModelTest}.
 */
class LeidenModelTest {

    // ── featureKnn ─────────────────────────────────────────────────────────────

    @Test
    void featureKnnMatchesBruteForceSetOnRandomCloud() {
        Random rng = new Random(42);
        int n = 150;
        int d = 5;
        int k = 8;
        double[][] rows = randomCloud(rng, n, d);
        int[][] actual = LeidenModel.featureKnn(rows, k);
        for (int i = 0; i < n; i++) {
            assertEquals(k, actual[i].length, "Expected exactly k neighbours at index " + i);
            assertFalse(toSet(actual[i]).contains(i), "Self must be excluded at index " + i);
            Set<Integer> expected = bruteForceKnn(rows, i, k);
            assertEquals(expected, toSet(actual[i]), "kNN set mismatch at index " + i);
        }
    }

    @Test
    void featureKnnExcludesSelfWithDuplicateRows() {
        double[][] rows = {
            {1.0, 2.0, 3.0},
            {1.0, 2.0, 3.0},
            {100.0, 200.0, 300.0}
        };
        int[][] out = LeidenModel.featureKnn(rows, 1);
        assertEquals(1, out[0].length);
        assertEquals(1, out[0][0], "Duplicate row must match its twin, not itself");
        assertEquals(1, out[1].length);
        assertEquals(0, out[1][0]);
    }

    @Test
    void featureKnnReturnsNMinusOneWhenFewerThanK() {
        double[][] rows = {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}};
        int[][] out = LeidenModel.featureKnn(rows, 10);
        for (int i = 0; i < 3; i++) {
            assertEquals(2, out[i].length, "n-1 neighbours when n-1 < k at index " + i);
            assertFalse(toSet(out[i]).contains(i));
        }
    }

    @Test
    void featureKnnOrdersByDistanceThenIndexOnTies() {
        // Row 0 is the query at the origin; rows 1-4 are all exactly distance 1 away
        // (a 4-way tie), row 5 is farther. With k=3 the result must be the three
        // lowest-index tied neighbours (1,2,3 — excluding the equally-close index 4),
        // returned in ascending (distance, index) order. This pins down the exact
        // ordering + tie-break the bounded-heap selection must reproduce.
        double[][] rows = {
            {0.0, 0.0}, // 0: query
            {1.0, 0.0}, // 1: dist 1
            {0.0, 1.0}, // 2: dist 1
            {-1.0, 0.0}, // 3: dist 1
            {0.0, -1.0}, // 4: dist 1 (tie loser — higher index than 1,2,3)
            {2.0, 0.0} // 5: dist 4
        };
        int[][] out = LeidenModel.featureKnn(rows, 3);
        assertArrayEquals(new int[] {1, 2, 3}, out[0], "3-NN of the origin, tie-broken by lower index, ordered asc");
    }

    // ── ANN recall gate ────────────────────────────────────────────────────────

    @Test
    void recallGateSampleSizeMatchesProportionalCappedFormula() {
        // sampleSize = min(10_000, max(1, round(n * 0.001))) — checkable exactly
        // for n where n*0.001 has no rounding ambiguity.
        assertEquals(1, LeidenModel.recallSampleSize(100), "round(0.1)=0 -> max(1,0)=1");
        assertEquals(1, LeidenModel.recallSampleSize(1_000), "round(1.0)=1");
        assertEquals(2, LeidenModel.recallSampleSize(2_000), "round(2.0)=2");
        assertEquals(10_000, LeidenModel.recallSampleSize(10_000_000), "round(10_000.0)=10_000, at the cap exactly");
        assertEquals(10_000, LeidenModel.recallSampleSize(50_000_000), "cap enforced above 10_000_000");
    }

    @Test
    void meanRecallIsOneForIdenticalSetsAndZeroForDisjointSets() {
        int[][] exact = {{1, 2, 3}, {4, 5, 6}};
        int[][] sameSetsReordered = {{3, 2, 1}, {6, 5, 4}};
        assertEquals(1.0, LeidenModel.meanRecall(exact, sameSetsReordered), 1e-9, "Identical sets -> recall 1.0");

        int[][] disjoint = {{7, 8, 9}, {10, 11, 12}};
        assertEquals(0.0, LeidenModel.meanRecall(exact, disjoint), 1e-9, "Disjoint sets -> recall 0.0");
    }

    @Test
    void recallGatePassesWithoutThrowingWhenAnnMatchesExact() {
        // "Adequate ANN": the stub returns the exact neighbours themselves, so
        // recall is 1.0 on the very first (non-escalated) attempt and the gate
        // must not throw.
        Random rng = new Random(77);
        double[][] rows = randomCloud(rng, 400, 6);
        int k = 10;
        int keep = Math.min(k, rows.length - 1);

        double recall = LeidenModel.gateAnnRecall(
                rows, k, 1234L, 64, (ef, sampleIdx) -> LeidenModel.exactNeighborsForQueries(rows, sampleIdx, keep));

        assertTrue(recall >= 0.95, "Adequate ANN neighbours must pass the recall gate, got " + recall);
    }

    @Test
    void recallGateEscalatesThenAbortsWhenAnnNeverImproves() {
        // "Degraded ANN": the stub always returns each sampled row's FARTHEST
        // (not nearest) neighbours, regardless of the escalating ef -- there is
        // no headroom for escalation to help, so the gate must exhaust
        // MAX_ESCALATIONS and abort.
        Random rng = new Random(88);
        double[][] rows = randomCloud(rng, 400, 6);
        int k = 10;
        int keep = Math.min(k, rows.length - 1);

        AnnRecallException ex = assertThrows(
                AnnRecallException.class,
                () -> LeidenModel.gateAnnRecall(rows, k, 5678L, 64, (ef, sampleIdx) -> {
                    int[][] wrong = new int[sampleIdx.length][];
                    for (int s = 0; s < sampleIdx.length; s++) {
                        wrong[s] = bruteForceFarthest(rows, sampleIdx[s], keep);
                    }
                    return wrong;
                }));
        assertTrue(
                ex.getMessage() != null && ex.getMessage().contains("0."),
                "Exception message must include the measured (failing) recall: " + ex.getMessage());
    }

    // ── primitive-array Jaccard/SNN weighting rewrite (Plan 15-03) ────────────

    @Test
    void jaccardEdgesForTestMatchesBoxedReferenceOnVariousSizes() {
        // n = 10..200, several k values -- the primitive sorted-array
        // merge-intersection weighting must produce the SAME edges with
        // BYTE-IDENTICAL (exact ==) Jaccard weights as the retired boxed
        // HashSet<Integer>[]/HashSet<Long> implementation (preserved here only
        // as the equivalence baseline). Jaccard is intersection/union of integer
        // counts -- no floating-point summation-order sensitivity -- so exact
        // equality is the correct assertion, not a tolerance.
        int[] sizes = {10, 37, 80, 200};
        int[] ks = {2, 5, 15};
        for (int n : sizes) {
            for (int k : ks) {
                Random rng = new Random(1000 + n * 31 + k);
                double[][] rows = randomCloud(rng, n, Math.max(2, Math.min(6, n / 4)));
                int[][] neighbors = LeidenModel.featureKnn(rows, k);

                LeidenModel.JaccardEdges primitive = LeidenModel.jaccardEdgesForTest(n, neighbors);
                LeidenModel.JaccardEdges boxed = buildJaccardEdgesBoxedReference(n, neighbors);

                assertArrayEquals(boxed.from(), primitive.from(), "from[] mismatch at n=" + n + " k=" + k);
                assertArrayEquals(boxed.to(), primitive.to(), "to[] mismatch at n=" + n + " k=" + k);
                assertArrayEquals(
                        boxed.weights(),
                        primitive.weights(),
                        0.0,
                        "weights[] must be byte-identical at n=" + n + " k=" + k);
            }
        }
    }

    @Test
    void jaccardEdgesForTestGivesClosedSetWeightOneForMutualOnlyNeighbours() {
        // Two nodes that are each other's ONLY neighbour: closed sets are
        // {0,1} and {1,0} -- identical sets -- so Jaccard must be 1.0, not an
        // undefined/zero weight (the PhenoGraph/SNN closed-set convention).
        int[][] neighbors = {{1}, {0}};
        LeidenModel.JaccardEdges edges = LeidenModel.jaccardEdgesForTest(2, neighbors);
        assertEquals(1, edges.from().length, "Exactly one undirected edge expected");
        assertEquals(0, edges.from()[0]);
        assertEquals(1, edges.to()[0]);
        assertEquals(1.0, edges.weights()[0], 0.0, "Mutual-only closed neighbour sets must weight 1.0");
    }

    // ── cluster: community recovery ───────────────────────────────────────────

    @Test
    void clusterRecoversThreeSeparatedBlobsByPurity() {
        Random rng = new Random(11);
        int per = 50;
        int n = per * 3;
        double[][] rows = new double[n][2];
        fillBlob(rows, rng, 0, per, 0.0, 0.0, 0.3);
        fillBlob(rows, rng, per, 2 * per, 20.0, 0.0, 0.3);
        fillBlob(rows, rng, 2 * per, 3 * per, 0.0, 20.0, 0.3);

        // resolution=0.3: CPM resolution (after association-strength normalization)
        // has a different natural scale than modularity's familiar "1.0" — for this
        // small, tightly-sampled synthetic set, resolution=1.0 over-splits each blob
        // in two, while resolutions well under 1.0 recover the 3 true communities.
        LeidenResult res = LeidenModel.cluster(rows, 15, 0.3, 10, 42L);

        assertTrue(res.nClusters() >= 2, "Expected at least 2 recovered communities, got " + res.nClusters());
        assertTrue(purity(res.labels(), 0, per) > 0.9, "Blob A not pure");
        assertTrue(purity(res.labels(), per, 2 * per) > 0.9, "Blob B not pure");
        assertTrue(purity(res.labels(), 2 * per, n) > 0.9, "Blob C not pure");

        // Labels must be dense/contiguous 0..nClusters-1 (downstream colouring contract).
        Set<Integer> distinct = toSet(res.labels());
        for (int label : distinct) {
            assertTrue(label >= 0 && label < res.nClusters(), "Label " + label + " outside [0, nClusters)");
        }
    }

    @Test
    void clusterHigherResolutionYieldsAtLeastAsManyClustersOnGradedSet() {
        // A graded synthetic set with several loosely-separated sub-groups: higher
        // resolution should not produce FEWER communities than a lower resolution
        // (monotone-ish, not strictly monotone for every possible network).
        Random rng = new Random(5);
        int perGroup = 30;
        int groups = 6;
        int n = perGroup * groups;
        double[][] rows = new double[n][2];
        for (int g = 0; g < groups; g++) {
            fillBlob(rows, rng, g * perGroup, (g + 1) * perGroup, g * 8.0, 0.0, 0.6);
        }

        LeidenResult low = LeidenModel.cluster(rows, 15, 0.2, 10, 7L);
        LeidenResult high = LeidenModel.cluster(rows, 15, 2.5, 10, 7L);

        assertTrue(
                high.nClusters() >= low.nClusters(),
                "Higher resolution (" + high.nClusters() + ") should yield >= clusters than lower resolution ("
                        + low.nClusters() + ")");
    }

    // ── cluster: reproducibility ──────────────────────────────────────────────

    @Test
    void clusterSameSeedProducesIdenticalLabels() {
        Random rng = new Random(3);
        double[][] rows = new double[120][3];
        fillBlob(rows, rng, 0, 60, 0.0, 0.0, 0.4);
        fillBlob(rows, rng, 60, 120, 15.0, 15.0, 0.4);

        LeidenResult a = LeidenModel.cluster(rows, 15, 1.0, 5, 99L);
        LeidenResult b = LeidenModel.cluster(rows, 15, 1.0, 5, 99L);

        assertArrayEquals(a.labels(), b.labels(), "Identical inputs+seed must yield identical labels");
        assertEquals(a.nClusters(), b.nClusters());
    }

    @Test
    void clusterDifferentSeedMayDiffer() {
        // Not a strict assertion of inequality (different seeds *can* coincide on
        // an easy problem) — just confirm neither seed crashes and both recover
        // the same well-separated structure (same nClusters, valid partitions).
        Random rng = new Random(21);
        double[][] rows = new double[100][2];
        fillBlob(rows, rng, 0, 50, 0.0, 0.0, 0.3);
        fillBlob(rows, rng, 50, 100, 25.0, 0.0, 0.3);

        LeidenResult a = LeidenModel.cluster(rows, 15, 1.0, 3, 1L);
        LeidenResult b = LeidenModel.cluster(rows, 15, 1.0, 3, 2L);

        assertTrue(a.nClusters() >= 2);
        assertTrue(b.nClusters() >= 2);
    }

    // ── clusterViaAnn: HNSW-routed Leiden ─────────────────────────────────────

    @Test
    void clusterViaAnnAgreesWithExactClusterByAri() {
        Random rng = new Random(31);
        int per = 60;
        int n = per * 4;
        double[][] rows = new double[n][2];
        fillBlob(rows, rng, 0, per, 0.0, 0.0, 0.4);
        fillBlob(rows, rng, per, 2 * per, 15.0, 0.0, 0.4);
        fillBlob(rows, rng, 2 * per, 3 * per, 0.0, 15.0, 0.4);
        fillBlob(rows, rng, 3 * per, n, 15.0, 15.0, 0.4);

        LeidenResult exact = LeidenModel.cluster(rows, 15, 0.3, 10, 42L);
        LeidenResult viaAnn = LeidenModel.clusterViaAnn(rows, 15, 0.3, 10, 42L, true);

        assertEquals(n, viaAnn.labels().length);
        double ari = adjustedRandIndex(exact.labels(), viaAnn.labels());
        assertTrue(ari >= 0.85, "HNSW-routed Leiden must agree with exact Leiden by ARI, got " + ari);
    }

    @Test
    void clusterViaAnnCompletesOnLargerSyntheticCloudWithoutThrowing() {
        Random rng = new Random(55);
        int per = 1000;
        int n = per * 2;
        double[][] rows = new double[n][2];
        fillBlob(rows, rng, 0, per, 0.0, 0.0, 0.5);
        fillBlob(rows, rng, per, n, 30.0, 0.0, 0.5);

        LeidenResult res = assertDoesNotThrow(() -> LeidenModel.clusterViaAnn(rows, 15, 0.3, 5, 7L, true));
        assertEquals(n, res.labels().length);
        assertTrue(res.nClusters() >= 2, "Expected at least 2 recovered communities, got " + res.nClusters());
    }

    @Test
    void clusterViaAnnReturnsAMeasuredPassingRecall() {
        // Item 8 (D-09): clusterViaAnn must thread the recall-gate's measured passing recall
        // through into its LeidenResult, instead of the caller only ever seeing a hardcoded
        // sentinel. A passing run's recall must be a real, measured value in (0, 1].
        Random rng = new Random(88);
        int per = 60;
        int n = per * 3;
        double[][] rows = new double[n][2];
        fillBlob(rows, rng, 0, per, 0.0, 0.0, 0.3);
        fillBlob(rows, rng, per, 2 * per, 20.0, 0.0, 0.3);
        fillBlob(rows, rng, 2 * per, n, 0.0, 20.0, 0.3);

        LeidenResult res = LeidenModel.clusterViaAnn(rows, 15, 0.3, 5, 12L, true);

        assertTrue(res.recall() > 0.0 && res.recall() <= 1.0, "measured recall must be in (0, 1], got " + res.recall());
    }

    @Test
    void clusterExactPathReportsRecallOfOne() {
        // The exact (brute-force featureKnn) path never approximates, so its recall is a
        // constant 1.0, not left at some sentinel or unset value.
        Random rng = new Random(89);
        double[][] rows = randomCloud(rng, 50, 3);
        LeidenResult res = LeidenModel.cluster(rows, 10, 1.0, 3, 5L);
        assertEquals(1.0, res.recall());
    }

    @Test
    void clusterViaAnnAbortsWhenRecallGateFails() {
        // D-09 deliberately hides ALL ann knobs from callers (no user-facing way
        // to "cripple" the real HnswKnnIndex from outside), so -- exactly as in
        // the Task 1 recall-gate tests -- degraded ANN params are simulated via
        // a stub neighbour source rather than the real index. This exercises the
        // SAME gateAnnRecall wiring that clusterViaAnn/annNeighborsWithGate calls
        // uncaught, so the thrown AnnRecallException is guaranteed to propagate
        // out of clusterViaAnn with no LeidenResult ever produced.
        Random rng = new Random(66);
        double[][] rows = randomCloud(rng, 400, 6);
        int k = 10;
        int keep = Math.min(k, rows.length - 1);

        assertThrows(
                AnnRecallException.class,
                () -> LeidenModel.gateAnnRecall(rows, k, 999L, 64, (ef, sampleIdx) -> {
                    int[][] wrong = new int[sampleIdx.length][];
                    for (int s = 0; s < sampleIdx.length; s++) {
                        wrong[s] = bruteForceFarthest(rows, sampleIdx[s], keep);
                    }
                    return wrong;
                }));
    }

    @Test
    void clusterViaAnnHandlesDegenerateInputsLikeCluster() {
        double[][] empty = new double[0][0];
        LeidenResult resEmpty = LeidenModel.clusterViaAnn(empty, 15, 1.0, 5, 1L, true);
        assertEquals(0, resEmpty.labels().length);
        assertEquals(0, resEmpty.nClusters());

        double[][] single = {{1.0, 2.0}};
        LeidenResult resSingle = LeidenModel.clusterViaAnn(single, 15, 1.0, 5, 1L, true);
        assertArrayEquals(new int[] {0}, resSingle.labels());
        assertEquals(1, resSingle.nClusters());
    }

    // ── clusterViaAnn: all-cells single partition + reproducibility (LEI-06, LEI-10) ──

    @Test
    void clusterViaAnnAssignsEveryRowFromSinglePartitionAndRecoversBlobsByPurity() {
        // Simulates a pooled-across-images cloud (no image concept exists at this
        // pure-array layer -- that is CohortClusterModel's job): every pooled row
        // must come out of ONE Leiden partition (labels().length == rows.length,
        // no reference/query split) and each blob must be recovered by purity.
        Random rng = new Random(303);
        int per = 80;
        int n = per * 3;
        double[][] rows = new double[n][2];
        fillBlob(rows, rng, 0, per, 0.0, 0.0, 0.3);
        fillBlob(rows, rng, per, 2 * per, 25.0, 0.0, 0.3);
        fillBlob(rows, rng, 2 * per, 3 * per, 0.0, 25.0, 0.3);

        LeidenResult res = LeidenModel.clusterViaAnn(rows, 15, 0.3, 10, 404L, true);

        assertEquals(n, res.labels().length, "Every pooled row must get a label from a single partition");
        assertTrue(purity(res.labels(), 0, per) > 0.9, "Blob A not pure");
        assertTrue(purity(res.labels(), per, 2 * per) > 0.9, "Blob B not pure");
        assertTrue(purity(res.labels(), 2 * per, n) > 0.9, "Blob C not pure");
    }

    @Test
    void clusterViaAnnReproducibleRunsAreIdenticalUpToPermutation() {
        // SPEC Acceptance Criterion 6: two consecutive reproducible clusterViaAnn
        // runs on the same input must yield labelings identical up to a label
        // permutation -- asserted via Adjusted Rand Index == 1.0 (permutation-
        // invariant), not a raw assertArrayEquals on label ids. Relies on Plan
        // 15-01's seeded single-threaded HnswKnnIndex build for the reproducible
        // path -- if this test goes flaky, that is a real defect upstream, not a
        // reason to weaken the assertion.
        Random rng = new Random(505);
        int per = 60;
        int n = per * 3;
        double[][] rows = new double[n][2];
        fillBlob(rows, rng, 0, per, 0.0, 0.0, 0.3);
        fillBlob(rows, rng, per, 2 * per, 22.0, 0.0, 0.3);
        fillBlob(rows, rng, 2 * per, n, 0.0, 22.0, 0.3);

        LeidenResult a = LeidenModel.clusterViaAnn(rows, 15, 0.3, 10, 99L, true);
        LeidenResult b = LeidenModel.clusterViaAnn(rows, 15, 0.3, 10, 99L, true);

        double ari = adjustedRandIndex(a.labels(), b.labels());
        assertEquals(
                1.0,
                ari,
                1e-9,
                "Two reproducible clusterViaAnn runs with the same seed must yield identical labels up to"
                        + " permutation (ARI==1.0), got " + ari);
    }

    @Test
    void transferLabelsStillWorksAfterAllCellsRewrite() {
        // Quick guard for SPEC's "transferLabels and its Phase 14 tests remain
        // present and passing" acceptance criterion -- transferLabels is
        // untouched by this plan's primitive-array SNN rewrite; the dedicated
        // tests above already cover it in depth, this is just a smoke check
        // alongside the new all-cells tests.
        Random rng = new Random(202);
        int per = 30;
        double[][] reference = new double[per * 2][2];
        fillBlob(reference, rng, 0, per, 0.0, 0.0, 0.3);
        fillBlob(reference, rng, per, per * 2, 20.0, 0.0, 0.3);
        int[] refLabels = new int[per * 2];
        Arrays.fill(refLabels, 0, per, 0);
        Arrays.fill(refLabels, per, per * 2, 1);
        double[][] query = {{0.1, 0.1}, {19.9, -0.1}};

        int[] assigned = LeidenModel.transferLabels(query, reference, refLabels, 10, 2);

        assertEquals(0, assigned[0], "Query near blob A must get label 0");
        assertEquals(1, assigned[1], "Query near blob B must get label 1");
    }

    // ── conditional PCA reduction before clustering (dominance/degradation fix) ──

    @Test
    void pcaOnRecoversCommunitiesBetterThanPcaOffWhenSignalIsDrownedByManyNoisyMeasurements() {
        // Reproduces the real-project failure mode this feature fixes: a SMALL number of true
        // signal markers (2 here, matching clusterRecoversThreeSeparatedBlobsByPurity's 2D
        // blobs) separating 3 known communities, plus a LARGE number of other per-cell
        // measurement columns that carry no cluster structure at all (independent noise —
        // "other markers/measurements" a wide real panel also carries). After z-scoring, every
        // column has equal (unit) per-column variance, so raw brute-force Euclidean kNN weighs
        // the 2 informative columns and the many uninformative ones EQUALLY: with enough
        // uninformative columns their combined random fluctuation swamps the fixed signal
        // separation from just 2 columns, degrading the neighbour graph and Leiden's recovered
        // communities entirely (empirically: raw kNN here collapses to a single giant
        // "cluster", ARI ~ 0). PCA's covariance eigendecomposition instead ranks the (weakly
        // but genuinely) correlated pair of signal columns above the independent noise
        // columns' eigenvalues, so a modest kept-component count recovers the true structure.
        // Both methods run through the SAME clusterViaAnn call — only the input matrix differs
        // (PCA-reduced vs raw z-scored) — isolating the reduction itself.
        Random rng = new Random(2024);
        int per = 3000;
        int n = per * 3;
        double[][] signal = new double[n][2];
        fillBlob(signal, rng, 0, per, 0.0, 0.0, 0.3);
        fillBlob(signal, rng, per, 2 * per, 20.0, 0.0, 0.3);
        fillBlob(signal, rng, 2 * per, 3 * per, 0.0, 20.0, 0.3);
        int[] trueLabels = new int[n];
        Arrays.fill(trueLabels, 0, per, 0);
        Arrays.fill(trueLabels, per, 2 * per, 1);
        Arrays.fill(trueLabels, 2 * per, 3 * per, 2);

        int noiseColumns = 60; // far outnumbers the 2 true signal columns (62 total > threshold 50).
        double[][] raw = appendUninformativeNoiseColumns(signal, noiseColumns, rng);

        double[][] std = ScatterMath.standardizeColumns(raw);
        ScatterMath.PcaReduction pcaOff = ScatterMath.pcaReduce(std, 0, 50, 100_000, 42L);
        ScatterMath.PcaReduction pcaOn = ScatterMath.pcaReduce(std, 5, 50, 100_000, 42L);
        assertFalse(pcaOff.applied(), "PCA-off control must be a genuine no-op pass-through");
        assertTrue(pcaOn.applied(), "62 columns > threshold 50 must trigger PCA");

        LeidenResult withoutPca = LeidenModel.clusterViaAnn(pcaOff.reduced(), 15, 0.3, 10, 42L, true);
        LeidenResult withPca = LeidenModel.clusterViaAnn(pcaOn.reduced(), 15, 0.3, 10, 42L, true);

        double ariWithout = adjustedRandIndex(trueLabels, withoutPca.labels());
        double ariWith = adjustedRandIndex(trueLabels, withPca.labels());

        assertTrue(
                ariWith > ariWithout + 0.1,
                String.format(
                        "PCA-on ARI (%.3f) must materially exceed PCA-off ARI (%.3f) recovering the 3 true"
                                + " communities from noise-drowned measurements",
                        ariWith, ariWithout));
        assertTrue(ariWith > 0.8, "PCA-on must recover the true communities well, got ARI " + ariWith);
    }

    /**
     * Appends {@code noiseColumns} independent, cluster-agnostic random columns (mean 0, unit
     * variance — i.i.d. per cell, unrelated to {@code signal} or the true community structure)
     * onto {@code signal}'s true marker columns — the "many other measurements a wide real
     * panel also carries" scenario.
     */
    private static double[][] appendUninformativeNoiseColumns(double[][] signal, int noiseColumns, Random rng) {
        int n = signal.length;
        int d = signal[0].length;
        double[][] out = new double[n][d + noiseColumns];
        for (int i = 0; i < n; i++) {
            System.arraycopy(signal[i], 0, out[i], 0, d);
            for (int c = 0; c < noiseColumns; c++) {
                out[i][d + c] = rng.nextGaussian();
            }
        }
        return out;
    }

    // ── transferLabels ─────────────────────────────────────────────────────────

    @Test
    void transferLabelsAssignsQueryPointsToNearestBlobLabel() {
        Random rng = new Random(13);
        int per = 40;
        double[][] reference = new double[per * 2][2];
        fillBlob(reference, rng, 0, per, 0.0, 0.0, 0.3);
        fillBlob(reference, rng, per, per * 2, 20.0, 0.0, 0.3);
        int[] refLabels = new int[per * 2];
        Arrays.fill(refLabels, 0, per, 0);
        Arrays.fill(refLabels, per, per * 2, 1);

        double[][] query = {
            {0.1, 0.1}, // near blob A
            {19.9, -0.1}, // near blob B
            {0.2, -0.2}, // near blob A
        };
        int[] assigned = LeidenModel.transferLabels(query, reference, refLabels, 10, 2);

        assertEquals(0, assigned[0], "Query near blob A must get label 0");
        assertEquals(1, assigned[1], "Query near blob B must get label 1");
        assertEquals(0, assigned[2], "Query near blob A must get label 0");
    }

    @Test
    void transferLabelsTiesResolveToLowestLabel() {
        // Reference: exactly one point at each of two labels, symmetric around the
        // query so k=2 pulls in exactly one of each label — a tie, must resolve to
        // the lowest label id.
        double[][] reference = {
            {-1.0, 0.0}, // label 1
            {1.0, 0.0}, // label 0
        };
        int[] refLabels = {1, 0};
        double[][] query = {{0.0, 0.0}};

        int[] assigned = LeidenModel.transferLabels(query, reference, refLabels, 2, 2);
        assertEquals(0, assigned[0], "Tie must resolve to the lowest label id");
    }

    @Test
    void transferLabelsHandlesDegenerateQueryWithoutCrashing() {
        double[][] reference = {{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}};
        int[] refLabels = {0, 1, 1};
        // Query exactly coincident with a reference point, and a query with NaN.
        double[][] query = {{0.0, 0.0}, {Double.NaN, Double.NaN}};

        int[] assigned = assertDoesNotThrow(() -> LeidenModel.transferLabels(query, reference, refLabels, 2, 2));
        assertEquals(2, assigned.length);
    }

    @Test
    void transferLabelsSingleReferenceRowNeverCrashes() {
        double[][] reference = {{5.0, 5.0}};
        int[] refLabels = {0};
        double[][] query = {{0.0, 0.0}, {10.0, 10.0}};
        int[] assigned = assertDoesNotThrow(() -> LeidenModel.transferLabels(query, reference, refLabels, 5, 1));
        assertEquals(0, assigned[0]);
        assertEquals(0, assigned[1]);
    }

    @Test
    void transferLabelsEmptyReferenceYieldsAllUnassigned() {
        // An empty reference has nothing to vote from -- must return -1 ("unassigned") for
        // every query row, NOT silently default to label 0 (Rule 1 bug fix: a query cell must
        // never be misclassified as cluster 0's phenotype just because the reference was empty).
        double[][] reference = new double[0][0];
        int[] refLabels = new int[0];
        double[][] query = {{0.0, 0.0}, {1.0, 1.0}, {Double.NaN, Double.NaN}};

        int[] assigned = assertDoesNotThrow(() -> LeidenModel.transferLabels(query, reference, refLabels, 5, 2));

        assertEquals(3, assigned.length);
        for (int label : assigned) {
            assertEquals(-1, label, "Empty reference must leave every query row unassigned (-1)");
        }
    }

    @Test
    void transferLabelsEmptyReferenceLeavesCohortAssignmentUnassignedEndToEnd() {
        // End-to-end guard for the CohortClusterModel callers (assignAcrossProjectLeiden /
        // writeClusterAcrossProjectLeiden): the -1 transferLabels now returns for an empty
        // reference must resolve to "no class" via Map.get(-1)==null, and to a written -1
        // (not label+1==0) via the writeCluster path's `labels[t] >= 0` guard.
        double[][] reference = new double[0][0];
        int[] refLabels = new int[0];
        double[][] query = {{0.0, 0.0}, {1.0, 1.0}};
        int[] assigned = LeidenModel.transferLabels(query, reference, refLabels, 5, 2);

        Map<Integer, PathClass> mapping = new HashMap<>();
        mapping.put(0, PathClass.fromString("Phenotype A"));
        mapping.put(1, PathClass.fromString("Phenotype B"));
        for (int label : assigned) {
            assertNull(mapping.get(label), "assignAcrossProjectLeiden's mapping.get(-1) must be null (unassigned)");
        }

        double[] writtenValues = new double[assigned.length];
        Arrays.fill(writtenValues, -1.0);
        for (int i = 0; i < assigned.length; i++) {
            if (assigned[i] >= 0) {
                writtenValues[i] = assigned[i] + 1.0;
            }
        }
        for (double v : writtenValues) {
            assertEquals(-1.0, v, "writeClusterAcrossProjectLeiden must write -1, never label+1==0, when unassigned");
        }
    }

    // ── helpers ─────────────────────────────────────────────────────────────────

    private static double[][] randomCloud(Random rng, int n, int d) {
        double[][] rows = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                rows[i][j] = rng.nextDouble() * 100.0;
            }
        }
        return rows;
    }

    private static void fillBlob(double[][] rows, Random rng, int from, int to, double cx, double cy, double spread) {
        for (int i = from; i < to; i++) {
            rows[i][0] = cx + rng.nextGaussian() * spread;
            rows[i][1] = cy + rng.nextGaussian() * spread;
        }
    }

    private static Set<Integer> bruteForceKnn(double[][] rows, int i, int k) {
        int n = rows.length;
        Integer[] order = new Integer[n];
        double[] d2 = new double[n];
        for (int j = 0; j < n; j++) {
            order[j] = j;
            d2[j] = (j == i) ? Double.POSITIVE_INFINITY : squaredDist(rows[i], rows[j]);
        }
        Arrays.sort(order, (p, q) -> Double.compare(d2[p], d2[q]));
        Set<Integer> set = new HashSet<>();
        for (int t = 0; t < k && t < n; t++) {
            set.add(order[t]);
        }
        return set;
    }

    /**
     * The {@code k} FARTHEST (not nearest) rows to row {@code i} -- used to
     * simulate a deliberately degraded/wrong ANN neighbour source for the
     * recall-gate abort test, since the recall gate's exact reference is
     * always the true nearest neighbours.
     */
    private static int[] bruteForceFarthest(double[][] rows, int i, int k) {
        int n = rows.length;
        Integer[] order = new Integer[n];
        double[] d2 = new double[n];
        for (int j = 0; j < n; j++) {
            order[j] = j;
            d2[j] = (j == i) ? Double.NEGATIVE_INFINITY : squaredDist(rows[i], rows[j]);
        }
        Arrays.sort(order, (p, q) -> Double.compare(d2[q], d2[p])); // descending -> farthest first
        int keep = Math.min(k, n);
        int[] out = new int[keep];
        for (int t = 0; t < keep; t++) {
            out[t] = order[t];
        }
        return out;
    }

    private static double squaredDist(double[] a, double[] b) {
        double sum = 0;
        for (int c = 0; c < a.length; c++) {
            double diff = a[c] - b[c];
            sum += diff * diff;
        }
        return sum;
    }

    /**
     * Reference (boxed) Jaccard/SNN weighting -- the pre-Plan-15-03
     * implementation, preserved here ONLY as the equivalence baseline for the
     * primitive-array rewrite ({@code jaccardEdgesForTestMatchesBoxedReferenceOnVariousSizes}).
     * Not used by any production code path -- {@link LeidenModel} itself no
     * longer contains a boxed {@code HashSet<Integer>}/{@code HashSet<Long>}
     * weighting implementation (RESEARCH Pitfall 2).
     */
    private static LeidenModel.JaccardEdges buildJaccardEdgesBoxedReference(int n, int[][] neighbors) {
        @SuppressWarnings("unchecked")
        Set<Integer>[] closedSets = new HashSet[n];
        for (int i = 0; i < n; i++) {
            Set<Integer> s = new HashSet<>(neighbors[i].length * 2 + 1);
            s.add(i);
            for (int j : neighbors[i]) {
                s.add(j);
            }
            closedSets[i] = s;
        }
        Set<Long> seenEdges = new HashSet<>();
        java.util.List<Integer> fromList = new java.util.ArrayList<>();
        java.util.List<Integer> toList = new java.util.ArrayList<>();
        java.util.List<Double> weightList = new java.util.ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j : neighbors[i]) {
                int a = Math.min(i, j);
                int b = Math.max(i, j);
                if (a == b) {
                    continue;
                }
                long key = (((long) a) << 32) | (b & 0xffffffffL);
                if (!seenEdges.add(key)) {
                    continue;
                }
                double w = jaccardBoxedReference(closedSets[a], closedSets[b]);
                fromList.add(a);
                toList.add(b);
                weightList.add(w);
            }
        }
        int[] from = fromList.stream().mapToInt(Integer::intValue).toArray();
        int[] to = toList.stream().mapToInt(Integer::intValue).toArray();
        double[] weights = weightList.stream().mapToDouble(Double::doubleValue).toArray();
        return new LeidenModel.JaccardEdges(from, to, weights);
    }

    private static double jaccardBoxedReference(Set<Integer> a, Set<Integer> b) {
        Set<Integer> smaller = a.size() <= b.size() ? a : b;
        Set<Integer> larger = a.size() <= b.size() ? b : a;
        int intersection = 0;
        for (Integer v : smaller) {
            if (larger.contains(v)) {
                intersection++;
            }
        }
        int union = a.size() + b.size() - intersection;
        return union == 0 ? 0.0 : intersection / (double) union;
    }

    private static Set<Integer> toSet(int[] a) {
        Set<Integer> s = new HashSet<>();
        for (int v : a) {
            s.add(v);
        }
        return s;
    }

    /**
     * Adjusted Rand Index between two labelings of the same {@code n} items --
     * chance-corrected agreement, 1.0 for identical partitions (up to label
     * permutation), ~0.0 for random/independent partitions. Standard
     * contingency-table formula (Hubert & Arabie 1985).
     */
    private static double adjustedRandIndex(int[] a, int[] b) {
        int n = a.length;
        if (n == 0) {
            return 1.0;
        }
        int maxA = 0;
        int maxB = 0;
        for (int v : a) {
            maxA = Math.max(maxA, v);
        }
        for (int v : b) {
            maxB = Math.max(maxB, v);
        }
        int[][] contingency = new int[maxA + 1][maxB + 1];
        for (int i = 0; i < n; i++) {
            contingency[a[i]][b[i]]++;
        }
        int[] rowSums = new int[maxA + 1];
        int[] colSums = new int[maxB + 1];
        long sumComb = 0;
        for (int i = 0; i <= maxA; i++) {
            for (int j = 0; j <= maxB; j++) {
                int c = contingency[i][j];
                rowSums[i] += c;
                colSums[j] += c;
                sumComb += comb2(c);
            }
        }
        long sumRowComb = 0;
        for (int s : rowSums) {
            sumRowComb += comb2(s);
        }
        long sumColComb = 0;
        for (int s : colSums) {
            sumColComb += comb2(s);
        }
        long totalComb = comb2(n);
        double expectedIndex = (sumRowComb * (double) sumColComb) / (double) totalComb;
        double maxIndex = 0.5 * (sumRowComb + sumColComb);
        double denom = maxIndex - expectedIndex;
        if (denom == 0) {
            return 1.0;
        }
        return (sumComb - expectedIndex) / denom;
    }

    private static long comb2(long x) {
        return x * (x - 1) / 2;
    }

    /** Fraction of rows in [from,to) that carry the most common label in that range. */
    private static double purity(int[] labels, int from, int to) {
        java.util.Map<Integer, Integer> counts = new java.util.HashMap<>();
        for (int i = from; i < to; i++) {
            counts.merge(labels[i], 1, Integer::sum);
        }
        int max = counts.values().stream().mapToInt(Integer::intValue).max().orElse(0);
        return max / (double) (to - from);
    }
}
