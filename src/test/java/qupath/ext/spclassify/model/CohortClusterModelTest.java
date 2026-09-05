package qupath.ext.spclassify.model;

import static org.junit.jupiter.api.Assertions.*;

import java.awt.image.BufferedImage;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;
import qupath.ext.spclassify.model.CohortClusterModel.CancellationToken;
import qupath.ext.spclassify.model.CohortClusterModel.CentroidsAndCounts;
import qupath.ext.spclassify.model.CohortClusterModel.ClusterOutcome;
import qupath.ext.spclassify.model.CohortClusterModel.MeasurementAssignment;
import qupath.ext.spclassify.model.CohortClusterModel.Pass2Outcome;
import qupath.ext.spclassify.model.CohortClusterModel.UuidKey;
import qupath.ext.spclassify.model.LeidenModel.LeidenResult;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;
import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.interfaces.ROI;

/**
 * Tests the deterministic core of the cohort clustering backend: distinct-index
 * sampling and nearest-centroid assignment (the rule that gives every image's
 * cells a cohort-consistent cluster label), plus the all-cells (LEI-06/LEI-08)
 * two-pass driver's pure seams: packed-UUID reorder-safe write-back via a single
 * global O(n) map ({@link CohortClusterModel#buildGlobalLabelMap}), pass-2
 * cancellation bookkeeping ({@link CohortClusterModel#runPass2Loop}), and
 * recall-gate abort handling
 * ({@link CohortClusterModel#clusterOrAbort}). These pure helpers are exercised
 * directly against synthetic/real {@link PathObject}s (no live {@code Project}/
 * {@code ImageData} required — see AnnotationLabelCollectorTest's real-hierarchy
 * construction pattern), mirroring what {@link CohortClusterModel#writeClusterAllCells}
 * itself calls.
 */
class CohortClusterModelTest {

    private static final ImagePlane PLANE = ImagePlane.getDefaultPlane();

    /** A small real detection with a real, stable {@code getID()} (like AnnotationLabelCollectorTest's pattern). */
    private static PathObject detectionAt(double x, double y) {
        ROI roi = ROIs.createRectangleROI(x, y, 5, 5, PLANE);
        return PathObjects.createDetectionObject(roi);
    }

    private static List<PathObject> detections(int n, double xOffset) {
        List<PathObject> cells = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            cells.add(detectionAt(xOffset + i * 10, 0));
        }
        return cells;
    }

    private static PathObjectHierarchy hierarchyWith(List<PathObject> objects) {
        PathObjectHierarchy h = new PathObjectHierarchy();
        for (PathObject o : objects) {
            h.addObject(o);
        }
        return h;
    }

    /**
     * A minimal, real {@link ImageData} wrapping {@code hierarchy}, backed by a fake
     * {@link ImageServer} (built via {@link Proxy} — {@code ImageServer} is an interface
     * and none of its methods are touched by anything {@code CohortClusterModel} calls on
     * an {@code ImageData}, which only ever reads {@code getHierarchy()}). Lets tests
     * construct genuine, reference-distinguishable {@code ImageData<BufferedImage>}
     * instances without a live QuPath project/server.
     */
    @SuppressWarnings("unchecked")
    private static ImageData<BufferedImage> fakeImageData(PathObjectHierarchy hierarchy) {
        ImageServer<BufferedImage> server = (ImageServer<BufferedImage>) Proxy.newProxyInstance(
                ImageServer.class.getClassLoader(), new Class[] {ImageServer.class}, (proxy, method, args) -> {
                    String name = method.getName();
                    return switch (name) {
                        case "toString" -> "FakeImageServer";
                        case "hashCode" -> System.identityHashCode(proxy);
                        case "equals" -> proxy == args[0];
                        case "close" -> null;
                        default -> throw new UnsupportedOperationException("Not implemented in fake server: " + name);
                    };
                });
        return new ImageData<>(server, hierarchy, ImageData.ImageType.UNSET);
    }

    // ── LEI-08: packed-UUID reorder-safe write-back ─────────────────────────────

    @Test
    void uuidWriteBackSurvivesReorder() {
        // Two synthetic "images": image A (4 cells) and image B (5 cells), each with
        // real PathObjects (real getID()s).
        List<PathObject> imageACells = detections(4, 0);
        List<PathObject> imageBCells = detections(5, 1000);
        hierarchyWith(imageACells); // constructed to mirror a real per-image hierarchy; not otherwise used
        hierarchyWith(imageBCells);

        // Simulate pass 1's pooling: capture every cell's packed UUID, in pooled
        // order (image A's cells first, then image B's).
        List<PathObject> pooledOrder = new ArrayList<>();
        pooledOrder.addAll(imageACells);
        pooledOrder.addAll(imageBCells);

        int m = pooledOrder.size();
        long[] msb = new long[m];
        long[] lsb = new long[m];
        int[] labels = new int[m];
        Map<PathObject, Integer> expectedLabelByCell = new HashMap<>();
        for (int i = 0; i < m; i++) {
            PathObject cell = pooledOrder.get(i);
            msb[i] = cell.getID().getMostSignificantBits();
            lsb[i] = cell.getID().getLeastSignificantBits();
            labels[i] = i % 3; // an arbitrary synthetic Leiden community label per pooled row
            expectedLabelByCell.put(cell, labels[i]);
        }

        // Build the SINGLE global reorder-independent lookup via the EXACT production
        // helper — no per-image ordinal filtering, built once for the whole cohort.
        Map<UuidKey, Integer> globalLabelMap = CohortClusterModel.buildGlobalLabelMap(msb, lsb, labels);

        // Read image B's cells back in a SHUFFLED order (simulating a second
        // readImageData() call returning cells in a different order) and confirm
        // every cell's looked-up label matches its pass-1 pooled-row label BY
        // UUID, not by position.
        List<PathObject> shuffledB = new ArrayList<>(imageBCells);
        Collections.shuffle(shuffledB, new Random(7));
        assertNotEquals(
                imageBCells, shuffledB, "shuffle must actually reorder the list for this test to be meaningful");

        for (PathObject cell : shuffledB) {
            UuidKey key = new UuidKey(
                    cell.getID().getMostSignificantBits(), cell.getID().getLeastSignificantBits());
            Integer looked = globalLabelMap.get(key);
            assertNotNull(looked, "every image-B cell must resolve a label via its UUID");
            assertEquals(
                    expectedLabelByCell.get(cell),
                    looked,
                    "label must match the pass-1 pooled-row label, not position");
        }

        // Also exercise a REVERSED read order for good measure.
        List<PathObject> reversedB = new ArrayList<>(imageBCells);
        Collections.reverse(reversedB);
        for (PathObject cell : reversedB) {
            UuidKey key = new UuidKey(
                    cell.getID().getMostSignificantBits(), cell.getID().getLeastSignificantBits());
            assertEquals(expectedLabelByCell.get(cell), globalLabelMap.get(key));
        }

        // image A's cells must ALSO resolve correctly through the SAME global map
        // (no ordinal filtering — one map serves every image).
        for (PathObject cell : imageACells) {
            UuidKey key = new UuidKey(
                    cell.getID().getMostSignificantBits(), cell.getID().getLeastSignificantBits());
            assertEquals(
                    expectedLabelByCell.get(cell),
                    globalLabelMap.get(key),
                    "image A's cells must resolve via the same global map used for image B");
        }
    }

    // ── LEI-08/LEI-10: pooling identity ──────────────────────────────────────────

    @Test
    void poolingIdentityEveryUuidResolvesToExactlyOneLabelPerImage() {
        List<PathObject> imageACells = detections(6, 0);
        List<PathObject> imageBCells = detections(3, 1000);

        List<PathObject> pooledOrder = new ArrayList<>();
        pooledOrder.addAll(imageACells);
        pooledOrder.addAll(imageBCells);
        int totalCells = imageACells.size() + imageBCells.size();

        int m = pooledOrder.size();
        long[] msb = new long[m];
        long[] lsb = new long[m];
        int[] labels = new int[m];
        for (int i = 0; i < m; i++) {
            PathObject cell = pooledOrder.get(i);
            msb[i] = cell.getID().getMostSignificantBits();
            lsb[i] = cell.getID().getLeastSignificantBits();
            labels[i] = i;
        }

        // Pooled row count == total detections across both images.
        assertEquals(totalCells, m, "pooled row count must equal total detections across all images");

        // A SINGLE global map (built once, O(n)) must resolve every pooled cell to its own
        // label, regardless of which image (A or B) it came from -- no per-image rescans.
        Map<UuidKey, Integer> globalMap = CohortClusterModel.buildGlobalLabelMap(msb, lsb, labels);

        assertEquals(totalCells, globalMap.size(), "global map must have exactly one entry per pooled cell");
        for (PathObject cell : imageACells) {
            UuidKey key = new UuidKey(
                    cell.getID().getMostSignificantBits(), cell.getID().getLeastSignificantBits());
            assertTrue(globalMap.containsKey(key), "image A cell must resolve in the global map");
        }
        for (PathObject cell : imageBCells) {
            UuidKey key = new UuidKey(
                    cell.getID().getMostSignificantBits(), cell.getID().getLeastSignificantBits());
            assertTrue(globalMap.containsKey(key), "image B cell must resolve in the global map");
        }

        // Lookup is independent of image order: building the map from a SHUFFLED pooled
        // order still resolves every cell to its original per-row label.
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < m; i++) {
            order.add(i);
        }
        Collections.shuffle(order, new Random(3));
        long[] shuffledMsb = new long[m];
        long[] shuffledLsb = new long[m];
        int[] shuffledLabels = new int[m];
        for (int i = 0; i < m; i++) {
            int src = order.get(i);
            shuffledMsb[i] = msb[src];
            shuffledLsb[i] = lsb[src];
            shuffledLabels[i] = labels[src];
        }
        Map<UuidKey, Integer> shuffledMap =
                CohortClusterModel.buildGlobalLabelMap(shuffledMsb, shuffledLsb, shuffledLabels);
        for (int i = 0; i < m; i++) {
            PathObject cell = pooledOrder.get(i);
            UuidKey key = new UuidKey(
                    cell.getID().getMostSignificantBits(), cell.getID().getLeastSignificantBits());
            assertEquals(globalMap.get(key), shuffledMap.get(key), "lookup must be independent of pooled row order");
        }
    }

    // ── LEI-10: cancel leaves already-written images intact ─────────────────────

    @Test
    void cancelLeavesWrittenIntactAndReportsUnwrittenImages() {
        List<String> images = List.of("img1", "img2", "img3", "img4");
        CancellationToken token = new CancellationToken();

        // "img2" is written successfully; cancel() fires WHILE processing img2 (as
        // if the user clicked Cancel mid-write), which runPass2Loop must only
        // observe on the NEXT iteration (it checks isCancelled() at the top of
        // each iteration, matching writeClusterAllCells's real per-image loop).
        List<String> processed = new ArrayList<>();
        Pass2Outcome outcome = CohortClusterModel.runPass2Loop(images, token, (ordinal, name) -> {
            processed.add(name);
            if (name.equals("img2")) {
                token.cancel();
                return true; // img2 itself still gets written before cancellation is observed
            }
            return true;
        });

        assertEquals(
                List.of("img1", "img2"), outcome.written(), "img1/img2 processed before cancellation was observed");
        assertEquals(List.of("img3", "img4"), outcome.notWritten(), "img3/img4 must be reported unwritten");
        assertTrue(outcome.cancelled(), "outcome must report the run was cancelled");
        assertEquals(
                List.of("img1", "img2"),
                processed,
                "processImage must never be invoked for img3/img4 after cancellation");
    }

    @Test
    void noCancellationProcessesEveryImage() {
        List<String> images = List.of("a", "b", "c");
        Pass2Outcome outcome =
                CohortClusterModel.runPass2Loop(images, new CancellationToken(), (ordinal, name) -> true);
        assertEquals(images, outcome.written());
        assertTrue(outcome.notWritten().isEmpty());
        assertFalse(outcome.cancelled());
    }

    @Test
    void failedImagesAreReportedNotWrittenWithoutStoppingTheLoop() {
        List<String> images = List.of("ok1", "fails", "ok2");
        Pass2Outcome outcome = CohortClusterModel.runPass2Loop(
                images, new CancellationToken(), (ordinal, name) -> !name.equals("fails"));
        assertEquals(List.of("ok1", "ok2"), outcome.written());
        assertEquals(List.of("fails"), outcome.notWritten());
        assertFalse(outcome.cancelled());
    }

    @Test
    void cancellationTokenReflectsCancelCall() {
        CancellationToken token = new CancellationToken();
        assertFalse(token.isCancelled());
        token.cancel();
        assertTrue(token.isCancelled());
    }

    // ── LEI-10: recall-gate abort writes nothing ─────────────────────────────────

    @Test
    void abortWritesNothingWhenRecallGateFails() {
        ClusterOutcome outcome = CohortClusterModel.clusterOrAbort(() -> {
            throw new AnnRecallException("ANN recall 0.42 remained below the required 0.95 threshold");
        });

        assertTrue(outcome.aborted(), "a thrown AnnRecallException must be reported as an abort");
        assertNull(outcome.result(), "no LeidenResult (and therefore no labels) on abort");
        assertNotNull(outcome.abortMessage());
        assertTrue(outcome.abortMessage().contains("0.42"));
    }

    @Test
    void successfulClusterCallIsNotAborted() {
        LeidenResult fakeResult = new LeidenResult(new int[] {0, 1, 0}, 2, 0.99);
        ClusterOutcome outcome = CohortClusterModel.clusterOrAbort(() -> fakeResult);

        assertFalse(outcome.aborted());
        assertSame(fakeResult, outcome.result());
        assertNull(outcome.abortMessage());
    }

    // ── centroidsAndCounts (per-cluster centroid + pooled count over labels) ────

    @Test
    void centroidsAndCountsComputesMeanZScoredRowPerCluster() {
        // 3 markers, 2 clusters: cluster 0 = rows 0,1; cluster 1 = rows 2,3,4.
        double[][] rows = {
            {1.0, 2.0, 3.0},
            {3.0, 4.0, 5.0},
            {10.0, 0.0, -10.0},
            {20.0, 0.0, -20.0},
            {30.0, 0.0, -30.0},
        };
        int[] labels = {0, 0, 1, 1, 1};

        CentroidsAndCounts result = CohortClusterModel.centroidsAndCounts(rows, labels, 2, 3);

        assertArrayEquals(new int[] {2, 3}, result.counts());
        assertArrayEquals(new double[] {2.0, 3.0, 4.0}, result.centroids()[0], 1e-9);
        assertArrayEquals(new double[] {20.0, 0.0, -20.0}, result.centroids()[1], 1e-9);
    }

    @Test
    void centroidsAndCountsLeavesEmptyClustersAtZero() {
        double[][] rows = {{5.0, 5.0}};
        int[] labels = {0};

        // nClusters=3 but only cluster 0 has any pooled row -- clusters 1/2 stay all-zero.
        CentroidsAndCounts result = CohortClusterModel.centroidsAndCounts(rows, labels, 3, 2);

        assertArrayEquals(new int[] {1, 0, 0}, result.counts());
        assertArrayEquals(new double[] {5.0, 5.0}, result.centroids()[0], 1e-9);
        assertArrayEquals(new double[] {0.0, 0.0}, result.centroids()[1], 1e-9);
        assertArrayEquals(new double[] {0.0, 0.0}, result.centroids()[2], 1e-9);
    }

    @Test
    void centroidsAndCountsSkipsOutOfRangeLabelsDefensively() {
        double[][] rows = {{1.0}, {2.0}, {3.0}};
        // Label -1 (unclustered sentinel) and an out-of-range label must never crash or
        // pollute a real cluster's centroid.
        int[] labels = {-1, 0, 99};

        CentroidsAndCounts result = CohortClusterModel.centroidsAndCounts(rows, labels, 1, 1);

        assertArrayEquals(new int[] {1}, result.counts());
        assertArrayEquals(new double[] {2.0}, result.centroids()[0], 1e-9);
    }

    @Test
    void centroidsAndCountsStayInMarkerSpaceRegardlessOfPcaClusteringInput() {
        // Requirement 4/Task 4 guard: when the all-cells driver PCA-reduces the pooled matrix
        // BEFORE clusterViaAnn, centroidsAndCounts must still be called on the ORIGINAL pooled
        // z-scored MARKER matrix (not the PCA-reduced one) -- this pins that contract down by
        // actually clustering in PC space and then computing centroids from the marker rows,
        // asserting the returned centroid width is nMarkers (not nComponents).
        Random rng = new Random(77);
        int per = 200;
        int n = per * 2;
        int nMarkers = 80; // > ScatterMath.PCA_DEFAULT_THRESHOLD (50) so PCA reduction applies.
        double[][] raw = new double[n][nMarkers];
        for (int i = 0; i < n; i++) {
            double blobShift = i < per ? 0.0 : 20.0;
            for (int j = 0; j < nMarkers; j++) {
                raw[i][j] = blobShift + rng.nextGaussian() * 0.5;
            }
        }
        double[][] markerSpace = ScatterMath.standardizeColumns(raw);

        ScatterMath.PcaReduction pca = ScatterMath.pcaReduce(markerSpace, 10, 50, 100_000, 42L);
        assertTrue(pca.applied(), "80 columns > threshold 50 must trigger PCA");
        assertEquals(10, pca.reduced()[0].length, "clustering input must be PC-space width");

        LeidenResult clustered = LeidenModel.clusterViaAnn(pca.reduced(), 15, 0.5, 5, 42L, true);
        int nClusters = Math.max(1, clustered.nClusters());

        // Centroids computed from the ORIGINAL marker-space matrix (never pca.reduced()).
        CentroidsAndCounts result =
                CohortClusterModel.centroidsAndCounts(markerSpace, clustered.labels(), nClusters, nMarkers);

        for (double[] centroid : result.centroids()) {
            assertEquals(
                    nMarkers,
                    centroid.length,
                    "centroid width must be nMarkers (marker space), not nComponents (PC space)");
        }
        int totalCounted = Arrays.stream(result.counts()).sum();
        assertEquals(n, totalCounted, "every pooled row must land in some cluster's count");
    }

    // ── writeClusterAllCells: cancel-during-write install guard ──────────────────

    /** A minimal fake {@link ProjectImageEntry} backed by an in-memory {@link ImageData}, no disk I/O. */
    @SuppressWarnings("unchecked")
    private static ProjectImageEntry<BufferedImage> fakeEntry(String imageName, ImageData<BufferedImage> data) {
        return (ProjectImageEntry<BufferedImage>) Proxy.newProxyInstance(
                ProjectImageEntry.class.getClassLoader(),
                new Class[] {ProjectImageEntry.class},
                (proxy, method, args) -> switch (method.getName()) {
                    case "getImageName" -> imageName;
                    case "readImageData" -> data;
                    case "saveImageData" -> null;
                    case "toString" -> "FakeEntry(" + imageName + ")";
                    case "hashCode" -> System.identityHashCode(proxy);
                    case "equals" -> proxy == args[0];
                    default ->
                        throw new UnsupportedOperationException(
                                "Method not implemented in fake entry: " + method.getName());
                });
    }

    @SuppressWarnings("unchecked")
    private static Project<BufferedImage> fakeProject(List<ProjectImageEntry<BufferedImage>> entries) {
        return (Project<BufferedImage>) Proxy.newProxyInstance(
                Project.class.getClassLoader(),
                new Class[] {Project.class},
                (proxy, method, args) -> switch (method.getName()) {
                    case "getImageList" -> entries;
                    case "toString" -> "FakeProject";
                    case "hashCode" -> System.identityHashCode(proxy);
                    case "equals" -> proxy == args[0];
                    default ->
                        throw new UnsupportedOperationException(
                                "Method not implemented in fake project: " + method.getName());
                });
    }

    /** {@code n} cells split into two well-separated marker-space blobs, so Leiden reliably finds >1 cluster. */
    private static List<PathObject> blobCells(List<String> markers, int n, double xOffset, Random rng) {
        List<PathObject> cells = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            PathObject cell = detectionAt(xOffset + i * 10, 0);
            double shift = i % 2 == 0 ? 0.0 : 25.0;
            for (String marker : markers) {
                cell.getMeasurementList().put(marker, shift + rng.nextGaussian());
            }
            cells.add(cell);
        }
        return cells;
    }

    @Test
    void writeClusterAllCellsInstallGuardIsSatisfiedOnCancelledButValidWrite() {
        // Fix 5 regression guard: writeClusterAllCells's own clustering (pass 1 pooling + the
        // single Leiden partition) always completes BEFORE the pass-2 write loop starts -- a
        // cancel only stops the write loop partway through. So a cancelled-but-non-aborted
        // AllCellsResult must still carry a fully populated nClusters/centroids/clusterCounts --
        // exactly what ScatterPlotView's install guard (fCentroids != null && fNClusters > 0) now
        // checks instead of gating on "not cancelled". This exercises the REAL production method
        // end-to-end (not just the pure runPass2Loop seam already covered above).
        List<String> markers = List.of("M1", "M2", "M3");
        Random rng = new Random(11);
        List<PathObject> image1Cells = blobCells(markers, 30, 0, rng);
        List<PathObject> image2Cells = blobCells(markers, 30, 1000, rng);

        ImageData<BufferedImage> data1 = fakeImageData(hierarchyWith(image1Cells));
        ImageData<BufferedImage> data2 = fakeImageData(hierarchyWith(image2Cells));
        var project = fakeProject(List.of(fakeEntry("image1", data1), fakeEntry("image2", data2)));

        CancellationToken token = new CancellationToken();
        // Cancel right after pass 2 writes the FIRST image (mirrors
        // cancelLeavesWrittenIntactAndReportsUnwrittenImages's mid-loop cancel timing, but here
        // driven by the real writeClusterAllCells log stream instead of a synthetic seam).
        List<String> logMessages = new ArrayList<>();
        java.util.function.Consumer<String> log = msg -> {
            logMessages.add(msg);
            if (msg.startsWith("Writing")) {
                token.cancel();
            }
        };

        CohortClusterModel.AllCellsResult result = CohortClusterModel.writeClusterAllCells(
                project,
                List.of("image1", "image2"),
                markers,
                10,
                1.0,
                1,
                42L,
                true,
                false,
                0,
                null,
                null,
                null, // batchShifts
                null,
                null,
                token,
                log,
                frac -> {});

        assertFalse(result.aborted(), "recall gate must not fail on this small, well-separated synthetic dataset");
        assertTrue(result.cancelled(), "the run must report cancellation (triggered after image1's write)");
        assertEquals(List.of("image1"), result.imagesWritten(), "image1 must have been written before cancel");
        assertEquals(List.of("image2"), result.imagesNotWritten(), "image2 must be reported unwritten after cancel");

        // The install-guard contract: nClusters/centroids/clusterCounts must be populated even
        // though the run was cancelled, because clustering itself already succeeded before any
        // write happened.
        assertTrue(result.nClusters() > 0, "a cancelled-but-valid run must still report a real cluster count");
        assertNotNull(result.centroids(), "centroids must be populated on a cancelled-but-valid run");
        assertEquals(result.nClusters(), result.centroids().length);
        assertNotNull(result.clusterCounts(), "clusterCounts must be populated on a cancelled-but-valid run");
        assertEquals(result.nClusters(), result.clusterCounts().length);
        assertTrue(result.cellsWritten() > 0, "the written image's cells must have received a cluster id");
    }

    @Test
    void writeClusterAllCellsCarriesTheMeasuredAnnRecallThroughToAllCellsResult() {
        // Item 8 (D-09): AllCellsResult.recall() must be the REAL measured recall from the
        // single clusterViaAnn partition on a successful run, not the old hardcoded -1.0
        // sentinel -- exercised end-to-end through the real production method.
        List<String> markers = List.of("M1", "M2", "M3");
        Random rng = new Random(21);
        List<PathObject> image1Cells = blobCells(markers, 30, 0, rng);
        ImageData<BufferedImage> data1 = fakeImageData(hierarchyWith(image1Cells));
        var project = fakeProject(List.of(fakeEntry("image1", data1)));

        CohortClusterModel.AllCellsResult result = CohortClusterModel.writeClusterAllCells(
                project,
                List.of("image1"),
                markers,
                10,
                1.0,
                1,
                42L,
                true,
                false,
                0,
                null,
                null,
                null, // batchShifts
                null,
                null,
                new CancellationToken(),
                msg -> {},
                frac -> {});

        assertFalse(result.aborted());
        assertFalse(result.cancelled());
        assertTrue(
                result.recall() > 0.0 && result.recall() <= 1.0,
                "a successful all-cells run must carry a real measured recall in (0, 1], got " + result.recall());
    }

    @Test
    void writeClusterAllCellsPass2IsUuidDrivenAndIgnoresClassChangeBetweenPasses() {
        // Item 6 regression guard: pass 2 write-back must be purely UUID-driven -- a cell that
        // matched the classFilter at pass 1 (pooled, present in globalLabelMap) must still get
        // its pass-1 cluster label even if its PathClass changes before pass 2 re-reads it. Only
        // a cell that NEVER matched the filter at pass 1 (absent from globalLabelMap) should be
        // written -1.
        List<String> markers = List.of("M1", "M2", "M3");
        Random rng = new Random(17);
        List<PathObject> cells = blobCells(markers, 40, 0, rng);
        PathClass tumor = PathClass.fromString("Tumor");
        PathClass stroma = PathClass.fromString("Stroma");
        for (PathObject cell : cells) {
            cell.setPathClass(tumor);
        }
        // Never matches the classFilter at all -- stays outside the pass-1 pooled population.
        PathObject neverMatched = cells.get(cells.size() - 1);
        neverMatched.setPathClass(stroma);
        // Matches at pass 1 (pooled with a real label), but its class flips before pass 2 runs.
        PathObject classChangedCell = cells.get(0);

        ImageData<BufferedImage> data = fakeImageData(hierarchyWith(cells));
        var project = fakeProject(List.of(fakeEntry("image1", data)));

        java.util.function.Consumer<String> log = msg -> {
            if (msg.equals("Running Leiden…")) {
                // Pass 1 pooling has already finished (this cell's UUID is already keyed into
                // the labels that will build globalLabelMap); pass 2 hasn't started yet.
                classChangedCell.setPathClass(stroma);
            }
        };

        CohortClusterModel.AllCellsResult result = CohortClusterModel.writeClusterAllCells(
                project,
                List.of("image1"),
                markers,
                10,
                1.0,
                1,
                42L,
                true,
                false,
                0,
                "Tumor",
                null,
                null, // batchShifts
                null,
                null,
                new CancellationToken(),
                log,
                frac -> {});

        assertFalse(result.aborted(), "recall gate must not fail on this small, well-separated synthetic dataset");
        assertFalse(result.cancelled());

        double changedValue = classChangedCell.getMeasurementList().get(CohortClusterModel.CLUSTER_MEASUREMENT);
        assertTrue(
                changedValue >= 1.0,
                "a cell that matched the filter at pass 1 must keep its pass-1 label even though its "
                        + "PathClass changed before pass 2 -- write-back is UUID-driven, not re-filtered");

        double neverMatchedValue = neverMatched.getMeasurementList().get(CohortClusterModel.CLUSTER_MEASUREMENT);
        assertEquals(-1.0, neverMatchedValue, "a cell that never matched the classFilter at pass 1 must still be -1");
    }

    // ── COF-08: poolAllCellsRaw returns RAW rows while poolAllCells z-scores ─────

    @Test
    void poolAllCellsRawReturnsRawRowsWhilePoolAllCellsZScores() {
        // Two images with a KNOWN single marker "M1" so the pooled column has a non-zero mean and
        // known raw values. poolAllCellsRaw must return those raw values verbatim, while
        // poolAllCells z-scores the SAME input — proving the cofactor estimator is fed RAW
        // intensities, not centred ones (COF-08 / critical correction #3: z-scored rows give
        // p50 ≈ 0 and a non-positive cofactor).
        List<String> markers = List.of("M1");
        List<String> images = List.of("imgA", "imgB");

        double[] aValues = {10, 20, 30, 40, 50};
        double[] bValues = {60, 70, 80};

        List<PathObject> aCells = new ArrayList<>();
        for (int i = 0; i < aValues.length; i++) {
            PathObject cell = detectionAt(i * 10, 0);
            cell.getMeasurementList().put("M1", aValues[i]);
            aCells.add(cell);
        }
        List<PathObject> bCells = new ArrayList<>();
        for (int i = 0; i < bValues.length; i++) {
            PathObject cell = detectionAt(1000 + i * 10, 0);
            cell.getMeasurementList().put("M1", bValues[i]);
            bCells.add(cell);
        }

        var project = fakeProject(List.of(
                fakeEntry("imgA", fakeImageData(hierarchyWith(aCells))),
                fakeEntry("imgB", fakeImageData(hierarchyWith(bCells)))));

        CancellationToken token = new CancellationToken();
        CohortClusterModel.PooledData raw =
                CohortClusterModel.poolAllCellsRaw(project, images, markers, null, null, null, null, token, s -> {});
        CohortClusterModel.PooledData z =
                CohortClusterModel.poolAllCells(project, images, markers, null, null, null, null, null, token, s -> {});

        // Both pool EVERY cell (no sampling): 5 + 3 = 8 rows.
        assertEquals(8, raw.rows().length, "poolAllCellsRaw must pool every cell (no sampling)");
        assertEquals(8, z.rows().length, "poolAllCells must pool the same 8 cells");

        // RAW rows equal the injected values exactly (sort the column so the assertion is
        // independent of per-image hierarchy iteration order).
        double[] rawCol = new double[raw.rows().length];
        for (int i = 0; i < raw.rows().length; i++) {
            rawCol[i] = raw.rows()[i][0];
        }
        double[] sortedRaw = rawCol.clone();
        Arrays.sort(sortedRaw);
        assertArrayEquals(
                new double[] {10, 20, 30, 40, 50, 60, 70, 80},
                sortedRaw,
                1e-9,
                "poolAllCellsRaw must return the raw injected intensities, un-standardized");

        double rawMean = Arrays.stream(rawCol).average().orElse(Double.NaN);
        assertEquals(45.0, rawMean, 1e-9, "the raw column mean is the arithmetic mean of the injected values");

        // Z-SCORED rows are centred: the z column mean is ≈ 0 and clearly NOT the raw values.
        double zMean = 0.0;
        for (int i = 0; i < z.rows().length; i++) {
            zMean += z.rows()[i][0];
        }
        zMean /= z.rows().length;
        assertEquals(0.0, zMean, 1e-6, "poolAllCells z-scores the column → mean ≈ 0");
        assertNotEquals(
                rawMean,
                zMean,
                1e-6,
                "the z-scored column mean (≈0) must differ from the raw column mean (≈45) — raw ≠ z-scored");
    }

    // ── Fix 4: k-means cohort assignment must run in the queryProjector's space ─

    @Test
    void writeClusterAcrossProjectAppliesQueryProjectorBeforeNearestCentroid() {
        // Regression guard: when the fit clustered in a reduced/rotated space (PCA in production),
        // the centroids passed here live in THAT space, not raw marker space -- so the query row
        // must be projected into the same space before nearestCentroid, or boundary cells land in
        // the wrong cluster relative to what the user actually previewed.
        List<String> markers = List.of("M1", "M2");
        double[] mean = {0.0, 0.0};
        double[] sd = {1.0, 1.0};
        // Centroids live in a space where the two marker dimensions are SWAPPED relative to raw
        // marker space -- e.g. the "clustering space" produced by a projector that swaps columns.
        double[][] centroids = {{0.0, 10.0}, {10.0, 0.0}};
        // Swaps the two columns -- a stand-in for a PCA projector to keep the test's arithmetic
        // exact and easy to reason about, while still exercising a REAL non-identity projector.
        java.util.function.UnaryOperator<double[][]> swapProjector = rows -> {
            double[][] out = new double[rows.length][];
            for (int i = 0; i < rows.length; i++) {
                out[i] = new double[] {rows[i][1], rows[i][0]};
            }
            return out;
        };

        PathObject cell = detectionAt(0, 0);
        cell.getMeasurementList().put("M1", 10.0);
        cell.getMeasurementList().put("M2", 0.0);

        // Without projection: raw query row [10, 0] is nearest centroids[1] = {10, 0} (distance 0)
        // -> label 1 -> written value 2.0.
        ImageData<BufferedImage> dataIdentity = fakeImageData(hierarchyWith(List.of(cell)));
        var projectIdentity = fakeProject(List.of(fakeEntry("img", dataIdentity)));
        CohortClusterModel.writeClusterAcrossProject(
                projectIdentity,
                List.of("img"),
                markers,
                mean,
                sd,
                centroids,
                null, // null projector == identity
                null,
                null,
                null,
                null,
                msg -> {},
                frac -> {});
        double writtenIdentity = cell.getMeasurementList().get(CohortClusterModel.CLUSTER_MEASUREMENT);
        assertEquals(2.0, writtenIdentity, 1e-9, "unprojected query row [10,0] is nearest centroids[1]={10,0}");

        // With the swap projector: the SAME raw query row [10,0] projects to [0,10], which is
        // nearest centroids[0] = {0, 10} (distance 0) -> label 0 -> written value 1.0. The
        // assignment MUST flip relative to the identity case above -- proving the projector is
        // actually applied before nearestCentroid, not ignored.
        PathObject cell2 = detectionAt(0, 0);
        cell2.getMeasurementList().put("M1", 10.0);
        cell2.getMeasurementList().put("M2", 0.0);
        ImageData<BufferedImage> dataProjected = fakeImageData(hierarchyWith(List.of(cell2)));
        var projectProjected = fakeProject(List.of(fakeEntry("img", dataProjected)));
        CohortClusterModel.writeClusterAcrossProject(
                projectProjected,
                List.of("img"),
                markers,
                mean,
                sd,
                centroids,
                swapProjector,
                null,
                null,
                null,
                null,
                msg -> {},
                frac -> {});
        double writtenProjected = cell2.getMeasurementList().get(CohortClusterModel.CLUSTER_MEASUREMENT);
        assertEquals(
                1.0,
                writtenProjected,
                1e-9,
                "projected query row [0,10] is nearest centroids[0]={0,10} -- the projected assignment must win");
        assertNotEquals(
                writtenIdentity,
                writtenProjected,
                "the SAME raw query row must be assigned to a DIFFERENT cluster once queryProjector is applied");

        // Explicit identity must reproduce the null-projector (current/legacy) behaviour exactly.
        PathObject cell3 = detectionAt(0, 0);
        cell3.getMeasurementList().put("M1", 10.0);
        cell3.getMeasurementList().put("M2", 0.0);
        ImageData<BufferedImage> dataExplicitIdentity = fakeImageData(hierarchyWith(List.of(cell3)));
        var projectExplicitIdentity = fakeProject(List.of(fakeEntry("img", dataExplicitIdentity)));
        CohortClusterModel.writeClusterAcrossProject(
                projectExplicitIdentity,
                List.of("img"),
                markers,
                mean,
                sd,
                centroids,
                java.util.function.UnaryOperator.identity(),
                null,
                null,
                null,
                null,
                msg -> {},
                frac -> {});
        double writtenExplicitIdentity = cell3.getMeasurementList().get(CohortClusterModel.CLUSTER_MEASUREMENT);
        assertEquals(
                writtenIdentity,
                writtenExplicitIdentity,
                "an explicit identity projector must reproduce the null-projector (PCA-off) behaviour exactly");
    }

    // ── selectImageSource (pool the open image from live data, not disk) ────────

    @Test
    void selectImageSourceReturnsLiveOpenDataWithoutTouchingDiskForTheOpenImage() {
        ImageData<BufferedImage> openData = fakeImageData(hierarchyWith(detections(2, 0)));
        AtomicInteger diskReads = new AtomicInteger();

        ImageData<BufferedImage> resolved = CohortClusterModel.selectImageSource("imgA", "imgA", openData, () -> {
            diskReads.incrementAndGet();
            return fakeImageData(hierarchyWith(detections(9, 500))); // a distinct, "stale disk" instance
        });

        assertSame(openData, resolved, "the open image must resolve to the exact live openData instance");
        assertEquals(0, diskReads.get(), "disk must never be read for the open image");
    }

    @Test
    void selectImageSourceReadsDiskExactlyOnceForNonOpenImages() {
        ImageData<BufferedImage> openData = fakeImageData(hierarchyWith(detections(2, 0)));
        ImageData<BufferedImage> diskData = fakeImageData(hierarchyWith(detections(9, 500)));
        AtomicInteger diskReads = new AtomicInteger();

        ImageData<BufferedImage> resolved = CohortClusterModel.selectImageSource("imgB", "imgA", openData, () -> {
            diskReads.incrementAndGet();
            return diskData;
        });

        assertSame(diskData, resolved, "a non-open image must resolve to whatever the disk reader supplies");
        assertEquals(1, diskReads.get(), "disk must be read exactly once for a non-open image");
    }

    @Test
    void selectImageSourceTreatsNullOpenNameAsNoOpenImage() {
        ImageData<BufferedImage> diskData = fakeImageData(hierarchyWith(detections(1, 0)));
        AtomicInteger diskReads = new AtomicInteger();

        ImageData<BufferedImage> resolved = CohortClusterModel.selectImageSource("imgA", null, null, () -> {
            diskReads.incrementAndGet();
            return diskData;
        });

        assertSame(diskData, resolved, "with no open image, every name must read from disk");
        assertEquals(1, diskReads.get());
    }

    // ── classForMeasurementValue / assignmentsFromMeasurement (all-cells assign) ─

    @Test
    void classForMeasurementValueDecodesOneBasedWrittenValue() {
        PathClass tumor = PathClass.fromString("Tumor");
        PathClass stroma = PathClass.fromString("Stroma");
        Map<Integer, PathClass> mapping = Map.of(0, tumor, 2, stroma);

        // Written value is 1-based (label + 1): cluster 0 -> value 1.0, cluster 2 -> value 3.0.
        assertEquals(tumor, CohortClusterModel.classForMeasurementValue(1.0, mapping));
        assertEquals(stroma, CohortClusterModel.classForMeasurementValue(3.0, mapping));
    }

    @Test
    void classForMeasurementValueReturnsNullForNaNUnclusteredOrUnmapped() {
        PathClass tumor = PathClass.fromString("Tumor");
        Map<Integer, PathClass> mapping = Map.of(0, tumor);

        assertNull(CohortClusterModel.classForMeasurementValue(Double.NaN, mapping), "missing measurement -> no class");
        // -1 (unclustered sentinel) decodes to label -2, never a real mapping key.
        assertNull(CohortClusterModel.classForMeasurementValue(-1.0, mapping), "unclustered cell -> no class");
        // Cluster 1 exists in the written data but the user chose "skip" for it (absent from mapping).
        assertNull(CohortClusterModel.classForMeasurementValue(2.0, mapping), "unmapped cluster -> no class (skip)");
    }

    @Test
    void assignmentsFromMeasurementMatchesOnlyMappedWrittenClusters() {
        PathClass tumor = PathClass.fromString("Tumor");
        PathClass stroma = PathClass.fromString("Stroma");
        Map<Integer, PathClass> mapping = Map.of(0, tumor, 1, stroma);

        PathObject clusterZero = detectionAt(0, 0); // will carry written value 1.0 -> cluster 0 -> tumor
        PathObject clusterOne = detectionAt(10, 0); // written value 2.0 -> cluster 1 -> stroma
        PathObject skipped = detectionAt(20, 0); // written value 3.0 -> cluster 2 -> not in mapping, skipped
        PathObject unclustered = detectionAt(30, 0); // written value -1.0 -> unclustered, skipped
        PathObject noMeasurement = detectionAt(40, 0); // never written, NaN, skipped

        clusterZero.getMeasurementList().put(CohortClusterModel.CLUSTER_MEASUREMENT, 1.0);
        clusterOne.getMeasurementList().put(CohortClusterModel.CLUSTER_MEASUREMENT, 2.0);
        skipped.getMeasurementList().put(CohortClusterModel.CLUSTER_MEASUREMENT, 3.0);
        unclustered.getMeasurementList().put(CohortClusterModel.CLUSTER_MEASUREMENT, -1.0);

        List<PathObject> cells = List.of(clusterZero, clusterOne, skipped, unclustered, noMeasurement);
        MeasurementAssignment assignment = CohortClusterModel.assignmentsFromMeasurement(cells, mapping);

        assertEquals(2, assignment.objects().size(), "only the two mapped, clustered cells are matched");
        assertEquals(List.of(clusterZero, clusterOne), assignment.objects());
        assertEquals(List.of(tumor, stroma), assignment.classes());
    }

    // ── nearestCentroid ──────────────────────────────────────────────────────

    @Test
    void nearestCentroidPicksClosestBySquaredEuclidean() {
        double[][] cents = {{0, 0}, {10, 10}};
        assertEquals(0, CohortClusterModel.nearestCentroid(new double[] {1, 1}, cents));
        assertEquals(1, CohortClusterModel.nearestCentroid(new double[] {9, 9}, cents));
        assertEquals(0, CohortClusterModel.nearestCentroid(new double[] {4.9, 4.9}, cents));
        assertEquals(1, CohortClusterModel.nearestCentroid(new double[] {5.1, 5.1}, cents));
    }

    @Test
    void nearestCentroidBreaksTiesToFirst() {
        double[][] cents = {{0, 0}, {2, 2}};
        // Equidistant point → the first centroid wins (strict < comparison).
        assertEquals(0, CohortClusterModel.nearestCentroid(new double[] {1, 1}, cents));
    }

    @Test
    void nearestCentroidWorksWithSingleCluster() {
        double[][] cents = {{3, 3, 3}};
        assertEquals(0, CohortClusterModel.nearestCentroid(new double[] {-5, 100, 0}, cents));
    }

    // ── sampleIndices ────────────────────────────────────────────────────────

    @Test
    void sampleIndicesReturnsAllWhenCountAtLeastN() {
        int[] all = CohortClusterModel.sampleIndices(5, 5, new Random(1));
        assertArrayEquals(new int[] {0, 1, 2, 3, 4}, all);

        int[] capped = CohortClusterModel.sampleIndices(3, 10, new Random(1));
        assertArrayEquals(new int[] {0, 1, 2}, capped);
    }

    @Test
    void sampleIndicesReturnsRequestedCountOfDistinctInRangeIndices() {
        int n = 100;
        int count = 20;
        int[] pick = CohortClusterModel.sampleIndices(n, count, new Random(42));

        assertEquals(count, pick.length);
        // All in range.
        assertTrue(Arrays.stream(pick).allMatch(i -> i >= 0 && i < n));
        // All distinct.
        assertEquals(count, Arrays.stream(pick).distinct().count(), "indices must be distinct");
    }

    @Test
    void sampleIndicesIsDeterministicForAFixedSeed() {
        int[] a = CohortClusterModel.sampleIndices(50, 10, new Random(7));
        int[] b = CohortClusterModel.sampleIndices(50, 10, new Random(7));
        assertArrayEquals(a, b, "same seed → same sample");
    }

    @Test
    void sampleIndicesCoversWholeRangeOverManyDraws() {
        // Sanity: with count==n via the fast path we get every index exactly once.
        int[] pick = CohortClusterModel.sampleIndices(8, 8, new Random(0));
        assertEquals(8, IntStream.of(pick).distinct().count());
    }

    // ── Growable*Array overflow-safe growth ─────────────────────────────────────

    @Test
    void growableLongArrayGrowsPastATinyForcedCapacityAndPreservesContents() {
        // Force an initial capacity of 1 so add() must grow several times over a handful
        // of elements — exercising nextGrowableCapacity's doubling path directly rather
        // than waiting for the real 1024-element default to fill up.
        CohortClusterModel.GrowableLongArray arr = new CohortClusterModel.GrowableLongArray(1);
        long[] expected = new long[50];
        for (int i = 0; i < expected.length; i++) {
            expected[i] = i * 7L;
            arr.add(expected[i]);
        }
        assertArrayEquals(expected, arr.toArray(), "growth must preserve every previously-added element in order");
    }

    @Test
    void growableIntArrayGrowsPastATinyForcedCapacityAndPreservesContents() {
        CohortClusterModel.GrowableIntArray arr = new CohortClusterModel.GrowableIntArray(1);
        int[] expected = new int[50];
        for (int i = 0; i < expected.length; i++) {
            expected[i] = i * 3;
            arr.add(expected[i]);
        }
        assertArrayEquals(expected, arr.toArray(), "growth must preserve every previously-added element in order");
    }

    @Test
    void nextGrowableCapacityClampsBelowIntOverflowInsteadOfWrappingNegative() {
        // A length whose doubling overflows a signed int (length*2 > Integer.MAX_VALUE) but
        // which is still comfortably below the clamped ceiling — the clamp must kick in and
        // return the ceiling itself, not a wrapped negative value or a NegativeArraySizeException.
        int length = 1_500_000_000; // length * 2 == 3,000,000,000 > Integer.MAX_VALUE
        int next = CohortClusterModel.nextGrowableCapacity(length);
        assertTrue(next > length, "capacity must still increase");
        assertTrue(next > 0, "clamped capacity must never be negative");
        assertEquals(Integer.MAX_VALUE - 8, next, "must clamp to the documented ceiling");
    }

    @Test
    void nextGrowableCapacityThrowsAtTheClampedCeiling() {
        int ceiling = Integer.MAX_VALUE - 8;
        assertThrows(IllegalStateException.class, () -> CohortClusterModel.nextGrowableCapacity(ceiling));
    }

    @Test
    void nextGrowableCapacityDoublesNormally() {
        assertEquals(2048, CohortClusterModel.nextGrowableCapacity(1024));
        assertEquals(2, CohortClusterModel.nextGrowableCapacity(1));
    }
}
