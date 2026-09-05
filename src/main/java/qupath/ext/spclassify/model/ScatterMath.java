package qupath.ext.spclassify.model;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.UnaryOperator;
import smile.feature.extraction.PCA;
import smile.manifold.UMAP;

/**
 * Pure numerical helpers for the cell scatter plot: column standardisation, the
 * PCA/UMAP 2-D embeddings, deterministic subsampling, and point-in-polygon
 * hit-testing. No JavaFX, no per-row state — every method is static and a pure
 * function of its arguments, so the embedding core is unit-testable in isolation
 * from the interactive {@code ScatterPlotView}.
 */
public final class ScatterMath {

    private ScatterMath() {}

    // ── Standardisation ──────────────────────────────────────────────────────

    /** Z-scores each column (per-column mean 0, sd 1; constant columns → 0). */
    public static double[][] standardizeColumns(double[][] data) {
        int p = data.length == 0 ? 0 : data[0].length;
        return standardizeColumns(data, new double[p], new double[p]);
    }

    /**
     * Z-scores each column, writing the per-column mean/sd into {@code outMean}/
     * {@code outSd} so the same transform can be replayed at cohort-assign time.
     * <p>
     * NaN-robust: a {@code NaN} entry (a missing QuPath measurement) is excluded
     * from that column's mean/sd computation rather than poisoning it, and is
     * itself imputed to 0 in the output (the column mean, in z-score terms) —
     * every OTHER cell in the column still gets a meaningful z-score instead of
     * the whole column collapsing to {@code NaN}. A column that is entirely
     * {@code NaN} (or constant) maps to all zeros, matching the existing
     * near-zero-sd behaviour. For fully-finite input this is byte-identical to
     * summing over every row (the non-NaN count equals {@code n}).
     */
    public static double[][] standardizeColumns(double[][] data, double[] outMean, double[] outSd) {
        int n = data.length;
        int p = n == 0 ? 0 : data[0].length;
        double[][] out = new double[n][p];
        for (int j = 0; j < p; j++) {
            double sum = 0;
            int count = 0;
            for (double[] row : data) {
                double v = row[j];
                if (!Double.isNaN(v)) {
                    sum += v;
                    count++;
                }
            }
            double mean = count > 0 ? sum / count : 0.0;
            double var = 0;
            for (double[] row : data) {
                double v = row[j];
                if (!Double.isNaN(v)) {
                    double d = v - mean;
                    var += d * d;
                }
            }
            double sd = count > 0 ? Math.sqrt(var / count) : 0.0;
            outMean[j] = mean;
            outSd[j] = sd;
            double inv = sd < 1e-9 ? 0.0 : 1.0 / sd;
            for (int i = 0; i < n; i++) {
                double v = data[i][j];
                out[i][j] = Double.isNaN(v) ? 0.0 : (v - mean) * inv;
            }
        }
        return out;
    }

    // ── Embeddings ───────────────────────────────────────────────────────────

    /** Projects all rows onto their first two principal components. */
    public static void fillPca(double[][] std, int n, double[] nx, double[] ny) {
        double[][] proj = PCA.fit(std).getProjection(2).apply(std);
        for (int i = 0; i < n; i++) {
            nx[i] = proj[i][0];
            ny[i] = proj[i][1];
        }
    }

    /**
     * Embeds the rows with UMAP, subsampling to {@code maxCells} when the input is
     * larger (pass {@link Integer#MAX_VALUE} to embed everything). Returns a
     * status notice (currently empty).
     *
     * @throws LinkageError if the native ARPACK library cannot be loaded (caller
     *     falls back to PCA)
     */
    public static String fillUmap(double[][] std, int n, double[] nx, double[] ny, int maxCells) {
        // UMAP: subsample if larger than the cap; embed a connected subset.
        int[] sub = (n > maxCells) ? randomSubsample(n, maxCells) : identity(n);
        // The status bar's "clustered · plotted" counts convey the subsample, so
        // no extra notice is emitted here.
        String notice = "";
        double[][] subMatrix = new double[sub.length][];
        for (int s = 0; s < sub.length; s++) {
            subMatrix[s] = std[sub[s]];
        }
        int neighbors = Math.min(15, subMatrix.length - 1);
        if (neighbors < 2) {
            throw new IllegalStateException("Too few cells for UMAP (need at least 3).");
        }
        UMAP umap = UMAP.of(subMatrix, neighbors);
        // coordinates[j] corresponds to subMatrix row umap.index[j].
        for (int j = 0; j < umap.coordinates.length; j++) {
            int orig = sub[umap.index[j]];
            nx[orig] = umap.coordinates[j][0];
            ny[orig] = umap.coordinates[j][1];
        }
        return notice;
    }

    // ── Conditional PCA reduction (pre-clustering) ───────────────────────────

    /**
     * Default max PCA components kept when reduction is applied (mirrors scanpy's
     * {@code sc.pp.pca} default {@code n_comps=50}).
     */
    public static final int PCA_DEFAULT_MAX_COMPONENTS = 50;

    /**
     * Default feature-count threshold above which PCA reduction applies (mirrors
     * scanpy's {@code sc.pp.neighbors}, which builds the graph on {@code X_pca}
     * once the matrix has more than 50 variables).
     */
    public static final int PCA_DEFAULT_THRESHOLD = 50;

    /**
     * Default cap on the row count PCA is FIT over; above this cap a deterministic
     * seeded subsample is fit and the resulting projection applied to every row —
     * bounds fit cost/memory independent of total row count (the all-cells cohort
     * path can pool tens of millions of rows; fitting {@code smile.feature.extraction.PCA}
     * needs the full covariance of whatever it is fit on, so the FIT set, not the
     * APPLY set, must stay bounded).
     */
    public static final int PCA_DEFAULT_FIT_SAMPLE_CAP = 100_000;

    /**
     * Fixed seed for the PCA subsample fit — deliberately independent of any
     * caller's own reproducibility toggle (e.g. the scatter view's "Sample
     * multiple seeds" / Leiden/k-means seed), so PCA reduction itself never
     * introduces nondeterminism regardless of that toggle's state.
     */
    public static final long PCA_DEFAULT_SEED = 42L;

    /**
     * Result of {@link #pcaReduce}: the (possibly unchanged) reduced matrix,
     * whether reduction was actually applied, the component count kept, the
     * cumulative variance proportion explained by those components (1.0 when not
     * applied — the identity "reduction" trivially explains all variance), and a
     * reusable projector that maps NEW rows (same original column count, same
     * z-scoring convention) into the SAME PC space — needed so a later "transfer"
     * step (kNN label transfer against a fitted reference) can project query rows
     * into the identical basis the reference was clustered in.
     */
    public record PcaReduction(
            double[][] reduced,
            boolean applied,
            int nComponents,
            double cumulativeVariance,
            UnaryOperator<double[][]> projector) {}

    /** {@link #pcaReduce} with the scanpy-mirroring defaults ({@link #PCA_DEFAULT_MAX_COMPONENTS} etc.). */
    public static PcaReduction pcaReduce(double[][] std) {
        return pcaReduce(
                std, PCA_DEFAULT_MAX_COMPONENTS, PCA_DEFAULT_THRESHOLD, PCA_DEFAULT_FIT_SAMPLE_CAP, PCA_DEFAULT_SEED);
    }

    /**
     * Conditional PCA dimensionality reduction, applied to the z-scored {@code std}
     * matrix BEFORE the clustering kNN graph is built (the scanpy {@code scale ->
     * PCA -> neighbors} recipe): real projects can carry 1000+ per-cell
     * measurements (each marker x several statistics x several compartments),
     * and unreduced Euclidean kNN over that many equally-weighted columns both
     * lets whichever marker happens to have the most measurements dominate the
     * distance, and suffers the usual high-dimensional distance-concentration
     * problem. Reusing {@code smile.feature.extraction.PCA} (already used by
     * {@link #fillPca}) keeps this deterministic — it is an EXACT covariance
     * eigendecomposition, not a randomized/seeded SVD.
     *
     * <p>A no-op (identity) pass-through when {@code maxComponents <= 0} (PCA
     * disabled), the column count is at or below {@code threshold} (a small,
     * curated panel — projecting onto >= p components is just a lossless
     * rotation that only adds cost), or there are fewer than 2 rows. Otherwise
     * reduces to {@code min(maxComponents, cols-1, rows-1)} components, fit on a
     * deterministic seeded subsample of size {@code fitSampleCap} when
     * {@code std.length > fitSampleCap} (bounding fit cost/memory independent of
     * total row count — critical at cohort scale), else fit on the full matrix.
     * The resulting projection (which carries the FIT's own center, from
     * {@code PCA.getProjection(int)}) is then applied to EVERY row of
     * {@code std} — not just the rows it was fit on — and is also returned as
     * {@link PcaReduction#projector} so a caller can project OTHER new rows
     * (e.g. a "transfer" query image's z-scored rows, same column count) into
     * the identical PC basis later.
     *
     * @param std z-scored input matrix (rows = cells, columns = markers/measurements)
     * @param maxComponents PCA components to keep when applied ({@code <= 0} disables PCA entirely)
     * @param threshold column-count threshold at/below which PCA is skipped (scanpy default: 50)
     * @param fitSampleCap row-count cap PCA is FIT over; above this a seeded subsample is fit and
     *     the fitted projection applied to every row
     * @param seed seed for the deterministic subsample fit (independent of any caller reproducibility toggle)
     */
    public static PcaReduction pcaReduce(
            double[][] std, int maxComponents, int threshold, int fitSampleCap, long seed) {
        int n = std.length;
        int p = n == 0 ? 0 : std[0].length;
        if (maxComponents <= 0 || p <= threshold || n < 2) {
            return new PcaReduction(std, false, p, 1.0, UnaryOperator.identity());
        }
        int nComp = Math.min(maxComponents, Math.min(p - 1, n - 1));
        if (nComp <= 0) {
            return new PcaReduction(std, false, p, 1.0, UnaryOperator.identity());
        }
        double[][] fitRows;
        if (n > fitSampleCap) {
            int[] sampleIdx = randomSubsample(n, fitSampleCap, seed);
            fitRows = new double[sampleIdx.length][];
            for (int i = 0; i < sampleIdx.length; i++) {
                fitRows[i] = std[sampleIdx[i]];
            }
        } else {
            fitRows = std;
        }
        PCA fitted = PCA.fit(fitRows);
        PCA projection = fitted.getProjection(nComp);
        double[] cumProp = fitted.cumulativeVarianceProportion();
        double cumVar = cumProp.length == 0 ? 1.0 : cumProp[Math.min(nComp, cumProp.length) - 1];
        double[][] reduced = projection.apply(std);
        return new PcaReduction(reduced, true, nComp, cumVar, projection::apply);
    }

    // ── Subsampling ──────────────────────────────────────────────────────────

    /** Identity index array {@code [0, 1, ..., n-1]}. */
    public static int[] identity(int n) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        }
        return a;
    }

    /**
     * Deterministic (fixed seed 42) subsample of {@code count} distinct indices
     * in {@code [0, n)}, returned sorted ascending. Uses a fixed seed so repeated
     * runs over the same data plot the same points.
     */
    public static int[] randomSubsample(int n, int count) {
        return randomSubsample(n, count, 42L);
    }

    /**
     * Deterministic (seeded) subsample of {@code count} distinct indices in
     * {@code [0, n)}, returned sorted ascending — same partial Fisher-Yates as
     * {@link #randomSubsample(int, int)} but with a caller-supplied seed, so a
     * consumer (e.g. {@link #pcaReduce}'s subsample-fit path) can use a seed
     * independent of this class's own fixed-42 plotting convention.
     */
    public static int[] randomSubsample(int n, int count, long seed) {
        int[] all = identity(n);
        Random rng = new Random(seed);
        // Partial Fisher-Yates: first `count` slots become the sample.
        for (int i = 0; i < count; i++) {
            int j = i + rng.nextInt(n - i);
            int tmp = all[i];
            all[i] = all[j];
            all[j] = tmp;
        }
        int[] out = new int[count];
        System.arraycopy(all, 0, out, 0, count);
        Arrays.sort(out);
        return out;
    }

    // ── Geometry ─────────────────────────────────────────────────────────────

    /**
     * Even-odd ray-cast point-in-polygon test. {@code poly} is a list of
     * {@code [x, y]} vertices; the polygon is treated as closed.
     */
    public static boolean pointInPolygon(double x, double y, List<double[]> poly) {
        boolean in = false;
        int n = poly.size();
        for (int i = 0, j = n - 1; i < n; j = i++) {
            double xi = poly.get(i)[0], yi = poly.get(i)[1];
            double xj = poly.get(j)[0], yj = poly.get(j)[1];
            boolean intersect = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi);
            if (intersect) {
                in = !in;
            }
        }
        return in;
    }
}
