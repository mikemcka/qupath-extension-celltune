package qupath.ext.spclassify.classifier;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.stream.IntStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.objects.PathObject;

/**
 * Unsupervised feature pruner with marker-group guardrails.
 *
 * <p>Designed to be run on the <b>labelled cell set</b> at the start of each
 * training round. The labelled set is small (typically tens to a few thousand
 * cells), so the pipeline runs in milliseconds and uses every cell — no
 * subsampling.
 *
 * <p>Pipeline:
 * <ol>
 *   <li><b>Sparsity / variance filter</b> — drop features that are non-zero in
 *       fewer than {@code minNonZeroCells} cells.</li>
 *   <li><b>Within-marker correlation removal</b> — within each marker group,
 *       keep the highest-variance feature and drop peers with |Pearson r|
 *       above the threshold. Groups are processed in parallel.</li>
 *   <li><b>Cross-marker correlation removal</b> — optional, off by default.
 *       Greedy keep-set across all surviving features.</li>
 *   <li><b>Per-group whitelist</b> — the {@code minKeptPerGroup} highest-variance
 *       features in every marker group are always kept, immune to the stages above
 *       (a group with fewer features keeps them all). This guarantees the classifier
 *       never goes blind to a marker and that each marker keeps its strongest few
 *       features even if they correlate.</li>
 * </ol>
 *
 * <p>The "marker group" of a feature is decided by {@link #extractGroup(String)}: the
 * prefix before the first {@code ": "}, else the token before the first underscore or
 * space, matched <b>case-insensitively</b>. This is intentionally distinct from the
 * UI category grouping in {@code FeatureSelectionPane} (markers / morphology /
 * neighbours / embeddings), which is semantic rather than structural.
 */
public final class FeaturePruner {

    private static final Logger logger = LoggerFactory.getLogger(FeaturePruner.class);

    private FeaturePruner() {}

    /**
     * Pruning configuration.
     *
     * @param minNonZeroCells            drop a feature if non-zero in fewer than this many cells
     * @param withinMarkerCorrThreshold  drop within-marker peers with |r| above this threshold
     * @param crossMarkerCorrThreshold   |r| threshold for cross-marker dropping; use 1.0 to disable
     * @param minKeptPerGroup            always keep this many highest-variance features per marker
     *                                   group, immune to the sparsity/correlation stages (a group
     *                                   with fewer features keeps them all); 0 disables the guardrail
     */
    public record PruneOptions(
            int minNonZeroCells,
            double withinMarkerCorrThreshold,
            double crossMarkerCorrThreshold,
            int minKeptPerGroup) {
        /** Defaults tuned for training-time pruning over labelled cells. */
        public static PruneOptions defaults() {
            return new PruneOptions(5, 0.95, 1.0, 5);
        }
    }

    /**
     * Result of a pruning pass.
     *
     * @param keptFeatures        surviving feature names (in original order)
     * @param originalCount       size of the input feature list
     * @param droppedConstant     features removed in the sparsity stage
     * @param droppedWithinMarker features removed by within-marker correlation
     * @param droppedCrossMarker  features removed by cross-marker correlation
     * @param forceKeptMarkers    marker groups where the top-N whitelist re-added a feature that the
     *                            correlation/sparsity stages had dropped
     * @param markersRetained     marker groups with ≥1 surviving feature
     * @param markersLost         marker groups that ended up with zero features (only when minKeptPerGroup == 0)
     * @param cellsScanned        number of cells contributing to stats
     */
    public record PruneResult(
            List<String> keptFeatures,
            int originalCount,
            int droppedConstant,
            int droppedWithinMarker,
            int droppedCrossMarker,
            List<String> forceKeptMarkers,
            List<String> markersRetained,
            List<String> markersLost,
            int cellsScanned) {}

    /**
     * Run the prune pipeline.
     *
     * @param detections   cells to base statistics on (usually the labelled set)
     * @param featureNames features to consider (in QuPath measurement-name form)
     * @param opts         configuration
     * @param progress     optional progress callback (may be {@code null})
     * @return pruning result; never {@code null}
     */
    public static PruneResult prune(
            Collection<PathObject> detections,
            List<String> featureNames,
            PruneOptions opts,
            Consumer<String> progress) {
        if (featureNames == null || featureNames.isEmpty()) {
            return new PruneResult(List.of(), 0, 0, 0, 0, List.of(), List.of(), List.of(), 0);
        }
        if (detections == null || detections.isEmpty()) {
            return new PruneResult(
                    List.copyOf(featureNames),
                    featureNames.size(),
                    0,
                    0,
                    0,
                    List.of(),
                    groupKeys(featureNames),
                    List.of(),
                    0);
        }

        // Materialise raw measurements into a dense [cell][feature] matrix and
        // delegate to the shared matrix-based pruner. Callers that already hold
        // pre-extracted (e.g. normalised, cross-image pooled) rows should call the
        // float[][] overload directly so pruning sees the same values the models
        // train on, rather than one image's raw measurements.
        final int nFeatures = featureNames.size();
        final PathObject[] cells = detections.toArray(new PathObject[0]);
        final int nCells = cells.length;
        final String[] names = featureNames.toArray(new String[0]);
        final float[][] data = new float[nCells][nFeatures];
        IntStream.range(0, nFeatures).parallel().forEach(j -> {
            String name = names[j];
            for (int i = 0; i < nCells; i++) {
                double v = cells[i].getMeasurementList().get(name);
                data[i][j] = Double.isNaN(v) ? 0f : (float) v;
            }
        });
        return pruneMatrix(data, featureNames, opts, progress);
    }

    /**
     * Prune from pre-extracted feature rows. Use this overload when pruning should
     * reflect the full training matrix the models will actually see — e.g. the
     * normalised, cross-image pooled rows assembled before training — rather than a
     * single image's raw measurements. Each row must be aligned to {@code featureNames}.
     *
     * @param rows         feature rows, each of length {@code featureNames.size()}
     * @param featureNames feature-column ordering
     * @param opts         pruning options
     * @param progress     optional progress sink (may be {@code null})
     * @return pruning result; never {@code null}
     */
    public static PruneResult prune(
            float[][] rows, List<String> featureNames, PruneOptions opts, Consumer<String> progress) {
        if (featureNames == null || featureNames.isEmpty()) {
            return new PruneResult(List.of(), 0, 0, 0, 0, List.of(), List.of(), List.of(), 0);
        }
        if (rows == null || rows.length == 0) {
            return new PruneResult(
                    List.copyOf(featureNames),
                    featureNames.size(),
                    0,
                    0,
                    0,
                    List.of(),
                    groupKeys(featureNames),
                    List.of(),
                    0);
        }
        final int nFeatures = featureNames.size();
        for (float[] row : rows) {
            if (row == null || row.length != nFeatures) {
                throw new IllegalArgumentException("Each prune row must have length " + nFeatures + " (got "
                        + (row == null ? "null" : row.length) + ")");
            }
        }
        return pruneMatrix(rows, featureNames, opts, progress);
    }

    /**
     * Shared core: prune a dense {@code [cell][feature]} matrix. Both public
     * {@code prune} overloads funnel here so the sparsity/variance and correlation
     * logic lives in one place.
     */
    private static PruneResult pruneMatrix(
            float[][] data, List<String> featureNames, PruneOptions opts, Consumer<String> progress) {
        final int nCells = data.length;
        final int nFeatures = featureNames.size();
        final String[] names = featureNames.toArray(new String[0]);

        report(progress, "Pruning " + nFeatures + " features against " + nCells + " labelled cells…");

        final String[] groupOf = new String[nFeatures];
        for (int j = 0; j < nFeatures; j++) {
            groupOf[j] = extractGroup(names[j]);
        }

        // ── Stage 0: stats + standardised columns (parallel by feature) ─────
        final double[] mean = new double[nFeatures];
        final double[] std = new double[nFeatures];
        final int[] nonZero = new int[nFeatures];
        final double[][] z = new double[nFeatures][];

        IntStream.range(0, nFeatures).parallel().forEach(j -> {
            double s = 0.0, ss = 0.0;
            int nz = 0;
            double[] raw = new double[nCells];
            for (int i = 0; i < nCells; i++) {
                double v = data[i][j];
                if (Double.isNaN(v)) v = 0.0;
                raw[i] = v;
                s += v;
                ss += v * v;
                if (v != 0.0) nz++;
            }
            double m = s / nCells;
            double varJ = Math.max(0.0, ss / nCells - m * m);
            double sd = Math.sqrt(varJ);
            mean[j] = m;
            std[j] = sd;
            nonZero[j] = nz;
            if (sd > 0.0) {
                double inv = 1.0 / sd;
                double[] zj = new double[nCells];
                for (int i = 0; i < nCells; i++) zj[i] = (raw[i] - m) * inv;
                z[j] = zj;
            }
        });

        // Why each feature was dropped (1 = near-constant, 2 = within-marker,
        // 3 = cross-marker), so net per-stage counts can be recomputed after the
        // Stage-4 whitelist re-adds protected features.
        final int[] dropReason = new int[nFeatures];

        // ── Stage 1: sparsity / constant filter ─────────────────────────────
        final boolean[] kept = new boolean[nFeatures];
        int droppedConstant = 0;
        for (int j = 0; j < nFeatures; j++) {
            if (nonZero[j] >= opts.minNonZeroCells() && std[j] > 0.0) {
                kept[j] = true;
            } else {
                dropReason[j] = 1;
                droppedConstant++;
            }
        }
        report(progress, "Sparsity: dropped " + droppedConstant + " near-constant features");

        // ── Stage 2: within-marker correlation (parallel by group) ──────────
        Map<String, List<Integer>> groupIndices = new LinkedHashMap<>();
        for (int j = 0; j < nFeatures; j++) {
            if (kept[j]) {
                groupIndices.computeIfAbsent(groupOf[j], k -> new ArrayList<>()).add(j);
            }
        }

        AtomicInteger droppedWithinAtom = new AtomicInteger(0);
        groupIndices.entrySet().parallelStream().forEach(entry -> {
            List<Integer> idxs = entry.getValue();
            if (idxs.size() < 2) return;
            idxs.sort((a, b) -> Double.compare(std[b], std[a]));
            List<Integer> keepSet = new ArrayList<>();
            for (int idx : idxs) {
                boolean redundant = false;
                for (int keptIdx : keepSet) {
                    double r = Math.abs(pearson(z[idx], z[keptIdx]));
                    if (r > opts.withinMarkerCorrThreshold()) {
                        redundant = true;
                        break;
                    }
                }
                if (redundant) {
                    kept[idx] = false; // disjoint indices across groups — safe
                    dropReason[idx] = 2;
                    droppedWithinAtom.incrementAndGet();
                } else {
                    keepSet.add(idx);
                }
            }
        });
        int droppedWithin = droppedWithinAtom.get();
        report(progress, "Within-marker: dropped " + droppedWithin + " redundant features");

        // ── Stage 3: cross-marker correlation (optional, sequential greedy) ─
        int droppedCross = 0;
        if (opts.crossMarkerCorrThreshold() < 1.0) {
            List<Integer> survivors = new ArrayList<>();
            for (int j = 0; j < nFeatures; j++) if (kept[j]) survivors.add(j);
            survivors.sort((a, b) -> Double.compare(std[b], std[a]));
            List<Integer> keepSet = new ArrayList<>();
            final double thr = opts.crossMarkerCorrThreshold();
            for (int idx : survivors) {
                final int idxF = idx;
                boolean redundant;
                if (keepSet.size() < 32) {
                    redundant = false;
                    for (int keptIdx : keepSet) {
                        if (groupOf[idxF].equals(groupOf[keptIdx])) continue;
                        if (Math.abs(pearson(z[idxF], z[keptIdx])) > thr) {
                            redundant = true;
                            break;
                        }
                    }
                } else {
                    redundant = keepSet.parallelStream()
                            .anyMatch(keptIdx -> !groupOf[idxF].equals(groupOf[keptIdx])
                                    && Math.abs(pearson(z[idxF], z[keptIdx])) > thr);
                }
                if (redundant) {
                    kept[idxF] = false;
                    dropReason[idxF] = 3;
                    droppedCross++;
                } else {
                    keepSet.add(idxF);
                }
            }
            report(progress, "Cross-marker: dropped " + droppedCross + " correlated features");
        }

        // ── Stage 4: per-group whitelist — always keep the top-N highest-variance
        // features in every marker group, immune to the sparsity/correlation stages.
        // N = opts.minKeptPerGroup(); a group with ≤N features keeps them all. This
        // subsumes the old "keep ≥1 per group" guardrail (N == 1 reproduces it).
        List<String> forceKept = new ArrayList<>();
        Map<String, List<Integer>> allGroupIndices = new LinkedHashMap<>();
        for (int j = 0; j < nFeatures; j++) {
            allGroupIndices.computeIfAbsent(groupOf[j], k -> new ArrayList<>()).add(j);
        }
        final int minKeep = Math.max(0, opts.minKeptPerGroup());
        if (minKeep > 0) {
            for (var entry : allGroupIndices.entrySet()) {
                List<Integer> idxs = new ArrayList<>(entry.getValue());
                idxs.sort((a, b) -> Double.compare(std[b], std[a]));
                int keepN = Math.min(minKeep, idxs.size());
                boolean reAdded = false;
                for (int t = 0; t < keepN; t++) {
                    int j = idxs.get(t);
                    if (!kept[j]) {
                        kept[j] = true;
                        reAdded = true;
                    }
                }
                if (reAdded) forceKept.add(entry.getKey());
            }
        }

        List<String> keptNames = new ArrayList<>();
        for (int j = 0; j < nFeatures; j++) {
            if (kept[j]) keptNames.add(names[j]);
        }

        // Net per-stage drop counts after the whitelist, so that
        // (originalCount − netConstant − netWithin − netCross) == keptFeatures.size()
        // and the reported breakdown matches the actual kept set.
        int netConstant = 0;
        int netWithin = 0;
        int netCross = 0;
        for (int j = 0; j < nFeatures; j++) {
            if (kept[j]) continue;
            switch (dropReason[j]) {
                case 1 -> netConstant++;
                case 2 -> netWithin++;
                case 3 -> netCross++;
                default -> {}
            }
        }

        Set<String> retainedSet = new LinkedHashSet<>();
        for (int j = 0; j < nFeatures; j++) {
            if (kept[j]) retainedSet.add(groupOf[j]);
        }
        List<String> markersRetained = new ArrayList<>(retainedSet);

        // Groups with zero survivors — only possible when the whitelist is disabled.
        List<String> markersLost = new ArrayList<>();
        for (var entry : allGroupIndices.entrySet()) {
            if (entry.getValue().stream().noneMatch(j -> kept[j])) {
                markersLost.add(entry.getKey());
            }
        }

        logger.info(
                "Pruned {} -> {} features ({} marker groups retained, {} force-kept, {} lost; {} cells)",
                nFeatures,
                keptNames.size(),
                markersRetained.size(),
                forceKept.size(),
                markersLost.size(),
                nCells);

        return new PruneResult(
                keptNames,
                nFeatures,
                netConstant,
                netWithin,
                netCross,
                forceKept,
                markersRetained,
                markersLost,
                nCells);
    }

    /**
     * Extract the marker-group key for a feature, so within-group redundant features
     * can be deduplicated and a guardrail can keep at least one per group.
     *
     * <p>Separators are tried in priority order:
     *
     * <ol>
     *   <li>{@code ": "} — the marker convention ({@code "CD3: Cell: Mean" -> "CD3"}).
     *       Checked first so it always wins, even over an earlier space
     *       ({@code "Some Marker: Mean" -> "Some Marker"}).
     *   <li>otherwise, the first {@code '_'} or {@code ' '} — whichever appears
     *       earlier ({@code "kronos_emb_0" -> "kronos"}, {@code "Distance to tumor" -> "Distance"}).
     * </ol>
     *
     * <p>A feature with no recognised separator (or one only at position 0) forms its
     * own singleton group keyed by the full feature name. Group keys are
     * <b>case-insensitive</b> (lower-cased), so {@code "CD3: Mean"} and {@code "cd3: Max"}
     * share a group.
     */
    public static String extractGroup(String featureName) {
        String prefix;
        // Marker-style "PREFIX: ..." takes priority regardless of other separators.
        int colon = featureName.indexOf(": ");
        if (colon > 0) {
            prefix = featureName.substring(0, colon);
        } else {
            // Otherwise group by the token before the first underscore or space.
            int cut = firstSeparator(featureName.indexOf('_'), featureName.indexOf(' '));
            prefix = cut > 0 ? featureName.substring(0, cut) : featureName;
        }
        return prefix.toLowerCase(java.util.Locale.ROOT);
    }

    /** Smaller of two {@code indexOf} results, ignoring {@code -1} (not found). */
    private static int firstSeparator(int a, int b) {
        if (a < 0) return b;
        if (b < 0) return a;
        return Math.min(a, b);
    }

    private static List<String> groupKeys(List<String> featureNames) {
        Set<String> groups = new LinkedHashSet<>();
        for (String name : featureNames) groups.add(extractGroup(name));
        return new ArrayList<>(groups);
    }

    /**
     * Pearson correlation between two pre-standardised vectors of equal length:
     * r = (1/N) * sum(a_i * b_i).
     */
    private static double pearson(double[] a, double[] b) {
        double sum = 0.0;
        int n = a.length;
        for (int i = 0; i < n; i++) sum += a[i] * b[i];
        return sum / n;
    }

    private static void report(Consumer<String> progress, String msg) {
        if (progress != null) {
            try {
                progress.accept(msg);
            } catch (Exception ignored) {
            }
        }
    }
}
