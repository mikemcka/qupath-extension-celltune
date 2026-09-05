package qupath.ext.spclassify.classifier;

import java.util.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.model.CellPrediction;
import qupath.ext.spclassify.model.PopulationSet;

/**
 * Multi-tier cell sampling for human review, matching the Python CellTune approach.
 * <p>
 * The sample is composed of five tiers, collected in priority order:
 * <ol>
 *   <li><b>FOV balance</b> — disagreement cells grouped by spatial region (FOV),
 *       prioritising regions with the highest disagreement fraction.
 *       Prevents one tissue area from dominating the review queue.</li>
 *   <li><b>Cell-type disagreement</b> — disagreement cells grouped by predicted
 *       class, prioritising classes with the highest disagreement fraction.
 *       {@code cellsPerType} cells are sampled per class.</li>
 *   <li><b>Rare cell types</b> — disagreement cells grouped by class, sorted by
 *       total population (rarest first). Ensures under-represented types get
 *       reviewed even when their disagreement rate is low.</li>
 *   <li><b>Preferred confusions</b> — user-specified confusion pairs
 *       ({@code "TypeA:TypeB"}) and/or preferred cell types that should be
 *       over-sampled for targeted review.</li>
 *   <li><b>Random fill</b> — remaining disagreement cells, shuffled randomly,
 *       to fill any unused disagreement budget.</li>
 * </ol>
 * Each tier marks sampled cells as used so later tiers cannot re-select them.
 * The final result is capped at {@code sampleSize}.
 */
public class UncertaintySampler {

    private static final Logger logger = LoggerFactory.getLogger(UncertaintySampler.class);

    // Base per-tier budgets (calibrated for sampleSize = 256, matching Python CellTune)
    private static final int BASE_FOV_BUDGET = 84; // FOV-balanced tier
    private static final int BASE_TYPE_BUDGET = 112; // cell-type disagreement tier
    private static final int BASE_RARE_BUDGET = 60; // rare cell-type tier
    private static final int BASE_PREF_BUDGET = 40; // preferred confusions tier
    private static final int BASE_CELLS_PER_FOV = 14;
    private static final int BASE_CELLS_PER_TYPE = 16;
    private static final int BASE_CELLS_PER_RARE = 10;
    private static final int BASE_CELLS_PER_PREF = 8;

    private UncertaintySampler() {} // utility class

    /**
     * Sample cells for review using the multi-tier strategy.
     *
     * @param predALL              all dual-model predictions
     * @param classNames           ordered class name list
     * @param agreementRates       per-class agreement rates (unused by tiered logic,
     *                             kept for API compatibility)
     * @param sampleSize           total number of cells to sample
     * @param preferredConfusions  confusion pairs to boost, e.g. {@code ["Epithelial:Goblet"]}
     * @param preferredTypes       cell types to boost when they appear in disagreements
     * @param fovMap               optional cellId → FOV/region label map; if non-null
     *                             enables FOV-balanced sampling (tier 0)
     * @param rng                  random number generator
     * @return list of sampled cell IDs in tier order
     */
    public static List<String> sample(
            PopulationSet predALL,
            List<String> classNames,
            double[] agreementRates,
            int sampleSize,
            List<String> preferredConfusions,
            List<String> preferredTypes,
            Map<String, String> fovMap,
            Random rng) {
        return sample(
                predALL,
                classNames,
                agreementRates,
                sampleSize,
                preferredConfusions,
                preferredTypes,
                fovMap,
                Set.of(),
                rng);
    }

    /**
     * Sample cells for review using the multi-tier strategy.
     *
     * @param reviewedCellIds already-reviewed cell IDs to exclude from sampling
     */
    public static List<String> sample(
            PopulationSet predALL,
            List<String> classNames,
            double[] agreementRates,
            int sampleSize,
            List<String> preferredConfusions,
            List<String> preferredTypes,
            Map<String, String> fovMap,
            Set<String> reviewedCellIds,
            Random rng) {
        if (preferredConfusions == null) preferredConfusions = List.of();
        if (preferredTypes == null) preferredTypes = List.of();
        if (reviewedCellIds == null) reviewedCellIds = Set.of();

        Map<String, CellPrediction> predMap = predALL.getAll();

        // ── 1. Partition cells and build per-class index ────────────────────
        List<String> disagreementIds = new ArrayList<>();

        // For each class: list of disagreement cell IDs involving that class
        Map<String, List<String>> disagreeByClass = new LinkedHashMap<>();
        // Per-class total count (from both model predictions)
        Map<String, Long> totalPerClass = new LinkedHashMap<>();
        // Per-class disagreement count
        Map<String, Long> disagreePerClass = new LinkedHashMap<>();

        for (String cn : classNames) {
            disagreeByClass.put(cn, new ArrayList<>());
            totalPerClass.put(cn, 0L);
            disagreePerClass.put(cn, 0L);
        }

        for (var entry : predMap.entrySet()) {
            String cellId = entry.getKey();
            if (reviewedCellIds.contains(cellId)) {
                continue;
            }
            CellPrediction pred = entry.getValue();
            String l1 = pred.getModel1Label(), l2 = pred.getModel2Label();

            // Count total per class (each model's prediction counts once)
            totalPerClass.merge(l1, 1L, Long::sum);
            if (!l1.equals(l2)) {
                totalPerClass.merge(l2, 1L, Long::sum);
            }

            if (pred.isDisagreement()) {
                disagreementIds.add(cellId);
                // Register cell in both classes' pools (matching Python's duplicated approach)
                disagreeByClass.computeIfAbsent(l1, k -> new ArrayList<>()).add(cellId);
                disagreeByClass.computeIfAbsent(l2, k -> new ArrayList<>()).add(cellId);
                disagreePerClass.merge(l1, 1L, Long::sum);
                disagreePerClass.merge(l2, 1L, Long::sum);
            }
        }

        if (disagreementIds.isEmpty()) {
            logger.info("No cells available for sampling.");
            return List.of();
        }

        // ── 2. Per-tier budgets, scaled linearly from the 256-cell defaults ──
        // The BASE_* constants are calibrated for sampleSize = 256 (matching
        // Python balanceSelectCells). Scaling by sampleSize/256 keeps the tier
        // proportions intact for smaller (or larger) review batches, so lower
        // tiers still get representation instead of being truncated away.
        double scale = sampleSize / 256.0;
        int fovBudget = (fovMap == null || fovMap.isEmpty()) ? 0 : scaleBudget(BASE_FOV_BUDGET, scale);
        int cellsPerFov = scaleBudget(BASE_CELLS_PER_FOV, scale);
        int typeBudget = scaleBudget(BASE_TYPE_BUDGET, scale);
        int rareBudget = scaleBudget(BASE_RARE_BUDGET, scale);
        int prefBudget =
                (preferredConfusions.isEmpty() && preferredTypes.isEmpty()) ? 0 : scaleBudget(BASE_PREF_BUDGET, scale);
        int cellsPerType = scaleBudget(BASE_CELLS_PER_TYPE, scale);
        int cellsPerRare = scaleBudget(BASE_CELLS_PER_RARE, scale);
        int cellsPerPref = scaleBudget(BASE_CELLS_PER_PREF, scale);

        Set<String> used = new LinkedHashSet<>();
        List<String> result = new ArrayList<>();

        // ── 3. Tier 0: FOV balance (highest disagreement fraction per region first)
        int tier0 = 0;
        if (fovBudget > 0) {
            // Group disagreement cells by FOV and compute per-FOV disagree fraction
            Map<String, List<String>> disagreeByFov = new LinkedHashMap<>();
            Map<String, Long> totalByFov = new LinkedHashMap<>();
            Map<String, Long> disagreeCountByFov = new LinkedHashMap<>();

            for (var entry : predMap.entrySet()) {
                String cellId = entry.getKey();
                String fov = fovMap.get(cellId);
                if (fov == null) continue;
                totalByFov.merge(fov, 1L, Long::sum);
                if (entry.getValue().isDisagreement()) {
                    disagreeByFov.computeIfAbsent(fov, k -> new ArrayList<>()).add(cellId);
                    disagreeCountByFov.merge(fov, 1L, Long::sum);
                }
            }

            // Sort FOVs by disagreement fraction (highest first)
            List<String> fovsByFraction = new ArrayList<>(disagreeByFov.keySet());
            fovsByFraction.sort((a, b) -> Double.compare(
                    classFraction(disagreeCountByFov, totalByFov, b),
                    classFraction(disagreeCountByFov, totalByFov, a)));

            tier0 = sampleTier(fovsByFraction, disagreeByFov, used, result, cellsPerFov, fovBudget, rng);
        }

        // ── 4. Tier 1: Cell-type disagreement (highest disagreement fraction first)
        List<String> byDisagreeFraction = new ArrayList<>(classNames);
        byDisagreeFraction.sort((a, b) -> Double.compare(
                classFraction(disagreePerClass, totalPerClass, b), classFraction(disagreePerClass, totalPerClass, a)));

        int tier1 = sampleTier(byDisagreeFraction, disagreeByClass, used, result, cellsPerType, typeBudget, rng);

        // ── 5. Tier 2: Rare cell types (rarest first by total count) ────────
        List<String> byRarity = new ArrayList<>(classNames);
        byRarity.sort((a, b) -> Long.compare(totalPerClass.getOrDefault(a, 0L), totalPerClass.getOrDefault(b, 0L)));

        int tier2 = sampleTier(byRarity, disagreeByClass, used, result, cellsPerRare, rareBudget, rng);

        // ── 6. Tier 3: Preferred confusions and types ───────────────────────
        int tier3 = 0;
        if (prefBudget > 0) {
            Map<String, List<String>> prefPools =
                    buildPreferredPools(preferredConfusions, preferredTypes, disagreementIds, predMap, used);

            for (var pool : prefPools.values()) {
                if (tier3 >= prefBudget) break;
                pool.removeIf(used::contains);
                Collections.shuffle(pool, rng);
                int take = Math.min(cellsPerPref, Math.min(pool.size(), prefBudget - tier3));
                for (int i = 0; i < take; i++) {
                    result.add(pool.get(i));
                    used.add(pool.get(i));
                    tier3++;
                }
            }
        }

        // ── 7. Tier 4: Random fill from remaining disagreements ─────────────
        List<String> remaining = new ArrayList<>(disagreementIds);
        remaining.removeIf(used::contains);
        Collections.shuffle(remaining, rng);
        int fillCount = Math.min(Math.max(0, sampleSize - result.size()), remaining.size());
        for (int i = 0; i < fillCount; i++) {
            result.add(remaining.get(i));
        }

        // Cap at requested sample size
        if (result.size() > sampleSize) {
            result = new ArrayList<>(result.subList(0, sampleSize));
        }

        logger.info(
                "Sampled {} cells: fov={}, type={}, rare={}, preferred={}, random={}",
                result.size(),
                tier0,
                tier1,
                tier2,
                tier3,
                fillCount);
        return result;
    }

    /**
     * Overload without FOV map — no FOV-balanced sampling.
     */
    public static List<String> sample(
            PopulationSet predALL,
            List<String> classNames,
            double[] agreementRates,
            int sampleSize,
            List<String> preferredConfusions,
            List<String> preferredTypes,
            Random rng) {
        return sample(
                predALL,
                classNames,
                agreementRates,
                sampleSize,
                preferredConfusions,
                preferredTypes,
                null,
                Set.of(),
                rng);
    }

    /**
     * Convenience overload — no preferred confusions/types, no FOV map, default RNG.
     */
    public static List<String> sample(
            PopulationSet predALL, List<String> classNames, double[] agreementRates, int sampleSize) {
        return sample(
                predALL, classNames, agreementRates, sampleSize, List.of(), List.of(), null, Set.of(), new Random());
    }

    // ── Private helpers ─────────────────────────────────────────────────────────

    /**
     * Scale a 256-calibrated tier budget by the requested sample size.
     * Floored at 1 so every tier still contributes at least one cell when its
     * base budget is non-zero, even for very small review batches.
     */
    private static int scaleBudget(int base256, double scale) {
        return Math.max(1, (int) Math.round(base256 * scale));
    }

    /**
     * Sample from class-keyed pools in the given class order, respecting
     * a per-group limit and overall tier budget. Marks sampled cells as used.
     *
     * @return number of cells actually sampled by this tier
     */
    private static int sampleTier(
            List<String> sortedClasses,
            Map<String, List<String>> poolByClass,
            Set<String> used,
            List<String> result,
            int perGroup,
            int maxBudget,
            Random rng) {
        int count = 0;
        for (String cn : sortedClasses) {
            if (count >= maxBudget) break;
            List<String> pool = new ArrayList<>(poolByClass.getOrDefault(cn, List.of()));
            pool.removeIf(used::contains);
            Collections.shuffle(pool, rng);
            int take = Math.min(perGroup, Math.min(pool.size(), maxBudget - count));
            for (int i = 0; i < take; i++) {
                result.add(pool.get(i));
                used.add(pool.get(i));
                count++;
            }
        }
        return count;
    }

    /**
     * Build per-group pools for preferred confusion pairs and types.
     * A confusion {@code "A:B"} matches cells where (pred1=A, pred2=B) or vice versa.
     * A preferred type {@code T} matches cells where pred1=T or pred2=T.
     */
    private static Map<String, List<String>> buildPreferredPools(
            List<String> confusions,
            List<String> types,
            List<String> disagreementIds,
            Map<String, CellPrediction> predMap,
            Set<String> used) {
        Map<String, List<String>> pools = new LinkedHashMap<>();

        for (String conf : confusions) {
            String[] parts = conf.split(":");
            if (parts.length != 2) continue;
            String a = parts[0].strip(), b = parts[1].strip();
            List<String> pool = new ArrayList<>();
            for (String id : disagreementIds) {
                if (used.contains(id)) continue;
                CellPrediction p = predMap.get(id);
                if ((a.equals(p.getModel1Label()) && b.equals(p.getModel2Label()))
                        || (b.equals(p.getModel1Label()) && a.equals(p.getModel2Label()))) {
                    pool.add(id);
                }
            }
            pools.put(conf, pool);
        }

        for (String type : types) {
            List<String> pool = new ArrayList<>();
            for (String id : disagreementIds) {
                if (used.contains(id)) continue;
                CellPrediction p = predMap.get(id);
                if (type.equals(p.getModel1Label()) || type.equals(p.getModel2Label())) {
                    pool.add(id);
                }
            }
            pools.put(type, pool);
        }

        return pools;
    }

    /** Safe fraction: disagreement count / total count for a class. */
    private static double classFraction(Map<String, Long> numerator, Map<String, Long> denominator, String key) {
        long num = numerator.getOrDefault(key, 0L);
        long den = denominator.getOrDefault(key, 1L);
        return den == 0 ? 0 : (double) num / den;
    }
}
