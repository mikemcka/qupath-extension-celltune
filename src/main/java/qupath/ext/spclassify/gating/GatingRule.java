package qupath.ext.spclassify.gating;

import java.util.*;

/**
 * Encodes a single cell type's gating rules: a primary marker expression
 * plus optional secondary/tertiary markers, converted into a numeric
 * marker table matching Python CellTune's encoding:
 * <ul>
 *   <li>{@code  1} — must-have (AND operand in primary expression)</li>
 *   <li>{@code  2} — or-expression (OR operand in primary expression)</li>
 *   <li>{@code  0} — secondary/tertiary (informative, not gating)</li>
 *   <li>{@code -1} — soft NOT (not mentioned, may be promoted to strict)</li>
 *   <li>{@code -3} — strict NOT (explicitly negated with {@code !})</li>
 * </ul>
 */
public class GatingRule {

    private final String cellType;
    private final String primaryExpression;
    private final String secondaryMarkers; // pipe-separated
    private final String tertiaryMarkers; // pipe-separated

    /** Parsed marker encoding: marker name → numeric code. */
    private final Map<String, Integer> markerEncoding;

    /**
     * @param cellType           cell type name
     * @param primaryExpression  boolean expression (e.g. "CD4&CD3&!IgA")
     * @param secondaryMarkers   pipe-separated marker list (may be null)
     * @param tertiaryMarkers    pipe-separated marker list (may be null)
     * @param allChannels        full ordered list of available channels
     */
    public GatingRule(
            String cellType,
            String primaryExpression,
            String secondaryMarkers,
            String tertiaryMarkers,
            List<String> allChannels) {
        this.cellType = cellType;
        this.primaryExpression = primaryExpression;
        this.secondaryMarkers = secondaryMarkers;
        this.tertiaryMarkers = tertiaryMarkers;
        this.markerEncoding = buildEncoding(allChannels);
    }

    // ── Public API ──────────────────────────────────────────────────────────

    public String getCellType() {
        return cellType;
    }

    public String getPrimaryExpression() {
        return primaryExpression;
    }

    /**
     * @return the numeric encoding for each channel. Keys are channel names,
     *         values are 1, 2, 0, -1, or -3.
     */
    public Map<String, Integer> getMarkerEncoding() {
        return Collections.unmodifiableMap(markerEncoding);
    }

    /**
     * Get the encoding value for a specific channel.
     *
     * @param channel channel name
     * @return encoding value (defaults to -1 if channel not present)
     */
    public int getEncoding(String channel) {
        return markerEncoding.getOrDefault(channel, -1);
    }

    /**
     * Get channels with a specific encoding.
     */
    public List<String> getChannelsWithEncoding(int code) {
        List<String> result = new ArrayList<>();
        for (var entry : markerEncoding.entrySet()) {
            if (entry.getValue() == code) result.add(entry.getKey());
        }
        return result;
    }

    /**
     * Promote soft NOTs (-1) to strict NOTs (-3) for channels that overlap
     * with another rule's must-have or or-expression markers.
     * <p>
     * This matches Python CellTune's {@code convert_not_to_strict_not()} logic:
     * when two cell types share primary markers but differ by which markers
     * they negate, the soft NOT is promoted to strict to prevent overlap.
     *
     * @param otherRule the other rule to compare against
     */
    public void promoteOverlappingSoftNots(GatingRule otherRule) {
        // Find our primary markers (encoding 1 or 2)
        Set<String> ourPrimary = new HashSet<>();
        for (var e : markerEncoding.entrySet()) {
            if (e.getValue() == 1 || e.getValue() == 2) ourPrimary.add(e.getKey());
        }

        // Find other rule's primary markers
        Set<String> theirPrimary = new HashSet<>();
        for (var e : otherRule.markerEncoding.entrySet()) {
            if (e.getValue() == 1 || e.getValue() == 2) theirPrimary.add(e.getKey());
        }

        // Check if there's overlap in primary markers
        boolean sharesPrimary = false;
        for (String m : ourPrimary) {
            if (theirPrimary.contains(m)) {
                sharesPrimary = true;
                break;
            }
        }
        if (!sharesPrimary) return;

        // For markers that are must-have(1) in the other rule but soft-NOT(-1) in ours,
        // promote to strict-NOT(-3)
        for (var e : otherRule.markerEncoding.entrySet()) {
            if (e.getValue() == 1) {
                Integer ourCode = markerEncoding.get(e.getKey());
                if (ourCode != null && ourCode == -1) {
                    markerEncoding.put(e.getKey(), -3);
                }
            }
        }

        // For markers that are or-expression(2) in the other rule but soft-NOT(-1) in ours,
        // promote to strict-NOT(-3) if the other rule has no missing must-have markers
        List<String> theirMustHave = otherRule.getChannelsWithEncoding(1);
        boolean allMustHavePositiveInUs = true;
        for (String m : theirMustHave) {
            Integer ourCode = markerEncoding.get(m);
            if (ourCode == null || ourCode < 0) {
                allMustHavePositiveInUs = false;
                break;
            }
        }
        if (allMustHavePositiveInUs) {
            for (var e : otherRule.markerEncoding.entrySet()) {
                if (e.getValue() == 2) {
                    Integer ourCode = markerEncoding.get(e.getKey());
                    if (ourCode != null && ourCode == -1) {
                        markerEncoding.put(e.getKey(), -3);
                    }
                }
            }
        }
    }

    // ── Internals ───────────────────────────────────────────────────────────

    private Map<String, Integer> buildEncoding(List<String> allChannels) {
        Map<String, Integer> encoding = new LinkedHashMap<>();

        // Start with all channels as soft NOT (-1)
        for (String ch : allChannels) {
            encoding.put(ch, -1);
        }

        // Parse primary expression
        if (primaryExpression != null && !primaryExpression.isBlank()) {
            var expr = new GatingExpression(primaryExpression);
            var cats = expr.categorize();
            for (String m : cats.mustHave()) {
                if (encoding.containsKey(m)) encoding.put(m, 1);
            }
            for (String m : cats.orExpression()) {
                if (encoding.containsKey(m)) encoding.put(m, 2);
            }
            for (String m : cats.not()) {
                if (encoding.containsKey(m)) encoding.put(m, -3);
            }
        }

        // Process secondary markers (pipe-separated) → encoding 0
        applySecondary(encoding, secondaryMarkers);
        // Tertiary markers merged with secondary (same as Python CellTune)
        applySecondary(encoding, tertiaryMarkers);

        return encoding;
    }

    private void applySecondary(Map<String, Integer> encoding, String markers) {
        if (markers == null || markers.isBlank()) return;
        for (String m : markers.split("\\|")) {
            m = m.strip();
            if (!m.isEmpty() && encoding.containsKey(m) && encoding.get(m) == -1) {
                encoding.put(m, 0);
            }
        }
    }

    @Override
    public String toString() {
        return "GatingRule[" + cellType + ": " + primaryExpression + "]";
    }
}
