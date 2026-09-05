package qupath.ext.spclassify.model;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Maps a cell-type name to marker channel names and optional gating rules.
 * <p>
 * Supports two CSV formats, auto-detected from the header:
 * <p>
 * <b>Simple format</b> (channel display only):
 * <pre>
 * CellType,Marker1,Marker2,Marker3
 * T-Cell,CD3,,
 * Macrophage,CD68,CD163,
 * </pre>
 *
 * <b>Rule format</b> (gating rules — matching Python CellTune):
 * <pre>
 * CellType,PrimaryMarker,SecondaryMarker,TertiaryMarker
 * CD8T,CD8,CD3,CD103|CD45|CD45RA
 * Plasma_CD38,CD38&amp;!IgA,,CD45|VIM
 * Macrophage,CD68|CD163|CD206,,CD14|CD38|VIM
 * </pre>
 * Additional columns (hex, Channel_1/2/3) are accepted and ignored.
 * <p>
 * When gating rules are present ({@link #hasGatingRules()} returns true),
 * marker channels for display are derived from the primary expression's
 * must-have and or-expression markers.
 */
public class CellTypeTable {

    /** Maximum number of marker channels per cell type (for display). */
    public static final int MAX_MARKERS = 5;

    private static final Logger logger = LoggerFactory.getLogger(CellTypeTable.class);

    private final Map<String, List<String>> table; // cellType → display markers
    private final Map<String, String> primaryMarkers; // cellType → primary expression
    private final Map<String, String> secondaryMarkers; // cellType → pipe-separated
    private final Map<String, String> tertiaryMarkers; // cellType → pipe-separated
    // normalized(cellType) → actual key, for tolerant lookup (see getMarkers). The
    // predicted class name and the CSV CellType are typed independently, so they can
    // differ in case/spacing/punctuation; this lets auto channel-switch still resolve.
    private final Map<String, String> normalizedIndex;
    private boolean hasRules = false;

    public CellTypeTable() {
        this.table = new LinkedHashMap<>();
        this.primaryMarkers = new LinkedHashMap<>();
        this.secondaryMarkers = new LinkedHashMap<>();
        this.tertiaryMarkers = new LinkedHashMap<>();
        this.normalizedIndex = new LinkedHashMap<>();
    }

    /**
     * Normalize a cell-type name to lowercase alphanumeric characters only, so lookups
     * tolerate case, spacing, and punctuation differences between the classifier's class
     * names and the CSV {@code CellType} column. Mirrors the channel-name normalization in
     * {@code ChannelSelector} so both sides of auto channel-switching match the same way.
     */
    static String normalizeType(String s) {
        if (s == null) return "";
        return s.toLowerCase().replaceAll("[^a-z0-9]", "");
    }

    /** Register {@code cellType} in the normalized index, warning on a normalized collision. */
    private void indexKey(String cellType) {
        String norm = normalizeType(cellType);
        if (norm.isEmpty()) return;
        String existing = normalizedIndex.putIfAbsent(norm, cellType);
        if (existing != null && !existing.equals(cellType)) {
            logger.warn(
                    "Cell types '{}' and '{}' normalize to the same key; auto channel-switch "
                            + "lookups will resolve to '{}'.",
                    cellType,
                    existing,
                    existing);
        }
    }

    // ── Simple format API (display markers) ─────────────────────────────────

    /**
     * Define a cell type with its display marker channels.
     */
    public void put(String cellType, List<String> markers) {
        List<String> trimmed = new ArrayList<>();
        for (int i = 0; i < Math.min(markers.size(), MAX_MARKERS); i++) {
            String m = markers.get(i);
            if (m != null && !m.isBlank()) {
                trimmed.add(m.strip());
            }
        }
        table.put(cellType, Collections.unmodifiableList(trimmed));
        indexKey(cellType);
    }

    /**
     * @return marker channels for the given cell type, or an empty list.
     *     <p>Tries an exact key match first; if none, falls back to a normalized
     *     (case/spacing/punctuation-insensitive) match so a predicted class name that
     *     differs only cosmetically from the CSV {@code CellType} still resolves.
     */
    public List<String> getMarkers(String cellType) {
        if (cellType == null) return Collections.emptyList();
        List<String> exact = table.get(cellType);
        if (exact != null) return exact;
        String actual = normalizedIndex.get(normalizeType(cellType));
        return actual == null ? Collections.emptyList() : table.getOrDefault(actual, Collections.emptyList());
    }

    /** @return all cell type names in insertion order */
    public Set<String> getCellTypes() {
        return Collections.unmodifiableSet(table.keySet());
    }

    /** @return number of defined cell types */
    public int size() {
        return table.size();
    }

    /** @return true if the table contains no entries */
    public boolean isEmpty() {
        return table.isEmpty();
    }

    // ── Gating rule API ─────────────────────────────────────────────────────

    /**
     * Define a cell type with gating rules (rule format).
     */
    public void putRule(String cellType, String primaryExpr, String secondary, String tertiary) {
        this.hasRules = true;
        primaryMarkers.put(cellType, primaryExpr);
        secondaryMarkers.put(cellType, secondary);
        tertiaryMarkers.put(cellType, tertiary);

        // Derive display markers from primary expression
        List<String> displayMarkers = new ArrayList<>();
        if (primaryExpr != null && !primaryExpr.isBlank()) {
            // Extract simple marker names (strip operators and parens)
            String[] tokens = primaryExpr.split("[&|!()]+");
            for (String t : tokens) {
                t = t.strip();
                if (!t.isEmpty() && displayMarkers.size() < MAX_MARKERS) {
                    displayMarkers.add(t);
                }
            }
        }
        table.put(cellType, Collections.unmodifiableList(displayMarkers));
        indexKey(cellType);
    }

    /** @return true if this table was loaded from a rule-format CSV */
    public boolean hasGatingRules() {
        return hasRules;
    }

    /** @return primary marker expression for the given cell type, or null */
    public String getPrimaryExpression(String cellType) {
        return primaryMarkers.get(cellType);
    }

    /** @return secondary markers (pipe-separated) for the given cell type, or null */
    public String getSecondaryMarkers(String cellType) {
        return secondaryMarkers.get(cellType);
    }

    /** @return tertiary markers (pipe-separated) for the given cell type, or null */
    public String getTertiaryMarkers(String cellType) {
        return tertiaryMarkers.get(cellType);
    }

    /**
     * Collect all unique channel names referenced across all rules
     * (primary, secondary, tertiary expressions).
     *
     * @return sorted list of all referenced channel names
     */
    public List<String> getAllRuleChannels() {
        Set<String> channels = new TreeSet<>(String.CASE_INSENSITIVE_ORDER);
        for (String cellType : table.keySet()) {
            addChannelsFrom(primaryMarkers.get(cellType), channels);
            addChannelsFrom(secondaryMarkers.get(cellType), channels);
            addChannelsFrom(tertiaryMarkers.get(cellType), channels);
        }
        return new ArrayList<>(channels);
    }

    private void addChannelsFrom(String expr, Set<String> channels) {
        if (expr == null || expr.isBlank()) return;
        for (String token : expr.split("[&|!()]+")) {
            token = token.strip();
            if (!token.isEmpty()) channels.add(token);
        }
    }

    // ── CSV I/O ─────────────────────────────────────────────────────────────

    /**
     * Load a CellTypeTable from a CSV file.
     * Auto-detects simple vs rule format from the header.
     */
    public static CellTypeTable loadFromCSV(Path path) throws IOException {
        CellTypeTable tbl = new CellTypeTable();
        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String header = reader.readLine();
            if (header == null) return tbl;

            // Detect format from header
            String headerLower = header.toLowerCase();
            boolean isRuleFormat = headerLower.contains("primarymarker");

            if (isRuleFormat) {
                return loadRuleFormat(tbl, header, reader);
            } else {
                return loadSimpleFormat(tbl, reader);
            }
        }
    }

    private static CellTypeTable loadSimpleFormat(CellTypeTable tbl, BufferedReader reader) throws IOException {
        String line;
        while ((line = reader.readLine()) != null) {
            line = line.strip();
            if (line.isEmpty()) continue;

            String[] parts = line.split(",", -1);
            if (parts.length < 1) continue;

            String cellType = parts[0].strip();
            if (cellType.isEmpty()) continue;

            List<String> markers = new ArrayList<>();
            for (int i = 1; i < parts.length && i <= MAX_MARKERS; i++) {
                markers.add(parts[i].strip());
            }
            tbl.put(cellType, markers);
        }
        return tbl;
    }

    private static CellTypeTable loadRuleFormat(CellTypeTable tbl, String header, BufferedReader reader)
            throws IOException {
        // Parse header to find column indices
        String[] cols = header.split(",", -1);
        int classCol = -1, primaryCol = -1, secondaryCol = -1, tertiaryCol = -1;
        for (int i = 0; i < cols.length; i++) {
            String col = cols[i].strip().toLowerCase();
            if (col.equals("celltype") || col.equals("class")) classCol = i;
            else if (col.equals("primarymarker")) primaryCol = i;
            else if (col.equals("secondarymarker")) secondaryCol = i;
            else if (col.equals("tertiarymarker")) tertiaryCol = i;
        }

        if (classCol < 0) {
            throw new IOException("Rule-format CSV must have a 'CellType' or 'class' column");
        }
        if (primaryCol < 0) {
            throw new IOException("Rule-format CSV must have a 'PrimaryMarker' column");
        }

        String line;
        while ((line = reader.readLine()) != null) {
            line = line.strip();
            if (line.isEmpty()) continue;

            String[] parts = line.split(",", -1);
            if (parts.length <= classCol) continue;

            String cellType = parts[classCol].strip();
            if (cellType.isEmpty()) continue;

            String primary = safeGet(parts, primaryCol);
            String secondary = safeGet(parts, secondaryCol);
            String tertiary = safeGet(parts, tertiaryCol);

            tbl.putRule(cellType, primary, secondary, tertiary);
        }
        return tbl;
    }

    private static String safeGet(String[] parts, int idx) {
        if (idx < 0 || idx >= parts.length) return null;
        String val = parts[idx].strip();
        return val.isEmpty() ? null : val;
    }

    /**
     * Save this table to a CSV file.
     * Uses rule format if gating rules are present, otherwise simple format.
     */
    public void saveToCSV(Path path) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            if (hasRules) {
                writer.write("CellType,PrimaryMarker,SecondaryMarker,TertiaryMarker");
                writer.newLine();
                for (String cellType : table.keySet()) {
                    writer.write(cellType);
                    writer.write(',');
                    writer.write(Objects.toString(primaryMarkers.get(cellType), ""));
                    writer.write(',');
                    writer.write(Objects.toString(secondaryMarkers.get(cellType), ""));
                    writer.write(',');
                    writer.write(Objects.toString(tertiaryMarkers.get(cellType), ""));
                    writer.newLine();
                }
            } else {
                StringBuilder header = new StringBuilder("CellType");
                for (int i = 1; i <= MAX_MARKERS; i++) {
                    header.append(",Marker").append(i);
                }
                writer.write(header.toString());
                writer.newLine();
                for (var entry : table.entrySet()) {
                    StringBuilder sb = new StringBuilder();
                    sb.append(entry.getKey());
                    List<String> markers = entry.getValue();
                    for (int i = 0; i < MAX_MARKERS; i++) {
                        sb.append(',');
                        if (i < markers.size()) {
                            sb.append(markers.get(i));
                        }
                    }
                    writer.write(sb.toString());
                    writer.newLine();
                }
            }
        }
    }

    @Override
    public String toString() {
        return "CellTypeTable[" + table.size() + " types" + (hasRules ? ", with gating rules" : "") + "]";
    }
}
