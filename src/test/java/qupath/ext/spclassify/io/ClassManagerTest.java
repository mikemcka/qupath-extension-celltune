package qupath.ext.spclassify.io;

import static org.junit.jupiter.api.Assertions.*;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests the GUI-free merge / undo / purge file logic of {@link ClassManager}.
 * The reversibility contract is: merging class X into Y encodes the label as
 * {@code "X-mergedInto(Y)"} (keeping the innermost original), and undo strips that
 * suffix back to X.
 */
class ClassManagerTest {

    private static final Gson GSON = new Gson();
    private static final Type MAP_TYPE = new TypeToken<Map<String, String>>() {}.getType();

    private static Path writeLabels(Path dir, Map<String, String> labels) throws Exception {
        Path p = dir.resolve("img.json");
        Files.writeString(p, GSON.toJson(labels), StandardCharsets.UTF_8);
        return p;
    }

    private static Map<String, String> readLabels(Path p) throws Exception {
        return GSON.fromJson(Files.readString(p, StandardCharsets.UTF_8), MAP_TYPE);
    }

    private static Map<String, String> map(String... kv) {
        Map<String, String> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) m.put(kv[i], kv[i + 1]);
        return m;
    }

    // ── innermostOriginal ────────────────────────────────────────────────────

    @Test
    void innermostOriginalUnwrapsChainedMerges() {
        assertEquals("A", ClassManager.innermostOriginal("A-mergedInto(B)"));
        assertEquals("A", ClassManager.innermostOriginal("A-mergedInto(B-mergedInto(C))"));
        assertEquals("Plain", ClassManager.innermostOriginal("Plain"));
        assertNull(ClassManager.innermostOriginal(null));
    }

    // ── mergeClassesInFile ───────────────────────────────────────────────────

    @Test
    void mergeEncodesSourceLabelsAndLeavesOthers(@TempDir Path tmp) throws Exception {
        Path f = writeLabels(tmp, map("c1", "A", "c2", "B", "c3", "A"));

        int count = ClassManager.mergeClassesInFile(f, List.of("A"), "Z");

        assertEquals(2, count);
        Map<String, String> after = readLabels(f);
        assertEquals("A-mergedInto(Z)", after.get("c1"));
        assertEquals("A-mergedInto(Z)", after.get("c3"));
        assertEquals("B", after.get("c2"), "unrelated class must be untouched");
    }

    @Test
    void mergePreservesInnermostOriginalWhenRemerging(@TempDir Path tmp) throws Exception {
        // c1 was already merged A→Z; now merge Z→W. The innermost original (A)
        // must be kept so a later undo can still recover it.
        Path f = writeLabels(tmp, map("c1", "A-mergedInto(Z)", "c2", "Z"));

        int count = ClassManager.mergeClassesInFile(f, List.of("Z"), "W");

        assertEquals(2, count);
        Map<String, String> after = readLabels(f);
        assertEquals("A-mergedInto(W)", after.get("c1"));
        assertEquals("Z-mergedInto(W)", after.get("c2"));
    }

    @Test
    void mergeIntoSelfIsNoOp(@TempDir Path tmp) throws Exception {
        Path f = writeLabels(tmp, map("c1", "A", "c2", "A"));
        int count = ClassManager.mergeClassesInFile(f, List.of("A"), "A");
        assertEquals(0, count);
        assertEquals("A", readLabels(f).get("c1"));
    }

    // ── purgeClassFromFile ───────────────────────────────────────────────────

    @Test
    void purgeRemovesByEffectiveClass(@TempDir Path tmp) throws Exception {
        Path f = writeLabels(tmp, map("c1", "A", "c2", "A-mergedInto(Z)", "c3", "B"));

        // Effective class of c2 is Z, so purging Z removes it.
        int removed = ClassManager.purgeClassFromFile(f, "Z");
        assertEquals(1, removed);
        Map<String, String> after = readLabels(f);
        assertFalse(after.containsKey("c2"));
        assertTrue(after.containsKey("c1"));
        assertTrue(after.containsKey("c3"));
    }

    // ── restoreMergedInMap (undo) ────────────────────────────────────────────

    @Test
    void undoRestoresOnlySuffixMatchingEntries() {
        Map<String, String> labels = map(
                "c1", "A-mergedInto(Z)",
                "c2", "Z", // natural Z, no history → keep
                "c3", "B-mergedInto(Z)",
                "c4", "C-mergedInto(W)"); // merged into a different target → keep

        List<String> restored = ClassManager.restoreMergedInMap(labels, "Z");

        assertEquals(List.of("A", "B"), restored);
        assertEquals("A", labels.get("c1"));
        assertEquals("B", labels.get("c3"));
        assertEquals("Z", labels.get("c2"), "natural target label must be left alone");
        assertEquals("C-mergedInto(W)", labels.get("c4"), "other-target merge must be left alone");
    }

    @Test
    void mergeThenUndoRoundTripsToOriginal() {
        Map<String, String> labels = map("c1", "A", "c2", "B", "c3", "A");
        // Simulate the in-file merge encoding, then undo it.
        for (var e : labels.entrySet()) {
            if ("A".equals(e.getValue())) e.setValue("A-mergedInto(Z)");
        }
        List<String> restored = ClassManager.restoreMergedInMap(labels, "Z");

        assertEquals(2, restored.size());
        assertEquals("A", labels.get("c1"));
        assertEquals("A", labels.get("c3"));
        assertEquals("B", labels.get("c2"));
    }
}
