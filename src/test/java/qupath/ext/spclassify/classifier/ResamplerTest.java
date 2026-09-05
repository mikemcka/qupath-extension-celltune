package qupath.ext.spclassify.classifier;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class ResamplerTest {

    // ── helpers ─────────────────────────────────────────────────────────────

    /** Build a list of n identical 2-feature rows for a given label value. */
    private static List<float[]> rows(int n, float f1, float f2) {
        List<float[]> list = new ArrayList<>();
        for (int i = 0; i < n; i++) list.add(new float[] {f1 + i * 0.001f, f2 + i * 0.001f});
        return list;
    }

    private static List<Integer> labels(int count, int label) {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < count; i++) list.add(label);
        return list;
    }

    private static int countLabel(List<Integer> labels, int target) {
        int count = 0;
        for (int l : labels) if (l == target) count++;
        return count;
    }

    // ── NONE ─────────────────────────────────────────────────────────────────

    @Test
    void noneStrategyReturnsShallowCopyUnchanged() {
        List<float[]> r = rows(5, 1f, 2f);
        List<Integer> l = labels(5, 0);

        Resampler.Result result = Resampler.apply(r, l, 1, ResamplingStrategy.NONE, null);

        assertEquals(5, result.rows().size());
        assertEquals(5, result.labels().size());
        // Verify it's a copy, not the same list instance
        result.rows().clear();
        assertEquals(5, r.size());
    }

    @Test
    void noneStrategyLeavesLabelsIntact() {
        List<float[]> r = new ArrayList<>();
        List<Integer> l = new ArrayList<>();
        r.add(new float[] {1f, 2f});
        l.add(0);
        r.add(new float[] {3f, 4f});
        l.add(1);

        Resampler.Result result = Resampler.apply(r, l, 2, ResamplingStrategy.NONE, null);

        assertEquals(0, result.labels().get(0));
        assertEquals(1, result.labels().get(1));
    }

    // ── SMOTE ────────────────────────────────────────────────────────────────

    @Test
    void smoteBalancesBinaryImbalance() {
        // 10 majority (class 0), 3 minority (class 1)
        List<float[]> r = new ArrayList<>();
        List<Integer> l = new ArrayList<>();
        r.addAll(rows(10, 0f, 0f));
        for (int i = 0; i < 10; i++) l.add(0);
        r.addAll(rows(3, 5f, 5f));
        for (int i = 0; i < 3; i++) l.add(1);

        Resampler.Result result = Resampler.apply(r, l, 2, ResamplingStrategy.SMOTE, null);

        int count0 = countLabel(result.labels(), 0);
        int count1 = countLabel(result.labels(), 1);

        // After SMOTE minority should be brought up toward majority count
        assertTrue(count1 >= 3, "Minority class should not shrink");
        assertTrue(count1 >= count0 * 0.5, "Minority class should be substantially upsampled");
        assertEquals(result.rows().size(), result.labels().size(), "Rows and labels must stay parallel");
    }

    @Test
    void smoteDoesNotModifyOriginalLists() {
        List<float[]> r = new ArrayList<>();
        List<Integer> l = new ArrayList<>();
        r.addAll(rows(6, 0f, 0f));
        for (int i = 0; i < 6; i++) l.add(0);
        r.addAll(rows(2, 5f, 5f));
        for (int i = 0; i < 2; i++) l.add(1);

        int origSize = r.size();
        Resampler.apply(r, l, 2, ResamplingStrategy.SMOTE, null);

        assertEquals(origSize, r.size(), "Original rows list must not be mutated");
    }

    @Test
    void smoteRowsAndLabelsAlwaysParallel() {
        List<float[]> r = new ArrayList<>();
        List<Integer> l = new ArrayList<>();
        r.addAll(rows(8, 0f, 0f));
        for (int i = 0; i < 8; i++) l.add(0);
        r.addAll(rows(2, 5f, 5f));
        for (int i = 0; i < 2; i++) l.add(1);

        Resampler.Result result = Resampler.apply(r, l, 2, ResamplingStrategy.SMOTE, null);

        assertEquals(result.rows().size(), result.labels().size());
    }

    // ── TOMEK ────────────────────────────────────────────────────────────────

    @Test
    void tomekDoesNotAddSamples() {
        // 5 per class — Tomek can only remove, never add
        List<float[]> r = new ArrayList<>();
        List<Integer> l = new ArrayList<>();
        r.addAll(rows(5, 0f, 0f));
        for (int i = 0; i < 5; i++) l.add(0);
        r.addAll(rows(5, 5f, 5f));
        for (int i = 0; i < 5; i++) l.add(1);

        Resampler.Result result = Resampler.apply(r, l, 2, ResamplingStrategy.TOMEK, null);

        assertTrue(result.rows().size() <= r.size(), "Tomek cannot add samples");
        assertEquals(result.rows().size(), result.labels().size());
    }

    // ── SMOTE_TOMEK ──────────────────────────────────────────────────────────

    @Test
    void smoteTomekOutputIsParallelAndNonEmpty() {
        List<float[]> r = new ArrayList<>();
        List<Integer> l = new ArrayList<>();
        r.addAll(rows(8, 0f, 0f));
        for (int i = 0; i < 8; i++) l.add(0);
        r.addAll(rows(2, 9f, 9f));
        for (int i = 0; i < 2; i++) l.add(1);

        Resampler.Result result = Resampler.apply(r, l, 2, ResamplingStrategy.SMOTE_TOMEK, null);

        assertFalse(result.rows().isEmpty());
        assertEquals(result.rows().size(), result.labels().size());
    }

    // ── log callback ─────────────────────────────────────────────────────────

    @Test
    void logCallbackReceivesMessages() {
        List<float[]> r = new ArrayList<>();
        List<Integer> l = new ArrayList<>();
        r.addAll(rows(6, 0f, 0f));
        for (int i = 0; i < 6; i++) l.add(0);
        r.addAll(rows(2, 5f, 5f));
        for (int i = 0; i < 2; i++) l.add(1);

        List<String> messages = new ArrayList<>();
        Resampler.apply(r, l, 2, ResamplingStrategy.SMOTE, messages::add);

        assertFalse(messages.isEmpty(), "Log callback should receive at least one message");
    }

    @Test
    void multiClassSmoteBalancesAllClasses() {
        // 3 classes: 10 / 3 / 3
        List<float[]> r = new ArrayList<>();
        List<Integer> l = new ArrayList<>();
        r.addAll(rows(10, 0f, 0f));
        for (int i = 0; i < 10; i++) l.add(0);
        r.addAll(rows(3, 5f, 0f));
        for (int i = 0; i < 3; i++) l.add(1);
        r.addAll(rows(3, 0f, 5f));
        for (int i = 0; i < 3; i++) l.add(2);

        Resampler.Result result = Resampler.apply(r, l, 3, ResamplingStrategy.SMOTE, null);

        int c0 = countLabel(result.labels(), 0);
        int c1 = countLabel(result.labels(), 1);
        int c2 = countLabel(result.labels(), 2);

        assertTrue(c1 > 3, "Class 1 should be upsampled");
        assertTrue(c2 > 3, "Class 2 should be upsampled");
        assertEquals(c0, 10, "Majority class should not change");
        assertEquals(result.rows().size(), result.labels().size());
    }

    // ── label-index validation (regression for the out-of-range guard) ──────

    @Test
    void labelOutOfRangeThrowsClearMessage() {
        List<float[]> r = rows(4, 0f, 0f);
        List<Integer> l = new ArrayList<>(List.of(0, 1, 2, 5)); // 5 invalid when nClasses=3
        var ex = assertThrows(
                IllegalArgumentException.class, () -> Resampler.apply(r, l, 3, ResamplingStrategy.SMOTE, null));
        assertTrue(ex.getMessage().contains("index 3"), "Message should name the offending index");
        assertTrue(ex.getMessage().contains("[0, 3)"), "Message should state the valid range");
    }

    @Test
    void negativeLabelThrows() {
        List<float[]> r = rows(3, 0f, 0f);
        List<Integer> l = new ArrayList<>(List.of(0, -1, 1));
        assertThrows(IllegalArgumentException.class, () -> Resampler.apply(r, l, 2, ResamplingStrategy.NONE, null));
    }

    @Test
    void nClassesBelowOneThrows() {
        List<float[]> r = rows(2, 0f, 0f);
        List<Integer> l = new ArrayList<>(List.of(0, 0));
        assertThrows(IllegalArgumentException.class, () -> Resampler.apply(r, l, 0, ResamplingStrategy.NONE, null));
    }

    @Test
    void validLabelsPassValidation() {
        List<float[]> r = rows(4, 0f, 0f);
        List<Integer> l = new ArrayList<>(List.of(0, 1, 1, 0));
        // Should not throw; NONE returns a copy unchanged.
        Resampler.Result result = Resampler.apply(r, l, 2, ResamplingStrategy.NONE, null);
        assertEquals(4, result.labels().size());
    }
}
