package qupath.ext.spclassify.model;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Manages ground-truth cell labels — the human-assigned class for each cell.
 * <p>
 * This is the central data structure that accumulates labels across review cycles.
 * Labels from landmarking, gating, and review mode all flow into the same
 * {@code LabelStore}, which then feeds the next training cycle.
 * <p>
 * Serialisable to/from JSON via Gson (all fields are simple maps/strings).
 */
public class LabelStore {

    private final String name;
    // cellId → class name. Wrapped in a synchronised map because the same store
    // is mutated from the JavaFX thread (manual labelling), the training thread,
    // and the image-switch listener. Single-key operations are guarded by the
    // wrapper; compound iterate/stream operations below additionally synchronise
    // on {@code labels} (required for safe iteration of a synchronised map).
    private final Map<String, String> labels;

    /**
     * Create an empty label store.
     *
     * @param name descriptive name (e.g. "Landmarks", "Labels_Classifier_1")
     */
    public LabelStore(String name) {
        this.name = name;
        this.labels = Collections.synchronizedMap(new LinkedHashMap<>());
    }

    /**
     * Create a label store pre-populated with labels.
     *
     * @param name   descriptive name
     * @param labels initial cellId → class name map (copied)
     */
    public LabelStore(String name, Map<String, String> labels) {
        this.name = name;
        this.labels = Collections.synchronizedMap(new LinkedHashMap<>(labels));
    }

    /** Assign a class label to a cell. Overwrites any existing label. */
    public void setLabel(String cellId, String className) {
        labels.put(cellId, className);
    }

    /** Remove a cell's label. */
    public void removeLabel(String cellId) {
        labels.remove(cellId);
    }

    /** Get a cell's label, or null if unlabelled. */
    public String getLabel(String cellId) {
        return labels.get(cellId);
    }

    /** @return true if the cell has a label in this store */
    public boolean hasLabel(String cellId) {
        return labels.containsKey(cellId);
    }

    /** @return unmodifiable view of all cellId → label mappings */
    public Map<String, String> getAllLabels() {
        return Collections.unmodifiableMap(labels);
    }

    /** @return total number of labelled cells */
    public int size() {
        return labels.size();
    }

    /**
     * Strip merge-history annotation from a raw label value, returning only the
     * effective (current) class name.
     * <p>
     * For example, {@code "test1-mergedInto(myType)"} returns {@code "myType"}.
     * Labels without a merge annotation are returned unchanged.
     *
     * @param rawLabel the raw label value as stored on disk (may be null)
     * @return the effective class name, or null if rawLabel is null
     */
    public static String effectiveClassName(String rawLabel) {
        if (rawLabel == null) return null;
        int start = rawLabel.indexOf("-mergedInto(");
        if (start >= 0 && rawLabel.endsWith(")")) {
            return rawLabel.substring(start + "-mergedInto(".length(), rawLabel.length() - 1);
        }
        return rawLabel;
    }

    /**
     * Strip all merge-history layers to reveal the very first class name.
     * E.g. {@code "a-mergedInto(b-mergedInto(c))"} → {@code "a"}.
     * Labels without a merge annotation are returned unchanged.
     *
     * @param rawLabel the raw label value (may be null)
     * @return the innermost original class name, or null if rawLabel is null
     */
    public static String innermostOriginal(String rawLabel) {
        if (rawLabel == null) return null;
        int idx = rawLabel.indexOf("-mergedInto(");
        return idx >= 0 ? rawLabel.substring(0, idx) : rawLabel;
    }

    /**
     * @return set of all distinct <em>effective</em> class names (merge-history
     * annotations stripped) that have at least one labelled cell
     */
    public Set<String> getClassNames() {
        synchronized (labels) {
            return labels.values().stream()
                    .map(LabelStore::effectiveClassName)
                    .collect(Collectors.toCollection(LinkedHashSet::new));
        }
    }

    /**
     * @return a new map of cellId → <em>effective</em> class name (merge-history
     * annotations stripped), suitable for training pipelines that must not see
     * the raw encoded form
     */
    public Map<String, String> getEffectiveLabels() {
        Map<String, String> result = new LinkedHashMap<>();
        synchronized (labels) {
            labels.forEach((id, raw) -> result.put(id, effectiveClassName(raw)));
        }
        return result;
    }

    /** @return count of labelled cells per class */
    public Map<String, Long> getClassCounts() {
        synchronized (labels) {
            return labels.values().stream()
                    .collect(Collectors.groupingBy(c -> c, LinkedHashMap::new, Collectors.counting()));
        }
    }

    /** @return all cell IDs that have a specific class label */
    public List<String> getCellsWithLabel(String className) {
        synchronized (labels) {
            return labels.entrySet().stream()
                    .filter(e -> className.equals(e.getValue()))
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toList());
        }
    }

    /** Merge all labels from another store into this one. Existing labels are overwritten. */
    public void mergeFrom(LabelStore other) {
        // Snapshot the source under its own lock, then put-all under ours, to
        // avoid holding two map monitors at once (deadlock-safe ordering).
        Map<String, String> snapshot;
        synchronized (other.labels) {
            snapshot = new LinkedHashMap<>(other.labels);
        }
        labels.putAll(snapshot);
    }

    /**
     * Remove all labels whose <em>effective</em> class name is not in the given
     * set of valid classes.  Labels that carry merge-history annotations are
     * matched by their effective (post-merge) class, not their raw stored value.
     *
     * @param validClasses the set of class names to keep
     * @return number of labels removed
     */
    public int retainClasses(Set<String> validClasses) {
        synchronized (labels) {
            int before = labels.size();
            labels.entrySet().removeIf(e -> !validClasses.contains(effectiveClassName(e.getValue())));
            return before - labels.size();
        }
    }

    /**
     * Rename labels whose <em>effective</em> class equals {@code oldName} to
     * {@code newName}. Raw values that already carry merge-history annotation
     * are matched by their effective (post-merge) class, so chained merges
     * stay in sync with the on-disk representation.
     *
     * @param oldName the effective class name to match
     * @param newName the new raw value to assign
     * @return number of labels renamed
     */
    public int renameClass(String oldName, String newName) {
        synchronized (labels) {
            int count = 0;
            for (var entry : labels.entrySet()) {
                if (oldName.equals(effectiveClassName(entry.getValue()))) {
                    entry.setValue(newName);
                    count++;
                }
            }
            return count;
        }
    }

    /**
     * Restore labels that were encoded by a merge operation.
     * Any label whose raw value ends with {@code mergedSuffix}
     * (e.g. {@code "-mergedInto(myType)"}) is reverted to the innermost
     * original class name (the part before the first {@code -mergedInto(}).
     *
     * @param mergedSuffix the suffix to match, e.g. {@code "-mergedInto(myType)"}
     * @return number of labels restored
     */
    public int restoreMergedLabels(String mergedSuffix) {
        synchronized (labels) {
            int count = 0;
            for (var entry : labels.entrySet()) {
                String raw = entry.getValue();
                if (raw != null && raw.endsWith(mergedSuffix)) {
                    int idx = raw.indexOf("-mergedInto(");
                    String original = idx >= 0 ? raw.substring(0, idx) : raw;
                    entry.setValue(original);
                    count++;
                }
            }
            return count;
        }
    }

    /** Create a deep copy of this label store. */
    public LabelStore copy() {
        synchronized (labels) {
            return new LabelStore(name, labels);
        }
    }

    public String getName() {
        return name;
    }

    @Override
    public String toString() {
        return "LabelStore[" + name + ", " + labels.size() + " labels, "
                + getClassNames().size() + " classes]";
    }
}
