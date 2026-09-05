package qupath.ext.spclassify.io;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javafx.application.Platform;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.model.LabelStore;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.projects.Project;

/**
 * Backend logic for the Class Control dialog.
 * <p>
 * Supports three operations on both QuPath's class panel and the persisted
 * {@code celltune/image-labels/*.json} label files:
 * <ul>
 *   <li><b>Add</b> — register a new PathClass in QuPath's available-class list.</li>
 *   <li><b>Delete</b> — remove a PathClass; optionally purge its labels from
 *       all image-label files.</li>
 *   <li><b>Merge</b> — reassign source classes to a target class, encoding the
 *       original class in the label value so the operation is reversible:
 *       {@code "test1"} merged into {@code "myType"} becomes
 *       {@code "test1-mergedInto(myType)"}.</li>
 * </ul>
 * All file I/O is performed synchronously on whatever thread calls these methods.
 * Callers that invoke from the JavaFX Application Thread should do so via a
 * background task to avoid blocking the UI.
 */
public final class ClassManager {

    private static final Logger logger = LoggerFactory.getLogger(ClassManager.class);
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    @SuppressWarnings("unchecked")
    private static final Type MAP_TYPE = new TypeToken<Map<String, String>>() {}.getType();

    /** Separator used to encode merge history inside a label value. */
    public static final String MERGE_SEPARATOR = "-mergedInto(";
    /** Closing character for the merge-history annotation. */
    public static final String MERGE_CLOSE = ")";

    private ClassManager() {}

    // ─────────────────────────────────────────────────────────────────────────
    //  Add
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Add a new PathClass to QuPath's available-class list.
     * <p>
     * If a class with the same name already exists the call is a no-op.
     * Must be called on the JavaFX Application Thread.
     *
     * @param qupath the QuPath GUI instance
     * @param name   the new class name (must not be blank)
     */
    public static void addPathClass(QuPathGUI qupath, String name) {
        if (name == null || name.isBlank()) return;
        runOnFx(() -> {
            var available = qupath.getAvailablePathClasses();
            boolean exists = available.stream().anyMatch(pc -> pc != null && name.equals(pc.getName()));
            if (!exists) {
                available.add(PathClass.fromString(name));
                syncProjectClasses(qupath);
                logger.info("Added PathClass '{}'", name);
            }
        });
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Delete
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Remove a PathClass from QuPath's available-class list and optionally purge
     * its labels from all image-label JSON files.
     * <p>
     * Must be called on the JavaFX Application Thread for the QuPath list update.
     * The file I/O is performed inline, so do not call this from the FX thread
     * on large projects — wrap in a background task first.
     *
     * @param qupath       the QuPath GUI instance
     * @param name         the class name to delete
     * @param purgeLabels  if true, labels with this effective class are removed
     *                     from every image-label JSON file
     * @return number of label entries purged across all files (0 if purgeLabels is false)
     * @throws IOException if a label file cannot be read or written
     */
    public static int deletePathClass(QuPathGUI qupath, String name, boolean purgeLabels) throws IOException {
        if (name == null || name.isBlank()) return 0;

        // Remove from QuPath available classes (must be on FX thread)
        runOnFx(() -> {
            var available = qupath.getAvailablePathClasses();
            available.removeIf(pc -> pc != null && name.equals(pc.getName()));
            syncProjectClasses(qupath);
            logger.info("Removed PathClass '{}'", name);
        });

        if (!purgeLabels) return 0;

        var project = qupath.getProject();
        if (project == null) return 0;

        int total = 0;
        for (Path jsonPath : listImageLabelFiles(project)) {
            total += purgeClassFromFile(jsonPath, name);
        }
        logger.info("Purged {} label entries for deleted class '{}'", total, name);
        return total;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Merge
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Merge one or more source classes into a target class.
     * <p>
     * For each source class label in every image-label JSON file, the raw value
     * is encoded as {@code "<original>-mergedInto(<target>)"}.  This preserves
     * the merge history inside the label file so the operation can be undone
     * (see {@link #undoMerge}).
     * <p>
     * The source PathClasses are removed from QuPath's available-class list and
     * the target PathClass is added if it does not already exist.
     * <p>
     * The supplied {@code inMemoryLabelStore} (may be null) is also updated
     * so the current session sees the change without a reload.
     *
     * @param qupath              the QuPath GUI instance
     * @param sourceClasses       classes to fold into the target
     * @param targetClass         the name of the resulting merged class
     * @param inMemoryLabelStore  the extension's current in-memory store, or null
     * @return number of label entries rewritten across all image files
     * @throws IOException if a label file cannot be read or written
     */
    public static int mergeClasses(
            QuPathGUI qupath, List<String> sourceClasses, String targetClass, LabelStore inMemoryLabelStore)
            throws IOException {
        if (sourceClasses == null || sourceClasses.isEmpty()) return 0;
        if (targetClass == null || targetClass.isBlank()) return 0;

        // Update QuPath available classes: add target, remove sources (must be on FX thread)
        runOnFx(() -> {
            var available = qupath.getAvailablePathClasses();
            boolean targetExists = available.stream().anyMatch(pc -> pc != null && targetClass.equals(pc.getName()));
            if (!targetExists) {
                available.add(PathClass.fromString(targetClass));
            }
            for (String src : sourceClasses) {
                if (!src.equals(targetClass)) {
                    available.removeIf(pc -> pc != null && src.equals(pc.getName()));
                }
            }
            syncProjectClasses(qupath);
        });

        // Rewrite JSON files
        int total = 0;
        var project = qupath.getProject();
        if (project != null) {
            for (Path jsonPath : listImageLabelFiles(project)) {
                total += mergeClassesInFile(jsonPath, sourceClasses, targetClass);
            }
        }

        // Update in-memory label store
        if (inMemoryLabelStore != null) {
            for (String src : sourceClasses) {
                if (!src.equals(targetClass)) {
                    String encodedTarget = src + MERGE_SEPARATOR + targetClass + MERGE_CLOSE;
                    inMemoryLabelStore.renameClass(src, encodedTarget);
                }
            }
        }

        logger.info(
                "Merged {} source class(es) into '{}': {} labels rewritten", sourceClasses.size(), targetClass, total);
        return total;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Undo merge
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Undo a previous merge by restoring labels that were encoded as
     * {@code "<original>-mergedInto(<target>)"} back to their original class name.
     * <p>
     * Only labels whose effective class matches {@code targetClass} AND whose raw
     * value carries a merge-history annotation are reverted.  Labels that were
     * naturally named {@code targetClass} (no history) are left unchanged.
     * <p>
     * The original source PathClasses are re-added to QuPath's available-class
     * list.  If no more labels reference {@code targetClass} after the undo the
     * caller may choose to delete it, but this method does not do so automatically.
     *
     * @param qupath              the QuPath GUI instance
     * @param targetClass         the merged class to undo
     * @param inMemoryLabelStore  the extension's current in-memory store, or null
     * @return number of label entries restored
     * @throws IOException if a label file cannot be read or written
     */
    public static int undoMerge(QuPathGUI qupath, String targetClass, LabelStore inMemoryLabelStore)
            throws IOException {
        if (targetClass == null || targetClass.isBlank()) return 0;

        int total = 0;
        var project = qupath.getProject();
        if (project != null) {
            for (Path jsonPath : listImageLabelFiles(project)) {
                total += undoMergeInFile(jsonPath, targetClass, qupath);
            }
        }

        // Restore in-memory label store
        if (inMemoryLabelStore != null) {
            String suffix = MERGE_SEPARATOR + targetClass + MERGE_CLOSE;
            inMemoryLabelStore.restoreMergedLabels(suffix);
        }

        logger.info("Undo merge into '{}': {} labels restored", targetClass, total);
        return total;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Helpers — file I/O
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * List all {@code *.json} files in the {@code celltune/image-labels/} directory.
     */
    public static List<Path> listImageLabelFiles(Project<?> project) throws IOException {
        Path dir = ProjectStateManager.getCellTuneDir(project).resolve("image-labels");
        if (!Files.isDirectory(dir)) return List.of();
        try (Stream<Path> s = Files.list(dir)) {
            return s.filter(p -> p.toString().endsWith(".json")).collect(Collectors.toList());
        }
    }

    private static Map<String, String> readLabelFile(Path path) throws IOException {
        String json = Files.readString(path, StandardCharsets.UTF_8);
        Map<String, String> map = GSON.fromJson(json, MAP_TYPE);
        return map != null ? new LinkedHashMap<>(map) : new LinkedHashMap<>();
    }

    private static void writeLabelFile(Path path, Map<String, String> labels) throws IOException {
        Files.writeString(path, GSON.toJson(labels), StandardCharsets.UTF_8);
    }

    /** Purge all entries whose effective class equals {@code className}. */
    static int purgeClassFromFile(Path path, String className) throws IOException {
        Map<String, String> labels = readLabelFile(path);
        int before = labels.size();
        labels.entrySet().removeIf(e -> className.equals(LabelStore.effectiveClassName(e.getValue())));
        int removed = before - labels.size();
        if (removed > 0) writeLabelFile(path, labels);
        return removed;
    }

    /** Rewrite entries matching any source class into the merge-encoded form. */
    static int mergeClassesInFile(Path path, List<String> sourceClasses, String targetClass) throws IOException {
        Map<String, String> labels = readLabelFile(path);
        int count = 0;
        for (var entry : labels.entrySet()) {
            String raw = entry.getValue();
            String effective = LabelStore.effectiveClassName(raw);
            if (sourceClasses.contains(effective) && !effective.equals(targetClass)) {
                // Encode: preserve the original (inner-most) class name, not an already-merged one
                String original = innermostOriginal(raw);
                entry.setValue(original + MERGE_SEPARATOR + targetClass + MERGE_CLOSE);
                count++;
            }
        }
        if (count > 0) writeLabelFile(path, labels);
        return count;
    }

    /**
     * Strip all merge-history layers to reveal the very first class name.
     * E.g. {@code "a-mergedInto(b-mergedInto(c))"} → {@code "a"}.
     */
    static String innermostOriginal(String raw) {
        if (raw == null) return raw;
        int idx = raw.indexOf(MERGE_SEPARATOR);
        return idx >= 0 ? raw.substring(0, idx) : raw;
    }

    /**
     * Restore labels whose value ends with {@code "-mergedInto(<targetClass>)"}
     * to their innermost original class.
     */
    private static int undoMergeInFile(Path path, String targetClass, QuPathGUI qupath) throws IOException {
        Map<String, String> labels = readLabelFile(path);
        List<String> restoredOriginals = restoreMergedInMap(labels, targetClass);
        for (String original : restoredOriginals) {
            // Re-add the original PathClass to QuPath if absent (idempotent).
            addPathClassOnFxThread(qupath, original);
        }
        if (!restoredOriginals.isEmpty()) writeLabelFile(path, labels);
        return restoredOriginals.size();
    }

    /**
     * In-place undo of a merge within a label map: every value ending with
     * {@code "-mergedInto(<targetClass>)"} is reverted to its innermost original
     * class. Pure (no I/O, no QuPath) so it can be unit-tested directly.
     *
     * @return the original class names restored, one per reverted entry (order
     *         preserved; duplicates kept so the count equals entries changed)
     */
    static List<String> restoreMergedInMap(Map<String, String> labels, String targetClass) {
        String suffix = MERGE_SEPARATOR + targetClass + MERGE_CLOSE;
        List<String> restoredOriginals = new ArrayList<>();
        for (var entry : labels.entrySet()) {
            String raw = entry.getValue();
            if (raw != null && raw.endsWith(suffix)) {
                String original = innermostOriginal(raw);
                entry.setValue(original);
                restoredOriginals.add(original);
            }
        }
        return restoredOriginals;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  QuPath class-list sync
    // ─────────────────────────────────────────────────────────────────────────

    private static void syncProjectClasses(QuPathGUI qupath) {
        var project = qupath.getProject();
        if (project == null) return;
        try {
            // Only update the project's class list. Do NOT call project.syncChanges()
            // here — that flushes the entire project (including any pending in-memory
            // image-list edits) to disk and can clobber unrelated state. QuPath will
            // persist the class list with the next normal project save.
            project.setPathClasses(new ArrayList<>(qupath.getAvailablePathClasses()));
        } catch (Exception e) {
            logger.warn("Could not sync PathClasses to project: {}", e.getMessage());
        }
    }

    private static void addPathClassOnFxThread(QuPathGUI qupath, String name) {
        if (name == null || name.isBlank()) return;
        addPathClass(qupath, name);
    }

    /**
     * Run {@code task} on the JavaFX Application Thread, blocking until it completes.
     * If already on the FX thread runs inline.  Any RuntimeException thrown by the
     * task is rethrown on the calling thread.
     */
    private static void runOnFx(Runnable task) {
        if (Platform.isFxApplicationThread()) {
            task.run();
            return;
        }
        CountDownLatch latch = new CountDownLatch(1);
        AtomicReference<RuntimeException> err = new AtomicReference<>();
        Platform.runLater(() -> {
            try {
                task.run();
            } catch (RuntimeException e) {
                err.set(e);
            } finally {
                latch.countDown();
            }
        });
        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted waiting for FX thread", e);
        }
        if (err.get() != null) throw err.get();
    }
}
