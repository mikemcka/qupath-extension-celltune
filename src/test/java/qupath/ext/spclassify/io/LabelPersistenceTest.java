package qupath.ext.spclassify.io;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.reflect.Proxy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import qupath.ext.spclassify.model.LabelStore;
import qupath.lib.projects.Project;

/**
 * Round-trip coverage for the per-image label persistence extracted into
 * {@link LabelPersistence}. Exercised through the {@link ProjectStateManager}
 * facade so both the delegation and the underlying logic are tested.
 */
class LabelPersistenceTest {

    @TempDir
    Path tempDir;

    @Test
    void savesAndLoadsMultiClassLabelsRoundTrip() throws Exception {
        Project<BufferedImage> project = fakeProject(tempDir.resolve("p/project.qpproj"));

        LabelStore store = new LabelStore("slide A.tif");
        store.setLabel("cell-1", "Tumour");
        store.setLabel("cell-2", "Stroma");

        assertFalse(ProjectStateManager.hasImageLabels(project, "slide A.tif"));
        ProjectStateManager.saveImageLabels(project, "slide A.tif", store);
        assertTrue(ProjectStateManager.hasImageLabels(project, "slide A.tif"));

        LabelStore loaded = ProjectStateManager.loadImageLabels(project, "slide A.tif");
        assertNotNull(loaded);
        assertEquals(2, loaded.size());
        assertEquals("Tumour", loaded.getLabel("cell-1"));
        assertEquals("Stroma", loaded.getLabel("cell-2"));
    }

    @Test
    void scopedLabelsAreIsolatedFromMultiClass() throws Exception {
        Project<BufferedImage> project = fakeProject(tempDir.resolve("scoped/project.qpproj"));

        LabelStore multi = new LabelStore("img");
        multi.setLabel("c", "Tumour");
        ProjectStateManager.saveImageLabels(project, null, "img", multi);

        LabelStore binary = new LabelStore("img");
        binary.setLabel("c", "CD4_pos");
        ProjectStateManager.saveImageLabels(project, "CD4", "img", binary);

        // Same image name, different scope -> independent files.
        assertEquals(
                "Tumour",
                ProjectStateManager.loadImageLabels(project, null, "img").getLabel("c"));
        assertEquals(
                "CD4_pos",
                ProjectStateManager.loadImageLabels(project, "CD4", "img").getLabel("c"));

        assertTrue(ProjectStateManager.hasImageLabels(project, "CD4", "img"));
        assertFalse(ProjectStateManager.hasImageLabels(project, "CD8", "img"));
    }

    @Test
    void loadReturnsNullWhenNoFile() throws Exception {
        Project<BufferedImage> project = fakeProject(tempDir.resolve("none/project.qpproj"));
        assertNull(ProjectStateManager.loadImageLabels(project, "never-saved"));
        assertNull(ProjectStateManager.loadImageLabels(project, "CD4", "never-saved"));
    }

    @Test
    void listsImageLabelFilesForScope() throws Exception {
        Project<BufferedImage> project = fakeProject(tempDir.resolve("list/project.qpproj"));

        LabelStore a = new LabelStore("a");
        a.setLabel("c", "X");
        LabelStore b = new LabelStore("b");
        b.setLabel("c", "Y");
        ProjectStateManager.saveImageLabels(project, "a", a);
        ProjectStateManager.saveImageLabels(project, "b", b);

        List<Path> files = ProjectStateManager.listImageLabelFiles(project, null);
        assertEquals(2, files.size());
        assertTrue(
                files.get(0)
                                .getFileName()
                                .toString()
                                .compareTo(files.get(1).getFileName().toString())
                        < 0,
                "files should be sorted");

        // Empty/non-existent scope dir -> empty list.
        assertTrue(ProjectStateManager.listImageLabelFiles(project, "no-such-scope")
                .isEmpty());
    }

    @Test
    void rawReadWriteRoundTrip() throws Exception {
        Path file = tempDir.resolve("raw/labels.json");
        assertEquals(Map.of(), ProjectStateManager.readImageLabelsRaw(file));

        ProjectStateManager.writeImageLabelsRaw(file, Map.of("c1", "A", "c2", "B"));
        Map<String, String> read = ProjectStateManager.readImageLabelsRaw(file);
        assertEquals(Map.of("c1", "A", "c2", "B"), read);
    }

    @Test
    void backupLabelsMergesDiskAndInMemory() throws Exception {
        Project<BufferedImage> project = fakeProject(tempDir.resolve("backup/project.qpproj"));

        // One image already persisted to disk.
        LabelStore onDisk = new LabelStore("other.tif");
        onDisk.setLabel("c-other", "Stroma");
        ProjectStateManager.saveImageLabels(project, "other.tif", onDisk);

        // Current open image only in memory.
        LabelStore inMemory = new LabelStore("current.tif");
        inMemory.setLabel("c-current", "Tumour");

        Path backup = ProjectStateManager.backupLabels(project, "current.tif", inMemory);
        assertNotNull(backup);
        assertTrue(Files.exists(backup));
        String json = Files.readString(backup);
        // Backup is a flat list of {cellId, className, imageName} records spanning both images.
        assertTrue(json.contains("c-other"), json);
        assertTrue(json.contains("c-current"), json);
        assertTrue(json.contains("other.tif"), json);
        assertTrue(json.contains("current.tif"), json);
    }

    @SuppressWarnings("unchecked")
    private static Project<BufferedImage> fakeProject(Path projectFile) {
        try {
            Files.createDirectories(projectFile.getParent());
            if (!Files.exists(projectFile)) {
                Files.createFile(projectFile);
            }
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }

        return (Project<BufferedImage>) Proxy.newProxyInstance(
                Project.class.getClassLoader(),
                new Class[] {Project.class},
                (proxy, method, args) -> switch (method.getName()) {
                    case "getPath" -> projectFile;
                    case "toString" -> "FakeProject(" + projectFile + ")";
                    case "hashCode" -> System.identityHashCode(proxy);
                    case "equals" -> proxy == args[0];
                    case "close" -> null;
                    default ->
                        throw new UnsupportedOperationException(
                                "Method not implemented in fake project: " + method.getName());
                });
    }
}
