package qupath.ext.spclassify.io;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
import qupath.ext.spclassify.model.CellPrediction;
import qupath.ext.spclassify.model.PopulationSet;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;

/**
 * Round-trip coverage for the per-image prediction persistence extracted into
 * {@link PredictionPersistence}. Exercised through the {@link ProjectStateManager}
 * facade so both the delegation and the underlying logic are tested.
 */
class PredictionPersistenceTest {

    @TempDir
    Path tempDir;

    private static final List<String> CLASSES = List.of("Tumour", "Stroma", "Immune");

    @Test
    void savesAndLoadsPredictionsRoundTrip() throws Exception {
        Project<BufferedImage> project = fakeProject(tempDir.resolve("p/project.qpproj"), List.of());

        PopulationSet predAll = new PopulationSet("Pred_ALL");
        predAll.put(
                "cell-1",
                new CellPrediction(
                        "cell-1",
                        "Tumour",
                        "Tumour",
                        new float[] {0.8f, 0.1f, 0.1f},
                        new float[] {0.7f, 0.2f, 0.1f},
                        CLASSES));
        predAll.put(
                "cell-2",
                new CellPrediction(
                        "cell-2",
                        "Stroma",
                        "Immune",
                        new float[] {0.1f, 0.6f, 0.3f},
                        new float[] {0.2f, 0.3f, 0.5f},
                        CLASSES));

        Path written = ProjectStateManager.saveImagePredictions(project, "slide A.tif", predAll);
        assertNotNull(written);
        assertTrue(Files.exists(written));

        PopulationSet loaded = ProjectStateManager.loadImagePredictions(project, "slide A.tif");
        assertNotNull(loaded);
        assertEquals(2, loaded.size());

        Map<String, CellPrediction> all = loaded.getAll();
        CellPrediction c1 = all.get("cell-1");
        assertNotNull(c1);
        assertEquals("Tumour", c1.getModel1Label());
        assertEquals("Tumour", c1.getModel2Label());
        assertEquals(CLASSES, c1.getClassNames());
        assertArrayEqualsExact(new float[] {0.8f, 0.1f, 0.1f}, c1.getModel1Probs());
        assertArrayEqualsExact(new float[] {0.7f, 0.2f, 0.1f}, c1.getModel2Probs());

        CellPrediction c2 = all.get("cell-2");
        assertNotNull(c2);
        assertEquals("Stroma", c2.getModel1Label());
        assertEquals("Immune", c2.getModel2Label());
    }

    @Test
    void emptyPredictionSetWritesNothing() throws Exception {
        Project<BufferedImage> project = fakeProject(tempDir.resolve("empty/project.qpproj"), List.of());
        assertNull(ProjectStateManager.saveImagePredictions(project, "img", new PopulationSet("Pred_ALL")));
        assertNull(ProjectStateManager.saveImagePredictions(project, "img", null));
    }

    @Test
    void loadReturnsNullWhenNoFile() throws Exception {
        Project<BufferedImage> project = fakeProject(tempDir.resolve("none/project.qpproj"), List.of());
        assertNull(ProjectStateManager.loadImagePredictions(project, "never-saved"));
    }

    @Test
    void listsImagesWithPredictions() throws Exception {
        Project<BufferedImage> project =
                fakeProject(tempDir.resolve("list/project.qpproj"), List.of("with-preds.tif", "without-preds.tif"));

        PopulationSet predAll = new PopulationSet("Pred_ALL");
        predAll.put(
                "cell-1",
                new CellPrediction(
                        "cell-1",
                        "Tumour",
                        "Tumour",
                        new float[] {0.8f, 0.1f, 0.1f},
                        new float[] {0.7f, 0.2f, 0.1f},
                        CLASSES));
        ProjectStateManager.saveImagePredictions(project, "with-preds.tif", predAll);

        List<String> imagesWithPreds = ProjectStateManager.listImagesWithPredictions(project);
        assertEquals(List.of("with-preds.tif"), imagesWithPreds);
    }

    private static void assertArrayEqualsExact(float[] expected, float[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], 1e-6f, "index " + i);
        }
    }

    @SuppressWarnings("unchecked")
    private static Project<BufferedImage> fakeProject(Path projectFile, List<String> imageNames) {
        try {
            Files.createDirectories(projectFile.getParent());
            if (!Files.exists(projectFile)) {
                Files.createFile(projectFile);
            }
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }

        List<ProjectImageEntry<BufferedImage>> entries =
                imageNames.stream().map(PredictionPersistenceTest::fakeEntry).toList();

        return (Project<BufferedImage>) Proxy.newProxyInstance(
                Project.class.getClassLoader(),
                new Class[] {Project.class},
                (proxy, method, args) -> switch (method.getName()) {
                    case "getPath" -> projectFile;
                    case "getImageList" -> entries;
                    case "toString" -> "FakeProject(" + projectFile + ")";
                    case "hashCode" -> System.identityHashCode(proxy);
                    case "equals" -> proxy == args[0];
                    case "close" -> null;
                    default ->
                        throw new UnsupportedOperationException(
                                "Method not implemented in fake project: " + method.getName());
                });
    }

    @SuppressWarnings("unchecked")
    private static ProjectImageEntry<BufferedImage> fakeEntry(String imageName) {
        return (ProjectImageEntry<BufferedImage>) Proxy.newProxyInstance(
                ProjectImageEntry.class.getClassLoader(),
                new Class[] {ProjectImageEntry.class},
                (proxy, method, args) -> switch (method.getName()) {
                    case "getImageName" -> imageName;
                    case "toString" -> "FakeEntry(" + imageName + ")";
                    case "hashCode" -> System.identityHashCode(proxy);
                    case "equals" -> proxy == args[0];
                    default ->
                        throw new UnsupportedOperationException(
                                "Method not implemented in fake entry: " + method.getName());
                });
    }
}
