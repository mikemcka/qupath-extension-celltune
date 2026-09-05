package qupath.ext.spclassify.io;

import static org.junit.jupiter.api.Assertions.*;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.reflect.Proxy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import qupath.lib.projects.Project;

class ProjectStateManagerBinaryGroundTruthTest {

    @TempDir
    Path tempDir;

    @Test
    void savesAndLoadsBinaryImportedTrainingRowsRoundTrip() throws Exception {
        Project<BufferedImage> project = fakeProject(tempDir.resolve("source-project/project.qpproj"));

        List<String> featureNames = List.of("feat_a", "feat_b");
        List<GroundTruthIO.TrainingRow> rows = List.of(
                new GroundTruthIO.TrainingRow("CD4_pos", new float[] {1.0f, 2.0f}),
                new GroundTruthIO.TrainingRow("CD4_neg", new float[] {3.5f, 4.5f}));

        ProjectStateManager.saveBinaryImportedTrainingData(project, "CD4", featureNames, rows);

        ProjectStateManager.BinaryImportedTrainingData loaded =
                ProjectStateManager.loadBinaryImportedTrainingData(project, "CD4");

        assertNotNull(loaded);
        assertEquals(featureNames, loaded.featureNames());
        assertEquals(2, loaded.rows().size());
        assertEquals("CD4_pos", loaded.rows().get(0).label());
        assertArrayEquals(new float[] {1.0f, 2.0f}, loaded.rows().get(0).features(), 1e-6f);
        assertEquals("CD4_neg", loaded.rows().get(1).label());
        assertArrayEquals(new float[] {3.5f, 4.5f}, loaded.rows().get(1).features(), 1e-6f);
    }

    @Test
    void returnsNullWhenBinaryImportedTrainingDataDoesNotExist() throws Exception {
        Project<BufferedImage> project = fakeProject(tempDir.resolve("target-project/project.qpproj"));
        assertNull(ProjectStateManager.loadBinaryImportedTrainingData(project, "CD8"));
    }

    @Test
    void deleteBinaryImportedTrainingDataRemovesPayload() throws Exception {
        Project<BufferedImage> project = fakeProject(tempDir.resolve("del-binary/project.qpproj"));

        ProjectStateManager.saveBinaryImportedTrainingData(
                project, "CD4", List.of("feat_a"), List.of(new GroundTruthIO.TrainingRow("CD4_pos", new float[] {1.0f
                })));
        assertNotNull(ProjectStateManager.loadBinaryImportedTrainingData(project, "CD4"));

        // First delete removes it and reports true; payload is gone afterwards.
        assertTrue(ProjectStateManager.deleteBinaryImportedTrainingData(project, "CD4"));
        assertNull(ProjectStateManager.loadBinaryImportedTrainingData(project, "CD4"));

        // Deleting again is a no-op and reports false.
        assertFalse(ProjectStateManager.deleteBinaryImportedTrainingData(project, "CD4"));
    }

    @Test
    void clearImportedTrainingDataStripsRowsButKeepsRestOfState() throws Exception {
        Project<BufferedImage> project = fakeProject(tempDir.resolve("clear-multi/project.qpproj"));

        // Seed a full classifier state (labels + class names + a model byte), then add imported rows.
        var labels = new qupath.ext.spclassify.model.LabelStore("CellTune");
        labels.setLabel("cell-1", "Tumour");
        ProjectStateManager.saveState(
                project,
                "MyClassifier",
                labels,
                List.of("feat_a", "feat_b"),
                List.of("Tumour", "Stroma"),
                new byte[] {1, 2, 3},
                null);
        ProjectStateManager.saveImportedTrainingData(
                project,
                List.of("feat_a", "feat_b"),
                List.of(new GroundTruthIO.TrainingRow("Tumour", new float[] {1.0f, 2.0f})));

        // Sanity: imported rows are present before clearing.
        assertNotNull(ProjectStateManager.decodeImportedTrainingRows(ProjectStateManager.loadState(project)));

        assertTrue(ProjectStateManager.clearImportedTrainingData(project));

        var state = ProjectStateManager.loadState(project);
        assertNotNull(state);
        // Imported rows + feature names are gone...
        assertNull(ProjectStateManager.decodeImportedTrainingRows(state));
        assertNull(ProjectStateManager.getImportedTrainingFeatureNames(state));
        // ...but the rest of the state is intact.
        assertEquals("MyClassifier", state.name);
        assertEquals(List.of("Tumour", "Stroma"), state.classNames);
        assertEquals("Tumour", state.labels.get("cell-1"));
        assertArrayEquals(new byte[] {1, 2, 3}, ProjectStateManager.decodeXGBoostModel(state));
    }

    @Test
    void clearImportedTrainingDataIsNoOpWithoutState() throws Exception {
        Project<BufferedImage> project = fakeProject(tempDir.resolve("clear-empty/project.qpproj"));
        assertFalse(ProjectStateManager.clearImportedTrainingData(project));
    }

    @SuppressWarnings("unchecked")
    private static Project<BufferedImage> fakeProject(Path projectFile) throws IOException {
        Files.createDirectories(projectFile.getParent());
        if (!Files.exists(projectFile)) {
            Files.createFile(projectFile);
        }

        return (Project<BufferedImage>) Proxy.newProxyInstance(
                Project.class.getClassLoader(), new Class[] {Project.class}, (proxy, method, args) -> {
                    String name = method.getName();
                    return switch (name) {
                        case "getPath" -> projectFile;
                        case "toString" -> "FakeProject(" + projectFile + ")";
                        case "hashCode" -> System.identityHashCode(proxy);
                        case "equals" -> proxy == args[0];
                        case "close" -> null;
                        default ->
                            throw new UnsupportedOperationException("Method not implemented in fake project: " + name);
                    };
                });
    }
}
