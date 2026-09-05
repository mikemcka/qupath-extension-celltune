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
import qupath.ext.spclassify.model.CellTypeTable;
import qupath.lib.projects.Project;

class ProjectStateManagerMarkerTableTest {

    @TempDir
    Path tempDir;

    @Test
    void savesAndLoadsSimpleFormatRoundTrip() throws Exception {
        Project<BufferedImage> project = fakeProject(tempDir.resolve("simple-project/project.qpproj"));

        CellTypeTable original = new CellTypeTable();
        original.put("T-Cell", List.of("CD3"));
        original.put("Macrophage", List.of("CD68", "CD163"));

        ProjectStateManager.saveMarkerTable(project, original);
        CellTypeTable loaded = ProjectStateManager.loadMarkerTable(project);

        assertNotNull(loaded);
        assertFalse(loaded.hasGatingRules());
        assertEquals(original.getCellTypes(), loaded.getCellTypes());
        assertEquals(List.of("CD3"), loaded.getMarkers("T-Cell"));
        assertEquals(List.of("CD68", "CD163"), loaded.getMarkers("Macrophage"));
    }

    @Test
    void savesAndLoadsRuleFormatRoundTrip() throws Exception {
        Project<BufferedImage> project = fakeProject(tempDir.resolve("rule-project/project.qpproj"));

        CellTypeTable original = new CellTypeTable();
        original.putRule("CD8T", "CD8", "CD3", "CD103|CD45|CD45RA");
        original.putRule("Plasma_CD38", "CD38&!IgA", null, "CD45|VIM");

        ProjectStateManager.saveMarkerTable(project, original);
        CellTypeTable loaded = ProjectStateManager.loadMarkerTable(project);

        assertNotNull(loaded);
        assertTrue(loaded.hasGatingRules());
        assertEquals(original.getCellTypes(), loaded.getCellTypes());

        assertEquals("CD8", loaded.getPrimaryExpression("CD8T"));
        assertEquals("CD3", loaded.getSecondaryMarkers("CD8T"));
        assertEquals("CD103|CD45|CD45RA", loaded.getTertiaryMarkers("CD8T"));

        assertEquals("CD38&!IgA", loaded.getPrimaryExpression("Plasma_CD38"));
        assertNull(loaded.getSecondaryMarkers("Plasma_CD38"));
        assertEquals("CD45|VIM", loaded.getTertiaryMarkers("Plasma_CD38"));
    }

    @Test
    void returnsNullWhenMarkerTableDoesNotExist() {
        Project<BufferedImage> project = fakeProject(tempDir.resolve("empty-project/project.qpproj"));
        assertNull(ProjectStateManager.loadMarkerTable(project));
    }

    @Test
    void savingNullOrEmptyClearsExistingFile() throws Exception {
        Project<BufferedImage> project = fakeProject(tempDir.resolve("clear-project/project.qpproj"));

        CellTypeTable original = new CellTypeTable();
        original.put("T-Cell", List.of("CD3"));
        ProjectStateManager.saveMarkerTable(project, original);
        assertNotNull(ProjectStateManager.loadMarkerTable(project));

        ProjectStateManager.saveMarkerTable(project, null);
        assertNull(ProjectStateManager.loadMarkerTable(project));

        ProjectStateManager.saveMarkerTable(project, original);
        assertNotNull(ProjectStateManager.loadMarkerTable(project));

        ProjectStateManager.saveMarkerTable(project, new CellTypeTable());
        assertNull(ProjectStateManager.loadMarkerTable(project));
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
