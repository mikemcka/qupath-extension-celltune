package qupath.ext.spclassify.io;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;
import java.util.zip.ZipFile;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ProjectStateManagerResetTest {

    @Test
    void zipDirectoryArchivesAllFilesWithRelativeNames(@TempDir Path tmp) throws Exception {
        Path dir = tmp.resolve("celltune");
        Files.createDirectories(dir.resolve("image-labels"));
        Files.writeString(dir.resolve("classifier-state.json"), "{}", StandardCharsets.UTF_8);
        Files.writeString(dir.resolve("image-labels/slide1.json"), "{\"a\":\"T\"}", StandardCharsets.UTF_8);

        Path zip = tmp.resolve("backup.zip");
        Path written = ProjectStateManager.zipDirectory(dir, zip);

        assertEquals(zip, written);
        assertTrue(Files.exists(zip));

        Set<String> entries = new HashSet<>();
        try (ZipFile zf = new ZipFile(zip.toFile())) {
            zf.stream().forEach(e -> entries.add(e.getName()));
        }
        assertTrue(entries.contains("classifier-state.json"), entries.toString());
        assertTrue(entries.contains("image-labels/slide1.json"), entries.toString());
        assertEquals(2, entries.size());
    }

    @Test
    void zipDirectoryReturnsNullForMissingOrEmptyDir(@TempDir Path tmp) throws Exception {
        assertNull(ProjectStateManager.zipDirectory(tmp.resolve("nope"), tmp.resolve("a.zip")));

        Path empty = tmp.resolve("empty");
        Files.createDirectories(empty);
        Path zip = tmp.resolve("b.zip");
        assertNull(ProjectStateManager.zipDirectory(empty, zip));
        // No archive should be left behind when there was nothing to back up.
        assertFalse(Files.exists(zip));
    }

    @Test
    void deleteDirectoryRecursivelyRemovesEverything(@TempDir Path tmp) throws Exception {
        Path dir = tmp.resolve("celltune");
        Files.createDirectories(dir.resolve("binary/CD3"));
        Files.writeString(dir.resolve("state.json"), "{}", StandardCharsets.UTF_8);
        Files.writeString(dir.resolve("binary/CD3/slide.json"), "{}", StandardCharsets.UTF_8);

        ProjectStateManager.deleteDirectoryRecursively(dir);

        assertFalse(Files.exists(dir));
    }

    @Test
    void deleteDirectoryRecursivelyIsNoOpWhenMissing(@TempDir Path tmp) throws Exception {
        // Should not throw.
        ProjectStateManager.deleteDirectoryRecursively(tmp.resolve("does-not-exist"));
    }
}
