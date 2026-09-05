package qupath.ext.spclassify.io;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;
import java.util.zip.ZipFile;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class FileSystemUtilitiesTest {

    @Test
    void zipDirectoryArchivesAllFilesWithRelativeNames(@TempDir Path tmp) throws Exception {
        Path dir = Files.createDirectory(tmp.resolve("state"));
        Files.writeString(dir.resolve("a.json"), "{}", StandardCharsets.UTF_8);
        Path sub = Files.createDirectory(dir.resolve("images"));
        Files.writeString(sub.resolve("b.json"), "{}", StandardCharsets.UTF_8);

        Path zip = tmp.resolve("out.zip");
        Path written = FileSystemUtilities.zipDirectory(dir, zip);

        assertEquals(zip, written);
        Set<String> entries = new HashSet<>();
        try (ZipFile zf = new ZipFile(zip.toFile())) {
            zf.entries().asIterator().forEachRemaining(e -> entries.add(e.getName()));
        }
        assertTrue(entries.contains("a.json"));
        assertTrue(entries.contains("images/b.json"), "entries should use forward slashes: " + entries);
    }

    @Test
    void zipDirectoryReturnsNullForMissingOrEmptyDir(@TempDir Path tmp) throws Exception {
        assertNull(FileSystemUtilities.zipDirectory(tmp.resolve("nope"), tmp.resolve("a.zip")));
        Path empty = Files.createDirectory(tmp.resolve("empty"));
        Path zip = tmp.resolve("empty.zip");
        assertNull(FileSystemUtilities.zipDirectory(empty, zip));
        assertFalse(Files.exists(zip), "no archive should be left behind for an empty dir");
    }

    @Test
    void deleteDirectoryRecursivelyRemovesEverything(@TempDir Path tmp) throws Exception {
        Path dir = Files.createDirectory(tmp.resolve("tree"));
        Files.writeString(dir.resolve("f.txt"), "x", StandardCharsets.UTF_8);
        Files.createDirectory(dir.resolve("nested"));
        Files.writeString(dir.resolve("nested").resolve("g.txt"), "y", StandardCharsets.UTF_8);

        FileSystemUtilities.deleteDirectoryRecursively(dir);
        assertFalse(Files.exists(dir));
    }

    @Test
    void deleteDirectoryRecursivelyIsNoOpWhenMissing(@TempDir Path tmp) {
        assertDoesNotThrow(() -> FileSystemUtilities.deleteDirectoryRecursively(tmp.resolve("does-not-exist")));
    }
}
