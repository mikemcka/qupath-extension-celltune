package qupath.ext.spclassify.io;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * Small, dependency-free filesystem helpers used by the project-state IO layer.
 * Extracted from {@code ProjectStateManager} so the pure file operations can be
 * reused and unit-tested in isolation.
 */
public final class FileSystemUtilities {

    private FileSystemUtilities() {} // utility class

    /**
     * Zip every file under {@code dir} into {@code zipTarget}, using paths
     * relative to {@code dir} (with forward slashes) as entry names.
     *
     * @return {@code zipTarget} if at least one file was written, or {@code null}
     *         if {@code dir} is not a directory or is empty (no archive is left
     *         behind in the empty case)
     */
    public static Path zipDirectory(Path dir, Path zipTarget) throws IOException {
        if (dir == null || !Files.isDirectory(dir)) {
            return null;
        }
        boolean wroteEntry = false;
        try (ZipOutputStream zos = new ZipOutputStream(Files.newOutputStream(zipTarget))) {
            try (var stream = Files.walk(dir)) {
                for (Path p : (Iterable<Path>) stream::iterator) {
                    if (Files.isDirectory(p)) {
                        continue;
                    }
                    String rel = dir.relativize(p).toString().replace('\\', '/');
                    zos.putNextEntry(new ZipEntry(rel));
                    Files.copy(p, zos);
                    zos.closeEntry();
                    wroteEntry = true;
                }
            }
        }
        if (!wroteEntry) {
            Files.deleteIfExists(zipTarget);
            return null;
        }
        return zipTarget;
    }

    /** Recursively delete a directory tree. A no-op if {@code dir} does not exist. */
    public static void deleteDirectoryRecursively(Path dir) throws IOException {
        if (dir == null || !Files.exists(dir)) {
            return;
        }
        try (var stream = Files.walk(dir)) {
            // Deepest paths first so directories are emptied before removal.
            for (Path p : stream.sorted(Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(p);
            }
        }
    }
}
