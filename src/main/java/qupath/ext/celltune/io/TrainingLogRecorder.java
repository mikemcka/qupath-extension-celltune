package qupath.ext.celltune.io;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.projects.Project;

/**
 * Writes a durable copy of a training run's log to {@code <project>/celltune/logs/}.
 * <p>
 * The on-screen training log lives only in a JavaFX {@code TextArea} that is discarded when the
 * progress window closes, so there is nothing to inspect after a slow run — and nothing at all
 * after an {@link OutOfMemoryError} or a JVM crash, which are exactly the cases worth
 * diagnosing. This tees the same {@code Consumer<String>} sink to a file.
 * <p>
 * Every line is flushed as it is written. That is deliberate: a buffered log that is lost on
 * crash defeats the purpose. Training emits on the order of hundreds of lines per run, so the
 * cost is irrelevant next to the work being logged.
 * <p>
 * Logging must never be the reason a training run fails. Construction falls back to a no-op
 * recorder when there is no project on disk, and every write swallows its {@link IOException}
 * after logging it once.
 */
public final class TrainingLogRecorder implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(TrainingLogRecorder.class);

    /** Directory name under {@code celltune/}. */
    private static final String LOGS_DIR = "logs";

    /** Number of most-recent log files to keep; older ones are pruned when a new run starts. */
    private static final int RETAIN_LOGS = 20;

    private final Path file;
    private final BufferedWriter writer;
    private boolean writeFailed = false;

    private TrainingLogRecorder(Path file, BufferedWriter writer) {
        this.file = file;
        this.writer = writer;
    }

    /**
     * Opens a recorder for a new training run.
     *
     * @param project the current project; may be {@code null} when an image is open outside a
     *                project, in which case a no-op recorder is returned
     * @return a recorder — never {@code null}, never throws
     */
    public static TrainingLogRecorder open(Project<?> project) {
        if (project == null || project.getPath() == null) {
            return new TrainingLogRecorder(null, null);
        }
        try {
            Path dir = ProjectStateManager.getCellTuneDir(project).resolve(LOGS_DIR);
            Files.createDirectories(dir);
            pruneOldLogs(dir);
            Path target =
                    dir.resolve("training-" + LocalDateTime.now().format(ProjectStateManager.TIMESTAMP_FMT) + ".log");
            BufferedWriter w = Files.newBufferedWriter(
                    target, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            logger.info("CellTune training log: {}", target);
            return new TrainingLogRecorder(target, w);
        } catch (IOException | RuntimeException ex) {
            logger.warn("Could not open a training log file; continuing without one", ex);
            return new TrainingLogRecorder(null, null);
        }
    }

    /** Deletes all but the {@value #RETAIN_LOGS} most recent {@code training-*.log} files. */
    private static void pruneOldLogs(Path dir) {
        try (Stream<Path> files = Files.list(dir)) {
            List<Path> logs = files.filter(p -> p.getFileName().toString().startsWith("training-"))
                    .filter(p -> p.getFileName().toString().endsWith(".log"))
                    .sorted(Comparator.comparing((Path p) -> p.getFileName().toString())
                            .reversed())
                    .toList();
            for (int i = RETAIN_LOGS; i < logs.size(); i++) {
                Files.deleteIfExists(logs.get(i));
            }
        } catch (IOException ex) {
            logger.debug("Could not prune old training logs in {}", dir, ex);
        }
    }

    /** @return the log file path, or {@code null} when no file is being written */
    public Path getFile() {
        return file;
    }

    /** @return true if this recorder is writing to a file */
    public boolean isRecording() {
        return writer != null && !writeFailed;
    }

    /**
     * Writes a header block identifying the run and its environment, so a log a user sends back
     * is self-contained. Values are supplied by the caller because this class deliberately knows
     * nothing about the classifier.
     *
     * @param title       a short run title, e.g. {@code "Multi-class training"}
     * @param settings    ordered {@code label -> value} pairs describing the run configuration
     */
    public void writeHeader(String title, List<String[]> settings) {
        Runtime rt = Runtime.getRuntime();
        accept("=".repeat(72));
        accept("CellTune training log — " + title);
        accept("Started      : " + LocalDateTime.now());
        accept("Java         : " + System.getProperty("java.version") + " (" + System.getProperty("os.name") + ")");
        accept("Processors   : " + rt.availableProcessors());
        accept("Max heap     : " + toMiB(rt.maxMemory()) + " MB");
        for (String[] kv : settings) {
            accept(String.format("%-13s: %s", kv[0], kv[1]));
        }
        accept("=".repeat(72));
    }

    private static long toMiB(long bytes) {
        return bytes / (1024L * 1024L);
    }

    /**
     * Appends one line. Safe to call from any thread and after a write failure; never throws.
     */
    public synchronized void accept(String message) {
        if (writer == null || writeFailed) {
            return;
        }
        try {
            writer.write(message == null ? "" : message);
            writer.newLine();
            writer.flush(); // survive an OOM or a hard kill
        } catch (IOException ex) {
            writeFailed = true;
            logger.warn("Training log write failed; further lines will not be recorded", ex);
        }
    }

    /**
     * Wraps an existing log sink so messages go to both it and this file.
     *
     * @param delegate the original sink (e.g. the one appending to the on-screen text area);
     *                 may be {@code null}
     * @return a sink that writes to both
     */
    public Consumer<String> tee(Consumer<String> delegate) {
        return msg -> {
            if (delegate != null) {
                delegate.accept(msg);
            }
            accept(msg);
        };
    }

    /**
     * Records a failure with its stack trace. Called for both ordinary exceptions and
     * {@link OutOfMemoryError}, so it takes {@link Throwable}.
     */
    public synchronized void recordFailure(String context, Throwable error) {
        accept("");
        accept("!! FAILED during: " + context);
        accept("!! " + error.getClass().getName() + ": " + error.getMessage());
        for (StackTraceElement el : error.getStackTrace()) {
            accept("!!     at " + el);
        }
        Throwable cause = error.getCause();
        if (cause != null && cause != error) {
            accept("!! caused by " + cause.getClass().getName() + ": " + cause.getMessage());
        }
    }

    @Override
    public synchronized void close() {
        if (writer == null) {
            return;
        }
        try {
            writer.flush();
            writer.close();
        } catch (IOException ex) {
            logger.debug("Failed to close training log", ex);
        }
    }

    /** Convenience for building the {@link #writeHeader} settings list. */
    public static List<String[]> settings(Object... labelValuePairs) {
        List<String[]> out = new ArrayList<>();
        for (int i = 0; i + 1 < labelValuePairs.length; i += 2) {
            out.add(new String[] {String.valueOf(labelValuePairs[i]), String.valueOf(labelValuePairs[i + 1])});
        }
        return out;
    }
}
