package qupath.ext.celltune.util;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Factory for the daemon-threaded background executors used across the
 * extension's UI. Centralises the previously-repeated "named daemon thread
 * factory" boilerplate so every background pool is consistently named (for
 * thread dumps) and marked daemon (so a stray pool never blocks JVM shutdown).
 * <p>
 * Callers own the returned {@link ExecutorService} and remain responsible for
 * shutting it down.
 */
public final class BackgroundExecutors {

    private BackgroundExecutors() {} // utility class

    /**
     * A {@link ThreadFactory} producing daemon threads with the given base name.
     * Threads are named {@code <baseName>} (single) or, for pools, the base name
     * is reused for every worker — matching the prior per-class behaviour.
     */
    public static ThreadFactory daemonThreadFactory(String baseName) {
        return r -> {
            Thread t = new Thread(r, baseName);
            t.setDaemon(true);
            return t;
        };
    }

    /** Single-threaded executor backed by one daemon thread named {@code name}. */
    public static ExecutorService newSingleThread(String name) {
        return Executors.newSingleThreadExecutor(daemonThreadFactory(name));
    }

    /** Fixed-size pool of {@code nThreads} daemon threads named {@code name}. */
    public static ExecutorService newFixedPool(int nThreads, String name) {
        return Executors.newFixedThreadPool(nThreads, daemonThreadFactory(name));
    }

    /**
     * A pool of at most {@code nThreads} daemon threads that lets idle workers die off.
     * <p>
     * For a pool that is used in short bursts and then reused: {@link #newFixedPool} would keep
     * every worker parked between bursts, and creating a fresh fixed pool per burst pays thread
     * construction every time and — worse — lets concurrent callers each hold a full-width pool,
     * so nothing bounds the total. This is long-lived and shared instead, with a keep-alive that
     * releases the threads once the work stops.
     * <p>
     * Unlike the other factories here, a caller does <em>not</em> shut this down: it is meant to
     * be a shared singleton, and the threads are daemon, so an idle pool costs nothing and never
     * blocks JVM shutdown.
     */
    public static ExecutorService newElasticPool(int nThreads, String name) {
        ThreadPoolExecutor exec = new ThreadPoolExecutor(
                nThreads, nThreads, 60L, TimeUnit.SECONDS, new LinkedBlockingQueue<>(), daemonThreadFactory(name));
        exec.allowCoreThreadTimeOut(true);
        return exec;
    }
}
