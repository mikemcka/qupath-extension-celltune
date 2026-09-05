package qupath.ext.spclassify.util;

/**
 * Single source of truth for how many threads the ML training path may use.
 * <p>
 * XGBoost, LightGBM, the hyperparameter tuner and the resampler each used to call
 * {@link Runtime#availableProcessors()} independently, so nothing could see the total budget and
 * concurrent stages oversubscribed the machine. Routing them all through here means one place
 * decides, and a user on a shared workstation can cap it.
 * <p>
 * With no override set this returns {@code availableProcessors()}, exactly what the individual
 * call sites returned before — adopting it is a behaviour-preserving refactor.
 * <p>
 * Deliberately free of QuPath GUI types so it stays unit-testable and usable from the pure-array
 * classes; the persistent preference that drives {@link #setOverride(int)} is wired up in the
 * extension's startup code.
 */
public final class TrainingThreads {

    /** 0 means "auto" — use every available processor. */
    private static volatile int override = 0;

    private TrainingThreads() {} // utility class

    /**
     * Caps the training thread budget.
     *
     * @param n desired thread count; {@code 0} (or negative) restores automatic sizing, and
     *          values above {@link Runtime#availableProcessors()} are clamped down to it
     */
    public static void setOverride(int n) {
        override = n <= 0 ? 0 : Math.min(n, Runtime.getRuntime().availableProcessors());
    }

    /** @return the configured override, or {@code 0} when sizing automatically */
    public static int getOverride() {
        return override;
    }

    /** @return the total number of threads the training path may use; always {@code >= 1} */
    public static int total() {
        int o = override;
        return o > 0 ? o : Math.max(1, Runtime.getRuntime().availableProcessors());
    }

    /**
     * Splits the budget across stages that will run at the same time, so {@code k} concurrent
     * LightGBM fits don't each request every core.
     *
     * @param concurrentTasks how many tasks will run simultaneously
     * @return threads each task should use; always {@code >= 1}
     */
    public static int forConcurrentTasks(int concurrentTasks) {
        return Math.max(1, total() / Math.max(1, concurrentTasks));
    }
}
