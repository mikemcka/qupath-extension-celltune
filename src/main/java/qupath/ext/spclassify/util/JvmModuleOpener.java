package qupath.ext.spclassify.util;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.Field;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.misc.Unsafe;

/**
 * Opens {@code java.base/java.lang} to all unnamed modules at runtime — the
 * programmatic equivalent of launching the JVM with
 * {@code --add-opens=java.base/java.lang=ALL-UNNAMED}.
 * <p>
 * Smile's PCA ({@code Matrix.svd} → OpenBLAS) and UMAP (spectral init → ARPACK)
 * load their native libraries through JavaCPP, which reflects into
 * {@code java.lang.Runtime} to associate the library with the calling class
 * loader. Under the Java module system (Java 16+) that reflection is blocked
 * unless the package is opened, producing an
 * {@link java.lang.reflect.InaccessibleObjectException} wrapped in an
 * {@link ExceptionInInitializerError}. Opening the package here means users do
 * not have to edit JVM arguments for the cell scatter plot to work.
 * <p>
 * This MUST run before any Smile native class is initialized: once a native
 * class such as {@code org.bytedeco.arpackng.global.arpack} fails to initialize,
 * the JVM poisons it permanently (subsequent access throws
 * {@link NoClassDefFoundError} for the rest of the session), so opening the
 * module afterwards is too late.
 */
public final class JvmModuleOpener {

    private static final Logger logger = LoggerFactory.getLogger(JvmModuleOpener.class);

    /** null = not yet attempted; otherwise the cached outcome. */
    private static volatile Boolean javaLangOpen;

    private JvmModuleOpener() {}

    /**
     * Ensures {@code java.base/java.lang} is open to unnamed modules. Idempotent,
     * thread-safe, and never throws — failures are logged and reported via the
     * return value so callers can fall back gracefully.
     *
     * @return {@code true} if the package is open (already, or opened by this
     *     call); {@code false} if it could not be opened
     */
    public static synchronized boolean ensureJavaLangOpen() {
        if (javaLangOpen != null) {
            return javaLangOpen;
        }
        Module base = Object.class.getModule();
        Module self = JvmModuleOpener.class.getModule();
        if (base.isOpen("java.lang", self)) {
            javaLangOpen = Boolean.TRUE;
            return true;
        }
        boolean ok = false;
        try {
            openPackageToAllUnnamed(base, "java.lang");
            ok = base.isOpen("java.lang", self);
        } catch (Throwable t) {
            logger.debug("Runtime add-opens of java.base/java.lang failed", t);
        }
        javaLangOpen = ok;
        if (ok) {
            logger.info("Opened java.base/java.lang at runtime " + "(enables Smile native PCA/UMAP without JVM flags)");
        } else {
            logger.warn("Could not open java.base/java.lang at runtime; Smile "
                    + "native PCA/UMAP may be unavailable. If needed, launch "
                    + "QuPath with --add-opens=java.base/java.lang=ALL-UNNAMED.");
        }
        return ok;
    }

    /**
     * Invokes the non-public {@code Module.implAddOpensToAllUnnamed(String)} via
     * the trusted {@code IMPL_LOOKUP}, obtained through {@code sun.misc.Unsafe}.
     * This is the same approach used by Lombok/ByteBuddy and works on Java 9–25.
     */
    @SuppressWarnings("removal") // sun.misc.Unsafe accessors: present through JDK 25
    private static void openPackageToAllUnnamed(Module base, String pkg) throws Throwable {
        Field theUnsafe = Unsafe.class.getDeclaredField("theUnsafe");
        theUnsafe.setAccessible(true);
        Unsafe unsafe = (Unsafe) theUnsafe.get(null);

        Field implLookupField = MethodHandles.Lookup.class.getDeclaredField("IMPL_LOOKUP");
        MethodHandles.Lookup implLookup = (MethodHandles.Lookup)
                unsafe.getObject(unsafe.staticFieldBase(implLookupField), unsafe.staticFieldOffset(implLookupField));

        MethodHandle addOpens = implLookup.findVirtual(
                Module.class, "implAddOpensToAllUnnamed", MethodType.methodType(void.class, String.class));
        addOpens.invoke(base, pkg);
    }
}
