package qupath.ext.spclassify.io;

/**
 * Shared CSV field-escaping helpers for the extension's exporters.
 * <p>
 * Two escaping policies are offered because the exporters historically chose
 * differently and changing either would alter on-disk output:
 * <ul>
 *   <li>{@link #quoteIfNeeded(String)} — RFC-4180 minimal quoting: only wrap a
 *       field in quotes when it contains a comma, quote, or newline. Used by the
 *       cell-table export.</li>
 *   <li>{@link #quoteAlways(String)} — always wrap in quotes. Used by the
 *       project-summary and pixel-stats exports.</li>
 * </ul>
 * Both double any embedded quote ({@code "} → {@code ""}) and render
 * {@code null} as an empty (unquoted) field.
 */
public final class CsvUtils {

    private CsvUtils() {} // utility class

    /**
     * Quote a CSV field only if it contains a comma, double-quote, or newline.
     * {@code null} or empty input yields an empty string.
     */
    public static String quoteIfNeeded(String value) {
        if (value == null || value.isEmpty()) return "";
        if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
            return "\"" + value.replace("\"", "\"\"") + "\"";
        }
        return value;
    }

    /**
     * Always wrap the field in double-quotes (doubling any embedded quote).
     * {@code null} input yields an empty string.
     */
    public static String quoteAlways(String value) {
        if (value == null) {
            return "";
        }
        return '"' + value.replace("\"", "\"\"") + '"';
    }
}
