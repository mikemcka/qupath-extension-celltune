package qupath.ext.spclassify.io;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class CsvUtilsTest {

    @Test
    void quoteIfNeededLeavesPlainValuesUnquoted() {
        assertEquals("Tumor", CsvUtils.quoteIfNeeded("Tumor"));
        assertEquals("", CsvUtils.quoteIfNeeded(""));
        assertEquals("", CsvUtils.quoteIfNeeded(null));
    }

    @Test
    void quoteIfNeededQuotesOnlyWhenSpecialCharsPresent() {
        assertEquals("\"a,b\"", CsvUtils.quoteIfNeeded("a,b"));
        assertEquals("\"a\"\"b\"", CsvUtils.quoteIfNeeded("a\"b"));
        assertEquals("\"a\nb\"", CsvUtils.quoteIfNeeded("a\nb"));
    }

    @Test
    void quoteAlwaysWrapsEverything() {
        assertEquals("\"Tumor\"", CsvUtils.quoteAlways("Tumor"));
        assertEquals("\"\"", CsvUtils.quoteAlways(""));
        assertEquals("", CsvUtils.quoteAlways(null));
    }

    @Test
    void quoteAlwaysDoublesEmbeddedQuotes() {
        assertEquals("\"a\"\"b\"", CsvUtils.quoteAlways("a\"b"));
        assertEquals("\"a,b\"", CsvUtils.quoteAlways("a,b"));
    }
}
