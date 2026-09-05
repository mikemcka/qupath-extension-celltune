package qupath.ext.spclassify.io;

import java.io.IOException;
import java.nio.file.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.spclassify.model.CellTypeTable;

/**
 * Imports a marker channel table from a CSV file.
 * <p>
 * Expected CSV format:
 * <pre>
 * CellType,Marker1,Marker2,Marker3
 * T-Cell,CD3,,
 * B-Cell,CD20,,
 * Macrophage,CD68,CD163,
 * </pre>
 * The first row is treated as a header and skipped.
 * Each subsequent row defines a cell type and up to 3 marker channel names.
 * Empty marker columns are ignored.
 */
public class MarkerTableImporter {

    private static final Logger logger = LoggerFactory.getLogger(MarkerTableImporter.class);

    private MarkerTableImporter() {} // utility class

    /**
     * Load a {@link CellTypeTable} from a CSV file.
     *
     * @param csvPath path to the marker table CSV
     * @return the parsed cell type table
     * @throws IOException if the file cannot be read or is malformed
     */
    public static CellTypeTable importFromCSV(Path csvPath) throws IOException {
        logger.info("Importing marker table from {}", csvPath);
        CellTypeTable table = CellTypeTable.loadFromCSV(csvPath);
        logger.info("Loaded {} cell types from marker table", table.size());
        if (table.isEmpty()) {
            logger.warn("Marker table is empty — no cell types were loaded from {}", csvPath);
        }
        return table;
    }
}
