package qupath.ext.spclassify.ui;

import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.scene.input.Clipboard;
import javafx.scene.input.ClipboardContent;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.Modality;
import javafx.stage.Stage;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;

/**
 * "How to cite" dialog: a copyable, always-available reference list so users can see their
 * citation obligations without hunting through the README. Lists the software itself, the
 * CellTune paper the workflow derives from (always cite), and the method papers whose
 * algorithms specific features reimplement (cite the ones whose feature you used).
 *
 * <p>The same information — in machine-readable form — lives in {@code CITATION.cff}.
 */
public final class CitationDialog {

    private CitationDialog() {} // utility class

    private static final String CITATIONS = """
            How to cite SP Classify
            =======================

            If you use SP Classify in your work, please cite BOTH the software and the
            CellTune paper it derives from, plus any method papers for the specific
            features you used.

            -- Software --
            Mckay, M. SP Classify: an active-learning cell classifier extension for QuPath.
            Zenodo. https://doi.org/10.5281/zenodo.21782421

            -- Always cite (the human-in-the-loop workflow this extension derives from) --
            CellTune (Keren Lab). Nature Methods (2026).
            https://doi.org/10.1038/s41592-026-03162-2

            -- Cite if you used the corresponding feature --
            * Cellular Neighborhoods (CN):
                Schurch CM, Bhate SS, Barlow GL, et al. Cell 182(5):1341-1359 (2020).
                https://doi.org/10.1016/j.cell.2020.07.005

            * Graph-based (Leiden) clustering:
                Traag VA, Waltman L, van Eck NJ. Sci Rep 9:5233 (2019).
                https://doi.org/10.1038/s41598-019-41695-z
                Pipeline mirrors the scanpy recipe:
                Wolf FA, Angerer P, Theis FJ. Genome Biol 19:15 (2018).
                https://doi.org/10.1186/s13059-017-1382-0

            * Cohort / all-cells Leiden (HNSW approximate-NN graph):
                Malkov YA, Yashunin DA. IEEE TPAMI 42(4):824-836 (2020).
                https://doi.org/10.1109/TPAMI.2018.2889473

            * arcsinh cofactor (0.05) for MIBI mass-spectrometry data:
                Hartmann FJ, Mrdjen D, McCaffrey E, et al. Nat Biotechnol 39:186-197 (2021).
                https://doi.org/10.1038/s41587-020-0651-8

            * Batch Normalisation (UniFORM):
                Wang K, Ait-Ahmad K, Kupp S, et al. bioRxiv 2024.12.06.626879 (2024);
                Cell Reports Methods (2025).
                https://doi.org/10.1101/2024.12.06.626879

            A machine-readable version is in CITATION.cff in the repository.
            """;

    /** Show the non-modal citation dialog, owned by the QuPath window. */
    public static void show(QuPathGUI qupath) {
        Stage dialog = new Stage();
        if (qupath != null && qupath.getStage() != null) {
            dialog.initOwner(qupath.getStage());
        }
        dialog.initModality(Modality.NONE);
        dialog.setTitle("SP Classify - How to Cite");

        Label header = new Label("Please cite the software and the CellTune paper, plus any "
                + "method papers for the features you used.");
        header.setWrapText(true);

        TextArea text = new TextArea(CITATIONS);
        text.setEditable(false);
        text.setWrapText(true);
        text.setStyle("-fx-font-family: monospace;");
        text.setPrefRowCount(24);
        text.setPrefColumnCount(72);
        VBox.setVgrow(text, Priority.ALWAYS);

        Button copyButton = new Button("Copy all");
        copyButton.setOnAction(e -> {
            ClipboardContent content = new ClipboardContent();
            content.putString(CITATIONS);
            Clipboard.getSystemClipboard().setContent(content);
            Dialogs.showInfoNotification("SP Classify", "Citations copied to clipboard.");
        });

        Button closeButton = new Button("Close");
        closeButton.setOnAction(e -> dialog.close());

        HBox buttons = new HBox(10, copyButton, closeButton);
        buttons.setPadding(new Insets(10, 0, 0, 0));

        VBox root = new VBox(10, header, text, buttons);
        root.setPadding(new Insets(15));

        dialog.setScene(new Scene(root, 620, 560));
        dialog.show();
    }
}
