package qupath.ext.celltune.ui;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import javafx.application.Platform;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.IntegerProperty;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Insets;
import javafx.geometry.Orientation;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.CheckMenuItem;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.MenuButton;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.ProgressIndicator;
import javafx.scene.control.RadioButton;
import javafx.scene.control.Separator;
import javafx.scene.control.Spinner;
import javafx.scene.control.TextField;
import javafx.scene.control.ToggleButton;
import javafx.scene.control.ToggleGroup;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.FileChooser;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javax.imageio.ImageIO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.celltune.BatchCorrection;
import qupath.ext.celltune.model.AnnRecallException;
import qupath.ext.celltune.model.CellFeatureExtractor;
import qupath.ext.celltune.model.CohortClusterModel;
import qupath.ext.celltune.model.FeatureNormalizer;
import qupath.ext.celltune.model.LeidenModel;
import qupath.ext.celltune.model.PopulationSet;
import qupath.ext.celltune.model.ScatterMath;
import qupath.ext.celltune.util.JvmModuleOpener;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.color.ColorMaps;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.prefs.PathPrefs;
import qupath.lib.gui.tools.MeasurementMapper;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjectFilter;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.objects.hierarchy.events.PathObjectSelectionListener;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;
import qupath.lib.roi.interfaces.ROI;
import smile.clustering.KMeans;
import smile.math.MathEx;

/**
 * Interactive 2D embedding scatter plot of cell detections.
 * <p>
 * Each cell is projected into a 2D dimensionality-reduction embedding (PCA or
 * UMAP) of its selected marker-mean measurements and drawn as a dot, coloured by
 * k-means cluster, predicted class, or a single marker's intensity. The user can
 * box- or lasso-select a region of the plot to select those cells in the QuPath
 * viewer; conversely, selecting cells in the viewer outlines them on the plot.
 * <p>
 * Rendering follows the canvas-based idiom of {@link IntensityHeatmapView} /
 * {@link ConfusionMatrixView}. Embedding and clustering run on a background
 * thread (they can take tens of seconds on large images) and marshal back to the
 * FX thread via {@link Platform#runLater}. SMILE supplies PCA, UMAP and k-means.
 */
public class ScatterPlotView {

    private static final Logger logger = LoggerFactory.getLogger(ScatterPlotView.class);

    /**
     * UMAP builds a k-NN graph and optimises a layout — expensive on very large
     * point sets. Above this count we embed a random subsample (other cells are
     * left unplotted, with a notice). PCA and k-means always run on all cells.
     */
    private static final int MAX_UMAP_CELLS = 20_000;

    private enum Embedding {
        PCA,
        UMAP
    }

    /** Where the plotted rows come from: the open image, or a project sample. */
    private enum Scope {
        CURRENT_IMAGE,
        PROJECT
    }

    /** Clustering algorithm: fixed-k k-means, or resolution-driven graph-based Leiden. */
    private enum ClusterMethod {
        KMEANS,
        LEIDEN
    }

    /** Feature-space kNN graph neighbours for Leiden (hidden default; scanpy-style). */
    private static final int LEIDEN_GRAPH_K = 15;
    /** Leiden random-starts when the reproducibility toggle is OFF. */
    private static final int LEIDEN_SINGLE_START = 1;
    /** Leiden random-starts when the reproducibility toggle is ON (mirrors k-means n_init). */
    private static final int LEIDEN_REPRODUCIBLE_STARTS = 10;
    /** Fixed seed used when the reproducibility toggle is ON, for run-to-run identical labels. */
    private static final long LEIDEN_SEED = 42L;
    /** k-means restarts (n_init) when the reproducibility toggle is ON; keeps the lowest-distortion fit. */
    private static final int KMEANS_MULTI_RESTARTS = 10;
    /** Fixed seed for the reproducible k-means restarts (mirrors {@link #LEIDEN_SEED}). */
    private static final long KMEANS_SEED = 42L;
    /**
     * Soft ceiling (D-10) on the estimated pooled cell count before an all-cells cohort run
     * triggers an extra confirm dialog. Configurable (persisted), not a hard block — a run above
     * this estimate can still proceed if the user confirms.
     */
    private static final IntegerProperty ALL_CELLS_SOFT_CEILING =
            PathPrefs.createPersistentPreference("celltune.allCellsSoftCeiling", 50_000_000);
    /**
     * Whether conditional PCA reduction runs before the clustering kNN graph (Leiden and
     * k-means both) — on by default, mirroring scanpy's {@code scale -> PCA -> neighbors}
     * recipe. Persisted so the choice survives across sessions.
     */
    private static final BooleanProperty PCA_ENABLED_PREF =
            PathPrefs.createPersistentPreference("celltune.clusterPcaEnabled", true);
    /** PCA components to keep when {@link #PCA_ENABLED_PREF} is on; persisted alongside it. */
    private static final IntegerProperty PCA_COMPONENTS_PREF = PathPrefs.createPersistentPreference(
            "celltune.clusterPcaComponents", ScatterMath.PCA_DEFAULT_MAX_COMPONENTS);
    // ── Per-row state (all arrays aligned by index 0..nRows-1; rebuilt on toggle)
    private Scope scope = Scope.CURRENT_IMAGE;
    private int nRows; // number of plotted rows
    private PathObject[] cells; // live cells (null in PROJECT scope)
    private String[] rowClass; // class name per row (PROJECT scope)
    private String[] rowImage; // source image per row (PROJECT scope)
    private final List<String> markerFeatures;
    private double[][] raw; // [nRows][nFeatures] raw marker values
    private double[][] std; // [nRows][nFeatures] z-scored columns
    private double[] ex; // embedding x (NaN = not embedded)
    private double[] ey; // embedding y
    private int[] cluster; // k-means label per row (-1 = none)
    private boolean[] selected; // mirrors the viewer selection / plot highlight
    private IdentityHashMap<PathObject, Integer> indexOf; // null in PROJECT scope

    // ── Latest fit, retained for the cohort assign + the centroid heatmap ──────
    private double[] fitMean; // per-marker mean over the active rows
    private double[] fitSd; // per-marker sd over the active rows
    private double[][] fitCentroids; // [k][selMarkers] z-scored centroids, ALWAYS marker space (heatmap display)
    // Per-cluster centroids in the SAME space the k-means partition was actually computed in — the
    // PC-reduced space when the fit applied conditional PCA, otherwise identical to fitCentroids
    // (marker space). Used ONLY for k-means cohort assignment (assignAcrossProject/
    // writeClusterAcrossProject), never for the Assign-dialog heatmap — that must stay marker-space/
    // interpretable and keeps reading fitCentroids. Null before any k-means fit, or after a Leiden
    // fit/all-cells write (only consulted on the k-means cohort-assign path).
    private double[][] fitAssignCentroids;
    private List<String> fitMarkers; // selected markers the fit used (column order)
    private String fitClassFilter; // within-class filter active at fit time
    // Effective cluster count of the latest fit: kSpinner's value for k-means, or
    // Leiden's own decided community count. Sizes clusterCounts/fitCentroids/mapping
    // uniformly for both methods (Leiden picks this; the user does not).
    private int fitNClusters;
    // Reference used by the Leiden cohort-assign path (kNN label transfer): the
    // fitted sample's z-scored active rows + their Leiden labels. Null for k-means
    // (which uses fitCentroids/nearestCentroid instead) or before any Leiden fit.
    private double[][] fitLeidenReference;
    private int[] fitLeidenReferenceLabels;
    // Reusable projector into the PC space fitLeidenReference lives in, when the latest
    // Recompute applied conditional PCA reduction (Requirement 3) — identity/no-op when PCA
    // was off or skipped by the feature-count threshold. Applied to every NEW z-scored query
    // row (marker space) immediately before a Leiden kNN label transfer, so the query and the
    // fitted reference always share one basis. Retained regardless of method (only consulted
    // when fitLeidenReference is non-null, i.e. after a Leiden fit).
    private java.util.function.UnaryOperator<double[][]> fitPcaProjector;
    private boolean fitPcaApplied;
    // Per-cluster pooled cell counts from the last completed all-cells write
    // (CohortClusterModel.AllCellsResult.clusterCounts()), overriding clusterCounts(k)'s
    // preview-row-derived counts so the Assign dialog shows the all-cells cluster sizes
    // (bug fix, LEI-09 companion). Null before any all-cells write, or after the next
    // preview Recompute clears it (see clearFit()).
    private int[] fitAllCellsCounts;
    // True after a successful (non-aborted, non-cancelled) "Cluster all cells" write, until
    // the next preview Recompute. While true, cluster->class assignment in PROJECT scope must
    // read each cell's WRITTEN Cluster measurement (CohortClusterModel.assignAcrossProjectByMeasurement)
    // instead of re-transferring against the preview Leiden reference, which no longer matches
    // what was actually persisted to the cohort.
    private boolean allCellsWriteActive;

    // ── Project scope ──────────────────────────────────────────────────────────
    private List<String> projectImages = new ArrayList<>();

    private final QuPathGUI qupath;
    private final String imageName;
    private final PopulationSet predictions; // may be null
    private final PathObjectHierarchy hierarchy;
    private final Runnable openClassControl; // opens the Class Control dialog (nullable)
    // Feature normalizer applied during extraction so clustering/embedding see the
    // same transformed values the classifier does. Captured at construction so the
    // sample fit and the cohort assign always use the same transform (nullable).
    private final FeatureNormalizer normalizer;

    // ── UI ─────────────────────────────────────────────────────────────────────
    private final Stage stage;
    // Assigned once during construction, after the controls its PlotModel /
    // lasso-mode lambdas read; non-final so the colour/marker combo handlers
    // (built earlier) can reference it.
    private ScatterPlotCanvas plot;
    private final ComboBox<Embedding> embeddingCombo;
    private final CheckBox fullUmapCheck;
    private final ComboBox<ClusterMethod> methodCombo;
    // Cohort-mode radio pair (D-04/D-05): project scope + Method=Leiden only. Net visibility is
    // driven by both the methodCombo listener and applyScopeOverrides() (see
    // updateCohortModeVisibility()).
    private final RadioButton cohortModeAllCells;
    private final RadioButton cohortModeTransfer;
    // Visible only during an all-cells run (D-12); cancels the in-flight CancellationToken.
    private final Button cancelAllCellsBtn;
    // The token for the in-flight all-cells run, if any (null otherwise) — set at the start of
    // writeClusterAllCellsAcrossProject() so cancelAllCellsBtn's action can reach it. Written on
    // the FX thread (set here, cleared in the FX-thread runLater alongside hiding the Cancel
    // button) and read on the FX thread (Cancel button's action); volatile guards against any
    // stale read if that ever changes.
    private volatile CohortClusterModel.CancellationToken allCellsToken;
    private final Spinner<Integer> kSpinner;
    private final Spinner<Double> resolutionSpinner;
    private final CheckBox reproducibleCheck;
    // Conditional PCA controls (Requirement 6): visible for both k-means and Leiden, unlike
    // kSpinner/resolutionSpinner which toggle with the method. Persisted via PCA_ENABLED_PREF /
    // PCA_COMPONENTS_PREF.
    private final CheckBox pcaEnabledCheck;
    private final Spinner<Integer> pcaComponentsSpinner;
    private final Label kLabel;
    private final Label resolutionLabel;
    private final ComboBox<ScatterPlotCanvas.ColorMode> colorCombo;
    private final ComboBox<String> markerCombo;
    private final TextField annotationField;
    private final ComboBox<String> classField;
    private final MenuButton clusterMarkersBtn;
    private final List<CheckMenuItem> clusterMarkerItems = new ArrayList<>();
    private final ToggleButton boxToggle;
    private final ToggleButton lassoToggle;
    private final Button applyClustersBtn;
    // Non-destructive viewer-overlay toggles (CN-dialog parity): paint the open image's
    // cells by their assigned phenotype cluster, or revert to QuPath's native class colours.
    private final Button clusterOverlayBtn;
    private final Button classViewBtn;
    private boolean clusterOverlayActive;
    // True when the written Cluster measurement no longer matches the current fit (set on
    // every Recompute / fit change). While false, toggling "By cluster" just re-activates the
    // overlay display instead of re-running the (potentially cohort-wide, saving) assignment.
    private boolean clusterMeasurementStale = true;
    private final ProgressIndicator progress;
    // A wide, prominent mirror of {@link #progress} in the status bar, so long
    // assign/apply runs (which lock the controls) show clear, legible progress
    // rather than only the small toolbar spinner.
    private final ProgressBar progressBar;
    private final Label statusLabel;

    // ── Scope (current image ↔ project) controls ───────────────────────────────
    private final ToggleButton imageScopeToggle;
    private final ToggleButton projectScopeToggle;
    private final Spinner<Integer> sampleSpinner;
    private final Button imagesBtn;
    private final Button reSampleBtn;
    private final HBox projectControls;
    private final HBox sampleControls;

    private PathObjectSelectionListener selectionListener;
    private boolean updatingSelection = false; // guard against self-triggered redraws
    private volatile boolean applying = false; // guard against concurrent apply runs
    private String statusNotice = ""; // scope/filter suffix, persisted across redraws

    /**
     * @param owner          parent stage
     * @param qupath         QuPath GUI (for viewer access + selection)
     * @param imageName      display name of the current image
     * @param markerFeatures ordered marker-mean measurement names to embed/cluster
     * @param cellList       detection objects to plot
     * @param predictions    population set for "colour by predicted class" (nullable)
     * @param openClassControl opens the Class Control dialog from the assign dialog
     *                         so users can add/delete classes (nullable)
     * @param normalizer       feature normalizer applied during extraction so
     *                         clustering matches the classifier (nullable)
     */
    public ScatterPlotView(
            Stage owner,
            QuPathGUI qupath,
            String imageName,
            List<String> markerFeatures,
            List<PathObject> cellList,
            PopulationSet predictions,
            Runnable openClassControl,
            FeatureNormalizer normalizer,
            Runnable onNewPlot) {
        this.qupath = qupath;
        this.imageName = imageName != null ? imageName : "Current Image";
        this.markerFeatures = List.copyOf(markerFeatures);
        this.predictions = predictions;
        this.openClassControl = openClassControl;
        this.normalizer = normalizer;
        this.hierarchy = qupath.getImageData() != null ? qupath.getImageData().getHierarchy() : null;

        // Per-row data arrays are filled by loadCurrentImageData(...) below, once
        // the controls exist (so the initial recompute can read them).

        // ── Build controls ─────────────────────────────────────────────────────
        stage = new Stage();

        // Created early so control listeners (e.g. embedding switch) can update it.
        statusLabel = new Label("");
        statusLabel.setPadding(new Insets(6, 8, 6, 8));

        embeddingCombo = new ComboBox<>();
        embeddingCombo.getItems().addAll(Embedding.PCA, Embedding.UMAP);
        embeddingCombo.setValue(Embedding.PCA);
        embeddingCombo.setTooltip(
                new javafx.scene.control.Tooltip("PCA is fast and linear. UMAP often separates phenotypes better "
                        + "but is MUCH slower — expect a noticeable wait on large cell "
                        + "counts (and longer still with “Full UMAP”)."));

        fullUmapCheck = new CheckBox("Full UMAP");
        fullUmapCheck.setTooltip(new javafx.scene.control.Tooltip("Embed ALL cells in UMAP instead of a "
                + String.format("%,d", MAX_UMAP_CELLS)
                + "-cell sample. Much slower and more memory-hungry on large "
                + "images, but plots every cell. Only affects UMAP — k-means "
                + "already clusters all cells regardless."));
        fullUmapCheck.setDisable(embeddingCombo.getValue() != Embedding.UMAP);
        embeddingCombo.valueProperty().addListener((o, a, b) -> {
            fullUmapCheck.setDisable(b != Embedding.UMAP);
            if (b == Embedding.UMAP) {
                statusLabel.setText(
                        "UMAP is much slower than PCA on large cell " + "counts — click “Recompute” when ready.");
            }
        });

        kSpinner = new Spinner<>(2, 50, 8);
        kSpinner.setEditable(true);
        kSpinner.setPrefWidth(70);

        methodCombo = new ComboBox<>();
        methodCombo.getItems().addAll(ClusterMethod.KMEANS, ClusterMethod.LEIDEN);
        methodCombo.setValue(ClusterMethod.KMEANS);
        methodCombo.setTooltip(new javafx.scene.control.Tooltip(
                "k-means needs a fixed cluster count (k) and assumes roughly spherical, "
                        + "equal-size clusters. Leiden is graph-based: it builds a "
                        + "feature-space nearest-neighbour graph and lets a resolution "
                        + "parameter decide the cluster count — no fixed k, better at "
                        + "finding rare/irregular populations (the scanpy/scimap/SPACEc "
                        + "phenotyping standard)."));

        kLabel = new Label("Clusters (k):");

        resolutionSpinner = new Spinner<>(0.1, 3.0, 1.0, 0.1);
        resolutionSpinner.setEditable(true);
        resolutionSpinner.setPrefWidth(70);
        resolutionSpinner.setTooltip(
                new javafx.scene.control.Tooltip("Leiden resolution: higher values find more, smaller communities; "
                        + "lower values find fewer, larger ones. Unlike k-means there is "
                        + "no fixed cluster count — Leiden decides it from the graph."));
        resolutionSpinner.focusedProperty().addListener((o, was, isNow) -> {
            if (!isNow) {
                commitSpinnerEditor(resolutionSpinner, 0.1, 3.0);
            }
        });
        resolutionLabel = new Label("Resolution:");

        reproducibleCheck = new CheckBox("Sample multiple seeds");
        reproducibleCheck.setTooltip(new javafx.scene.control.Tooltip(
                "Run clustering from several random starts and keep the best-quality partition "
                        + "(lowest k-means distortion / best Leiden CPM), so repeated runs with the "
                        + "same settings give identical, higher-quality clusters. Off runs a single, "
                        + "faster pass whose result may vary run to run. Applies to both k-means and "
                        + "Leiden."));

        pcaEnabledCheck = new CheckBox("Reduce dims (PCA)");
        pcaEnabledCheck.setSelected(PCA_ENABLED_PREF.get());
        pcaEnabledCheck.setTooltip(new javafx.scene.control.Tooltip(
                "Project onto principal components before building the clustering kNN graph, "
                        + "when the marker panel is wide (> " + ScatterMath.PCA_DEFAULT_THRESHOLD
                        + " measurements). Fixes two problems with wide panels: markers with more "
                        + "measurements no longer dominate the distance, and high-dimensional distance "
                        + "concentration no longer degrades the neighbour graph (the scanpy/Seurat "
                        + "standard). On by default; applies to both k-means and Leiden. Below the "
                        + "threshold this is a no-op — the existing small-panel behaviour is unchanged."));
        pcaComponentsSpinner = new Spinner<>(2, 500, PCA_COMPONENTS_PREF.get());
        pcaComponentsSpinner.setEditable(true);
        pcaComponentsSpinner.setPrefWidth(70);
        pcaComponentsSpinner.setDisable(!pcaEnabledCheck.isSelected());
        pcaComponentsSpinner.setTooltip(
                new javafx.scene.control.Tooltip("Principal components to keep when PCA reduction is on (default "
                        + ScatterMath.PCA_DEFAULT_MAX_COMPONENTS + ", mirrors scanpy)."));
        pcaComponentsSpinner.focusedProperty().addListener((o, was, isNow) -> {
            if (!isNow) {
                commitSpinnerEditor(pcaComponentsSpinner);
                PCA_COMPONENTS_PREF.set(pcaComponentsSpinner.getValue());
            }
        });
        pcaComponentsSpinner.valueProperty().addListener((o, a, b) -> PCA_COMPONENTS_PREF.set(b));
        pcaEnabledCheck.selectedProperty().addListener((o, a, b) -> {
            PCA_ENABLED_PREF.set(b);
            pcaComponentsSpinner.setDisable(!b);
        });

        boolean methodIsLeiden = methodCombo.getValue() == ClusterMethod.LEIDEN;
        kLabel.managedProperty().bind(kLabel.visibleProperty());
        kSpinner.managedProperty().bind(kSpinner.visibleProperty());
        resolutionLabel.managedProperty().bind(resolutionLabel.visibleProperty());
        resolutionSpinner.managedProperty().bind(resolutionSpinner.visibleProperty());
        reproducibleCheck.managedProperty().bind(reproducibleCheck.visibleProperty());
        kLabel.setVisible(!methodIsLeiden);
        kSpinner.setVisible(!methodIsLeiden);
        resolutionLabel.setVisible(methodIsLeiden);
        resolutionSpinner.setVisible(methodIsLeiden);
        // "Sample multiple seeds" applies to both methods (k-means n_init restarts and
        // Leiden random starts), so it stays visible regardless of the selected method.
        reproducibleCheck.setVisible(true);

        // Cohort-mode radio pair (D-04/D-05): the cohort ASSIGNMENT mechanism for Leiden
        // specifically — NOT a replacement for methodCombo, which selects the clustering
        // algorithm (k-means vs Leiden). Net visibility = project scope AND Method == LEIDEN,
        // enforced by updateCohortModeVisibility() below.
        ToggleGroup cohortModeGroup = new ToggleGroup();
        cohortModeAllCells = new RadioButton("Cluster all cells");
        cohortModeTransfer = new RadioButton("Transfer from sample");
        cohortModeAllCells.setToggleGroup(cohortModeGroup);
        cohortModeTransfer.setToggleGroup(cohortModeGroup);
        cohortModeAllCells.setSelected(true); // D-05: all-cells (exact/true-scanpy) is the default
        cohortModeAllCells.setTooltip(new javafx.scene.control.Tooltip(
                "Exact / true-scanpy: pool every cell across the selected images into one HNSW kNN "
                        + "graph and run a SINGLE Leiden partition over the whole cohort "
                        + "(sc.tl.leiden-style). Default — slower, but every cell is genuinely "
                        + "clustered rather than approximated."));
        cohortModeTransfer.setTooltip(new javafx.scene.control.Tooltip(
                "Fast / approximate: fit Leiden on the pooled sample only, then assign every other "
                        + "cell by kNN label transfer against that labelled sample (sc.tl.ingest-style)."));
        cohortModeAllCells.managedProperty().bind(cohortModeAllCells.visibleProperty());
        cohortModeTransfer.managedProperty().bind(cohortModeTransfer.visibleProperty());
        cohortModeAllCells.setVisible(false);
        cohortModeTransfer.setVisible(false);
        cohortModeGroup.selectedToggleProperty().addListener((obs, old, sel) -> {
            if (sel == null && old != null) {
                old.setSelected(true); // keep exactly one selected
            }
        });

        cancelAllCellsBtn = new Button("Cancel");
        cancelAllCellsBtn.managedProperty().bind(cancelAllCellsBtn.visibleProperty());
        cancelAllCellsBtn.setVisible(false);
        cancelAllCellsBtn.setTooltip(new javafx.scene.control.Tooltip(
                "Stop the all-cells run after the image currently being written finishes. "
                        + "Already-written images keep their Cluster measurement (no rollback)."));
        cancelAllCellsBtn.setOnAction(e -> {
            if (allCellsToken != null) {
                allCellsToken.cancel();
                cancelAllCellsBtn.setDisable(true);
                statusLabel.setText("Cancelling — finishing the current image…");
            }
        });

        methodCombo.valueProperty().addListener((o, a, b) -> {
            boolean leiden = b == ClusterMethod.LEIDEN;
            kLabel.setVisible(!leiden);
            kSpinner.setVisible(!leiden);
            resolutionLabel.setVisible(leiden);
            resolutionSpinner.setVisible(leiden);
            updateCohortModeVisibility();
        });

        annotationField = new TextField();
        annotationField.setPromptText("name (blank = all cells)");
        annotationField.setPrefWidth(150);
        annotationField.setTooltip(
                new javafx.scene.control.Tooltip("Only cluster cells whose centroid falls inside an annotation "
                        + "whose name (or classification) contains this text. "
                        + "Leave blank to use all cells."));
        annotationField.setOnAction(e -> recompute()); // Enter re-runs

        classField = new ComboBox<>();
        classField.setEditable(true);
        classField.setPromptText("class (blank = all)");
        classField.setPrefWidth(150);
        classField.getItems().add("");
        for (PathClass pc : qupath.getAvailablePathClasses()) {
            if (pc != null && pc.getName() != null) {
                classField.getItems().add(pc.toString());
            }
        }
        classField.setValue("");
        classField.setTooltip(
                new javafx.scene.control.Tooltip("Only cluster cells whose current QuPath classification contains "
                        + "this text — e.g. assign Immune/Tumour/Other via Apply Clusters, "
                        + "then drill in with \"Immune\". Combines with the annotation "
                        + "filter. Leave blank for all classes."));
        classField.setOnAction(e -> recompute());

        clusterMarkersBtn = new MenuButton("Cluster markers (all)");
        clusterMarkersBtn.setTooltip(
                new javafx.scene.control.Tooltip("Markers used for k-means and the embedding. Uncheck markers to "
                        + "sub-cluster on a focused panel (e.g. immune markers only). "
                        + "Values are re-standardized over the active cells each run."));
        for (String marker : this.markerFeatures) {
            CheckMenuItem item = new CheckMenuItem(marker);
            item.setSelected(true);
            item.selectedProperty().addListener((o, a, b) -> updateClusterMarkersLabel());
            clusterMarkerItems.add(item);
            clusterMarkersBtn.getItems().add(item);
        }

        Button recomputeBtn = new Button("Recompute");
        recomputeBtn.setTooltip(
                new javafx.scene.control.Tooltip("Re-fit k-means + the embedding on the current rows (the open image, "
                        + "or the project sample). Does not re-sample — use “Images…” for "
                        + "that in project scope."));
        recomputeBtn.setOnAction(e -> recompute());

        // ── Scope toggle: cluster the open image, or a project-wide sample ───────
        ToggleGroup scopeGroup = new ToggleGroup();
        imageScopeToggle = new ToggleButton("Current image");
        projectScopeToggle = new ToggleButton("Project");
        imageScopeToggle.setToggleGroup(scopeGroup);
        projectScopeToggle.setToggleGroup(scopeGroup);
        imageScopeToggle.setSelected(true);
        imageScopeToggle.setTooltip(
                new javafx.scene.control.Tooltip("Cluster every cell of the open image, with full viewer "
                        + "interaction (box/lasso select, click-to-select)."));
        projectScopeToggle.setTooltip(
                new javafx.scene.control.Tooltip("Fit one k-means on a bounded sample pooled across selected images, "
                        + "then assign it consistently across the whole cohort. Viewer "
                        + "selection is plot-only here (sampled cells aren't all open)."));
        scopeGroup.selectedToggleProperty().addListener((obs, old, sel) -> {
            if (sel == null && old != null) {
                old.setSelected(true); // keep exactly one selected
            }
        });
        imageScopeToggle.setOnAction(e -> {
            if (imageScopeToggle.isSelected()) {
                switchToImageScope();
            }
        });
        projectScopeToggle.setOnAction(e -> {
            if (projectScopeToggle.isSelected()) {
                switchToProjectScope();
            }
        });

        sampleSpinner = new Spinner<>(1000, 5_000_000, 50_000, 10_000);
        sampleSpinner.setEditable(true);
        sampleSpinner.setPrefWidth(110);
        sampleSpinner.setTooltip(
                new javafx.scene.control.Tooltip("Max cells to sample. Project scope: cells pooled to FIT k-means "
                        + "(every cell is still classified by Assign — 50k is plenty for "
                        + "stable centroids). Current-image scope: a random subsample of "
                        + "the open image to plot + cluster. Press Enter or click "
                        + "“Re-sample” to apply."));
        // Editable spinners don't commit typed text to the value unless focus
        // leaves the editor — commit on focus loss so getValue() is current.
        sampleSpinner.focusedProperty().addListener((o, was, isNow) -> {
            if (!isNow) {
                commitSpinnerEditor(sampleSpinner);
            }
        });
        // Enter in the editor commits and re-samples with the new cap.
        sampleSpinner.getEditor().setOnAction(e -> {
            commitSpinnerEditor(sampleSpinner);
            reSample();
        });

        imagesBtn = new Button("Images…");
        imagesBtn.setTooltip(
                new javafx.scene.control.Tooltip("Choose which project images to draw the sample from. Picking "
                        + "images does NOT sample — click “Re-sample” afterwards."));
        imagesBtn.setOnAction(e -> chooseProjectImages());

        reSampleBtn = new Button("Re-sample");
        reSampleBtn.setTooltip(
                new javafx.scene.control.Tooltip("Draw a fresh sample using the “Sample” cap: in Project scope a "
                        + "pooled sample across the chosen images; in Current-image scope "
                        + "a random subsample of the open image. Does not cluster — click "
                        + "“Recompute” to cluster."));
        reSampleBtn.setOnAction(e -> reSample());

        // Images… is project-only; the Sample cap + Re-sample apply to both scopes.
        projectControls = new HBox(8, imagesBtn);
        projectControls.setAlignment(Pos.CENTER_LEFT);
        projectControls.managedProperty().bind(projectControls.visibleProperty());
        projectControls.setVisible(false);

        Button newBtn = new Button("New clustering session");
        newBtn.setTooltip(
                new javafx.scene.control.Tooltip(
                        "Start a fresh clustering session — re-select measurements and re-cluster. Use this to change "
                                + "the marker set or scope; reopening from the menu otherwise restores this window as it was."));
        newBtn.setDisable(onNewPlot == null);
        newBtn.setOnAction(e -> {
            if (onNewPlot != null) {
                onNewPlot.run();
            }
        });

        sampleControls = new HBox(8, new Label("Sample:"), sampleSpinner, reSampleBtn, newBtn);
        sampleControls.setAlignment(Pos.CENTER_LEFT);

        progress = new ProgressIndicator();
        progress.setPrefSize(20, 20);
        progress.setVisible(false);

        // Mirror the spinner's progress/visibility onto a wide status-bar bar so
        // every long operation has obvious feedback. Bound, so the existing
        // progress.setProgress(...) / setVisible(...) calls drive it unchanged.
        progressBar = new ProgressBar();
        progressBar.setPrefWidth(220);
        progressBar.progressProperty().bind(progress.progressProperty());
        progressBar.visibleProperty().bind(progress.visibleProperty());
        progressBar.managedProperty().bind(progressBar.visibleProperty());

        markerCombo = new ComboBox<>();
        markerCombo.getItems().addAll(this.markerFeatures);
        if (!this.markerFeatures.isEmpty()) {
            markerCombo.setValue(this.markerFeatures.get(0));
        }
        markerCombo.setDisable(true);

        colorCombo = new ComboBox<>();
        colorCombo
                .getItems()
                .addAll(
                        ScatterPlotCanvas.ColorMode.CLUSTER,
                        ScatterPlotCanvas.ColorMode.CLASS,
                        ScatterPlotCanvas.ColorMode.MARKER);
        colorCombo.setValue(ScatterPlotCanvas.ColorMode.CLUSTER);
        colorCombo.setOnAction(e -> {
            markerCombo.setDisable(colorCombo.getValue() != ScatterPlotCanvas.ColorMode.MARKER);
            plot.redraw();
        });

        markerCombo.setOnAction(e -> {
            if (colorCombo.getValue() == ScatterPlotCanvas.ColorMode.MARKER) {
                plot.redraw();
            }
        });

        ToggleGroup selGroup = new ToggleGroup();
        boxToggle = new ToggleButton("Box");
        lassoToggle = new ToggleButton("Lasso");
        boxToggle.setToggleGroup(selGroup);
        lassoToggle.setToggleGroup(selGroup);
        boxToggle.setSelected(true);
        // Keep one always selected.
        selGroup.selectedToggleProperty().addListener((obs, old, sel) -> {
            if (sel == null && old != null) {
                old.setSelected(true);
            }
        });

        applyClustersBtn = new Button("Apply Clusters…");
        applyClustersBtn.setOnAction(e -> applyClustersToClasses());

        clusterOverlayBtn = new Button("By cluster");
        clusterOverlayBtn.setTooltip(new javafx.scene.control.Tooltip(
                "Paint every cell in the open image by its assigned phenotype cluster (a full-tissue "
                        + "map: each cell is assigned to its nearest cluster — nearest-centroid for k-means, "
                        + "kNN label transfer for Leiden). Non-destructive: writes a numeric \"Cluster\" "
                        + "measurement and a viewer overlay, never changing the cell's classification. "
                        + "Cleared when this window closes."));
        clusterOverlayBtn.setOnAction(e -> showClusterOverlay());

        classViewBtn = new Button("By classification");
        classViewBtn.setTooltip(new javafx.scene.control.Tooltip(
                "Remove the cluster overlay and show QuPath's native classification colours "
                        + "(the phenotype classes, e.g. those assigned via Apply Clusters)."));
        classViewBtn.setOnAction(e -> showClassificationView());

        Button exportBtn = new Button("Export PNG…");
        exportBtn.setOnAction(e -> exportAsPng());

        Button closeBtn = new Button("Close");
        closeBtn.setOnAction(e -> stage.close());

        // Row 1 = "what to compute": embedding + clustering method + the Recompute action.
        HBox row1 = new HBox(
                8,
                new Label("Embedding:"),
                embeddingCombo,
                fullUmapCheck,
                new Separator(Orientation.VERTICAL),
                new Label("Method:"),
                methodCombo,
                kLabel,
                kSpinner,
                resolutionLabel,
                resolutionSpinner,
                reproducibleCheck,
                pcaEnabledCheck,
                new Label("PCA comps:"),
                pcaComponentsSpinner,
                cohortModeAllCells,
                cohortModeTransfer,
                recomputeBtn);
        row1.setAlignment(Pos.CENTER_LEFT);

        // Row 1b = "on what data": scope toggles + per-scope sampling controls.
        HBox rowScope = new HBox(
                8,
                new Label("Scope:"),
                imageScopeToggle,
                projectScopeToggle,
                projectControls,
                sampleControls,
                progress,
                cancelAllCellsBtn);
        rowScope.setAlignment(Pos.CENTER_LEFT);

        HBox rowFilter = new HBox(
                8,
                new Label("Annotation:"),
                annotationField,
                new Label("Within class:"),
                classField,
                clusterMarkersBtn);
        rowFilter.setAlignment(Pos.CENTER_LEFT);

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);
        HBox row2 = new HBox(
                8,
                new Label("Colour by:"),
                colorCombo,
                new Label("Marker:"),
                markerCombo,
                spacer,
                new Label("Select:"),
                boxToggle,
                lassoToggle,
                applyClustersBtn,
                exportBtn,
                closeBtn);
        row2.setAlignment(Pos.CENTER_LEFT);

        // Viewer-overlay row (CN-dialog parity): recolour the open image's cells by their
        // assigned cluster, or revert to native classification colours. Non-destructive.
        HBox rowViewer = new HBox(8, new Label("Colour cells in image:"), clusterOverlayBtn, classViewBtn);
        rowViewer.setAlignment(Pos.CENTER_LEFT);

        VBox top = new VBox(6, row1, rowScope, rowFilter, rowViewer, row2);
        top.setPadding(new Insets(8));

        // The visual layer owns the canvas + drawing + box/lasso/legend gestures;
        // it reads plotted-row state via the PlotModel (this view) and reports
        // gestures back through the callbacks. Built here, after every control its
        // lambdas read (lassoToggle, the combos, kSpinner) already exists.
        plot = new ScatterPlotCanvas(
                asPlotModel(),
                () -> lassoToggle.isSelected(),
                this::onRegionGesture,
                this::selectCluster,
                this::updateStatus);

        // The canvas component fills the centre and resizes with the window.
        statusLabel.setMaxWidth(Double.MAX_VALUE);
        HBox.setHgrow(statusLabel, Priority.ALWAYS);
        HBox bottomBar = new HBox(8, statusLabel, progressBar);
        bottomBar.setAlignment(Pos.CENTER_LEFT);
        bottomBar.setPadding(new Insets(0, 8, 0, 0));

        BorderPane root = new BorderPane();
        root.setTop(top);
        root.setCenter(plot.getNode());
        root.setBottom(bottomBar);

        stage.initOwner(owner);
        stage.initModality(Modality.NONE);
        stage.setTitle("CellTune — Cell Scatter Plot");
        stage.setScene(new Scene(root, 1000, 720));
        stage.setResizable(true);
        stage.setOnHidden(e -> {
            removeSelectionListener();
            clearClusterOverlay(); // revert the viewer to classification colours on close
        });

        // Load the open image's cells but DON'T cluster yet — clustering only
        // runs when the user clicks Recompute (or changes scope / re-samples).
        loadCurrentImageData(cellList);
        applyScopeOverrides();
        installSelectionListener();
        statusLabel.setText(currentImageLoadedMessage(cellList.size()));
        plot.redraw();
    }

    /** Status text after loading the open image, noting any subsample. */
    private String currentImageLoadedMessage(int total) {
        if (nRows < total) {
            return String.format("Subsampled %,d of %,d cell(s) — click “Recompute” to cluster.", nRows, total);
        }
        return String.format("%,d cell(s) loaded — click “Recompute” to cluster.", nRows);
    }

    // ── Data loading (rebuilds the per-row arrays on a scope switch) ────────────

    /**
     * Loads the open image's detections as the plotted rows (CURRENT_IMAGE scope),
     * randomly subsampling down to the current "Sample" cap when the image has
     * more cells than the cap.
     */
    private void loadCurrentImageData(List<PathObject> cellList) {
        scope = Scope.CURRENT_IMAGE;
        int cap = sampleSpinner.getValue();
        List<PathObject> used = cellList;
        if (cellList.size() > cap) {
            int[] pick = ScatterMath.randomSubsample(cellList.size(), cap);
            used = new ArrayList<>(cap);
            for (int idx : pick) {
                used.add(cellList.get(idx));
            }
        }
        int n = used.size();
        int nFeat = markerFeatures.size();
        nRows = n;
        cells = used.toArray(new PathObject[0]);
        rowClass = null;
        rowImage = null;
        ex = new double[n];
        ey = new double[n];
        cluster = new int[n];
        selected = new boolean[n];
        indexOf = new IdentityHashMap<>(n * 2);
        // Leave points unplotted (NaN) and unclustered until Recompute runs.
        java.util.Arrays.fill(ex, Double.NaN);
        java.util.Arrays.fill(ey, Double.NaN);
        for (int i = 0; i < n; i++) {
            indexOf.put(cells[i], i);
            cluster[i] = -1;
        }
        var extractor = new CellFeatureExtractor(markerFeatures, normalizer);
        float[] flat = extractor.extractMatrix(used);
        raw = new double[n][nFeat];
        for (int i = 0; i < n; i++) {
            int off = i * nFeat;
            for (int j = 0; j < nFeat; j++) {
                raw[i][j] = flat[off + j];
            }
        }
        std = ScatterMath.standardizeColumns(raw);
        clearFit();
    }

    /** Loads a project-wide pooled sample as the plotted rows (PROJECT scope). */
    private void loadProjectData(CohortClusterModel.SampleData sd) {
        scope = Scope.PROJECT;
        int n = sd.sampledCells();
        nRows = n;
        cells = null;
        indexOf = null;
        rowClass = sd.rowClass();
        rowImage = sd.rowImage();
        ex = new double[n];
        ey = new double[n];
        cluster = new int[n];
        selected = new boolean[n];
        // Leave points unplotted (NaN) and unclustered until Recompute runs.
        java.util.Arrays.fill(ex, Double.NaN);
        java.util.Arrays.fill(ey, Double.NaN);
        for (int i = 0; i < n; i++) {
            cluster[i] = -1;
        }
        raw = sd.raw(); // already [n][nFeat] in markerFeatures column order
        std = ScatterMath.standardizeColumns(raw);
        clearFit();
    }

    private void clearFit() {
        fitMean = null;
        fitSd = null;
        fitCentroids = null;
        fitAssignCentroids = null;
        fitMarkers = null;
        fitClassFilter = null;
        fitNClusters = 0;
        fitLeidenReference = null;
        fitLeidenReferenceLabels = null;
        fitPcaProjector = null;
        fitPcaApplied = false;
        fitAllCellsCounts = null;
        allCellsWriteActive = false;
        clusterMeasurementStale = true;
    }

    /** Show the scatter plot window. */
    public void show() {
        stage.show();
        stage.toFront();
    }

    /** Hide the window. The instance (and its fit / clusters) stays reusable via {@link #show()}. */
    public void close() {
        stage.close();
    }

    /** The image this plot was built for — used to decide whether a retained window can be reused. */
    public boolean isForImage(String name) {
        return imageName != null && imageName.equals(name);
    }

    /**
     * Whether a retained window can be reused for the image now open. A project-scope plot pools
     * across many images, so it is independent of which single image is active and is always
     * reusable; a current-image plot is only reusable for the same image it was built for.
     */
    public boolean isReusableFor(String name) {
        return scope == Scope.PROJECT || isForImage(name);
    }

    // ── Embedding + clustering (background thread) ──────────────────────────────

    private void recompute() {
        // Project scope with no sample loaded yet: draw the sample first, then
        // cluster once it's in (sampling is async, so it re-enters recompute()).
        if (scope == Scope.PROJECT && nRows == 0) {
            if (projectImages == null || projectImages.isEmpty()) {
                statusLabel.setText("Click “Images…” to choose project images first.");
            } else {
                runProjectSample(true);
            }
            return;
        }
        final Embedding embedding = embeddingCombo.getValue();
        final ClusterMethod method = methodCombo.getValue();
        final int k = kSpinner.getValue();
        final double resolution = resolutionSpinner.getValue();
        final boolean reproducible = reproducibleCheck.isSelected();
        final boolean pcaEnabled = pcaEnabledCheck.isSelected();
        final int pcaMaxComponents = pcaComponentsSpinner.getValue();
        final int n = nRows;
        final String keyword = annotationField.getText();
        final String classKeyword = classField.getValue();
        final int[] selCols = selectedMarkerColumns();
        final int umapCap = fullUmapCheck.isSelected() ? Integer.MAX_VALUE : MAX_UMAP_CELLS;
        if (n == 0) {
            statusLabel.setText("No cells to plot.");
            return;
        }
        if (selCols.length < 2) {
            statusLabel.setText("Select at least 2 cluster markers (see “Cluster markers”).");
            return;
        }

        progress.setVisible(true);
        statusLabel.setText("Computing " + embedding + " embedding…");
        setControlsDisabled(true);

        new Thread(
                        () -> {
                            String notice = "";
                            try {
                                // Smile's PCA/UMAP load native libs via JavaCPP, which needs
                                // java.base/java.lang opened. The extension opens it at startup;
                                // this is a defensive no-op if already done.
                                JvmModuleOpener.ensureJavaLangOpen();

                                // Restrict clustering/embedding to cells inside matching
                                // annotation(s) AND of the matching class (all cells when blank).
                                int[] activeIdx = computeActiveIndices(keyword, classKeyword);
                                final int m = activeIdx.length;

                                // Full-length outputs; non-active cells stay unclustered/unplotted.
                                int[] newCluster = new int[n];
                                java.util.Arrays.fill(newCluster, -1);
                                double[] nx = new double[n];
                                double[] ny = new double[n];
                                java.util.Arrays.fill(nx, Double.NaN);
                                java.util.Arrays.fill(ny, Double.NaN);

                                boolean annoFilter = keyword != null && !keyword.isBlank();
                                boolean classFilter = classKeyword != null && !classKeyword.isBlank();
                                boolean filtered = annoFilter || classFilter;
                                String scopeDesc = describeScope(keyword, classKeyword);
                                if (m == 0) {
                                    final String fNotice = filtered ? " (no cells matching " + scopeDesc + ")" : "";
                                    Platform.runLater(() -> {
                                        System.arraycopy(nx, 0, ex, 0, n);
                                        System.arraycopy(ny, 0, ey, 0, n);
                                        cluster = newCluster;
                                        clearFit();
                                        progress.setVisible(false);
                                        setControlsDisabled(false);
                                        plot.redraw();
                                        appendStatusNotice(fNotice);
                                    });
                                    return;
                                }

                                // Active feature matrix: the selected marker columns of the active
                                // cells, re-standardized over that subset so sub-clustering scales
                                // to the subpopulation (not the whole image).
                                double[][] activeRaw = new double[m][selCols.length];
                                for (int j = 0; j < m; j++) {
                                    double[] src = raw[activeIdx[j]];
                                    for (int c = 0; c < selCols.length; c++) {
                                        activeRaw[j][c] = src[selCols[c]];
                                    }
                                }
                                double[] mean = new double[selCols.length];
                                double[] sd = new double[selCols.length];
                                double[][] active = ScatterMath.standardizeColumns(activeRaw, mean, sd);

                                // Conditional PCA reduction (Requirement 2): applied to the z-scored
                                // active matrix BEFORE the clustering kNN graph (Leiden AND k-means both),
                                // never before the 2D display embedding below (fillPca/fillUmap still see
                                // `active`, unchanged — the embedding is display-only and unaffected). A
                                // no-op (applied=false, clusterMatrix == active) when the checkbox is off
                                // or the active marker-column count is at/below ScatterMath's threshold,
                                // preserving the existing small-panel behaviour exactly (regression guard).
                                ScatterMath.PcaReduction pcaReduction = ScatterMath.pcaReduce(
                                        active,
                                        pcaEnabled ? pcaMaxComponents : 0,
                                        ScatterMath.PCA_DEFAULT_THRESHOLD,
                                        ScatterMath.PCA_DEFAULT_FIT_SAMPLE_CAP,
                                        ScatterMath.PCA_DEFAULT_SEED);
                                double[][] clusterMatrix = pcaReduction.reduced();
                                if (pcaReduction.applied()) {
                                    logger.info(
                                            "PCA: {} → {} comps, {}% variance",
                                            selCols.length,
                                            pcaReduction.nComponents(),
                                            String.format("%.1f", pcaReduction.cumulativeVariance() * 100));
                                }

                                // ── Cluster the active subset (k-means or Leiden) ────────────────
                                int[] subCluster = new int[m];
                                int nClustersEff;
                                double[][] leidenRef = null;
                                int[] leidenRefLabels = null;
                                // D-09: the single-image/preview fit can cheaply surface its own measured
                                // ANN recall too (clusterViaAnn now always returns one) — -1 (not shown)
                                // for the k-means branch, which never touches the ANN gate.
                                double leidenRecall = -1.0;
                                if (method == ClusterMethod.LEIDEN) {
                                    int randomStarts = reproducible ? LEIDEN_REPRODUCIBLE_STARTS : LEIDEN_SINGLE_START;
                                    long seed = reproducible ? LEIDEN_SEED : System.nanoTime();
                                    // D-06: the single-image / interactive-preview fit routes its kNN
                                    // graph build through the HNSW ANN index (LEI-07), not brute-force
                                    // featureKnn. A recall-gate failure here is rare but must not crash
                                    // the Recompute thread or leave the UI disabled — surface it and
                                    // bail out gracefully, exactly like the m==0 branch above.
                                    LeidenModel.LeidenResult leidenResult;
                                    try {
                                        leidenResult = LeidenModel.clusterViaAnn(
                                                clusterMatrix,
                                                LEIDEN_GRAPH_K,
                                                resolution,
                                                randomStarts,
                                                seed,
                                                reproducible);
                                    } catch (AnnRecallException are) {
                                        logger.warn("Leiden preview ANN recall gate failed: {}", are.getMessage());
                                        Platform.runLater(() -> {
                                            progress.setVisible(false);
                                            setControlsDisabled(false);
                                            statusLabel.setText("Leiden preview: ANN recall too low — try more "
                                                    + "cells / different markers.");
                                        });
                                        return;
                                    }
                                    System.arraycopy(leidenResult.labels(), 0, subCluster, 0, m);
                                    nClustersEff = Math.max(1, leidenResult.nClusters());
                                    leidenRecall = leidenResult.recall();
                                    // Retained for the cohort-scope Leiden assign path (kNN label
                                    // transfer against this labelled sample, Task 5) — the SAME space
                                    // (PC-reduced when PCA applied, marker space otherwise) Leiden was
                                    // clustered in, so a later transfer projects its query rows the
                                    // same way before voting (fitPcaProjector, stored below).
                                    leidenRef = clusterMatrix;
                                    leidenRefLabels = subCluster.clone();
                                } else {
                                    int kEff = Math.min(k, m);
                                    if (kEff >= 2) {
                                        // "Sample multiple seeds" ON -> n_init restarts, keeping the
                                        // lowest-distortion fit (scikit-learn's n_init), seeded from a
                                        // fixed value for run-to-run identical labels; OFF -> a single,
                                        // faster fit. Mirrors the Leiden random-starts branch above and
                                        // NeighborhoodModel's k-means restart policy.
                                        int restarts = reproducible ? KMEANS_MULTI_RESTARTS : 1;
                                        if (reproducible) {
                                            MathEx.setSeed(KMEANS_SEED);
                                        }
                                        KMeans best = KMeans.fit(clusterMatrix, kEff);
                                        for (int r = 1; r < restarts; r++) {
                                            KMeans cand = KMeans.fit(clusterMatrix, kEff);
                                            if (cand.distortion < best.distortion) {
                                                best = cand;
                                            }
                                        }
                                        System.arraycopy(best.y, 0, subCluster, 0, m);
                                    }
                                    nClustersEff = k;
                                }
                                for (int j = 0; j < m; j++) {
                                    newCluster[activeIdx[j]] = subCluster[j];
                                }

                                // Per-cluster z-scored centroids (nClustersEff rows; empty clusters
                                // stay 0). ALWAYS computed in the original marker space (`active`),
                                // regardless of whether clustering ran on a PCA-reduced matrix
                                // (Requirement 2) — never `clusterMatrix`. Retained for the cohort
                                // assign and the assignment heatmap — computed post-hoc for Leiden
                                // exactly like k-means, for DISPLAY only (Leiden's cohort assignment
                                // itself uses kNN label transfer against fitLeidenReference, not
                                // these centroids).
                                double[][] cents = new double[nClustersEff][selCols.length];
                                int[] centCount = new int[nClustersEff];
                                for (int j = 0; j < m; j++) {
                                    int lab = subCluster[j];
                                    if (lab < 0 || lab >= nClustersEff) {
                                        continue;
                                    }
                                    centCount[lab]++;
                                    for (int c = 0; c < selCols.length; c++) {
                                        cents[lab][c] += active[j][c];
                                    }
                                }
                                for (int lab = 0; lab < nClustersEff; lab++) {
                                    if (centCount[lab] > 0) {
                                        for (int c = 0; c < selCols.length; c++) {
                                            cents[lab][c] /= centCount[lab];
                                        }
                                    }
                                }

                                // Per-cluster centroids in the CLUSTERING space (Fix 4/Requirement:
                                // k-means cohort assignment must run in the SAME space clustering
                                // happened in) — PC space when PCA was applied above, otherwise
                                // identical to `cents` (both are then `active`-space). Computed the
                                // same way as `cents` but over `clusterMatrix`, not `active`, so
                                // `nearestCentroid` against these later reproduces the SAME Voronoi
                                // partition the preview scatter/k-means fit actually used — unlike
                                // `cents` (kept marker-space, unchanged, for the interpretable
                                // Assign-dialog heatmap).
                                int assignCols = clusterMatrix.length > 0 ? clusterMatrix[0].length : selCols.length;
                                double[][] assignCents = new double[nClustersEff][assignCols];
                                for (int j = 0; j < m; j++) {
                                    int lab = subCluster[j];
                                    if (lab < 0 || lab >= nClustersEff) {
                                        continue;
                                    }
                                    for (int c = 0; c < assignCols; c++) {
                                        assignCents[lab][c] += clusterMatrix[j][c];
                                    }
                                }
                                for (int lab = 0; lab < nClustersEff; lab++) {
                                    if (centCount[lab] > 0) {
                                        for (int c = 0; c < assignCols; c++) {
                                            assignCents[lab][c] /= centCount[lab];
                                        }
                                    }
                                }
                                final double[] fMean = mean;
                                final double[] fSd = sd;
                                final double[][] fCents = cents;
                                final double[][] fAssignCents = assignCents;
                                final List<String> fMarkers = markerNamesFor(selCols);
                                final int fNClusters = nClustersEff;
                                final double[][] fLeidenRef = leidenRef;
                                final int[] fLeidenRefLabels = leidenRefLabels;
                                final java.util.function.UnaryOperator<double[][]> fPcaProjector =
                                        pcaReduction.projector();
                                final boolean fPcaApplied = pcaReduction.applied();
                                final int fPcaComponents = pcaReduction.nComponents();
                                final double fPcaCumVariance = pcaReduction.cumulativeVariance();

                                // ── Embedding on the active subset ───────────────────────────────
                                double[] subX = new double[m];
                                double[] subY = new double[m];
                                java.util.Arrays.fill(subX, Double.NaN);
                                java.util.Arrays.fill(subY, Double.NaN);

                                if (embedding == Embedding.PCA) {
                                    ScatterMath.fillPca(active, m, subX, subY);
                                } else {
                                    try {
                                        notice = ScatterMath.fillUmap(active, m, subX, subY, umapCap);
                                    } catch (LinkageError err) {
                                        // UMAP's spectral layout loads the native ARPACK library
                                        // through JavaCPP, which uses reflection into java.base.
                                        // On JVMs started without
                                        // --add-opens=java.base/java.lang=ALL-UNNAMED that load
                                        // fails with an Error (ExceptionInInitializerError, then
                                        // NoClassDefFoundError on retries, or UnsatisfiedLinkError)
                                        // — all LinkageError, not Exception. Fall back to PCA so the
                                        // plot still renders on any system.
                                        logger.warn(
                                                "UMAP unavailable ({}); falling back to PCA. Launch "
                                                        + "QuPath with "
                                                        + "--add-opens=java.base/java.lang=ALL-UNNAMED to "
                                                        + "enable UMAP.",
                                                err.toString());
                                        logger.debug("UMAP native load failure detail", err);
                                        java.util.Arrays.fill(subX, Double.NaN);
                                        java.util.Arrays.fill(subY, Double.NaN);
                                        ScatterMath.fillPca(active, m, subX, subY);
                                        notice = " (UMAP unavailable — showing PCA)";
                                    }
                                }
                                for (int j = 0; j < m; j++) {
                                    nx[activeIdx[j]] = subX[j];
                                    ny[activeIdx[j]] = subY[j];
                                }

                                if (filtered) {
                                    notice = notice + String.format(" (%,d cells in %s)", m, scopeDesc);
                                }
                                if (selCols.length < markerFeatures.size()) {
                                    notice = notice
                                            + String.format(" · %d/%d markers", selCols.length, markerFeatures.size());
                                }
                                if (fPcaApplied) {
                                    notice = notice
                                            + String.format(
                                                    " · PCA: %d → %d comps, %.1f%% variance",
                                                    selCols.length, fPcaComponents, fPcaCumVariance * 100);
                                }
                                if (leidenRecall >= 0) {
                                    notice = notice + String.format(" · ANN recall %.3f", leidenRecall);
                                }

                                final String fNotice = notice;
                                final String fClassFilter = classKeyword;
                                Platform.runLater(() -> {
                                    System.arraycopy(nx, 0, ex, 0, n);
                                    System.arraycopy(ny, 0, ey, 0, n);
                                    cluster = newCluster;
                                    fitMean = fMean;
                                    fitSd = fSd;
                                    fitCentroids = fCents;
                                    fitAssignCentroids = fAssignCents;
                                    fitMarkers = fMarkers;
                                    fitClassFilter = fClassFilter;
                                    fitNClusters = fNClusters;
                                    fitLeidenReference = fLeidenRef;
                                    fitLeidenReferenceLabels = fLeidenRefLabels;
                                    fitPcaProjector = fPcaProjector;
                                    fitPcaApplied = fPcaApplied;
                                    // A new preview fit invalidates any retained all-cells state (LEI-09
                                    // companion fix): the Assign dialog must go back to showing this
                                    // preview's counts/heatmap, and cluster->class assignment must go back
                                    // to the preview-reference transfer path.
                                    fitAllCellsCounts = null;
                                    allCellsWriteActive = false;
                                    clusterMeasurementStale =
                                            true; // new fit — any written Cluster measurement is now stale
                                    progress.setVisible(false);
                                    setControlsDisabled(false);
                                    plot.redraw();
                                    appendStatusNotice(fNotice);
                                });
                            } catch (Throwable ex) {
                                // Throwable, not Exception: Smile's native loaders fail with
                                // Errors (e.g. ExceptionInInitializerError) when java.lang is not
                                // open. Catch them here so they never reach the uncaught-exception
                                // dialog, and give the user an actionable message.
                                logger.error("Failed to compute scatter embedding", ex);
                                boolean nativeIssue = ex instanceof LinkageError;
                                final String msg = nativeIssue
                                        ? "Embedding failed: native math libraries unavailable. "
                                                + "Launch QuPath with "
                                                + "--add-opens=java.base/java.lang=ALL-UNNAMED."
                                        : "Embedding failed: " + ex.getMessage();
                                Platform.runLater(() -> {
                                    progress.setVisible(false);
                                    setControlsDisabled(false);
                                    statusLabel.setText(msg);
                                });
                            }
                        },
                        "CellTune-ScatterEmbedding")
                .start();
    }

    /**
     * Indices of cells eligible for clustering/embedding, intersecting two
     * optional filters: inside an annotation whose label contains
     * {@code annoKeyword}, and whose current classification contains
     * {@code classKeyword} (both case-insensitive; blank = no restriction). The
     * annotation test mirrors Review-mode's membership check.
     */
    private int[] computeActiveIndices(String annoKeyword, String classKeyword) {
        int n = nRows;
        boolean annoFilter = annoKeyword != null && !annoKeyword.isBlank();
        boolean classFilter = classKeyword != null && !classKeyword.isBlank();

        // Project scope: annotations belong to one image's hierarchy, so the only
        // filter that applies across a pooled cohort sample is the within-class
        // one, tested against each row's carried class.
        if (scope == Scope.PROJECT) {
            if (!classFilter) {
                return identityIndices(n);
            }
            String kw = classKeyword.trim().toLowerCase();
            List<Integer> idx = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                String rc = rowClass[i];
                if (rc != null && rc.toLowerCase().contains(kw)) {
                    idx.add(i);
                }
            }
            return toIntArray(idx);
        }

        if (!annoFilter && !classFilter) {
            return identityIndices(n);
        }

        // Matching annotation ROIs (only needed for the annotation filter).
        List<ROI> rois = new ArrayList<>();
        if (annoFilter && hierarchy != null) {
            String kw = annoKeyword.trim().toLowerCase();
            for (PathObject anno : hierarchy.getAnnotationObjects()) {
                ROI roi = anno.getROI();
                if (roi == null) {
                    continue;
                }
                String label = annotationLabel(anno);
                if (label != null && label.toLowerCase().contains(kw)) {
                    rois.add(roi);
                }
            }
            if (rois.isEmpty()) {
                return new int[0];
            }
        }
        String classKw = classFilter ? classKeyword.trim().toLowerCase() : null;

        List<Integer> idx = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (classFilter) {
                PathClass pc = cells[i].getPathClass();
                if (pc == null || !pc.toString().toLowerCase().contains(classKw)) {
                    continue;
                }
            }
            if (annoFilter) {
                ROI cr = cells[i].getROI();
                if (cr == null) {
                    continue;
                }
                double cx = cr.getCentroidX();
                double cy = cr.getCentroidY();
                boolean inside = false;
                for (ROI r : rois) {
                    if (r.contains(cx, cy)) {
                        inside = true;
                        break;
                    }
                }
                if (!inside) {
                    continue;
                }
            }
            idx.add(i);
        }
        int[] out = new int[idx.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = idx.get(i);
        }
        return out;
    }

    /** Human-readable description of the active-subset filters for the status bar. */
    private static String describeScope(String annoKeyword, String classKeyword) {
        boolean anno = annoKeyword != null && !annoKeyword.isBlank();
        boolean cls = classKeyword != null && !classKeyword.isBlank();
        if (anno && cls) {
            return String.format("class “%s” inside “%s”", classKeyword.trim(), annoKeyword.trim());
        }
        if (cls) {
            return "class “" + classKeyword.trim() + "”";
        }
        if (anno) {
            return "annotation “" + annoKeyword.trim() + "”";
        }
        return "all cells";
    }

    /** Annotation display label: explicit name, else PathClass name, else null. */
    private static String annotationLabel(PathObject anno) {
        String name = anno.getName();
        if (name != null && !name.isBlank()) {
            return name;
        }
        PathClass pc = anno.getPathClass();
        if (pc != null) {
            String pcName = pc.getName();
            if (pcName != null && !pcName.isBlank()) {
                return pcName;
            }
        }
        return null;
    }

    // ── Apply clusters → QuPath classifications ─────────────────────────────────

    /**
     * Opens the shared {@link ClusterAssignmentPane} (centroid heatmap + class
     * dropdowns), then either writes the chosen classes onto the open image's
     * cells (CURRENT_IMAGE scope) or streams the mapping across the whole cohort
     * (PROJECT scope). The CellTune ground-truth label store is never touched.
     */
    private void applyClustersToClasses() {
        if (applying) {
            return;
        }
        // fitNClusters is the effective cluster count of the last Recompute: the
        // k-spinner's value for k-means, or Leiden's own decided community count
        // (Leiden picks this — the user does not select a k for it).
        final int k = fitNClusters > 0 ? fitNClusters : kSpinner.getValue();
        int[] counts = clusterCounts(k);
        if (counts == null) {
            Dialogs.showWarningNotification("CellTune", "No clusters available yet — run Recompute first.");
            return;
        }
        if (scope == Scope.CURRENT_IMAGE && hierarchy == null) {
            Dialogs.showWarningNotification("CellTune", "No image is open to classify.");
            return;
        }
        if (scope == Scope.PROJECT && fitCentroids == null) {
            Dialogs.showWarningNotification("CellTune", "No fit available yet — run Recompute first.");
            return;
        }

        List<String> heatMarkers = (fitMarkers != null) ? fitMarkers : markerFeatures;
        Map<Integer, PathClass> mapping = ClusterAssignmentPane.show(
                stage,
                scope == Scope.PROJECT ? "Assign Cohort Clusters to Classes" : "Assign Clusters to Classes",
                k,
                counts,
                fitCentroids,
                heatMarkers,
                this::availableClassNames,
                plot::clusterColor,
                openClassControl);
        if (mapping == null) {
            return; // cancelled
        }
        if (mapping.isEmpty()) {
            statusLabel.setText("No clusters mapped — nothing changed.");
            return;
        }

        if (scope == Scope.PROJECT) {
            // Bug fix: after a completed "Cluster all cells" write, the persisted Cluster
            // measurement — not the preview Leiden reference (which no longer matches what
            // was actually written) — is the source of truth for cluster identity.
            if (allCellsWriteActive) {
                assignAcrossProjectByMeasurement(mapping);
            } else {
                assignAcrossProject(mapping);
            }
            return;
        }

        int affected = 0;
        for (int i = 0; i < nRows; i++) {
            if (mapping.containsKey(cluster[i])) {
                affected++;
            }
        }
        boolean confirmed = Dialogs.showConfirmDialog(
                "Replace classifications",
                String.format(
                        "Set the QuPath classification of %,d cell(s) across "
                                + "%d cluster(s)? This replaces any existing class on "
                                + "those cells.",
                        affected, mapping.size()));
        if (!confirmed) {
            return;
        }
        applyClusterMapping(mapping);
    }

    /**
     * Streams the cluster → class mapping across the selected project images,
     * assigning every matching cell to its nearest cohort centroid and saving each
     * image. Honors the within-class filter so a sub-clustering only rewrites cells
     * of that class.
     */
    private void assignAcrossProject(Map<Integer, PathClass> mapping) {
        final Project<BufferedImage> project = qupath.getProject();
        if (project == null) {
            Dialogs.showErrorMessage("CellTune", "No project is open.");
            return;
        }
        boolean ok = Dialogs.showConfirmDialog(
                "Assign across project",
                String.format(
                        "Assign %d cluster(s) as classifications to every "
                                + "matching cell across %d image(s)? Each image is saved. "
                                + "This replaces existing classes on the assigned cells.",
                        mapping.size(), projectImages.size()));
        if (!ok) {
            return;
        }

        var available = qupath.getAvailablePathClasses();
        for (PathClass pc : mapping.values()) {
            if (pc != null && !available.contains(pc)) {
                available.add(pc);
            }
        }

        final List<String> images = new ArrayList<>(projectImages);
        final List<String> markers = fitMarkers;
        final double[] mean = fitMean;
        final double[] sd = fitSd;
        // k-means cohort assign uses centroids in the SAME space clustering ran in (PC space when
        // the fit applied PCA) -- NOT fitCentroids, which stays marker-space for the heatmap.
        final double[][] assignCents = fitAssignCentroids;
        final String classFilter = fitClassFilter;
        // Leiden cohort assign uses kNN label transfer against the labelled fitted
        // sample (Task 5); non-null exactly when the last Recompute used Leiden.
        final double[][] leidenRef = fitLeidenReference;
        final int[] leidenRefLabels = fitLeidenReferenceLabels;
        final int nClustersForAssign = fitNClusters;
        final java.util.function.UnaryOperator<double[][]> pcaProjector = fitPcaProjector;

        applying = true;
        setControlsDisabled(true);
        progress.setProgress(ProgressIndicator.INDETERMINATE_PROGRESS);
        progress.setVisible(true);
        statusLabel.setText("Assigning classifications across " + images.size() + " image(s)…");

        new Thread(
                        () -> {
                            try {
                                ImageData<BufferedImage> openData = qupath.getImageData();
                                String openName = null;
                                if (openData != null) {
                                    ProjectImageEntry<BufferedImage> openEntry = project.getEntry(openData);
                                    if (openEntry != null) {
                                        openName = openEntry.getImageName();
                                    }
                                }
                                long total;
                                if (leidenRef != null && leidenRefLabels != null) {
                                    total = CohortClusterModel.assignAcrossProjectLeiden(
                                            project,
                                            images,
                                            markers,
                                            mean,
                                            sd,
                                            leidenRef,
                                            leidenRefLabels,
                                            LEIDEN_GRAPH_K,
                                            nClustersForAssign,
                                            pcaProjector,
                                            mapping,
                                            classFilter,
                                            normalizer,
                                            openData,
                                            openName,
                                            msg -> Platform.runLater(() -> statusLabel.setText(msg)),
                                            frac -> Platform.runLater(() -> progress.setProgress(frac)));
                                } else {
                                    total = CohortClusterModel.assignAcrossProject(
                                            project,
                                            images,
                                            markers,
                                            mean,
                                            sd,
                                            assignCents,
                                            pcaProjector,
                                            mapping,
                                            classFilter,
                                            normalizer,
                                            openData,
                                            openName,
                                            msg -> Platform.runLater(() -> statusLabel.setText(msg)),
                                            frac -> Platform.runLater(() -> progress.setProgress(frac)));
                                }
                                final long fTotal = total;
                                Platform.runLater(() -> {
                                    QuPathViewer viewer = qupath.getViewer();
                                    if (viewer != null) {
                                        viewer.repaint();
                                    }
                                    statusLabel.setText(String.format(
                                            "Done — assigned %,d cell(s) across %d image(s).", fTotal, images.size()));
                                });
                                logger.info("Cohort assign wrote {} cells across {} images", fTotal, images.size());
                            } catch (Throwable t) {
                                logger.error("Project assignment failed", t);
                                Platform.runLater(() -> statusLabel.setText("Assign failed: " + t.getMessage()));
                            } finally {
                                Platform.runLater(() -> {
                                    progress.setVisible(false);
                                    progress.setProgress(ProgressIndicator.INDETERMINATE_PROGRESS);
                                    setControlsDisabled(false);
                                    applying = false;
                                });
                            }
                        },
                        "CellTune-CohortAssign")
                .start();
    }

    /**
     * Streams the cluster→class mapping across the selected project images by reading
     * each cell's already-written {@link CohortClusterModel#CLUSTER_MEASUREMENT} value
     * ({@link CohortClusterModel#assignAcrossProjectByMeasurement}) instead of
     * re-transferring against a preview fit — the correct path when
     * {@link #allCellsWriteActive} is true (the current cluster state came from a
     * completed "Cluster all cells" write, so the persisted measurement is the
     * source of truth, not any in-memory preview reference). Mirrors
     * {@link #assignAcrossProject}'s confirm/thread/progress/save shape.
     */
    private void assignAcrossProjectByMeasurement(Map<Integer, PathClass> mapping) {
        final Project<BufferedImage> project = qupath.getProject();
        if (project == null) {
            Dialogs.showErrorMessage("CellTune", "No project is open.");
            return;
        }
        boolean ok = Dialogs.showConfirmDialog(
                "Assign across project",
                String.format(
                        "Assign %d cluster(s) as classifications to every matching cell across %d "
                                + "image(s), using the written all-cells \"%s\" measurement? Each image is "
                                + "saved. This replaces existing classes on the assigned cells.",
                        mapping.size(), projectImages.size(), CohortClusterModel.CLUSTER_MEASUREMENT));
        if (!ok) {
            return;
        }

        var available = qupath.getAvailablePathClasses();
        for (PathClass pc : mapping.values()) {
            if (pc != null && !available.contains(pc)) {
                available.add(pc);
            }
        }

        final List<String> images = new ArrayList<>(projectImages);

        applying = true;
        setControlsDisabled(true);
        progress.setProgress(ProgressIndicator.INDETERMINATE_PROGRESS);
        progress.setVisible(true);
        statusLabel.setText("Assigning classifications across " + images.size() + " image(s)…");

        new Thread(
                        () -> {
                            try {
                                ImageData<BufferedImage> openData = qupath.getImageData();
                                String openName = null;
                                if (openData != null) {
                                    ProjectImageEntry<BufferedImage> openEntry = project.getEntry(openData);
                                    if (openEntry != null) {
                                        openName = openEntry.getImageName();
                                    }
                                }
                                long total = CohortClusterModel.assignAcrossProjectByMeasurement(
                                        project,
                                        images,
                                        mapping,
                                        openData,
                                        openName,
                                        msg -> Platform.runLater(() -> statusLabel.setText(msg)),
                                        frac -> Platform.runLater(() -> progress.setProgress(frac)));
                                final long fTotal = total;
                                Platform.runLater(() -> {
                                    QuPathViewer viewer = qupath.getViewer();
                                    if (viewer != null) {
                                        viewer.repaint();
                                    }
                                    statusLabel.setText(String.format(
                                            "Done — assigned %,d cell(s) across %d image(s) (from written %s "
                                                    + "measurement).",
                                            fTotal, images.size(), CohortClusterModel.CLUSTER_MEASUREMENT));
                                });
                                logger.info(
                                        "All-cells cohort assign (by measurement) wrote {} cells across {} images",
                                        fTotal,
                                        images.size());
                            } catch (Throwable t) {
                                logger.error("Project assignment (by measurement) failed", t);
                                Platform.runLater(() -> statusLabel.setText("Assign failed: " + t.getMessage()));
                            } finally {
                                Platform.runLater(() -> {
                                    progress.setVisible(false);
                                    progress.setProgress(ProgressIndicator.INDETERMINATE_PROGRESS);
                                    setControlsDisabled(false);
                                    applying = false;
                                });
                            }
                        },
                        "CellTune-CohortAssignMeasurement")
                .start();
    }

    /**
     * Writes the cluster→class mapping onto the cells on a background thread,
     * applying {@code setPathClass} in chunks marshalled to the FX thread so the
     * UI stays responsive (and shows progress) for large cell counts.
     */
    private void applyClusterMapping(Map<Integer, PathClass> mapping) {
        // Register any newly typed classes so they appear in the project list
        // (we are on the FX thread here, before the worker starts).
        var available = qupath.getAvailablePathClasses();
        for (PathClass pc : mapping.values()) {
            if (pc != null && !available.contains(pc)) {
                available.add(pc);
            }
        }

        applying = true;
        setControlsDisabled(true);
        progress.setProgress(0);
        progress.setVisible(true);
        statusLabel.setText("Applying cluster classifications…");

        new Thread(
                        () -> {
                            final int n = nRows;
                            final int chunk = 5000;
                            final List<PathObject> changed = new ArrayList<>();
                            try {
                                for (int start = 0; start < n; start += chunk) {
                                    final int end = Math.min(n, start + chunk);
                                    final List<PathObject> objs = new ArrayList<>();
                                    final List<PathClass> classes = new ArrayList<>();
                                    for (int i = start; i < end; i++) {
                                        PathClass pc = mapping.get(cluster[i]);
                                        if (pc != null) {
                                            objs.add(cells[i]);
                                            classes.add(pc);
                                        }
                                    }

                                    final java.util.concurrent.CountDownLatch latch =
                                            new java.util.concurrent.CountDownLatch(1);
                                    Platform.runLater(() -> {
                                        try {
                                            for (int j = 0; j < objs.size(); j++) {
                                                objs.get(j).setPathClass(classes.get(j));
                                            }
                                        } finally {
                                            latch.countDown();
                                        }
                                    });
                                    latch.await();

                                    changed.addAll(objs);
                                    final double frac = end / (double) n;
                                    final int done = changed.size();
                                    Platform.runLater(() -> {
                                        progress.setProgress(frac);
                                        statusLabel.setText(String.format("Applying classifications… %,d cells", done));
                                    });
                                }

                                Platform.runLater(() -> {
                                    hierarchy.fireObjectClassificationsChangedEvent(this, changed);
                                    QuPathViewer viewer = qupath.getViewer();
                                    if (viewer != null) {
                                        viewer.repaint();
                                    }
                                    plot.redraw(); // refresh CLASS colouring on the plot
                                    statusLabel.setText(String.format(
                                            "Applied %d cluster→class mapping(s) to %,d cell(s).",
                                            mapping.size(), changed.size()));
                                });
                                logger.info(
                                        "Applied cluster→class mapping to {} cells ({} clusters)",
                                        changed.size(),
                                        mapping.size());
                            } catch (InterruptedException ie) {
                                Thread.currentThread().interrupt();
                                Platform.runLater(() -> statusLabel.setText("Apply cancelled."));
                            } finally {
                                Platform.runLater(() -> {
                                    progress.setVisible(false);
                                    progress.setProgress(ProgressIndicator.INDETERMINATE_PROGRESS);
                                    setControlsDisabled(false);
                                    applying = false;
                                });
                            }
                        },
                        "CellTune-ApplyClusters")
                .start();
    }

    // ── Non-destructive viewer overlay: paint cells by cluster (CN-dialog parity) ─────

    /**
     * Assign every cell in the open image to its nearest phenotype cluster (nearest-centroid
     * for k-means, kNN label transfer for Leiden against the labelled fitted sample), write
     * the id as a non-destructive numeric {@link CohortClusterModel#CLUSTER_MEASUREMENT}, and activate a
     * {@link MeasurementMapper} overlay so the tissue is painted by cluster — the phenotype
     * analogue of the CN dialog's spatial CN colouring. The classification is never changed;
     * the overlay is reset when this window closes or "By classification" is clicked.
     *
     * <p>When the fit used a within-class filter (sub-clustering), only cells of that class
     * are painted; the rest keep their phenotype colour. Runs off the FX thread — for Leiden
     * on a very large image the kNN label transfer can take a while (a progress spinner shows).
     */
    private void showClusterOverlay() {
        if (applying) {
            return;
        }
        QuPathViewer viewer = qupath.getViewer();
        if (viewer == null || viewer.getImageData() == null) {
            Dialogs.showWarningNotification("CellTune", "No open image to recolour.");
            return;
        }
        boolean haveFit = fitCentroids != null || (fitLeidenReference != null && fitLeidenReferenceLabels != null);
        if (!haveFit) {
            Dialogs.showWarningNotification("CellTune", "No clusters available yet — run Recompute first.");
            return;
        }

        // Fast path: the Cluster measurement is already written for the current fit and the
        // open image still carries it, so toggling back from "By classification" is just a
        // display flip — re-activate the overlay without re-running the (possibly cohort-wide,
        // saving) assignment.
        if (!clusterMeasurementStale && openImageHasClusterMeasurement()) {
            activateClusterMapper(viewer, fitNClusters);
            clusterOverlayActive = true;
            statusLabel.setText("Showing cluster overlay.");
            return;
        }

        // Project scope: write the Cluster measurement to every cell across every selected
        // image (persistent + saved), then overlay the open one. Current-image scope stays
        // an in-memory, single-image overlay below.
        if (scope == Scope.PROJECT) {
            // Cohort-mode radio pair (D-04): all-cells is the true-scanpy two-pass driver;
            // transfer is the retained Phase 14 kNN label-transfer path (unchanged dispatch).
            if (methodCombo.getValue() == ClusterMethod.LEIDEN && cohortModeAllCells.isSelected()) {
                writeClusterAllCellsAcrossProject();
            } else {
                writeClusterMeasurementAcrossProject();
            }
            return;
        }

        final ImageData<BufferedImage> data = viewer.getImageData();
        final List<String> markers = (fitMarkers != null) ? fitMarkers : markerFeatures;
        final double[] mean = fitMean;
        final double[] sd = fitSd;
        final double[][] cents = fitCentroids;
        // Clustering-space centroids (PC space when PCA was applied, else == cents) for the
        // k-means overlay, so single-image assignment matches the previewed partition (same
        // fix as the cohort paths). fitPcaProjector below projects the query into that space.
        final double[][] assignCents = fitAssignCentroids;
        final double[][] leidenRef = fitLeidenReference;
        final int[] leidenRefLabels = fitLeidenReferenceLabels;
        final int nClusters = fitNClusters;
        final String classFilter = fitClassFilter;
        final java.util.function.UnaryOperator<double[][]> pcaProjector = fitPcaProjector;

        applying = true;
        setControlsDisabled(true);
        progress.setProgress(ProgressIndicator.INDETERMINATE_PROGRESS);
        progress.setVisible(true);
        statusLabel.setText("Assigning clusters across the image…");

        new Thread(
                        () -> {
                            try {
                                var hier = data.getHierarchy();
                                var cellCol = hier.getCellObjects();
                                List<PathObject> cells =
                                        new ArrayList<>(cellCol.isEmpty() ? hier.getDetectionObjects() : cellCol);
                                int n = cells.size();
                                int nFeat = markers.size();
                                float[] flat = new CellFeatureExtractor(markers, normalizer).extractMatrix(cells);
                                boolean standardize = mean != null && sd != null;

                                // Standardized feature matrix in the same space as the fit.
                                double[][] z = new double[n][nFeat];
                                for (int i = 0; i < n; i++) {
                                    int off = i * nFeat;
                                    for (int j = 0; j < nFeat; j++) {
                                        double v = flat[off + j];
                                        z[i][j] = standardize ? (sd[j] < 1e-9 ? 0.0 : (v - mean[j]) / sd[j]) : v;
                                    }
                                }

                                // Label every cell: Leiden -> kNN transfer vs the labelled fit sample; else nearest
                                // centroid.
                                int[] labels;
                                if (leidenRef != null && leidenRefLabels != null) {
                                    // Project the query rows into the SAME space fitLeidenReference was
                                    // clustered in — identity when the fit did not apply PCA.
                                    double[][] zProjected = pcaProjector != null ? pcaProjector.apply(z) : z;
                                    labels = LeidenModel.transferLabels(
                                            zProjected,
                                            leidenRef,
                                            leidenRefLabels,
                                            LEIDEN_GRAPH_K,
                                            Math.max(1, nClusters));
                                } else {
                                    // Project the query rows into the same space the k-means
                                    // partition lives in (identity when the fit did not apply PCA),
                                    // then match against the clustering-space centroids — so the
                                    // overlay agrees with the previewed partition under PCA.
                                    double[][] zProjected = pcaProjector != null ? pcaProjector.apply(z) : z;
                                    labels = new int[n];
                                    for (int i = 0; i < n; i++) {
                                        labels[i] = nearestCentroidIndex(zProjected[i], assignCents);
                                    }
                                }

                                // 1-based ids; cells outside the clustered (within-class) population get -1.
                                double[] values = new double[n];
                                int assigned = 0;
                                for (int i = 0; i < n; i++) {
                                    if (labels[i] >= 0 && matchesClassFilter(cells.get(i), classFilter)) {
                                        values[i] = labels[i] + 1.0;
                                        assigned++;
                                    } else {
                                        values[i] = -1.0;
                                    }
                                }

                                final int fAssigned = assigned;
                                Platform.runLater(() -> {
                                    for (int i = 0; i < cells.size(); i++) {
                                        cells.get(i)
                                                .getMeasurementList()
                                                .put(CohortClusterModel.CLUSTER_MEASUREMENT, values[i]);
                                    }
                                    hier.fireHierarchyChangedEvent(this);
                                    activateClusterMapper(viewer, nClusters);
                                    clusterOverlayActive = true;
                                    clusterMeasurementStale = false; // measurement now matches the fit
                                    statusLabel.setText(String.format(
                                            "Painted %,d cell(s) by cluster (%d cluster(s)) — non-destructive.",
                                            fAssigned, Math.max(1, nClusters)));
                                });
                            } catch (Throwable t) {
                                logger.error("Cluster overlay failed", t);
                                Platform.runLater(() -> statusLabel.setText("Overlay failed: " + t.getMessage()));
                            } finally {
                                Platform.runLater(() -> {
                                    progress.setVisible(false);
                                    progress.setProgress(ProgressIndicator.INDETERMINATE_PROGRESS);
                                    setControlsDisabled(false);
                                    applying = false;
                                });
                            }
                        },
                        "CellTune-ClusterOverlay")
                .start();
    }

    /**
     * Project-scope "By cluster": assign every cell across every selected image to its
     * nearest cluster and write the non-destructive {@link CohortClusterModel#CLUSTER_MEASUREMENT} to each,
     * saving every image (the persistent, cohort-wide analogue of the single-image overlay).
     * Then activate the viewer overlay on whichever image is currently open. The cell
     * classification is never changed.
     */
    private void writeClusterMeasurementAcrossProject() {
        final Project<BufferedImage> project = qupath.getProject();
        if (project == null) {
            Dialogs.showErrorMessage("CellTune", "No project is open.");
            return;
        }
        boolean ok = Dialogs.showConfirmDialog(
                "Write cluster measurement across project",
                String.format(
                        "Assign a cluster to every cell across %d image(s), writing a non-destructive "
                                + "\"%s\" measurement, and save each image? Classifications are not changed.",
                        projectImages.size(), CohortClusterModel.CLUSTER_MEASUREMENT));
        if (!ok) {
            return;
        }

        final List<String> images = new ArrayList<>(projectImages);
        final List<String> markers = (fitMarkers != null) ? fitMarkers : markerFeatures;
        final double[] mean = fitMean;
        final double[] sd = fitSd;
        // k-means cohort write uses centroids in the SAME space clustering ran in (PC space when
        // the fit applied PCA) -- NOT fitCentroids, which stays marker-space for the heatmap.
        final double[][] assignCents = fitAssignCentroids;
        final double[][] leidenRef = fitLeidenReference;
        final int[] leidenRefLabels = fitLeidenReferenceLabels;
        final int nClusters = fitNClusters;
        final String classFilter = fitClassFilter;
        final java.util.function.UnaryOperator<double[][]> pcaProjector = fitPcaProjector;

        applying = true;
        setControlsDisabled(true);
        progress.setProgress(0);
        progress.setVisible(true);
        statusLabel.setText("Writing cluster measurement across " + images.size() + " image(s)…");

        new Thread(
                        () -> {
                            try {
                                ImageData<BufferedImage> openData = qupath.getImageData();
                                String openName = null;
                                if (openData != null) {
                                    ProjectImageEntry<BufferedImage> openEntry = project.getEntry(openData);
                                    if (openEntry != null) {
                                        openName = openEntry.getImageName();
                                    }
                                }
                                long total;
                                if (leidenRef != null && leidenRefLabels != null) {
                                    total = CohortClusterModel.writeClusterAcrossProjectLeiden(
                                            project,
                                            images,
                                            markers,
                                            mean,
                                            sd,
                                            leidenRef,
                                            leidenRefLabels,
                                            LEIDEN_GRAPH_K,
                                            nClusters,
                                            pcaProjector,
                                            classFilter,
                                            normalizer,
                                            openData,
                                            openName,
                                            msg -> Platform.runLater(() -> statusLabel.setText(msg)),
                                            frac -> Platform.runLater(() -> progress.setProgress(frac)));
                                } else {
                                    total = CohortClusterModel.writeClusterAcrossProject(
                                            project,
                                            images,
                                            markers,
                                            mean,
                                            sd,
                                            assignCents,
                                            pcaProjector,
                                            classFilter,
                                            normalizer,
                                            openData,
                                            openName,
                                            msg -> Platform.runLater(() -> statusLabel.setText(msg)),
                                            frac -> Platform.runLater(() -> progress.setProgress(frac)));
                                }
                                final long fTotal = total;
                                Platform.runLater(() -> {
                                    QuPathViewer viewer = qupath.getViewer();
                                    if (viewer != null && viewer.getImageData() != null) {
                                        activateClusterMapper(viewer, nClusters);
                                        clusterOverlayActive = true;
                                    }
                                    clusterMeasurementStale =
                                            false; // measurement now matches the fit across the cohort
                                    statusLabel.setText(String.format(
                                            "Wrote cluster to %,d cell(s) across %d image(s) — saved, non-destructive.",
                                            fTotal, images.size()));
                                });
                                logger.info(
                                        "Cohort cluster measurement wrote {} cells across {} images",
                                        total,
                                        images.size());
                            } catch (Throwable t) {
                                logger.error("Project cluster-measurement write failed", t);
                                Platform.runLater(() -> statusLabel.setText("Cluster write failed: " + t.getMessage()));
                            } finally {
                                Platform.runLater(() -> {
                                    progress.setVisible(false);
                                    progress.setProgress(ProgressIndicator.INDETERMINATE_PROGRESS);
                                    setControlsDisabled(false);
                                    applying = false;
                                });
                            }
                        },
                        "CellTune-ClusterMeasure")
                .start();
    }

    /**
     * Project-scope "Cluster all cells" (D-04/D-05, LEI-06): the true-scanpy two-pass all-cells
     * driver. Unlike {@link #writeClusterMeasurementAcrossProject}, this pools EVERY cell across
     * the selected images (no sample cap) into ONE HNSW kNN graph and runs a SINGLE
     * {@link LeidenModel#clusterViaAnn} partition over the whole cohort, then writes the
     * {@link CohortClusterModel#CLUSTER_MEASUREMENT} back by UUID lookup ({@link CohortClusterModel#writeClusterAllCells}).
     * Adds a soft-ceiling confirm (D-10), per-phase progress + a mid-run Cancel (D-11/D-12), and
     * re-syncs the legend to the FINAL all-cells cluster count (LEI-09), not the preview
     * subsample's {@link #fitNClusters}.
     * <p>
     * Bug fix: on any non-aborted write (including one the user CANCELLED partway through the
     * write loop — clustering itself always completes before that loop starts, so the partition
     * is already fully valid by then), this installs the FULL all-cells fit-state (not just the
     * on-slide overlay via {@link #activateClusterMapper}) — {@link #fitNClusters}/
     * {@link #fitCentroids}/{@link #fitAllCellsCounts}/{@link #fitMarkers}/{@link #fitMean}/
     * {@link #fitSd} — and sets {@link #allCellsWriteActive}, so the scatter legend
     * ({@code plot.redraw()}) and the "Assign Cohort Clusters to Classes" dialog
     * ({@link #applyClustersToClasses}, via {@link #clusterCounts}) both reflect the all-cells
     * partition, and subsequent cluster→class assignment reads the WRITTEN measurement
     * ({@link #assignAcrossProjectByMeasurement}) instead of re-transferring against the now-stale
     * preview Leiden reference — even for the images that already got the write before cancel.
     */
    private void writeClusterAllCellsAcrossProject() {
        final Project<BufferedImage> project = qupath.getProject();
        if (project == null) {
            Dialogs.showErrorMessage("CellTune", "No project is open.");
            return;
        }
        boolean ok = Dialogs.showConfirmDialog(
                "Cluster all cells across project",
                String.format(
                        "Pool EVERY cell across %d image(s) into one graph, run a single Leiden partition "
                                + "over the whole cohort, and write a non-destructive \"%s\" measurement to every "
                                + "cell, saving each image? This clusters the full cohort (not a sample) and can "
                                + "take a long time on large projects. Classifications are not changed.",
                        projectImages.size(), CohortClusterModel.CLUSTER_MEASUREMENT));
        if (!ok) {
            return;
        }

        final List<String> images = new ArrayList<>(projectImages);
        final List<String> markers = (fitMarkers != null) ? fitMarkers : markerFeatures;
        final double resolution = resolutionSpinner.getValue();
        final boolean reproducible = reproducibleCheck.isSelected();
        final int randomStarts = reproducible ? LEIDEN_REPRODUCIBLE_STARTS : LEIDEN_SINGLE_START;
        final long seed = reproducible ? LEIDEN_SEED : System.nanoTime();
        final boolean pcaEnabled = pcaEnabledCheck.isSelected();
        final int pcaMaxComponents = pcaComponentsSpinner.getValue();
        final String classFilter = fitClassFilter;
        final CohortClusterModel.CancellationToken token = new CohortClusterModel.CancellationToken();
        allCellsToken = token;

        applying = true;
        setControlsDisabled(true);
        cancelAllCellsBtn.setDisable(false);
        cancelAllCellsBtn.setVisible(true);
        progress.setProgress(0);
        progress.setVisible(true);
        statusLabel.setText("Estimating pooled cell count across " + images.size() + " image(s)…");

        new Thread(
                        () -> {
                            try {
                                ImageData<BufferedImage> openData = qupath.getImageData();
                                String openName = null;
                                if (openData != null) {
                                    ProjectImageEntry<BufferedImage> openEntry = project.getEntry(openData);
                                    if (openEntry != null) {
                                        openName = openEntry.getImageName();
                                    }
                                }

                                // Soft ceiling (D-10): a cheap count-only pre-scan (no feature
                                // extraction) so an accidental oversized run gets an extra confirm
                                // before pass 1/2 even start.
                                long estimate = estimatePooledCellCount(project, images, token);
                                if (token.isCancelled()) {
                                    Platform.runLater(() -> statusLabel.setText("Cancelled before pooling started."));
                                    return;
                                }
                                int ceiling = ALL_CELLS_SOFT_CEILING.get();
                                if (estimate > ceiling) {
                                    boolean proceed = confirmOnFx(
                                            "Large all-cells run",
                                            String.format(
                                                    "The selected images have an estimated %,d cells, above the "
                                                            + "configured %,d-cell soft ceiling. This can take a "
                                                            + "long time and use a lot of memory. Continue anyway?",
                                                    estimate, ceiling));
                                    if (!proceed) {
                                        Platform.runLater(() -> statusLabel.setText(String.format(
                                                "Cancelled — estimated %,d cells exceeds the soft ceiling.",
                                                estimate)));
                                        return;
                                    }
                                }

                                CohortClusterModel.AllCellsResult result = CohortClusterModel.writeClusterAllCells(
                                        project,
                                        images,
                                        markers,
                                        LEIDEN_GRAPH_K,
                                        resolution,
                                        randomStarts,
                                        seed,
                                        reproducible,
                                        pcaEnabled,
                                        pcaMaxComponents,
                                        classFilter,
                                        normalizer,
                                        BatchCorrection.loadIfEnabled(project),
                                        openData,
                                        openName,
                                        token,
                                        msg -> Platform.runLater(() -> statusLabel.setText(msg)),
                                        frac -> Platform.runLater(() -> progress.setProgress(frac)));

                                if (result.aborted()) {
                                    Platform.runLater(() -> Dialogs.showErrorMessage(
                                            "CellTune",
                                            "ANN recall gate failed — no Cluster measurement was written. Try "
                                                    + "more cells, different markers, or re-run. (See the status "
                                                    + "log for the measured recall.)"));
                                    return;
                                }

                                final int fNClusters = result.nClusters();
                                final long fCellsWritten = result.cellsWritten();
                                final int fWrittenCount = result.imagesWritten().size();
                                final int fNotWrittenCount =
                                        result.imagesNotWritten().size();
                                final boolean fCancelled = result.cancelled();
                                final double fRecall = result.recall();
                                final double[][] fCentroids = result.centroids();
                                final int[] fClusterCounts = result.clusterCounts();
                                final double[] fPooledMean = result.mean();
                                final double[] fPooledSd = result.sd();
                                Platform.runLater(() -> {
                                    // Bug fix (T-16-xx / cancel-mid-write re-sync gap): result.aborted() already
                                    // returned above, so by this point the all-cells partition (nClusters/
                                    // centroids/counts) is always fully valid — clustering completes BEFORE the
                                    // write loop runs, and a cancel only stops the write loop partway through
                                    // writing already-clustered images. Install the fit-state whenever the run
                                    // was NOT aborted (guarding defensively on a populated centroids/nClusters
                                    // pair, which also excludes the degenerate empty-images case), NOT only when
                                    // the write wasn't cancelled — otherwise a cancelled-but-valid write leaves
                                    // the legend/Assign dialog bound to the stale preview fit while some images
                                    // already carry the real all-cells Cluster measurement.
                                    if (fCentroids != null && fNClusters > 0) {
                                        // Bug fix (LEI-09 re-sync gap): install the ALL-CELLS fit-state —
                                        // not just re-color the on-slide overlay — so the scatter legend
                                        // AND the "Assign Cohort Clusters to Classes" dialog both reflect
                                        // the final all-cells cluster count/heatmap, not the preview
                                        // subsample's last Recompute fit.
                                        fitNClusters = fNClusters;
                                        fitCentroids = fCentroids;
                                        // Not a k-means fit — the all-cells write is Leiden-only (D-04/D-05), and
                                        // cluster->class assignment reads the WRITTEN measurement
                                        // (allCellsWriteActive),
                                        // never assignAcrossProject/writeClusterAcrossProject's centroid path.
                                        fitAssignCentroids = null;
                                        fitAllCellsCounts = fClusterCounts;
                                        fitMarkers = markers;
                                        fitMean = fPooledMean;
                                        fitSd = fPooledSd;
                                        // Not a preview Leiden kNN-transfer fit — cluster->class assignment
                                        // must read the written measurement instead (see allCellsWriteActive).
                                        fitLeidenReference = null;
                                        fitLeidenReferenceLabels = null;
                                        fitPcaProjector = null;
                                        fitPcaApplied = false;
                                        allCellsWriteActive = true;

                                        QuPathViewer viewer = qupath.getViewer();
                                        if (viewer != null && viewer.getImageData() != null) {
                                            // LEI-09: re-sync to the FINAL all-cells cluster count, not
                                            // the preview subsample's fitNClusters.
                                            activateClusterMapper(viewer, fNClusters);
                                            clusterOverlayActive = true;
                                        }
                                        clusterMeasurementStale = false;
                                        // Rebuild the scatter legend now that fitNClusters/fitCentroids
                                        // reflect the all-cells partition (PlotModel.clusterCount() reads
                                        // fitNClusters — redraw so the legend picks it up immediately).
                                        plot.redraw();
                                    }
                                    // D-09: report the measured ANN recall from the all-cells partition
                                    // (LeidenModel.LeidenResult.recall(), threaded through
                                    // AllCellsResult.recall()); the >= 0 guard now only ever excludes the
                                    // aborted/cancel-before-clustering early-return sentinel (-1.0), never
                                    // a real successful run — see CohortClusterModel.AllCellsResult javadoc.
                                    String recallMsg =
                                            fRecall >= 0 ? String.format("ANN recall %.3f — passed. ", fRecall) : "";
                                    if (fCancelled) {
                                        statusLabel.setText(String.format(
                                                "%sCancelled — wrote cluster to %,d cell(s) across %d image(s); "
                                                        + "%d image(s) not written.",
                                                recallMsg, fCellsWritten, fWrittenCount, fNotWrittenCount));
                                    } else {
                                        statusLabel.setText(String.format(
                                                "%sWrote cluster to %,d cell(s) across %d image(s) (%d cluster(s)) "
                                                        + "— saved, non-destructive.",
                                                recallMsg, fCellsWritten, fWrittenCount, fNClusters));
                                    }
                                });
                                logger.info(
                                        "All-cells cluster measurement wrote {} cells across {} image(s) "
                                                + "({} cluster(s), cancelled={})",
                                        fCellsWritten,
                                        fWrittenCount,
                                        fNClusters,
                                        fCancelled);
                            } catch (Throwable t) {
                                logger.error("All-cells cluster write failed", t);
                                Platform.runLater(
                                        () -> statusLabel.setText("All-cells cluster write failed: " + t.getMessage()));
                            } finally {
                                Platform.runLater(() -> {
                                    progress.setVisible(false);
                                    progress.setProgress(ProgressIndicator.INDETERMINATE_PROGRESS);
                                    setControlsDisabled(false);
                                    cancelAllCellsBtn.setVisible(false);
                                    allCellsToken = null;
                                    applying = false;
                                });
                            }
                        },
                        "CellTune-AllCellsCluster")
                .start();
    }

    /**
     * Cheap count-only pre-scan (D-10, RESEARCH.md Open Question 3): sums each selected image's
     * detection count WITHOUT extracting marker features, so the soft-ceiling estimate costs one
     * extra full project read rather than duplicating pass 1's feature extraction. Honors
     * cancellation between images.
     */
    private long estimatePooledCellCount(
            Project<BufferedImage> project, List<String> images, CohortClusterModel.CancellationToken token) {
        Map<String, ProjectImageEntry<BufferedImage>> byName = new java.util.LinkedHashMap<>();
        for (ProjectImageEntry<BufferedImage> entry : project.getImageList()) {
            byName.put(entry.getImageName(), entry);
        }
        long total = 0;
        for (String name : images) {
            if (token.isCancelled()) {
                break;
            }
            ProjectImageEntry<BufferedImage> entry = byName.get(name);
            if (entry == null) {
                continue;
            }
            try {
                ImageData<BufferedImage> imageData = entry.readImageData();
                if (imageData != null) {
                    total += imageData.getHierarchy().getDetectionObjects().size();
                }
            } catch (Exception e) {
                logger.debug("Soft-ceiling pre-scan: could not read [{}] — skipped", name, e);
            }
        }
        return total;
    }

    /**
     * Shows a confirm dialog from a background thread by marshalling it onto the FX thread and
     * blocking (via {@link java.util.concurrent.CountDownLatch}) until the user answers — mirrors
     * {@code CohortClusterModel.applyMeasurement}'s existing FX-marshalling shape.
     */
    private boolean confirmOnFx(String title, String message) {
        final boolean[] result = {false};
        var latch = new java.util.concurrent.CountDownLatch(1);
        Platform.runLater(() -> {
            try {
                result[0] = Dialogs.showConfirmDialog(title, message);
            } finally {
                latch.countDown();
            }
        });
        try {
            latch.await();
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        }
        return result[0];
    }

    /** Reverts the viewer to native classification colours (removes the cluster overlay). */
    private void showClassificationView() {
        QuPathViewer viewer = qupath.getViewer();
        if (viewer != null) {
            viewer.getOverlayOptions().resetMeasurementMapper();
            viewer.repaintEntireImage();
        }
        clusterOverlayActive = false;
        statusLabel.setText("Showing classification colours.");
    }

    /** Removes any active cluster overlay (called when the window closes). Null-safe, silent. */
    private void clearClusterOverlay() {
        if (!clusterOverlayActive) {
            return;
        }
        QuPathViewer viewer = qupath.getViewer();
        if (viewer != null) {
            viewer.getOverlayOptions().resetMeasurementMapper();
            viewer.repaintEntireImage();
        }
        clusterOverlayActive = false;
    }

    /**
     * Builds and installs a {@link MeasurementMapper} that paints each cell by its 1-based
     * {@link CohortClusterModel#CLUSTER_MEASUREMENT} using the plot's per-cluster colours, so the image matches
     * the scatter-plot legend. Cells with -1 keep their phenotype colour.
     */
    private void activateClusterMapper(QuPathViewer viewer, int nClusters) {
        int n = Math.max(2, nClusters);
        int[] r = new int[n];
        int[] g = new int[n];
        int[] b = new int[n];
        for (int c = 0; c < n; c++) {
            Color col = plot.clusterColor(Math.min(c, Math.max(0, nClusters - 1)));
            r[c] = (int) Math.round(col.getRed() * 255);
            g[c] = (int) Math.round(col.getGreen() * 255);
            b[c] = (int) Math.round(col.getBlue() * 255);
        }
        ColorMaps.ColorMap cm = ColorMaps.createColorMap(CohortClusterModel.CLUSTER_MEASUREMENT, r, g, b);
        MeasurementMapper mm = new MeasurementMapper(
                cm,
                CohortClusterModel.CLUSTER_MEASUREMENT,
                viewer.getImageData().getHierarchy().getDetectionObjects());
        mm.setDisplayMinValue(1);
        mm.setDisplayMaxValue(Math.max(2, nClusters));
        mm.setExcludeOutsideRange(true); // -1 (unclustered) cells keep their phenotype colour
        viewer.getOverlayOptions().setMeasurementMapper(mm);
        viewer.repaintEntireImage();
    }

    /** True when the open image's cells already carry a {@link CohortClusterModel#CLUSTER_MEASUREMENT} value. */
    private boolean openImageHasClusterMeasurement() {
        QuPathViewer viewer = qupath.getViewer();
        if (viewer == null || viewer.getImageData() == null) {
            return false;
        }
        var hier = viewer.getImageData().getHierarchy();
        var col = hier.getCellObjects();
        var cells = col.isEmpty() ? hier.getDetectionObjects() : col;
        for (PathObject cell : cells) {
            if (!Double.isNaN(cell.getMeasurementList().get(CohortClusterModel.CLUSTER_MEASUREMENT))) {
                return true;
            }
        }
        return false;
    }

    /** Argmin Euclidean distance to the (z-scored) centroids; -1 if none. */
    private static int nearestCentroidIndex(double[] z, double[][] centroids) {
        if (centroids == null || centroids.length == 0) {
            return -1;
        }
        int best = -1;
        double bestD = Double.POSITIVE_INFINITY;
        for (int c = 0; c < centroids.length; c++) {
            double[] cen = centroids[c];
            double d = 0;
            int len = Math.min(z.length, cen.length);
            for (int j = 0; j < len; j++) {
                double diff = z[j] - cen[j];
                d += diff * diff;
            }
            if (d < bestD) {
                bestD = d;
                best = c;
            }
        }
        return best;
    }

    /** True when the cell's classification contains {@code keyword} (blank/null matches all); mirrors the fit filter. */
    private static boolean matchesClassFilter(PathObject cell, String keyword) {
        if (keyword == null || keyword.isBlank()) {
            return true;
        }
        PathClass pc = cell.getPathClass();
        return pc != null && pc.toString().toLowerCase().contains(keyword.trim().toLowerCase());
    }

    private void setControlsDisabled(boolean disabled) {
        embeddingCombo.setDisable(disabled);
        fullUmapCheck.setDisable(disabled || embeddingCombo.getValue() != Embedding.UMAP);
        kSpinner.setDisable(disabled);
        pcaEnabledCheck.setDisable(disabled);
        pcaComponentsSpinner.setDisable(disabled || !pcaEnabledCheck.isSelected());
        annotationField.setDisable(disabled);
        classField.setDisable(disabled);
        clusterMarkersBtn.setDisable(disabled);
        applyClustersBtn.setDisable(disabled);
        clusterOverlayBtn.setDisable(disabled);
        classViewBtn.setDisable(disabled);
        imageScopeToggle.setDisable(disabled);
        projectScopeToggle.setDisable(disabled);
        sampleSpinner.setDisable(disabled);
        imagesBtn.setDisable(disabled);
        reSampleBtn.setDisable(disabled);
        if (!disabled) {
            applyScopeOverrides();
        }
    }

    /**
     * Applies scope-specific control state on top of the enabled baseline: in
     * project scope the annotation filter is meaningless (annotations live in one
     * image's hierarchy) so it is disabled, the Images… button is shown, and the
     * apply button reads as a cohort-wide assign. The Sample cap + Re-sample stay
     * visible in both scopes.
     */
    private void applyScopeOverrides() {
        boolean project = scope == Scope.PROJECT;
        annotationField.setDisable(project);
        projectControls.setVisible(project);
        applyClustersBtn.setText(project ? "Assign Clusters…" : "Apply Clusters…");
        // In project scope "By cluster" writes + saves the Cluster measurement across all
        // selected images; in current-image scope it's an in-memory single-image overlay.
        clusterOverlayBtn.setText(project ? "By cluster (all images)" : "By cluster");
        imageScopeToggle.setSelected(!project);
        projectScopeToggle.setSelected(project);
        updateCohortModeVisibility();
    }

    /**
     * Net visibility for the cohort-mode radio pair (D-04): visible only when BOTH the scope is
     * PROJECT and the selected Method is LEIDEN — driven from here (scope changes) and from
     * {@code methodCombo}'s value listener (method changes), never duplicated elsewhere.
     */
    private void updateCohortModeVisibility() {
        boolean visible = scope == Scope.PROJECT && methodCombo.getValue() == ClusterMethod.LEIDEN;
        cohortModeAllCells.setVisible(visible);
        cohortModeTransfer.setVisible(visible);
    }

    // ── Scope switching ─────────────────────────────────────────────────────────

    /** Switches back to clustering the open image, re-reading its detections. */
    private void switchToImageScope() {
        if (scope == Scope.CURRENT_IMAGE) {
            applyScopeOverrides();
            return;
        }
        if (hierarchy == null) {
            Dialogs.showErrorMessage("CellTune", "No image is open.");
            projectScopeToggle.setSelected(true);
            return;
        }
        List<PathObject> live = hierarchy.getObjects(null, PathObject.class).stream()
                .filter(PathObjectFilter.DETECTIONS_ALL)
                .toList();
        java.util.Arrays.fill(selected, false);
        loadCurrentImageData(live);
        applyScopeOverrides();
        statusLabel.setText(currentImageLoadedMessage(live.size()));
        plot.redraw();
    }

    /**
     * Switches to project scope: prompts for images but does NOT sample yet.
     * Sampling waits for an explicit “Re-sample” (or “Recompute”).
     */
    private void switchToProjectScope() {
        var project = qupath.getProject();
        if (project == null) {
            Dialogs.showErrorMessage("CellTune", "Open a project to cluster across multiple images.");
            imageScopeToggle.setSelected(true);
            return;
        }
        List<String> chosen = pickImages(project);
        if (chosen == null || chosen.isEmpty()) {
            // Cancelled: restore the toggle to whatever scope is actually active.
            applyScopeOverrides();
            return;
        }
        projectImages = chosen;
        enterEmptyProjectScope();
    }

    /** Re-picks the project images (project scope only); does not sample. */
    private void chooseProjectImages() {
        var project = qupath.getProject();
        if (project == null) {
            return;
        }
        List<String> chosen = pickImages(project);
        if (chosen == null || chosen.isEmpty()) {
            return;
        }
        projectImages = chosen;
        enterEmptyProjectScope();
    }

    /**
     * Enters project scope with no rows loaded yet, prompting the user to draw a
     * sample. Keeps the plot empty until “Re-sample”/“Recompute” is clicked.
     */
    private void enterEmptyProjectScope() {
        scope = Scope.PROJECT;
        nRows = 0;
        cells = null;
        indexOf = null;
        rowClass = null;
        rowImage = null;
        ex = new double[0];
        ey = new double[0];
        cluster = new int[0];
        selected = new boolean[0];
        raw = new double[0][];
        std = new double[0][];
        clearFit();
        applyScopeOverrides();
        statusLabel.setText(String.format(
                "Picked %d image(s) — click “Re-sample” to draw a sample, then " + "“Recompute” to cluster.",
                projectImages.size()));
        plot.redraw();
    }

    /**
     * Draws a fresh sample using the current cap, scope-aware: a pooled cohort
     * sample in project scope, or a random subsample of the open image in
     * current-image scope. Does not cluster.
     */
    private void reSample() {
        commitSpinnerEditor(sampleSpinner);
        if (scope == Scope.PROJECT) {
            if (qupath.getProject() == null) {
                return;
            }
            if (projectImages == null || projectImages.isEmpty()) {
                statusLabel.setText("Click “Images…” to choose project images first.");
                return;
            }
            runProjectSample(false);
        } else {
            if (hierarchy == null) {
                statusLabel.setText("No image is open.");
                return;
            }
            List<PathObject> live = hierarchy.getObjects(null, PathObject.class).stream()
                    .filter(PathObjectFilter.DETECTIONS_ALL)
                    .toList();
            java.util.Arrays.fill(selected, false);
            loadCurrentImageData(live);
            applyScopeOverrides();
            statusLabel.setText(currentImageLoadedMessage(live.size()));
            plot.redraw();
        }
    }

    /**
     * Forces an editable spinner to commit its typed text to the value (JavaFX
     * does not do this until focus leaves the editor), clamping to range.
     */
    private static void commitSpinnerEditor(Spinner<Integer> spinner) {
        if (!spinner.isEditable()) {
            return;
        }
        var factory = spinner.getValueFactory();
        if (factory == null) {
            return;
        }
        try {
            Integer parsed =
                    factory.getConverter().fromString(spinner.getEditor().getText());
            if (parsed != null) {
                factory.setValue(parsed);
            }
        } catch (RuntimeException ex) {
            // Unparseable text: restore the editor to the last valid value.
            spinner.getEditor().setText(factory.getConverter().toString(factory.getValue()));
        }
    }

    /**
     * Forces an editable {@code Spinner<Double>} to commit its typed text to the
     * value, clamping to {@code [min, max]} (mirrors the {@code Integer} overload
     * above; the Leiden resolution spinner uses this).
     */
    private static void commitSpinnerEditor(Spinner<Double> spinner, double min, double max) {
        if (!spinner.isEditable()) {
            return;
        }
        var factory = spinner.getValueFactory();
        if (factory == null) {
            return;
        }
        try {
            Double parsed =
                    factory.getConverter().fromString(spinner.getEditor().getText());
            if (parsed != null) {
                factory.setValue(Math.max(min, Math.min(max, parsed)));
            }
        } catch (RuntimeException ex) {
            spinner.getEditor().setText(factory.getConverter().toString(factory.getValue()));
        }
    }

    private List<String> pickImages(Project<BufferedImage> project) {
        List<String> allNames = new ArrayList<>();
        for (var entry : project.getImageList()) {
            allNames.add(entry.getImageName());
        }
        return new ImageSelectionPane(stage, allNames, imageName).showAndWait();
    }

    /**
     * Streams a bounded pooled sample across {@link #projectImages}, then loads it
     * as the plotted rows. Runs the sample pass off the FX thread. When
     * {@code clusterAfter} is true (e.g. triggered from Recompute) it clusters the
     * sample once loaded; otherwise it stops and waits for an explicit Recompute.
     */
    private void runProjectSample(boolean clusterAfter) {
        final Project<BufferedImage> project = qupath.getProject();
        if (project == null) {
            return;
        }
        final List<String> images = new ArrayList<>(projectImages);
        final int cap = sampleSpinner.getValue();

        progress.setVisible(true);
        progress.setProgress(ProgressIndicator.INDETERMINATE_PROGRESS);
        setControlsDisabled(true);
        statusLabel.setText(String.format("Sampling up to %,d cells across %d image(s)…", cap, images.size()));

        new Thread(
                        () -> {
                            try {
                                CohortClusterModel.SampleData sd = CohortClusterModel.sample(
                                        project,
                                        images,
                                        markerFeatures,
                                        cap,
                                        normalizer,
                                        msg -> Platform.runLater(() -> statusLabel.setText(msg)));
                                if (sd.sampledCells() < 2) {
                                    Platform.runLater(() -> {
                                        progress.setVisible(false);
                                        setControlsDisabled(false);
                                        statusLabel.setText("Not enough cells sampled — staying on current image.");
                                        switchToImageScope();
                                    });
                                    return;
                                }
                                Platform.runLater(() -> {
                                    loadProjectData(sd);
                                    progress.setVisible(false);
                                    setControlsDisabled(false);
                                    applyScopeOverrides();
                                    if (clusterAfter) {
                                        recompute();
                                    } else {
                                        statusLabel.setText(String.format(
                                                "Sampled %,d cell(s) across %d image(s) — click "
                                                        + "“Recompute” to cluster.",
                                                sd.sampledCells(), sd.imageCount()));
                                        plot.redraw();
                                    }
                                });
                            } catch (Throwable t) {
                                logger.error("Project sampling failed", t);
                                Platform.runLater(() -> {
                                    progress.setVisible(false);
                                    setControlsDisabled(false);
                                    statusLabel.setText("Sampling failed: " + t.getMessage());
                                    switchToImageScope();
                                });
                            }
                        },
                        "CellTune-CohortSample")
                .start();
    }

    // ── Small shared helpers ─────────────────────────────────────────────────────

    /** Cells per cluster (length k), or null if no rows are clustered yet. */
    private int[] clusterCounts(int k) {
        // After a completed all-cells write, show the all-cells partition's own pooled
        // counts (not the preview subsample's cluster[] tally) so the Assign dialog's
        // cell counts match what was actually written to the cohort.
        if (fitAllCellsCounts != null && fitAllCellsCounts.length == k) {
            return fitAllCellsCounts;
        }
        int[] counts = new int[k];
        boolean any = false;
        for (int i = 0; i < nRows; i++) {
            int c = cluster[i];
            if (c >= 0 && c < k) {
                counts[c]++;
                any = true;
            }
        }
        return any ? counts : null;
    }

    private List<String> availableClassNames() {
        List<String> classNames = new ArrayList<>();
        for (PathClass pc : qupath.getAvailablePathClasses()) {
            if (pc != null && pc.getName() != null) {
                classNames.add(pc.toString());
            }
        }
        return classNames;
    }

    private List<String> markerNamesFor(int[] cols) {
        List<String> out = new ArrayList<>(cols.length);
        for (int c : cols) {
            out.add(markerFeatures.get(c));
        }
        return out;
    }

    private static int[] identityIndices(int n) {
        int[] all = new int[n];
        for (int i = 0; i < n; i++) {
            all[i] = i;
        }
        return all;
    }

    private static int[] toIntArray(List<Integer> list) {
        int[] out = new int[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = list.get(i);
        }
        return out;
    }

    private void updateClusterMarkersLabel() {
        int total = clusterMarkerItems.size();
        int sel = 0;
        for (CheckMenuItem it : clusterMarkerItems) {
            if (it.isSelected()) {
                sel++;
            }
        }
        clusterMarkersBtn.setText(
                sel == total ? "Cluster markers (all)" : String.format("Cluster markers (%d/%d)", sel, total));
    }

    /** Indices into {@link #markerFeatures} of the checked cluster markers. */
    private int[] selectedMarkerColumns() {
        List<Integer> cols = new ArrayList<>();
        for (int j = 0; j < clusterMarkerItems.size(); j++) {
            if (clusterMarkerItems.get(j).isSelected()) {
                cols.add(j);
            }
        }
        int[] out = new int[cols.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = cols.get(i);
        }
        return out;
    }

    /** Canvas callback for a completed box/lasso gesture: select + redraw. */
    private void onRegionGesture(boolean[] hit) {
        pushOrHighlight(hit);
        plot.redraw();
    }

    /**
     * In CURRENT_IMAGE scope, pushes the hit rows to the QuPath viewer selection
     * (our listener mirrors it back). In PROJECT scope the rows are pooled from
     * many images that aren't all open, so the selection is reflected on the plot
     * only (a visual highlight to read a region's class/marker).
     */
    private void pushOrHighlight(boolean[] hit) {
        if (scope == Scope.CURRENT_IMAGE && hierarchy != null) {
            List<PathObject> hits = new ArrayList<>();
            for (int i = 0; i < nRows; i++) {
                if (hit[i]) {
                    hits.add(cells[i]);
                }
            }
            updatingSelection = true;
            try {
                if (hits.isEmpty()) {
                    hierarchy.getSelectionModel().clearSelection();
                } else {
                    hierarchy.getSelectionModel().setSelectedObjects(hits, null);
                }
            } finally {
                updatingSelection = false;
            }
            QuPathViewer viewer = qupath.getViewer();
            if (viewer != null) {
                viewer.repaint();
            }
            applySelection(hits);
        } else {
            System.arraycopy(hit, 0, selected, 0, nRows);
        }
    }

    private void applySelection(List<PathObject> objs) {
        java.util.Arrays.fill(selected, false);
        for (PathObject o : objs) {
            Integer idx = indexOf.get(o);
            if (idx != null) {
                selected[idx] = true;
            }
        }
    }

    /**
     * Selects every row in the given k-means cluster — in the viewer
     * (CURRENT_IMAGE scope) or as a plot-only highlight (PROJECT scope).
     */
    private void selectCluster(int c) {
        boolean[] hit = new boolean[nRows];
        int count = 0;
        for (int i = 0; i < nRows; i++) {
            if (cluster[i] == c) {
                hit[i] = true;
                count++;
            }
        }
        pushOrHighlight(hit);
        plot.redraw();
        statusLabel.setText(String.format(
                "Selected cluster %d — %,d cell(s)%s.", c, count, scope == Scope.PROJECT ? " (plot highlight)" : ""));
    }

    // ── Viewer → plot selection sync ─────────────────────────────────────────────

    private void installSelectionListener() {
        if (hierarchy == null) {
            return;
        }
        selectionListener = (pathObjectSelected, previousObject, allSelected) -> Platform.runLater(() -> {
            if (updatingSelection || scope == Scope.PROJECT) {
                // PROJECT scope: rows aren't the open image's live cells, so
                // there's nothing to mirror (and indexOf is null).
                return;
            }
            java.util.Arrays.fill(selected, false);
            if (allSelected != null) {
                for (PathObject o : allSelected) {
                    Integer idx = indexOf.get(o);
                    if (idx != null) {
                        selected[idx] = true;
                    }
                }
            }
            plot.redraw();
        });
        hierarchy.getSelectionModel().addPathObjectSelectionListener(selectionListener);
    }

    private void removeSelectionListener() {
        if (hierarchy != null && selectionListener != null) {
            hierarchy.getSelectionModel().removePathObjectSelectionListener(selectionListener);
        }
        selectionListener = null;
    }

    // ── Plot model (read-only view the canvas draws from) ────────────────────────

    /**
     * Exposes the live per-row state to {@link ScatterPlotCanvas} as a read-only
     * {@link ScatterPlotCanvas.PlotModel}. The arrays are reassigned on every
     * scope switch / recompute, so the getters return the current references each
     * draw rather than a captured snapshot.
     */
    private ScatterPlotCanvas.PlotModel asPlotModel() {
        return new ScatterPlotCanvas.PlotModel() {
            @Override
            public int nRows() {
                return nRows;
            }

            @Override
            public double[] ex() {
                return ex;
            }

            @Override
            public double[] ey() {
                return ey;
            }

            @Override
            public int[] cluster() {
                return cluster;
            }

            @Override
            public boolean[] selected() {
                return selected;
            }

            @Override
            public double[][] raw() {
                return raw;
            }

            @Override
            public PathObject[] cells() {
                return cells;
            }

            @Override
            public String[] rowClass() {
                return rowClass;
            }

            @Override
            public PopulationSet predictions() {
                return predictions;
            }

            @Override
            public List<String> markerFeatures() {
                return markerFeatures;
            }

            @Override
            public boolean projectScope() {
                return scope == Scope.PROJECT;
            }

            @Override
            public ScatterPlotCanvas.ColorMode colorMode() {
                return colorCombo.getValue();
            }

            @Override
            public String embeddingName() {
                return embeddingCombo.getValue().name();
            }

            @Override
            public int clusterCount() {
                // The number of clusters actually present in the current display. Leiden
                // decides its own count (fitNClusters, e.g. 26), which is unrelated to the
                // k-means "Clusters (k)" spinner — using the spinner here capped the legend
                // and the colour palette at k, cycling hues (c % k) for higher Leiden labels.
                // Fall back to the spinner only before the first Recompute (no fit yet).
                return fitNClusters > 0 ? fitNClusters : kSpinner.getValue();
            }

            @Override
            public String markerName() {
                return markerCombo.getValue();
            }
        };
    }

    // ── Status + export ──────────────────────────────────────────────────────────

    private void updateStatus() {
        int plotted = 0;
        int sel = 0;
        int clustered = 0;
        for (int i = 0; i < nRows; i++) {
            if (!Double.isNaN(ex[i])) {
                plotted++;
            }
            if (selected[i]) {
                sel++;
            }
            if (cluster[i] >= 0) {
                clustered++;
            }
        }
        // When UMAP is subsampled, k-means still clusters every (active) cell but
        // only a subset is drawn — show both so the gap is never mistaken for
        // "only N cells were clustered".
        String counts = (plotted == clustered)
                ? String.format("%,d cells", clustered)
                : String.format("%,d clustered · %,d plotted", clustered, plotted);
        String scopeDesc = (scope == Scope.PROJECT)
                ? String.format(
                        "Project sample (%d image%s)", projectImages.size(), projectImages.size() == 1 ? "" : "s")
                : imageName;
        // Report the cluster count actually on display: Leiden decides its own count
        // (fitNClusters), which is unrelated to the k-means "Clusters (k)" spinner.
        // Fall back to the spinner only before the first Recompute (no fit yet).
        int shownClusters = fitNClusters > 0 ? fitNClusters : kSpinner.getValue();
        statusLabel.setText(String.format(
                "%s  ·  %s  ·  %s  ·  %,d cluster(s)  ·  %,d selected%s",
                scopeDesc, embeddingCombo.getValue(), counts, shownClusters, sel, statusNotice));
    }

    private void appendStatusNotice(String notice) {
        // Persisted across redraws via updateStatus() (redraw() rebuilds the label).
        statusNotice = (notice == null) ? "" : notice;
        updateStatus();
    }

    private void exportAsPng() {
        FileChooser fc = new FileChooser();
        fc.setTitle("Save Scatter Plot as PNG");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("PNG Image", "*.png"));
        fc.setInitialFileName("cell_scatter.png");
        File file = fc.showSaveDialog(stage);
        if (file == null) {
            return;
        }
        try {
            WritableImage image = plot.snapshot();
            ImageIO.write(SwingFXUtils.fromFXImage(image, null), "png", file);
            Dialogs.showInfoNotification("CellTune", "Exported scatter plot: " + file.getName());
            logger.info("Exported cell scatter plot to {}", file);
        } catch (IOException ex) {
            logger.error("Failed to export scatter plot", ex);
            Dialogs.showErrorMessage("CellTune", "Failed to export PNG: " + ex.getMessage());
        }
    }
}
