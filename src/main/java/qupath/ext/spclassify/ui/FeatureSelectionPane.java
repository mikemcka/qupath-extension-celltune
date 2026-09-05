package qupath.ext.spclassify.ui;

import java.util.*;
import java.util.stream.Collectors;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.control.cell.CheckBoxTreeCell;
import javafx.scene.layout.*;
import javafx.stage.Modality;
import javafx.stage.Stage;

/**
 * A dialog for selecting which cell measurement features to include in ML
 * training (also reused for choosing measurements for the intensity heatmap and
 * scatter plot).
 * <p>
 * Designed for large panels (COMET, MIBI, etc.) with 1000+ features per cell.
 * Features are grouped into a checkbox tree so the list is navigable at scale:
 * <ul>
 *   <li>One group per <b>marker</b> (the name before the first {@code ": "},
 *       e.g. {@code "DAPI_AF"}, {@code "mCherry_S1 - TRITC_AF"}). The group's
 *       parent checkbox selects/clears every feature for that marker at once.</li>
 *   <li>A <b>Morphology / Shape</b> group for compartment-only measurements
 *       ({@code "Cell: Area µm^2"}, {@code "Cell: ErosionBin_1: …"}, etc.).</li>
 *   <li>A <b>Neighbors</b> group for neighbor-aggregate measurements
 *       ({@code "Neighbors: Mean: Cell: Area µm^2"}, etc.), when present. Leaf
 *       labels keep the full name so the {@code Neighbors:} context is not lost.</li>
 *   <li>An <b>Embeddings</b> group for dimensionality-reduction features
 *       (UMAP / PCA / t-SNE / embedding columns), when present.</li>
 *   <li>An <b>Other / Uncategorized</b> catch-all for feature names that match
 *       none of the above patterns, so nothing is silently misfiled.</li>
 * </ul>
 * Plus an instant search box, expand/collapse and select-all/clear-all helpers,
 * and a selected/total counter. Pre-selects all features by default; the caller
 * may pass a subset to pre-select instead.
 */
public class FeatureSelectionPane {

    /** Group label for compartment-only shape/morphology measurements. */
    static final String GROUP_MORPHOLOGY = "Morphology / Shape";
    /** Group label for dimensionality-reduction / embedding features. */
    static final String GROUP_EMBEDDINGS = "Embeddings";
    /** Group label for neighbor-aggregate features ({@code "Neighbors: …"}). */
    static final String GROUP_NEIGHBORS = "Neighbors";
    /** Catch-all group for features whose name matches no known pattern. */
    static final String GROUP_OTHER = "Other / Uncategorized";

    /** Compartment tokens that mark a measurement as morphology rather than a marker. */
    private static final Set<String> COMPARTMENTS = Set.of("Cell", "Cytoplasm", "Membrane", "Nucleus");

    /** Tokens (case-insensitive, whole-token match) that mark an embedding feature. */
    private static final Set<String> EMBEDDING_TOKENS =
            Set.of("umap", "tsne", "pca", "phenograph", "leiden", "latent", "emb", "embed");

    private final Stage stage;
    private final List<FeatureItem> allFeatures = new ArrayList<>();
    private final List<String> groupOrder = new ArrayList<>();
    private final Map<String, List<FeatureItem>> itemsByGroup = new LinkedHashMap<>();

    private final TreeView<Node> tree;
    private final TextField searchField;
    private final Label countLabel;
    private final List<CheckBoxTreeItem<Node>> visibleLeaves = new ArrayList<>();

    private boolean confirmed = false;

    /**
     * Create a feature selection dialog.
     *
     * @param owner        parent stage
     * @param featureNames all available feature names
     * @param preSelected  features to pre-select (if null or empty, all are selected)
     */
    public FeatureSelectionPane(Stage owner, List<String> featureNames, List<String> preSelected) {
        stage = new Stage();

        Set<String> selected = (preSelected != null && !preSelected.isEmpty())
                ? new LinkedHashSet<>(preSelected)
                : new LinkedHashSet<>(featureNames);

        for (String name : featureNames) {
            allFeatures.add(new FeatureItem(name, selected.contains(name)));
        }
        buildGroups();

        // Search/filter.
        searchField = new TextField();
        searchField.setPromptText("Search features…");
        HBox.setHgrow(searchField, Priority.ALWAYS);
        searchField.textProperty().addListener((o, a, b) -> rebuildTree());

        // Tree with checkboxes.
        tree = new TreeView<>();
        tree.setShowRoot(false);
        tree.setCellFactory(CheckBoxTreeCell.forTreeView());
        tree.setPrefSize(560, 460);
        VBox.setVgrow(tree, Priority.ALWAYS);

        countLabel = new Label();

        // Group / selection helpers.
        Button selectAllBtn = new Button("Select All");
        selectAllBtn.setOnAction(e -> setVisible(true));
        Button clearAllBtn = new Button("Clear All");
        clearAllBtn.setOnAction(e -> setVisible(false));
        Button expandBtn = new Button("Expand All");
        expandBtn.setOnAction(e -> setExpanded(true));
        Button collapseBtn = new Button("Collapse All");
        collapseBtn.setOnAction(e -> setExpanded(false));

        HBox btnRow = new HBox(
                6,
                selectAllBtn,
                clearAllBtn,
                new Separator(javafx.geometry.Orientation.VERTICAL),
                expandBtn,
                collapseBtn);
        btnRow.setAlignment(Pos.CENTER_LEFT);

        // OK / Cancel.
        Button okBtn = new Button("OK");
        okBtn.setDefaultButton(true);
        okBtn.setPrefWidth(80);
        okBtn.setOnAction(e -> {
            confirmed = true;
            stage.close();
        });

        Button cancelBtn = new Button("Cancel");
        cancelBtn.setCancelButton(true);
        cancelBtn.setPrefWidth(80);
        cancelBtn.setOnAction(e -> stage.close());

        HBox okCancelRow = new HBox(
                8,
                countLabel,
                new Region() {
                    {
                        HBox.setHgrow(this, Priority.ALWAYS);
                    }
                },
                okBtn,
                cancelBtn);
        okCancelRow.setAlignment(Pos.CENTER);

        HBox filterRow = new HBox(6, new Label("Filter:"), searchField);
        filterRow.setAlignment(Pos.CENTER_LEFT);

        VBox root = new VBox(8, filterRow, btnRow, tree, okCancelRow);
        root.setPadding(new Insets(10));

        rebuildTree();
        updateCount();

        stage.initOwner(owner);
        stage.initModality(Modality.APPLICATION_MODAL);
        stage.setTitle("Select Features for Training");
        stage.setScene(new Scene(root, 640, 600));
        stage.setMinWidth(420);
        stage.setMinHeight(380);
    }

    /**
     * Override the dialog window title (default: "Select Features for Training").
     *
     * @param title the window title to display
     */
    public void setTitle(String title) {
        stage.setTitle(title);
    }

    /**
     * Show the dialog and block until the user confirms or cancels.
     *
     * @return the selected feature names, or null if cancelled
     */
    public List<String> showAndWait() {
        confirmed = false;
        stage.showAndWait();
        if (!confirmed) return null;

        return allFeatures.stream()
                .filter(FeatureItem::isSelected)
                .map(FeatureItem::getName)
                .collect(Collectors.toList());
    }

    // ── Grouping ─────────────────────────────────────────────────────────────────

    /** Bucket every feature into its group and compute the group display order. */
    private void buildGroups() {
        for (FeatureItem item : allFeatures) {
            String group = groupOf(item.getName());
            itemsByGroup.computeIfAbsent(group, k -> new ArrayList<>()).add(item);
        }

        // Markers first (alphabetical), then Morphology, Neighbors, Embeddings, Other.
        Set<String> special = Set.of(GROUP_MORPHOLOGY, GROUP_NEIGHBORS, GROUP_EMBEDDINGS, GROUP_OTHER);
        List<String> markers = itemsByGroup.keySet().stream()
                .filter(g -> !special.contains(g))
                .sorted(String.CASE_INSENSITIVE_ORDER)
                .collect(Collectors.toList());
        groupOrder.addAll(markers);
        if (itemsByGroup.containsKey(GROUP_MORPHOLOGY)) {
            groupOrder.add(GROUP_MORPHOLOGY);
        }
        if (itemsByGroup.containsKey(GROUP_NEIGHBORS)) {
            groupOrder.add(GROUP_NEIGHBORS);
        }
        if (itemsByGroup.containsKey(GROUP_EMBEDDINGS)) {
            groupOrder.add(GROUP_EMBEDDINGS);
        }
        if (itemsByGroup.containsKey(GROUP_OTHER)) {
            groupOrder.add(GROUP_OTHER);
        }
    }

    /**
     * Classify a feature name into a group: the marker name, {@link #GROUP_MORPHOLOGY},
     * {@link #GROUP_NEIGHBORS}, {@link #GROUP_EMBEDDINGS}, or {@link #GROUP_OTHER}
     * for names matching no known pattern.
     */
    static String groupOf(String name) {
        if (name == null) {
            return GROUP_OTHER;
        }
        if (isEmbedding(name)) {
            return GROUP_EMBEDDINGS;
        }
        int idx = name.indexOf(": ");
        if (idx <= 0) {
            return GROUP_OTHER;
        }
        String prefix = name.substring(0, idx);
        if (COMPARTMENTS.contains(prefix)) {
            return GROUP_MORPHOLOGY;
        }
        if (prefix.equals(GROUP_NEIGHBORS)) {
            return GROUP_NEIGHBORS;
        }
        return prefix;
    }

    /**
     * Detect dimensionality-reduction / embedding feature names by splitting on
     * non-alphanumeric separators and matching whole tokens. This recognises
     * forms like {@code "UMAP 1"}, {@code "Embedding_0"}, {@code "PCA_2"}, and
     * {@code "kronos_emb_0"} without misfiring on markers that merely contain
     * the letters (e.g. {@code "PECAM"}).
     */
    static boolean isEmbedding(String name) {
        if (name == null) {
            return false;
        }
        for (String token : name.split("[^A-Za-z0-9]+")) {
            String t = token.toLowerCase(Locale.ROOT);
            if (t.isEmpty()) {
                continue;
            }
            if (t.startsWith("embedding") || EMBEDDING_TOKENS.contains(t)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Extract common prefixes (text before the first {@code ": "}) from feature
     * names, sorted case-insensitively. Retained as a shared utility for other
     * panes ({@code CellTableExportPane}, {@code ClusteringNormalizationPane}) that build
     * their own prefix dropdowns.
     */
    static List<String> discoverPrefixes(List<String> featureNames) {
        Set<String> prefixes = new LinkedHashSet<>();
        for (String name : featureNames) {
            int colonIdx = name.indexOf(": ");
            if (colonIdx > 0) {
                prefixes.add(name.substring(0, colonIdx + 2)); // include ": "
            }
        }
        List<String> sorted = new ArrayList<>(prefixes);
        sorted.sort(String.CASE_INSENSITIVE_ORDER);
        return sorted;
    }

    /** Display label for a leaf — strips the marker prefix for marker groups. */
    static String leafLabel(String name, String group) {
        if (name == null) {
            return "";
        }
        if (!group.equals(GROUP_MORPHOLOGY)
                && !group.equals(GROUP_EMBEDDINGS)
                && !group.equals(GROUP_NEIGHBORS)
                && !group.equals(GROUP_OTHER)) {
            String prefix = group + ": ";
            if (name.startsWith(prefix)) {
                return name.substring(prefix.length());
            }
        }
        return name;
    }

    // ── Tree building / filtering ────────────────────────────────────────────────

    private void rebuildTree() {
        String search = searchField.getText() == null
                ? ""
                : searchField.getText().toLowerCase(Locale.ROOT).strip();
        boolean searching = !search.isEmpty();

        visibleLeaves.clear();
        CheckBoxTreeItem<Node> root = new CheckBoxTreeItem<>(new Node("root", null, null));

        for (String group : groupOrder) {
            List<FeatureItem> items = itemsByGroup.get(group);
            List<FeatureItem> matching = search.isEmpty()
                    ? items
                    : items.stream()
                            .filter(i -> i.getName().toLowerCase(Locale.ROOT).contains(search))
                            .collect(Collectors.toList());
            if (matching.isEmpty()) {
                continue;
            }

            String groupLabel = group + " (" + items.size() + ")";
            CheckBoxTreeItem<Node> groupNode = new CheckBoxTreeItem<>(new Node(groupLabel, null, group));
            groupNode.setExpanded(searching);

            for (FeatureItem item : matching) {
                CheckBoxTreeItem<Node> leaf =
                        new CheckBoxTreeItem<>(new Node(leafLabel(item.getName(), group), item, group));
                groupNode.getChildren().add(leaf);
                leaf.setSelected(item.isSelected());
                leaf.selectedProperty().addListener((o, was, is) -> {
                    item.setSelected(is);
                    updateCount();
                });
                visibleLeaves.add(leaf);
            }
            root.getChildren().add(groupNode);
        }

        tree.setRoot(root);
    }

    private void setVisible(boolean selected) {
        for (CheckBoxTreeItem<Node> leaf : visibleLeaves) {
            leaf.setSelected(selected);
        }
        updateCount();
    }

    private void setExpanded(boolean expanded) {
        if (tree.getRoot() == null) {
            return;
        }
        for (TreeItem<Node> group : tree.getRoot().getChildren()) {
            group.setExpanded(expanded);
        }
    }

    private void updateCount() {
        long selected = allFeatures.stream().filter(FeatureItem::isSelected).count();
        countLabel.setText(selected + " / " + allFeatures.size() + " selected");
    }

    // ── Inner classes ───────────────────────────────────────────────────────────

    /** A feature with a selected state. */
    static class FeatureItem {
        private final String name;
        private boolean selected;

        FeatureItem(String name, boolean selected) {
            this.name = name;
            this.selected = selected;
        }

        String getName() {
            return name;
        }

        boolean isSelected() {
            return selected;
        }

        void setSelected(boolean s) {
            this.selected = s;
        }

        @Override
        public String toString() {
            return name;
        }
    }

    /** Tree node value: a group header (item == null) or a leaf feature. */
    static final class Node {
        private final String label;
        private final FeatureItem item;
        private final String group;

        Node(String label, FeatureItem item, String group) {
            this.label = label;
            this.item = item;
            this.group = group;
        }

        boolean isGroup() {
            return item == null;
        }

        FeatureItem item() {
            return item;
        }

        String group() {
            return group;
        }

        @Override
        public String toString() {
            return label;
        }
    }
}
