package qupath.ext.spclassify.gating;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Parses and evaluates boolean marker expressions for cell-type gating rules.
 * <p>
 * Supports the same syntax as Python CellTune's {@code FormulaTree}:
 * <ul>
 *   <li>{@code CD3}           — single marker (must-have)</li>
 *   <li>{@code CD4&CD3}       — AND (must have both)</li>
 *   <li>{@code CD68|CD163}    — OR (any one suffices)</li>
 *   <li>{@code CD38&!IgA}     — NOT (must NOT have IgA)</li>
 *   <li>{@code A&(B|C)}       — grouping with parentheses</li>
 * </ul>
 * <p>
 * After parsing, {@link #categorize()} classifies each operand into:
 * <ul>
 *   <li><b>mustHave</b> (encoding 1)  — AND operands not inside an OR</li>
 *   <li><b>orExpression</b> (encoding 2) — operands inside an OR group</li>
 *   <li><b>not</b> (encoding -3)       — operands preceded by {@code !}</li>
 * </ul>
 */
public class GatingExpression {

    // ── AST node types ──────────────────────────────────────────────────────

    sealed interface Node permits ItemNode, NotNode, AndNode, OrNode {
        boolean[] evaluate(Map<String, boolean[]> masks);

        String toText();
    }

    record ItemNode(String name) implements Node {
        @Override
        public boolean[] evaluate(Map<String, boolean[]> masks) {
            return masks.get(name); // null if marker not found
        }

        @Override
        public String toText() {
            return name;
        }
    }

    record NotNode(Node operand) implements Node {
        @Override
        public boolean[] evaluate(Map<String, boolean[]> masks) {
            boolean[] inner = operand.evaluate(masks);
            if (inner == null) return null;
            boolean[] result = new boolean[inner.length];
            for (int i = 0; i < inner.length; i++) result[i] = !inner[i];
            return result;
        }

        @Override
        public String toText() {
            return "!" + operand.toText();
        }
    }

    record AndNode(Node left, Node right) implements Node {
        @Override
        public boolean[] evaluate(Map<String, boolean[]> masks) {
            boolean[] l = left.evaluate(masks);
            boolean[] r = right.evaluate(masks);
            if (l == null) return r;
            if (r == null) return l;
            boolean[] result = new boolean[l.length];
            for (int i = 0; i < l.length; i++) result[i] = l[i] && r[i];
            return result;
        }

        @Override
        public String toText() {
            return "(" + left.toText() + " & " + right.toText() + ")";
        }
    }

    record OrNode(Node left, Node right) implements Node {
        @Override
        public boolean[] evaluate(Map<String, boolean[]> masks) {
            boolean[] l = left.evaluate(masks);
            boolean[] r = right.evaluate(masks);
            if (l == null) return r;
            if (r == null) return l;
            boolean[] result = new boolean[l.length];
            for (int i = 0; i < l.length; i++) result[i] = l[i] || r[i];
            return result;
        }

        @Override
        public String toText() {
            return "(" + left.toText() + " | " + right.toText() + ")";
        }
    }

    // ── Categorisation result ───────────────────────────────────────────────

    /** Classification of operands in a parsed expression. */
    public record MarkerCategories(List<String> mustHave, List<String> orExpression, List<String> not) {}

    // ── Fields ──────────────────────────────────────────────────────────────

    private static final Pattern TOKEN_PATTERN = Pattern.compile("[\\w_-]+|[&|!()]");

    private final String expression;
    private final Node root;
    private final Set<String> validMarkers;

    /**
     * Parse a marker expression.
     *
     * @param expression   the boolean expression string (e.g. "CD4&CD3&!IgA")
     * @param validMarkers set of known marker/channel names for validation;
     *                     if null, all tokens are accepted
     */
    public GatingExpression(String expression, Set<String> validMarkers) {
        this.expression = expression;
        this.validMarkers = validMarkers;
        if (expression == null || expression.isBlank()) {
            this.root = null;
        } else {
            List<String> tokens = tokenize(expression);
            int[] pos = {0};
            this.root = parseOr(tokens, pos);
            if (pos[0] < tokens.size()) {
                throw new IllegalArgumentException("Unexpected token after complete expression: " + tokens.get(pos[0]));
            }
        }
    }

    /** Parse without marker validation. */
    public GatingExpression(String expression) {
        this(expression, null);
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /** @return true if this expression is empty (no markers) */
    public boolean isEmpty() {
        return root == null;
    }

    /** @return the original expression string */
    public String getExpression() {
        return expression;
    }

    /**
     * Evaluate the expression against per-marker boolean masks.
     *
     * @param masks map of marker-name → boolean array (true = marker positive for that cell)
     * @return boolean array (true = cell passes gating rule), or null if no markers matched
     */
    public boolean[] evaluate(Map<String, boolean[]> masks) {
        if (root == null) return null;
        return root.evaluate(masks);
    }

    /**
     * Categorise all operands into must-have, or-expression, and not lists.
     * This mirrors Python CellTune's {@code FormulaTree.categorize_operands()}.
     *
     * @return categorised marker lists
     */
    public MarkerCategories categorize() {
        List<String> mustHave = new ArrayList<>();
        List<String> orExpr = new ArrayList<>();
        List<String> not = new ArrayList<>();
        if (root != null) {
            categorizeRecursive(root, mustHave, orExpr, not, false);
        }
        return new MarkerCategories(mustHave, orExpr, not);
    }

    /**
     * Collect all marker names referenced in this expression.
     *
     * @return unmodifiable set of marker names
     */
    public Set<String> getAllMarkers() {
        Set<String> markers = new LinkedHashSet<>();
        if (root != null) collectMarkers(root, markers);
        return Collections.unmodifiableSet(markers);
    }

    @Override
    public String toString() {
        return root == null ? "" : root.toText();
    }

    // ── Tokenizer ───────────────────────────────────────────────────────────

    private List<String> tokenize(String expr) {
        List<String> tokens = new ArrayList<>();
        Matcher m = TOKEN_PATTERN.matcher(expr);
        while (m.find()) {
            tokens.add(m.group());
        }
        return tokens;
    }

    // ── Recursive descent parser ────────────────────────────────────────────

    private Node parseOr(List<String> tokens, int[] pos) {
        Node node = parseAnd(tokens, pos);
        while (pos[0] < tokens.size() && "|".equals(tokens.get(pos[0]))) {
            pos[0]++; // consume '|'
            node = new OrNode(node, parseAnd(tokens, pos));
        }
        return node;
    }

    private Node parseAnd(List<String> tokens, int[] pos) {
        Node node = parsePrimary(tokens, pos);
        while (pos[0] < tokens.size() && "&".equals(tokens.get(pos[0]))) {
            pos[0]++; // consume '&'
            node = new AndNode(node, parsePrimary(tokens, pos));
        }
        return node;
    }

    private Node parsePrimary(List<String> tokens, int[] pos) {
        if (pos[0] >= tokens.size()) {
            throw new IllegalArgumentException("Unexpected end of expression");
        }
        String token = tokens.get(pos[0]);
        if ("!".equals(token)) {
            pos[0]++;
            return new NotNode(parsePrimary(tokens, pos));
        }
        if ("(".equals(token)) {
            pos[0]++;
            Node inner = parseOr(tokens, pos);
            if (pos[0] >= tokens.size() || !")".equals(tokens.get(pos[0]))) {
                throw new IllegalArgumentException("Missing closing parenthesis");
            }
            pos[0]++; // consume ')'
            return inner;
        }
        // Must be a marker name
        if (validMarkers != null && !validMarkers.contains(token)) {
            throw new IllegalArgumentException("Unknown marker: " + token + ". Available: " + validMarkers);
        }
        pos[0]++;
        return new ItemNode(token);
    }

    // ── Tree traversal helpers ──────────────────────────────────────────────

    private void categorizeRecursive(
            Node node, List<String> mustHave, List<String> orExpr, List<String> not, boolean insideOr) {
        switch (node) {
            case ItemNode item -> {
                if (insideOr) {
                    if (!orExpr.contains(item.name())) orExpr.add(item.name());
                } else {
                    if (!mustHave.contains(item.name())) mustHave.add(item.name());
                }
            }
            case NotNode notNode -> {
                if (notNode.operand() instanceof ItemNode item) {
                    if (!not.contains(item.name())) not.add(item.name());
                } else {
                    categorizeRecursive(notNode.operand(), mustHave, orExpr, not, insideOr);
                }
            }
            case AndNode and -> {
                categorizeRecursive(and.left(), mustHave, orExpr, not, insideOr);
                categorizeRecursive(and.right(), mustHave, orExpr, not, insideOr);
            }
            case OrNode or -> {
                categorizeRecursive(or.left(), mustHave, orExpr, not, true);
                categorizeRecursive(or.right(), mustHave, orExpr, not, true);
            }
        }
    }

    private void collectMarkers(Node node, Set<String> markers) {
        switch (node) {
            case ItemNode item -> markers.add(item.name());
            case NotNode not -> collectMarkers(not.operand(), markers);
            case AndNode and -> {
                collectMarkers(and.left(), markers);
                collectMarkers(and.right(), markers);
            }
            case OrNode or -> {
                collectMarkers(or.left(), markers);
                collectMarkers(or.right(), markers);
            }
        }
    }
}
