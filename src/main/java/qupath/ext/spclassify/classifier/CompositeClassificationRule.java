package qupath.ext.spclassify.classifier;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import qupath.ext.spclassify.io.BinaryClassifierRegistry;

/**
 * Immutable named composite rule composed of ordered marker polarity conditions.
 */
public final class CompositeClassificationRule {

    private static final int MAX_RULE_NAME_LENGTH = 120;
    private static final int MAX_CONDITIONS = 128;

    public enum Polarity {
        POSITIVE('+'),
        NEGATIVE('-');

        private final char symbol;

        Polarity(char symbol) {
            this.symbol = symbol;
        }

        public char symbol() {
            return symbol;
        }

        public static Polarity fromToken(String token) {
            if (token == null || token.isBlank()) {
                throw new IllegalArgumentException("Polarity token must not be blank");
            }
            String normalized = token.trim().toLowerCase();
            return switch (normalized) {
                case "+", "pos", "positive" -> POSITIVE;
                case "-", "neg", "negative" -> NEGATIVE;
                default -> throw new IllegalArgumentException("Invalid polarity token: " + token);
            };
        }

        public static Polarity fromSymbol(char symbol) {
            return switch (symbol) {
                case '+' -> POSITIVE;
                case '-' -> NEGATIVE;
                default -> throw new IllegalArgumentException("Invalid polarity symbol: " + symbol);
            };
        }
    }

    public static final class MarkerCondition {
        private final String markerName;
        private final Polarity polarity;

        private MarkerCondition(String markerName, Polarity polarity) {
            this.markerName = markerName;
            this.polarity = Objects.requireNonNull(polarity, "polarity");
        }

        public static MarkerCondition of(String markerName, Polarity polarity) {
            return new MarkerCondition(normalizeAndValidateMarkerName(markerName), polarity);
        }

        public String markerName() {
            return markerName;
        }

        public Polarity polarity() {
            return polarity;
        }
    }

    private final String name;
    private final List<MarkerCondition> conditions;
    private final String expression;

    private CompositeClassificationRule(String name, List<MarkerCondition> conditions) {
        this.name = normalizeAndValidateRuleName(name);
        this.conditions = validateConditions(conditions);
        this.expression = formatExpression(this.conditions);
    }

    public static CompositeClassificationRule of(String name, List<MarkerCondition> conditions) {
        return new CompositeClassificationRule(name, conditions);
    }

    public static CompositeClassificationRule parse(String name, String expression) {
        return new CompositeClassificationRule(name, parseExpression(expression));
    }

    public static List<MarkerCondition> parseExpression(String expression) {
        if (expression == null || expression.isBlank()) {
            throw new IllegalArgumentException("Rule expression must not be blank");
        }

        String[] tokens = expression.trim().split(":");
        if (tokens.length == 0) {
            throw new IllegalArgumentException("Rule expression has no marker tokens");
        }
        if (tokens.length > MAX_CONDITIONS) {
            throw new IllegalArgumentException("Rule expression exceeds max condition count: " + MAX_CONDITIONS);
        }

        List<MarkerCondition> parsed = new ArrayList<>(tokens.length);
        for (String rawToken : tokens) {
            String token = rawToken == null ? "" : rawToken.trim();
            if (token.length() < 2) {
                throw new IllegalArgumentException("Invalid rule token: " + rawToken);
            }
            char suffix = token.charAt(token.length() - 1);
            Polarity polarity = Polarity.fromSymbol(suffix);
            String markerName = token.substring(0, token.length() - 1).trim();
            parsed.add(MarkerCondition.of(markerName, polarity));
        }
        return validateConditions(parsed);
    }

    public static String formatExpression(List<MarkerCondition> conditions) {
        List<MarkerCondition> validated = validateConditions(conditions);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < validated.size(); i++) {
            if (i > 0) {
                sb.append(':');
            }
            MarkerCondition condition = validated.get(i);
            sb.append(condition.markerName()).append(condition.polarity().symbol());
        }
        return sb.toString();
    }

    public String name() {
        return name;
    }

    public List<MarkerCondition> conditions() {
        return conditions;
    }

    public String expression() {
        return expression;
    }

    private static String normalizeAndValidateRuleName(String rawName) {
        if (rawName == null || rawName.isBlank()) {
            throw new IllegalArgumentException("Rule name must not be blank");
        }
        String name = rawName.trim();
        if (name.length() > MAX_RULE_NAME_LENGTH) {
            throw new IllegalArgumentException("Rule name exceeds max length: " + MAX_RULE_NAME_LENGTH);
        }
        return name;
    }

    private static List<MarkerCondition> validateConditions(List<MarkerCondition> rawConditions) {
        if (rawConditions == null || rawConditions.isEmpty()) {
            throw new IllegalArgumentException("Rule must contain at least one marker condition");
        }
        if (rawConditions.size() > MAX_CONDITIONS) {
            throw new IllegalArgumentException("Rule exceeds max condition count: " + MAX_CONDITIONS);
        }

        LinkedHashSet<String> seenMarkers = new LinkedHashSet<>();
        List<MarkerCondition> validated = new ArrayList<>(rawConditions.size());
        for (MarkerCondition condition : rawConditions) {
            if (condition == null) {
                throw new IllegalArgumentException("Rule contains null marker condition");
            }
            String marker = normalizeAndValidateMarkerName(condition.markerName());
            String dedupeKey = marker.toLowerCase();
            if (!seenMarkers.add(dedupeKey)) {
                throw new IllegalArgumentException("Duplicate marker in rule: " + marker);
            }
            validated.add(MarkerCondition.of(marker, condition.polarity()));
        }
        return List.copyOf(validated);
    }

    private static String normalizeAndValidateMarkerName(String rawMarker) {
        if (rawMarker == null || rawMarker.isBlank()) {
            throw new IllegalArgumentException("Marker name must not be blank");
        }
        String marker = rawMarker.trim();
        String sanitized = BinaryClassifierRegistry.sanitizeMarkerName(marker);
        if (!sanitized.equals(marker)) {
            throw new IllegalArgumentException(
                    "Marker name contains unsupported characters for composite rules: " + rawMarker);
        }
        return sanitized;
    }

    @Override
    public String toString() {
        return name + "[" + expression + "]";
    }
}
