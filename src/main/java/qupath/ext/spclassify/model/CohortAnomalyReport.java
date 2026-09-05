package qupath.ext.spclassify.model;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Immutable output contracts for project-level cohort anomaly analysis.
 */
public final class CohortAnomalyReport {

    public static final String RARE_ENRICHMENT = "RARE_ENRICHMENT";
    public static final String OUTLIER_COMPOSITION = "COMPOSITION_OUTLIER";
    public static final String OUTLIER_DISAGREEMENT = "HIGH_DISAGREEMENT";

    public static final Thresholds DEFAULT_THRESHOLDS = new Thresholds(0.01, 20L, 3.0, 3.0, 1.0, 0.65, 0.35, 0.5);

    private final List<ImageAnomaly> images;

    public CohortAnomalyReport(List<ImageAnomaly> images) {
        this.images = images == null ? List.of() : List.copyOf(images);
    }

    public List<ImageAnomaly> images() {
        return images;
    }

    public Map<String, ImageAnomaly> byImageName() {
        var byName = new LinkedHashMap<String, ImageAnomaly>(images.size());
        for (var image : images) {
            byName.put(image.imageName(), image);
        }
        return byName;
    }

    public record Thresholds(
            double rareBaselineCutoff,
            long rareMinCount,
            double rareEnrichmentFoldMin,
            double robustZOutlierThreshold,
            double smoothingAlpha,
            double compositionWeight,
            double disagreementWeight,
            double confidenceDisagreementThreshold) {

        public Thresholds {
            if (rareBaselineCutoff < 0 || rareBaselineCutoff > 1) {
                throw new IllegalArgumentException("rareBaselineCutoff must be in [0, 1]");
            }
            if (rareMinCount < 1) {
                throw new IllegalArgumentException("rareMinCount must be >= 1");
            }
            if (rareEnrichmentFoldMin < 1.0) {
                throw new IllegalArgumentException("rareEnrichmentFoldMin must be >= 1");
            }
            if (robustZOutlierThreshold < 0) {
                throw new IllegalArgumentException("robustZOutlierThreshold must be >= 0");
            }
            if (smoothingAlpha <= 0) {
                throw new IllegalArgumentException("smoothingAlpha must be > 0");
            }
            if (compositionWeight < 0 || disagreementWeight < 0) {
                throw new IllegalArgumentException("weights must be >= 0");
            }
            if (Math.abs((compositionWeight + disagreementWeight) - 1.0) > 1e-6) {
                throw new IllegalArgumentException("compositionWeight + disagreementWeight must equal 1");
            }
            if (confidenceDisagreementThreshold < 0 || confidenceDisagreementThreshold > 1) {
                throw new IllegalArgumentException("confidenceDisagreementThreshold must be in [0, 1]");
            }
        }
    }

    public record ImageInput(String imageName, long predictedCells, long disagreements, Map<String, Long> classCounts) {

        public ImageInput {
            imageName = imageName == null ? "" : imageName;
            predictedCells = Math.max(0L, predictedCells);
            disagreements = Math.max(0L, disagreements);
            classCounts = classCounts == null ? Map.of() : Map.copyOf(classCounts);
        }
    }

    public record ClassEnrichment(
            String className,
            long count,
            double baselineFraction,
            double imageFraction,
            double enrichmentFold,
            boolean rareCandidate,
            boolean highlighted) {}

    public record ImageAnomaly(
            String imageName,
            double anomalyScore,
            boolean flagged,
            List<String> flagReasons,
            Map<String, ClassEnrichment> enrichmentByClass,
            double compositionDistance,
            double disagreementRate,
            double compositionRobustZ,
            double disagreementRobustZ,
            List<String> highlightedClasses) {

        public ImageAnomaly {
            imageName = imageName == null ? "" : imageName;
            anomalyScore = Math.max(0.0, anomalyScore);
            flagReasons = flagReasons == null ? List.of() : List.copyOf(flagReasons);
            enrichmentByClass = enrichmentByClass == null ? Map.of() : Map.copyOf(enrichmentByClass);
            highlightedClasses = highlightedClasses == null ? List.of() : List.copyOf(highlightedClasses);
        }
    }
}
