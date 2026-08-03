package qupath.ext.celltune.ui;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import javafx.application.Platform;
import qupath.ext.celltune.classifier.DualModelClassifier;
import qupath.ext.celltune.io.ProjectStateManager;
import qupath.ext.celltune.model.CellFeatureExtractor;
import qupath.ext.celltune.model.FeatureNormalizer;
import qupath.ext.celltune.model.LabelStore;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.projects.Project;
import qupath.lib.projects.ProjectImageEntry;

/**
 * Background training-pipeline helpers extracted from {@code ClassificationPanel.doTrain}.
 * <p>
 * These are the two self-contained, IO-bound concerns the god-method delegates to while
 * running on the training thread:
 * <ul>
 *   <li>{@link #poolLabelsFromOtherImages} — read saved per-image labels across the project,
 *       open each labelled image, and extract supplementary training rows;</li>
 *   <li>{@link #applyToTargetImages} — classify a set of target images in parallel and persist
 *       the per-image predictions (batch apply).</li>
 * </ul>
 * The residual orchestration (validation, feature prep, progress UI, the
 * {@code trainAndPredict} call, classifier-state save and FX completion) stays in
 * {@code doTrain} because it is bound to the panel's controls and the JavaFX lifecycle.
 * Behaviour is preserved 1:1 — this is a verbatim move behind explicit parameters, so it
 * still requires manual QuPath QA (no automated coverage of the interactive train path).
 */
final class TrainingOrchestrator {

    private TrainingOrchestrator() {} // utility class

    /** Supplementary training rows pooled from other project images (parallel lists). */
    record PooledLabels(List<float[]> rows, List<String> labels) {}

    /**
     * Pool labelled cells from every <em>other</em> project image (within {@code scope}) into
     * supplementary training rows. Skips images with no saved label file (cheap existence check)
     * before the expensive {@code readImageData()}. Always returns non-null (possibly empty) lists.
     *
     * @param project          the QuPath project
     * @param currentImageData the open image (excluded from pooling)
     * @param scope            per-image label scope (sanitized binary marker, or null = multi-class)
     * @param featureNames     the model's feature-column ordering
     * @param normalizer       feature normalizer applied during extraction (may be null)
     * @param trainLog         progress sink
     */
    static PooledLabels poolLabelsFromOtherImages(
            Project<BufferedImage> project,
            ImageData<BufferedImage> currentImageData,
            String scope,
            List<String> featureNames,
            FeatureNormalizer normalizer,
            Consumer<String> trainLog) {
        List<float[]> supplementaryRows = new ArrayList<>();
        List<String> supplementaryLabels = new ArrayList<>();

        List<ProjectImageEntry<BufferedImage>> allEntries = project.getImageList();
        var currentEntry = project.getEntry(currentImageData);

        trainLog.accept("Pooling labels from other project images…");
        for (var entry : allEntries) {
            if (currentEntry != null && entry.equals(currentEntry)) continue;

            // Fast check: skip images with no saved label file to avoid
            // the expensive readImageData() call for unlabelled images.
            if (!ProjectStateManager.hasImageLabels(project, scope, entry.getImageName())) continue;

            try {
                // Load saved labels first (no image I/O required)
                LabelStore otherLabels = new LabelStore("temp");
                try {
                    var savedLabels = ProjectStateManager.loadImageLabels(project, scope, entry.getImageName());
                    if (savedLabels != null) {
                        otherLabels.mergeFrom(savedLabels);
                    }
                } catch (Exception lsEx) {
                    // No saved labels for this image — that's fine
                }

                if (otherLabels.size() == 0) continue;

                // Only now open the image to extract features
                var otherImageData = entry.readImageData();
                var otherHierarchy = otherImageData.getHierarchy();
                var otherDetections = otherHierarchy.getDetectionObjects();
                if (otherDetections.isEmpty()) continue;

                var otherExtractor = new CellFeatureExtractor(featureNames);
                otherExtractor.setNormalizer(normalizer);
                Map<String, PathObject> otherCellById = new LinkedHashMap<>();
                for (PathObject cell : otherDetections) {
                    otherCellById.put(cell.getID().toString(), cell);
                }

                int added = 0;
                for (var labelEntry : otherLabels.getAllLabels().entrySet()) {
                    PathObject cell = otherCellById.get(labelEntry.getKey());
                    if (cell == null) continue;
                    supplementaryRows.add(otherExtractor.extractRow(cell));
                    // Strip merge-history annotation so training sees the effective class
                    supplementaryLabels.add(LabelStore.effectiveClassName(labelEntry.getValue()));
                    added++;
                }
                if (added > 0) {
                    int addedFinal = added;
                    trainLog.accept("  + " + addedFinal + " labelled cells from " + entry.getImageName());
                }
            } catch (Exception ex) {
                trainLog.accept("  ! Could not read " + entry.getImageName() + " (" + ex.getMessage() + ")");
            }
        }
        trainLog.accept("Pooled total: " + supplementaryRows.size() + " rows.");
        return new PooledLabels(supplementaryRows, supplementaryLabels);
    }

    /**
     * Apply the trained classifier to a set of target images in parallel, firing a hierarchy
     * change on the FX thread, saving the image data, and persisting the per-image predictions.
     *
     * @param project              the QuPath project
     * @param currentImageData     the open image (skipped if present in {@code targetImages})
     * @param targetImages         image names to classify
     * @param workers              requested worker threads (capped at the target count, min 1)
     * @param classifier           the trained dual-model classifier
     * @param featureNames         the model's feature-column ordering
     * @param normalizer           feature normalizer applied during extraction (may be null)
     * @param hierarchyEventSource source object for {@code fireHierarchyChangedEvent}
     * @param trainLog             progress sink
     * @return the number of images successfully classified and saved
     */
    static int applyToTargetImages(
            Project<BufferedImage> project,
            ImageData<BufferedImage> currentImageData,
            List<String> targetImages,
            int workers,
            DualModelClassifier classifier,
            List<String> featureNames,
            FeatureNormalizer normalizer,
            Object hierarchyEventSource,
            Consumer<String> trainLog) {
        var currentEntry = project.getEntry(currentImageData);
        String currentImageName = currentEntry != null ? currentEntry.getImageName() : null;

        // Parallelise per-image classification using the user-chosen worker
        // count from the sidebar spinner. Capped at the number of target
        // images. Booster.predict in XGBoost4J / LightGBM4J is thread-safe
        // and predictOnly(populateSets=false) does not mutate shared state.
        int parallelism = Math.min(workers, targetImages.size());
        parallelism = Math.max(1, parallelism);
        trainLog.accept(
                "Applying classifier to " + targetImages.size() + " target image(s), " + parallelism + " at once…");

        var poolExec = Executors.newFixedThreadPool(parallelism, r -> {
            Thread t = new Thread(r, "CellTune-BatchPredict");
            t.setDaemon(true);
            return t;
        });
        var appliedCounter = new AtomicInteger(0);
        var futures = new ArrayList<Future<?>>();

        for (String imgName : targetImages) {
            if (currentImageName != null && currentImageName.equals(imgName)) continue;

            var entryOpt = project.getImageList().stream()
                    .filter(en -> en.getImageName().equals(imgName))
                    .findFirst();
            if (entryOpt.isEmpty()) continue;

            final var entry = entryOpt.get();
            final String imgNameFinal = imgName;
            futures.add(poolExec.submit(() -> {
                try {
                    trainLog.accept("[" + imgNameFinal + "] Loading…");
                    var otherImageData = entry.readImageData();
                    if (otherImageData == null) return;
                    var otherDetections = otherImageData.getHierarchy().getDetectionObjects();
                    if (otherDetections.isEmpty()) {
                        trainLog.accept("[" + imgNameFinal + "] Skipped (no detections)");
                        return;
                    }

                    var otherExtractor = new CellFeatureExtractor(featureNames);
                    otherExtractor.setNormalizer(normalizer);
                    var otherPredAll = classifier.predictAndCollect(
                            otherDetections, otherExtractor, m -> trainLog.accept("[" + imgNameFinal + "] " + m));

                    var saveReady = new CountDownLatch(1);
                    Platform.runLater(() -> {
                        otherImageData.getHierarchy().fireHierarchyChangedEvent(hierarchyEventSource);
                        saveReady.countDown();
                    });
                    try {
                        saveReady.await();
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        return;
                    }

                    entry.saveImageData(otherImageData);
                    // Persist the per-image PopulationSet so the Project
                    // Prediction Summary and review-mode sampling can pull
                    // disagreements from this image.
                    try {
                        ProjectStateManager.saveImagePredictions(project, imgNameFinal, otherPredAll);
                    } catch (Exception persistEx) {
                        trainLog.accept("[" + imgNameFinal + "] WARN: could not save predictions JSON: "
                                + persistEx.getMessage());
                    }
                    int n = appliedCounter.incrementAndGet();
                    trainLog.accept("[" + imgNameFinal + "] Saved (" + n + "/" + targetImages.size() + ")");
                } catch (Exception ex) {
                    trainLog.accept("[" + imgNameFinal + "] ERROR: " + ex.getMessage());
                }
            }));
        }

        poolExec.shutdown();
        for (var f : futures) {
            try {
                f.get();
            } catch (Exception ex) {
                // already logged inside the task
            }
        }
        return appliedCounter.get();
    }
}
