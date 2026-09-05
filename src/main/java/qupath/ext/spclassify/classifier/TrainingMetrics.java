package qupath.ext.spclassify.classifier;

import java.util.List;

/**
 * Per-class classification metrics computed on a held-out evaluation set.
 *
 * <p>Mirrors sklearn's {@code classification_report}: per-class precision,
 * recall, F1, and support, plus overall accuracy, macro F1, and weighted F1.
 * Computed by predicting the eval set with a model trained only on the 80%
 * train portion of an 80/20 stratified split.
 *
 * @param label         human-readable label (e.g. "Model 1 (XGBOOST) - Train")
 * @param classNames    class name list (length nClasses)
 * @param precision     per-class precision (length nClasses)
 * @param recall        per-class recall (length nClasses)
 * @param f1            per-class F1 (length nClasses)
 * @param support       per-class support — number of true samples (length nClasses)
 * @param accuracy      overall fraction correct
 * @param macroF1       unweighted mean of per-class F1
 * @param weightedF1    support-weighted mean of per-class F1
 * @param total         total number of evaluated samples
 * @param confusionMatrix square matrix [nClasses][nClasses] where
 *                       {@code confusionMatrix[trueIdx][predIdx]} is the
 *                       count of samples whose true class index is
 *                       {@code trueIdx} and whose predicted class index is
 *                       {@code predIdx}. Rows sum to {@code support[i]}.
 */
public record TrainingMetrics(
        String label,
        List<String> classNames,
        double[] precision,
        double[] recall,
        double[] f1,
        int[] support,
        double accuracy,
        double macroF1,
        double weightedF1,
        int total,
        int[][] confusionMatrix) {

    /**
     * Compute per-class metrics from raw probabilities.
     *
     * @param label       descriptive label for this metric block
     * @param classNames  ordered class names
     * @param trueLabels  ground-truth class indices (length n)
     * @param predProba   per-sample class probabilities, [n][nClasses]
     */
    public static TrainingMetrics compute(
            String label, List<String> classNames, float[] trueLabels, float[][] predProba) {
        int n = predProba.length;
        int nClasses = classNames.size();
        int[][] cm = new int[nClasses][nClasses];
        int correct = 0;

        for (int i = 0; i < n; i++) {
            int trueIdx = (int) trueLabels[i];
            int predIdx = 0;
            float best = predProba[i][0];
            for (int c = 1; c < nClasses; c++) {
                if (predProba[i][c] > best) {
                    best = predProba[i][c];
                    predIdx = c;
                }
            }
            if (trueIdx >= 0 && trueIdx < nClasses) {
                cm[trueIdx][predIdx]++;
                if (trueIdx == predIdx) correct++;
            }
        }

        double[] precision = new double[nClasses];
        double[] recall = new double[nClasses];
        double[] f1 = new double[nClasses];
        int[] support = new int[nClasses];

        double weightedF1Sum = 0;
        double macroF1Sum = 0;

        for (int c = 0; c < nClasses; c++) {
            int tp = cm[c][c];
            int rowSum = 0, colSum = 0;
            for (int j = 0; j < nClasses; j++) {
                rowSum += cm[c][j];
                colSum += cm[j][c];
            }
            support[c] = rowSum;
            precision[c] = colSum > 0 ? (double) tp / colSum : 0;
            recall[c] = rowSum > 0 ? (double) tp / rowSum : 0;
            f1[c] = (precision[c] + recall[c]) > 0 ? 2.0 * precision[c] * recall[c] / (precision[c] + recall[c]) : 0;
            macroF1Sum += f1[c];
            weightedF1Sum += f1[c] * support[c];
        }

        double accuracy = n > 0 ? (double) correct / n : 0;
        double macroF1 = nClasses > 0 ? macroF1Sum / nClasses : 0;
        double weightedF1 = n > 0 ? weightedF1Sum / n : 0;

        return new TrainingMetrics(
                label, List.copyOf(classNames), precision, recall, f1, support, accuracy, macroF1, weightedF1, n, cm);
    }

    /**
     * Render this metric block as a sklearn-style classification report.
     */
    public String toFormattedReport() {
        StringBuilder sb = new StringBuilder();
        sb.append(label).append('\n');
        sb.append(String.format("%-24s %10s %10s %10s %10s%n", "class", "precision", "recall", "f1", "support"));
        sb.append("-".repeat(68)).append('\n');
        for (int c = 0; c < classNames.size(); c++) {
            sb.append(String.format(
                    "%-24s %10.3f %10.3f %10.3f %10d%n",
                    truncate(classNames.get(c), 24), precision[c], recall[c], f1[c], support[c]));
        }
        sb.append("-".repeat(68)).append('\n');
        sb.append(String.format("%-24s %10s %10s %10.3f %10d%n", "accuracy", "", "", accuracy, total));
        sb.append(String.format("%-24s %10s %10s %10.3f %10d%n", "macro F1", "", "", macroF1, total));
        sb.append(String.format("%-24s %10s %10s %10.3f %10d%n", "weighted F1", "", "", weightedF1, total));
        return sb.toString();
    }

    private static String truncate(String s, int max) {
        return s.length() > max ? s.substring(0, max - 1) + "\u2026" : s;
    }
}
