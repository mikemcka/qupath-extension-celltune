package qupath.ext.celltune.classifier;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.base.cart.SplitRule;
import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.data.measure.NominalScale;
import smile.data.type.DataTypes;
import smile.data.type.StructField;
import smile.data.vector.IntVector;

/**
 * Random Forest classifier backed by the SMILE library
 * ({@link smile.classification.RandomForest}).
 * <p>
 * Trees use the entropy (information-gain) split rule — the closest analogue to
 * Python CellTune's {@code criterion='log_loss'} — with {@code mtry = sqrt(p)}
 * random features per split and bootstrap sampling, then average class
 * probabilities across all trees (SMILE soft prediction).
 * <p>
 * Designed to be a drop-in alongside {@link XGBoostModel} and {@link LightGBMModel}
 * with the same method signatures. Replaced a hand-rolled pure-Java CART forest;
 * the serialized model format changed accordingly (see {@link #toBytes()}).
 */
public class RandomForestModel {

    private static final Logger logger = LoggerFactory.getLogger(RandomForestModel.class);

    /** Magic header identifying the SMILE-backed serialization format (v1). */
    private static final int MAGIC = 0x52465331; // "RFS1"

    private static final int FORMAT_VERSION = 1;

    /** Minimum samples in a leaf node (matches the former implementation). */
    private static final int NODE_SIZE = 1;

    private int nClasses;
    private List<String> classNames;
    private List<String> featureNames;
    private int nTrees;

    /**
     * Synthetic, formula-safe feature column names ({@code f0..f{p-1}}). Real
     * measurement names (e.g. "CD3: Cell: Mean") contain spaces/colons that would
     * break SMILE formula parsing, so the DataFrame always uses these instead.
     */
    private String[] featureCols;

    /** The trained SMILE model (null until trained / after {@link #close()}). */
    private RandomForest forest;

    // ── Training ────────────────────────────────────────────────────────────────

    /**
     * Train a Random Forest model.
     * <p>
     * Parameter mapping from the boosted-tree API:
     * <ul>
     *   <li>{@code numRounds} → number of trees in the forest</li>
     *   <li>{@code maxDepth} → max tree depth</li>
     *   <li>{@code eta} → ignored (no learning rate in RF)</li>
     *   <li>{@code subsample} → bootstrap sample rate (1.0 = sampling with
     *       replacement, the standard bootstrap)</li>
     * </ul>
     */
    public void train(
            float[] flatData,
            float[] labels,
            int nSamples,
            int nFeatures,
            List<String> classNames,
            List<String> featureNames,
            int numRounds,
            int maxDepth,
            float eta,
            float subsample)
            throws Exception {

        this.nClasses = classNames.size();
        this.classNames = List.copyOf(classNames);
        this.featureNames = List.copyOf(featureNames);
        this.nTrees = numRounds;
        this.featureCols = makeFeatureCols(nFeatures);

        double[][] x = toMatrix(flatData, nSamples, nFeatures);
        int[] y = new int[nSamples];
        for (int i = 0; i < nSamples; i++) {
            y[i] = (int) labels[i];
        }

        // Build a classification DataFrame: numeric feature columns + a nominal
        // response column. Declaring all class levels on the response guarantees
        // a fixed-length posteriori in our class order, even if some class is
        // absent from the training labels.
        String[] levels = this.classNames.toArray(new String[0]);
        DataFrame df = DataFrame.of(x, featureCols)
                .merge(IntVector.of(new StructField("class", DataTypes.IntegerType, new NominalScale(levels)), y));
        Formula formula = Formula.lhs("class");

        int mtry = Math.max(1, (int) Math.sqrt(nFeatures));
        int maxNodes = Math.max(2, nSamples);
        double subsampleRate = Math.min(Math.max(subsample, 1e-3f), 1.0f);

        logger.info(
                "Random Forest (SMILE) training: {} trees, max_depth={}, mtry={}, "
                        + "{} samples, {} features, {} classes",
                nTrees,
                maxDepth,
                mtry,
                nSamples,
                nFeatures,
                nClasses);

        this.forest = RandomForest.fit(
                formula, df, numRounds, mtry, SplitRule.ENTROPY, maxDepth, maxNodes, NODE_SIZE, subsampleRate);

        logger.info("Random Forest training complete ({} trees)", nTrees);
    }

    // ── Prediction ──────────────────────────────────────────────────────────────

    /**
     * Predict class probabilities for multiple cells.
     *
     * @param flatData  row-major feature matrix (nSamples × nFeatures)
     * @param nSamples  number of samples
     * @param nFeatures number of features
     * @return probability matrix [nSamples][nClasses]
     */
    public float[][] predictProba(float[] flatData, int nSamples, int nFeatures) throws Exception {
        if (forest == null) {
            throw new IllegalStateException("Random Forest model is not trained.");
        }
        double[][] x = toMatrix(flatData, nSamples, nFeatures);
        DataFrame df = DataFrame.of(x, featureCols);

        float[][] result = new float[nSamples][nClasses];
        double[] posteriori = new double[nClasses];
        for (int i = 0; i < nSamples; i++) {
            forest.predict(df.get(i), posteriori);
            for (int c = 0; c < nClasses; c++) {
                result[i][c] = (float) posteriori[c];
            }
        }
        return result;
    }

    /**
     * Predict the single best class index for each sample.
     */
    public int[] predict(float[] flatData, int nSamples, int nFeatures) throws Exception {
        float[][] probs = predictProba(flatData, nSamples, nFeatures);
        int[] preds = new int[nSamples];
        for (int i = 0; i < nSamples; i++) {
            int best = 0;
            for (int c = 1; c < nClasses; c++) {
                if (probs[i][c] > probs[i][best]) {
                    best = c;
                }
            }
            preds[i] = best;
        }
        return preds;
    }

    // ── Serialisation ───────────────────────────────────────────────────────────

    /**
     * Serialise the entire model (metadata + SMILE forest) to a byte array.
     * <p>
     * Format: {@code MAGIC, version, nClasses, nTrees, classNames, featureNames,
     * len(forest), forest-object-bytes}. The leading magic distinguishes this
     * from the legacy hand-rolled tree format (see {@link #loadFromBytes}).
     */
    public byte[] toBytes() throws Exception {
        if (forest == null) {
            throw new IllegalStateException("Random Forest model is not trained.");
        }
        ByteArrayOutputStream forestBaos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(forestBaos)) {
            oos.writeObject(forest);
        }
        byte[] forestBytes = forestBaos.toByteArray();

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream out = new DataOutputStream(baos);
        out.writeInt(MAGIC);
        out.writeInt(FORMAT_VERSION);
        out.writeInt(nClasses);
        out.writeInt(nTrees);
        out.writeInt(classNames.size());
        for (String cn : classNames) {
            out.writeUTF(cn);
        }
        out.writeInt(featureNames.size());
        for (String fn : featureNames) {
            out.writeUTF(fn);
        }
        out.writeInt(forestBytes.length);
        out.write(forestBytes);
        out.flush();
        return baos.toByteArray();
    }

    /**
     * Load a model from a byte array produced by {@link #toBytes()}.
     *
     * @throws IOException if the bytes are in the legacy (pre-SMILE) format, which
     *                     cannot be migrated — such models must be retrained.
     */
    public void loadFromBytes(byte[] bytes, List<String> classNames, List<String> featureNames) throws Exception {
        DataInputStream in = new DataInputStream(new ByteArrayInputStream(bytes));

        int magic = in.readInt();
        if (magic != MAGIC) {
            throw new IOException("Saved Random Forest model uses the legacy format and is no "
                    + "longer compatible after the SMILE upgrade. Please "
                    + "retrain the classifier.");
        }
        in.readInt(); // format version (only v1 today)

        this.nClasses = in.readInt();
        this.nTrees = in.readInt();

        int nCN = in.readInt();
        List<String> cn = new ArrayList<>(nCN);
        for (int i = 0; i < nCN; i++) {
            cn.add(in.readUTF());
        }
        this.classNames = List.copyOf(cn);

        int nFN = in.readInt();
        List<String> fn = new ArrayList<>(nFN);
        for (int i = 0; i < nFN; i++) {
            fn.add(in.readUTF());
        }
        this.featureNames = List.copyOf(fn);
        this.featureCols = makeFeatureCols(this.featureNames.size());

        int forestLen = in.readInt();
        byte[] forestBytes = new byte[forestLen];
        in.readFully(forestBytes);
        try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(forestBytes))) {
            this.forest = (RandomForest) ois.readObject();
        }
    }

    /** Release resources. */
    public void close() {
        forest = null;
    }

    // ── Accessors ───────────────────────────────────────────────────────────────

    public boolean isTrained() {
        return forest != null;
    }

    public int getNumClasses() {
        return nClasses;
    }

    public List<String> getClassNames() {
        return classNames;
    }

    public List<String> getFeatureNames() {
        return featureNames;
    }

    // ── Feature Importance ───────────────────────────────────────────────────────

    /**
     * Compute feature importance as normalised mean decrease in impurity (SMILE's
     * {@link RandomForest#importance()}).
     * <p>
     * Random Forest importance is class-agnostic, so the same per-feature vector
     * is returned for every class (matching the former behaviour).
     *
     * @return importance matrix [nClasses][nFeatures], values in [0, 1] (sum to 1)
     */
    public double[][] computeSplitImportance() throws Exception {
        if (forest == null) {
            throw new IllegalStateException("Random Forest model is not trained.");
        }
        int nFeatures = featureNames.size();
        double[] counts = forest.importance();

        // SMILE importance is ordered by the formula's predictors, which here are
        // featureCols (f0..f{p-1}) in feature order — already aligned. Guard length.
        double[] imp = new double[nFeatures];
        for (int f = 0; f < nFeatures && f < counts.length; f++) {
            imp[f] = counts[f];
        }

        double total = 0;
        for (double c : imp) {
            total += c;
        }
        if (total > 0) {
            for (int f = 0; f < nFeatures; f++) {
                imp[f] /= total;
            }
        }

        double[][] result = new double[nClasses][nFeatures];
        for (int c = 0; c < nClasses; c++) {
            result[c] = imp.clone();
        }
        return result;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────────

    private static String[] makeFeatureCols(int nFeatures) {
        String[] cols = new String[nFeatures];
        for (int j = 0; j < nFeatures; j++) {
            cols[j] = "f" + j;
        }
        return cols;
    }

    private static double[][] toMatrix(float[] flatData, int nSamples, int nFeatures) {
        double[][] x = new double[nSamples][nFeatures];
        for (int i = 0; i < nSamples; i++) {
            int off = i * nFeatures;
            for (int j = 0; j < nFeatures; j++) {
                x[i][j] = flatData[off + j];
            }
        }
        return x;
    }
}
