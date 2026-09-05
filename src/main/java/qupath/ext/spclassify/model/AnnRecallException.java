package qupath.ext.spclassify.model;

/**
 * Thrown when the approximate-nearest-neighbour (HNSW) recall gate measures a
 * recall that remains below the required threshold even after escalating the
 * query-time {@code ef} parameter up to its cap (D-07/D-08). When this is
 * thrown, the caller (e.g. {@link LeidenModel#clusterViaAnn}) has produced NO
 * {@code Cluster} labels for the run — this is the phase's correctness gate
 * (D-08): an under-recall ANN graph must never silently feed a wrong
 * partition downstream. Unchecked (a {@link RuntimeException}) because it
 * signals an environment/data condition the caller cannot recover from
 * inline; the UI layer is expected to surface the message (which includes the
 * final measured recall) to the user.
 */
public final class AnnRecallException extends RuntimeException {

    public AnnRecallException(String message) {
        super(message);
    }
}
