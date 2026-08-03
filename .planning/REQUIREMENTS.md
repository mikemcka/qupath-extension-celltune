# Requirements: CellTune QuPath Extension

Defined: 2026-04-29
Core Value: Bioimaging researchers can iteratively improve cell classifiers by reviewing model disagreements without leaving QuPath.

## v1 Requirements

Requirements for milestone v1.1 Reliability and Verification Hardening.

### Verification Evidence

- [x] EVID-01: Phase 1 has a formal VERIFICATION artifact migrated from existing UAT evidence.
- [x] EVID-02: Phase 2 has a formal VERIFICATION artifact proving composite apply and batch workflows.
- [x] EVID-03: Phase 3 has a formal VERIFICATION artifact proving named-rule save, reload, and apply behavior.

### Nyquist Validation

- [ ] NYQ-01: Phase 1 has VALIDATION coverage with nyquist_compliant status and explicit sampling evidence.
- [ ] NYQ-02: Phase 2 has VALIDATION coverage with nyquist_compliant status and explicit sampling evidence.
- [ ] NYQ-03: Phase 3 has VALIDATION coverage with nyquist_compliant status and explicit sampling evidence.
- [ ] NYQ-04: Phase 4 has VALIDATION coverage with nyquist_compliant status and explicit sampling evidence.

### Reliability Hardening

- [ ] REL-01: Feature selection and normalization settings persist through saveState and reload correctly.
- [ ] REL-02: LabelStore operations are thread-safe during concurrent training and labeling.
- [ ] REL-03: Prediction-only classifier load validates feature column names, not only feature count.

### UX Safety

- [ ] UX-01: Labelling attempts outside a project context show a clear warning instead of silent data loss.
- [ ] UX-02: Binary classification feature-importance view uses a single shared chart mode instead of duplicate class views.

### Regression Tests

- [ ] TST-01: Automated tests cover new reliability and validation paths introduced in this milestone.

### Documentation and Governance

- [x] GOV-01: Phase 10 has a formal code review artifact for project prediction summary implementation.
- [x] DOC-01: README documents Project Prediction Summary workflow including open-image and CSV export actions.
- [x] DOC-02: Standalone build documentation exists with JDK 25 prerequisites and compile/test/shadowJar commands.

## v1.7 Requirements

Requirements for milestone v1.7 Training Performance.

### Training Performance

- [x] PERF-01: Training emits a durable per-run log to `<project>/celltune/logs/` with per-phase timings, dataset shape and heap high-water mark, surviving a crash or OOM.
- [x] PERF-02: An OutOfMemoryError during training produces a specific actionable dialog, a complete log, and a re-enabled Train button — never a silent hang.
- [x] PERF-03: Resampler produces bit-identical output to the pre-optimisation implementation for every strategy, verified by checksum tests, and invariant to thread count.
- [x] PERF-04: Resampler runs at most twice per Train click (was three), with the duplicated 80% fold computed once.
- [x] PERF-05: LightGBM early stopping uses LightGBM's native incremental validation rather than re-scoring the full validation set every round.
- [x] PERF-06: No native Booster handles are leaked by the training or tuning paths.
- [ ] PERF-07: Labelled-cell feature extraction happens once per Train click, in parallel, and cross-image label pooling is parallel and order-deterministic. *(deferred — not executed in phase 18)*
- [x] PERF-08: Users can skip the train/val metrics step and set an explicit training thread budget; both default to today's behaviour.

## v2 Requirements

### Composite Evolution

- COMP-06: Support OR-groups and nested composite definitions for advanced phenotype logic.
- COMP-07: Add confidence-gated composite assignment controls beyond fixed threshold decisions.

### Cohort Analytics

- COH-01: Project summary computes cohort baseline proportions and per-image rare-type enrichment scores for predicted classes.
- COH-02: Project summary computes robust outlier metrics (composition distance and disagreement anomaly) and exposes a ranked anomaly score per image.
- COH-03: Project summary UI supports flagged-image filtering and class-targeted review workflows for unusual cell-type distributions.
- COH-04: Automated tests cover cohort scoring math and summary CSV export schema for anomaly analytics fields.

### Binary Ground Truth Portability

- [ ] XFER-01: Export all registered binary classifiers with persisted marker-specific ground-truth measurement rows into a single portable bundle.
- [ ] XFER-02: Importing a bundle into another project registers missing binary classifiers and restores per-marker training rows for immediate reuse.
- [ ] XFER-03: Import supports explicit merge/replace behavior per marker and reports schema mismatches without mutating unaffected markers.
- [ ] XFER-04: Automated tests verify bundle round-trip fidelity plus malformed-bundle and schema-mismatch handling.

### Cellular Neighborhood Analytics

- [ ] CN-01: Each cell on the current image gets a local-neighborhood cell-type composition vector (kNN or radius) built from its live PathClass classification.
- [ ] CN-02: Composition vectors are k-means clustered into cellular neighborhoods stored as a non-destructive numeric CN measurement, leaving getPathClass() unchanged.
- [ ] CN-03: A CN × cell-type enrichment heatmap, a per-image CN-frequency CSV, and a non-destructive Classification ⟷ CN viewer color toggle are available from the dialog.
- [ ] CN-04: Automated tests cover kNN/radius neighbor finding (self-excluded), composition normalization, clustering, and per-CN mean composition.

### Graph-based Phenotype Clustering (Leiden)

- [ ] LEI-01: The Scatter Plots & Clustering dialog offers a Method selector {k-means, Leiden}; choosing Leiden replaces the "Clusters (k)" control with a resolution control plus a reproducibility toggle, and lets Leiden decide the cluster count.
- [ ] LEI-02: Leiden clusters the same z-scored active marker matrix k-means uses — via a feature-space kNN graph, Jaccard/SNN edge weights, and the CWTS Leiden algorithm — returning a label array that drives plot colouring, legend, and cluster→class assignment identically to k-means.
- [ ] LEI-03: Project/cohort scope supports Leiden by fitting on the pooled sample and assigning every cell across images via kNN label transfer (majority vote of nearest sampled cells), leaving the k-means centroid path unchanged.
- [ ] LEI-04: A pure-array LeidenModel (feature kNN, edge weighting, community detection, kNN label transfer) is covered by JUnit tests on synthetic graphs (community recovery by purity/modularity, self-exclusion, resolution behaviour, label transfer), with no QuPath/JavaFX types.
- [ ] LEI-05: Leiden runs are reproducible run-to-run via a seeded RNG + random-starts, exposed as a reproducibility toggle mirroring the k-means multi-restart option.

### Graph-based Phenotype Clustering — All-Cells (True-Scanpy)

- [x] LEI-06: Cohort/project Leiden offers an all-cells mode that clusters every cell across all images in one graph (pool → single CWTS Leiden partition → write labels back), selectable alongside the retained Phase 14 kNN label-transfer mode.
- [x] LEI-07: Leiden kNN graph construction (single-image and cohort) uses an approximate-NN index (HNSW) whose recall is validated at runtime against exact brute-force featureKnn on a subsample and gated at ≥95% (auto-tune then abort on failure).
- [x] LEI-08: All-cells cohort write is memory-safe via two passes (pool marker features + record per-cell identity, releasing hierarchies; then re-read and write) and maps community labels back to cells by stable PathObject UUID.
- [x] LEI-09: The interactive scatter/UMAP preview remains subsample-based for resolution selection while the persisted Cluster measurement is produced by the full all-cells run; the preview-vs-final divergence is surfaced to the user.
- [x] LEI-10: Automated tests cover cohort pooling/identity mapping, the ANN recall gate vs exact kNN, UUID-keyed label write-back, and all-cells community recovery on synthetic clouds.

### Graph-based Phenotype Clustering — Conditional PCA Reduction

- [x] PCA-01: With feature count above the configurable threshold (default 50) and PCA enabled, the clustering kNN graph (single-image, all-cells cohort, and Leiden kNN-transfer) is built on the PCA-projected matrix, verified by tests asserting the matrix fed to the graph builder has the reduced column count.
- [x] PCA-02: With feature count at or below the threshold, or PCA disabled via the "Reduce dims (PCA)" checkbox, clustering is byte-identical to the pre-PCA behaviour (regression guard on the existing curated-panel path).
- [x] PCA-03: Two reproducible runs (seed 42) on the same above-threshold input yield identical PCA projections and identical downstream cluster labels — the exact (non-randomized) Smile PCA introduces no nondeterminism.
- [x] PCA-04: On a synthetic dataset where a small number of true signal columns are drowned by many independent noise columns, PCA-on recovers the known communities at materially higher ARI than PCA-off, demonstrating the dominance/degradation fix.
- [x] PCA-05: The all-cells cohort path fits one PCA projection on a bounded seeded subsample and applies it to every pooled row; per-cluster centroids remain in original marker space regardless of the space clustering ran in, and peak fit memory stays within a documented bound independent of total cell count.
- [x] PCA-06: Component count kept and cumulative variance explained are reported to the status line/log.

### Normalization / Cofactor Assistance (v1.6)

- [ ] COF-01: User can launch a cofactor-suggestion tool from the Normalise Features workflow.
- [ ] COF-02: User selects which measurements/markers to calibrate the cofactor from via a grouped, searchable picker, independently of the features chosen for normalization.
- [ ] COF-03: Suggestions are computed from the project's in-memory cell measurements, with no external geojson/file streaming.
- [ ] COF-04: The tool derives a per-feature suggested cofactor from each selected feature's background-vs-signal intensity structure.
- [ ] COF-05: The tool presents a per-feature results table (feature, value-scale summary, suggested cofactor).
- [ ] COF-06: The tool reports one recommended global cofactor aggregated across the selected features.
- [ ] COF-07: User can apply the recommended cofactor into the existing Normalise Features cofactor input in one action.
- [ ] COF-08: User can choose the estimation scope — the open image, or the whole project (pooled).

## Out of Scope

| Feature | Reason |
|---------|--------|
| New model families beyond existing stack | v1.1 focuses on reliability and verification hardening only. |
| New UI workflow families unrelated to current Active concerns | Scope is constrained to safety and correctness improvements. |
| Migration off QuPath 0.7 APIs | Compatibility changes are deferred to a future milestone. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| EVID-01 | Phase 5 | Complete |
| EVID-02 | Phase 5 | Complete |
| EVID-03 | Phase 5 | Complete |
| NYQ-01 | Phase 6 | Pending |
| NYQ-02 | Phase 6 | Pending |
| NYQ-03 | Phase 6 | Pending |
| NYQ-04 | Phase 6 | Pending |
| REL-01 | Phase 7 | Pending |
| REL-02 | Phase 7 | Pending |
| REL-03 | Phase 7 | Pending |
| UX-01 | Phase 8 | Pending |
| UX-02 | Phase 8 | Pending |
| TST-01 | Phase 9 | Pending |
| GOV-01 | Phase 10 | Complete |
| DOC-01 | Phase 10 | Complete |
| DOC-02 | Phase 10 | Complete |
| COH-01 | Phase 11 | Complete |
| COH-02 | Phase 11 | Complete |
| COH-03 | Phase 11 | Complete |
| COH-04 | Phase 11 | Complete |
| XFER-01 | Phase 12 | Pending |
| XFER-02 | Phase 12 | Pending |
| XFER-03 | Phase 12 | Pending |
| XFER-04 | Phase 12 | Pending |
| CN-01 | Phase 13 | Pending |
| CN-02 | Phase 13 | Pending |
| CN-03 | Phase 13 | Pending |
| CN-04 | Phase 13 | Pending |
| LEI-01 | Phase 14 | Pending |
| LEI-02 | Phase 14 | Pending |
| LEI-03 | Phase 14 | Pending |
| LEI-04 | Phase 14 | Pending |
| LEI-05 | Phase 14 | Pending |
| LEI-06 | Phase 15 | Complete |
| LEI-07 | Phase 15 | Complete |
| LEI-08 | Phase 15 | Complete |
| LEI-09 | Phase 15 | Complete |
| LEI-10 | Phase 15 | Complete |
| PCA-01 | Phase 16 | Complete |
| PCA-02 | Phase 16 | Complete |
| PCA-03 | Phase 16 | Complete |
| PCA-04 | Phase 16 | Complete |
| PCA-05 | Phase 16 | Complete |
| PCA-06 | Phase 16 | Complete |
| COF-01 | Phase 17 | Verified (UAT pending) |
| COF-02 | Phase 17 | Verified (UAT pending) |
| COF-03 | Phase 17 | Verified (code) |
| COF-04 | Phase 17 | Verified (code) |
| COF-05 | Phase 17 | Verified (UAT pending) |
| COF-06 | Phase 17 | Verified (code) |
| COF-07 | Phase 17 | Verified (UAT pending) |
| COF-08 | Phase 17 | Verified (code) |

Coverage:
- requirements listed in this file: 54
- mapped to phases: 52
- unmapped: 2 (COMP-06, COMP-07)

---
Requirements defined: 2026-04-29
Last updated: 2026-04-29 after phase 10 documentation and governance refresh
