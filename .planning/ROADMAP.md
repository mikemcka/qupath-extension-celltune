# Roadmap - CellTune QuPath Extension

## Milestones

- [x] v1.0 Binary Composite Classification (shipped 2026-04-29) - see [v1.0 archive](milestones/v1.0-ROADMAP.md)
- [ ] v1.1 Reliability and Verification Hardening (in progress)
- [x] v1.2 Cohort Outlier Analytics (shipped 2026-04-30)
- [ ] v1.3 Cross-Project Binary Ground Truth Portability (planned)
- [ ] v1.4 Cellular Neighborhood Analytics (planned)
- [ ] v1.5 Graph-based Phenotype Clustering (planned)
- [ ] v1.6 Normalization / Cofactor Assistance (planned)

## Milestone v1.1 - Reliability and Verification Hardening

Goal: Close verification-evidence debt and harden reliability risks found after v1.0 shipment.

### Phase 5 - Verification Artifact Backfill

Goal: Normalize legacy milestone evidence by creating formal phase verification artifacts for phases 1-3.
Depends on: v1.0 archives and existing phase summaries/UAT evidence.
Requirements: EVID-01, EVID-02, EVID-03
Success Criteria:
1. Phase 1 has VERIFICATION artifact mapped to its shipped behaviors.
2. Phase 2 has VERIFICATION artifact covering single-image and batch composite flows.
3. Phase 3 has VERIFICATION artifact for named-rule persistence and apply flows.
4. Milestone audit no longer flags orphaned requirements caused by missing phase verification files.
Plans: 2 plans

Plans:
- [x] 05-01-PLAN.md - Create formal verification artifacts for phases 1 and 2 (completed 2026-04-29)
- [x] 05-02-PLAN.md - Create phase 3 verification artifact and refresh v1.0 audit evidence (completed 2026-04-29)

### Phase 6 - Nyquist Validation Coverage

Goal: Add Nyquist VALIDATION coverage for all milestone phases and produce explicit compliance status.
Depends on: Phase 5 verification artifacts.
Requirements: NYQ-01, NYQ-02, NYQ-03, NYQ-04
Success Criteria:
1. Each phase 1-4 has VALIDATION artifact with nyquist_compliant field.
2. Validation files include reproducible checks and sampling rationale.
3. Missing-phase warnings in audit are removed for Nyquist coverage.
Plans: To be planned

### Phase 7 - Reliability Core Hardening

Goal: Fix core reliability defects in persistence, concurrency safety, and feature-schema validation.
Depends on: Phase 6.
Requirements: REL-01, REL-02, REL-03
Success Criteria:
1. saveState/reload preserves selected features and normalization settings.
2. LabelStore operations are safe under concurrent training and labeling usage.
3. predictOnly rejects mismatched feature name schemas and surfaces actionable errors.
Plans: To be planned

### Phase 8 - UX Safety and Explainability Polish

Goal: Improve user-facing safety feedback and binary explainability display behavior.
Depends on: Phase 7.
Requirements: UX-01, UX-02
Success Criteria:
1. Out-of-project labeling attempts show explicit warning.
2. Binary SHAP display avoids duplicate class views and uses a single coherent mode.
Plans: To be planned

### Phase 9 - Regression Tests for Hardening

Goal: Add automated tests covering reliability and verification hardening paths.
Depends on: Phases 7-8.
Requirements: TST-01
Success Criteria:
1. New test suite covers persistence, schema validation, and key verification-path regressions.
2. CI or local test command demonstrates stable pass for new hardening tests.
Plans: To be planned

### Phase 10 - Documentation and Governance Refresh

Goal: Review recent summary-dialog delivery and align planning plus user/build documentation.
Depends on: Phase 5 baseline artifacts and current summary feature implementation.
Requirements: GOV-01, DOC-01, DOC-02
Success Criteria:
1. Code review artifact records findings and residual risk for summary feature changes.
2. README documents Project Prediction Summary usage including open-image and CSV export.
3. Dedicated build documentation provides reproducible JDK 25 and Gradle wrapper commands.
Plans: 1 plan

Plans:
- [x] 10-01-PLAN.md - Review summary code and align planning, README, and build docs (completed 2026-04-29)


## Milestone v1.2 - Cohort Outlier Analytics

Goal: Detect rare cell-type enrichment and image-level composition anomalies from project-wide prediction summaries.

### Phase 11 - Project Summary Cohort Outlier Analytics

Goal: Add cohort baseline analytics, rare-type enrichment ranking, and anomaly-aware project summary review workflows.
Depends on: Phase 10 summary pane baseline and persisted per-image Pred_ALL predictions.
Requirements: COH-01, COH-02, COH-03, COH-04
Success Criteria:
1. Project summary computes per-image enrichment vs cohort baselines for predicted classes.
2. Project summary computes robust anomaly scores for unusual composition and disagreement behavior.
3. Summary UI can filter flagged images and target specific rare cell classes for review.
4. CSV export includes anomaly fields and automated tests cover scoring and export schema.
Plans: 2 plans

Plans:
- [x] 11-01-PLAN.md - Build cohort anomaly scoring contracts and analytics engine with tests (completed 2026-04-30)
- [x] 11-02-PLAN.md - Integrate anomaly insights into summary UI filters and CSV export (completed 2026-04-30)

## Milestone v1.3 - Cross-Project Binary Ground Truth Portability

Goal: Move binary classifier training evidence between projects so marker-specific classifiers can be reconstructed without relabelling from scratch.

### Phase 12 - Binary Ground Truth Bundle Export/Import

Goal: Export ground truth measurements from all registered binary classifiers in one project and import them into another project with schema-aware merge behavior.
Depends on: Phase 1 binary classifier registry/persistence and existing ground-truth CSV import/export foundations.
Requirements: XFER-01, XFER-02, XFER-03, XFER-04
Success Criteria:
1. A single export action produces a bundle containing marker-wise ground-truth measurements for every registered binary classifier with available training rows.
2. Importing a bundle registers missing binary classifiers in the target project and persists imported training rows per marker.
3. Import flow reports per-marker outcomes (imported, replaced, skipped, schema mismatch) without corrupting existing classifier state.
4. Automated tests cover bundle round-trip, malformed bundle handling, and merge/replace behavior.
Plans: 2 plans

Plans:
- [ ] 12-01-PLAN.md - Build bundle contract, persistence hooks, and automated round-trip tests
- [ ] 12-02-PLAN.md - Wire QuPath menu actions and guided import/export workflow with verification checkpoint

## Milestone v1.4 - Cellular Neighborhood Analytics

Goal: Partition tissue into recurring spatial micro-environments by clustering each cell on its local cell-type composition (Schürch/Nolan cellular neighborhoods), non-destructively, on the current image.

### Phase 13 - CN Spatial Clustering

Goal: Add cellular-neighborhood (CN) spatial clustering: per-cell local composition vectors → k-means → non-destructive CN measurement, with an enrichment heatmap, CN-frequency CSV, and a Classification ⟷ CN viewer color toggle.
Depends on: Existing scatter/cluster machinery (ScatterPlotView, ScatterMath, CohortClusterModel), DistanceMeasurementsDialog STRtree pattern, and IntensityHeatmapView.
Requirements: CN-01, CN-02, CN-03, CN-04
Success Criteria:
1. Each cell on the current image gets a CN from its local-neighborhood cell-type composition (kNN or radius) derived from live classifications.
2. CN is stored as a non-destructive numeric measurement; phenotype (getPathClass()) is unchanged.
3. Dialog provides a CN × cell-type enrichment heatmap, a per-image CN-frequency CSV, and a non-destructive Classification ⟷ CN color toggle.
4. Automated tests cover neighbor finding (kNN + radius, self-excluded), composition normalization, clustering, and per-CN mean composition.
Plans: 1 plan

Plans:
- [ ] 13-01-PLAN.md - Build NeighborhoodModel + dialog + enrichment heatmap/CSV + menu wiring with unit tests (filed from prior plan-mode draft)


## Milestone v1.5 - Graph-based Phenotype Clustering

Goal: Add Leiden graph-based community detection as a selectable alternative to k-means in the interactive Scatter Plots & Clustering (phenotyping) workflow, for both current-image and whole-project scope — the field-standard method (scanpy/scimap/SPACEc) for resolving non-spherical and rare cell populations.

### Phase 14 - Leiden Phenotype Clustering

Goal: Add a Method {k-means, Leiden} selector to the scatter clustering dialog; Leiden builds a feature-space kNN graph (Jaccard-weighted) and runs the CWTS Leiden algorithm with a resolution control and reproducibility toggle, reusing the existing colouring/legend/assignment machinery and adding a kNN label-transfer path for cohort scope.
Depends on: Existing scatter/cluster machinery (ScatterPlotView, ScatterMath, CohortClusterModel) and the k-means multi-restart reproducibility pattern (NeighborhoodModel). Research: .planning/notes/leiden-clustering-design.md.
Requirements: LEI-01, LEI-02, LEI-03, LEI-04, LEI-05
Success Criteria:
1. Dialog offers Method {k-means, Leiden}; Leiden shows resolution + reproducibility controls and decides the cluster count.
2. Leiden clusters the same z-scored active matrix via kNN graph + CWTS Leiden and drives colouring/legend/assignment identically to k-means.
3. Project scope fits Leiden on the pooled sample and assigns all cells across images via kNN label transfer; k-means cohort path unchanged.
4. Automated tests cover feature kNN, community recovery, resolution behaviour, reproducibility, and label transfer.
Plans: 1 plan

Plans:
- [ ] 14-01-PLAN.md - Bundle CWTS Leiden + build LeidenModel (kNN graph/Leiden/label transfer), wire Method selector + resolution/reproducibility controls, cohort kNN-transfer, docs, with unit tests

### Phase 15 - All-Cells Leiden Clustering (True-Scanpy)

Goal: Replace the Phase 14 cohort kNN label-transfer (sc.tl.ingest-style) with a single all-cohort clustering (sc.tl.leiden-style): pool every cell across all project images into one feature matrix, build an approximate-NN (HNSW) kNN graph over all cells, SNN/Jaccard-weight it, run one CWTS Leiden partition, and write each cell's community label back to its source image — no label transfer.
Depends on: Phase 14 (LeidenModel feature kNN/edge weighting/CWTS Leiden, CohortClusterModel streaming passes, ScatterPlotView cohort write action).
Requirements: LEI-06, LEI-07, LEI-08, LEI-09, LEI-10
Success Criteria:
1. Cohort clustering offers an all-cells mode (pool all cells → one graph → single Leiden → write labels back) selectable alongside the retained kNN label-transfer mode.
2. The Leiden kNN graph (single-image and cohort) is built with an approximate-NN index (HNSW) and validated at runtime against exact brute-force featureKnn by a recall gate (≥95%, auto-tune then abort) on a subsample.
3. Write-back is memory-safe (two-pass: pool features + record identity, release hierarchies; re-read + write) and maps labels to cells by stable PathObject UUID.
4. The interactive scatter/UMAP preview stays subsample-based for resolution selection; the persisted Cluster measurement comes from the full all-cells run, with the preview-vs-final divergence documented.
5. Automated tests cover pooling/identity mapping, the ANN recall gate vs exact kNN, UUID write-back, and all-cells community recovery on synthetic clouds.
Plans: 5 plans
- [x] 15-01-PLAN.md - Bundle hnswlib-core (implementation+shadow) and add the pure HnswKnnIndex ANN kNN wrapper with seeded-deterministic build + recall tests
- [x] 15-02-PLAN.md - Route Leiden through HNSW (clusterViaAnn) guarded by the runtime recall gate (AnnRecallException abort); HNSW-vs-exact ARI tests
- [x] 15-03-PLAN.md - Primitive-array SNN/Jaccard rewrite for cohort scale (byte-identical equivalence) + all-cells recovery/reproducibility tests
- [x] 15-04-PLAN.md - CohortClusterModel two-pass all-cells driver (poolAllCells + writeClusterAllCells) with packed-UUID write-back + cancellation; reorder/pooling/cancel tests
- [x] 15-05-PLAN.md - ScatterPlotView cohort-mode radio pair, soft-ceiling confirm, per-phase progress + cancel + recall status, preview re-sync, and USER_GUIDE/CLAUDE.md fidelity-gap docs (completed 2026-07-06, human-verify checkpoint approved incl. full multi-image project)

### Phase 16 - PCA Dimensionality Reduction for Clustering

Goal: Add an optional, deterministic PCA reduction step between z-scoring and the clustering kNN graph build (single-image preview, all-cells cohort driver, Leiden kNN-transfer), mirroring scanpy's `scale -> PCA -> neighbors -> leiden` recipe and closing the third and final documented scanpy-fidelity gap ("no PCA before neighbours").
Depends on: Phase 15 (all-cells Leiden clustering, ScatterMath/CohortClusterModel/ScatterPlotView machinery this PCA step is inserted in front of).
Requirements: PCA-01, PCA-02, PCA-03, PCA-04, PCA-05, PCA-06
Success Criteria:
1. Above a configurable feature-count threshold (default 50) with PCA enabled, the clustering kNN graph is built on the PCA-projected matrix in all three entry points (single-image preview, all-cells cohort, Leiden kNN-transfer).
2. At or below the threshold, or with PCA disabled, clustering is byte-identical to the pre-PCA behaviour.
3. PCA reduction is deterministic (exact Smile covariance eigendecomposition, no randomized SVD) and demonstrably fixes the noise-dominance/distance-degradation failure mode on synthetic data (dominance ARI test).
4. The all-cells cohort path fits the projection on a bounded seeded subsample and applies it to every pooled row; per-cluster centroids stay in original marker space regardless of clustering space; component count and variance explained are reported to the status line.
Plans: 1 plan (recorded retroactively — implemented inline, then documented)

Plans:
- [x] 16-01-PLAN.md - ScatterMath.pcaReduce conditional dimensionality reduction, wired into the all-cells driver + Leiden transfer + scatter-plot preview fit/UI, with dominance/determinism/marker-space-centroid tests and USER_GUIDE/CLAUDE.md fidelity-gap docs (completed 2026-07-06, recorded retroactively)


## Milestone v1.6 - Normalization / Cofactor Assistance

Goal: Let users derive a sound arcsinh normalization cofactor from their own in-project cell measurements, in-app, instead of guessing or hardcoding a platform default.

### Phase 17 - In-QuPath Cofactor Suggestion

Goal: Add a "Suggest cofactor…" tool to the Normalise Features workflow that estimates a good arcsinh cofactor from the project's in-memory cell measurements. The user picks a calibration feature-set from a grouped, searchable picker (independent of the normalize set, no hardcoded measurement name); the tool estimates each selected feature's background/signal structure, presents a per-feature table plus one recommended global cofactor, and applies it into the existing cofactor spinner. Open-image or whole-project (pooled) scope.
Depends on: Phase 5 (feature normalisation — NormalizationPane, FeatureNormalizer), the FeatureSelectionPane grouping/search UI, and Phase 15 (CohortClusterModel.poolAllCells for project-scope pooling).
Requirements: COF-01, COF-02, COF-03, COF-04, COF-05, COF-06, COF-07, COF-08
Success Criteria:
1. A "Suggest cofactor…" control reachable from the Normalise Features workflow opens a grouped, searchable feature picker whose calibration selection is independent of the normalize selection.
2. The tool computes suggestions from in-memory cell measurements (no external geojson/file streaming) for the chosen scope — open image or whole project pooled.
3. The tool displays a per-feature results table (feature → value-scale summary → suggested cofactor) plus one recommended global cofactor aggregated across the selected features.
4. Applying the recommendation sets the existing Normalise Features cofactor input to that value in a single action.
5. On a known raw-fluorescence-scale panel the recommended global cofactor lands in a data-driven range (tens, not the platform default), consistent with the MIBI/COMET cohort assessments.

**Plans:** 4 plans

Plans:
**Wave 1** *(parallel — no file overlap):*
- [x] 17-01-PLAN.md - CofactorEstimator pure-array background-percentile estimator + tests (Wave 1)
- [x] 17-02-PLAN.md - poolAllCellsRaw raw all-cells pooling sibling + test (Wave 1)

**Wave 2** *(blocked on Wave 1 — depends on 01+02):*
- [x] 17-03-PLAN.md - CofactorSuggestionDialog non-modal tool window (picker/scope/table/apply) (Wave 2)

**Wave 3** *(blocked on Wave 2 — depends on 03):*
- [x] 17-04-PLAN.md - NormalizationPane Suggest button + spinner write-back wiring (Wave 3)

Cross-cutting constraints:
- **Suggest window owned by `NormalizationPane.getStage()`, never `qupath.getStage()`** (critical correction #2; `NormalizationPane` is `APPLICATION_MODAL`) — spans plans 03 (uses the owner) and 04 (adds the `getStage()` accessor and passes it).
- **Estimator fed RAW values** via the new `CohortClusterModel.poolAllCellsRaw` sibling and `CellFeatureExtractor(..., null)` — never the z-scored `poolAllCells` (correction #3) — spans plans 02 (produces raw) and 03 (consumes raw).


## Milestone v1.7 - Training Performance

Goal: Make the human-in-the-loop train -> review -> retrain cycle fast enough to iterate on, and make training failures diagnosable, without changing a single prediction.

### Phase 18 - ML Training Performance and Diagnostics

Goal: Remove the algorithmic cost that dominated training (O(n^2) resampling run three times per click, quadratic LightGBM early stopping, six model fits of which two are discarded), and add the per-run logging and OOM handling needed to see where time goes. Every optimisation is bit-identical, gated by a checksum oracle captured against the pre-change implementation.
Depends on: Phase 1 (classifier core), and the existing Resampler/DualModelClassifier/TrainValMetricsComputer structure.
Requirements: PERF-01, PERF-02, PERF-03, PERF-04, PERF-05, PERF-06, PERF-08
Success Criteria:
1. Resampling on the real 19-class label set is at least an order of magnitude faster; measured 75x on the default SMOTE+Tomek (167.29s -> 2.22s on 16 cores).
2. Resampler output is bit-identical for every strategy and invariant to thread count, proven by ResamplerGoldenTest against constants captured pre-change.
3. Resampling runs at most twice per Train click instead of three times.
4. Each run writes a durable log with a per-phase timing/heap breakdown to <project>/celltune/logs/.
5. An OutOfMemoryError is reported with the phase that failed and leaves the UI usable.

**Plans:** 1 plan (executed inline, recorded retroactively)

Plans:
- [x] 18-01-PLAN.md - Resampler primitive/memoised/parallel fast core, shared 80/20 fold, LightGBM native early stopping, XGBoost hygiene, training log + phase timings + OOM capture, metrics opt-out and thread budget (completed 2026-08-03)

Cross-cutting constraints:
- **Bit-identical output is the acceptance criterion**, not a goal — ResamplerGoldenTest constants were captured against the pre-change code and must never be re-baselined to make a change pass.
- **Tomek's tie-break had to become explicit** (argmin, lowest index wins) before the search could be reordered or parallelised; the previous ascending-scan-with-strict-< got it implicitly.
- Deferred deliberately because they change results: HyperparameterTuner's missing min_gain_to_split, parallel dual-model fitting, tuner thread oversubscription, ADASYN's growing-scan quirk. Also deferred: PERF-07 feature-extraction parallelism.


## Progress

| Phase | Milestone | Requirements | Plans Complete | Status |
|-------|-----------|--------------|----------------|--------|
| 5. Verification Artifact Backfill | v1.1 | EVID-01,EVID-02,EVID-03 | 2/2 | Complete (2026-04-29) |
| 6. Nyquist Validation Coverage | v1.1 | NYQ-01,NYQ-02,NYQ-03,NYQ-04 | 0/0 | Not started |
| 7. Reliability Core Hardening | v1.1 | REL-01,REL-02,REL-03 | 0/0 | Not started |
| 8. UX Safety and Explainability Polish | v1.1 | UX-01,UX-02 | 0/0 | Not started |
| 9. Regression Tests for Hardening | v1.1 | TST-01 | 0/0 | Not started |
| 10. Documentation and Governance Refresh | v1.1 | GOV-01,DOC-01,DOC-02 | 1/1 | Complete (2026-04-29) |
| 11. Project Summary Cohort Outlier Analytics | v1.2 | COH-01,COH-02,COH-03,COH-04 | 2/2 | Complete (2026-04-30) |
| 12. Binary Ground Truth Bundle Export/Import | v1.3 | XFER-01,XFER-02,XFER-03,XFER-04 | 0/2 | Not started |
| 13. CN Spatial Clustering | v1.4 | CN-01,CN-02,CN-03,CN-04 | 0/1 | Planned |
| 14. Leiden Phenotype Clustering | v1.5 | LEI-01,LEI-02,LEI-03,LEI-04,LEI-05 | 0/1 | Planned |
| 15. All-Cells Leiden Clustering (True-Scanpy) | v1.5 | LEI-06,LEI-07,LEI-08,LEI-09,LEI-10 | 5/5 | Complete (2026-07-06) |
| 16. PCA Dimensionality Reduction | v1.5 | PCA-01..PCA-06 | 1/1 | Complete (2026-07-06) |
| 17. In-QuPath Cofactor Suggestion | v1.6 | COF-01..COF-08 | 4/4 | Verified (code); human UAT pending |
| 18. ML Training Performance and Diagnostics | v1.7 | PERF-01..PERF-06, PERF-08 | 1/1 | Executed + benchmarked; human UAT pending |
