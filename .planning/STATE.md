---
gsd_state_version: 1.0
milestone: v1.7
milestone_name: Training Performance
status: awaiting_uat
last_updated: "2026-08-03T00:00:00.000Z"
progress:
  total_phases: 1
  completed_phases: 0
  total_plans: 1
  completed_plans: 1
  percent: 100
---

# Project State

## Current Position

Phase: 18 — ML Training Performance and Diagnostics (v1.7)
Plan: 1/1 complete (executed inline on branch perf/18-01-training-instrumentation, commit 73eef4a)
Status: Executed + benchmarked — awaiting live-QuPath human UAT (6 sections in 18-HUMAN-UAT.md)
Headline: SMOTE+Tomek on the real 19-class label set went 167.29s → 2.22s (75x) with bit-identical
output; resampling now runs twice per Train click instead of three times. Full suite + spotlessCheck
clean; shadowJar builds.

Previous phase: 17 — In-QuPath Cofactor Suggestion (v1.6), executed + code-verified, human UAT
still pending (5 items in 17-HUMAN-UAT.md).
Last activity: 2026-08-03 — Phase 18 executed inline: Resampler primitive/memoised/parallel fast core
(bit-identical, 75x on the default strategy), shared 80/20 fold, LightGBM native early stopping + GPU
probe removal, XGBoost watch/leak hygiene, training log + phase timings + OOM capture, metrics opt-out
and thread budget. `./gradlew check` clean, shadowJar builds. Benchmarked on a 16-core node against the
real COMET label distribution.
Prior activity: 2026-07-08 — Phase 17 executed (hand-orchestrated, gsd-sdk absent; 4 plans sequential on 0.2.1-update). Full suite 428 tests / 0 failures, spotlessCheck clean. Verifier: 5/5 must-haves + 8/8 COF requirements verified in code, both corrections (#2 owner-stage, #3 raw pooling) confirmed; status human_needed (5 live-UI checks only). Code review: 0 crit/0 high/1 med (MED-01 NaN-drop contract dead via extractMatrix 0f coercion)/4 low — advisory.

## Carry-Forward Context

- Phase 18 (v1.7) benchmark harness lives at `/vast/scratch/users/mckay.m/celltune-bench` (Gradle-free
  standalone runner + snapshotted pre-change classes, for detached SLURM runs) and as
  `ResamplerBenchmark` gated on `-Dcelltune.bench=true`. Baseline numbers are in 18-CONTEXT.md.
- Phase 18 test data: real label distribution taken from `/vast/scratch/users/mckay.m/exported_rois_qp_proj`
  (14 images, 5,839 labelled cells, 19 effective classes after collapsing `X-mergedInto(Y)`, 1,780 raw
  features). That project is the UAT target.
- **ResamplerGoldenTest constants are an oracle captured against the pre-optimisation code.** A failure
  means the resampled training set changed, which changes the trained model. Never re-baseline them to
  make a change pass.

- Milestone v1.5 (Graph-based Phenotype Clustering) COMPLETE — Phases 14 (Leiden transfer), 15 (all-cells true-scanpy Leiden), 16 (conditional PCA) all recorded and complete.
- v1.1 reliability/verification hardening debt (phases 6-9) still pending; also open: Phase 12 (binary ground-truth bundle), Phase 13 (CN spatial clustering).
- Previous milestone archived: v1.0 Binary Composite Classification.
- Verification evidence debt for phases 1-3 is normalized via formal VERIFICATION artifacts.
- Documentation and build guidance now include project prediction summary workflow and reproducible build steps.
- Remaining milestone scope: Nyquist validation coverage and reliability hardening phases 6-9.
- Additional completed phase: 11 (v1.2) cohort outlier analytics for project summary rare-type enrichment and anomaly triage.

## Deferred Items (Carried)

| Category | Item | Status |
|----------|------|--------|
| verification | phase-01-verification-md-missing | resolved in phase 5 |
| verification | phase-02-verification-md-missing | resolved in phase 5 |
| verification | phase-03-verification-md-missing | resolved in phase 5 |
| validation | nyquist-validation-phase-1-to-4-missing | active in v1.1 |
| performance | perf-07-feature-extraction-parallelism | deferred from phase 18 (labelled rows still extracted twice per click; training path still serial; cross-image pooling still sequential) |
| correctness | tuner-missing-min-gain-to-split | deferred from phase 18 — HyperparameterTuner scores a different model than LightGBMModel configures; fixing it changes tuned hyperparameters |
| performance | parallel-dual-model-fitting | deferred from phase 18 (~1.4-1.7x) — splitting the thread budget changes float accumulation order in histogram construction |
| performance | tuner-thread-oversubscription | deferred from phase 18 — threadsPerFoldLGB = totalCores with 5 concurrent folds; also per-trial fold-matrix rebuild |
| correctness | adasyn-growing-scan-quirk | deferred from phase 18 — knnAll scans a list grown by earlier classes' synthetics while k is fixed pre-loop; preserved bit-for-bit so checksums hold |

## Session Continuity

- Recommended next action: **Phase 18 manual QuPath UAT.** Build the fat JAR
  (`./gradlew shadowJar` → `build/libs/qupath-extension-celltune-0.2.3-all.jar`), delete any older
  CellTune JAR from the extensions folder, restart QuPath, open
  `/vast/scratch/users/mckay.m/exported_rois_qp_proj/project.qpproj`, and walk
  `.planning/phases/18-ml-training-performance/18-HUMAN-UAT.md`. The critical checks are §2 (predicted
  classes and resampled distribution unchanged vs a pre-change run) and §6 (LightGBM early stopping is
  the one change that is not formally bit-identical). Then open a PR from
  `perf/18-01-training-instrumentation`.
- Also still open: Phase 17 (In-QuPath Cofactor Suggestion, v1.6) is EXECUTED + code-verified; **manual QuPath UAT pending** — build the fat JAR (`build/libs/qupath-extension-celltune-0.2.1-all.jar`, produced by orchestrator), load it in QuPath v0.7, and walk the 5 checks in `.planning/phases/17-cofactor-suggestion/17-HUMAN-UAT.md` (arcsinh-only Suggest… button; Modality.NONE window interactive under the APPLICATION_MODAL Normalise pane; picker independence; per-feature table + global over both scopes; no-mutation Apply). Then `/gsd-verify-work 17` to record results, or open a PR from `0.2.1-update`. Optional cleanup: address code-review MED-01 (whole-project heterogeneous-panel NaN→0f) via `/gsd-code-review-fix 17` or defer. Other open work: Phase 13 CN spatial clustering; Phase 12 binary ground-truth bundle; `/gsd-plan-phase 6` to resume v1.1 hardening.
- Note for phase-15 verification: `AllCellsResult.recall` is still the documented `-1.0` sentinel (carried from 15-04) — 15-05's status line shows the measured ANN recall only when a real value is available and omits the clause otherwise, deliberately not fabricating a number. A future small `LeidenModel`/`CohortClusterModel` API change could fully satisfy D-09's exact "ANN recall 0.982 — passed" wording; this is a known, documented gap, not a defect.
- Note carried from 15-01: HnswKnnIndex's reproducible=true build is "best-effort deterministic" (documented caveat) — HnswIndex cannot be subclassed from external source in jelmerk 1.2.1 (bytecode-verified); do not attempt the seeded-assignLevel subclass approach again without new information
- Note: conditional PCA (`ScatterMath.pcaReduce`, commits `41415fa`..`decc05f`) was added inline after 15-05's checkpoint approval and has now been recorded as Phase 16 (16-01-PLAN.md/16-01-SUMMARY.md, PCA-01..06 complete in REQUIREMENTS.md) — no further recording action needed.
- Alternative: manually verify phase 14 Leiden flow in QuPath (build/libs/qupath-extension-celltune-0.2.0-all.jar); or /gsd-execute-phase 13 (CN spatial clustering); or /gsd-plan-phase 6 to resume v1.1 hardening
