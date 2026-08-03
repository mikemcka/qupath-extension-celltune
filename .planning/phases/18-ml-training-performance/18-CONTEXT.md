# Phase 18 — ML Training Performance and Diagnostics (v1.7)

## Why

Training was slow, and dramatically worse with resampling on. Profiling the actual code path
showed the cause was algorithmic, not hardware:

- `Resampler` was brute-force O(n²·d), single-threaded, and ran **three times per Train click**.
- Two of those three runs were provably identical — early stopping and the train/val metrics
  both built an 80/20 fold from the same rows with the same ratio and the same seed (42), then
  resampled it with the same strategy.
- LightGBM's early stopping was accidentally O(rounds²): it called `predictForMat` on the whole
  validation fold every round, re-scoring all trees from scratch and re-marshalling the matrix
  across JNI.
- LightGBM's GPU probe ran a **full `numRounds` training loop** under `device_type=gpu` before
  falling back to CPU, on every `train()` call, against CPU-only binaries.
- The default path fitted six models per click; two were evaluation copies discarded immediately.
- There was no way to see where time went, and `OutOfMemoryError` (an `Error`, not an `Exception`)
  escaped the training thread's handler entirely — silent death, hung progress dialog, Train
  button stuck disabled.

GPU was considered and rejected: see "Explicitly not doing" below.

## Baseline (measured, not estimated)

Real pooled label distribution from the project's 14 images — 5,839 labelled cells across 19
effective classes (`X-mergedInto(Y)` collapsed to `Y`), majority 1,348, minority 132. SMOTE has to
synthesise 19,773 rows, taking the set to 25,612 before Tomek. 200 features (a realistic
post-`FeaturePruner` width; the raw panel is 1,780). 16-core node.

| Strategy | Before | After | Speedup |
|---|---|---|---|
| SMOTE | 0.91 s | 0.16 s | 5.7x |
| TOMEK | 6.21 s | 0.15 s | 41x |
| **SMOTE_TOMEK** (default) | **167.29 s** | **2.22 s** | **75x** |
| ADASYN | 12.67 s | 1.08 s | 11.7x |
| ADASYN_TOMEK | 155.51 s | 3.33 s | 47x |

Output row counts were identical in every case (25,612 / 4,892 / 15,730 / 25,507 / 21,275) —
independent confirmation of bit-identity on real-shaped data, beyond the unit fixtures.

At three runs per click the default strategy alone accounted for ~500 s; it is now ~4 s across
the two remaining runs.

Harness: `ResamplerBenchmark` (JUnit, gated on `-Dcelltune.bench=true`) and a Gradle-free
standalone runner under `/vast/scratch/users/mckay.m/celltune-bench` for detached SLURM runs.

## What changed

**`Resampler`** — internal rewrite, public surface unchanged:
- Primitive `float[][]`/`int[]` working buffer with exactly-sized capacity, replacing
  `List<float[]>`/`List<Integer>`; removes `Integer` unboxing from Tomek's innermost loop.
- SMOTE/ADASYN kNN memoised. The generation loop cycles `i % m` over an unchanging
  `classIndices` reading unchanging rows, so the identical neighbour set was being recomputed
  `ceil(needed/m)` times.
- Tomek walks a per-class index partition (different-class filter becomes structural), abandons a
  distance once it provably cannot beat the current best, and runs the outer loop in parallel.
- Removal is a single compacting pass instead of `ArrayList.remove(int)` per index.
- The "nearest different-class neighbour" tie-break is now **explicit** (argmin, lowest index
  wins) rather than implied by an ascending scan with strict `<`. That is what makes the result
  independent of visit order, and therefore of thread count.

**Orchestration** — `TrainValMetricsComputer.prepare` builds the shared 80/20 `PreparedFold` once;
`DualModelClassifier` passes it to the metrics step. Degrades to building its own when early
stopping is off or both models are Random Forest. `compute`'s original signature is retained as a
delegating overload so its injected-trainer testability is untouched.

**LightGBM** — `addValidData` + `getEval(1)` for native incremental validation (the reference
dataset is mandatory: LightGBM rejects a valid set with different bin mappers). Falls back to the
old manual scoring, decided once, if `getEval` misbehaves. GPU probe removed, mirroring
`XGBoostModel`. `updateOneIter`'s `is_finished` return is now honoured.

**XGBoost** — dropped the unused `train` watch (scored the full training matrix every round with
no consumer); disposed the leaked `Booster` in `findBestRounds` and in the tuner's fold evaluation
(~100 per tuned model).

**Diagnostics** — `TrainingLogRecorder` writes a flushed-per-line log to
`<project>/celltune/logs/training-<timestamp>.log` with a self-contained header; `PhaseTimer`
records per-phase wall time and heap, ending in a descending "where the time went" summary;
`OutOfMemoryError` is caught and reported with the phase that was in flight and the heap
high-water mark; `TrainingMemoryCheck` warns pre-flight (replacing the never-called
`CellTuneExtension.checkTrainingMemory`, extended to account for resampling inflation). Text-area
appends are batched instead of one `Platform.runLater` per line.

**Controls** (both default to today's behaviour) — "Train/val metrics" checkbox to skip the two
throwaway model fits; `celltune.trainingThreads` preference plus a "Train threads:" spinner,
routed through the new `util/TrainingThreads` which `XGBoostModel`, `LightGBMModel` and
`Resampler` now consult instead of each calling `availableProcessors()` independently.

## Bit-identity guarantee

`ResamplerGoldenTest` is the oracle. Three deterministic fixtures (moderate imbalance;
cross-class exact duplicates to pin tie-breaks; severe imbalance where `needed/m` ~ 99 to stress
the memoisation) x five strategies, asserting exact output size, exact per-class counts, and an
FNV-1a checksum over the raw bits of every value and label. The constants were captured against
the **pre-change** implementation. Plus: invariance across thread counts, stability across
repeated parallel runs, and "input lists never mutated" for every strategy.

The one place output is not formally bit-identical is LightGBM's `bestRound`, which now comes
from LightGBM's own log-loss rather than the hand-rolled one. Same formula and same `/n`; they can
only diverge if two rounds tie to ~15 significant figures. Worth confirming on real data before
relying on it.

## Explicitly not doing

- **GPU (`xgboost4j-gpu`)** — hundreds of MB added to the fat JAR, Linux x86_64 only, needs a
  matching CUDA driver, and contradicts the "install only the shadow JAR" rule. LightGBM4J ships
  no GPU build at all, so model 2 would stay on CPU regardless. Most importantly it targets the
  wrong cost: GPU `hist` pays off above ~1e5-1e6 rows, while labelled training sets here are
  1e3-1e5 and the bottleneck was O(n²) resampling.
- **HNSW-backed approximate resampling** — would make Tomek ~O(n log n), but jelmerk 1.2.1's
  unseeded `assignLevel` would make the *training set itself* non-reproducible run to run.
- **Tomek symmetry halving** — 2x on top of an already 40x win, at real risk to the tie-break
  invariant everything else rests on.

## Deferred (changes results — separate PR, with before/after comparison)

- `HyperparameterTuner` is missing `min_gain_to_split=10`, so it scores a different model than
  `LightGBMModel.buildParams` configures. Real bug; fixing it changes tuned hyperparameters.
- Parallel model-1/model-2 training (~1.4-1.7x): halving each model's thread budget changes float
  accumulation order in histogram construction.
- `HyperparameterTuner` thread oversubscription (`threadsPerFoldLGB = totalCores` with 5
  concurrent folds) and its per-trial fold-matrix rebuild.
- ADASYN's `knnAll` scans a row list that has grown with earlier classes' synthetics while `k` is
  fixed from a pre-loop capture. Arguably a bug; preserved bit-for-bit here so the checksums hold.
- 18-05 (feature extraction / cross-image pooling parallelism) was planned but not executed:
  labelled rows are still extracted twice per click, the training path still uses the serial
  `extractRow`, and `TrainingOrchestrator.poolLabelsFromOtherImages` is still sequential.

## Verification status

`./gradlew check` clean (full suite + spotlessCheck); `shadowJar` builds. **Manual QuPath UAT
outstanding** — see `18-HUMAN-UAT.md`.
