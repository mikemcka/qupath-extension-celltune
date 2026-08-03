# Phase 18 — first real-world run

Source: `exported_rois_qp_proj/celltune/logs/training-20260803_135557.log` (user's own run, 14 cores,
120 GB heap, post-phase-18 JAR).

## Shape

385,778 cells in the open image; 650 local labels + 4,266 pooled from 13 other images = 4,916
training cells. Auto-prune 4,205 → 1,886 features. **35 effective classes.** SMOTE+Tomek, early
stopping on, auto-tune off, metrics on.

The 35 classes are genuine, not a merged-name leak — pooling collapses `X-mergedInto(Y)` at
`TrainingOrchestrator:114`. The header's `Classes: 22` is the *current image's* label-store count,
which reads misleadingly next to "Training data: … 35 classes". Worth relabelling.

## Where the time went (730.7 s total)

| Phase | Time | Share |
|---|---|---|
| fit XGBoost | 230.77 s | 31.6% |
| early stop: XGBoost | 217.01 s | 29.7% |
| train/val metrics | 208.87 s | 28.6% |
| predict 385,778 cells | 37.94 s | 5.2% |
| fit LightGBM | 13.14 s | 1.8% |
| early stop: LightGBM | 12.10 s | 1.7% |
| resample (full) | 6.39 s | 0.9% |
| resample (80% fold) | 4.27 s | 0.6% |
| collect + extract | 0.19 s | 0.0% |
| flatten matrix | 0.05 s | 0.0% |

Peak heap 19,949 MB of 122,880 MB.

## Reading

**The resampling fix landed.** Both passes together are 10.7 s — 1.5% of the run. Pre-phase-18
this workload would have spent minutes there, three times over. It is no longer a target.

**XGBoost is now ~88% of the run** (~640 s across its three fits) against LightGBM's ~25 s on
identical data — an 18x gap. Structural: `multi:softprob` builds one tree per class per round, so
35 classes x 127 rounds is 4,445 trees, each scanning ~1,500 of the 1,886 pruned features.
LightGBM is fast partly because `min_gain_to_split=10` prunes hard, which shows in the slightly
lower validation score (0.739 vs 0.763).

**XGBoost is effectively trained three times**: the early-stop search boosts to ~147 rounds on the
80% fold, the metrics step retrains 127 rounds on *that same fold*, and the final fit retrains 127
rounds on 100%. After the phase-18 fold sharing, the first two run on a byte-identical array
(`PreparedFold.trainData()`, consumed at `TrainValMetricsComputer:225`), so one of them is
redundant.

`M1 train=1.000 / val=0.763` — XGBoost fits the training fold perfectly. Expected with 1,886
features and 35 classes over 4,916 labelled cells, but it does mean depth/rounds are generous for
the amount of label available.

## Actions

1. **Available today, no code change:** unchecking "Train/val metrics" removes 208.87 s (28.6%),
   taking this run from 12.2 to ~8.7 min.
2. **Best-round snapshot** (prototyped, being measured): have the early-stop search serialise the
   booster at its winning round via `toByteArray()` and hand it to the metrics step via
   `XGBoost.loadModel`, instead of retraining it. xgboost4j 2.1.4 has no `sliceModel`, so
   snapshot-at-best is the only way to capture that state. Small-scale check: predictions
   bit-identical (0 differing floats), snapshot overhead unmeasurable.
3. **Feature count** is the remaining lever — XGBoost hist cost is roughly linear in it, and
   auto-prune still leaves 1,886.

## Batch apply

Ran at 8 images at once (user raised it from the default 1) and overlapped loads with predictions
cleanly, reaching 13/14 saved by the end of the captured log.

---

# Follow-up: XGBoost best-round snapshot (measured)

`SnapBench` (standalone, `/vast/scratch/users/mckay.m/celltune-bench`), 14 cores, at the shape of
the field run: 35 classes, 1,886 features, 6,851 train rows, 996 val rows, maxRounds 200.

|  | time |
|---|---|
| A) search (no snapshot) | 426.58 s |
| A) retrain to best round | 424.32 s |
| **A) total today** | **850.89 s** |
| B) search (+ snapshot) | 476.27 s |
| B) loadModel from bytes | 0.12 s |
| **B) total proposed** | **476.38 s** |
| **saved** | **374.51 s (44.0%)** |

**Predictions identical — 0 differing floats, max abs diff 0.000e+00.**

This is the worst case for the snapshot: validation loss improved on all 200 rounds, so 200
snapshots were taken of a model that reached 9,981 KB. The field run's search stopped at round 127
of 200 with improvements tapering, so both the overhead and the absolute saving are smaller there —
the metrics XGBoost fit was ~195 s, so expect ~160-175 s net, around 22% of that 730 s run.

Notable: of the +49.69 s snapshot overhead, only **4.29 s** is `toByteArray` itself. The rest is GC
churn from ~2 GB of transient arrays. That is why `searchRounds` takes a `snapshot` flag rather
than always snapshotting — `DualModelClassifier` sets it only when the metrics step will actually
consume the result (`computeMetrics && nRealSamples >= 20`). If overhead ever matters more, the
lever is snapshotting less often, not serialising faster.

## Implementation

`XGBoostModel.searchRounds(..., boolean snapshot, ...)` returns `RoundSearch(bestRounds,
bestModel)`; `findBestRounds` is now a delegating overload. `DualModelClassifier` passes the bytes
to `computeTrainValMetrics`, whose trainer lambdas restore via `loadFromBytes` instead of fitting —
**guarded on array identity** against `PreparedFold.trainData()`, so if the metrics step ever
trains on anything other than the fold the search used, it silently falls back to a real fit.

`XGBoostRoundSearchTest` (4 tests, run against the real natives — verified not skipped) pins:
the restored snapshot predicts identically to a fresh fit of the same round count; taking a
snapshot does not change which round the search picks; no snapshot is produced when not requested;
`findBestRounds` still returns just the count.

Only XGBoost is wired up. LightGBM's whole search is ~12 s, so there is nothing to reclaim there.
