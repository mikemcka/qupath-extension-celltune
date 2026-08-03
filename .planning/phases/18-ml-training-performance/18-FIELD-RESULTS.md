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
