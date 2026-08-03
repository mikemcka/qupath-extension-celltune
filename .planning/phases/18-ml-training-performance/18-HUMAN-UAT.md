# Phase 18 — Human UAT (live QuPath)

Build: `./gradlew shadowJar` → `build/libs/qupath-extension-celltune-0.2.3-all.jar`.
**Delete any older `qupath-extension-celltune-*-all.jar` from the extensions folder first**, then
copy this one in and restart QuPath fully.

Test project: `/vast/scratch/users/mckay.m/exported_rois_qp_proj/project.qpproj`
(14 images, ~5,839 labelled cells across 19 effective classes, state in `celltune/`).

## 1. Speed — the headline claim

Train with the defaults (Balancing = SMOTE + Tomek, Early stopping on, Auto-tune off, Train/val
metrics on) and watch the training log.

- [ ] Training completes in a small fraction of the previous time. The three resampling passes
      that dominated the run should now be seconds, not minutes.
- [ ] The log ends with a "Where the time went" table. Confirm resampling is no longer the top
      entry — if it still is, capture the log, the remaining phases are the next targets.

## 2. Results unchanged — the safety claim

> **Leave `CPU threads` at 0 (auto) for this whole section.** It now feeds XGBoost's `nthread` and
> LightGBM's `num_threads`, and both build their histograms with a thread-count-dependent summation
> order — so changing it *will* move predictions slightly. That is expected behaviour, not a
> regression, but it invalidates the diff below. Before this change the value was always
> `availableProcessors()`, so a same-machine comparison was automatically like-for-like.

- [ ] The log's `Resampled distribution:` line reports the **same** per-class counts and total as
      a pre-change run. (`git stash` the branch, build the old JAR, and keep that log to diff, or
      compare against a log from before this change.)
- [ ] Reported macro-F1 (`Macro F1: M1 train=… val=… | M2 train=… val=…`) matches the old run to
      the printed precision.
- [ ] Export the cell table before and after and diff the predicted class column — expect no
      differences. Any diff is a defect, not a tolerance.

## 3. Training log file

- [ ] `<project>/celltune/logs/training-<timestamp>.log` exists after a run and contains the
      header (cells, labelled, features, classes, balancing, model types, threads, max heap),
      the per-phase timings, and the summary.
- [ ] Run training a few times and confirm old logs are pruned to the newest 20.
- [ ] Open a single image **outside** a project and train: it must still work, just without a log
      file.

## 4. Out-of-memory path

Launch QuPath with a deliberately small heap (Edit → Preferences → Maximum memory, or `-Xmx2g`)
and train on the largest image with all features selected.

- [ ] A pre-flight dialog warns before training starts, naming the estimate and the heap, and
      offers Proceed/Cancel. Cancel must abort cleanly.
- [ ] If it does run out of memory: a specific dialog appears naming the phase, the Train button
      is re-enabled, and the progress dialog is not left spinning. Previously this hung silently.
- [ ] The log file ends with `!! FAILED during: <phase>` plus the heap high-water mark.

## 5. New controls

- [ ] **Train/val metrics** checkbox, default ON. Unchecking it makes training visibly faster and
      the log says "Skipping train/val metrics (unchecked)". The Training Metrics button then
      reports no metrics available rather than showing stale ones from an earlier run.
- [ ] **CPU threads:** spinner, default 0 (= auto). Set it to 2, retrain, and confirm the log's
      `Threads:` line reflects it and CPU use drops. Confirm it persists across a QuPath restart.
- [ ] Confirm the two labels read as different knobs: **CPU threads** (inside one training run)
      versus **Images at once** (parallel *applying* of a classifier to other images).

## 6. Auto-tune with early stopping — the metrics must describe the deployed model

The round search can keep its winning XGBoost model so the metrics step does not refit it. That is
only valid when the hyperparameters have not moved since, and auto-tune replaces all of them.

- [ ] Model 1 = XGBoost, **Early stopping ON, Auto-tune ON**, Train/val metrics ON. Train.
- [ ] The log must **not** show the "kept the round-N model" line — with auto-tune on, the search
      is told not to snapshot, and the metrics step does a real fit with the tuned settings.
- [ ] The rounds/depth the log reports for the final fit are the tuned ones, and the reported
      macro-F1 is consistent with them (a wildly optimistic or stale-looking score is the symptom
      this guards against).
- [ ] Re-run with Auto-tune OFF: the "kept the round-N model" line *should* appear, and the
      train/val metrics should be unchanged from a pre-change run.

## 7. XGBoost `max_bin` — a speed/accuracy trade you have to price yourself

XGBoost is ~87% of a training run and `max_bin` is the only remaining lever on it. Off by default
(Edit → Preferences → CellTune → **XGBoost histogram bins**, 0 = XGBoost's own 256), because
**it changes predictions**. The benchmark measured 2.0× at 128 bins and 2.6× at 64 on the real
shape, but on synthetic data — the timing ratios transfer, the accuracy effect does not.

Do this **after** §2, so the baseline diff is done against unmodified defaults.

- [ ] With bins = 0, note the training log's `fit XGBoost` and `early stop: XGBoost` times, the
      `Macro F1` line, and export the cell table.
- [ ] Set bins = 128, retrain. Confirm the header records `XGB max_bin: 128` — a log without that
      line cannot be compared against another.
- [ ] Expect roughly half the XGBoost time. Compare macro-F1 and diff the exported class column
      against the bins = 0 export: **predictions will differ** — the question is by how much and
      whether the disagreements land on cells you care about.
- [ ] Repeat at 64 if 128 looks acceptable. Keep whichever setting trades time for accuracy in the
      direction you want; there is no correct answer here, only your answer for this panel.

## 8. Auto-tune now tunes the real model

The tuner cross-validated LightGBM without `min_gain_to_split=10` and then applied the winning
settings to a booster that has it. Both fold evaluators now share the model classes' own parameter
builders. This **changes auto-tuned results** — it should improve them, since the search is finally
scoring what gets deployed. Default runs are untouched: the tuner only runs with Auto-tune ticked.

- [ ] Model 2 = LightGBM, **Auto-tune ON**. Train, and note the tuned rounds/depth/rate/subsample
      the log reports plus the resulting macro-F1.
- [ ] Compare against an auto-tuned run on the baseline JAR. Expect **different chosen
      hyperparameters** — that is the fix working, not a regression. What matters is that the
      macro-F1 does not get worse.
- [ ] Auto-tune should also be somewhat faster: with `min_gain_to_split` applied, folds run out of
      worthwhile splits and stop early instead of grinding through every round as no-ops.

## 9. LightGBM early stopping (the one non-bit-identical change)

Early stopping now reads LightGBM's own log-loss instead of a hand-rolled one.

- [ ] With Early stopping on, the log's `LightGBM early stopping: best round N/M` line reports a
      plausible N, and training does not regress in quality (compare macro-F1 with a pre-change
      run).
- [ ] Optional deeper check: if N differs from the old run, confirm the resulting validation
      macro-F1 is equivalent. A shift of a round or two is acceptable; a quality drop is not.
