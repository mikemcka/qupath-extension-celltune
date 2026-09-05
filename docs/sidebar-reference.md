# Reference: every setting in the sidebar

| Control | Default | What it does |
|---|---|---|
| **Rounds** | 500 | Maximum boosting rounds (50–1000). With Early stopping on this is a **limit**, not a target — models stop when they stop improving, so it usually costs nothing. With Early stopping **off** it is used literally; lower it. |
| **Max depth** | 6 | Tree depth (2–15). Higher = more complex interactions, more overfit risk. |
| **CPU threads** | 0 (all) | How many compute resources training may use. 0 = all of it. Lower it to keep working while training runs. Remembered between sessions. See §5.3. |
| **Images at once** | 1 | How many images are classified at once *after* training (1–8). Uses memory, not processor. Doesn't affect training itself. |
| **Model 1** | XGBoost | First ensemble model. |
| **Model 2** | LightGBM | Second ensemble model. **Pick a different type** for meaningful disagreement. |
| **Pool labels from all images** | ✅ | Train on labels from every image; auto-on/locked in binary mode. |
| **Enable data balancing** | ✅ | Apply resampling. Hides the strategy dropdown when off. |
| **Strategy** | SMOTE + Tomek | Resampling algorithm — see §5.4 table. |
| **Auto-tune hyperparameters** | ❌ | Automatically searches for better settings by training **200 models**. Hours on a big panel — plan for overnight. |
| **Early stopping** | ✅ | Stops training once the model stops improving, so you don't waste time. |
| **Train/val metrics** | ✅ | Produces the **Training Metrics** report. Costs about a third of training time; untick it to train faster and go without the report. |
| **Show top 10 feature importance after training** | ✅ | Auto-open SHAP plot after training. |
| **Auto-prune features** | ✅ | Drop near-constant & redundant features across the pooled, normalised training set before training; the top 5 highest-variance features per group are always kept. Non-destructive. See §[4.1](setup.md#41-select-features). |
| **Restrict to features shared with imported data** | ❌ | Case-insensitive intersection with imported ground-truth columns. |
| **Sample current image only** | ❌ | Restrict sampling/review to the open image. |
| **Filter by annotation keywords** | (blank) | Comma-separated substring filter on annotation names. |
| **Apply to which images...** | (all) | Open dual-list selector. Button label updates with count. |
| **Manual Label Mode** | — | Open floating labelling toolbar. |
| **Train** | — | Start training. Requires ≥10 labelled cells. |
| **Plot Confusion...** | (disabled) | Inter-model agreement matrix. Unlocks after training. |
| **Training Metrics** | (disabled) | Per-class precision/recall/F1 on 20% held-out split (≥20 labelled cells). |
| **Feature Importance...** | (disabled) | SHAP top-N per class. Unlocks after training. |
| **Enter Review Mode** | (disabled) | Sample disagreement cells for human review. Unlocks after predictions exist. |

### 14.1 Reference: preferences

Under **Edit → Preferences → SP Classify**. These are set once and left alone, which is why they aren't in the sidebar.

| Preference | Default | What it does |
|---|---|---|
| **Enable** | ✅ | Turn the extension off without uninstalling it. |
| **XGBoost histogram bins** | 0 | **Leave at 0.** How many cut-off values the model tries per measurement; 0 = the standard 256, which is the most accurate. Lowering it trades accuracy for speed. |

**XGBoost histogram bins — leave this alone.**

**The default is the most accurate setting. Changing it trades accuracy for speed, and SP Classify is built on the assumption that you would rather wait and get the better answer.** The rest of this section is here so you know what the setting is if you meet it — not as a suggestion to change it.

*What it is.* The classifier works by asking yes/no questions about one measurement at a time — *"is this cell's CD8 above 412?"* To find a good cut-off it has to try candidates. Trying every value in your data would be exact but painfully slow, so instead it sorts your cells by that measurement, chops them into buckets, and only tries the cut-offs *between* buckets. This setting is how many buckets. The standard 256 gives a possible cut-off at roughly every 0.4% of your cells, which is fine enough that you are unlikely to lose a real boundary.

*What lowering it does.* Fewer buckets means fewer cut-offs to test, so training is faster — 128 is about twice as fast, 64 about two and a half times. But it also means fewer places the model is allowed to cut. If two cell types are separated by a narrow intensity window on some marker and that window falls inside a single bucket, the model can no longer split them there.

*If you genuinely need the speed* — a very large panel, a deadline — then treat it as an experiment rather than a switch: train once at the default and export the cell table, then again at 128, and compare both the Training Metrics and the two exported class columns. Some cells **will** be classified differently. Only you can judge whether they are cells that matter. The training log header records the value used (`XGB max_bin: …`), so runs stay comparable afterwards.

---
