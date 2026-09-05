# SP Classify for QuPath

A [QuPath](https://qupath.github.io/) 0.7 extension that brings **human-in-the-loop active
learning** to cell classification in highly multiplexed images. It trains two gradient-boosted
models (XGBoost + LightGBM) simultaneously, flags the cells where they disagree, and presents
those disputed cells for review — an iterative loop that progressively improves accuracy from a
few hundred labels rather than thousands.

!!! info "Citing SP Classify"
    If you use this tool, please cite both the software and the CellTune paper it derives from,
    plus any method papers for the features you use. See **[How to cite](how-to-cite.md)**.

## What it does

- **Dual-model disagreement detection** — two different models (default **XGBoost + LightGBM**;
  Random Forest also available) train on the same labels; the cells where they disagree are
  flagged for review. Training runs on the CPU.
- **Weighted uncertainty sampling** — a 5-tier strategy (FOV balance → most-confused classes →
  rare types → user-specified confusions → random fill) puts review effort where it matters.
- **Interactive review mode** — step through sampled disagreements and accept or correct a
  prediction in one click, with per-cell spatial context and optional auto channel-switching.
- **Beyond classification** — marker-based gating, project-wide clustering (k-means & Leiden),
  cellular-neighborhood analysis, intensity heatmaps, batch normalisation (UniFORM), spatial
  distance measurements, and whole-image QC.

## The active-learning loop

```
① Seed labels via Manual label mode
        │
        ▼
② Train dual classifiers (XGBoost + LightGBM)
        │
        ▼
③ View inter-model confusion matrix → per-class agreement rates
        │
        ▼
④ Enter review mode → step through disagreement cells → assign/correct labels
        │
        ▼
⟳ Merge new labels → retrain → repeat until satisfied
```

## Getting started

1. **[Install & launch](install.md)** the extension (QuPath 0.7, Java 25).
2. Follow the **[multi-class quick start](quickstart-multiclass.md)** or the
   **[binary + composite quick start](quickstart-binary.md)**.
3. Dig into the **[full multi-class workflow](multiclass.md)** and **[review mode](review-mode.md)**.

!!! note "Relationship to CellTune"
    SP Classify provides similar functionality to parts of
    [CellTune](https://celltune.org/) by the Keren Lab, reimplemented as a QuPath extension. It is
    an independent tool, not the official CellTune software.
