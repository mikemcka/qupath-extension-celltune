# Quick start — multi-class workflow

Build one classifier that distinguishes any number of cell types (e.g. T-cell / B-cell / Macrophage / Tumour / Stroma).

```
Select Features  →  Clustering Normalisation  →  Create classes (Class Control)
       ↓
Import Marker Table (optional, for auto channel switching)
       ↓
Manual Label Mode  → label ~20–50 cells across the open image
       ↓
Apply to which images... → choose images to predict on
       ↓
Set Images at once, pick settings (Pool labels, Balancing, Early stopping, etc.)
       ↓
Train  →  inspect Training Metrics + Confusion Matrix
       ↓
Enter Review Mode  →  correct cells the models disagree on
       ↓
(repeat: more labels → Train → Review)
       ↓
Project Prediction Summary  →  flag outlier slides → re-label as needed
       ↓
Export Cell Table  or  Export Ground Truth
```

Detail per step is in §[4](setup.md), §[5](multiclass.md), §[7](review-mode.md).

---
