# Project directory layout

Everything the extension writes is under `<project>/celltune/`:

```
celltune/
├── classifier-state.json         # Multi-class model (features, classes, model bytes, labels, normalisation)
├── composite-rules.json          # Saved CompositeClassificationRule objects (advanced/programmatic)
├── marker-table.json             # Imported marker table (auto channel switching) — persists across restarts
├── binary-registry.json          # markerName → state file path
├── labels_backup_YYYYMMDD_HHMMSS.json   # Auto-snapshot before each Train
│
├── image-labels/                 # Multi-class labels, one JSON per image
│   ├── slide1.json               #   { "<cellId>": "T-Cell", ... }
│   └── ...
│
├── binary-image-labels/<marker>/ # Same per-image JSON, scoped per binary marker
│   ├── CD3/slide1.json
│   └── ...
│
├── binary/                       # Binary classifier state files
│   ├── CD3.json
│   ├── CD8.json
│   └── ...
│
├── image-sampled/                # Cell IDs already sampled for review
├── image-predictions/            # Per-image Pred_ALL — consumed by Project Prediction Summary
```

JSON throughout. Model bytes are Base64-encoded inside the state files. Safe to commit `celltune/` to git if you want shared review history.

---
