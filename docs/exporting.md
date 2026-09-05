# Exporting results

### 12.1 Cell table export

**Menu:** *Extensions → SP Classify → Export ▸ Cell Table...*

For each selected image, writes `<ImageName>.csv` to your chosen folder with one row per detection:

| Column | Notes |
|---|---|
| `Image` | Source image name |
| `CellID` | QuPath cell UUID |
| `CentroidX_um` / `CentroidY_um` | Centroid in microns, 2 decimals (falls back to pixel × calibration) |
| `Area_um2` | Cell area in microns², 2 decimals |
| `Classification` | Current `PathClass` (empty if unclassified) |
| `ParentAnnotations` | All ancestor annotations, joined with `; ` |
| `ContainingAnnotations` | Every annotation whose ROI geometrically contains the cell centroid (captures overlapping regions the hierarchy discards), joined with `; ` |
| `Geometry_um` / `Geometry_px` | *(optional)* WKT `POLYGON` of the ROI outline, in microns or pixels — only written when polygon export is enabled |
| feature columns | One column per measurement **or metadata field** you tick in the export dialog |

Before exporting, a **Select Columns for Cell Table Export** dialog opens. It mirrors the *Select Features* dialog — search box, prefix dropdown, **Select Prefix** / **Clear Prefix**, **Select All** / **Clear All**, and a per-row checkbox — so you can pick exactly which columns land in the CSV. It pre-selects the curated subset (whole-cell means + any distance measurements). The chooser lists the **numeric measurements first, then the string metadata fields** (e.g. `CN Class`, `… original class`) — so text labels that aren't numeric measurements can now be exported too; filter for them by name if the list is long. Below the list, tick **Export cell polygons (geometry)** to include the ROI outline, and use the **Units** dropdown to choose **Microns (µm)** (`Geometry_um`) or **Pixels** (`Geometry_px`). Numeric measurements resolve to their value, metadata columns to their text value, and anything a cell doesn't have is written as `NA`.

### 12.2 Ground truth export & import

The extension's ground-truth files are a portable representation of your labelled cells **and** their feature vectors — they let you reuse labels across projects/workstations.

#### Export

**Menu:** *Extensions → SP Classify → Export ▸ Ground Truth...*

Header (commented):
```
# SP Classify Ground Truth Export
# Image: my_image.ome.tiff
# Exported: 2026-06-02T14:30:45
Image,Label,CentroidX,CentroidY,Feature1,Feature2,...
```

Exports **raw** feature values only — the values the classifier trains/predicts on. (Earlier versions offered a normalised `__norm` column set; that was removed when normalisation became clustering-only.) Only labelled cells are exported.

In multi-class mode the export pools labels from the current image plus all other project images. In **binary mode** use the dedicated menu item **Export ▸ Active Binary Ground Truth...** — it scopes to the active marker and includes previously-imported training rows from prior projects (so you can losslessly round-trip between projects).

#### Import

**Menu:** *Extensions → SP Classify → Import ▸ Ground Truth...*

After picking the CSV you choose one of two modes:

1. **Spatial Match** (per-image) — each imported row is matched to the nearest detection by centroid distance (you set the max threshold, default 20 px). Rows outside the threshold are skipped. Use this when you're re-importing labels onto the **same** image they were exported from.
2. **Training Data Only** (cross-project) — imports the feature vectors + labels without mapping back to cells. Use this when the source image isn't open in the current project; the rows feed straight into the next training run as if they were locally-labelled cells. The sidebar shows the count as `Imported rows: N`.

The binary equivalents are **Import ▸ Active Binary Ground Truth...** — same modes, but scoped to the active marker.

> **There is no "ground truth bundle" (ZIP)** currently — only the per-CSV import/export described here. The `.planning/phases/12` document scopes a bundle format as a future feature.

---
