# Reference: every SP Classify menu item

All under *Extensions → SP Classify*.

| Item | Requires | Action |
|---|---|---|
| Binary Classifiers... | Project | Open the binary classifier manager (create/open/delete per-marker classifiers). |
| Composite Classification... | Project + ≥1 trained binary | Apply trained binary classifiers and assign composite labels. |
| Class Control... | Project | Add/Delete/Merge/Undo Merge classes. |
| Select Features... | Project | Pick which measurement columns are used for training. |
| Clustering Normalisation | Project | Per-feature arcsinh/sqrt with shared cofactor (clustering-only; classifier uses raw). |
| Batch Normalisation... | Project | UniFORM per-image marker-intensity alignment across a cohort; fit + QC, streamed into clustering + ML or written as `(batchnorm)` columns. See §[19](batch-normalisation.md). |
| Project Prediction Summary... | Project | Cohort QC, anomaly scoring, per-image flags. |
| Image Pixel Prescreen... | Project | Cells-free whole-image QC: per-channel pixel statistics on a low-res pyramid level, cohort z-scores, verdicts/flags (background-heavy, saturated, weak signal, intensity outlier), CSV export. See §[17](pixel-prescreen.md). |
| Intensity Heatmaps... | Open image with detections | Phenotype × marker mean-intensity heatmap (z-score coloured), per-image / project-combined, PNG/CSV export. See §[9](intensity-heatmaps.md). |
| Generate Distance Measurements... | Project | Batch spatial distances (annotation-signed, cross-class, same-class NN) across selected images. See §[10](distance-measurements.md). |
| Scatter Plots and Clustering... | Open image with detections | Interactive PCA/UMAP embedding + k-means clustering, annotation/class gating, cluster→class assignment, and a **Scope** toggle for cohort-wide clustering across images in the same window. See §[11](scatter-clustering.md). |
| Export ▸ Cell Table... | Open image with detections | One CSV per selected image. |
| Export ▸ Ground Truth... | Open image with labels (multi-class) | Portable labels + feature vectors CSV. |
| Export ▸ Active Binary Ground Truth... | Binary mode active + open image with labels | Same as above, scoped to active marker. |
| Import ▸ Marker Table... | Open image | Load cell-type → markers mapping for review channel switching. |
| Import ▸ Ground Truth... | Open image (multi-class) | Spatial-match or training-data-only mode. |
| Import ▸ Active Binary Ground Truth... | Binary mode active + open image | Same as above, scoped to active marker. |
| Utility Scripts ▸ Filter Cells by Size & Circularity... | Open image with cells | Remove cells outside optional area/circularity bounds (current image). See §[13.1](utility-scripts.md#131-filter-cells-by-size--circularity). |
| Utility Scripts ▸ Resolve Hierarchy... | Open image or project | Rebuild parent/child relationships (`resolveHierarchy()`); current image or whole project. See §[13.2](utility-scripts.md#132-resolve-hierarchy). |
| Utility Scripts ▸ Import GeoJSON Objects... | Open image | Import objects from a (gzipped) GeoJSON into the current image — **small-to-medium files only**. See §[13.4](utility-scripts.md#134-import-geojson-objects). |
| Utility Scripts ▸ Export Annotation Regions... | Open image | Export annotation ROIs from the current image as polygon-masked OME-TIFF(s) — **single-image, small-to-medium**. See §[13.5](utility-scripts.md#135-export-annotation-regions). |
| Utility Scripts ▸ Delete Measurements by Keyword... | Open image or project | **Destructive:** delete detection measurements matching a keyword, with preview/confirm. See §[13.3](utility-scripts.md#133-delete-measurements-by-keyword). |
| Utility Scripts ▸ Reset Project State... | Project | **Destructive:** wipe the project's `celltune/` state (labels, models, predictions, settings) for a clean slate; writes a backup zip first, typed-`RESET` confirm, optional per-image artifact stripping. See §[13.6](utility-scripts.md#136-reset-project-state). |

---
