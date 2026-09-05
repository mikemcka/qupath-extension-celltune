# Utility scripts

*Extensions → SP Classify → **Utility Scripts***

A grab-bag of common housekeeping operations that would otherwise live in one-off Groovy scripts. Each prompts for its parameters and reports what it did.

### 13.1 Filter Cells by Size & Circularity

Removes cell detections that are likely mis-segmented or artefacts. A dialog takes an optional **Min** and **Max** for both **Cell area (µm²)** and **Circularity** — leave any field blank for no bound. A cell is removed if it violates *any* active bound (e.g. `area > 500` **or** `circularity < 0.7`). Cells missing either measurement are skipped, not removed. The number of cells to be removed is shown for confirmation first; the operation acts on the **current image** only.

### 13.2 Resolve Hierarchy

Rebuilds parent/child relationships from ROI containment — equivalent to the `resolveHierarchy()` scripting call. Choose **Current image** (resolves and refreshes immediately) or **All project images** (confirms first, then resolves and saves every entry). Project-wide work runs in the background so QuPath stays responsive; the open image updates straight away.

### 13.3 Delete Measurements by Keyword

> ⚠️ **Destructive and not undoable.** Double-check the keyword against your actual measurement names — a loose keyword can delete more columns than you intend.

Removes every detection measurement whose name contains a keyword (case-insensitive by default; tick **Case sensitive** to match exactly). Choose **Current image** or **All project images**. Before deleting, the extension previews the exact list of matching columns and asks you to confirm — if nothing matches, it aborts. Project-wide saves each entry (open image first, the rest in the background).

### 13.4 Import GeoJSON Objects

> ⚠️ **For small-to-medium GeoJSON only.** This importer loads the whole file into QuPath's memory, so very large files (hundreds of MB / millions of objects) can exhaust the heap and crash QuPath. For those, use the dedicated headless pipeline instead: [github.com/BioimageAnalysisCoreWEHI/import_large_geojson](https://github.com/BioimageAnalysisCoreWEHI/import_large_geojson).

Imports annotations and detections from a `.geojson` (or gzipped `.geojson.gz`) file into the **current image**. Pick the file, then choose whether to **clear existing objects first** and whether to **resolve the hierarchy** afterwards (off by default — it is O(n²) and slow for many objects). Parsing streams the file feature-by-feature on a background thread; objects are added annotations-first (locked), then detections, and the image data is saved automatically.

### 13.5 Export Annotation Regions

> ⚠️ **Single-image, small-to-medium exports.** Pixels are streamed tile-by-tile so memory stays bounded, but very large regions or whole-project batch exports are far faster headless on HPC. For those, use the dedicated pipeline: [github.com/BioimageAnalysisCoreWEHI/export_large_annotation_regions](https://github.com/BioimageAnalysisCoreWEHI/export_large_annotation_regions).

Exports one or more annotation ROIs from the **current image** as polygon-**masked** OME-TIFFs — pixels outside the annotation shape are zeroed, so you get the annotation region rather than its rectangular bounding box. Enter a comma-separated list of annotation names (leave blank to export **all** annotations), set the **downsample**, **tile size**, **writer threads**, **compression** (LZW by default), and whether to write **BigTIFF** and a **pyramid**, then choose an output directory. Each region is written to `<image>__<annotation>.ome.tif` on a background thread, and a notification reports how many succeeded. Requires QuPath's built-in Bio-Formats extension (loaded by default).

### 13.6 Reset Project State

> ⚠️ **Destructive.** Permanently deletes everything the extension has saved for this project. Intended for starting over — e.g. when you've **copied a project** to trial different ML options and want a clean slate, since the `celltune/` state travels with the copy.

Deletes the project's entire `celltune/` folder: all labels and per-image label files, trained classifiers (multi-class **and** binary) and predictions, feature selection, normalisation, marker table, composite rules, and sampling/review state. It also resets the running session so nothing re-saves the old state.

**Safety net:** before deleting anything, a timestamped **`celltune_backup_<timestamp>.zip`** is written to the project folder. To undo a reset, unzip it back into the project folder (recreating `celltune/`). The action is guarded by a typed **`RESET`** confirmation.

**Images and detections are kept.** The extension's ground-truth **label points** and the **cell classifications** (predictions) it paints onto cells live in each image's `.qpdata`, *not* in `celltune/`. They are left in place unless you tick **"Also clear SP Classify label points and all cell classifications from every image"**, which strips classified point annotations and clears every cell's classification across **all** project images (this rewrites each image's data and runs on a background thread). Tissue/region annotations and unclassified points are never touched.

---
