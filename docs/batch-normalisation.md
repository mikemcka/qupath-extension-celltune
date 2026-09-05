# Batch normalisation (UniFORM)

Multiplex staining varies image-to-image — the same marker can sit at a different intensity on different slides or runs. **Batch normalisation** aligns each image's marker-intensity distribution to a common reference so that clustering and the classifier see one consistent intensity scale across a cohort, instead of learning the batch. SP Classify implements the **feature-level UniFORM** method (Wang et al., *Cell Reports Methods* 2025; see [README ▸ References](how-to-cite.md)): for each marker it aligns per-image log-intensity histograms by the rigid shift that best matches a reference, and that shift maps back to a single per-image multiplicative **gain** per channel. Because it is a translation in log-space, the distribution's *shape* is preserved — only its location moves — so it is conservative about erasing real biology.

Open it from **Extensions ▸ SP Classify ▸ Batch Normalisation…**.

> The gain is computed from each channel's **Cell: Mean** intensities and then applied to every statistic of that channel. Only intensity measurements are corrected; foundation-model embeddings are excluded. Nothing is overwritten unless you explicitly write columns — see §19.4.

### 19.1 When to use it

- You cluster or train **across multiple images/slides** stained in different runs and see clusters or classes that track the *slide* rather than the biology.
- It complements the clustering normalisation of §[4.2](setup.md#42-clustering-normalisation) (arcsinh/sqrt): that applies the **same** transform to every image and so cannot remove per-slide offsets (it says as much in its own limitations). Batch normalisation is the missing per-image step. Single-image analysis does not need it.

### 19.2 Fitting — step by step

1. **Correct measurements** — *Choose measurements…* picks the marker intensities to align (embeddings are excluded automatically).
2. **Images** — *Choose images…* picks the cohort. *Also include projects ▸ Add project…* pools images from other SP Classify projects into the same fit (they must share the marker/measurement names); *Clear* resets.
3. **Batch grouping** (optional) — *Assign batches…* opens an Image → Batch table. Assign by double-clicking a cell, selecting rows → *Assign selected → batch…*, **Auto-detect from name**, or **Load CSV…**. The grouping drives per-batch mode and the QC view.
4. **Granularity** —
   - **Per image** — each image is aligned to the reference independently (finest correction).
   - **Per batch** — images pooled within a batch are aligned together (uses the grouping above).
5. **Advanced** — **Bins** (log-histogram resolution, default 1024), **Cells/image** (subsample cap for the fit, default 50,000), **Workers** (images processed in parallel).
6. **Run fit** — computes the per-image/per-batch gains and saves them to `<project>/celltune/batch-shifts.json`. The fit persists across sessions and can be re-run any time.

### 19.3 QC — did it work?

**Show QC** opens *Batch Normalisation — QC*: per-marker log-intensity density curves (one per batch) with a spread (SD) readout. **Lower SD = better aligned** — the curves should overlap after correction. Scan a few markers to confirm the batches were pulled together without collapsing genuine structure.

### 19.4 How it's applied

Two independent options:

- **Streamed (recommended)** — tick **"Use batch-corrected values in clustering + ML"**. Clustering *and* classifier training/inference then multiply each cell's measurements by that image's fitted gain **in memory** before use: no columns are written, nothing on the cells changes, and the correction is applied consistently at every seam (clustering, training, and single-image auto-classify / batch-apply to other images). It is a persistent project preference (`celltune.useBatchCorrection`), so it stays on until you untick it, and is a no-op when no fit exists.
- **Written columns** — **Write corrected columns** materialises the corrected values as new `…(batchnorm)` measurement columns (for export or inspection). The raw columns are left intact.

### 19.5 Tips & cautions

- **Fit before you cluster or train** — the streamed toggle only does anything once a fit exists in the project.
- **Grouping matters for per-batch mode** — with everything in one batch, per-batch mode is just a single-reference alignment; *Auto-detect from name* bootstraps groups from filename conventions.
- **It can over-correct** — treating each slide as a batch can erase real biology if a cohort genuinely differs by group. QC each marker, and don't batch-correct across groups you expect to differ (mirrors the caution in §[18.8](neighborhoods.md#188-tips--cautions)).
- **Cross-project fits** require a shared marker panel — measurement names must match across the pooled projects.

---
