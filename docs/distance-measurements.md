# Distance measurements (spatial analysis)

**Menu:** *Extensions → SP Classify → Generate Distance Measurements...*

A project-wide batch tool that adds spatial distance columns to your cell measurements — useful for downstream neighbourhood / spatial-statistics analysis. It runs across as many project images as you select, loading and saving each one for you.

It can generate three independent measurement families (tick any combination):

| Computation | What it writes per cell | Backed by |
|---|---|---|
| **Detection-to-annotation signed distances** | `Signed distance to annotation <class> <unit>` — negative inside the annotation, positive outside. | QuPath `DistanceTools.detectionToAnnotationDistancesSigned` |
| **Cross-class centroid distances** | `Distance to detection <class> <unit>` — nearest centroid-to-centroid distance to a cell of every *other* class. | QuPath `DistanceTools.detectionCentroidDistances` |
| **Same-class nearest-neighbour distances (excludes self)** | `Distance to other <class> <unit>` — distance to the nearest *other* cell of the **same** class. | The extension (spatially indexed; see below) |

`<unit>` is `µm` when a pixel size is available (from calibration or the override below), otherwise `px`.

### Dialog options

- **Images** — checklist of every project image, with **All** / **None** / **Current only** buttons. All are ticked by default.
- **Pixel size (µm/pixel)** — optional. Pre-filled from the current image's calibration when available.
  - Leave **blank** to use each image's own existing calibration.
  - Enter a value to override calibration for *every* selected image so results come out in microns.
  - **Persist this pixel size to each image's calibration on save** — when ticked, the override is written into each image's calibration metadata (so future measurements also use this scale). When unticked, the override is reverted after the run.
- **Skip images where all selected measurements already exist** (default on) — before computing, the extension scans every cell. If all cells already carry every measurement the selected computations would produce, the image is skipped entirely (no recompute, no re-save). This makes interrupted runs cheap to resume. It is **all-or-nothing per image**: if even one selected measurement is missing, the whole image is recomputed, guaranteeing internally consistent results. Untick to force recomputation (e.g. after changing classes).
- **Parallel image workers** (1–N cores) — how many images are processed at the same time.
  - The heavy distance maths for a *single* image already spreads across all CPU cores, so raising this mostly overlaps disk load/save (I/O) with compute.
  - **Many small images:** higher worker counts can speed up the batch.
  - **A few very large images (hundreds of thousands of cells):** 1–2 workers is often fastest — each image then gets the full CPU and uses less memory.

### Running it

Click **Apply**. The log area streams per-image progress, e.g.:

```
Starting on 41 image(s)…
Using 1 parallel image worker(s) (cores=14).
[slide1.ome.tif] Loading…
[slide1.ome.tif] Skipped — all selected measurements already present.
[slide2.ome.tif] Same-class nearest-neighbour distances…
[slide2.ome.tif]   Tumour: 82770 cells in 18830 ms → Distance to other Tumour µm
[slide2.ome.tif] Saved.
```

Classes with only a single cell are reported as `Skipping '<class>' (n=1)` for the same-class computation (a lone cell has no same-class neighbour). Each processed image is saved back to the project automatically. **Close** dismisses the dialog.

> **Performance note.** For large numbers of small images (10-20K cells) use a higher number of workers, for large images (500k+ cells) use one or 2 workers.

---
