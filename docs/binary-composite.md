# Binary + composite workflow in detail

Use this when you want **per-marker** classifiers (one for CD3 positive/negative, another for CD8 positive/negative, etc.) and then combine them into composite cell types. Great for smaller panels or functional markers like Ki67.

### 6.1 Create a binary classifier

**Menu:** *Extensions → SP Classify → Binary Classifiers...*

- **Create...** → enter a marker name (e.g. `CD3`). Marker names are sanitised to safe filesystem characters.
- The marker is registered in `<project>/celltune/binary-registry.json` and a state file `<project>/celltune/binary/CD3.json` is created when you first train.

Select the marker in the list and click **Open**. The dialog closes; the main sidebar switches into **Binary Mode** with a blue banner: `Active binary mode: CD3`.

In binary mode:
- The class buttons in Manual Label Mode are restricted to `CD3_pos` and `CD3_neg` (so you can't accidentally label across markers).
- **Pool labels from all images** is auto-enabled and locked — each marker classifier always trains on its full pooled label set.
- Settings, sampling, review, and metrics work exactly the same as multi-class.

Train, review, and iterate until you're happy. Then click **Exit Binary Mode** to return to multi-class.

Repeat for every marker you want in the composite.

### 6.2 Composite classification

**Menu:** *Extensions → SP Classify → Composite Classification...*

- **Markers** — checkbox per trained binary classifier. **All** / **None** buttons above. Only markers that have been trained (have a saved XGBoost model) appear.
- **Images** — checkbox per project image. **All** / **None** / **Current only** buttons above.
- **Prepend current primary classification (colour follows primary)** — see below.
- **Apply** — runs the classifiers.

**How it works:**
- The currently open image is classified **in-memory** — viewer updates immediately, no save/reload.
- Every other selected image is read from disk, classified, and written back (logged in the panel's text area).
- Each cell gets a composite `PathClass` named by joining the marker results alphabetically:
  - `CD3+:CD8-:CD45+` etc.
  - `+` if the binary classifier's positive probability ≥ 0.5, else `-`.

**Prepend current primary classification:**
- When **off** (default): composite name is markers only (`CD3+:CD8-`), QuPath auto-assigns the class colour.
- When **on**: each cell's current primary `PathClass` is captured **before** any reassignment and prepended (`Tumour:CD3+:CD8-`). The composite class's colour is set to the primary class's colour, so the viewer keeps your existing multi-class colouring. Cells with no current primary fall back to binary-only naming.

> Use the merge mode after you've already run a multi-class classifier — the multi-class result becomes the cell-type "backbone" and the binary classifiers add functional state.

---
