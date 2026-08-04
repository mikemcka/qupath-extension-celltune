# CellTune Classifier for QuPath

[![DOI](https://zenodo.org/badge/1214148430.svg)](https://doi.org/10.5281/zenodo.21782421)

## This java extension provides similar functionality as parts of CellTune by the Keren Lab into QuPath. If you use this tool for your analysis please also cite [the CellTune preprint](https://www.biorxiv.org/content/10.1101/2025.05.05.652215v1).

A [QuPath](https://qupath.github.io/) 0.7 extension that brings **active learning** to cell classification. It trains two gradient-boosted models (XGBoost + LightGBM) simultaneously, identifies cells where the models disagree, and presents those disputed cells for human review — creating an iterative loop that progressively improves classification accuracy.


## Features

- **Dual-model disagreement detection** — two *different* models (default **XGBoost + LightGBM**; **Random Forest** also available) train on the same labels, and the cells where they disagree are flagged for review. Boosted models use GPU (CUDA) when available, else CPU.
- **Weighted uncertainty sampling** — disagreement cells are drawn with a 5-tier strategy (FOV balance → most-confused classes → rare types → user-specified confusions → random fill) so review effort lands where it matters instead of on easy or already-covered cells.
- **Interactive review mode** — step cell-by-cell through the sampled disagreements and accept a model's prediction (or override it) with one click, with per-cell spatial context.

See **[User Guide §7 — Review mode](USER_GUIDE.md#7-after-training--review-mode)** for the full tier breakdown and review toolbar, and **[§5](USER_GUIDE.md#5-multi-class-workflow-in-detail)** for training settings.


## The Active Learning Loop

```
① Seed labels via Manual label mode
                    │
                    ▼
② Train dual classifiers (XGBoost + LightGBM)
                    │
                    ▼
③ View inter-model confusion matrix
   → per-class agreement rates
                    │
                    ▼
④ Enter review mode → choose sample size
   → step through disagreement cells
   → assign or correct labels
                    │
                    ▼
   ⟳ Merge new labels → retrain → repeat
     until satisfied with classification quality
```

## Requirements

| Component | Version |
|-----------|---------|
| QuPath | 0.7.0 or later |
| Java | 25 (required by QuPath 0.7) |

The extension JAR bundles XGBoost4J, LightGBM4J, and a pure-Java Random Forest — no separate ML library installation is needed.

## Installation

1. Download the latest `qupath-extension-celltune-*-all.jar` from the [Releases](../../releases) page (or build from source — see below)
2. **If upgrading, delete any older `qupath-extension-celltune-*-all.jar` from the extensions directory first.** QuPath loads every JAR it finds there, so leaving an old version behind means the stale extension may load instead of the new one (you'll see old behaviour even after "updating"). Only one such JAR should be present.
3. Drag and drop the JAR onto the QuPath window, or copy it to your QuPath extensions directory:
   - **Windows:** `C:\Users\<you>\QuPath\v0.7\extensions\`
   - **Linux:** `~/.local/share/QuPath/v0.7/extensions/`
   - **macOS:** `~/Library/Application Support/QuPath/v0.7/extensions/`

   (In QuPath, *Extensions > Installed extensions* opens this folder directly.)
4. Restart QuPath — the extension appears under **Extensions > CellTune Classifier**

## Quick Start

1. **Open a project** with an image that has cell detections (run *Analyze > Cell detection* first if needed)
2. **Label seed cells** — create point annotations on detected cells, then set the annotation's class (e.g. `CD4T`, `Bcell`, `Macrophage`). Aim for ≥20-30 cells per class.
3. **Import a marker table** (optional) — *Extensions > CellTune Classifier > Import Marker Table* — a CSV mapping cell types to up to 3 marker channel names, used for auto-channel switching during review
4. **Select features** (optional) — *Extensions > CellTune Classifier > Select Features* — choose which measurements to include in training. Features are shown in a grouped, searchable checkbox tree (one group per marker, plus Morphology / Shape, Neighbors, Embeddings, and Other / Uncategorized) so 1000+-column panels stay navigable.
5. **Clustering normalisation** (optional, clustering-only) — *Extensions > CellTune Classifier > Clustering Normalisation* — apply arcsinh or sqrt transforms to selected features for the clustering / scatter-plot / gating workflows (the classifier always trains and predicts on raw values). Match the cofactor to your intensity scale: ~25–50 for raw fluorescence panels (COMET, CODEX; scale-dependent) or 0.05 for MIBI mass spectrometry (Hartmann et al. 2021; see [References](#references)).
6. **Train** — click *Train* in the CellTune Classifier panel (or *Extensions > CellTune Classifier > Run CellTune Classification…*). If features haven't been selected yet, you'll be prompted to select them or use all. A confirmation dialog shows the feature and label counts, **Model 1** and **Model 2** type selectors (default: XGBoost + LightGBM), resampling, auto-tune, and early stopping options. If the project has multiple images, a dual-list image selector lets you choose which images to apply the trained classifier to. A progress dialog shows real-time training status including GPU/CPU device info.
7. **Plot confusions** — click *Plot Confusions* to see the inter-model confusion matrix with per-class agreement rates and F1 scores
8. **Feature importance** (optional) — *Extensions > CellTune Classifier > Feature Importance…* — opens a bar chart of the top 10 features by mean |SHAP| value with a class selector. Alternatively tick **"Show top 10 feature importance after training"** in the training dialog to show it automatically.
9. **Review** — click *Enter Review Mode*. You'll be prompted for how many disagreement cells to review (default 200), then step through disputed cells one-by-one. Each cell shows coloured prediction buttons (e.g. `XGB: CD4 (87%)`, `LGB: Bcell (65%)`) — click to accept. Use the *All Classes* dropdown if neither prediction is correct. If you switch to a different image, the trained classifier is automatically applied so you can review that image immediately.
10. **Retrain** — after reviewing, click Train again. The confusion matrix should improve. Repeat until satisfied.
11. **Export** — *Export Cell Table* opens a column picker (same search/prefix/select-all controls as *Select Features*) so you can choose which measurement columns to export, with an optional tick-box to include cell polygons in micron or pixel coordinates, then saves all cells as CSV. *Export AnnData* exports AnnData-compatible CSV with a Python H5AD conversion script. *Export Ground Truth* saves labelled cells for transfer to other images.

### Marker Table Format For Automated Channel Switching

A simple CSV with up to 5 marker columns. Trailing columns may be left blank.

```csv
CellType,Marker1,Marker2,Marker3,Marker4,Marker5
CD4T,CD4,CD3,,,
Bcell,CD20,,,,
Macrophage,CD68,CD163,CD11b,,
Treg,CD4,CD25,FOXP3,CD3,
```

## Building from Source

See [CLAUDE.md](CLAUDE.md#build--test) for prerequisites (JDK 25), platform-specific commands, and install steps. In short:

```bash
export JAVA_HOME=/path/to/jdk-25
./gradlew shadowJar
# → build/libs/qupath-extension-celltune-0.2.4-all.jar
```

## Project Structure

Source lives under `src/main/java/qupath/ext/celltune/`, organised into `model/`, `classifier/`, `gating/`, `ui/`, and `io/` packages, with `CellTuneExtension.java` as the entry point. See [CLAUDE.md](CLAUDE.md#architecture) for the package-by-package architecture and key classes.


## Technology Stack

| Layer | Technology |
|-------|-----------|
| Build system | Gradle 9.2.1 (Kotlin DSL) + QuPath conventions plugin |
| Extension host | QuPath 0.7 |
| ML model 1 | XGBoost4J 2.1.4 (Scala 2.13) |
| ML model 2 | LightGBM4J 4.6.0-2 |
| ML model 3 | Random Forest (pure Java, no external dependency) |
| UI framework | JavaFX (bundled with QuPath) |
| Serialisation | JSON (Gson, bundled with QuPath) |

## License

Licensed under the **[GNU General Public License v3.0 only](LICENSE)** (GPL-3.0-only).

Copyright (C) 2026 mikemcka.

> This extension bundles third-party libraries whose own licenses constrain how it may be
> distributed — most notably **QuPath** (GPL-3.0) and **Smile** (dual-licensed **GPL-3.0 /
> commercial**). Using Smile under its open-source arm makes the distributed combined work
> GPL-3.0, so GPL-3.0 is the strictest license this project is bound by. See the bundled-library
> licenses under [Acknowledgements](#acknowledgements).

## Acknowledgements

### References
- **[CellTune](https://celltune.org/)** by the [Keren Lab](https://www.weizmann.ac.il/mcb/Keren/home) —  The human in the loop cell classification workflow that this extension derives functions from. See [the CellTune preprint](https://www.biorxiv.org/content/10.1101/2025.05.05.652215v1).
- **Cellular neighborhoods** — the CN analysis (§18 of the [User Guide](USER_GUIDE.md)) implements the neighbourhood-clustering method of **Schürch CM, Bhate SS, Barlow GL, et al. "Coordinated Cellular Neighborhoods Orchestrate Antitumoral Immunity at the Colorectal Cancer Invasive Front." *Cell* 182(5):1341–1359.e19 (2020). [doi:10.1016/j.cell.2020.07.005](https://doi.org/10.1016/j.cell.2020.07.005)**. If you use the cellular neighborhoods feature in your analysis, please cite this paper. Reference implementation: [nolanlab/NeighborhoodCoordination](https://github.com/nolanlab/NeighborhoodCoordination) (the authors' Python/Jupyter code) — the extension reimplements the same kNN-window + k-means approach in Java.
- **Leiden clustering & the scanpy recipe** — the graph-based clustering (§11 of the [User Guide](USER_GUIDE.md)) uses the **Leiden algorithm**: **Traag VA, Waltman L, van Eck NJ. "From Louvain to Leiden: guaranteeing well-connected communities." *Scientific Reports* 9:5233 (2019). [doi:10.1038/s41598-019-41695-z](https://doi.org/10.1038/s41598-019-41695-z)** (implemented here via the CWTS `networkanalysis` library — same authors as the Python `leidenalg`). The pipeline deliberately mirrors **scanpy**'s single-cell recipe (scale → PCA → neighbours → Leiden): **Wolf FA, Angerer P, Theis FJ. "SCANPY: large-scale single-cell gene expression data analysis." *Genome Biology* 19:15 (2018). [doi:10.1186/s13059-017-1382-0](https://doi.org/10.1186/s13059-017-1382-0)**. The extension's Leiden clustering was validated against scanpy's Leiden on the Schürch et al. CODEX CRC dataset. If graph-based clustering is central to your analysis, please cite the Leiden algorithm and scanpy.
- **Approximate nearest neighbours (HNSW)** — the scalable kNN graph behind cohort/all-cells Leiden uses **Hierarchical Navigable Small World graphs**: **Malkov YA, Yashunin DA. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs." *IEEE TPAMI* 42(4):824–836 (2020). [doi:10.1109/TPAMI.2018.2889473](https://doi.org/10.1109/TPAMI.2018.2889473)** (implemented here via the jelmerk `hnswlib-core` library).
- **arcsinh feature normalisation (MIBI)** — the recommended arcsinh cofactor of **0.05** for MIBI mass-spectrometry mean intensities (§4.2 of the [User Guide](USER_GUIDE.md)) follows **Hartmann FJ, Mrdjen D, McCaffrey E, et al. "Single-cell metabolic profiling of human cytotoxic T cells." *Nature Biotechnology* 39:186–197 (2021). [doi:10.1038/s41587-020-0651-8](https://doi.org/10.1038/s41587-020-0651-8)** (bioRxiv preprint: "Multiplexed single-cell metabolic profiles organize the spectrum of cytotoxic human T cells"), as used in the [squidpy MIBI-TOF tutorial](https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_mibitof.html).
- **[pixel-patrol](https://pypi.org/project/pixel-patrol/)** (MIT) — the whole-image, per-channel pixel-statistics approach behind the **image pixel prescreen** (which summary statistics to compute and how images are flagged) was adapted from pixel-patrol. The Java implementation (`model/ImagePixelStats`, `model/ImagePixelStatsReader`, `model/PixelCohortAnalyzer`) is original to this project.
- **[qupath-extension-xgboost](https://github.com/zindy/qupath-extension-xgboost)** by [Zindy](https://github.com/zindy) — a QuPath 0.7 XGBoost extension whose project structure, Gradle configuration, and XGBoost4J integration patterns served as a reference implementation for this extension.
- Built on the [QuPath extension template](https://github.com/qupath/qupath-extension-template).

### Bundled libraries
The shadow ("fat") JAR bundles the following third-party libraries; their licenses apply to the distributed JAR:

| Library | License | Used for |
|---------|---------|----------|
| [QuPath](https://qupath.github.io/) | GPL-3.0 | Host platform the extension runs in |
| [Smile](https://github.com/haifengl/smile) (`smile-core`) | Dual: **GPL-3.0 / commercial** (© Haifeng Li / SMILE.AI, LLC) | PCA, UMAP, k-means for the cell scatter plot & clustering |
| [XGBoost4J](https://github.com/dmlc/xgboost) | Apache-2.0 | Gradient-boosted model 1 |
| [LightGBM4J](https://github.com/metarank/lightgbm4j) wrapping [LightGBM](https://github.com/microsoft/LightGBM) | MIT / MIT | Gradient-boosted model 2 |
| [Bytedeco JavaCPP presets](https://github.com/bytedeco/javacpp-presets) → native [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS), [ARPACK-NG](https://github.com/opencollab/arpack-ng) | Apache-2.0 / BSD-3-Clause / BSD-3-Clause | Native BLAS/LAPACK for Smile's PCA/UMAP (pulled in transitively by Smile) |
| [CWTS networkanalysis](https://github.com/CWTSLeiden/networkanalysis) | MIT | Leiden community detection (CPM) for graph-based cell clustering (§11) |
| [hnswlib-core](https://github.com/jelmerk/hnswlib) (jelmerk) | Apache-2.0 | HNSW approximate-nearest-neighbour kNN graph for Leiden at cohort/all-cells scale |
