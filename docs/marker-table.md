# Marker table format

A **marker table** maps each cell type to the marker channels that identify it. It drives
[auto channel-switching during review](review-mode.md) — when you land on a predicted `CD4T`
cell, the viewer can show the `CD4`/`CD3` channels automatically.

Import one via **Extensions ▸ SP Classify ▸ Import Marker Table**.

## Simple format

A CSV with up to 5 marker columns. Trailing columns may be left blank. A ready-to-edit example
lives at
[`examples/marker-table-example.csv`](https://github.com/mikemcka/qupath-extension-sp-classify/blob/main/examples/marker-table-example.csv).

```csv
CellType,Marker1,Marker2,Marker3,Marker4,Marker5
CD4T,CD4,CD3,,,
CD8T,CD8,CD3,,,
Treg,CD4,CD25,FOXP3,CD3,
Bcell,CD20,CD19,,,
Macrophage,CD68,CD163,CD11b,,
```

## How names are matched

!!! tip "Matching is tolerant — but the `CellType` column should track your class names"
    The `CellType` column should match the class names you assign to labelled cells. Matching is
    **case-, spacing-, and punctuation-insensitive**, so `CD4 T`, `cd4t`, and `CD4-T` are treated
    as the same type. `Marker` names are matched to image channels the same way, so a channel
    named `CD3 (Opal 570)` still matches the marker `CD3`.

    If a predicted type isn't found in the table, or none of its markers match any channel, the
    viewer's channels are **left unchanged** (nothing is hidden).

## Rule format (gating)

The importer also accepts a **rule format** for composite gating, auto-detected from a
`PrimaryMarker` column. See **[Binary + composite workflow](binary-composite.md)** for how gating
rules are written and applied.

```csv
CellType,PrimaryMarker,SecondaryMarker,TertiaryMarker
CD8T,CD8&CD3,CD45,CD103|CD45RA
Macrophage,CD68|CD163|CD206,,CD14|CD38|VIM
```
