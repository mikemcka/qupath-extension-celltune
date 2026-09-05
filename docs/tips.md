# Tips, tricks and known limitations

- **Label at least 20–30 cells per class** before the first Train, then trust the disagreement-driven Review Mode to grow your label set efficiently.
- **Selecting cells** Hold Ctrl key on windows/linux or Command on Mac to select multiple cells at the same time to label.
- **Channel Viewer** View > Channel viewer is very useful, it will display a view of the target area which all channels selected in channel select displayed.
- **Resolve hierarchy before exporting** We also have noticed a bug where parent annotations are missing for exported cells, ContainingAnnotations will display them correctly.
- **F1 scores can lie.** A held-out 20% split is honest within an image but optimistic across the project. Always sanity-check on a few unseen slides before believing the metrics.
- **Pick different model types for Model 1 and Model 2.** Two XGBoosts won't disagree much, which kills the whole point.
- **Images at once caps at 8.** Expect ~2–4 GB of memory per image on COMET data. It's a different thing from **CPU threads** — see §5.3.
- **If training feels slow, read the log first.** Every run ends with a "Where the time went" table in `<project>/celltune/logs/`. It tells you which step to actually do something about instead of guessing.
- **The marker table persists per project.** On import it's saved to `<project>/celltune/marker-table.json` and reloaded automatically when you reopen the project, so auto-channel-switching during review survives QuPath restarts — no re-import needed. (*Reset Project State* clears it along with the rest of the `celltune/` folder.)
- **Project Prediction Summary needs ≥5 images** to give meaningful robust z-scores. On 2–3 image projects, treat the Anomaly column as overview or guide.
- **Composite class colours.** Without "Prepend primary", QuPath generates a colour per unique composite name — you can end up with hundreds. Tick "Prepend primary" and your existing multi-class palette is preserved.
- **No `.qpdata` save** when navigating from Project Prediction Summary — this is deliberate (saving large slides is slow and pointless for navigation). Manually save the image after editing it.

---

*Spot a missing step or an inaccurate label? Open an issue on the GitHub repo. Screenshots welcome.*
