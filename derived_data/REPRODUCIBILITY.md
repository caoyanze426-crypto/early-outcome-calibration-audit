> **Release note (PAPER2_W7B).** In this repository these derived files
> live under `derived_data/` (the pre-finalization package used the
> directory name `repository_candidate/`). File contents are unchanged
> except for the single corrected literal documented in
> `derived_data/PROVENANCE.md`, and the references to the author-decision
> files of the pre-finalization workspace, which this release replaces
> with the release level `README.md` / `LICENSE` / `LICENSE-DATA`.

# REPRODUCIBILITY

> **Release note (PAPER2_W7B).** This is the pre-finalization
> reproducibility note, retained verbatim; the release-level companion
> documents are `reproducibility/ENVIRONMENT.md`,
> `reproducibility/DATASET_REVISIONS.md`,
> `reproducibility/RUN_INSTRUCTIONS.md` and
> `reproducibility/FROZEN_LINEAGE.md`.

## Environment used for the rendering in this submission

    Python          3.14.3
    matplotlib      3.11.2
    numpy / pandas  as bundled with the above runtime
    plots written   vector PDF + SVG (svg.fonttype=none, pdf.fonttype=42) and
                    300-dpi PNG previews

The scientific computation itself (feature construction, LightGBM training,
bootstraps) was executed inside the frozen phases; this package contains the
result files, not the training environment.

## Frozen analysis configuration (for reading the data files)

    predictor id      P0B_ReferenceFree_LightGBM_Dense_AF_FoldLocalFE
    policy            current_safe_stop, dual, success_thr = failure_thr = 0.95,
                      min_step = 0, consecutive = 1
    gap convention    signed calibration gap = mean calibrated decision score
                      at the stop point minus empirical precision of the
                      decided rows (predicted minus realized)
    bootstrap units   task-cluster bootstrap over instance_id (Phase 1A: 2000
                      valid replicates, seed 4201; Phase 1A-D: 5000 replicates,
                      seed 42031; Phase 1B: 2000 replicates per threshold, seeds
                      42100/42110/42120/42130 and 42101/42111/42121/42131)
    eligibility       decision minimum 20 per cell/fold; persistence criteria as
                      recorded in phase1a_target_persistence.csv and
                      phase1ad_primary_gate.json

## Regenerating the main figures from this package

    Figure 1  study-design schematic; no numeric data (conceptual)
    Figure 2  data/figure2_plot_data.csv          (raw + corrected gaps per cell)
    Figure 3  data/figure3_plot_data.csv,
              data/figure3_annotation_values.json (occurrences + overlays)
    Figure 4  data/figure4_threshold_data.csv,
              data/figure4_cross_benchmark_data.csv

    Tables 1-3 use data/table1_design_data.csv, table2_primary_targets.csv,
    table3_terminal_boundary.csv (note: the frozen table3 file carries a
    six-field section-B row layout against a five-name header; the trailing
    status field is the sixth column).

Every numeric value rendered in the submission figures was re-verified against
these files by machine audit (`FIGURE_RENDER_AUDIT.csv`: 105 numeric values, 0
mismatches) and the same procedure was applied to the supplement figures
(`APPENDIX_VISUAL_AUDIT.csv`).

## Retrieval of the source corpora

    SWE corpus   tarsur385/swebench-verified-trajectories
                 revision 773748a7c1222e8a642a7059821498e14293562a
    TB corpus    yoonholee/terminalbench-trajectories
                 revision 04e8940f5b6736a7ce8d22224fe2f2af74163ed2 (apache-2.0)

The frozen study consumed exactly these revisions; no other snapshot was used.
