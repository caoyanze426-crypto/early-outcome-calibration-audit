# ENVIRONMENT

Values below are transcribed from the frozen records; nothing is inferred.

## Runtime recorded for the frozen analysis

    Python          3.14.3 (tags/v3.14.3:323c59a, Feb  3 2026, 16:04:56) [MSC v.1944 64 bit (AMD64)]
    Upstream        earlyeval @ 7fd1a8e5b755e1f7ab642bcae77473b53e2bf1d0
                    (MIT; clone not modified; upstream requirements list:
                    numpy, pandas, pyarrow, scipy, scikit-learn, lightgbm,
                    joblib, pyyaml, tqdm, tiktoken, transformers, torch,
                    sentencepiece)

## Runtime recorded for the submission artwork

    Python          3.14.3
    matplotlib      3.11.2
    numpy / pandas  as bundled with that runtime
    output formats  vector PDF + SVG (svg.fonttype=none, pdf.fonttype=42) and
                    300-dpi PNG previews

## What is reproduced by what

The scientific computation itself (feature construction, LightGBM training,
bootstraps, threshold sweeps) was executed inside the frozen phases; the
aggregate result files of those phases are shipped under `derived_data/`, and
the code that produced them is shipped under `code/`. The artwork in
`figures/` and `tables/` was rendered from the plot-ready source data of the
same frozen phases (`derived_data/data/`), with the render audits in
`derived_data/audits/`.

The frozen analysis configuration (predictor id, policy constants, gap
convention, bootstrap seeds and units, eligibility rules) is recorded verbatim
in `derived_data/REPRODUCIBILITY.md` and in the per-phase protocols under
`protocols/`.
