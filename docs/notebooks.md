# Notebooks

[← Back to index](index.md)

Two Jupyter notebooks live in `module/` and complement the Python scripts. They are
run with the same `MUSE_PN` environment (see [Installation](installation.md)).

## `ICF_ABUND_NGC6153.ipynb`

The main analysis notebook (≈77 cells). It sits *after* the diagnostics and ionic
abundances in the [pipeline](overview.md) and computes:

- the **ω weight map** — the fraction of Hβ emitted by the cold, H-poor component —
  predicted by machine-learning models (CatBoost / ai4neb ANN emulators trained on
  photoionization-model grids). Serialized outputs: `w_pred_CatBoost.joblib`,
  `w_pred_convolved.joblib`;
- **Ionization Correction Factors (ICFs)** for unseen ionization stages, also with
  machine-learning techniques;
- the **total elemental abundances** of the warm and cold components, and the
  abundance discrepancy analysis of the paper.

Its ω outputs are consumed by `figuras_articulo.py` (fig12–fig15, table 5) through
the `omega_mask.joblib` / `omega_1_mask.joblib` files (created by
`fig12_smooth_omega`).

## `Results.ipynb`

A lighter notebook (≈22 cells) used to explore and display the results: loading the
computed maps, quick-look figures and sanity checks. It does not produce files
required by the rest of the pipeline.
