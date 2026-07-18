# Installation

[← Back to index](index.md)

## 1. Conda environment

A conda environment is highly recommended:

```bash
conda create -n MUSE_PN "python==3.10.13" numpy matplotlib pandas scipy astropy \
    h5py joblib catboost ipykernel SQLAlchemy pymysql jupyterlab tensorflow
conda activate MUSE_PN
```

## 2. PyNeb

The pipeline is built on [PyNeb](https://github.com/Morisset/PyNeb_devel)
(nebular emission-line analysis: atoms, emissivities, diagnostics, observations):

```bash
pip install -U PyNeb
```

To use the development version of PyNeb instead:

```bash
pip install -U git+https://github.com/Morisset/PyNeb_devel.git
```

## 3. ai4neb (ANN emulators)

[ai4neb](https://github.com/VGomezLlanos/AI4neb) provides the machine-learning
regression models used to accelerate the Te/Ne diagnostics (and the CatBoost ω
prediction):

```bash
pip install -U git+https://github.com/VGomezLlanos/AI4neb.git
```

Without ai4neb the code still imports (`diagnostics.py` detects its absence), but
`get_TeNe` should then be called with `use_ANN=False`, which makes the pixel-by-pixel
`getCrossTemDen` inversion much slower.

## 4. Data and paths

The pipeline needs two data sets, whose locations are **hardcoded** in
[`module/constants/observation_parameters.py`](configuration.md) and must be edited
first:

| Parameter | Content |
|---|---|
| `DATA_DIR` | Directory with the per-line emission maps, one FITS file per line named `NGC6153_MUSE_b_<label>.fits` (plus `*error*` files) |
| `OBS_INT_FILE` | ASCII file with the line fluxes of the integrated spectrum (PyNeb format `lines_in_rows_err_cols`) |
| `FITS_DIR` | Directory with the Gaussian-fit velocity maps (only used for kinematics figures) |

Also check `OBJ_NAME`/`OBS_NAME` (file naming pattern) and the rest of the
[configuration reference](configuration.md).

## 5. Trained models and caches

Some steps rely on files produced by previous steps and shipped/created in `module/`:

- `new_ai4neb/`, `ai4neb/` — trained ANN emulators of the diagnostics (created on
  first run, reused afterwards);
- `w_pred_CatBoost.joblib`, `w_pred_convolved.joblib` — ω weight map predicted by the
  ML model of the notebook;
- `omega_mask.joblib`, `omega_1_mask.joblib` — masked ω and 1−ω maps created by
  `figuras_articulo.fig12_smooth_omega`, required by fig13/fig14/fig15 and table 5;
- `constants/wcs_NGC6153.joblib` — WCS of the MUSE field (loaded at import).

## 6. Run

```bash
cd module
python -c "import figuras_articulo as fa; fa.create_figures(); fa.create_tables()"
```

or work interactively — see the [Quick start](index.md#quick-start) and the
[notebooks](notebooks.md).
