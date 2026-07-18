# Configuration reference

[← Back to index](index.md)

All pipeline settings live in `module/constants/observation_parameters.py`; every
module imports from it. This page documents each parameter group.

## Data paths (edit these first)

| Parameter | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `/home/vero/...` | Directory with the per-line IFU FITS maps. **Hardcoded absolute path — must be adapted.** |
| `OBJ_NAME` | `NGC6153` | Object name, used to build the file pattern |
| `OBS_NAME` | `DATA_DIR/NGC6153_MUSE_b_*.fits` | Glob pattern matching the line maps |
| `OBS_INT_FILE` | `/home/vero/...` | ASCII file with the integrated-spectrum fluxes. **Must be adapted.** |
| `FITS_DIR` | `/home/vgomez/...` | Gaussian-fit velocity maps (kinematics only). **Must be adapted.** |

## Reading options (passed to `pn.Observation`)

| Parameter | Default | Meaning |
|---|---|---|
| `FILE_FORMAT` | `fits_IFU` | PyNeb file format of the IFU maps |
| `FILE_FORMAT_INT` | `lines_in_rows_err_cols` | Format of the integrated-flux file |
| `CORRECTED` | `False` | Data are not already extinction-corrected |
| `ERROR_STR` | `error` | Tag identifying the FITS files holding the errors |
| `ERR_IS_RELATIVE` | `False` | Whether errors are relative or absolute |
| `ERROR_DEFAULT` | `0.05` | Default error assigned when none is available |
| `ADD_ERR_DEFAULT` | `True` | Add the default error quadratically |
| `CUTOUT2D_POSITION`, `CUTOUT2D_SIZE` | `None` | Optional 2D cutout (center pixel and size) of the maps |

## Fluxes, cleaning, masking

| Parameter | Default | Meaning |
|---|---|---|
| `FLUX_NORM` | `1e-20` | FITS flux unit; maps are multiplied by it |
| `CLEAN_ERROR` | `1e-5` | Pixels whose error equals `ERROR_DEFAULT` within this tolerance are flagged NaN (no real measurement) |
| `CUT_HB` | `0.005` | Spaxels with Hβ < `CUT_HB` × max(Hβ) are masked out |

## Monte Carlo error propagation

| Parameter | Default | Meaning |
|---|---|---|
| `N_MC` | `None` | Number of MC realizations of the IFU maps (`None` disables) |
| `N_MC_INT` | `500` | MC realizations of the integrated spectrum |
| `RANDOM_SEED` | `42` | Seed for the MC noise and the ANN training |

## Extinction correction

| Parameter | Default | Meaning |
|---|---|---|
| `CORREC_LAW` | `F99` | Extinction law (Fitzpatrick 1999) |
| `R_V` | `3.1` | Total-to-selective extinction ratio |
| `TEM_CHB` | `8000` | Te [K] for the theoretical H I ratios (based on Te[S III]) |
| `DEN_CHB` | `3400` | Ne [cm⁻³] for the theoretical H I ratios (based on Ne[Cl III]) |
| `EBV_MIN` | `0.` | E(B−V) values below this are reset to 0 |
| `EXTINCTION_LABEL` | Hα + 4 Paschen lines | H I lines ratioed to Hβ; the median E(B−V) of the ratios is adopted |

## Recombination-contamination correction

| Parameter | Default | Meaning |
|---|---|---|
| `TE_CORR` | `6000` | Te [K] assumed for the cold recombining gas |
| `DEN_CORR` | `1e4` | Ne [cm⁻³] assumed for the cold recombining gas |
| `REC_LABEL` | `O2r_4649.13A` | O II RL used as reference for the [O II] correction; may also be `O2r_4661.63A`. Using 4649.13 means the 4649.13+4650.84 blend of multiplet V1 is considered |

## Te/Ne diagnostics

| Parameter | Default | Meaning |
|---|---|---|
| `DIAGNOSTICS_DICT` | 9 pairs | Key → (Te diagnostic, Ne diagnostic) solved simultaneously; e.g. `'N2S2': ('[NII] 5755/6548', '[SII] 6731/6716')` |
| `TENE_HIGH` | `S3Cl3` | Default diagnostic for the high-ionization zone |
| `IP_CUT` | `17` | Ionization potential [eV] separating the low/high-ionization zones |
| `USE_ANN` | `True` | Use ai4neb ANN emulators in `getCrossTemDen` |
| `ANN_FOLDER` | `./ai4neb/` | Cache directory of the trained ANNs |
| `ANN_INST_KWARGS`, `ANN_INIT_KWARGS` | — | Instantiation/initialization settings of the scikit-learn ANN (architecture (10, 20, 10), tanh, lbfgs) |

## Paschen-jump temperature

| Parameter | Default | Meaning |
|---|---|---|
| `DEN_PJ` | `1e4` | Density assumed for the theoretical PJ(Te) curve |
| `Hep_PJ` | `0.12` | He⁺/H⁺ used for the He continuum contribution |
| `Hepp_PJ` | `0.004` | He⁺⁺/H⁺ used for the He continuum contribution |

## Plotting

| Parameter | Default | Meaning |
|---|---|---|
| `PLOT_DIAGNOSTICS` | `True` | Plot every diagnostic map in `get_TeNe` |
| `VMIN_DEN`, `VMAX_DEN` | `3`, `4` | Display range of log₁₀(Ne) |
| `VMIN_TEM`, `VMAX_TEM` | `7500`, `12000` | Display range of Te [K] |

## Kinematics (velocity maps)

| Parameter | Default | Meaning |
|---|---|---|
| `C_KMS` | `299792.46` | Speed of light [km/s] |
| `V_SYS` | `35` | Systemic velocity [km/s] (Richer et al. 2022) |
| `VMIN`, `VMAX` | `−35`, `35` | Velocity display range [km/s] |
| `FLUX_LIM` | `−18` | log flux threshold for the velocity maps |
| `FITS_INFO` | tuple | Selected Gaussian-fit maps: (FITS filename, lab wavelength, PyNeb label, display name, plane index) |

## WCS

| Parameter | Default | Meaning |
|---|---|---|
| `wcs_file` | `constants/wcs_NGC6153.joblib` | Precomputed astropy WCS of the MUSE field, loaded at import into `WCS` |
