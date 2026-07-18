# `obs_int.py` — integrated 1D spectrum

[← Back to index](../index.md)

Stand-alone counterpart of [observation](observation.md) for the spatially
integrated spectrum: same processing chain, applied only to the line fluxes measured
on the summed MUSE spectrum.

## Quick use

```python
from obs_int import get_obs_int

obs_int = get_obs_int()               # full default chain
I = obs_int.getIntens()               # dereddened intensities
# element [0] is the measured value; [1:] are the Monte Carlo realizations
```

## Differences with `observation.py`

| Aspect | `Observations` (IFU) | `Obs_int` (integrated) |
|---|---|---|
| Input | Per-line FITS maps + integrated file | Integrated file only |
| Normalization | `FLUX_NORM`, integrated value also divided by n_spaxels | `FLUX_NORM` only |
| Hβ masking | Yes (`CUT_HB`) | Not applicable |
| Monte Carlo | `N_MC` (often `None`) | `N_MC_INT` (default 500) |
| Result | `self.obs` | `self.obs_int` |

## Class `Obs_int`

Same method names and physics as `Observations` (see the
[observation page](observation.md) for details): `read_obs_int`, `norm_obs_int`,
`redefine_lines`, `add_MC`, `red_cor_obs`, `correct_by_cHb`,
`correct_NII_recomb`, `correct_OII_recomb`, `get_obs`.

## Module function

- **`get_obs_int(**kwargs)`** — one-call wrapper returning the calibrated
  `pn.Observation` (kwargs as in `Observations.get_obs`: `tem_cHb`, `den_cHb`,
  `tem_rec`, `den_rec`, `oii_rec_label`, `corr_NII`, `corr_OII`).
