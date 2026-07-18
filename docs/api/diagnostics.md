# `diagnostics.py` — electron temperature and density

[← Back to index](../index.md)

Derives the Te/Ne maps from the calibrated observation, using CEL ratio pairs
(ANN-accelerated), the Paschen jump and the He I 7281/6678 ratio.

## Quick use

```python
from observation import get_obs
from diagnostics import get_TeNe

obs = get_obs()
TeNe, obs = get_TeNe(obs=obs, plot=False)

TeNe['N2S2']['Te']    # Te[N II] map (K)
TeNe['N2S2']['Ne']    # Ne[S II] map (cm^-3)
TeNe['PJ']['Te']      # Paschen-jump temperature
TeNe['He1']['Te']     # He I temperature
```

The result dictionary has one entry per key of `DIAGNOSTICS_DICT`
(`N2S2`, `S3Cl3`, `S3S2`, `S3Ar4`, `Ar3Cl3`, `Ar3S2`, `Ar4Cl3`, `Ar4S2`, `Ar4Ar4`),
each with `'Te'` and `'Ne'`, plus `'PJ'` and `'He1'` (temperature only).

## Functions

| Function | Role |
|---|---|
| `make_diags(obs, use_ANN)` | Build the `pn.Diagnostics` object; register diagnostics from the observed lines and add [Ar IV] 7170/4740 |
| `add_gCTD(diags, obs, label, diag1, diag2, TeNe, ...)` | Solve one Te×Ne pair with `getCrossTemDen`. First run trains and saves the ai4neb ANN in `new_ai4neb/<label>`; later runs reload it (`force=True` retrains) |
| `add_T_PJ(TeNe, obs, den, Hep, Hepp)` | Paschen-jump Te: tabulate the theoretical jump with `pn.Continuum.BJ_HI` (8100/8400 Å continua vs H I 9229 Å) and invert by interpolation |
| `add_T_He(TeNe, obs)` | He I Te from I(7281)/I(6678) with the density-dependent fit of Méndez-Delgado et al. (2021); uses `TeNe['N2S2']['Ne']`, so N2S2 must exist |
| `get_TeNe(obs=None, use_ANN=USE_ANN, plot=PLOT_DIAGNOSTICS)` | **Main driver**: loops over `DIAGNOSTICS_DICT`, then adds PJ and He I. Returns `(TeNe, obs)` |
| `plot_tem_den(TeNe, key_diag, ...)` | Side-by-side log Ne / Te maps of one diagnostic |
| `plot_PJ(TeNe, obs, **kwargs)` / `plot_He(TeNe, obs, **kwargs)` | Plot the PJ / He I temperature map, computing it first if missing |

## Notes

- **ANN acceleration**: `getCrossTemDen` inverts two line ratios per pixel; on
  200×200 pixels (× Monte Carlo) the exact solver is slow, so an ai4neb neural
  network is trained once per diagnostic to emulate the inversion. Delete the
  `new_ai4neb/` folder (or pass `force=True` to `add_gCTD`) to retrain.
- If ai4neb is not installed, the module still imports (`AI4NEB_INSTALLED = False`);
  call `get_TeNe(use_ANN=False)`.
- The Paschen-jump and He I temperatures probe the gas emitting the H/He
  **recombination** spectrum; they come out much lower than the CEL temperatures —
  the signature of the cold component (paper fig. 10).
