# `observation.py` — IFU line maps

[← Back to index](../index.md)

Turns the raw per-line FITS maps (plus the integrated spectrum) into a fully
calibrated PyNeb `Observation`. See also [obs_int](obs_int.md) for the
integrated-spectrum-only counterpart.

## Quick use

```python
from observation import get_obs

obs = get_obs()                       # full default chain
obs = get_obs(corr_NII=False)         # skip the [N II] 5755 correction
obs = get_obs(tem_rec=4000)           # different recombination temperature

I_Hb = obs.getIntens()['H1r_4861A']   # dereddened Hβ map (flat array)
```

## Class `Observations`

State-carrying builder. Boolean flags (`isExtinctionCorrected`, `isNIICorrected`, ...)
ensure each step runs only once. Methods, in the order run by `get_obs`:

| Method | Role |
|---|---|
| `read_obs_IFU()` | Read the per-line FITS maps (`OBS_NAME`) into a `pn.Observation` |
| `read_obs_int()` | Read the integrated fluxes (`OBS_INT_FILE`) |
| `norm_obs_int()` | Scale integrated fluxes by `FLUX_NORM` / n_spaxels |
| `norm_obs_IFU()` | Scale the maps by `FLUX_NORM`, NaN-flag pixels with no real measurement, and store the integrated value of each line in pixel `[0]` of its map |
| `mask_by_Hb()` | NaN-mask spaxels with Hβ < `CUT_HB` × max(Hβ) |
| `redefine_lines()` | Define blends: O I 7773+ (sum of the triplet), O II 4649.13+4650.84, [Ne IV] 4726+ |
| `add_MC()` | Add `N_MC` Monte Carlo realizations (Gaussian noise, fixed seed) |
| `red_cor_obs(...)` | Derive E(B−V) from H I ratio(s) — median over several ratios when iterable — and deredden (`correctData`) |
| `correct_by_cHb(tem, den)` | Compute theoretical Hα/Paschen ratios at (Te, Ne) with `pn.RecAtom('H', 1)` and call `red_cor_obs` |
| `correct_NII_recomb(tem_rec, den_rec)` | Subtract the recombination contribution to [N II] 5755, estimated from N II 5679 (P91/FSL11 emissivities) |
| `correct_OII_recomb(tem_rec, den_rec, rec_label)` | Subtract the recombination contribution to [O II] 7320/7330, estimated from an O II RL (P91/SSB17 emissivities) |
| `get_obs(...)` | Run the whole chain (parameters: `tem_cHb`, `den_cHb`, `tem_rec`, `den_rec`, `oii_rec_label`, `corr_NII`, `corr_OII`) |

## Module functions

- **`get_obs(**kwargs)`** — one-call wrapper: build `Observations`, run the chain,
  return the `pn.Observation`.
- **`get_wcs()`** — return the WCS of the observation (note: rebuilds the whole
  observation; prefer the `WCS` constant for plotting).

## Physics notes

- The **recombination corrections** matter because in a nebula with cold H-poor
  clumps, part of the auroral [N II] 5755 and [O II] 7320/7330 flux comes from
  recombination of N⁺⁺/O⁺⁺ in the cold gas, not from collisional excitation; left
  uncorrected it biases Te([N II]) and the O⁺ abundance upwards.
- Pixel `[0]` of every map holds the **integrated-spectrum value**, so integrated
  quantities travel with the maps (used by the tables of the paper).
