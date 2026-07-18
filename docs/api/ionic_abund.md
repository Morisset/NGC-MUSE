# `ionic_abund.py` — ionic abundances

[← Back to index](../index.md)

Converts dereddened line intensities into ionic abundances (X^i+/H⁺) with PyNeb's
`getIonAbundance`, choosing the Te/Ne zone per ion and optionally applying the
warm/cold two-component weighting.

## Quick use

```python
from ionic_abund import set_abunds

# Simple: every line, CEL zones, no two-component weighting
abund = set_abunds(TeNe, obs)

# Paper recipe: RLs at (2000 K, 1e4 cm^-3), omega weighting
import joblib
w   = joblib.load('omega_mask.joblib')     # cold Hβ fraction
w_1 = joblib.load('omega_1_mask.joblib')   # warm Hβ fraction
abund = set_abunds(TeNe, obs, Te_rec=2000, Ne_rec=10000, w=w, w_1=w_1)

abund['O3_4959A']        # O++/H+ map from [O III] 4959 (warm component)
abund['O2r_4649.13A']    # O++/H+ map from O II RLs (cold component)
```

## Functions

| Function | Role |
|---|---|
| `create_rec_atom(line)` | Build a `pn.RecAtom` (Case A for C II/O I, Case B otherwise) and return it with the IP needed to create the emitting ion |
| `create_col_atom(line)` | Build a `pn.Atom` for a CEL, with the corresponding IP (0 for neutral emitters) |
| `create_atom(line)` | Dispatch on the label (trailing `r` → recombination) and return `(atom, IP, rec_line)` |
| `select_TeNe(TeNe, IP, consider_3zones=False, use_ar3=False)` | Zone assignment: IP < 17 eV → `N2S2`; IP ≥ 17 eV → `S3Cl3`; with `consider_3zones` and IP ≥ 35 eV → Ar-based Te ([Ar III]/[Cl III] alone, or averaged with [Ar IV] weighted by typical Ar⁺⁺/Ar³⁺ abundances) |
| `select_Te_rec(TeNe, Te, Ne, Te_rec)` | Te for the heavy-element RLs: `'He'` (He I diag), `'PJ'`, `'PJ_ANN'`, `None` (keep CEL Te), or a constant [K] |
| `set_abunds_ori(TeNe, obs, ...)` | Original single-component version (no ω weighting) |
| `set_abunds(TeNe, obs, w=None, w_1=None, ...)` | **Paper version**: two-component scheme (see below) |

## The two-component scheme (`set_abunds`)

For each valid line of the observation:

1. build/cache the PyNeb atom and get the ion's IP;
2. pick Te/Ne from the IP (`select_TeNe`, kwargs `consider_3zones`/`use_ar3` are
   forwarded);
3. if the line is a **recombination line**: use `tem_HI_RLs` for the H I emissivity;
   He I lines get Te(He I) with Ne(N2S2), He II lines keep their zone's Te/Ne,
   heavy-element RLs get `Te_rec`/`Ne_rec`; Hβ weight = `w` (cold fraction ω);
4. if the line is a **CEL**: use `tem_HI`; Hβ weight = `w_1` (warm fraction 1−ω);
5. call `atom.getIonAbundance(intensity/Hβ, Te, Ne, to_eval, Hbeta=weight, tem_HI=...)`.

The ω weighting means each family of lines is normalized to the Hβ actually emitted
by its own component — the abundances of the cold clumps become physically
meaningful instead of diluted over all the gas.

Returned value: `dict` mapping `line.label` → abundance array (same shape as the
maps); skipped lines are `None`.
