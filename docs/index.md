# NGC-MUSE documentation

Python library developed to analyse VLT/MUSE integral-field data of the planetary
nebula **NGC 6153**, published in Gómez-Llanos et al. (2024). It is built on top of
[PyNeb](https://github.com/Morisset/PyNeb_devel), the Python package for the analysis
of emission-line nebulae.

## Contents

| Page | What it covers |
|---|---|
| [Overview](overview.md) | What the pipeline does, its architecture and data flow |
| [Installation](installation.md) | Conda environment, PyNeb, ai4neb, data and paths |
| [Configuration](configuration.md) | Reference of every parameter in `module/constants/observation_parameters.py` |
| [Notebooks](notebooks.md) | Role of `ICF_ABUND_NGC6153.ipynb` and `Results.ipynb` |
| **API reference** | |
| [observation](api/observation.md) | Reading/dereddening the IFU line maps (`Observations`) |
| [obs_int](api/obs_int.md) | Same for the integrated 1D spectrum (`Obs_int`) |
| [diagnostics](api/diagnostics.md) | Electron temperature/density maps (`get_TeNe`) |
| [ionic_abund](api/ionic_abund.md) | Ionic abundances (`set_abunds`) |
| [figuras_articulo](api/figuras_articulo.md) | Reproducing the paper's figures and tables |
| [utils](api/utils.md) | Label formatting, image extraction, plotting helpers |

## Quick start

Once the [installation](installation.md) is done and the paths in
`module/constants/observation_parameters.py` point to your data:

```python
# From the module/ directory
from observation import get_obs        # calibrated IFU observation
from diagnostics import get_TeNe       # Te/Ne diagnostic maps
from ionic_abund import set_abunds     # ionic abundances

obs = get_obs()                        # read, deredden, correct
TeNe, obs = get_TeNe(obs=obs)          # Te/Ne for every diagnostic
abund = set_abunds(TeNe, obs)          # X^i+/H+ maps for every line
```

To regenerate every figure and table of the paper:

```python
import figuras_articulo as fa          # note: heavy computation at import
fa.create_figures()
fa.create_tables()
```
