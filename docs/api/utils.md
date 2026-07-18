# `utils/` — misc and plotting helpers

[← Back to index](../index.md)

## `utils/misc.py`

Label formatting and map extraction.

| Function | Role |
|---|---|
| `get_label_str(label, split=False, latex=False)` | PyNeb label → readable name: `'O3_5007A'` → `'[O III] 5007'`; with `latex=True` uses the paper macros `\forb{}{}` / `\perm{}{}`; with `split=True` returns `(elem, spec_roman, wl_str, is_forbidden)` |
| `get_label_str_ori(label, split=False)` | Original pre-LaTeX version, kept for reference |
| `check_recomb(label)` | True if the label is a recombination line (atom ends with `r`) |
| `get_image(obs, data=None, label=None, N_MC=None, type_='median', returnObs=True)` | Reshape a line's flat array (or `data`) to the 2D map; collapse the Monte Carlo axis with `type_` = `'median'`, `'mean'`, `'std'` or `'orig'` (first realization); a tuple label returns the image ratio |

```python
from utils.misc import get_image
im  = get_image(obs, label='O3_5007A')                  # 2D map
std = get_image(obs, label='O3_5007A', N_MC=500, type_='std')   # MC uncertainty map
```

## `utils/plots.py`

Display helpers used everywhere (they import the field `WCS` and `N_MC` from the
configuration).

| Function | Role |
|---|---|
| `create_axis(n_line_maps, suptitle="", n_columns=3, ..., show_ticks=False)` | Figure + grid of axes sized for N maps; `show_ticks=True` uses the WCS projection (RA/Dec) |
| `plot_image(image, label="", fig=None, ax=None, ...)` | Display one map; accepts 2D images, flat 40 000-element vectors (reshaped 200×200) or flat MC cubes (first realization shown); optional colorbar, WCS ticks, `vmin`/`vmax`/`cmap` |
| `plot_ionic_ab(abund_dic, abund_keys=None, dex_range=0.5)` | Grid of log ionic-abundance maps, color range = median ± `dex_range` dex |
| `plot_fluxes(obs, returnObs=True, **kwargs)` | log-flux maps of every line, RLs and CELs in separate figures |
| `plot_ann_test(pred, ann, tem_diag=..., den_diag=...)` | Predicted-vs-true Te and log Ne scatter plots to validate an ai4neb ANN |

```python
from utils.plots import plot_image
import numpy as np
plot_image(np.log10(obs.getIntens()['H1r_4861A']), label='log I(Hβ)',
           cmap='inferno', vmin=-17, vmax=-14.8)
```
