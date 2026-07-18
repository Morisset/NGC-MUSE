# `figuras_articulo.py` — paper figures and tables

[← Back to index](../index.md)

Reproduces every figure (`module/paper_figures/*.pdf`) and LaTeX table
(`module/paper_tables/*.tex`) of Gómez-Llanos et al. (2024).

> **Import cost**: importing this module immediately builds the calibrated
> observation and computes all Te/Ne diagnostic maps (module-level `obs` and
> `TeNe`), which takes minutes and needs valid data paths. All functions then share
> these two objects.

```python
import figuras_articulo as fa
fa.create_figures()   # fig1 ... fig15
fa.create_tables()    # tab4, tab5, tab6, tabA1
```

## Figures

| Function | Output | Content |
|---|---|---|
| `fig1_rgb_image` | `rgb_image.pdf` | RGB composite: R = ([N II]+[S II])/2, G = Hβ, B = He II 4686 |
| `fig2_obs_fluxes` | `obs_fluxes.pdf` | Grid of observed log-flux maps of 20 key lines |
| `figx_Kr_fluxes` | `kr_iii_iv_fluxes.pdf` | Extra: [Kr III] 6827 and [Kr IV] 5868 flux maps |
| `fig3_logFHb` | `log_FHb.pdf` | Hβ surface-brightness map |
| `fig4_cHb_dist` | `cHb_dist.pdf` | c(Hβ) histograms for different Te/Ne assumptions + literature values |
| `fig5_cHb` | `cHb.pdf` | Adopted c(Hβ) map |
| `fig6_emis_NII_rec` | `emis_NII_rec.pdf` | Theoretical N II j(5755)/j(5679) recombination ratio vs Te |
| `fig7_Te_NII_recomb_corr` | `Te_NII_recomb_corr.pdf` | Te([N II]) without/with the recombination correction (Te_rec = 2000/4000/6000 K); recomputes everything 4× (slow) |
| `fig8_NII_OII_recomb_corr` | `NII_OII_recomb_corr.pdf` | [N II] 5755 and [O II] 7330+ maps before/after correction |
| `fig9_TeNe_CELs` | `TeNe_CELs_b.pdf` | CEL Te (top) and log Ne (bottom) for 4 cross-diagnostics |
| `fig10_Te_HeI_PJ_SIII` | `Te_HeI_PJ_SIII.pdf` | Te(He I) vs Te(PJ) vs Te([S III]) |
| `fig11_ab_o2r` | `ab_o2r.pdf` | O⁺⁺/H⁺ from the O II 4649+ and 4661 RLs + histograms |
| `fig12_smooth_omega` | `2_omega_smooth.pdf` | 1/ω and 1/(1−ω) maps; **also creates `omega_mask.joblib` / `omega_1_mask.joblib`** needed by fig13–15 and tab5 |
| `fig13_ion_ab_with_omega` | `ion_ab_with_omega.pdf` | Grid of 17 ionic-abundance maps with ω weighting |
| `fig14_O_adf_acf` | `O_adf_acf.pdf` | ADF vs ACF maps of O⁺ and O⁺⁺ |
| `fig15_hep_warm_cold` | `hep_warm_cold.pdf` | He⁺/H⁺ of the warm and cold components (two-component He I inversion) |

Run order matters: `fig12_smooth_omega` must run before fig13/fig14/fig15 (it
creates the masked ω files); `create_figures()` respects this.

## Tables

| Function | Output | Content |
|---|---|---|
| `tab4_intTeNe` | `int_tene.tex` | Te/Ne of the integrated spectrum for every diagnostic (± MC std) |
| `tab5_int_ion_ab` | `ionic_ab_7_recipes.tex` | Integrated ionic abundances under 7 recipes (from pure-CEL to full 3-zone + ω schemes) |
| `tab6_O_mass_frac` | `O_mass_frac.tex` | Cold-to-warm oxygen mass ratios M^c/M^w for O⁺ and O⁺⁺ |
| `tabA1_int_fluxes` | `int_fluxes.tex` | Observed and dereddened integrated fluxes (Hβ = 100) |

## Helpers

| Function | Role |
|---|---|
| `get_int_std(abund, label)` | (12+log abundance of pixel [0], std over the array) |
| `print2(to_print, f)` | Print to screen and write to the open LaTeX file |
| `norm_data(data)` | NaN-clean and normalize a map to its maximum |
| `ion_prefix(label)` | LaTeX ionic label of a line (e.g. `O$^{2+}$/H$^+$`) |
| `convolve_omega(new_filename)` | Smooth/interpolate/mask the raw CatBoost ω map and save it |

The `atomic_data` dictionary at the end of the file lists the PyNeb atomic data sets
used in the paper.
