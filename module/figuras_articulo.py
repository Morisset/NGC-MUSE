"""Reproduction of the figures and tables of Gomez-Llanos et al. (2024).

Every figure of the paper (saved to ``paper_figures/*.pdf``) and every
LaTeX table (saved to ``paper_tables/*.tex``) is produced by a dedicated
function of this module: ``fig1_rgb_image`` ... ``fig15_hep_warm_cold``
(plus the extra ``figx_Kr_fluxes``) and ``tab4_intTeNe`` ...
``tabA1_int_fluxes``. The drivers `create_figures` and `create_tables`
run them all in order.

Important: at import time the module builds the calibrated observation
(``obs = get_obs(...)``) and computes all the Te/Ne diagnostic maps
(``TeNe``), which takes a while and requires the data paths of
`constants/observation_parameters.py` to be valid. All figure/table
functions then share these module-level ``obs`` and ``TeNe``.

Several functions rely on the "omega" weight maps: omega is the fraction
of the Hbeta emission coming from the cold, H-poor component, predicted
by a machine-learning model in the ``ICF_ABUND_NGC6153.ipynb`` notebook
and stored in ``w_pred_CatBoost.joblib`` / ``w_pred_convolved.joblib``.
`fig12_smooth_omega` masks and saves them as ``omega_mask.joblib`` and
``omega_1_mask.joblib``, which `fig13_ion_ab_with_omega`,
`fig14_O_adf_acf`, `fig15_hep_warm_cold` and `tab5_int_ion_ab` reload.

The ``atomic_data`` dictionary at the end of the file lists, for
reference, the PyNeb atomic data sets used in the paper.
"""

import matplotlib.pyplot as plt
from observation import get_obs
from obs_int import get_obs_int
from diagnostics import get_TeNe
from ionic_abund import set_abunds
from utils.plots import plot_image, create_axis
from utils.misc import get_label_str
from constants.observation_parameters import TEM_CHB, DEN_CHB, TE_CORR, DEN_CORR, WCS
from pyneb.utils.misc import parseAtom
from scipy.ndimage.filters import gaussian_filter
from astropy.convolution import Gaussian2DKernel
from pyneb.core.pynebcore import getHbEmissivity
from astropy.convolution import interpolate_replace_nans
import numpy as np
import pyneb as pn
import joblib

omega_filename = 'w_pred_convolved.joblib'

# Shared by all figure/table functions: calibrated observation and Te/Ne
# diagnostic maps, computed once at import time.
obs = get_obs(corr_OII = True, corr_NII = True)
TeNe, obs = get_TeNe(obs = obs, plot=False)

def get_int_std(abund, label):
    """Return (12+log abundance of the integrated spectrum, std over MC/pixels).

    ``abund[label][0]`` holds the integrated-spectrum value (stored in
    element 0 by convention); the standard deviation is computed over the
    full log-abundance array.
    """
    ab = abund[label]
    ab_int = 12 + np.log10(ab[0])
    std_int = np.nanstd(np.log10(ab))
    return ab_int, std_int

def print2(to_print, f):
    """Print a line to screen and append it to the open file ``f``."""
    print(to_print)
    f.write(to_print + '\n')

def norm_data(data):
    """Replace non-finite values by NaN and normalize the map to its maximum."""
    data_clean = np.where(np.isfinite(data), data, np.nan)
    return data_clean/np.nanmax(data_clean)

def ion_prefix(label):
    """Build the LaTeX ionic-abundance label (e.g. 'O$^{2+}$/H$^+$') of a line.

    For collisionally excited lines the emitting ion has the charge of
    the spectroscopic stage minus one ([O III] -> O2+); for recombination
    lines (label ending in 'r') the spectrum number is the ion charge
    itself (O II RLs -> O2+... expressed here from the parsed spectrum).
    """
    spec = label.split('_')[0]
    #wl = label.split('_')[1]
    if spec[-1] == 'r':
        spec = '{}'.format(spec[0:-1])
        forb = False
    else:
        forb = True

    elem, ion = parseAtom(spec)

    if forb:
        ionic = str(int(ion)-1)
    else:
        ionic = ion

    if ionic == '1':
        prefix = '+'
    elif ionic == '0':
        prefix = '0'
    else:
        prefix = ionic + '+'

    return r'%s$^{%s}$/H$^+$'%(elem,prefix)

def convolve_omega(new_filename):
    """Smooth, interpolate and mask the ML-predicted omega map, then save it.

    Loads the raw CatBoost prediction (``w_pred_CatBoost.joblib``),
    fills its NaN holes by convolution with a 2D Gaussian kernel, masks
    the regions where the N II 5679 recombination line is too faint to
    constrain omega, and dumps the result to ``new_filename``
    (typically ``w_pred_convolved.joblib``). A control figure comparing
    raw, interpolated and masked versions (with [N II] 6548 contours) is
    saved to ``paper_figures_std/n2_contours.pdf``.

    Parameters
    ----------
    new_filename : str
        Path of the joblib file where the convolved omega map is saved.
    """
    line_intens = obs.getIntens()["N2_6548A"]
    sigma = 2 # this depends on how noisy your data is
    data = line_intens.reshape(200,200)
    data_filt = gaussian_filter(data, sigma)

    omega = joblib.load('w_pred_CatBoost.joblib')
    grid = omega.reshape(200,200)


    gaussian_2D_kernel = Gaussian2DKernel(3)

    result = interpolate_replace_nans(grid, gaussian_2D_kernel)
    N3r = obs.getIntens(returnObs=True)['N2r_5679.56A']
    mask_N3r = np.where(np.log10(N3r) > -18.5, 1, np.nan)
    mask = mask_N3r.reshape(200,200)
    omega_to_save = result*mask

    fig, ax = create_axis(3, n_columns=3, scale_y = 4.2, scale_x = 5)
    fig.subplots_adjust(left = 0.18, bottom = 0.198, right = 0.96, top = 0.893, wspace = 0)

    ax[0].contour((data_filt), colors = 'r')
    ax[0].imshow(grid, vmin = 0.01, vmax = 0.18)

    ax[1].contour((data_filt), colors = 'r')
    ax[1].imshow(grid, interpolation='nearest', vmin = 0.01, vmax = 0.18)

    ax[2].contour((data_filt), colors = 'r')
    ax[2].imshow(omega_to_save, vmin = 0.01, vmax = 0.18)
    fig.savefig('paper_figures_std/n2_contours.pdf')
    joblib.dump(omega_to_save.ravel(), new_filename)

#---------------------------------------FIGURES-----------------------------------------------------------------------------------------
def fig1_rgb_image(figname = "paper_figures/rgb_image.pdf"):
    """Fig. 1: RGB composite of the nebula.

    Red = ([N II] 6548 + [S II] 6716)/2 (low ionization), green =
    Hbeta, blue = He II 4686 (high ionization); each channel normalized
    to its maximum.
    """
    He2 = (obs.getIntens(returnObs=True)['He2r_4686A']).reshape(200,200)
    He2_n = norm_data(He2)

    Hb = (obs.getIntens(returnObs=True)['H1r_4861A']).reshape(200,200)
    Hb_n = norm_data(Hb)

    N2 = (obs.getIntens(returnObs=True)['N2_6548A']).reshape(200,200)
    N2_n = norm_data(N2)

    S2 = (obs.getIntens(returnObs=True)['S2_6716A']).reshape(200,200)
    S2_n = norm_data(S2)

    N2_S2 = (N2_n + S2_n)/2

    rgb = np.dstack((N2_S2, Hb_n, He2_n))

    plt.rcParams['font.size'] = 12

    fig, ax = create_axis(1, n_columns = 1, show_ticks = True, scale_x = 6, scale_y = 5)

    plot_image(rgb, create_colorbar = False, fig = fig, ax = ax, show_ticks = True)
    fig.subplots_adjust(top = 0.98, bottom = 0.1, left = 0.25)
    plt.savefig(figname)

def fig2_obs_fluxes(figname = 'paper_figures/obs_fluxes.pdf'):
    """Fig. 2: grid of observed (uncorrected) log flux maps of 20 key lines.

    One panel per line, with hand-tuned display ranges per line
    (``lines_dict``), from He I/He II to the faint Kr IV line.
    """
    lines_dict={'He1r_5876A': (-17.5, -15.1),
                'He2r_4686A': (-18.3, -15.2),
                'C2r_6462.0A': (-18.5, -16.8),
                'N1_5198A': (-18, -16.8),
                'N2_6548A': (-18, -14.8),
                'N2r_5679.56A': (-18.4, -16.65),
                'N3r_4641A': (-18, -16),
                'O1_6300A': (-17.6, -15.6),
                'O1r_7773+': (-18.2, -16.8),
                'O2_7330A+': (-18, -15.8),
                'O2r_4649.13A': (-18, -16.25),
                'O3_4959A': (-16.5, -14.2),
                'S2_6731A': (-17.6, -15.1),
                'S3_9069A': (-17, -14.3),
                'Cl3_5538A': (-18.5, -16.6),
                'Cl4_8046A': (-18.5, -16.15),
                'Ar3_7136A': (-16.8, -14.85),
                'Ar4_4740A': (-18.2, -16.3),
                'Ar5_7005A': (-18.4, -17.4),
                'Kr4_5868A': (-18.2, -17.5),
                }

    labels = list(lines_dict.keys())
    n_lines = len(labels)

    plt.rcParams['font.size'] = 7
    fig, axs = create_axis(n_lines, n_columns=4, scale_x=2.5, scale_y=2)
    fig.subplots_adjust(left = 0.052, right = 0.952, top = 0.962, wspace = 0.25, hspace = 0.2)
    for index, label in enumerate(labels):
        line = obs.getIntens(returnObs=True)[label]
        vmin, vmax = lines_dict[label]
        plot_image(np.log10(line), vmin = vmin, vmax = vmax,
        label = get_label_str(label), cmap = 'plasma', fig = fig,
        ax = axs.ravel()[index], title_size=7)
    #axs.ravel()[-1].remove()
    fig.savefig(figname)

def figx_Kr_fluxes(figname = 'paper_figures/kr_iii_iv_fluxes.pdf'):
    """Extra figure: observed flux maps of the [Kr III] 6827 and [Kr IV] 5868 lines."""
    lines_dict = {'Kr4_5868A': (-18.5, -17.4), 'Kr3_6827A': (-19,-17)}

    labels = list(lines_dict.keys())

    plt.rcParams['font.size'] = 12
    fig, axs = create_axis(2, n_columns=2, scale_y = 4.2, scale_x = 5)
    fig.subplots_adjust(left = 0.08, bottom = 0.198, right = 0.995, top = 0.893, wspace = 0.14, hspace = 0.1)

    for index, label in enumerate(labels):
        line = obs.getIntens(returnObs=True)[label]
        vmin, vmax = lines_dict[label]
        plot_image(np.log10(line), vmin = vmin, vmax = vmax,
                   label = get_label_str(label), cmap = 'plasma', fig = fig,
                   ax = axs.ravel()[index], title_size=12)
    fig.savefig(figname)

def fig3_logFHb(figname = "paper_figures/log_FHb.pdf"):
    """Fig. 3: observed Hbeta surface-brightness map, log F(Hbeta)."""
    plt.rcParams['font.size'] = 14
    hb_obs = obs.getIntens(returnObs=True)['H1r_4861A']
    fig, ax = create_axis(1, n_columns=1) #scale_y = 4.2, scale_x = 5
    fig.subplots_adjust(left = 0.04, bottom = 0.1, right = 0.96, top = 0.893, wspace = 0)
    plot_image(np.log10(hb_obs), fig = fig, ax = ax, vmin = -17.03, vmax = -14.78,
                label = r'log F(H$\beta$)', title_size = 14, cmap = 'inferno')
    plt.savefig(figname)

def fig4_cHb_dist(figname = 'paper_figures/cHb_dist.pdf'):
    """Fig. 4: distributions of c(Hbeta) for different Te/Ne assumptions.

    Recomputes the extinction map with the Te/Ne of several diagnostics
    (N2S2, S3Cl3, PJ, Ar3Cl3, Ar4Cl3, and the adopted constant
    TEM_CHB/DEN_CHB) and plots the histograms, with the c(Hbeta)
    literature values of NGC 6153 overplotted as dashed vertical lines.
    """
    cHbeta = {}

    tem_den = {'N2S2':r'T$_e$([N II]), N$_e$([S II])', 'S3Cl3':r'T$_e$([S III]), N$_e$([Cl III])',
            'PJ':r'T$_e$(PJ), N$_e$([Cl III])', 'Ar3Cl3':r'T$_e$([Ar III]), N$_e$([Cl III])', 'Ar4Cl3':r'T$_e$([Ar IV]), N$_e$([Cl III])'}

    for key in tem_den.keys():

        if key == 'PJ':
            den = TeNe['S3Cl3']['Ne']
        else:
            den = TeNe[key]['Ne']
        obs = get_obs(tem_cHb = TeNe[key]['Te'], den_cHb = den)

        cHbeta[key] = obs.extinction.cHbeta

    obs = get_obs(tem_cHb = TEM_CHB, den_cHb = DEN_CHB)

    cHbeta['T_{}_N_{}'.format(TEM_CHB, DEN_CHB)] = obs.extinction.cHbeta
    tem_den['T_{}_N_{}'.format(TEM_CHB, DEN_CHB)] = r'T$_e$={}, N$_e$={}'.format(TEM_CHB, DEN_CHB)

    colors = ['r','g', 'b', 'magenta', 'cyan', 'orange']
    keys = ('N2S2', 'S3Cl3', 'PJ', 'Ar3Cl3', 'Ar4Cl3', 'T_{}_N_{}'.format(TEM_CHB, DEN_CHB))

    plt.rcParams['font.size'] = 22
    fig, ax = plt.subplots(figsize = (12,8.5))

    for index, key in enumerate(keys):

        ax.hist(cHbeta[key], bins = np.linspace(0.8, 1.4, 100), color = colors[index], alpha = 0.4, label = tem_den[key], density = True)
        ax.vlines(np.nanmedian(cHbeta[key]), 0, 17, color = colors[index])

    ax.vlines(0.96, 0, 17, linestyle = '--', color = colors[0], label = 'Kingsburgh & Barlow (1994)')
    ax.vlines(1.27, 0, 17, linestyle = '--', color = colors[1], label = 'Liu et al. (2000)')
    ax.vlines(1.19, 0, 17, linestyle = '--', color = colors[2], label = 'Pottasch et al. (2003)')
    ax.vlines(1.15, 0, 17, linestyle = '--', color = 'purple', label = 'Tsamis et al. (2008)')
    ax.vlines(1.32, 0, 17, linestyle = '--', color = colors[3], label = 'McNabb et al. (2016)')
    ax.vlines(0.76, 0, 17, linestyle = '--', color = colors[4], label = 'Richer et al. (2022)')

    ax.set_ylim(0,16)
    ax.legend(loc = 2)
    ax.set_xlabel(r'c(H$\beta$)');
    fig.tight_layout()
    fig.savefig(figname)

def fig5_cHb(figname = "paper_figures/cHb.pdf"):
    """Fig. 5: map of the adopted extinction coefficient c(Hbeta)."""
    plt.rcParams['font.size'] = 14
    fig, ax = create_axis(1, n_columns=1) #scale_y = 4.2, scale_x = 5
    fig.subplots_adjust(left = 0.04, bottom = 0.1, right = 0.96, top = 0.893, wspace = 0)
    plot_image(obs.extinction.cHbeta, fig = fig, ax = ax, vmin = 1.115, vmax = 1.295,
            label = r'c(H$\beta$)', title_size=14, cmap = 'inferno')
    plt.savefig(figname)

def fig6_emis_NII_rec(figname = "paper_figures/emis_NII_rec.pdf"):
    """Fig. 6: theoretical N II recombination ratio j(5755)/j(5679) vs Te.

    Curves for several densities, computed with the P91 (5755) and FSL11
    (5679) recombination data; this is the ratio used to correct
    [N II] 5755 for its recombination contribution.
    """
    plt.rcParams['font.size'] = 12

    linestyles = ['-',':', '--', '-.', '-']

    tem = np.linspace(1000,8000, 50)
    den = np.logspace(2,4,5)

    pn.atomicData.setDataFile('n_ii_rec_P91.func')
    N2rP = pn.RecAtom('N', 2, case='B')
    pn.atomicData.setDataFile('n_ii_rec_FSL11.func')
    N2rF = pn.RecAtom('N', 2, case='B')

    R = (N2rP.getEmissivity(tem, den, label='5755.', product=True) /
        N2rF.getEmissivity(tem, den, label='5679.56', product=True))

    fig, ax = plt.subplots(figsize = (5,5))
    for index, d in enumerate(den):
        d_str = np.round(np.log10(d),1)
        ax.plot(tem, R.T[index], label = r'Ne = $10^{%.1f}$ cm$^{-3}$'%d_str, ls = linestyles[index])

    ax.set_xlabel('Te [K]')
    ax.legend()
    ax.set_ylabel(r'j$_{5755}$/j$_{5679}$')
    fig.savefig(figname)

def fig7_Te_NII_recomb_corr(figname = "paper_figures/Te_NII_recomb_corr.pdf"):
    """Fig. 7: effect of the [N II] recombination correction on Te([N II]).

    Four Te([N II]) maps: without correction, and with the correction
    computed assuming a recombination temperature of 2000, 4000 and
    6000 K. This is expensive: the observation and diagnostics are
    recomputed four times.
    """
    font_size = 12
    plt.rcParams['font.size'] = font_size

    Ne_recom = DEN_CORR
    obs_no_corr = get_obs(corr_NII = False)
    obs_corr_2000 = get_obs(tem_rec = 2000, den_rec = Ne_recom)
    obs_corr_4000 = get_obs(tem_rec = 4000, den_rec = Ne_recom)
    obs_corr_6000 = get_obs(tem_rec = 6000, den_rec = Ne_recom)

    TeNe_no_corr, _ = get_TeNe(obs = obs_no_corr, plot = False )
    TeNe_corr_2000, _ = get_TeNe(obs = obs_corr_2000, plot=False)
    TeNe_corr_4000, _ = get_TeNe(obs = obs_corr_4000, plot=False)
    TeNe_corr_6000, _ = get_TeNe(obs = obs_corr_6000, plot=False)

    fig, ax = create_axis(4, n_columns=2) #scale_y=3.8, scale_x=4.9
    fig.subplots_adjust(left = 0.01, wspace = 0.05, top = 0.95, bottom = 0.05, right = 0.99, hspace = 0.15)
    cmap = 'plasma'
    vmin = 7500
    vmax = 12000
    plot_image(TeNe_no_corr["N2S2"]["Te"], vmin = vmin, vmax = vmax, label = r"T$_e$([N II])",
            fig = fig, ax = ax[0,0], cmap = cmap, title_size=font_size)

    plot_image(TeNe_corr_2000["N2S2"]["Te"], vmin = vmin, vmax = vmax, label = r"T$_e$([N II]), T$_{e,R}$ = 2000 K",
            fig = fig, ax = ax[0,1], cmap = cmap, title_size=font_size)

    plot_image(TeNe_corr_4000["N2S2"]["Te"], vmin = vmin, vmax = vmax, label = r"T$_e$([N II]) T$_{e,R}$ = 4000 K",
            fig = fig, ax = ax[1,0], cmap = cmap, title_size=font_size)

    plot_image(TeNe_corr_6000["N2S2"]["Te"], vmin = vmin, vmax = vmax, label = r"T$_e$([N II]) T$_{e,R}$ = 6000 K",
            fig = fig, ax = ax[1,1], cmap = cmap, title_size=font_size)

    fig.savefig(figname)

def fig8_NII_OII_recomb_corr(figname = "paper_figures/NII_OII_recomb_corr.pdf"):
    """Fig. 8: [N II] 5755 and [O II] 7330+ maps before/after recombination correction.

    Left column: total (uncorrected) dereddened fluxes; right column: the
    collisional part after subtracting the recombination contribution.
    """
    obs_uncorr = get_obs(corr_NII = False, corr_OII = False)
    obs_corr = get_obs(tem_rec = TE_CORR, den_rec = DEN_CORR)

    N2_total = obs_uncorr.getIntens(returnObs=False)["N2_5755A"]
    N2_col = obs_corr.getIntens(returnObs=False)["N2_5755A"]

    O2_total = obs_uncorr.getIntens(returnObs=False)['O2_7330A+']
    O2_col = obs_corr.getIntens(returnObs=False)['O2_7330A+']

    font_size = 12
    plt.rcParams['font.size'] = font_size

    cmap = 'plasma'

    fig, ax = create_axis(4, n_columns=2)#scale_y=3.8, scale_x=4.9
    fig.subplots_adjust(left = 0.01, wspace = 0.05, top = 0.95, bottom = 0.05, hspace = 0.15)

    plot_image(np.log10(N2_total), vmin = -17.3, vmax = -15.2, label = r"log F([N II] $\lambda$5755)",
            fig = fig, ax = ax[0,0], cmap = cmap, title_size = font_size, create_colorbar = False)

    plot_image(np.log10(N2_col), vmin = -17.3, vmax = -15.2, label = r"log F([N II] $\lambda$5755)",
            fig = fig, ax = ax[0,1], cmap = cmap, title_size = font_size, create_colorbar = True)

    plot_image(np.log10(O2_total), vmin = -18, vmax = -15, label = r"log F([O II] $\lambda$7330+)",
            fig = fig, ax = ax[1,0], cmap = cmap, title_size = font_size, create_colorbar = False)

    plot_image(np.log10(O2_col), vmin = -18, vmax = -15, label = r"log F([O II] $\lambda$7330+)",
            fig = fig, ax = ax[1,1], cmap = cmap, title_size = font_size, create_colorbar = True)

    fig.savefig(figname)

def fig9_TeNe_CELs(figname = "paper_figures/TeNe_CELs_b.pdf"):
    """Fig. 9: CEL Te (top row) and log Ne (bottom row) maps.

    Columns: N2S2, S3Cl3, Ar3Cl3 and Ar4Cl3 cross-diagnostics.
    """
    plt.rcParams['font.size'] = 12
    font_size = 12

    fig, ax = create_axis(8, n_columns=4)
    fig.subplots_adjust(left = 0.01, wspace = 0.05, top = 0.95, bottom = 0.05, right = 0.99, hspace = 0.15)

    cmap = 'inferno'

    plot_image(TeNe['N2S2']['Te'], fig = fig, ax = ax[0,0], vmin = 7250, vmax = 10000, title_size=font_size,
                label = 'T$_e$([N II]) (n$_e$([S II]))', cmap = cmap)
    plot_image(np.log10(TeNe['N2S2']['Ne']), fig = fig, ax = ax[1,0], vmin = 3, vmax = 4, title_size=font_size,
                label = 'log n$_e$([S II]) (T$_e$([N II]))', cmap = cmap)

    plot_image(TeNe['S3Cl3']['Te'], fig = fig, ax = ax[0,1], vmin = 7250, vmax = 10000, title_size=font_size,
                label = 'T$_e$([S III]) (n$_e$([Cl III]))', cmap = cmap)
    plot_image(np.log10(TeNe['S3Cl3']['Ne']), fig = fig, ax = ax[1,1], vmin = 3, vmax = 4, title_size=font_size,
                label = 'log n$_e$([Cl III]) (T$_e$([S III]))', cmap = cmap)

    plot_image(TeNe['Ar3Cl3']['Te'], fig = fig, ax = ax[0,2], vmin = 7250, vmax = 10000, title_size=font_size,
                label = 'T$_e$([Ar III]) (n$_e$([Cl III]))', cmap = cmap)
    plot_image(np.log10(TeNe['Ar3Cl3']['Ne']), fig = fig, ax = ax[1,2], vmin = 3, vmax = 4, title_size=font_size,
                label = 'log n$_e$([Cl III]) (T$_e$([Ar III]))', cmap = cmap)

    plot_image(TeNe['Ar4Cl3']['Te'], fig = fig, ax = ax[0,3], vmin = 8500, vmax = 14000, title_size=font_size,
                label = 'T$_e$([Ar IV]) (n$_e$([Cl III]))', cmap = cmap)
    plot_image(np.log10(TeNe['Ar4Cl3']['Ne']), fig = fig, ax = ax[1,3], vmin = 3, vmax = 4, title_size=font_size,
                label = 'log n$_e$([Cl III]) (T$_e$([Ar IV]))', cmap = cmap)

    fig.savefig(figname)

def fig10_Te_HeI_PJ_SIII(figname = "paper_figures/Te_HeI_PJ_SIII.pdf"):
    """Fig. 10: comparison of Te(He I), Te(Paschen jump) and Te([S III]).

    The RL/continuum temperatures (left, middle) are markedly lower than
    the CEL temperature (right) — the signature of the cold component.
    """
    font_size = 12
    plt.rcParams['font.size'] = font_size

    fig, ax = create_axis(3, n_columns=3)#scale_y=4.2, scale_x=4.9
    fig.subplots_adjust(left = 0.01, top = 0.9, bottom = 0.1, right = 0.9)

    plot_image(TeNe["He1"]["Te"], vmin = 4000, vmax = 9000, label = r"T$_e$(He I)", fig = fig, ax = ax[0], title_size=font_size, cmap = "plasma")
    plot_image(TeNe["PJ"]["Te"], vmin = 4000, vmax = 9000, label = r"T$_e$(PJ)", fig = fig, ax = ax[1], title_size=font_size, cmap = "plasma")
    plot_image(TeNe["S3Cl3"]["Te"], vmin = 4000, vmax = 9000, label = r"T$_e$([S III])", fig = fig, ax = ax[2], title_size=font_size, cmap = "plasma")
    fig.savefig(figname)

def fig11_ab_o2r(figname = 'paper_figures/ab_o2r.pdf'):
    """Fig. 11: O++ abundance from the O II 4649+ and 4661 recombination lines.

    Maps of 12+log(O2+/H+) from both RLs (computed at Te=2000 K,
    Ne=1e4 cm-3) and the histograms of the two maps, to check their
    consistency.
    """
    labels = ['O2r_4649.13A', 'O2r_4661.63A']

    abunds = {}

    for label in labels:
        abund_dic = set_abunds(TeNe, obs, label = label, exclude_elem=('H', ),
                               Te_rec=2000, Ne_rec=10000,
                               tem_HI = None)
        abunds[label] = abund_dic[label]

    plt.rcParams['font.size'] = 12

    fig, ax = create_axis(3, n_columns=3)

    ax1 = ax[0]
    ax2 = ax[1]
    ax3 = ax[2]

    fig.subplots_adjust(left = 0.01, top = 0.9, bottom = 0.15, right = 0.95)

    plot_image(12+np.log10(abunds['O2r_4649.13A']), vmin = 9, vmax =  9.8,
                   label = ion_prefix('O2r_4649.13A') + '(' + get_label_str('O2r_4649.13A')+'+50' + ')',
                   cmap = 'plasma', fig = fig, ax = ax1)

    plot_image(12+np.log10(abunds['O2r_4661.63A']), vmin = 9, vmax =  9.8,
                    label = ion_prefix('O2r_4661.63A') + '(' + get_label_str('O2r_4661.63A') + ')',
                   cmap = 'plasma', fig = fig, ax = ax2)

    ax3.hist(12+np.log10(abunds['O2r_4649.13A']), bins = np.linspace(8.5,11,50), alpha = 0.5,
                label = get_label_str('O2r_4649.13A') + '+50')
    ax3.hist(12+np.log10(abunds['O2r_4661.63A']), bins = np.linspace(8.5,11,50), alpha = 0.5,
                label = get_label_str('O2r_4661.63A') )
    ax3.set_xlabel(r'12+log(O$^{2+}$/H$^+$)')
    ax3.legend()

    fig.savefig(figname)

def fig12_smooth_omega(figname = "paper_figures/2_omega_smooth.pdf"):
    """Fig. 12: maps of 1/omega and 1/(1-omega), and creation of the masked weights.

    Loads the convolved omega map (fraction of Hbeta emitted by the cold
    component), masks it where the Paschen-jump temperature is not lower
    than 0.85 x Te([S III]) (i.e. where no cold component is detected),
    and saves the resulting weight maps to ``omega_mask.joblib`` and
    ``omega_1_mask.joblib`` — side products required by fig13, fig14,
    fig15 and tab5.
    """
    omega = joblib.load(omega_filename).reshape(200,200)

    te_diff = TeNe['PJ']['Te'] - 0.85 * TeNe['S3Cl3']['Te']

    grid = te_diff.reshape(200,200)

    gaussian_2D_kernel = Gaussian2DKernel(3)

    result = interpolate_replace_nans(grid, gaussian_2D_kernel)
    N3r = obs.getIntens(returnObs=True)['N2r_5679.56A']
    mask_N3r = np.where(np.log10(N3r) > -18.5, 1, np.nan)
    mask = mask_N3r.reshape(200,200)
    te_smooth = result*mask

    plt.rcParams['font.size'] = 12

    fig, ax = create_axis(2, n_columns=2) #scale_y=3.8, scale_x=4.9
    fig.subplots_adjust(left = 0.05, top = 0.9, bottom = 0.1, right = 0.95)

    mask = np.where(te_smooth < 0, 1, np.nan)

    omega_mask = (omega*mask).reshape(40000)
    omega_1_mask = (1-omega).reshape(40000)
    joblib.dump(omega_mask, 'omega_mask.joblib')
    joblib.dump(omega_1_mask, 'omega_1_mask.joblib')

    plot_image(1/(omega*mask), fig = fig, ax = ax[0], vmin = 1, vmax = 13,
                title_size = 12, cmap = 'viridis', label = r'1/$\omega$')
    plot_image(1/(1-omega), fig = fig, ax = ax[1], vmin = 1, vmax = 1.3,
                title_size = 12, cmap = 'viridis', label = r'1/(1-$\omega$)')
    plt.savefig(figname)

def fig13_ion_ab_with_omega(figname = "paper_figures/ion_ab_with_omega.pdf"):
    """Fig. 13: grid of ionic-abundance maps with the omega weighting applied.

    For each of 17 lines, the abundance is computed with `set_abunds`
    using Te_rec=2000 K, Ne_rec=1e4 cm-3 and the omega/1-omega Hbeta
    weights, so RL abundances refer to the cold component and CEL
    abundances to the warm one. Requires the joblib files created by
    `fig12_smooth_omega`.
    """
    lines_dict={'C2r_6462.0A': (10.2-0.5, 10.75-0.5),
                'N1_5198A': (5, 7.1),
                'N2_6548A': (6.1, 8.2),
                'N2r_5679.56A': (10.1-0.5, 10.7-0.5),
                'O1_6300A': (5, 8),
                'O1r_7773+': (9.9-0.5, 10.4-0.5),
                'O2_7330A+': (6.75, 8.2),
                'O2r_4649.13A': (10.7-0.5, 11.15-0.5),
                'O3_4959A': (8.65, 9),
                'S2_6731A': (4.9, 6.9),
                'S3_9069A': (6.5, 7.2),
                'Cl3_5538A': (5, 5.5),
                'Cl4_8046A': (4.3, 5.1),
                'Ar3_7136A': (6.2, 6.7),
                'Ar4_4740A': (5.5, 6.5),
                'Ar5_7005A': (3.5, 4.7),
                'Kr4_5868A': (3.5, 4.35),
                }

    labels = list(lines_dict.keys())
    n_lines = len(labels)

    omega = joblib.load('omega_mask.joblib')
    omega_1 = joblib.load('omega_1_mask.joblib')


    plt.rcParams['font.size'] = 7
    fig, axs = create_axis(n_lines, n_columns=4, scale_x=2.5, scale_y=2)
    fig.subplots_adjust(left = 0.052, right = 0.952, top = 0.962, wspace = 0.25, hspace = 0.2)

    for index, label in enumerate(labels):
        abund_dic = set_abunds(TeNe, obs, label = label, exclude_elem=('H', ), Te_rec=2000, Ne_rec=10000, w = omega, w_1 = omega_1)
        line_ab = abund_dic[label]

        vmin, vmax = lines_dict[label]
        plot_image(np.log10(line_ab)+12, vmin = vmin, vmax = vmax,
                   label = ion_prefix(label) + '(' + get_label_str(label) + ')',
                   cmap = 'plasma', fig = fig,
                   ax = axs.ravel()[index], title_size=7)
    for ax in axs.ravel()[-3:]:
        ax.remove()
    fig.savefig(figname)

def fig14_O_adf_acf(figname = "paper_figures/O_adf_acf.pdf"):
    """Fig. 14: abundance discrepancy factor maps of O+ and O++, before/after omega.

    Left column: classical ADF (RL abundance / CEL abundance) for O+
    (O I 7773+ vs [O II] 7330+) and O++ (O II 4649+ vs [O III] 4959).
    Right column: the same ratios once each component is normalized to
    its own Hbeta fraction (called ACF, abundance contrast factor).
    """
    labels = ('O1r_7773+', 'O2_7330A+', 'O2r_4649.13A', 'O3_4959A')
    abund_ori = {}
    abund_corr = {}

    w = joblib.load('omega_mask.joblib')
    w_1 = joblib.load('omega_1_mask.joblib')

    for label in labels:
        abund_o = set_abunds(TeNe, obs, label = label, exclude_elem=('H', ), Te_rec=2000, Ne_rec=10000)
        abund_c = set_abunds(TeNe, obs, label = label, exclude_elem=('H', ), Te_rec=2000, Ne_rec=10000, w = w, w_1 = w_1)

        abund_ori[label] = abund_o[label]
        abund_corr[label] = abund_c[label]

    adf_op_ori = abund_ori['O1r_7773+'] / abund_ori['O2_7330A+']
    adf_opp_ori = abund_ori['O2r_4649.13A'] / abund_ori['O3_4959A']

    adf_op_corr = abund_corr['O1r_7773+'] / abund_corr['O2_7330A+']
    adf_opp_corr = abund_corr['O2r_4649.13A'] / abund_corr['O3_4959A']

    font_size = 12
    plt.rcParams['font.size'] = font_size


    fig, ax = create_axis(4, n_columns=2 ) #scale_y=3.8, scale_x=4.9
    #fig.subplots_adjust(left = 0.09, wspace = 0.1, top = 0.93, bottom = 0.13, right = 0.98, hspace = 0.5)

    plot_image(np.log10(adf_op_ori), vmax = 2.8, vmin = 0.25, label = r'log ADF(O$^{+}$)',
                fig = fig, ax = ax[0,0], cmap = "viridis", title_size=font_size)
    plot_image(np.log10(adf_opp_ori), vmax = 2, vmin = 0.25, label = r'log ADF(O$^{++}$)',
                fig = fig, ax = ax[1,0], cmap = "viridis", title_size=font_size)

    plot_image(np.log10(adf_op_corr), vmax = 2.8, vmin = 0.25, label = r'log ACF(O$^{+}$)',
                fig = fig, ax = ax[0,1], cmap = "viridis", title_size=font_size)
    plot_image(np.log10(adf_opp_corr), vmax = 2, vmin = 0.25, label = r'log ACF(O$^{++}$)',
                fig = fig, ax = ax[1,1], cmap = "viridis", title_size=font_size)

    fig.savefig(figname)

def fig15_hep_warm_cold(figname = "paper_figures/hep_warm_cold.pdf"):
    """Fig. 15: He+/H+ abundance maps of the warm and cold components.

    Solves the two-component decomposition of the He I 6678 and 7281
    lines: assuming (Te, Ne) = (8300 K, 3e3 cm-3) for the warm gas and
    (2000 K, 1e4 cm-3) for the cold gas, the pair of line equations is
    inverted analytically to yield He+/H+ in each component, normalizing
    with the omega/1-omega Hbeta weights. The integrated values (pixel 0)
    are printed.
    """
    pn.atomicData.setDataFile('he_i_rec_S96_caseB.hdf5')

    omega = joblib.load('omega_mask.joblib')
    omega_1 = joblib.load('omega_1_mask.joblib')

    he1 = pn.RecAtom('He', 1)
    he2 = pn.RecAtom('He', 2)
    e_72 = lambda T, den: he1.getEmissivity(tem=T, den=den, label='7281.0') / getHbEmissivity(T, den)
    e_66 = lambda T, den: he1.getEmissivity(tem=T, den=den, label='6678.0') / getHbEmissivity(T, den)
    Hb = obs.getLine(label='H1r_4861A').corrIntens
    I_66 = obs.getLine(label='He1r_6678A').corrIntens/Hb
    I_72 = obs.getLine(label='He1r_7281A').corrIntens/Hb
    T_w, T_c = 8300, 2000
    dens_w, dens_c = 3e3, 1e4
    e_66_w = e_66(T_w, dens_w)
    e_66_c = e_66(T_c, dens_c)
    e_72_w = e_72(T_w, dens_w)
    e_72_c = e_72(T_c, dens_c)

    Hep_w = (I_66 - I_72 * e_66_c/e_72_c) / (omega_1) / (e_66_w - e_72_w/e_72_c*e_66_c)
    Hep_c = (I_66 - I_72 * e_66_w/e_72_w) / omega / (e_66_c - e_72_c/e_72_w*e_66_w)
    plt.rcParams['font.size'] = 12

    fig, ax = create_axis(2, n_columns=2)
    fig.subplots_adjust(left = 0.05, top = 0.9, bottom = 0.1, right = 0.95)

    plot_image(12+np.log10(Hep_w), fig = fig, ax = ax[0], label = 'He+/H+ warm', vmin = 10.7, vmax = 11.7, title_size = 12)
    plot_image(12+np.log10(Hep_c), fig = fig, ax = ax[1], label = 'He+/H+ cold', vmin = 10.7, vmax = 11.7, title_size = 12)
    print(12+np.log10(Hep_w[0]), 12+np.log10(Hep_c[0]) )

    fig.savefig(figname)

def create_figures():
    """Produce every figure of the paper (fig1 ... fig15) in paper_figures/."""
    fig1_rgb_image()
    fig2_obs_fluxes()
    fig3_logFHb()
    fig4_cHb_dist()
    fig5_cHb()
    fig6_emis_NII_rec()
    fig7_Te_NII_recomb_corr()
    fig8_NII_OII_recomb_corr()
    fig9_TeNe_CELs()
    fig10_Te_HeI_PJ_SIII()
    fig11_ab_o2r()
    fig12_smooth_omega()
    fig13_ion_ab_with_omega()
    fig14_O_adf_acf()
    fig15_hep_warm_cold()

#---------------------------------------------TABLES-----------------------------------------------------------------------------------
def tab4_intTeNe(tex_filename='paper_tables/int_tene.tex'):
    """Table 4: Te and Ne of the integrated spectrum for every diagnostic.

    Runs the whole diagnostic machinery on the integrated observation
    (with its Monte Carlo realizations) and writes one LaTeX row per
    diagnostic: value (element 0) +/- standard deviation over the MC
    realizations. The Ar III/Ar IV weighted-average temperature is added,
    and the PJ/He I/average rows have no density entry.
    """
    labels_diags = {'N2S2':  "\\Te(\\forb{N}{ii} 5755/6584),  \\Ne(\\forb{S}{ii} 6716/6731)",

                    'S3S2':  '\\Te(\\forb{S}{iii} 6312/9069),  \\Ne(\\forb{S}{ii} 6716/6731)',
                    'S3Cl3': '\\Te(\\forb{S}{iii} 6312/9069),  \\Ne(\\forb{Cl}{iii} 5518/5538)',
                    'S3Ar4': '\\Te(\\forb{S}{iii} 6312/9069),  \\Ne(\\forb{Ar}{iv} 4711/4740)',

                    'Ar3S2': '\\Te(\\forb{Ar}{iii} 5192/7136),  \\Ne(\\forb{S}{ii} 6716/6731)',
                    'Ar3Cl3':'\\Te(\\forb{Ar}{iii} 5192/7136),  \\Ne(\\forb{Cl}{iii} 5518/5538)',

                    'Ar4S2': '\\Te(\\forb{Ar}{iv} 4740/7170),  \\Ne(\\forb{S}{ii} 6716/6731)',
                    'Ar4Cl3':'\\Te(\\forb{Ar}{iv} 4740/7170),  \\Ne(\\forb{Cl}{iii} 5518/5538)',
                    'Ar4Ar4':'\\Te(\\forb{Ar}{iv} 4740/7170),  \\Ne(\\forb{Ar}{iv} 4711/4740)',

                    'Ar3_Ar4_average': '\\Te(average \\forb{Ar}{iii}, \\forb{Ar}{iv})',

                    'PJ': 'PJ',

                    'He1': '{\hei} $\lambda$7281/$\lambda$6678'
    }
    ne_exeptions = ['Ar3_Ar4_average', 'PJ', 'He1']

    obs_int = get_obs_int()
    TeNe_int, _ = get_TeNe(obs = obs_int, plot = False)

    # Ar III / Ar IV weighted-average temperature (same weights as in
    # ionic_abund.select_TeNe).
    ar_pp = 3e-6
    ar_ppp = 1e-6
    w_ar = ar_pp / (ar_pp + ar_ppp)
    Te = w_ar * (TeNe_int['Ar3Cl3']['Te']) + (1-w_ar) * (TeNe_int['Ar4Ar4']['Te'])
    TeNe_int['Ar3_Ar4_average'] = {'Te':Te}

    with open(tex_filename, 'w') as f:
        for k in labels_diags:
            Te = TeNe_int[k]['Te']
            try:
                Ne = TeNe_int[k]['Ne']
            except:
                Ne = np.ones_like(Te)*np.nan
            if k in ne_exeptions:
                print2('{:61s} & {:5.0f} $\pm$ {:4.0f} & --- \\\\'.format(labels_diags[k], Te[0],
                                                                                        np.nanstd(Te)),
                    f)
            else:
                print2('{:61s} & {:5.0f} $\pm$ {:4.0f} & {:4.0f} $\pm$ {:4.0f} \\\\'.format(labels_diags[k], Te[0],
                                                                                        np.nanstd(Te),
                                                                                        Ne[0],
                                                                                        np.nanstd(Ne)),
                    f)
        print2('\hline',f)

def tab5_int_ion_ab(tex_filename='paper_tables/ionic_ab_7_recipes.tex'):
    """Table 5: integrated ionic abundances under 7 different recipes.

    For each of the 20 reference lines, the ionic abundance of the
    integrated spectrum is computed with 7 combinations of assumptions
    (columns of the table):

    1. CEL Te/Ne for everything (no special RL treatment);
    2. RLs at Te=2000 K, Ne=1e4 cm-3;
    3. as 2 plus H I emissivity at the Paschen-jump temperature;
    4. as 2 plus the omega/1-omega Hbeta weights (two components);
    5. as 4 with a 3-zone scheme using Te[Ar III] for the highest IPs;
    6. as 4 with the 3-zone scheme using the Ar III/Ar IV average;
    7. as 6 with the CEL H I emissivity fixed at Te=8300 K.

    Writes one LaTeX row per line: 12+log(X/H+) +/- MC standard
    deviation for the 7 recipes. Requires ``omega_mask.joblib``.
    """
    obs_int = get_obs_int()
    TeNe_int, _ = get_TeNe(obs = obs_int, plot = False)
    w_int = joblib.load('omega_mask.joblib')[0]
    w_1_int = joblib.load('omega_1_mask.joblib')[0]

    labels = ['He1r_5876A',
              'He2r_4686A',
              'C2r_6462.0A',
              'N1_5198A',
              'N2_6548A',
              'N2r_5679.56A',
              'O1_6300A',
              'O1r_7773+',
              'O2_7330A+',
              'O2r_4649.13A',
              'O2r_4661.63A',
              'O3_4959A',
              'S2_6731A',
              'S3_9069A',
              'Cl3_5538A',
              'Cl4_8046A',
              'Ar3_7136A',
              'Ar4_4740A',
              'Ar5_7005A',
              'Kr4_5868A',
            ]

    abund_r1 = {}
    abund_r2 = {}
    abund_r3 = {}
    abund_r4 = {}
    abund_r5 = {}
    abund_r6 = {}
    abund_r7 = {}

    for label in labels:
        ab_r1 = set_abunds(TeNe_int, obs_int, label = label, exclude_elem=('H', ), Te_rec=None, Ne_rec=None, w = None, w_1 = None,
                    tem_HI = None, tem_HI_RLs = None, consider_3zones = False)

        ab_r2 = set_abunds(TeNe_int, obs_int, label = label, exclude_elem=('H', ), Te_rec=2000, Ne_rec=10000, w = None,  w_1 = None,
                    tem_HI = None, tem_HI_RLs = None, consider_3zones = False)

        ab_r3 = set_abunds(TeNe_int, obs_int, label = label, exclude_elem=('H', ), Te_rec=2000, Ne_rec=10000, w = None,  w_1 = None,
                    tem_HI = TeNe_int['PJ']['Te'], tem_HI_RLs = TeNe_int['PJ']['Te'], consider_3zones = False)

        ab_r4 = set_abunds(TeNe_int, obs_int, label = label, exclude_elem=('H', ), Te_rec=2000, Ne_rec=10000, w = w_int,  w_1 = w_1_int,
                    tem_HI = None, tem_HI_RLs = None, consider_3zones = False)

        ab_r5 = set_abunds(TeNe_int, obs_int, label = label, exclude_elem=('H', ), Te_rec=2000, Ne_rec=10000, w = w_int,  w_1 = w_1_int,
                    tem_HI = None, tem_HI_RLs = None, consider_3zones = True, use_ar3 = True)

        ab_r6 = set_abunds(TeNe_int, obs_int, label = label, exclude_elem=('H', ), Te_rec=2000, Ne_rec=10000, w = w_int,  w_1 = w_1_int,
                    tem_HI = None, tem_HI_RLs = None, consider_3zones = True, use_ar3 = False)

        ab_r7 = set_abunds(TeNe_int, obs_int, label = label, exclude_elem=('H', ), Te_rec=2000, Ne_rec=10000, w = w_int,  w_1 = w_1_int,
                    tem_HI = 8300*np.ones_like(TeNe_int['N2S2']['Te']), tem_HI_RLs = None, consider_3zones = True, use_ar3 = False)

        abund_r1[label] = ab_r1[label]
        abund_r2[label] = ab_r2[label]
        abund_r3[label] = ab_r3[label]
        abund_r4[label] = ab_r4[label]
        abund_r5[label] = ab_r5[label]
        abund_r6[label] = ab_r6[label]
        abund_r7[label] = ab_r7[label]

    with open(tex_filename, 'w') as f:
        for line in obs_int.getSortedLines(crit='mass'):
            if (line.is_valid) and (line.elem != 'H') and (line.label in abund_r1):
                tit = get_label_str(line.label, latex = True)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ab_1_int, std_1_int = get_int_std(abund_r1, line.label)
                    ab_2_int, std_2_int = get_int_std(abund_r2, line.label)
                    ab_3_int, std_3_int = get_int_std(abund_r3, line.label)
                    ab_4_int, std_4_int = get_int_std(abund_r4, line.label)
                    ab_5_int, std_5_int = get_int_std(abund_r5, line.label)
                    ab_6_int, std_6_int = get_int_std(abund_r6, line.label)
                    ab_7_int, std_7_int = get_int_std(abund_r7, line.label)


                to_print = '{:15s}&{:5.2f}$\pm${:4.2f}&{:5.2f}$\pm${:4.2f}&{:5.2f}$\pm${:4.2f}&{:5.2f}$\pm${:4.2f}&{:5.2f}$\pm${:4.2f}&{:5.2f}$\pm${:4.2f}&{:5.2f}$\pm${:4.2f} \\\\'.format(tit,
                                                             ab_1_int, std_1_int,
                                                             ab_2_int, std_2_int,
                                                             ab_3_int, std_3_int,
                                                             ab_4_int, std_4_int,
                                                             ab_5_int, std_5_int,
                                                             ab_6_int, std_6_int,
                                                             ab_7_int, std_7_int)

                print2(to_print,f)
        print2('\hline',f)

def tab6_O_mass_frac(tex_filename='paper_tables/O_mass_frac.tex',
                     tem_w=8300, tem_c=2000, ne_w=3400, ne_c=10000):
    """Table 6: cold-to-warm oxygen mass ratios M^c/M^w for O+ and O++.

    The mass ratio follows from the RL/CEL line ratio of each ion
    (O II 4649+ vs [O III] 4959 for O++; O I 7773+ vs [O II] 7330+ for
    O+), the theoretical emissivities of each line at its component's
    (Te, Ne), and the density ratio of the two components.

    Parameters
    ----------
    tex_filename : str, optional
        Output LaTeX file.
    tem_w, tem_c : float, optional
        Temperatures [K] of the warm and cold components.
    ne_w, ne_c : float, optional
        Densities [cm-3] of the warm and cold components.
    """
    pn.log_.level=2
    pn.atomicData.setDataFile('o_ii_rec_SSB17-B-opt.hdf5')

    O2rS = pn.RecAtom('O', 2, case='B')
    emis_O2r = (O2rS.getEmissivity(tem=tem_c, den=ne_c, label='4649.13', product=False) +
                O2rS.getEmissivity(tem=tem_c, den=ne_c, label='4650.84', product=False))

    O3 = pn.Atom('O', 3)
    emis_O3 = O3.getEmissivity(tem=tem_w, den=ne_w, wave=4959, product=False)

    mass_ratio_Opp = (emis_O3 / emis_O2r *
                    (obs.getLine(label='O2r_4649.13A').corrIntens)[0] /
                    (obs.getLine(label='O3_4959A').corrIntens)[0] *
                    ne_w / ne_c )

    O1r = pn.RecAtom('O', 1, case='A')
    emis_O1r = O1r.getEmissivity(tem=tem_c, den=ne_c, label='7773+', product=False)

    O2 = pn.Atom('O', 2)
    emis_O2 = (O2.getEmissivity(tem=tem_w, den=ne_w, wave=7331, product=False) +
                O2.getEmissivity(tem=tem_w, den=ne_w, wave=7329, product=False))

    mass_ratio_Op = (emis_O2 / emis_O1r *
                    (obs.getLine(label='O1r_7773+').corrIntens)[0] /
                    (obs.getLine(label='O2_7330A+').corrIntens)[0] *
                    ne_w / ne_c )
    with open(tex_filename, 'w') as f:
        print2('$T_e^w$ [K] & {},{} \\\\'.format(str(tem_w)[:-3],str(tem_w)[-3:]), f)
        print2('$T_e^c$ [K] & {},{} \\\\'.format(str(tem_c)[:-3],str(tem_c)[-3:]), f)
        print2('$n_e^w$ [cm$^{{-3}}$]  & {},{} \\\\'.format(str(ne_w)[:-3],str(ne_w)[-3:]), f)
        print2('$n_e^c$ [cm$^{{-3}}$]  & {},{} \\\\'.format(str(ne_c)[:-3],str(ne_c)[-3:]), f)
        print2('M$^c$/M$^w$(O$^+$) & {:.1f} \\\\'.format(mass_ratio_Op), f)
        print2('M$^c$/M$^w$(O$^{{2+}}$) & {:.1f} \\\\'.format(mass_ratio_Opp), f)
        print2('\hline',f)

def tabA1_int_fluxes(tex_filename='paper_tables/int_fluxes.tex'):
    """Table A1: observed and dereddened integrated line fluxes.

    One LaTeX row per line of the integrated spectrum, sorted by
    wavelength: F(observed) and I(dereddened) on the scale Hbeta = 100,
    with uncertainties from the Monte Carlo realizations. The number of
    decimals adapts to the size of the error.
    """
    obs_int = get_obs_int()
    Hb = obs_int.getLine(label='H1r_4861A')
    with open(tex_filename, 'w') as f:
        for l in obs_int.getSortedLines(crit='wave'):
            I_obs = (l.obsIntens / Hb.obsIntens * 100)
            e_obs = (l.obsError)[0] * I_obs[0]
            I_cor = (l.corrIntens / Hb.corrIntens * 100)
            mask = np.isfinite(I_cor)
            e_cor = np.std(I_cor[mask])

            elem, spec, wl, forb  = get_label_str(l.label, split = True)
            if forb:
                line = r'\forb{%s}{%s}'%(elem, spec.lower())
            else:
                line = r'\perm{%s}{%s}'%(elem, spec.lower())
            if e_obs > 0.1:
                to_print = '{:11s} & {:7s} & {:8.1f} $\pm$ {:6.1f} & {:8.1f} $\pm$ {:6.1f} \\\\'.format(line, wl, I_obs[0], e_obs, I_cor[0], e_cor)
            elif e_cor > 0.01:
                to_print = '{:11s} & {:7s} & {:8.2f} $\pm$ {:6.2f} & {:8.2f} $\pm$ {:6.2f} \\\\'.format(line, wl, I_obs[0], e_obs, I_cor[0], e_cor)
            else:
                to_print = '{:11s} & {:7s} & {:8.3f} $\pm$ {:6.3f} & {:8.3f} $\pm$ {:6.3f} \\\\'.format(line, wl, I_obs[0], e_obs, I_cor[0], e_cor)

            print2(to_print, f)

        f.write(r'\hline')

def create_tables():
    """Produce every LaTeX table of the paper (tab4, tab5, tab6, tabA1) in paper_tables/."""
    tab4_intTeNe()
    tab5_int_ion_ab()
    tab6_O_mass_frac()
    tabA1_int_fluxes()

# -------------------ATOMIC DATA--------------------------------------------------------------
# PyNeb atomic data sets used in the paper (for reference).
atomic_data = {'H1': ['h_i_rec_SH95.hdf5'],
               'N2': ['n_ii_rec_P91.func', 'n_ii_rec_FSL11.func', 'n_ii_atom_FFT04.dat', 'n_ii_coll_T11.dat'],
               'O2': ['o_ii_rec_P91.func', 'o_ii_rec_SSB17-B-opt.hdf5'],
               'S2': ['s_ii_atom_RGJ19.dat', 's_ii_coll_TZ10.dat'],
               'S3': ['s_iii_atom_FFTI06.dat', 's_iii_coll_TG99.dat'],
               'Cl3': ['cl_iii_atom_RGJ19.dat', 'cl_iii_coll_BZ89.dat'],
               'Ar3': ['ar_iii_atom_MB09.dat', 'ar_iii_coll_MB09.dat'],
               'Ar4': ['ar_iv_atom_RGJ19.dat', 'ar_iv_coll_RB97.dat']
               }
