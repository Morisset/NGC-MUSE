"""Electron temperature and density (Te, Ne) diagnostics for NGC 6153.

This module derives Te/Ne maps from the emission-line maps prepared by
`observation.py`, using three families of diagnostics:

1. Collisionally excited line (CEL) ratio pairs, solved simultaneously
   with PyNeb's ``Diagnostics.getCrossTemDen``. The pairs are listed in
   ``DIAGNOSTICS_DICT`` (in `constants/observation_parameters.py`), e.g.
   ``'N2S2'`` = Te[N II 5755/6548] x Ne[S II 6731/6716]. To make the
   pixel-by-pixel (plus Monte Carlo) computation affordable, the inversion
   can be emulated by an artificial neural network (ai4neb); trained
   networks are cached on disk and reused (`add_gCTD`).
2. The Paschen jump of the H I continuum around 8100/8400 A, giving the
   temperature of the ionized gas emitting the recombination continuum
   (`add_T_PJ`).
3. The He I 7281/6678 line ratio, following Mendez-Delgado et al. (2021)
   (`add_T_He`).

The main entry point is `get_TeNe`, which returns a dictionary
``TeNe[diag] = {'Te': map, 'Ne': map}`` for every diagnostic, together
with the observation used. `plot_tem_den`, `plot_PJ` and `plot_He` are
quick-look display helpers.
"""

import pyneb as pn
import numpy as np
import os
import matplotlib.pyplot as plt
from observation import get_obs
from scipy.interpolate import interp1d
try:
    from ai4neb import manage_RM
    AI4NEB_INSTALLED = True
    pn.config.import_ai4neb()
except:
    AI4NEB_INSTALLED = False
from utils.plots import create_axis, plot_image
from constants.observation_parameters import *

def make_diags(obs, use_ANN):
    """Build the PyNeb Diagnostics object for an observation.

    Creates a ``pn.Diagnostics`` instance, registers every diagnostic that
    can be computed from the lines present in ``obs``, and adds the
    [Ar IV] 7170/4740 temperature-sensitive ratio which is not part of the
    PyNeb defaults.

    Parameters
    ----------
    obs : pn.Observation
        Observation holding the (dereddened) line maps.
    use_ANN : bool
        If True, configure the object so ``getCrossTemDen`` uses ai4neb
        ANN emulators (settings ``ANN_INST_KWARGS``/``ANN_INIT_KWARGS``
        from `constants/observation_parameters.py`).

    Returns
    -------
    pn.Diagnostics
        Ready-to-use diagnostics object.
    """
    diags = pn.Diagnostics()
    if use_ANN:
        diags.ANN_inst_kwargs = ANN_INST_KWARGS
        diags.ANN_init_kwargs = ANN_INIT_KWARGS
    diags.addDiagsFromObs(obs)
    diags.addDiag('[ArIV] 7170/4740', ('Ar4', 'L(7170)/L(4740)', 'RMS([E(4740), E(7170)])'))
    return diags

def add_gCTD(diags, obs, label, diag1, diag2, TeNe, use_ANN=True, limit_res=True, force=False, save=True, **kwargs):
    """Solve one Te x Ne diagnostic pair and store the result in ``TeNe``.

    Wraps ``diags.getCrossTemDen(diag1, diag2)``: the simultaneous
    solution of a temperature-sensitive and a density-sensitive line
    ratio. The first time it is run (or if ``force=True``) the ANN
    emulator is created, trained and stored in the ``new_ai4neb/`` folder;
    subsequent calls reload the already-trained network, which makes the
    map-scale computation fast.

    Parameters
    ----------
    diags : pn.Diagnostics
        Diagnostics object from `make_diags`.
    obs : pn.Observation
        Observation providing the line intensities.
    label : str
        Key under which the result is stored in ``TeNe`` (e.g. ``'N2S2'``);
        also used as the filename of the cached ANN.
    diag1, diag2 : str
        PyNeb names of the temperature and density diagnostics
        (e.g. ``'[NII] 5755/6548'`` and ``'[SII] 6731/6716'``).
    TeNe : dict
        Dictionary of results, updated in place and returned.
    use_ANN : bool, optional
        Use the ai4neb ANN emulator instead of the exact iterative solver.
    limit_res : bool, optional
        Passed to ``getCrossTemDen``; restrict results to the validity
        domain of the ANN training set.
    force : bool, optional
        Retrain the ANN even if a stored one exists.
    save : bool, optional
        Save a newly trained ANN to disk for later reuse.
    **kwargs
        Forwarded to ``getCrossTemDen``.

    Returns
    -------
    dict
        ``TeNe`` with a new entry ``label -> {'Te': ..., 'Ne': ...}``.
    """
    if not AI4NEB_INSTALLED and use_ANN:
        print('ai4neb not installed')
    if force:
        ANN = None
    else:
        ANN = manage_RM(RM_filename='new_ai4neb/'+label)
        if not ANN.model_read:
            ANN = None
    Te, Ne = diags.getCrossTemDen(diag1, diag2, obs=obs, use_ANN=use_ANN, ANN=ANN,
                                       limit_res=limit_res, **kwargs)
    if use_ANN and ANN is None and save:
        diags.ANN.save_RM(filename='new_ai4neb/'+label, save_train=True, save_test=True)
    TeNe[label] = {'Te': Te, 'Ne': Ne}
    return TeNe

def plot_tem_den(TeNe, key_diag, vmin_den = VMIN_DEN, vmax_den = VMAX_DEN, vmin_tem = VMIN_TEM, vmax_tem = VMAX_TEM, **kwargs):
    """Plot the Ne (log scale) and Te maps of one diagnostic side by side.

    Parameters
    ----------
    TeNe : dict
        Diagnostic results from `get_TeNe`.
    key_diag : str
        Key of the diagnostic to display (e.g. ``'N2S2'``); the panel
        titles are taken from ``DIAGNOSTICS_DICT``.
    vmin_den, vmax_den : float, optional
        Color range for log10(Ne) [cm-3].
    vmin_tem, vmax_tem : float, optional
        Color range for Te [K].
    **kwargs
        Forwarded to ``utils.plots.create_axis``.
    """
    fig, ax = create_axis(2, n_columns=2, **kwargs)
    ax_den = ax[0]
    ax_tem = ax[1]
    plot_image(np.log10(TeNe[key_diag]["Ne"]), fig = fig, ax = ax_den,
    label = 'Ne {}'.format(DIAGNOSTICS_DICT[key_diag][1]), vmin = vmin_den, vmax = vmax_den)
    plot_image(TeNe[key_diag]["Te"], fig = fig, ax = ax_tem,
    label = 'Te {}'.format(DIAGNOSTICS_DICT[key_diag][0]), vmin = vmin_tem, vmax = vmax_tem)

def add_T_PJ(TeNe, obs, den, Hep, Hepp):
    """Compute the Paschen-jump temperature map and store it as TeNe['PJ'].

    The H I continuum jump across the Paschen limit is quantified by the
    observed ratio PJ = (C_8100 - C_8400) / I(H I 9229), where C_8100 and
    C_8400 are pseudo-continuum measurements on the blue and red side of
    the jump. A theoretical PJ(Te) curve is tabulated with PyNeb's
    ``Continuum.BJ_HI`` (for fixed density and He+ /He++ abundances) and
    inverted by interpolation to convert the observed ratio into a
    temperature.

    Parameters
    ----------
    TeNe : dict
        Diagnostic results, updated in place with key ``'PJ'``.
    obs : pn.Observation
        Observation providing ``H1r_8100A``, ``H1r_8400A`` (continua) and
        ``H1r_9229A``.
    den : float
        Adopted electron density [cm-3] for the theoretical curve.
    Hep, Hepp : float
        Adopted He+/H+ and He++/H+ abundances, which set the He
        contribution to the continuum.

    Returns
    -------
    dict
        ``TeNe`` with ``TeNe['PJ']['Te']`` added (no density is derived).
    """
    cont = pn.Continuum()
    # Tabulate the theoretical Paschen jump on a Te grid, then invert it.
    tab_tem = np.linspace(500, 30000, 100)
    tab_den = np.ones_like(tab_tem) * den
    tab_Hep = np.ones_like(tab_tem) * Hep
    tab_Hepp = np.ones_like(tab_tem) * Hepp

    tab_PJ =  cont.BJ_HI(tab_tem, tab_den, tab_Hep, tab_Hepp, wl_bbj = 8100, wl_abj = 8400, HI_label='9_3')
    tem_inter = interp1d(tab_PJ, tab_tem, bounds_error=False)

    TeNe['PJ'] = {}
    C_8100 = obs.getIntens()['H1r_8100A']
    C_8400 = obs.getIntens()['H1r_8400A']
    HI = obs.getIntens()['H1r_9229A']
    with np.errstate(divide='ignore', invalid='ignore'):
        PJ_HI = (C_8100 - C_8400) /  HI
    TeNe['PJ']['Te'] = tem_inter(PJ_HI)
    return TeNe

def add_T_He(TeNe, obs):
    """Compute the He I temperature map and store it as TeNe['He1'].

    Uses the density-dependent linear fit of Mendez-Delgado et al. (2021,
    MNRAS 502, 1703): Te(He I) = alpha(Ne) * R_He + beta(Ne), where
    R_He = I(He I 7281)/I(He I 6678). The alpha and beta coefficients are
    tabulated as a function of density and interpolated at the
    ``TeNe['N2S2']`` density of each pixel, so the [S II] density map must
    have been computed first.

    Parameters
    ----------
    TeNe : dict
        Diagnostic results, updated in place with key ``'He1'``; must
        already contain ``TeNe['N2S2']['Ne']``.
    obs : pn.Observation
        Observation providing ``He1r_7281A`` and ``He1r_6678A``.

    Returns
    -------
    dict
        ``TeNe`` with ``TeNe['He1']['Te']`` added.

    References
    ----------
    Mendez-Delgado et al. (2021), MNRAS 502, 1703.
    """
    TeNe['He1'] = {}
    # alpha(Ne) and beta(Ne) coefficients from Mendez-Delgado et al. (2021).
    dens = np.asarray((100,   500,  1000,  2000,  3000,  4000,  5000,  6000,  7000,
                       8000,  9000, 10000, 12000, 15000, 20000, 25000, 30000, 40000,
                       45000, 50000))
    alpha = np.asarray((92984, 81830, 77896, 69126, 65040, 62517, 60744, 59402, 58334,
                        57456, 56715, 56077, 55637, 55087, 54364, 53796, 53329, 52591,
                        52289, 52019))
    beta = np.asarray((-7455, -6031, -5527, -4378, -3851, -3529, -3305 ,-3137, -3004,
                       -2895, -2804, -2726, -2676, -2611, -2523, -2452, -2392, -2297,
                       -2257, -2222))
    alpha_int = interp1d(dens, alpha, bounds_error=False)
    beta_int = interp1d(dens, beta, bounds_error=False)

    alphas = alpha_int(TeNe['N2S2']['Ne'])
    betas = beta_int(TeNe['N2S2']['Ne'])
    with np.errstate(divide='ignore', invalid='ignore'):
        R_He = obs.getIntens()['He1r_7281A'] / obs.getIntens()['He1r_6678A']

    Te = alphas * R_He + betas
    Te[np.isinf(Te)] = np.nan

    TeNe['He1']['Te'] = Te
    return TeNe

def get_TeNe(obs = None, use_ANN = USE_ANN, plot = PLOT_DIAGNOSTICS):
    """Compute every Te/Ne diagnostic map: the main driver of this module.

    Steps:

    1. Load (or reuse) the observation via ``observation.get_obs``.
    2. Solve each CEL diagnostic pair of ``DIAGNOSTICS_DICT`` with
       `add_gCTD` (ANN-accelerated ``getCrossTemDen``).
    3. Add the Paschen-jump temperature (`add_T_PJ`, constants ``DEN_PJ``,
       ``Hep_PJ``, ``Hepp_PJ``) and the He I temperature (`add_T_He`).

    Parameters
    ----------
    obs : pn.Observation, optional
        Observation to use; if None it is built with ``get_obs()``.
    use_ANN : bool, optional
        Use ai4neb ANN emulators (default from configuration).
    plot : bool, optional
        If True, display the Te/Ne maps of every diagnostic.

    Returns
    -------
    TeNe : dict
        ``TeNe[diag] = {'Te': map, 'Ne': map}`` for each key of
        ``DIAGNOSTICS_DICT``, plus ``'PJ'`` and ``'He1'`` (Te only).
    obs : pn.Observation
        The observation actually used (handy when it was created here).
    """
    if obs is None:
        obs = get_obs()
    diags = make_diags(obs, use_ANN = use_ANN)
    TeNe = {}

    if not os.path.exists(ANN_FOLDER):
        os.mkdir(ANN_FOLDER)

    pn.log_.timer('Starting', quiet=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        for key in DIAGNOSTICS_DICT.keys():
            diagnostic = DIAGNOSTICS_DICT[key]
            TeNe = add_gCTD(diags, obs, key, diagnostic[0], diagnostic[1], TeNe, use_ANN = use_ANN)
    pn.log_.timer('ANN getCrossTemDen done')

    if plot:
        for key in DIAGNOSTICS_DICT.keys():
            plot_tem_den(TeNe, key)

    TeNe = add_T_PJ(TeNe, obs, DEN_PJ, Hep_PJ, Hepp_PJ)
    TeNe = add_T_He(TeNe, obs)

    return TeNe, obs

def plot_PJ(TeNe, obs, **kwargs):
    """Plot the Paschen-jump temperature map (computing it if missing).

    Parameters
    ----------
    TeNe : dict
        Diagnostic results; ``'PJ'`` is added via `add_T_PJ` if absent.
    obs : pn.Observation
        Observation used to compute the map when needed.
    **kwargs
        Forwarded to ``utils.plots.plot_image`` (e.g. ``vmin``, ``vmax``).

    Returns
    -------
    dict
        The (possibly updated) ``TeNe`` dictionary.
    """
    if "PJ" not in TeNe.keys():
        TeNe = add_T_PJ(TeNe, obs, DEN_PJ, Hep_PJ, Hepp_PJ)
    plot_image(TeNe["PJ"]["Te"], **kwargs)
    return TeNe

def plot_He(TeNe, obs, **kwargs):
    """Plot the He I temperature map (computing it if missing).

    Parameters
    ----------
    TeNe : dict
        Diagnostic results; ``'He1'`` is added via `add_T_He` if absent.
    obs : pn.Observation
        Observation used to compute the map when needed.
    **kwargs
        Forwarded to ``utils.plots.plot_image``.

    Returns
    -------
    dict
        The (possibly updated) ``TeNe`` dictionary.
    """
    if "He1" not in TeNe.keys():
        TeNe = add_T_He(TeNe, obs)
    plot_image(TeNe["He1"]["Te"], **kwargs)
    return TeNe
