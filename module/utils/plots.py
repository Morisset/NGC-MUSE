"""Shared plotting utilities for the NGC 6153 MUSE maps.

Helpers used across the pipeline to display 2D emission-line maps and
derived quantities (Te, Ne, ionic abundances):

- `create_axis`    : build a grid of matplotlib axes sized for N maps.
- `plot_image`     : display one 2D map (handles flattened 1D inputs and
  Monte Carlo cubes), with optional WCS ticks and colorbar.
- `plot_ionic_ab`  : grid of log10 ionic-abundance maps.
- `plot_fluxes`    : grids of log10 flux maps, recombination lines and
  collisionally excited lines in separate figures.
- `plot_ann_test`  : predicted-vs-true scatter plots to validate an
  ai4neb ANN emulator of a Te/Ne diagnostic.

The world coordinate system (``WCS``) and Monte Carlo size (``N_MC``) come
from `constants/observation_parameters.py`.
"""

import numpy as np
import matplotlib.pyplot as plt
from constants.observation_parameters import WCS, N_MC
from utils.misc import check_recomb, get_image, get_label_str

def create_axis(n_line_maps, suptitle:str = "", n_columns = 3, scale_x = 5, scale_y = 4,
                size = 30, top = 0.95, bottom = 0.05, hspace = 0.3, wspace = 0.2, show_ticks=False):
    """Create a figure with a grid of axes able to hold ``n_line_maps`` maps.

    The number of rows is derived from ``n_line_maps`` and ``n_columns``;
    the figure size scales with the grid dimensions.

    Parameters
    ----------
    n_line_maps : int
        Number of maps (subplots) to accommodate.
    suptitle : str, optional
        Figure-level title.
    n_columns : int, optional
        Number of columns in the grid (default 3).
    scale_x, scale_y : float, optional
        Width/height [inches] allotted to each subplot.
    size : int, optional
        Font size of the suptitle.
    top, bottom, hspace, wspace : float, optional
        Passed to ``fig.subplots_adjust`` to control margins and spacing.
    show_ticks : bool, optional
        If True, create the axes with the NGC 6153 WCS projection so that
        RA/Dec ticks can be shown.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : np.ndarray of matplotlib.axes.Axes
        The grid of axes (2D array as returned by ``plt.subplots``).
    """
    n_rows = n_line_maps // n_columns
    residue = n_line_maps % n_columns
    if residue > 0:
        n_rows += 1
    figsize = (int(scale_x*n_columns), int(scale_y*n_rows))
    if show_ticks:
        fig, axs = plt.subplots(n_rows, n_columns, figsize = figsize, subplot_kw={'projection': WCS})
    else:
        fig, axs = plt.subplots(n_rows, n_columns, figsize = figsize)
    fig.suptitle(suptitle, size = size)
    fig.subplots_adjust(top = top, bottom = bottom, hspace = hspace, wspace = wspace)
    return fig, axs

def plot_image(image:np.ndarray, label:str="", fig = None, ax = None, show_ticks = False,
               create_colorbar = True, cmap = 'viridis', title_size = 12, **kwargs):
    """Display a single 2D map of the nebula.

    Accepts either a 2D image or the flattened 1D arrays used internally
    by PyNeb Observations: a 40000-element vector is reshaped to the
    200x200 MUSE field, and a longer vector is assumed to be a Monte Carlo
    cube (200, 200, N_MC+1) of which only the first (original) realization
    is shown.

    Parameters
    ----------
    image : np.ndarray
        Map to display (2D image or flattened 1D vector, see above).
    label : str, optional
        Title of the panel (e.g. the line name).
    fig, ax : matplotlib Figure/Axes, optional
        Where to draw; a new single-panel figure is created if ``ax`` is
        None.
    show_ticks : bool, optional
        If True, label the axes with RA/Dec (requires WCS axes, see
        `create_axis`); otherwise draw a faint reference grid without tick
        labels.
    create_colorbar : bool, optional
        Add a colorbar to the right of the panel.
    cmap : str, optional
        Matplotlib colormap name (default ``'viridis'``).
    title_size : int, optional
        Font size of the panel title.
    **kwargs
        Forwarded to ``ax.imshow`` (e.g. ``vmin``, ``vmax``).
    """
    if np.ndim(image) == 1:
        # Flattened inputs: 200x200 map, or 200x200x(N_MC+1) MC cube of
        # which only the original (index 0) realization is displayed.
        if np.size(image) == 40000:
            image_plot = image.reshape(200,200)
        else:
            image_plot = image.reshape(200,200,N_MC+1)[:,:,0]
    else:
        image_plot = image
    if ax is None:
        fig, ax = create_axis(1, n_columns=1)

    im = ax.imshow(image_plot, cmap = cmap, origin = "lower", **kwargs)
    if show_ticks:
        ax.grid(color='gray', linestyle='--', linewidth=1)
    else:
        ax.grid(color='gray', linestyle='--', linewidth=1, alpha =0.5)
        ax.set_xticks([18, 72, 126, 180])
        ax.set_xticklabels(['', '', '', ''])
        ax.set_yticks([25, 75, 125, 175])
        ax.set_yticklabels(['', '', '', ''])
    if create_colorbar:
        cax = fig.add_axes([ax.get_position().x1,
                    ax.get_position().y0,
                    0.01,
                    ax.get_position().height])
    ax.set_title(label, size = title_size)
    if show_ticks:
        ax.set_xlabel("Right Ascension")
        ax.set_ylabel("Declination")
    if create_colorbar:
        fig.colorbar(im, cax = cax)

def plot_ionic_ab(abund_dic, abund_keys:list=None, dex_range = 0.5):
    """Plot a grid of ionic-abundance maps in log scale.

    Each map is shown as log10(X^i+/H+) with the color range centred on
    the median of the map and spanning +/- ``dex_range`` dex.

    Parameters
    ----------
    abund_dic : dict
        Mapping ``line label -> abundance map`` as returned by
        ``ionic_abund.set_abunds``.
    abund_keys : list of str, optional
        Subset (and order) of labels to plot; all keys of ``abund_dic``
        if None.
    dex_range : float, optional
        Half-width of the color range in dex around the median.
    """
    if abund_keys is None:
        n_maps = len(abund_dic)
        lines = list(abund_dic.keys())
    else:
        n_maps = len(abund_keys)
        lines = abund_keys
    fig, axs = create_axis(n_maps)
    for index, line in enumerate(lines):
        try:
            ab = abund_dic[line]
            log_median = np.log10(np.nanmedian(ab))
            vmin = log_median - dex_range
            vmax = log_median + dex_range
            plot_image(np.log10(ab), vmin = vmin, vmax = vmax, label = line, fig = fig, ax = axs.ravel()[index])
        except:
            print(line)


def plot_fluxes(obs, returnObs=True, **kwargs):
    """Plot the log10 flux maps of every line in an observation.

    Two figures are produced: one gathering the recombination lines and
    one for the collisionally excited lines (classification based on the
    line label, see ``utils.misc.check_recomb``).

    Parameters
    ----------
    obs : pn.Observation
        Observation holding the line maps.
    returnObs : bool, optional
        If True plot the observed (uncorrected) intensities; if False the
        extinction-corrected ones.
    **kwargs
        Forwarded to `plot_image` (e.g. ``vmin``, ``vmax``, ``cmap``).
    """
    lines_labels = [obs.getSortedLines()[index].label for index in range(len(obs.getSortedLines()))]
    n_recomb = sum([check_recomb(label) for label in lines_labels])
    n_coll = len(lines_labels) - n_recomb

    fig_r, axs_r = create_axis(n_recomb, suptitle = "Recombination lines")
    index_r = 0

    fig_c, axs_c = create_axis(n_coll, suptitle = "Collisionaly excited lines")
    index_c = 0

    for label in lines_labels:
        if check_recomb(label):
            image = get_image(obs = obs, label=label, type_='orig', returnObs=returnObs)
            plot_image(np.log10(image), get_label_str(label), fig = fig_r, ax = axs_r.ravel()[index_r], **kwargs)
            index_r += 1
        else:
            image = get_image(obs = obs, label=label, type_='orig', returnObs=returnObs)
            plot_image(np.log10(image), get_label_str(label), fig = fig_c, ax = axs_c.ravel()[index_c], **kwargs)
            index_c += 1

def plot_ann_test(pred, ann, tem_diag:str = 'OII 4649/4089', den_diag:str = 'O II 4649/mult V1'):
    """Validate an ai4neb ANN emulator: predicted vs true Te and Ne.

    Draws two scatter panels comparing the ANN predictions with the test
    set of the trained model: Te (left, colored by log Ne) and log Ne
    (right, colored by Te), each with the 1:1 line.

    Parameters
    ----------
    pred : np.ndarray
        ANN predictions on the test set, shape (N, 2): column 0 is
        Te/1e4 K, column 1 is log10(Ne).
    ann : ai4neb.manage_RM
        Trained model, whose ``y_test`` attribute holds the true values in
        the same convention.
    tem_diag, den_diag : str, optional
        Names of the diagnostics, used in the panel titles.
    """
    fig, ax = plt.subplots(1,2, figsize = (17,7))
    fontsize = 14
    lim_min = (np.min(pred[:,0]))*1e4 - 100
    lim_max = (np.max(pred[:,0]))*1e4 + 100
    cb0 = ax[0].scatter(ann.y_test[:,0]*1e4, pred[:,0]*1e4, c = ann.y_test[:,1], cmap = 'jet')
    cbar1 = fig.colorbar(cb0, ax = ax[0])
    cbar1.ax.tick_params(labelsize=fontsize)
    cbar1.set_label(label='log (Ne) [cm^-3]', size=fontsize)
    ax[0].set_title('Temperature diagnostic {}'.format(tem_diag), size = fontsize, weight='bold')
    ax[0].tick_params(axis='y', labelsize=fontsize)
    ax[0].tick_params(axis='x', labelsize=fontsize, rotation= 45)
    ax[0].plot((lim_min, lim_max),(lim_min, lim_max), color ='k')
    ax[0].set_xlim(lim_min, lim_max)
    ax[0].set_ylim(lim_min, lim_max)
    ax[0].set_xlabel('Te [K] (Real)', size = fontsize)
    ax[0].set_ylabel('Te [K] (Predicted)', size = fontsize)

    lim_min = np.min(pred[:,1])-0.1
    lim_max = np.max(pred[:,1])+0.1
    cb1 = ax[1].scatter(ann.y_test[:,1], pred[:,1], c = 1e4 * ann.y_test[:,0], cmap = 'jet')
    cbar2 = fig.colorbar(cb1, ax = ax[1], label = 'Te [K]')
    cbar2.ax.tick_params(labelsize=fontsize)
    cbar2.set_label(label='Te [K]', size=fontsize)
    ax[1].tick_params(axis='both', labelsize=fontsize)
    ax[1].set_title('Density diagnostic {}'.format(den_diag), size = fontsize, weight='bold')
    ax[1].plot((lim_min, lim_max),(lim_min, lim_max), color ='k')
    ax[1].set_xlim(lim_min, lim_max)
    ax[1].set_ylim(lim_min, lim_max)
    ax[1].set_xlabel('log(Ne [cm^-3]) (Real)', size = fontsize)
    ax[1].set_ylabel('log(Ne [cm^-3]) (Predicted)', size = fontsize);
