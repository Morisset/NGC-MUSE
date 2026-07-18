"""Miscellaneous helpers: line-label formatting and map extraction.

PyNeb line labels look like ``O3_5007A`` (CEL), ``O2r_4649.13A``
(recombination line, note the trailing ``r`` on the atom) or
``O2_7330A+`` (blend, trailing ``+``). The functions here convert such
labels into human-readable or LaTeX strings (`get_label_str`,
`get_label_str_ori`), test whether a label is a recombination line
(`check_recomb`), and reshape the flat intensity arrays of a PyNeb
Observation into 2D images, optionally collapsing the Monte Carlo axis
(`get_image`).
"""

from pyneb.utils.misc import parseAtom, int_to_roman
import numpy as np

def get_label_str(label, split = False, latex = False):
    """Convert a PyNeb line label into a readable (or LaTeX) line name.

    Examples: ``'O3_5007A'`` -> ``'[O III] 5007'``;
    ``'O2r_4649.13A'`` -> ``'O II 4649'``;
    ``'O2_7330A+'`` -> ``'[O II] 7330+'``.
    With ``latex=True``, the ion is wrapped in the paper's LaTeX macros
    ``\\forb{elem}{spec}`` (forbidden/CEL) or ``\\perm{elem}{spec}``
    (permitted/recombination).

    Parameters
    ----------
    label : str
        PyNeb line label (``atom_wavelength``, e.g. ``'N2r_5679.56A'``).
    split : bool, optional
        If True, return the pieces instead of a formatted string.
    latex : bool, optional
        If True, use the LaTeX macros for the ion name.

    Returns
    -------
    str or tuple
        The formatted line name, or (if ``split``) the tuple
        ``(elem, spec_roman, wavelength_str, is_forbidden)``.
    """
    atom = label.split('_')[0]
    wl = label.split('_')[1]
    if atom[-1] == 'r':
        atom = '{}'.format(atom[0:-1])
        forb = False
    else:
        forb = True

    elem, spec = parseAtom(atom)

    spec_rom = int_to_roman(int(spec))

    if forb:
        if latex:
            line = '\\forb{%s}{%s}'%(elem, spec_rom.lower())
        else:
            line = '[{} {}]'.format(elem, spec_rom)
    else:
        if latex:
            line = '\\perm{%s}{%s}'%(elem, spec_rom.lower())
        else:
            line = '{} {}'.format(elem, spec_rom)

    if wl[-1] == '+':
        wl = wl[0:-1]
        blend_str = '+'
    else:
        blend_str=''
    if wl[-1] == 'A':
        wl = wl[0:-1]
    wl = wl.split('.')[0]
    if split:
        return elem, spec_rom, wl + blend_str, forb
    else:
        return '{} {}{}'.format(line, wl, blend_str)

def get_label_str_ori(label, split = False):
    """Original (pre-LaTeX) version of `get_label_str`, kept for reference.

    Same conversion of a PyNeb label into ``'[O III] 5007'``-style
    strings, without the LaTeX option.

    Parameters
    ----------
    label : str
        PyNeb line label.
    split : bool, optional
        If True, return ``(ion_str, wavelength_str)`` instead of a single
        formatted string.

    Returns
    -------
    str or tuple
        The formatted line name, or its pieces if ``split``.
    """
    lab1 = label.split('_')[0]
    lab3 = label.split('_')[1]
    if lab1[-1] == 'r':
        lab1 = '{}'.format(lab1[0:-1])
        forb = False
    else:
        forb = True

    lab1, lab2 = parseAtom(lab1)

    lab2 = int_to_roman(int(lab2))

    if forb:
        lab1 = '[{} {}]'.format(lab1, lab2)
    else:
        lab1 = '{} {}'.format(lab1, lab2)

    if lab3[-1] == '+':
        lab3 = lab3[0:-1]
        blend_str = '+'
    else:
        blend_str=''
    if lab3[-1] == 'A':
        lab3 = lab3[0:-1]
    lab3 = lab3.split('.')[0]
    if split:
        return lab1, lab3 + blend_str
    else:
        return '{} {}{}'.format(lab1, lab3, blend_str)

def check_recomb(label:str):
    """Return True if the label is a recombination line (atom ends with 'r')."""
    ion = label.split("_")[0]
    return ion[-1] == "r"


def get_image(obs, data=None, label=None, N_MC = None, type_='median', returnObs=True):
    """Extract a 2D image from a PyNeb Observation, collapsing MC if needed.

    Reshapes the flat intensity array of a line (or an arbitrary ``data``
    array — useful when only the reshape is wanted) to the 2D shape of
    the observation. When Monte Carlo realizations are present
    (``N_MC`` not None), the MC axis is collapsed according to ``type_``.
    If ``label`` is a 2-tuple of labels, the ratio of the two
    corresponding images is returned.

    Parameters
    ----------
    obs : pn.Observation
        Observation providing the data and the target shape
        (``obs.data_shape``).
    data : np.ndarray, optional
        Raw array to reshape instead of reading a line from ``obs``.
    label : str or tuple of str, optional
        Line label to extract, or a tuple ``(label1, label2)`` to get the
        image ratio label1/label2.
    N_MC : int, optional
        If not None, the data include Monte Carlo realizations and the
        third axis is collapsed with ``type_``.
    type_ : str, optional
        'median': median of the MC and observations.
        'mean': mean of the MC and observations.
        'std': std of the MC and observations.
        'orig': the original (first) realization, no collapsing.
    returnObs : bool, optional
        True returns the observed intensities;
        False returns the extinction-corrected ones.

    Returns
    -------
    np.ndarray
        The 2D image (or 3D array when ``N_MC`` is None and the data
        include the MC axis).
    """
    if label is not None:
        if isinstance(label, tuple):
            with np.errstate(divide='ignore', invalid='ignore'):
                to_return = (get_image(label=label[0], type_=type_ ,returnObs=returnObs) /
                             get_image(label=label[1], type_=type_, returnObs=returnObs))
            return to_return
        d2return = obs.getIntens(returnObs=returnObs)[label]
    else:
        d2return = data # data in, data out: used when only the reshape is wanted
    if N_MC is None:
        return d2return.reshape(obs.data_shape)
    else:
        if type_ == 'median':
            return np.nanmedian(d2return.reshape(obs.data_shape), 2)
        if type_ == 'mean':
            return np.nanmean(d2return.reshape(obs.data_shape), 2)
        elif type_ == 'std':
            return np.nanstd(d2return.reshape(obs.data_shape), 2)
        elif type_ == 'orig':
            return d2return.reshape(obs.data_shape)[:,:,0]
        else:
            print('type_ must be one of median, mean, std, or orig')
