"""Ionic abundance determination from emission-line intensity maps.

This module converts the dereddened line intensities stored in a PyNeb
``Observation`` (built by `observation.py` or `obs_int.py`) into ionic
abundances (X^i+/H+), using the electron temperature (Te) and density (Ne)
maps computed in `diagnostics.py`.

The general logic is:

1. For each observed emission line, build the appropriate PyNeb atom model:
   a ``pn.RecAtom`` for recombination lines (labels ending in ``r``, e.g.
   ``O2r_4649.13A``) or a ``pn.Atom`` for collisionally excited lines (CELs,
   e.g. ``O3_5007A``). See `create_atom`.
2. Choose which Te/Ne diagnostic applies to the emitting ion, based on its
   ionization potential (IP): low-ionization ions use the [N II]/[S II]
   diagnostic, higher-ionization ions use [S III]/[Cl III] (and optionally
   Ar-based diagnostics in a 3-zone scheme). See `select_TeNe`.
3. Call ``atom.getIonAbundance`` with the line intensity relative to Hbeta.

Two drivers are provided: `set_abunds_ori` (original, simple 2-zone scheme)
and `set_abunds` (extended version supporting the warm/cold two-component
model of the nebula through the Hbeta weight maps ``w`` and ``w_1``, used to
derive abundances in the H-poor cold clumps and the "normal" warm gas
separately).

All Te/Ne inputs are the ``TeNe`` dictionaries returned by
``diagnostics.get_TeNe`` (keys like ``'N2S2'``, ``'S3Cl3'``, ``'He1'``,
``'PJ'``, each holding ``'Te'`` and ``'Ne'`` arrays).
"""

import pyneb as pn
import numpy as np

def create_rec_atom(line):
    """Build the PyNeb recombination atom associated to an observed line.

    Parameters
    ----------
    line : pn.EmissionLine
        Observed line (from ``obs.getSortedLines()``). Its ``atom``
        attribute is a recombination label such as ``'O2r'`` or ``'N2r'``.

    Returns
    -------
    atom : pn.RecAtom
        Recombination atom model, with ``extrapolate=True`` so emissivities
        can be evaluated slightly outside the tabulated Te/Ne grid.
    IP : float
        Ionization potential [eV] needed to *create* the emitting ion
        (e.g. for O II recombination lines the emitting ion is O++, so the
        IP of O+ is returned). Used by `select_TeNe` to pick the Te/Ne zone.

    Notes
    -----
    Case A recombination is used for C II and O I lines, Case B for all
    other ions (the standard choice for nebulae, where the Lyman lines are
    optically thick).
    """
    if line.atom in ('C2r', 'O1r'):
        case = 'A'
    else:
        case = 'B'
    atom = pn.RecAtom(line.elem, line.spec, case=case, extrapolate=True)
    IP = pn.utils.physics.IP[atom.elem][atom.spec-1]
    return atom, IP

def create_col_atom(line):
    """Build the PyNeb collisionally excited (CEL) atom for an observed line.

    Parameters
    ----------
    line : pn.EmissionLine
        Observed line (e.g. ``[O III] 5007``); ``line.atom`` is a CEL label
        such as ``'O3'``.

    Returns
    -------
    atom : pn.Atom
        Collisionally excited line atom model.
    IP : float
        Ionization potential [eV] needed to create the emitting ion
        (0 for neutral emitters such as [O I], for which no previous
        ionization stage exists).
    """
    atom = pn.Atom(line.elem, line.spec)
    if atom.spec-2 < 0:
        IP = 0.
    else:
        IP = pn.utils.physics.IP[atom.elem][atom.spec-2]
    return atom, IP

def create_atom(line):
    """Build the PyNeb atom (recombination or CEL) for an observed line.

    Dispatches to `create_rec_atom` if the atom label ends with ``'r'``
    (PyNeb convention for recombination lines, e.g. ``'O2r'``), otherwise
    to `create_col_atom`.

    Parameters
    ----------
    line : pn.EmissionLine
        Observed line.

    Returns
    -------
    atom : pn.Atom or pn.RecAtom
        Atom model to be used with ``getIonAbundance``.
    IP : float
        Ionization potential [eV] used to select the Te/Ne zone.
    rec_line : bool
        True if the line is a recombination line.
    """
    if line.atom[-1] == 'r':
        rec_line = True
        atom, IP = create_rec_atom(line)
    else:
        rec_line = False
        atom, IP = create_col_atom(line)
    return atom, IP, rec_line

def select_TeNe(TeNe, IP, consider_3zones = False, use_ar3 = False):
    """Select the Te and Ne maps appropriate for an ion, given its IP.

    Implements the ionization-stratification scheme of the paper:

    - IP < 17 eV (low-ionization zone, e.g. N+, S+, O+):
      Te[N II 5755/6548] and Ne[S II 6731/6716] (diagnostic ``'N2S2'``).
    - IP >= 17 eV (medium/high-ionization zone, e.g. O++, Ne++):
      Te[S III 6312/9069] and Ne[Cl III 5538/5518] (diagnostic ``'S3Cl3'``).
    - Optionally, with ``consider_3zones=True`` and IP >= 35 eV (highest
      ionization ions, e.g. Ar3+, Ne3+): use the [Ar III]/[Cl III]
      diagnostic, either alone (``use_ar3=True``) or averaged with the
      [Ar IV]-based temperature, weighted by typical Ar++/Ar3+ abundances.

    Parameters
    ----------
    TeNe : dict
        Diagnostic results from ``diagnostics.get_TeNe`` (keys ``'N2S2'``,
        ``'S3Cl3'``, ``'Ar3Cl3'``, ``'Ar4Ar4'``, ... each a dict with
        ``'Te'`` and ``'Ne'`` arrays).
    IP : float
        Ionization potential [eV] needed to create the emitting ion.
    consider_3zones : bool, optional
        If True, use a third (high-ionization) zone for IP >= 35 eV.
    use_ar3 : bool, optional
        In the 3-zone case, use Te[Ar III]/Ne[Cl III] alone instead of the
        Ar III/Ar IV weighted mean temperature.

    Returns
    -------
    Te, Ne : np.ndarray
        Electron temperature [K] and density [cm-3] maps to be used for
        this ion.
    """
    if IP < 17:
        Te = TeNe['N2S2']['Te']
        Ne = TeNe['N2S2']['Ne']
    else:
        Te = TeNe['S3Cl3']['Te']
        Ne = TeNe['S3Cl3']['Ne']
    if (consider_3zones) & (IP >= 35):
        if use_ar3:
            Te = TeNe['Ar3Cl3']['Te']
            Ne = TeNe['Ar3Cl3']['Ne']
        else:
            # Weighted mean of the Ar III- and Ar IV-based temperatures,
            # with weights set by typical Ar++ and Ar3+ ionic abundances.
            ar_pp = 3e-6
            ar_ppp = 1e-6
            w_ar = ar_pp / (ar_pp + ar_ppp)
            Te = w_ar * (TeNe['Ar3Cl3']['Te']) + (1-w_ar) * (TeNe['Ar4Ar4']['Te'])
            Ne = TeNe['Ar3Cl3']['Ne']
    return Te, Ne

def select_Te_rec(TeNe, Te, Ne, Te_rec):
    """Select the electron temperature to use for recombination lines.

    Recombination-line (RL) abundances are very sensitive to the adopted
    Te; in the presence of cold H-poor clumps the RLs are emitted mostly by
    cold gas, so a Te lower than the CEL temperature may be appropriate.

    Parameters
    ----------
    TeNe : dict
        Diagnostic results from ``diagnostics.get_TeNe``.
    Te : np.ndarray
        Default (CEL) temperature map, returned when ``Te_rec`` is None.
    Ne : np.ndarray
        Density map; only used to broadcast a constant ``Te_rec`` to the
        right shape.
    Te_rec : {'He', 'PJ', 'PJ_ANN', None} or float
        Which temperature to use for RLs:

        - ``'He'``     : Te from the He I 7281/6678 ratio (``TeNe['He1']``).
        - ``'PJ'``     : Te from the Paschen jump (``TeNe['PJ']``).
        - ``'PJ_ANN'`` : Te from the ANN-based Paschen jump fit.
        - ``None``     : keep the CEL temperature ``Te`` unchanged.
        - float        : use this constant temperature [K] everywhere.

    Returns
    -------
    np.ndarray
        Temperature map to be used for the recombination lines.
    """
    if Te_rec == 'He':
        return TeNe['He1']['Te']
    elif Te_rec == 'PJ':
        return TeNe['PJ']['Te']
    elif Te_rec == 'PJ_ANN':
        return TeNe['PJ_ANN']['Te']
    elif Te_rec is None:
        return Te
    else:
        return Te_rec * np.ones_like(Ne)

def set_abunds_ori(TeNe, obs, label = None, tem_HI=None, exclude_elem=('H',),
                   Te_rec = None, abund_dic = None, Ne_cte = None):
    """Compute ionic abundances for all (or one) observed lines (original scheme).

    Loops over the valid lines of ``obs``, builds the corresponding PyNeb
    atom, selects the Te/Ne zone from the ion IP (`select_TeNe`) and calls
    ``atom.getIonAbundance`` on the line intensity normalized to Hbeta.

    This is the original, single-component version: all lines are assumed
    to be emitted by the same gas (no warm/cold weighting). See
    `set_abunds` for the two-component version used in the paper.

    Parameters
    ----------
    TeNe : dict
        Diagnostic results from ``diagnostics.get_TeNe``.
    obs : pn.Observation
        Observation with dereddened intensities (``line.corrIntens``).
    label : str, optional
        If given, only compute the abundance for this line label
        (e.g. ``'O3_5007A'``); otherwise process every line.
    tem_HI : float or np.ndarray, optional
        Temperature used for the H I emissivity in ``getIonAbundance``
        (passed as ``tem_HI``); if None, the same Te as the line is used.
    exclude_elem : tuple of str, optional
        Elements to skip (by default hydrogen, which is the reference).
    Te_rec : {'He', 'PJ', 'PJ_ANN', None} or float, optional
        Temperature choice for recombination lines (see `select_Te_rec`).
    abund_dic : dict, optional
        Existing dictionary to update; a new one is created if None.
    Ne_cte : float, optional
        If given, use this constant density [cm-3] instead of the
        diagnostic density map.

    Returns
    -------
    dict
        Mapping ``line.label -> ionic abundance array`` (X^i+/H+, same
        shape as the intensity maps); lines that were skipped are set
        to None.
    """
    if abund_dic is None:
        abund_dic = {}
    Hbeta = obs.getIntens()['H1r_4861A']

    atom_dic = {}

    for line in obs.getSortedLines():
        if (line.label == label or label is None) & (line.elem not in exclude_elem) & (line.is_valid):
            # Atom models are cached in atom_dic: one PyNeb atom per ion.
            if line.atom not in atom_dic:
                atom, IP, rec_line = create_atom(line)
                atom_dic[line.atom] = (atom, IP, rec_line)
            else:
                atom, IP, rec_line = atom_dic[line.atom]

            Te, Ne = select_TeNe(TeNe, IP)

            if Ne_cte is not None:
                Ne = Ne_cte * np.ones_like(Te)

            if rec_line:
                Te = select_Te_rec(TeNe, Te, Ne, Te_rec)

            abund_dic[line.label] = atom.getIonAbundance(line.corrIntens/Hbeta, Te, Ne,
                                                         to_eval=line.to_eval, Hbeta=1., tem_HI=tem_HI)
        elif line.label not in abund_dic.keys():
            abund_dic[line.label] = None
    return abund_dic

def set_abunds(TeNe, obs, w = None, w_1 = None, label = None, tem_HI=None, tem_HI_RLs = None, exclude_elem=('H',),
                   Te_rec = None, abund_dic = None, Ne_rec = None, **kwargs):
    """Compute ionic abundances with the warm/cold two-component scheme.

    Extended version of `set_abunds_ori` used for the NGC 6153 analysis:
    the nebula is modelled as a warm component (emitting the CELs) plus a
    cold, H-poor component (emitting most of the heavy-element
    recombination lines). The Hbeta flux is shared between the components
    through the weight maps ``w`` (cold fraction, applied to RLs) and
    ``w_1`` (warm fraction, applied to CELs), so each family of lines is
    normalized to the Hbeta actually emitted by its own component.

    Additional differences with `set_abunds_ori`:

    - He I lines always use Te from the He I diagnostic
      (``TeNe['He1']``) with the low-ionization density ``TeNe['N2S2']``;
      He II lines keep the CEL Te/Ne of their zone.
    - Separate H I temperatures may be given for CELs (``tem_HI``) and for
      RLs (``tem_HI_RLs``), matching the temperature of the component each
      family is emitted in.
    - Extra keyword arguments (e.g. ``consider_3zones``, ``use_ar3``) are
      forwarded to `select_TeNe`.

    Parameters
    ----------
    TeNe : dict
        Diagnostic results from ``diagnostics.get_TeNe``.
    obs : pn.Observation
        Observation with dereddened intensities.
    w : np.ndarray or float, optional
        Fraction of Hbeta emitted by the cold component (weight applied
        when computing recombination-line abundances). If None, 1 is used.
    w_1 : np.ndarray or float, optional
        Fraction of Hbeta emitted by the warm component (weight applied to
        collisionally excited lines). If None, 1 is used.
    label : str, optional
        If given, only compute the abundance of this line.
    tem_HI : float or np.ndarray, optional
        H I emissivity temperature for CELs.
    tem_HI_RLs : float or np.ndarray, optional
        H I emissivity temperature for RLs (cold component).
    exclude_elem : tuple of str, optional
        Elements to skip (default: hydrogen).
    Te_rec : {'He', 'PJ', 'PJ_ANN', None} or float, optional
        Temperature choice for the heavy-element RLs (see `select_Te_rec`).
    abund_dic : dict, optional
        Existing dictionary to update; a new one is created if None.
    Ne_rec : float, optional
        If given, constant density [cm-3] adopted for the RLs.
    **kwargs
        Forwarded to `select_TeNe` (``consider_3zones``, ``use_ar3``).

    Returns
    -------
    dict
        Mapping ``line.label -> ionic abundance array``; skipped lines are
        set to None.
    """
    if abund_dic is None:
        abund_dic = {}
    Hbeta = obs.getIntens()['H1r_4861A']

    atom_dic = {}

    for line in obs.getSortedLines():
        if (line.label == label or label is None) & (line.elem not in exclude_elem) & (line.is_valid):
            # Atom models are cached in atom_dic: one PyNeb atom per ion.
            if line.atom not in atom_dic:
                atom, IP, rec_line = create_atom(line)
                atom_dic[line.atom] = (atom, IP, rec_line)
            else:
                atom, IP, rec_line = atom_dic[line.atom]

            Te, Ne = select_TeNe(TeNe, IP, **kwargs)

            if rec_line:
                tem_HI_adopted = tem_HI_RLs
                if 'He1r' in line.atom:
                    # He I: use the He I-based temperature with the
                    # low-ionization density.
                    Te = TeNe['He1']['Te']
                    Ne = TeNe['N2S2']['Ne']
                elif 'He2r' in line.atom:
                    # He II: keep the CEL Te/Ne of its ionization zone.
                    pass
                else:
                    # Heavy-element RLs: temperature set by Te_rec.
                    Te = select_Te_rec(TeNe, Te, Ne, Te_rec)

                if Ne_rec is not None:
                    Ne = Ne_rec * np.ones_like(Te)

                if w is not None:
                    Hb_w = w
                else:
                    Hb_w = 1
            else:
                tem_HI_adopted = tem_HI
                if w_1 is not None:
                    Hb_w = w_1
                else:
                    Hb_w = 1

            abund_dic[line.label] = atom.getIonAbundance(line.corrIntens/Hbeta, Te, Ne,
                                                         to_eval=line.to_eval, Hbeta=Hb_w, tem_HI=tem_HI_adopted)
        elif line.label not in abund_dic.keys():
            abund_dic[line.label] = None
    return abund_dic
