"""Reading and preparation of the integrated 1D spectrum of NGC 6153.

Stand-alone counterpart of `observation.py` for the spatially integrated
spectrum only: it applies the same processing chain (normalization, line
blend definitions, Monte Carlo replication, extinction correction, and
[N II]/[O II] recombination-contamination corrections) to the line fluxes
measured on the summed MUSE spectrum, without touching the IFU maps.

Differences with `observation.py`:

- only ``OBS_INT_FILE`` is read (no FITS maps, so no Hbeta masking and no
  injection of integrated values into map pixels);
- the flux normalization is simply ``FLUX_NORM`` (no division by the
  number of spaxels);
- the number of Monte Carlo realizations is ``N_MC_INT`` (typically a few
  hundred, whereas the IFU maps often run with ``N_MC = None``).

The main entry point is the module-level `get_obs_int`, which returns the
calibrated ``pn.Observation`` of the integrated spectrum. See the
docstrings of `observation.Observations` for the physics of each step;
they are identical here.
"""

import pyneb as pn
import numpy as np
from utils.plots import plot_image
from constants.observation_parameters import *

class Obs_int(object):
    """Builder of the calibrated PyNeb Observation for the integrated spectrum.

    Mirrors `observation.Observations` (same method names and logic) but
    operates only on ``self.obs_int``. Boolean flags record which steps
    have been applied so each is performed once. Typical usage::

        obj = Obs_int()
        obj.get_obs()              # runs the whole chain
        obs_int = obj.obs_int      # calibrated pn.Observation

    or, equivalently, the module-level function ``get_obs_int()``.
    """

    def __init__(self):
        self.obs_int = None
        self.isNormalized_obs_int = False
        self.areLinesRedefined = False
        self.isExtinctionCorrected = False
        self.isNIICorrected = False
        self.isOIICorrected = False
        self.N_MC = N_MC_INT

    def read_obs_int(self):
        """Read the integrated line fluxes (``OBS_INT_FILE``) into ``self.obs_int``.

        Builds a ``pn.Observation`` from the ASCII file in
        ``lines_in_rows_err_cols`` format. Does nothing if already read.
        """
        if self.obs_int is None:
            self.obs_int = pn.Observation(OBS_INT_FILE,
                            fileFormat = FILE_FORMAT_INT,
                            corrected = CORRECTED,
                            errIsRelative=ERR_IS_RELATIVE,
                            err_default = ERROR_DEFAULT,
                            addErrDefault = ADD_ERR_DEFAULT) #ADD_ERR_DEFAULT

    def norm_obs_int(self):
        """Scale the integrated fluxes by ``FLUX_NORM`` (the flux unit).

        Unlike ``Observations.norm_obs_int``, no division by the number of
        spaxels is applied here: the integrated spectrum is analysed on
        its own. Reads the observation first if needed.
        """
        if (self.obs_int is not None) & (not self.isNormalized_obs_int):
            for line in self.obs_int.getSortedLines():
                line.obsIntens *= FLUX_NORM
            self.isNormalized_obs_int = True
        else:
            self.read_obs_int()
            self.norm_obs_int()

    def redefine_lines(self):
        """Define line sums and blends (same definitions as in the IFU case).

        - O I 7771/7773/7775 summed into ``O1r_7773+``;
        - ``O2r_4649.13A`` redefined as the 4649.13 + 4650.84 blend of
          O II multiplet V1;
        - ``Ne4_4726A+`` defined as the 4-3 plus 5-3 transitions of
          [Ne IV].
        """
        if (self.obs_int is not None) & (not self.areLinesRedefined):
            # Add the sum of these 3 lines under a single blend label
            self.obs_int.addSum(('O1r_7771A', 'O1r_7773A', 'O1r_7775A'), 'O1r_7773+')
            # Remove the individual components
            self.obs_int.removeLine('O1r_7771A')
            self.obs_int.removeLine('O1r_7773A')
            self.obs_int.removeLine('O1r_7775A')
            # Define 4649.13 as the sum of the two V1 multiplet components
            self.obs_int.getLine(label='O2r_4649.13A').to_eval = 'L(4649.13) + L(4650.84)'
            # Define 4726+ as the sum of the 4-3 and 5-3 transitions of Ne4
            self.obs_int.getLine(label='Ne4_4726A+').to_eval = 'I(4,3)+I(5,3)'
            self.areLinesRedefined = True
        else:
            print('Redefine lines not done. Observations must be read first')

    def add_MC(self):
        """Add ``N_MC_INT`` Monte Carlo realizations for error propagation.

        Uses PyNeb ``addMonteCarloObs`` with the fixed ``RANDOM_SEED``.
        Skipped when ``self.N_MC`` is None.
        """
        if (self.obs_int is not None) & (self.N_MC is not None):
            print(f"Adding N = {self.N_MC} Monte Carlo")
            self.obs_int.addMonteCarloObs(self.N_MC, random_seed = RANDOM_SEED)
        else:
            print('MC not added. Observations must be read first or N_MC is None')

    def red_cor_obs(self, EBV_min=None, r_theo = 2.86,
                    label1="H1r_6563A", label2="H1r_4861A"):
        """Compute E(B-V) from H I line ratios and deredden the data.

        Identical to ``Observations.red_cor_obs``: compares the observed
        ``label1``/``label2`` ratio(s) with the theoretical ``r_theo``; if
        ``r_theo`` is iterable, the median E(B-V) over the ratios is
        adopted. ``correctData`` then applies the extinction law
        (``CORREC_LAW``, ``R_V``).

        Parameters
        ----------
        EBV_min : float, optional
            E(B-V) values below this are reset to 0.
        r_theo : float or sequence of float, optional
            Theoretical ratio(s); default 2.86 (Case B Halpha/Hbeta).
        label1 : str or sequence of str, optional
            Line(s) compared to the reference line.
        label2 : str, optional
            Reference line (Hbeta).
        """
        if (self.obs_int is not None) & (not self.isExtinctionCorrected):
            self.obs_int.extinction.R_V = R_V
            print('Redifined R_V = '+str(self.obs_int.extinction.R_V))
            try:
                _ = r_theo[0]
                # Iterable r_theo: several Balmer/Paschen ratios are used to
                # compute the extinction (as for M1-42), with these lines:
                # "H1r_6563A", "H1r_9229A", "H1r_8750A", 'H1r_8863A', 'H1r_9015A'
                r_theo_is_iterable = True
            except:
                r_theo_is_iterable = False

            if r_theo_is_iterable:
                self.EBV = []
                for l1, r in zip(label1, r_theo):
                    self.obs_int.def_EBV(label1=l1, label2=label2, r_theo=r)
                    self.EBV.append(self.obs_int.extinction.E_BV)
                self.obs_int.extinction.E_BV = np.nanmedian(self.EBV, 0)
            else:
                self.obs_int.def_EBV(label1=label1, label2=label2, r_theo=r_theo)

            if EBV_min is not None:
                mask = self.obs_int.extinction.E_BV < EBV_min
                # This will only work once the class is defined; kept as a
                # comment for now.
                """pn.log_.message('number of spaxels with EBV < {} : {}/{}'.format(EBV_min, mask.sum(),len(mask)),
                                calling='PipeLine.red_cor_obs')"""
                self.obs_int.extinction.E_BV[mask] = 0.

            self.obs_int.correctData()

    def correct_by_cHb(self, tem, den):
        """Deredden using theoretical H I ratios at a given Te and Ne.

        Same scheme as ``Observations.correct_by_cHb``: theoretical
        Halpha and Paschen (9-3, 12-3, 11-3, 10-3) ratios to Hbeta are
        computed with ``pn.RecAtom('H', 1)`` at (``tem``, ``den``) and
        passed to `red_cor_obs` with the ``EXTINCTION_LABEL`` lines.

        Parameters
        ----------
        tem : float
            Electron temperature [K] (``TEM_CHB`` by default in `get_obs`).
        den : float
            Electron density [cm-3] (``DEN_CHB``).
        """
        if (self.obs_int is not None) & (not self.isExtinctionCorrected):
            HI = pn.RecAtom('H',1)
            get_r_theo = lambda label:HI.getEmissivity(tem, den, label=label, product=False)  / (
                                    HI.getEmissivity(tem, den, label='4_2', product=False) )
            R_THEO = [get_r_theo('3_2'), get_r_theo('9_3'), get_r_theo('12_3'), get_r_theo('11_3'), get_r_theo('10_3')]

            self.red_cor_obs(EBV_min = EBV_MIN,
                            label1=EXTINCTION_LABEL,
                            r_theo=R_THEO)
            self.isExtinctionCorrected = True
        else:
            print('Correction not done, observations must be read or correction was already done.')

    def correct_NII_recomb(self, tem_rec, den_rec):
        """Remove the recombination contribution from [N II] 5755.

        Same correction as ``Observations.correct_NII_recomb``: the
        recombination part of the auroral 5755 line is estimated from the
        N II 5679 recombination line using the theoretical 5755/5679
        recombination ratio (P91 for 5755, FSL11 for 5679) at
        (``tem_rec``, ``den_rec``) and subtracted from the observed value.

        Parameters
        ----------
        tem_rec : float
            Temperature [K] of the recombining gas (``TE_CORR``).
        den_rec : float
            Density [cm-3] of the recombining gas (``DEN_CORR``).
        """
        if (self.obs_int is not None) & (not self.isNIICorrected):
            I_5755 = self.obs_int.getIntens()['N2_5755A']
            I_5679 = self.obs_int.getIntens()['N2r_5679.56A']
            pn.atomicData.setDataFile('n_ii_rec_P91.func')
            N2rP = pn.RecAtom('N', 2, case='B')
            pn.atomicData.setDataFile('n_ii_rec_FSL11.func')
            N2rF = pn.RecAtom('N', 2, case='B')
            R_5755_5679 = (N2rP.getEmissivity(tem_rec, den_rec, label='5755.', product=False) /
                        N2rF.getEmissivity(tem_rec, den_rec, label='5679.56', product=False))
            with np.errstate(divide='ignore', invalid='ignore'):
                I_5755R = R_5755_5679 * I_5679
                I_5755_new = I_5755 - I_5755R
            for line in self.obs_int.lines:
                if line.label == 'N2_5755A':
                    line.corrIntens = I_5755_new
            self.isNIICorrected = True
        else:
            print('NII recombination correction not done, observations must be read or correction was already done.')

    def correct_OII_recomb(self, tem_rec, den_rec, rec_label):
        """Remove the recombination contribution from [O II] 7320/7330.

        Same correction as ``Observations.correct_OII_recomb``: the
        recombination part of the 7325+ blend is estimated from the O II
        recombination line ``rec_label`` (P91 emissivity for 7325+, SSB17
        for the reference line) and the corrected flux redistributed
        between the 7319+ and 7330+ components pro rata of their observed
        values.

        Parameters
        ----------
        tem_rec : float
            Temperature [K] of the recombining gas.
        den_rec : float
            Density [cm-3] of the recombining gas.
        rec_label : str
            Reference O II recombination line: ``'O2r_4649.13A'`` or
            ``'O2r_4661.63A'``.
        """
        if (self.obs_int is not None) & (not self.isOIICorrected):
            I_7320 = self.obs_int.getIntens()['O2_7319A+']
            I_7330 = self.obs_int.getIntens()['O2_7330A+']
            I_7325 = I_7320 + I_7330

            I_REC = self.obs_int.getIntens()[rec_label]

            pn.atomicData.setDataFile('o_ii_rec_P91.func')
            O2rP = pn.RecAtom('O', 2, case='B')
            pn.atomicData.setDataFile('o_ii_rec_SSB17-B-opt.hdf5')
            O2rS = pn.RecAtom('O', 2, case='B')
            wave_str = rec_label.split('_')[1][:-1]
            emisP = O2rP.getEmissivity(tem_rec, den_rec, label='7325+', product=False)
            if rec_label == 'O2r_4649.13A':
                emisR = O2rS.getEmissivity(tem_rec, den_rec, label='4649.13', product=False) \
                    + O2rS.getEmissivity(tem_rec, den_rec, label='4650.84', product=False)
            else:
                emisR = O2rS.getEmissivity(tem_rec, den_rec, label=wave_str, product=False)
            with np.errstate(divide='ignore', invalid='ignore'):
                R_7325_REC = emisP / emisR
                I_7325_new = I_7325 - R_7325_REC * I_REC

            for line in self.obs_int.lines:
                with np.errstate(divide='ignore', invalid='ignore'):
                    if line.label == 'O2_7319A+':
                        line.corrIntens = I_7325_new * I_7320 / I_7325
                    if line.label == 'O2_7330A+':
                        line.corrIntens = I_7325_new * I_7330 / I_7325
            self.isOIICorrected = True
        else:
            print('OII recombination correction not done, observations must be read or correction was already done.')

    def get_obs(self, tem_cHb = TEM_CHB, den_cHb = DEN_CHB, tem_rec = TE_CORR, den_rec = DEN_CORR, oii_rec_label = REC_LABEL,
                corr_NII = True, corr_OII = True):
        """Run the whole preparation chain on the integrated spectrum.

        Resets the object, then successively: reads the integrated
        fluxes, normalizes them, redefines the blended lines, adds the
        Monte Carlo realizations, corrects for extinction and (optionally)
        removes the recombination contamination of [N II] 5755 and
        [O II] 7320/7330. The result is available as ``self.obs_int``.

        Parameters
        ----------
        tem_cHb, den_cHb : float, optional
            Te [K] and Ne [cm-3] for the theoretical H I ratios of the
            extinction correction (defaults ``TEM_CHB``, ``DEN_CHB``).
        tem_rec, den_rec : float, optional
            Te/Ne of the recombining gas for the [N II] and [O II]
            corrections (defaults ``TE_CORR``, ``DEN_CORR``).
        oii_rec_label : str, optional
            O II reference line for the [O II] correction (``REC_LABEL``).
        corr_NII, corr_OII : bool, optional
            Enable/disable each recombination correction.
        """
        self.__init__()
        self.read_obs_int()
        self.norm_obs_int()
        self.redefine_lines()
        self.add_MC()
        self.correct_by_cHb(tem = tem_cHb, den = den_cHb)
        if corr_NII:
            self.correct_NII_recomb(tem_rec = tem_rec, den_rec = den_rec)
        if corr_OII:
            self.correct_OII_recomb(tem_rec = tem_rec, den_rec = den_rec, rec_label = oii_rec_label)


def get_obs_int(**kwargs):
    """Build and return the calibrated integrated-spectrum ``pn.Observation``.

    Convenience wrapper: instantiates `Obs_int`, runs the complete
    preparation chain (see ``Obs_int.get_obs``) and returns the resulting
    PyNeb Observation.

    Parameters
    ----------
    **kwargs
        Forwarded to ``Obs_int.get_obs`` (e.g. ``tem_cHb``, ``corr_NII``,
        ``oii_rec_label``).

    Returns
    -------
    pn.Observation
        Dereddened, corrected integrated observation.
    """
    obj = Obs_int()
    obj.get_obs(**kwargs)
    return obj.obs_int
