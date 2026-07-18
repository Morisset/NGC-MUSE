"""Reading and preparation of the MUSE emission-line maps of NGC 6153.

This module wraps everything needed to turn the raw per-line FITS maps
(plus the integrated 1D spectrum) into a fully calibrated PyNeb
``Observation`` ready for the diagnostic (`diagnostics.py`) and abundance
(`ionic_abund.py`) steps:

- reading the IFU maps and the integrated spectrum (paths and reading
  options come from `constants/observation_parameters.py`);
- flux normalization and cleaning of suspicious errors;
- masking of low-surface-brightness spaxels (Hbeta cut);
- definition of line blends/sums (O I 7773+, O II 4649+4650, ...);
- optional Monte Carlo replication of the observation for error
  propagation;
- extinction correction from Balmer/Paschen H I line ratios;
- subtraction of the recombination contribution to the [N II] 5755 and
  [O II] 7320/7330 auroral lines, which would otherwise bias the derived
  temperatures.

The `Observations` class carries the state (which corrections have been
applied); the module-level helper `get_obs` runs the full chain with the
default parameters and returns the resulting ``pn.Observation``. The
companion module `obs_int.py` does the same for the integrated spectrum
alone.
"""

import pyneb as pn
import numpy as np
from utils.plots import plot_image
from constants.observation_parameters import *

class Observations(object):
    """Builder of the calibrated PyNeb Observation for the IFU maps.

    The class stores the IFU observation (``self.obs``) and the
    integrated-spectrum observation (``self.obs_int``), plus boolean flags
    recording which processing steps have already been applied, so that
    each step is only performed once. The typical usage is simply::

        obj = Observations()
        obj.get_obs()          # runs the whole chain
        obs = obj.obs          # calibrated pn.Observation

    or, equivalently, the module-level function ``get_obs()``.
    """

    def __init__(self):
        self.obs = None
        self.obs_int = None
        self.isNormalized_obs_int = False
        self.isNormalized_obs_IFU = False
        self.areLinesRedefined = False
        self.isExtinctionCorrected = False
        self.isNIICorrected = False
        self.isOIICorrected = False

    def read_obs_IFU(self):
        """Read the per-line IFU FITS maps into ``self.obs``.

        Builds a ``pn.Observation`` from the files matching ``OBS_NAME``
        (one FITS map per emission line, ``fits_IFU`` format), with the
        error-handling and optional 2D-cutout options defined in
        `constants/observation_parameters.py`. Does nothing if the
        observation was already read.
        """
        if self.obs is None:
            self.obs = pn.Observation(OBS_NAME,
                          fileFormat=FILE_FORMAT,
                          corrected = CORRECTED,
                          correcLaw = CORREC_LAW,
                          errStr = ERROR_STR,
                          errIsRelative = ERR_IS_RELATIVE,
                          err_default = ERROR_DEFAULT,
                          addErrDefault = ADD_ERR_DEFAULT,
                          Cutout2D_position = CUTOUT2D_POSITION,
                          Cutout2D_size = CUTOUT2D_SIZE)

    def read_obs_int(self):
        """Read the integrated 1D spectrum line fluxes into ``self.obs_int``.

        Builds a ``pn.Observation`` from the ASCII file ``OBS_INT_FILE``
        (format ``lines_in_rows_err_cols``). Does nothing if already read.
        """
        if self.obs_int is None:
            self.obs_int = pn.Observation(OBS_INT_FILE,
                            fileFormat = FILE_FORMAT_INT,
                            corrected = CORRECTED,
                            errIsRelative=ERR_IS_RELATIVE,
                            err_default = ERROR_DEFAULT,
                            addErrDefault = ADD_ERR_DEFAULT)

    def norm_obs_int(self):
        """Normalize the integrated fluxes to the mean flux per spaxel.

        Multiplies the integrated intensities by ``FLUX_NORM`` and divides
        by the number of spaxels of the IFU field, so they can be inserted
        into the IFU observation (see `norm_obs_IFU`) on the same scale as
        the individual spaxels. Reads both observations first if needed.
        """
        if (self.obs_int is not None) & (self.obs is not None) & (not self.isNormalized_obs_int):
            for line in self.obs_int.getSortedLines():
                line.obsIntens *= FLUX_NORM / self.obs.origin_fits_shape[0] / self.obs.origin_fits_shape[1]
            self.isNormalized_obs_int = True

        else:
            self.read_obs_int()
            self.read_obs_IFU()
            self.norm_obs_int()

    def norm_obs_IFU(self):
        """Normalize the IFU maps and inject the integrated values.

        This method:

        - multiplies every line map by ``FLUX_NORM`` (the FITS flux unit);
        - flags as NaN the pixels whose error is suspiciously equal to the
          default error (within ``CLEAN_ERROR``), i.e. pixels with no real
          measurement;
        - stores the integrated-spectrum intensity and error of each line
          in pixel [0] of the corresponding map, so the integrated values
          travel with the IFU observation (they are retrieved later by the
          figure/table scripts).
        """
        if not self.isNormalized_obs_int:
            self.norm_obs_int()

        obs_int_dict = self.obs_int.getIntens(returnObs = True)
        err_int_dict = self.obs_int.getError(returnObs = True)

        if not self.isNormalized_obs_IFU:
            for line in self.obs.getSortedLines():
                line.obsIntens *= FLUX_NORM
                if CLEAN_ERROR is not None:
                    mask_err = np.abs(line.obsError - ERROR_DEFAULT) < CLEAN_ERROR
                    line.obsIntens[mask_err] = np.nan
                try:
                    line.obsIntens[0] =  obs_int_dict[line.label][0]
                    line.obsError[0] = err_int_dict[line.label][0]
                except:
                    print(f'Integrated value for {line.label} not done')
            self.isNormalized_obs_IFU = True
        else:
            print('Noting done. IFU observations already normalized')

    def mask_by_Hb(self):
        """Mask the faint outer spaxels using an Hbeta surface-brightness cut.

        Every spaxel whose Hbeta flux is below ``CUT_HB`` times the peak
        Hbeta flux is set to NaN in all line maps, removing the noisy
        nebular outskirts from the analysis.
        """
        if self.obs is not None:
            Hb = self.obs.getIntens(returnObs=True)['H1r_4861A']
            mask_Hb = np.where(Hb > (np.nanmax(Hb) * CUT_HB), True, False)
            for line in self.obs.getSortedLines():
                line.obsIntens[~mask_Hb] = np.nan
        else:
            print('Mask by Hb not done. Observations must be read first.')

    def redefine_lines(self):
        """Define line sums and blends used by the analysis.

        - The O I 7771, 7773, 7775 triplet is summed into a single line
          labelled ``O1r_7773+`` and the individual components removed.
        - ``O2r_4649.13A`` is redefined as the blend 4649.13 + 4650.84 of
          O II multiplet V1 (the two components are not resolved by MUSE).
        - ``Ne4_4726A+`` is defined as the sum of the 4-3 and 5-3
          transitions of [Ne IV].
        """
        if (self.obs is not None) & (not self.areLinesRedefined):
            # Add the sum of these 3 lines under a single blend label
            self.obs.addSum(('O1r_7771A', 'O1r_7773A', 'O1r_7775A'), 'O1r_7773+')
            # Remove the individual components
            self.obs.removeLine('O1r_7771A')
            self.obs.removeLine('O1r_7773A')
            self.obs.removeLine('O1r_7775A')
            # Define 4649.13 as the sum of the two V1 multiplet components
            self.obs.getLine(label='O2r_4649.13A').to_eval = 'L(4649.13) + L(4650.84)'
            # Define 4726+ as the sum of the 4-3 and 5-3 transitions of Ne4
            self.obs.getLine(label='Ne4_4726A+').to_eval = 'I(4,3)+I(5,3)'
            self.areLinesRedefined = True
        else:
            print('Redefine lines not done. Observations must be read first')

    def add_MC(self):
        """Add Monte Carlo realizations of the observation for error propagation.

        Replicates every line ``N_MC`` times with Gaussian noise drawn
        from the observed errors (PyNeb ``addMonteCarloObs``), using the
        fixed ``RANDOM_SEED`` for reproducibility. Skipped when ``N_MC``
        is None.
        """
        if (self.obs is not None) & (N_MC is not None):
            print(f"Adding N = {N_MC} Monte Carlo")
            self.obs.addMonteCarloObs(N_MC, random_seed = RANDOM_SEED)
        else:
            print('MC not added. Observations must be read first or N_MC is None')

    def red_cor_obs(self, EBV_min=None, r_theo = 2.86,
                    label1="H1r_6563A", label2="H1r_4861A"):
        """Compute E(B-V) from H I line ratios and deredden the data.

        The color excess is derived by comparing the observed ratio
        ``label1``/``label2`` with its theoretical value ``r_theo``. If
        ``r_theo`` is iterable, several H I lines are used (each paired
        with ``label2``) and the median E(B-V) map is adopted — this is
        the multi-ratio scheme used with the Balmer + Paschen lines listed
        in ``EXTINCTION_LABEL``. Finally ``correctData`` applies the
        extinction law (``CORREC_LAW``, ``R_V``) to every line.

        Parameters
        ----------
        EBV_min : float, optional
            Spaxels with E(B-V) below this value are reset to 0 (avoids
            unphysical negative extinction).
        r_theo : float or sequence of float, optional
            Theoretical ratio(s) of ``label1``/``label2``; default 2.86 is
            the Case B Halpha/Hbeta value.
        label1 : str or sequence of str, optional
            Line(s) compared to the reference line.
        label2 : str, optional
            Reference line (Hbeta).
        """
        if (self.obs is not None) & (not self.isExtinctionCorrected):
            self.obs.extinction.R_V = R_V
            print('Redifined R_V = '+str(self.obs.extinction.R_V))
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
                    self.obs.def_EBV(label1=l1, label2=label2, r_theo=r)
                    self.EBV.append(self.obs.extinction.E_BV)
                self.obs.extinction.E_BV = np.nanmedian(self.EBV, 0)
            else:
                self.obs.def_EBV(label1=label1, label2=label2, r_theo=r_theo)

            if EBV_min is not None:
                mask = self.obs.extinction.E_BV < EBV_min
                # This will only work once the class is defined; kept as a
                # comment for now.
                """pn.log_.message('number of spaxels with EBV < {} : {}/{}'.format(EBV_min, mask.sum(),len(mask)),
                                calling='PipeLine.red_cor_obs')"""
                self.obs.extinction.E_BV[mask] = 0.

            self.obs.correctData()

    def correct_by_cHb(self, tem, den):
        """Deredden using theoretical H I ratios at a given Te and Ne.

        Computes with ``pn.RecAtom('H', 1)`` the theoretical ratios of
        Halpha 6563 and the Paschen lines 9229 (9-3), 8750 (12-3),
        8863 (11-3) and 9015 (10-3) relative to Hbeta at the given
        temperature and density, then calls `red_cor_obs` with those
        ratios and the ``EXTINCTION_LABEL`` line list.

        Parameters
        ----------
        tem : float
            Electron temperature [K] for the theoretical emissivities
            (``TEM_CHB``, based on Te[S III], by default in `get_obs`).
        den : float
            Electron density [cm-3] (``DEN_CHB``, based on Ne[Cl III]).
        """
        if (self.obs is not None) & (not self.isExtinctionCorrected):
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

        The auroral [N II] 5755 line contains a contribution from
        recombination of N++, which would bias Te[N II] upwards. It is
        estimated from the pure recombination line N II 5679: the
        theoretical 5755/5679 recombination ratio is computed at
        (``tem_rec``, ``den_rec``) using the Pequignot et al. 1991 (P91)
        coefficients for 5755 and Fang, Storey & Liu 2011 (FSL11) for
        5679, and the scaled N II 5679 intensity is subtracted from the
        observed 5755 map (stored in ``line.corrIntens``).

        Parameters
        ----------
        tem_rec : float
            Temperature [K] assumed for the recombination emission
            (cold-component temperature, ``TE_CORR`` by default).
        den_rec : float
            Density [cm-3] assumed for the recombination emission
            (``DEN_CORR``).
        """
        if (self.obs is not None) & (not self.isNIICorrected):
            I_5755 = self.obs.getIntens()['N2_5755A']
            I_5679 = self.obs.getIntens()['N2r_5679.56A']
            pn.atomicData.setDataFile('n_ii_rec_P91.func')
            N2rP = pn.RecAtom('N', 2, case='B')
            pn.atomicData.setDataFile('n_ii_rec_FSL11.func')
            N2rF = pn.RecAtom('N', 2, case='B')
            R_5755_5679 = (N2rP.getEmissivity(tem_rec, den_rec, label='5755.', product=False) /
                        N2rF.getEmissivity(tem_rec, den_rec, label='5679.56', product=False))
            with np.errstate(divide='ignore', invalid='ignore'):
                I_5755R = R_5755_5679 * I_5679
                I_5755_new = I_5755 - I_5755R
            for line in self.obs.lines:
                if line.label == 'N2_5755A':
                    line.corrIntens = I_5755_new
            self.isNIICorrected = True
        else:
            print('NII recombination correction not done, observations must be read or correction was already done.')

    def correct_OII_recomb(self, tem_rec, den_rec, rec_label):
        """Remove the recombination contribution from [O II] 7320/7330.

        Analogous to `correct_NII_recomb` for the [O II] 7325+ auroral
        blend: the recombination contribution of O++ is estimated from an
        observed O II recombination line (``rec_label``), using the P91
        emissivity of the 7325+ blend and the SSB17 (Storey, Sochi &
        Bastin 2017) emissivity of the reference line, both at
        (``tem_rec``, ``den_rec``). The corrected 7325+ intensity is then
        redistributed between the 7319+ and 7330+ components pro rata of
        their observed fluxes.

        Parameters
        ----------
        tem_rec : float
            Temperature [K] assumed for the recombination emission.
        den_rec : float
            Density [cm-3] assumed for the recombination emission.
        rec_label : str
            Label of the O II recombination line used as reference;
            ``'O2r_4649.13A'`` (blend 4649.13+4650.84 of multiplet V1) or
            ``'O2r_4661.63A'``.
        """
        if (self.obs is not None) & (not self.isOIICorrected):
            I_7320 = self.obs.getIntens()['O2_7319A+']
            I_7330 = self.obs.getIntens()['O2_7330A+']
            I_7325 = I_7320 + I_7330

            I_REC = self.obs.getIntens()[rec_label]

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

            for line in self.obs.lines:
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
        """Run the whole preparation chain on a fresh observation.

        Resets the object, then successively: reads the IFU maps and the
        integrated spectrum, normalizes both, applies the Hbeta mask,
        redefines the blended lines, adds the Monte Carlo realizations,
        corrects for extinction, and (optionally) removes the
        recombination contamination of [N II] 5755 and [O II] 7320/7330.
        The result is available as ``self.obs``.

        Parameters
        ----------
        tem_cHb, den_cHb : float, optional
            Te [K] and Ne [cm-3] used for the theoretical H I ratios of
            the extinction correction (defaults ``TEM_CHB``, ``DEN_CHB``).
        tem_rec, den_rec : float, optional
            Te/Ne of the recombining gas used in the [N II] and [O II]
            corrections (defaults ``TE_CORR``, ``DEN_CORR``).
        oii_rec_label : str, optional
            O II recombination line used as reference for the [O II]
            correction (default ``REC_LABEL``).
        corr_NII, corr_OII : bool, optional
            Enable/disable each recombination correction.
        """
        self.__init__()
        self.read_obs_IFU()
        self.read_obs_int()
        self.norm_obs_int()
        self.norm_obs_IFU()
        self.mask_by_Hb()
        self.redefine_lines()
        self.add_MC()
        self.correct_by_cHb(tem = tem_cHb, den = den_cHb)
        if corr_NII:
            self.correct_NII_recomb(tem_rec = tem_rec, den_rec = den_rec)
        if corr_OII:
            self.correct_OII_recomb(tem_rec = tem_rec, den_rec = den_rec, rec_label = oii_rec_label)


def get_obs(**kwargs):
    """Build and return the fully calibrated IFU ``pn.Observation``.

    Convenience wrapper: instantiates `Observations`, runs the complete
    preparation chain (see ``Observations.get_obs``) and returns the
    resulting PyNeb Observation.

    Parameters
    ----------
    **kwargs
        Forwarded to ``Observations.get_obs`` (e.g. ``tem_cHb``,
        ``corr_NII``, ``oii_rec_label``).

    Returns
    -------
    pn.Observation
        Dereddened, corrected observation with all line maps.
    """
    obj = Observations()
    obj.get_obs(**kwargs)
    return obj.obs

def get_wcs():
    """Return the WCS of the IFU observation (rebuilds the observation)."""
    obs = get_obs()
    return obs.wcs
