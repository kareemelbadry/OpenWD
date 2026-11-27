"""He I lines without released Stark tables, broadened with impact parameters.

The Tremblay et al. (2026) / Beauchamp tables used for ``HELIUM_I_LINES``
cover the 36 optical transitions from the ``n = 2`` terms (2723--7283 A).
This module adds the He I lines redward of them that STARK-B tabulates:
the ``2s-2p`` lines at 10830 and 20581 A and the lines from the ``n = 3``
terms (8361 A to 7.4 um), including He I 12528, 12785/12790, 17002,
18686/18697, 19089 and 21120/21132.

Atomic data
    Vacuum wavelengths, fine-structure components and oscillator strengths
    are from the NIST ASD (Kramida et al. 2024, ver. 5.12), the source of
    ``HELIUM_I_LINES``.  Each line is one LS multiplet.  For the strongly
    singlet-triplet mixed ``nf`` upper terms the intercombination multiplet
    from the same lower term (e.g. ``3d 3D - 4f 1F``) is added as a component,
    so the line carries the full NIST oscillator strength to that upper
    configuration (``3d - 4f``: f = 1.015, the hydrogenic value).
Stark broadening
    Electron-impact full widths and shifts at ``N_e = 1e16 cm^-3`` from
    STARK-B (Sahal-Brechot, Dimitrijevic & Moreau; dataset "He I, SCP"):
    Dimitrijevic & Sahal-Brechot (1989, Bull. Obs. Astron. Belgrade 141, 57;
    1990, A&AS 82, 519) and Terzi, Sahal-Brechot & Ben Lakhdar (1998, A&AS
    133, 229).  They are interpolated in log T, clamped to 2500--80000 K, and
    scaled linearly with perturber density.  The He II-impact width and shift
    are included where STARK-B gives them throughout the DB regime (10830,
    20581, 21120); elsewhere STARK-B leaves the ions to the quasi-static
    regime and no ion term is added.  ``isolated_line_parameter`` is the
    STARK-B ``C``: the isolated-line approximation holds for
    ``N_e < C / w_e(1e16)``.  10830 and 20581 are isolated throughout DB
    photospheres; the lines to ``n = 4-6`` nd/nf levels are not above
    ``N_e ~ 1e16``, where forbidden components and quasi-static ion splitting
    appear.  Their widths are then extrapolated linearly (as is conventional
    for these tables), which is the same order as the hydrogenic linear-Stark
    width of the analogous H I line, but their profiles are approximate.
Neutral broadening
    Ali--Griem resonance broadening by ground-state He for every ``np 1P``
    level of the transition (``helium_i_resonance_hwhm_angstrom``) plus the
    Unsold van der Waals width used for the optical lines, here with the
    Kurucz/SYNTHE level difference ``<r^2>_u - <r^2>_l`` (``2.5 n*^4``),
    because the lower ``n = 3`` radii are not negligible.
Other
    Thermal Doppler and classical radiative damping as for the optical
    lines; exact Voigt profile in frequency (Humlicek W4) about the shifted
    centre of each component.  Lower populations are LTE: the ``n <= 2``
    terms come from the EOS state, the ``n = 3`` terms use their NIST term
    energies with the EOS partition function and the Hummer--Mihalas
    occupation probability of the ``n = 3`` shell, as the EOS does for that
    shell.  The upper-level survival factor is the same as for the optical
    lines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._faddeeva import humlicek_w4
from .atmosphere import Atmosphere
from .constants import (
    BOLTZMANN,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE_ESU,
    HELIUM_FIRST_IONIZATION_ENERGY,
    HELIUM_MASS,
    HYDROGEN_IONIZATION_ENERGY,
    LIGHT_SPEED,
    PI,
    PLANCK,
)
from .eos import helium_occupation_probability
from .helium import (
    HELIUM_I_RESONANCE_LINES,
    _require_helium_state,
    helium_i_resonance_hwhm_angstrom,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HeliumImpactLine:
    """One He I LS multiplet with STARK-B impact broadening parameters.

    ``components`` holds ``(vacuum wavelength in A, fraction of gf)``.
    ``lower_term_index`` is the ``HeliumLTEState`` low-term column of an
    ``n = 2`` lower term, or ``None`` for an ``n = 3`` term whose population
    is computed here.  ``*_resonance_level`` is ``n`` for an ``np 1P`` level
    (resonance-broadened by ground-state He), else ``None``.  Stark widths and
    shifts are STARK-B full widths and shifts in A at a perturber density of
    1e16 cm^-3, positive to the red.
    """

    name: str
    lower_term: str
    upper_term: str
    lower_term_index: int | None
    lower_principal_quantum_number: int
    upper_principal_quantum_number: int
    lower_wavenumber: float
    lower_statistical_weight: float
    upper_wavenumber: float
    absorption_oscillator_strength: float
    components: tuple[tuple[float, float], ...]
    lower_resonance_level: int | None
    upper_resonance_level: int | None
    stark_temperature: tuple[float, ...]
    electron_fwhm_1e16: tuple[float, ...]
    electron_shift_1e16: tuple[float, ...]
    isolated_line_parameter: float
    ion_fwhm_1e16: tuple[float, ...] | None = None
    ion_shift_1e16: tuple[float, ...] | None = None

    @property
    def wavelength_vacuum_angstrom(self) -> float:
        """gf-weighted vacuum centre (wavenumber mean of the components)."""

        return 1.0 / sum(
            fraction / wavelength for wavelength, fraction in self.components
        )


# Generated by tools/build_helium_impact_lines.py from NIST ASD v5.12 lines (vacuum,
# fik) and the STARK-B He I SCP dataset (retrieved 2026-11-27).  Names use the
# STARK-B air wavelength.
HELIUM_I_IMPACT_LINES = (
    HeliumImpactLine(
        'HeI8362', '1s.3s 3S', '1s.6p 3P*',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=6,
        lower_wavenumber=183236.789095, lower_statistical_weight=3.0, upper_wavenumber=195192.745041,
        absorption_oscillator_strength=0.0119959,
        components=((8364.032160, 1.000000),),
        lower_resonance_level=None, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(17.3, 20.7, 23.4, 24.3, 24.3, 23.1),
        electron_shift_1e16=(6.06, 5.13, 3.98, 2.81, 1.0, 0.758),
        isolated_line_parameter=4.7e+17,
    ),
    HeliumImpactLine(
        'HeI9464', '1s.3s 3S', '1s.5p 3P*',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=5,
        lower_wavenumber=183236.789095, lower_statistical_weight=3.0, upper_wavenumber=193800.713160,
        absorption_oscillator_strength=0.0229192,
        components=((9466.132950, 0.111108), (9466.185171, 0.888892),),
        lower_resonance_level=None, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(10.4, 11.7, 13.3, 13.8, 13.9, 13.4),
        electron_shift_1e16=(4.84, 3.7, 2.8, 1.33, 0.518, 0.127),
        isolated_line_parameter=1e+18,
    ),
    HeliumImpactLine(
        'HeI9603', '1s.3s 1S', '1s.6p 1P*',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=6,
        lower_wavenumber=184864.826715, lower_statistical_weight=1.0, upper_wavenumber=195274.905860,
        absorption_oscillator_strength=0.02419,
        components=((9606.074900, 1.000000),),
        lower_resonance_level=None, upper_resonance_level=6,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(31.9, 39.8, 43.4, 43.6, 41.3, 37.7),
        electron_shift_1e16=(-8.82, -8.63, -6.57, -4.1, -1.61, -0.584),
        isolated_line_parameter=1.3e+17,
    ),
    HeliumImpactLine(
        'HeI9703', '1s.3p 3P*', '1s.7s 3S',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=7,
        lower_wavenumber=185564.599176, lower_statistical_weight=9.0, upper_wavenumber=195868.234370,
        absorption_oscillator_strength=0.00407222,
        components=((9705.282786, 0.888884), (9705.550710, 0.111116),),
        lower_resonance_level=None, upper_resonance_level=None,
        stark_temperature=(5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(35.1, 36.6, 38.0, 40.8, 38.8),
        electron_shift_1e16=(20.5, 20.8, 16.1, 10.9, 5.01),
        isolated_line_parameter=1.5e+18,
    ),
    HeliumImpactLine(
        'HeI10311', '1s.3p 3P*', '1s.6d 3D',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=6,
        lower_wavenumber=185564.599176, lower_statistical_weight=9.0, upper_wavenumber=195260.070049,
        absorption_oscillator_strength=0.053016,
        components=((10314.061195, 0.888880), (10314.358341, 0.111120),),
        lower_resonance_level=None, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(32.7, 43.7, 49.2, 49.0, 46.2, 41.6),
        electron_shift_1e16=(-7.32, -5.63, -4.74, -2.82, -1.22, 0.0605),
        isolated_line_parameter=2.6e+16,
    ),
    HeliumImpactLine(
        'HeI10668', '1s.3p 3P*', '1s.6s 3S',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=6,
        lower_wavenumber=185564.599176, lower_statistical_weight=9.0, upper_wavenumber=194936.117091,
        absorption_oscillator_strength=0.00823398,
        components=((10670.593992, 0.888880), (10670.917865, 0.111120),),
        lower_resonance_level=None, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(19.8, 21.3, 22.8, 22.8, 24.3, 23.9),
        electron_shift_1e16=(10.8, 13.4, 13.9, 11.1, 7.5, 4.44),
        isolated_line_parameter=2.9e+18,
    ),
    HeliumImpactLine(
        'HeI10830', '1s.2s 3S', '1s.2p 3P*',
        lower_term_index=1, lower_principal_quantum_number=2, upper_principal_quantum_number=2,
        lower_wavenumber=159855.971776, lower_statistical_weight=3.0, upper_wavenumber=169086.907646,
        absorption_oscillator_strength=0.539222,
        components=((10832.057482, 0.111090), (10833.216761, 0.333332), (10833.306454, 0.555578),),
        lower_resonance_level=None, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(0.106, 0.107, 0.114, 0.131, 0.156, 0.188),
        electron_shift_1e16=(-0.0496, -0.0603, -0.0631, -0.0622, -0.0475, -0.0383),
        ion_fwhm_1e16=(0.0293, 0.0299, 0.0307, 0.0316, 0.0328, 0.0343),
        ion_shift_1e16=(-0.0114, -0.0132, -0.015, -0.017, -0.0192, -0.0216),
        isolated_line_parameter=1.1e+20,
    ),
    HeliumImpactLine(
        'HeI10997', '1s.3d 3D', '1s.6p 3P*',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=6,
        lower_wavenumber=186101.553763, lower_statistical_weight=15.0, upper_wavenumber=195192.745041,
        absorption_oscillator_strength=0.00155118,
        components=((10999.658564, 1.000000),),
        lower_resonance_level=None, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(30.5, 36.3, 40.9, 42.2, 42.1, 39.9),
        electron_shift_1e16=(11.2, 9.56, 8.88, 6.32, 4.14, 1.68),
        isolated_line_parameter=8.1e+17,
    ),
    HeliumImpactLine(
        'HeI11013', '1s.3s 1S', '1s.5p 1P*',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=5,
        lower_wavenumber=184864.826715, lower_statistical_weight=1.0, upper_wavenumber=193942.459688,
        absorption_oscillator_strength=0.050484,
        components=((11016.087600, 1.000000),),
        lower_resonance_level=None, upper_resonance_level=5,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(26.9, 29.2, 29.5, 28.7, 26.8, 24.3),
        electron_shift_1e16=(-7.86, -7.82, -5.84, -3.73, -2.35, -1.12),
        isolated_line_parameter=3e+17,
    ),
    HeliumImpactLine(
        'HeI11045', '1s.3p 1P*', '1s.6d 1D',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=6,
        lower_wavenumber=186209.362333, lower_statistical_weight=3.0, upper_wavenumber=195260.767903,
        absorption_oscillator_strength=0.05629,
        components=((11048.007872, 1.000000),),
        lower_resonance_level=3, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(45.3, 57.2, 63.9, 63.4, 58.6, 51.9),
        electron_shift_1e16=(6.45, 6.64, 5.66, 4.39, 3.1, 2.06),
        isolated_line_parameter=1.8e+16,
    ),
    HeliumImpactLine(
        'HeI11226', '1s.3p 1P*', '1s.6s 1S',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=6,
        lower_wavenumber=186209.362333, lower_statistical_weight=3.0, upper_wavenumber=195114.866095,
        absorption_oscillator_strength=0.0070371,
        components=((11229.011033, 1.000000),),
        lower_resonance_level=3, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(34.9, 36.4, 37.2, 37.7, 38.0, 35.7),
        electron_shift_1e16=(16.6, 20.0, 19.8, 15.3, 10.3, 5.08),
        isolated_line_parameter=1.3e+18,
    ),
    HeliumImpactLine(
        'HeI11969', '1s.3p 3P*', '1s.5d 3D',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=5,
        lower_wavenumber=185564.599176, lower_statistical_weight=9.0, upper_wavenumber=193917.150918,
        absorption_oscillator_strength=0.124569,
        components=((11972.345277, 0.888879), (11972.740308, 0.111121),),
        lower_resonance_level=None, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(27.4, 31.7, 33.2, 32.0, 29.5, 26.4),
        electron_shift_1e16=(-5.44, -4.28, -3.36, -2.11, -0.922, -0.00675),
        isolated_line_parameter=6e+16,
    ),
    HeliumImpactLine(
        'HeI12528', '1s.3s 3S', '1s.4p 3P*',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=4,
        lower_wavenumber=183236.789095, lower_statistical_weight=3.0, upper_wavenumber=191217.054618,
        absorption_oscillator_strength=0.0500939,
        components=((12530.749540, 0.111109), (12530.931601, 0.888891),),
        lower_resonance_level=None, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(6.22, 7.03, 8.19, 8.75, 9.08, 8.99),
        electron_shift_1e16=(3.28, 2.45, 1.8, 1.0, 0.352, -0.0314),
        isolated_line_parameter=3.6e+18,
    ),
    HeliumImpactLine(
        'HeI12756', '1s.3d 1D', '1s.5p 1P*',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=5,
        lower_wavenumber=186104.964095, lower_statistical_weight=5.0, upper_wavenumber=193942.459688,
        absorption_oscillator_strength=0.0018677,
        components=((12759.177829, 1.000000),),
        lower_resonance_level=None, upper_resonance_level=5,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(37.7, 40.6, 40.6, 39.1, 36.0, 32.3),
        electron_shift_1e16=(-10.3, -9.98, -7.54, -5.28, -2.97, -1.39),
        isolated_line_parameter=4e+17,
    ),
    HeliumImpactLine(
        'HeI12785', '1s.3d 3D', '1s.5f 3F*',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=5,
        lower_wavenumber=186101.553763, lower_statistical_weight=15.0, upper_wavenumber=193921.118761,
        absorption_oscillator_strength=0.157029,
        components=((12788.419540, 0.799999), (12788.487736, 0.200001),),
        lower_resonance_level=None, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(9.82, 9.67, 8.9, 7.97, 7.15, 6.5),
        electron_shift_1e16=(0.815, 1.05, 1.06, 1.02, 0.791, 0.664),
        isolated_line_parameter=1.1e+17,
    ),
    HeliumImpactLine(
        'HeI12790', '1s.3d 1D', '1s.5f 1F*',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=5,
        lower_wavenumber=186104.964095, lower_statistical_weight=5.0, upper_wavenumber=193921.128275,
        absorption_oscillator_strength=0.156924,
        components=((12794.005390, 1.000000),),
        lower_resonance_level=None, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(30.1, 34.5, 34.6, 32.1, 28.7, 25.1),
        electron_shift_1e16=(0.4, 0.893, 1.23, 1.16, 1.01, 0.964),
        isolated_line_parameter=1.2e+16,
    ),
    HeliumImpactLine(
        'HeI12846', '1s.3p 3P*', '1s.5s 3S',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=5,
        lower_wavenumber=185564.599176, lower_statistical_weight=9.0, upper_wavenumber=193346.988738,
        absorption_oscillator_strength=0.0225392,
        components=((12849.471632, 0.888880), (12849.941277, 0.111120),),
        lower_resonance_level=None, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(12.0, 13.5, 14.3, 14.9, 15.9, 16.1),
        electron_shift_1e16=(7.1, 8.71, 8.96, 7.8, 5.81, 4.23),
        isolated_line_parameter=7.5e+18,
    ),
    HeliumImpactLine(
        'HeI12968', '1s.3p 1P*', '1s.5d 1D',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=5,
        lower_wavenumber=186209.362333, lower_statistical_weight=3.0, upper_wavenumber=193918.287298,
        absorption_oscillator_strength=0.14134,
        components=((12971.977346, 1.000000),),
        lower_resonance_level=3, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(40.6, 47.2, 48.1, 45.2, 40.6, 35.4),
        electron_shift_1e16=(7.09, 6.85, 5.71, 4.4, 2.99, 1.91),
        isolated_line_parameter=5.4e+16,
    ),
    HeliumImpactLine(
        'HeI12985', '1s.3d 3D', '1s.5p 3P*',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=5,
        lower_wavenumber=186101.553763, lower_statistical_weight=15.0, upper_wavenumber=193800.713160,
        absorption_oscillator_strength=0.00414151,
        components=((12988.424265, 0.911102), (12988.499185, 0.088898),),
        lower_resonance_level=None, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(20.6, 22.8, 25.6, 26.4, 26.4, 25.2),
        electron_shift_1e16=(10.1, 8.24, 7.2, 4.83, 3.0, 1.15),
        isolated_line_parameter=2e+18,
    ),
    HeliumImpactLine(
        'HeI13412', '1s.3p 1P*', '1s.5s 1S',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=5,
        lower_wavenumber=186209.362333, lower_statistical_weight=3.0, upper_wavenumber=193663.509489,
        absorption_oscillator_strength=0.018502,
        components=((13415.350932, 1.000000),),
        lower_resonance_level=3, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(23.8, 25.1, 25.4, 25.4, 25.7, 24.4),
        electron_shift_1e16=(12.7, 14.6, 14.0, 11.5, 8.12, 4.99),
        isolated_line_parameter=1.9e+18,
    ),
    HeliumImpactLine(
        'HeI15084', '1s.3s 1S', '1s.4p 1P*',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=4,
        lower_wavenumber=184864.826715, lower_statistical_weight=1.0, upper_wavenumber=191492.709304,
        absorption_oscillator_strength=0.14392,
        components=((15087.774820, 1.000000),),
        lower_resonance_level=None, upper_resonance_level=4,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(23.5, 23.9, 23.2, 22.2, 20.8, 18.9),
        electron_shift_1e16=(-8.28, -7.85, -6.65, -4.81, -3.1, -2.15),
        isolated_line_parameter=1.1e+18,
    ),
    HeliumImpactLine(
        'HeI17002', '1s.3p 3P*', '1s.4d 3D',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=4,
        lower_wavenumber=185564.599176, lower_statistical_weight=9.0, upper_wavenumber=191444.482673,
        absorption_oscillator_strength=0.477627,
        components=((17007.053836, 0.888874), (17007.826541, 0.111126),),
        lower_resonance_level=None, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(24.9, 25.8, 25.7, 24.3, 22.3, 20.1),
        electron_shift_1e16=(-4.55, -3.94, -2.83, -1.92, -1.08, -0.314),
        isolated_line_parameter=2e+17,
    ),
    HeliumImpactLine(
        'HeI18556', '1s.3d 1D', '1s.4p 1P*',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=4,
        lower_wavenumber=186104.964095, lower_statistical_weight=5.0, upper_wavenumber=191492.709304,
        absorption_oscillator_strength=0.0091817,
        components=((18560.640144, 1.000000),),
        lower_resonance_level=None, upper_resonance_level=4,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(41.0, 42.9, 41.2, 38.1, 34.1, 30.1),
        electron_shift_1e16=(-5.25, -4.96, -3.83, -3.09, -2.1, -1.23),
        isolated_line_parameter=1.5e+16,
    ),
    HeliumImpactLine(
        'HeI18686', '1s.3d 3D', '1s.4f 3F*',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=4,
        lower_wavenumber=186101.553763, lower_statistical_weight=15.0, upper_wavenumber=191451.878155,
        absorption_oscillator_strength=1.0146,
        components=((18690.419415, 0.800000), (18690.551870, 0.200000),),
        lower_resonance_level=None, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(21.0, 20.7, 19.0, 17.0, 15.3, 13.9),
        electron_shift_1e16=(1.74, 2.25, 2.27, 2.18, 1.69, 1.42),
        isolated_line_parameter=2.4e+17,
    ),
    HeliumImpactLine(
        'HeI18697', '1s.3d 1D', '1s.4f 1F*',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=4,
        lower_wavenumber=186104.964095, lower_statistical_weight=5.0, upper_wavenumber=191451.894855,
        absorption_oscillator_strength=1.01534,
        components=((18702.346966, 1.000000),),
        lower_resonance_level=None, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(23.2, 22.8, 20.9, 18.6, 16.5, 14.8),
        electron_shift_1e16=(-0.924, -0.00335, 0.483, 0.863, 0.794, 0.869),
        isolated_line_parameter=1.9e+17,
    ),
    HeliumImpactLine(
        'HeI19089', '1s.3p 1P*', '1s.4d 1D',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=4,
        lower_wavenumber=186209.362333, lower_statistical_weight=3.0, upper_wavenumber=191446.453139,
        absorption_oscillator_strength=0.64827,
        components=((19094.570570, 1.000000),),
        lower_resonance_level=3, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(46.5, 47.7, 45.2, 40.9, 35.9, 31.1),
        electron_shift_1e16=(10.1, 9.13, 7.69, 5.74, 3.71, 2.32),
        isolated_line_parameter=1.8e+17,
    ),
    HeliumImpactLine(
        'HeI19543', '1s.3d 3D', '1s.4p 3P*',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=4,
        lower_wavenumber=186101.553763, lower_statistical_weight=15.0, upper_wavenumber=191217.054618,
        absorption_oscillator_strength=0.0221813,
        components=((19548.173720, 0.111121), (19548.444456, 0.799979), (19548.597469, 0.088900),),
        lower_resonance_level=None, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(17.6, 18.8, 21.1, 22.2, 22.3, 21.8),
        electron_shift_1e16=(9.39, 8.41, 6.44, 4.69, 2.52, 1.29),
        isolated_line_parameter=8.7e+18,
    ),
    HeliumImpactLine(
        'HeI20581', '1s.2s 1S', '1s.2p 1P*',
        lower_term_index=2, lower_principal_quantum_number=2, upper_principal_quantum_number=2,
        lower_wavenumber=166277.437635, lower_statistical_weight=1.0, upper_wavenumber=171134.894441,
        absorption_oscillator_strength=0.37639,
        components=((20586.904630, 1.000000),),
        lower_resonance_level=None, upper_resonance_level=2,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(0.717, 0.75, 0.798, 0.876, 1.0, 1.13),
        electron_shift_1e16=(-0.391, -0.47, -0.455, -0.427, -0.313, -0.251),
        ion_fwhm_1e16=(0.147, 0.156, 0.166, 0.177, 0.191, 0.208),
        ion_shift_1e16=(-0.0857, -0.0999, -0.115, -0.131, -0.148, -0.167),
        isolated_line_parameter=2.1e+20,
    ),
    HeliumImpactLine(
        'HeI21120', '1s.3p 3P*', '1s.4s 3S',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=4,
        lower_wavenumber=185564.599176, lower_statistical_weight=9.0, upper_wavenumber=190298.110654,
        absorption_oscillator_strength=0.145242,
        components=((21125.824412, 0.888875), (21127.093920, 0.111125),),
        lower_resonance_level=None, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(11.1, 13.7, 15.5, 17.0, 17.9, 18.6),
        electron_shift_1e16=(5.8, 7.07, 7.76, 7.63, 6.23, 5.08),
        ion_fwhm_1e16=(float("nan"), 1.9, 2.12, 2.36, 2.63, 2.94),
        ion_shift_1e16=(float("nan"), 1.32, 1.65, 1.96, 2.29, 2.63),
        isolated_line_parameter=2.4e+19,
    ),
    HeliumImpactLine(
        'HeI21132', '1s.3p 1P*', '1s.4s 1S',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=4,
        lower_wavenumber=186209.362333, lower_statistical_weight=3.0, upper_wavenumber=190940.223749,
        absorption_oscillator_strength=0.10254,
        components=((21137.799480, 1.000000),),
        lower_resonance_level=3, upper_resonance_level=None,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(27.1, 28.7, 28.8, 28.4, 28.4, 27.3),
        electron_shift_1e16=(13.7, 15.6, 15.4, 13.2, 9.95, 7.34),
        isolated_line_parameter=4.7e+18,
    ),
    HeliumImpactLine(
        'HeI74351', '1s.3s 1S', '1s.3p 1P*',
        lower_term_index=None, lower_principal_quantum_number=3, upper_principal_quantum_number=3,
        lower_wavenumber=184864.826715, lower_statistical_weight=1.0, upper_wavenumber=186209.362333,
        absorption_oscillator_strength=0.62608,
        components=((74375.121500, 1.000000),),
        lower_resonance_level=None, upper_resonance_level=3,
        stark_temperature=(2500.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0),
        electron_fwhm_1e16=(180.0, 185.0, 182.0, 176.0, 171.0, 162.0),
        electron_shift_1e16=(-81.0, -84.7, -75.0, -63.5, -47.5, -36.0),
        isolated_line_parameter=5.8e+19,
    ),
)


_RESONANCE_BY_LEVEL = {
    line.upper_principal_quantum_number: line for line in HELIUM_I_RESONANCE_LINES
}
_STARK_REFERENCE_DENSITY = 1.0e16
# Kurucz/SYNTHE Unsold constant for H perturbers and the He-perturber scaling
# already used for the optical He I lines (helium_i_line_mass_absorption_coefficient).
_UNSOLD_HYDROGEN_COEFFICIENT = 4.5e-9
_UNSOLD_HELIUM_PERTURBER_SCALE = 0.42
# Numerical support of each component: 200 Lorentz half widths (0.3% of the
# Lorentz area lies outside), at least 12 Doppler sigma, at most 20% of the
# wavelength (the impact Lorentzian is not meant to reach the next series).
_SUPPORT_LORENTZ_HALF_WIDTHS = 200.0
_SUPPORT_DOPPLER_SIGMAS = 12.0
_SUPPORT_MAXIMUM_FRACTION = 0.2


def _effective_principal_quantum_number(wavenumber: float) -> float:
    binding = HELIUM_FIRST_IONIZATION_ENERGY - PLANCK * LIGHT_SPEED * wavenumber
    return float(np.sqrt(HYDROGEN_IONIZATION_ENERGY / binding))


def _interpolate_in_log_temperature(
    line_temperature: tuple[float, ...],
    values: tuple[float, ...],
    temperature: FloatArray,
    *,
    logarithmic_values: bool,
) -> FloatArray:
    table_temperature = np.asarray(line_temperature, dtype=np.float64)
    table_value = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(table_value)
    log_table = np.log10(table_temperature[valid])
    log_temperature = np.clip(np.log10(temperature), log_table[0], log_table[-1])
    if logarithmic_values:
        return np.power(
            10.0, np.interp(log_temperature, log_table, np.log10(table_value[valid]))
        )
    return np.interp(log_temperature, log_table, table_value[valid])


def helium_i_impact_stark_parameters(
    line: HeliumImpactLine,
    temperature: ArrayLike,
    electron_density: ArrayLike,
    singly_ionized_helium_density: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return the impact Stark HWHM and shift in Angstrom.

    Electron (and, where tabulated, He II) STARK-B full widths are halved to
    the HWHM convention of this package; shifts are positive to the red.
    Widths scale linearly with perturber density; shifts only up to the
    isolated-line limit ``C / w_e``.
    """

    temperature, electron_density, ion_density = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(electron_density, dtype=np.float64),
        np.asarray(singly_ionized_helium_density, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0)
        or np.any(~np.isfinite(electron_density)) or np.any(electron_density < 0.0)
        or np.any(~np.isfinite(ion_density)) or np.any(ion_density < 0.0)
    ):
        raise ValueError("invalid thermodynamic state for He I impact widths")
    electron_fwhm = _interpolate_in_log_temperature(
        line.stark_temperature, line.electron_fwhm_1e16, temperature,
        logarithmic_values=True,
    )
    # Beyond N_e = C / w_e(1e16) the line is not isolated: the impact width
    # keeps its linear density scaling, but the quadratic-Stark shift is not
    # defined there and is held at its value at that density.
    isolated_scale = line.isolated_line_parameter / electron_fwhm / _STARK_REFERENCE_DENSITY
    electron_scale = electron_density / _STARK_REFERENCE_DENSITY
    hwhm = 0.5 * electron_scale * electron_fwhm
    shift = np.minimum(electron_scale, isolated_scale) * _interpolate_in_log_temperature(
        line.stark_temperature, line.electron_shift_1e16, temperature,
        logarithmic_values=False,
    )
    if line.ion_fwhm_1e16 is not None:
        ion_scale = ion_density / _STARK_REFERENCE_DENSITY
        hwhm = hwhm + 0.5 * ion_scale * _interpolate_in_log_temperature(
            line.stark_temperature, line.ion_fwhm_1e16, temperature,
            logarithmic_values=True,
        )
        assert line.ion_shift_1e16 is not None
        shift = shift + np.minimum(ion_scale, isolated_scale) * _interpolate_in_log_temperature(
            line.stark_temperature, line.ion_shift_1e16, temperature,
            logarithmic_values=False,
        )
    return hwhm, shift


def helium_i_impact_neutral_hwhm_angstrom(
    line: HeliumImpactLine,
    wavelength_angstrom: float,
    temperature: ArrayLike,
    ground_state_density: ArrayLike,
) -> FloatArray:
    """Return the neutral-He (resonance plus van der Waals) HWHM in Angstrom."""

    temperature = np.asarray(temperature, dtype=np.float64)
    ground = np.asarray(ground_state_density, dtype=np.float64)
    hwhm = np.zeros(np.broadcast(temperature, ground).shape)
    for level in (line.lower_resonance_level, line.upper_resonance_level):
        if level is None:
            continue
        resonance = _RESONANCE_BY_LEVEL[level]
        hwhm = hwhm + helium_i_resonance_hwhm_angstrom(
            wavelength_angstrom,
            resonance.wavelength_vacuum_angstrom,
            resonance.absorption_oscillator_strength,
            ground,
        )
    upper_level = _effective_principal_quantum_number(line.upper_wavenumber)
    lower_level = _effective_principal_quantum_number(line.lower_wavenumber)
    radius_difference = 2.5 * (upper_level**4 - lower_level**4)
    damping_rate = (
        _UNSOLD_HYDROGEN_COEFFICIENT
        * _UNSOLD_HELIUM_PERTURBER_SCALE
        * ground
        * (temperature / 10_000.0) ** 0.3
        * abs(radius_difference) ** 0.4
    )
    center_cm = wavelength_angstrom * 1.0e-8
    return hwhm + center_cm**2 * damping_rate / (4.0 * PI * LIGHT_SPEED) * 1.0e8


def helium_i_impact_lower_population(
    atmosphere: Atmosphere, line: HeliumImpactLine
) -> tuple[FloatArray, FloatArray]:
    """Return the LTE lower-term number density and occupation probability."""

    state = _require_helium_state(atmosphere)
    if line.lower_term_index is not None:
        return (
            state.neutral_level_population_density[:, line.lower_term_index],
            state.neutral_level_occupation_probability[:, line.lower_term_index],
        )
    occupation = _occupation(atmosphere, float(line.lower_principal_quantum_number))
    boltzmann = np.exp(
        -PLANCK * LIGHT_SPEED * line.lower_wavenumber
        / (BOLTZMANN * atmosphere.temperature)
    )
    population = (
        state.neutral_he_density
        * line.lower_statistical_weight
        * occupation
        * boltzmann
        / state.neutral_partition_function
    )
    return population, occupation


def _occupation(atmosphere: Atmosphere, principal: float) -> FloatArray:
    state = _require_helium_state(atmosphere)
    return helium_occupation_probability(
        state.neutral_he_density,
        state.electron_density,
        atmosphere.temperature,
        principal,
        neutral_h_density=atmosphere.neutral_h_density,
        neutral_radius_scale=state.neutral_radius_scale,
        hydrogen_neutral_radius_scale=(
            atmosphere.hydrogen_lte_state.neutral_radius_scale
            if atmosphere.hydrogen_lte_state is not None else 1.0
        ),
        correlated_microfields=(state.microfield_model == "qmhd"),
    )


def _voigt_per_angstrom(
    wavelength: FloatArray, center: float, sigma: float, hwhm: float
) -> FloatArray:
    """Area-normalized Voigt evaluated in frequency, per Angstrom at centre."""

    center_cm = center * 1.0e-8
    per_angstrom = LIGHT_SPEED / center_cm**2 * 1.0e-8
    sigma_nu = sigma * per_angstrom
    gamma_nu = hwhm * per_angstrom
    detuning = LIGHT_SPEED / (wavelength * 1.0e-8) - LIGHT_SPEED / center_cm
    scale = 1.0 / (sigma_nu * np.sqrt(2.0))
    return (
        humlicek_w4(detuning * scale, gamma_nu * scale).real
        * scale / np.sqrt(PI) * per_angstrom
    )


def helium_i_impact_line_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    lines: tuple[HeliumImpactLine, ...] = HELIUM_I_IMPACT_LINES,
    include_occupation_probability: bool = True,
) -> FloatArray:
    """Return the LTE opacity of ``HELIUM_I_IMPACT_LINES`` in cm^2 g^-1."""

    state = _require_helium_state(atmosphere)
    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1 or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0) or np.any(np.diff(wavelength) < 0.0)
    ):
        raise ValueError("wavelength_angstrom must be a finite positive increasing 1D array")
    result = np.zeros((wavelength.size, atmosphere.n_depth), dtype=np.float64)
    if wavelength.size == 0 or not lines:
        return result
    integrated_cross_section = (
        PI * ELEMENTARY_CHARGE_ESU**2 / (ELECTRON_MASS * LIGHT_SPEED)
    )
    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    thermal_fraction = np.sqrt(BOLTZMANN * temperature / HELIUM_MASS) / LIGHT_SPEED
    ground = state.neutral_level_population_density[:, 0]
    for line in lines:
        lowest = min(component[0] for component in line.components)
        highest = max(component[0] for component in line.components)
        if (
            wavelength[-1] < lowest * (1.0 - _SUPPORT_MAXIMUM_FRACTION)
            or wavelength[0] > highest * (1.0 + _SUPPORT_MAXIMUM_FRACTION)
        ):
            continue
        lower_population, lower_occupation = helium_i_impact_lower_population(
            atmosphere, line
        )
        survival = np.ones(atmosphere.n_depth)
        if include_occupation_probability:
            upper_occupation = _occupation(
                atmosphere, float(line.upper_principal_quantum_number)
            )
            survival = np.minimum(
                upper_occupation
                / np.maximum(lower_occupation, np.finfo(np.float64).tiny),
                1.0,
            )
        center = line.wavelength_vacuum_angstrom
        stark_hwhm, stark_shift = helium_i_impact_stark_parameters(
            line, temperature, state.electron_density,
            state.singly_ionized_he_density,
        )
        neutral_hwhm = helium_i_impact_neutral_hwhm_angstrom(
            line, center, temperature, ground
        )
        center_cm = center * 1.0e-8
        radiative_hwhm = (
            center_cm**2 * 2.47342e-22 * (LIGHT_SPEED / center_cm) ** 2
            / (4.0 * PI * LIGHT_SPEED) * 1.0e8
        )
        lorentz_hwhm = stark_hwhm + neutral_hwhm + radiative_hwhm
        weight = (
            integrated_cross_section
            * line.absorption_oscillator_strength
            * lower_population
            * survival
            / atmosphere.mass_density
        )
        for component_wavelength, fraction in line.components:
            component_cm = component_wavelength * 1.0e-8
            stimulated = -np.expm1(
                -PLANCK * LIGHT_SPEED / (component_cm * BOLTZMANN * temperature)
            )
            to_frequency = 1.0e8 * component_cm**2 / LIGHT_SPEED
            for depth in range(atmosphere.n_depth):
                if weight[depth] <= 0.0:
                    continue
                shifted = component_wavelength + float(stark_shift[depth])
                sigma = component_wavelength * float(thermal_fraction[depth])
                hwhm = float(lorentz_hwhm[depth])
                half_width = min(
                    max(
                        _SUPPORT_LORENTZ_HALF_WIDTHS * hwhm,
                        _SUPPORT_DOPPLER_SIGMAS * sigma,
                    ),
                    _SUPPORT_MAXIMUM_FRACTION * component_wavelength,
                )
                start = int(np.searchsorted(wavelength, shifted - half_width))
                stop = int(np.searchsorted(wavelength, shifted + half_width, side="right"))
                if stop <= start:
                    continue
                profile = _voigt_per_angstrom(
                    wavelength[start:stop], shifted, sigma, hwhm
                )
                result[start:stop, depth] += (
                    weight[depth] * fraction * stimulated[depth]
                    * profile * to_frequency
                )
    return result


def helium_i_impact_line_structure_screen(
    atmosphere: Atmosphere,
    first_below_photosphere: int,
    *,
    threshold: float = 1.0e-3,
    lines: tuple[HeliumImpactLine, ...] = HELIUM_I_IMPACT_LINES,
) -> tuple[HeliumImpactLine, ...]:
    """Lines whose Doppler-core optical depth above tau_R ~ 1 reaches ``threshold``.

    Same strict upper bound as the structure screen of the optical lines:
    the 10000-K thermal Doppler peak, no damping, no occupation suppression.
    """

    from ._compat import trapezoid

    integrated_cross_section = (
        PI * ELEMENTARY_CHARGE_ESU**2 / (ELECTRON_MASS * LIGHT_SPEED)
    )
    thermal_velocity_fraction = np.sqrt(BOLTZMANN * 10_000.0 / HELIUM_MASS) / LIGHT_SPEED
    retained = []
    for line in lines:
        center = line.wavelength_vacuum_angstrom
        center_cm = center * 1.0e-8
        lower_population, _ = helium_i_impact_lower_population(atmosphere, line)
        doppler_peak_per_angstrom = 1.0 / (np.sqrt(2.0 * PI) * center * thermal_velocity_fraction)
        stimulated = -np.expm1(
            -PLANCK * LIGHT_SPEED / (center_cm * BOLTZMANN * atmosphere.temperature)
        )
        mass_opacity = (
            integrated_cross_section * line.absorption_oscillator_strength
            * lower_population * stimulated * doppler_peak_per_angstrom
            * 1.0e8 * center_cm**2 / LIGHT_SPEED / atmosphere.mass_density
        )
        depth = trapezoid(
            mass_opacity[:first_below_photosphere],
            atmosphere.column_mass[:first_below_photosphere],
        )
        if depth >= threshold:
            retained.append(line)
    return tuple(retained)
