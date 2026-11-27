"""He I lines redward of the tabulated optical set (wd_spectra.helium_impact)."""

import numpy as np
import pytest

from wd_spectra import (
    HELIUM_I_IMPACT_LINES,
    HELIUM_I_LINES,
    gray_helium_atmosphere,
    helium_i_impact_line_mass_absorption_coefficient,
    helium_i_resonance_hwhm_angstrom,
)
from wd_spectra._compat import trapezoid
from wd_spectra.constants import (
    BOLTZMANN,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE_ESU,
    LIGHT_SPEED,
    PI,
    PLANCK,
)
from wd_spectra.helium_impact import (
    helium_i_impact_lower_population,
    helium_i_impact_neutral_hwhm_angstrom,
    helium_i_impact_stark_parameters,
)


LINES = {line.name: line for line in HELIUM_I_IMPACT_LINES}


def test_line_list_is_complete_and_disjoint_from_the_tabulated_lines():
    assert len(LINES) == len(HELIUM_I_IMPACT_LINES) == 31
    for name in ("HeI10830", "HeI20581", "HeI12528", "HeI12785", "HeI12790",
                 "HeI17002", "HeI18686", "HeI18697", "HeI19089", "HeI21120",
                 "HeI21132", "HeI15084", "HeI9464"):
        assert name in LINES
    reddest_tabulated = max(line.wavelength_vacuum_angstrom for line in HELIUM_I_LINES)
    for line in HELIUM_I_IMPACT_LINES:
        assert line.wavelength_vacuum_angstrom > reddest_tabulated
        assert sum(fraction for _, fraction in line.components) == pytest.approx(1.0, abs=1e-5)
        assert line.absorption_oscillator_strength > 0.0
        assert len(line.stark_temperature) == len(line.electron_fwhm_1e16) == len(line.electron_shift_1e16)
        assert np.all(np.diff(line.stark_temperature) > 0.0)
        assert line.upper_wavenumber > line.lower_wavenumber
        # Components lie at the NIST term separation.
        assert 1.0e8 / line.wavelength_vacuum_angstrom == pytest.approx(
            line.upper_wavenumber - line.lower_wavenumber, rel=2e-4
        )


def test_nist_values_of_the_strongest_infrared_lines():
    he10830 = LINES["HeI10830"]
    # NIST ASD 5.12: 2s 3S1 - 2p 3P0,1,2 at 10832.057, 10833.217, 10833.306 A
    # (vacuum), multiplet f = 0.5392 with gf in the ratio 1:3:5.
    wavelengths = [w for w, _ in he10830.components]
    fractions = [f for _, f in he10830.components]
    assert wavelengths == pytest.approx([10832.057, 10833.217, 10833.306], abs=2e-3)
    assert fractions == pytest.approx([1 / 9, 3 / 9, 5 / 9], abs=2e-3)
    assert he10830.absorption_oscillator_strength == pytest.approx(0.5392, rel=1e-3)
    he20581 = LINES["HeI20581"]
    assert he20581.wavelength_vacuum_angstrom == pytest.approx(20586.905, abs=1e-3)
    assert he20581.absorption_oscillator_strength == pytest.approx(0.3764, rel=1e-3)
    # 3d - 4f carries the hydrogenic total once the singlet-triplet mixed 4f
    # intercombination multiplets are included.
    for name in ("HeI18686", "HeI18697"):
        assert LINES[name].absorption_oscillator_strength == pytest.approx(1.015, rel=2e-3)


def test_stark_parameters_match_starkb_and_scale_with_density():
    line = LINES["HeI10830"]
    # STARK-B (Dimitrijevic & Sahal-Brechot 1990), T = 20000 K, N = 1e16:
    # electrons w = 0.131 A, d = -0.0622 A; He II w = 0.0316 A, d = -0.017 A.
    hwhm, shift = helium_i_impact_stark_parameters(line, 20_000.0, 1.0e16, 0.0)
    assert hwhm == pytest.approx(0.5 * 0.131, rel=1e-12)
    assert shift == pytest.approx(-0.0622, rel=1e-12)
    hwhm, shift = helium_i_impact_stark_parameters(line, 20_000.0, 3.0e16, 2.0e16)
    assert hwhm == pytest.approx(0.5 * (3 * 0.131 + 2 * 0.0316), rel=1e-12)
    assert shift == pytest.approx(3 * -0.0622 + 2 * -0.017, rel=1e-12)
    # log-T interpolation between 10000 K and 20000 K, clamped outside.
    middle, _ = helium_i_impact_stark_parameters(line, np.sqrt(2.0) * 1.0e4, 1.0e16, 0.0)
    assert middle == pytest.approx(0.5 * np.sqrt(0.114 * 0.131), rel=1e-12)
    hot, _ = helium_i_impact_stark_parameters(line, 2.0e5, 1.0e16, 0.0)
    assert hot == pytest.approx(0.5 * 0.188, rel=1e-12)
    # A line without tabulated ion impact widths ignores the ion density.
    no_ion = LINES["HeI17002"]
    assert no_ion.ion_fwhm_1e16 is None
    first, _ = helium_i_impact_stark_parameters(no_ion, 20_000.0, 1.0e16, 0.0)
    second, _ = helium_i_impact_stark_parameters(no_ion, 20_000.0, 1.0e16, 1.0e17)
    assert first == second == pytest.approx(0.5 * 24.3)
    # The shift stops growing at the STARK-B isolated-line density C / w.
    limit = no_ion.isolated_line_parameter / 24.3
    _, at_limit = helium_i_impact_stark_parameters(no_ion, 20_000.0, limit, 0.0)
    wide, beyond = helium_i_impact_stark_parameters(no_ion, 20_000.0, 10.0 * limit, 0.0)
    assert beyond == pytest.approx(at_limit) == pytest.approx(-1.92 * limit / 1.0e16)
    assert wide == pytest.approx(0.5 * 24.3 * 10.0 * limit / 1.0e16)


def test_resonance_broadening_only_for_levels_connected_to_the_ground_state():
    temperature, ground = 12_000.0, 1.0e19
    singlet = LINES["HeI20581"]
    assert singlet.upper_resonance_level == 2 and singlet.lower_resonance_level is None
    triplet = LINES["HeI10830"]
    assert triplet.upper_resonance_level is None and triplet.lower_resonance_level is None
    center = singlet.wavelength_vacuum_angstrom
    resonance = helium_i_resonance_hwhm_angstrom(center, 584.33391989, 0.2762, ground)
    total = helium_i_impact_neutral_hwhm_angstrom(singlet, center, temperature, ground)
    van_der_waals_only = helium_i_impact_neutral_hwhm_angstrom(
        triplet, triplet.wavelength_vacuum_angstrom, temperature, ground
    )
    assert total > resonance > 0.0
    # Resonance dominates the van der Waals width of the 2s-2p singlet.
    assert resonance > 2.0 * (total - resonance)
    assert 0.0 < van_der_waals_only < resonance
    # 3p 1P lower level: resonance from the 537-A line.
    assert LINES["HeI19089"].lower_resonance_level == 3


def test_opacity_integrates_to_the_oscillator_strength():
    atmosphere = gray_helium_atmosphere(20_000.0, 8.0, n_depth=12)
    state = atmosphere.helium_lte_state
    line = LINES["HeI20581"]
    center = line.wavelength_vacuum_angstrom
    wavelength = center + np.concatenate((
        -np.geomspace(4000.0, 1.0e-3, 4000), [0.0], np.geomspace(1.0e-3, 4000.0, 4000)
    ))
    opacity = helium_i_impact_line_mass_absorption_coefficient(
        atmosphere, wavelength, lines=(line,), include_occupation_probability=False
    )
    depth = 6
    frequency = LIGHT_SPEED / (wavelength * 1.0e-8)
    integral = -trapezoid(opacity[:, depth], frequency)
    lower = state.neutral_level_population_density[depth, line.lower_term_index]
    stimulated = -np.expm1(
        -PLANCK * LIGHT_SPEED / (center * 1.0e-8 * BOLTZMANN * atmosphere.temperature[depth])
    )
    expected = (
        PI * ELEMENTARY_CHARGE_ESU**2 / (ELECTRON_MASS * LIGHT_SPEED)
        * line.absorption_oscillator_strength * lower * stimulated
        / atmosphere.mass_density[depth]
    )
    # The truncated Lorentz support keeps all but ~0.3% of the line area.
    assert integral == pytest.approx(expected, rel=6e-3)
    assert integral < expected


def test_n3_lower_population_is_boltzmann_relative_to_the_n2_terms():
    atmosphere = gray_helium_atmosphere(20_000.0, 8.0, n_depth=12)
    state = atmosphere.helium_lte_state
    line = LINES["HeI12528"]  # 3s 3S lower term
    population, occupation = helium_i_impact_lower_population(atmosphere, line)
    reference = state.neutral_level_population_density[:, 1]  # 2s 3S
    reference_occupation = state.neutral_level_occupation_probability[:, 1]
    expected = (
        reference * occupation / reference_occupation
        * np.exp(
            -PLANCK * LIGHT_SPEED * (line.lower_wavenumber - 159_855.971_776)
            / (BOLTZMANN * atmosphere.temperature)
        )
    )
    assert np.allclose(population, expected, rtol=1e-10)
    assert np.all((occupation > 0.0) & (occupation <= 1.0))


def test_opacity_is_local_finite_and_absent_from_the_optical():
    atmosphere = gray_helium_atmosphere(20_000.0, 8.0, n_depth=10)
    wavelength = np.unique(np.r_[np.geomspace(3000.0, 100_000.0, 6000), 10833.306, 10900.0])
    opacity = helium_i_impact_line_mass_absorption_coefficient(atmosphere, wavelength)
    assert np.all(np.isfinite(opacity)) and np.all(opacity >= 0.0)
    # In photospheric layers nothing reaches the tabulated optical lines.  (The
    # deepest gray layers, N_e > 1e21, broaden 10830 by thousands of A.)
    photospheric = atmosphere.electron_density < 1.0e19
    assert np.all(opacity[wavelength < 6600.0][:, photospheric] == 0.0)
    core = opacity[np.searchsorted(wavelength, 10833.306)]
    wing = opacity[np.searchsorted(wavelength, 10900.0)]
    assert np.all(core[photospheric] > 10.0 * wing[photospheric])
