"""Narrow He II and He I resonance line cores in the LTE helium structure grid."""

import numpy as np
import pytest

from wd_spectra._compat import trapezoid
from wd_spectra.atmosphere import _resolved_line_core_wavelengths, helium_continuum_atmosphere
from wd_spectra.constants import BOLTZMANN, HELIUM_MASS, LIGHT_SPEED
from wd_spectra.helium import (
    HELIUM_I_RESONANCE_LINES,
    HELIUM_II_LINES,
    helium_i_resonance_line_mass_absorption_coefficient,
    helium_ii_line_mass_absorption_coefficient,
)


def test_core_offsets_start_inside_the_doppler_core_and_grow_geometrically():
    temperature, center, ratio = 12_000.0, 303.78, 1.2
    grid = _resolved_line_core_wavelengths([center], temperature, HELIUM_MASS, step_ratio=ratio)
    doppler = center * np.sqrt(2.0 * BOLTZMANN * temperature / HELIUM_MASS) / LIGHT_SPEED
    offsets = grid[grid > center] - center
    assert center in grid and grid.size == 2 * offsets.size + 1
    np.testing.assert_allclose(np.sort(center - grid[grid < center]), offsets, rtol=0, atol=1e-12)
    assert offsets[0] == pytest.approx(0.1 * doppler, rel=1e-12)
    assert offsets[-1] == pytest.approx(5.0, rel=1e-12)
    assert np.all(offsets[1:] / offsets[:-1] <= ratio * (1 + 1e-12))
    assert _resolved_line_core_wavelengths([], temperature, HELIUM_MASS).size == 0
    with pytest.raises(ValueError):
        _resolved_line_core_wavelengths([center], 0.0, HELIUM_MASS)
    with pytest.raises(ValueError):
        _resolved_line_core_wavelengths([center], temperature, HELIUM_MASS, step_ratio=1.0)


@pytest.fixture(scope="module")
def seed():
    return helium_continuum_atmosphere(30_000.0, 8.0, n_depth=20, rosseland_frequency_points=80)


def _window_integral(opacity, grid, center):
    inside = np.unique(grid[(grid >= center - 5.0) & (grid <= center + 5.0)])
    values = opacity(inside)
    return trapezoid(values, inside, axis=0 if values.shape[0] == inside.size else 1)


def _reference(opacity, center):
    offsets = np.geomspace(1.0e-6, 5.0, 3000)
    return _window_integral(opacity, np.r_[center - offsets[::-1], center, center + offsets], center)


def test_he_ii_euv_core_area_is_recovered_on_the_structure_grid(seed):
    line = next(line for line in HELIUM_II_LINES if line.name == "HeII1-2")
    center = line.wavelength_vacuum_angstrom

    def opacity(wave):
        return helium_ii_line_mass_absorption_coefficient(seed, wave, lines=(line,))

    reference = _reference(opacity, center)
    linear = center + np.arange(-5.0, 5.0001, 0.2)
    resolved = np.r_[linear, _resolved_line_core_wavelengths([center], seed.temperature.min(), HELIUM_MASS)]
    # The former linear grid weights the sampled centre by a whole 0.2 A step.
    assert np.max(_window_integral(opacity, linear, center) / reference) > 5.0
    np.testing.assert_allclose(_window_integral(opacity, resolved, center), reference, rtol=0.015)


def test_he_i_resonance_core_area_is_recovered_on_the_structure_grid(seed):
    centers = [line.wavelength_vacuum_angstrom for line in HELIUM_I_RESONANCE_LINES]
    center = centers[0]

    def opacity(wave):
        return helium_i_resonance_line_mass_absorption_coefficient(seed, wave, include_occupation_probability=True)

    # The reference resolves every resonance core inside the 584 A window.
    offsets = np.geomspace(1.0e-6, 5.0, 3000)
    reference = _window_integral(opacity, np.r_[center - offsets[::-1], center, center + offsets], center)
    uv = np.arange(480.0, 700.0001, 0.5)
    resolved = np.r_[uv, _resolved_line_core_wavelengths(centers, seed.temperature.min(), HELIUM_MASS)]
    # The former 0.5 A grid misses the 584 A core between its points.
    assert np.min(_window_integral(opacity, uv, center) / reference) < 0.1
    np.testing.assert_allclose(_window_integral(opacity, resolved, center), reference, rtol=0.015)
