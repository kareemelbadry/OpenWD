"""Opt-in magneto-ionic free electrons with the cyclotron resonance (include_cyclotron_absorption).

The opacity checks are those of the original implementation (upstream tests/test_dah.py before PR 61).
"""

from pathlib import Path

import numpy as np
import pytest

from wd_spectra._compat import trapezoid
from wd_spectra.atmosphere import gray_hydrogen_atmosphere
from wd_spectra.constants import ELECTRON_MASS, ELEMENTARY_CHARGE_ESU, LIGHT_SPEED, PI
from wd_spectra.magnetic import (
    MagneticPhysics, synthesize_magnetic_hydrogen_spectrum, uniform_field_surface_cells,
)
from wd_spectra.magnetic_atomic import read_h2db_energy_database, read_h2db_transition_database
from wd_spectra.magnetic_continuum import (
    hydrogen_free_free_mass_absorption_coefficient,
    isotropic_free_electron_opacity,
    magnetized_free_electron_manifolds,
)
from wd_spectra.models.common import ModelData
from wd_spectra.opacity import electron_scattering_mass_coefficient

# Stokes I and V with the option off, written by fork main 3b71c0e before the option existed.
OFF_REFERENCE = Path(__file__).parent / "data/approved_regressions/current/dah-cyclotron-off-300mg.npz"


@pytest.fixture(scope="module")
def atmosphere():
    return gray_hydrogen_atmosphere(15_000.0, 8.0, n_depth=24, correlated_microfields=True)


def test_free_electrons_recover_zero_field_free_free_and_thomson(atmosphere):
    wavelength = np.geomspace(1_000.0, 50_000.0, 400)
    absorption, scattering = isotropic_free_electron_opacity(atmosphere, wavelength, 1.0e-7)
    np.testing.assert_allclose(
        absorption, hydrogen_free_free_mass_absorption_coefficient(atmosphere, wavelength), rtol=1e-8
    )
    np.testing.assert_allclose(
        scattering,
        np.broadcast_to(electron_scattering_mass_coefficient(atmosphere)[None, :], scattering.shape),
        rtol=1e-8,
    )


def test_cyclotron_resonance_has_the_classical_strength_and_far_field_limit(atmosphere):
    field = 200.0
    omega_c = ELEMENTARY_CHARGE_ESU * field * 1e6 / (ELECTRON_MASS * LIGHT_SPEED)
    center = 2.0 * PI * LIGHT_SPEED / omega_c * 1e8
    wavelength = center * np.linspace(0.97, 1.03, 40001)
    manifolds = magnetized_free_electron_manifolds(atmosphere, wavelength, field, 0.7)
    depth = 12
    extinction = manifolds.absorption[2][:, depth] + manifolds.scattering[2][:, depth]
    baseline = manifolds.absorption[1][:, depth] + manifolds.scattering[1][:, depth]
    angular = 2.0 * PI * LIGHT_SPEED / (wavelength * 1e-8)
    per_electron = -trapezoid(extinction - baseline, angular) * (
        atmosphere.mass_density[depth] / atmosphere.electron_density[depth]
    )
    # Integrated resonant mode cross-section 4 pi^2 e^2 / (m_e c), the classical oscillator.
    assert per_electron == pytest.approx(
        4.0 * PI**2 * ELEMENTARY_CHARGE_ESU**2 / (ELECTRON_MASS * LIGHT_SPEED), rel=0.03
    )
    # Far from resonance every mode tends to the zero-field value.
    far = magnetized_free_electron_manifolds(atmosphere, np.array([500.0, 600.0]), 0.1, 0.5)
    for triple in (far.absorption, far.scattering):
        np.testing.assert_allclose(triple[0], triple[2], rtol=1e-3)


def test_cyclotron_dispersion_has_the_line_sign_convention(atmosphere):
    # Positive on the low-frequency (red) side, as frequency_hilbert_dispersion for lines.
    omega_c = ELEMENTARY_CHARGE_ESU * 200e6 / (ELECTRON_MASS * LIGHT_SPEED)
    center = 2.0 * PI * LIGHT_SPEED / omega_c * 1e8
    grid = center * np.linspace(0.95, 1.05, 2001)
    dispersion = magnetized_free_electron_manifolds(atmosphere, grid, 200.0, 0.5).dispersion[2]
    assert dispersion[np.searchsorted(grid, center * 1.003), 12] > 0.0
    assert dispersion[np.searchsorted(grid, center * 0.997), 12] < 0.0


def test_option_off_is_unchanged_and_on_changes_the_polarization():
    data = ModelData.default().h2db_balmer_subset
    atmosphere = gray_hydrogen_atmosphere(17_000.0, 8.0, n_depth=24, correlated_microfields=True)
    cells = uniform_field_surface_cells(300.0, field_angle_deg=40.0, signed_geometry=True)
    with np.load(OFF_REFERENCE) as reference:
        wavelength, flux_i, flux_v = reference["wavelength"], reference["I"], reference["V"]
    spectra = {}
    for option in (False, True):
        physics = MagneticPhysics(
            read_h2db_transition_database(data), read_h2db_energy_database(data),
            include_rwa_photoionization=True, include_magnetic_eos=True, line_regime="h2db",
            include_cyclotron_absorption=option,
        )
        spectra[option] = synthesize_magnetic_hydrogen_spectrum(
            atmosphere, wavelength, cells, physics,
            return_stokes=True, polarized_transfer="full-stokes-iquv",
        )
    off, on = spectra[False], spectra[True]
    np.testing.assert_allclose(off.surface_flux_lambda, flux_i, rtol=2e-8, atol=0.0)
    np.testing.assert_allclose(off.surface_flux_v_lambda, flux_v, rtol=2e-8, atol=1e-12 * flux_i.max())
    assert off.metadata["free_electrons"].startswith("zero-field")
    assert on.metadata["free_electrons"].startswith("magneto-ionic")
    # lambda_c = 3571 A at 300 MG: the resonance and its wing move V/I by percents.
    shift = np.abs(on.surface_flux_v_lambda / on.surface_flux_lambda - flux_v / flux_i)
    assert shift.max() > 0.01
