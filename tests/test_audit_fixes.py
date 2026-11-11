"""Regression tests for the verified 2026-11-10 audit findings.

Each test pins one defect: a physics constant, a sum rule, or an input that
used to crash or return NaN.
"""

import numpy as np
import pytest

import wd_spectra.validation.magnetic_da as magnetic_da
from wd_spectra.atmosphere import (
    _metal_line_opacity_sampling_grid,
    gray_hydrogen_atmosphere,
)
from wd_spectra.helium import helium_minus_free_free_description
from wd_spectra.magnetic_continuum import _rwa_branching_fractions
from wd_spectra.radiative_transfer import (
    coherent_scattering_feautrier_field,
    feautrier_radiation_field,
    radiation_field,
)
from wd_spectra.spectrum import synthesize_balmer_spectrum
from wd_spectra.validation.observed import (
    convolve_constant_resolving_power,
    convolve_variable_log_gaussian,
)


@pytest.mark.parametrize("n", range(1, 8))
def test_rwa_branching_fractions_obey_zero_field_sum_rule(n):
    """sum_l (2l+1)(plus + minus) = n**2 at every photoelectron energy.

    The n = 7, l = 6 minus branch was 3x too small.
    """

    x = np.array([1.0e-3, 0.1, 1.0, 10.0, 49.0, 1.0e3, 1.0e6])
    total = np.zeros_like(x)
    for l in range(n):
        plus, minus = _rwa_branching_fractions(n, l, x)
        total += (2 * l + 1) * (plus + minus)
    np.testing.assert_allclose(total, n**2, rtol=1.0e-12)


def test_dah_estimated_noise_recovers_white_noise_sigma():
    """The high-pass scatter keeps 0.91 of sigma, not 0.8."""

    rng = np.random.default_rng(20261110)
    ratios = [
        np.median(
            magnetic_da._estimated_noise(1.0 + 0.02 * rng.standard_normal(20_000))
        )
        / 0.02
        for _ in range(10)
    ]
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.01)


def test_weak_field_scoring_ignores_nonpositive_epoch_pixels(monkeypatch):
    """A negative pixel used to misalign flux and normalized arrays."""

    original = magnetic_da._weak_epochs("gd9", None)
    damaged = []
    for wavelength, flux, inverse_variance in original:
        flux = flux.copy()
        center = np.argmin(np.abs(wavelength - 6564.6))
        flux[center] = -abs(flux[center])
        damaged.append((wavelength, flux, inverse_variance))
    monkeypatch.setattr(magnetic_da, "_weak_epochs", lambda key, root: damaged)
    wavelength = np.linspace(3_800.0, 7_000.0, 64_001)
    flux = np.ones_like(wavelength)
    for _, center, _, _ in magnetic_da.DAH_WEAK_LINE_WINDOWS:
        flux *= 1.0 - 0.5 * np.exp(-0.5 * ((wavelength - center) / 8.0) ** 2)
    score = magnetic_da.score_dah_weak_field_profiles("gd9", wavelength, flux)
    assert np.isfinite(score["mean_rms"])


@pytest.mark.parametrize("solver", ["feautrier", "scattering"])
def test_feautrier_solvers_reject_zero_first_optical_depth(solver):
    tau = np.array([0.0, 0.1, 1.0, 10.0, 100.0])
    source = np.array([1.0, 1.2, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="positive at the first depth"):
        if solver == "feautrier":
            feautrier_radiation_field(tau, source)
        else:
            coherent_scattering_feautrier_field(
                tau, source, np.full(5, 0.5), np.full(5, 0.5)
            )
    # The directional solver integrates from tau = 0 and stays valid.
    field = radiation_field(tau, source)
    assert np.all(np.isfinite(field.flux))


def test_constant_resolution_convolution_merges_coincident_wavelengths():
    wavelength = np.linspace(4_000.0, 4_010.0, 201)
    wavelength = np.sort(
        np.append(wavelength, np.nextafter(wavelength[100], np.inf))
    )
    flux = 1.0 + 0.1 * np.sin(wavelength)
    result = convolve_constant_resolving_power(wavelength, flux, 20_000.0)
    assert result.shape == wavelength.shape
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("spacing", [100.0, 300.0])
def test_lsf_convolutions_stay_finite_on_coarse_source_grids(spacing):
    """Source grids much coarser than the LSF used to give NaN (0/0)."""

    wavelength = np.arange(3_800.0, 9_200.0 + spacing, spacing)
    flux = 1.0 + 1.0e-4 * (wavelength - 6_000.0)
    output = np.linspace(4_000.0, 9_000.0, 3_000)
    sigma = np.full(output.size, 2.3e-4)
    variable = convolve_variable_log_gaussian(wavelength, flux, output, sigma)
    constant = convolve_constant_resolving_power(
        wavelength, flux, 1_800.0, output_wavelength_angstrom=output
    )
    expected = 1.0 + 1.0e-4 * (output - 6_000.0)
    np.testing.assert_allclose(variable, expected, rtol=2.0e-3)
    np.testing.assert_allclose(constant, expected, rtol=2.0e-3)


def test_opacity_sampling_grid_without_lines_in_planck_band_is_empty():
    grid, n_wing = _metal_line_opacity_sampling_grid(
        np.array([1_334.5, 1_335.7]),
        opacity_sampling_resolution=1_000.0,
        effective_temperature=4_000.0,
    )
    assert grid.size == 0
    assert n_wing == 0


def test_balmer_synthesis_refuses_array_continuum_with_h2_lines():
    atmosphere = gray_hydrogen_atmosphere(10_000.0, 8.0, n_depth=6)
    wavelength = np.linspace(4_000.0, 7_000.0, 50)
    opacity = np.full((wavelength.size, atmosphere.n_depth), 0.1)
    with pytest.raises(ValueError, match="scalar continuum_opacity"):
        synthesize_balmer_spectrum(
            atmosphere,
            wavelength,
            continuum_opacity=opacity,
            h2_line_data=object(),
        )


@pytest.mark.parametrize(
    "prescription", ["automatic", "john1968", "john1994", "partial-wave"]
)
def test_helium_minus_metadata_names_the_prescription_used(prescription):
    descriptions = {
        name: helium_minus_free_free_description(name)
        for name in ("automatic", "john1968", "john1994", "partial-wave")
    }
    assert len(set(descriptions.values())) == 4
    assert descriptions[prescription]
    with pytest.raises(ValueError):
        helium_minus_free_free_description("unknown")
