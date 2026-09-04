import numpy as np
import pytest

from wd_spectra._compat import trapezoid
from wd_spectra import (
    gray_hydrogen_atmosphere,
    planck_lambda_angstrom,
    synthesize_balmer_spectrum,
    synthesize_gray_spectrum,
)
from wd_spectra.constants import STEFAN_BOLTZMANN


def test_planck_function_integrates_to_sigma_t4_over_pi():
    temperature = 13_000.0
    wavelength = np.geomspace(1.0, 1.0e8, 30_000)
    integral = trapezoid(
        planck_lambda_angstrom(wavelength, temperature), wavelength
    )
    np.testing.assert_allclose(
        np.pi * integral, STEFAN_BOLTZMANN * temperature**4, rtol=1.0e-6
    )


@pytest.mark.parametrize("effective_temperature", [6_000.0, 12_000.0, 40_000.0])
def test_gray_spectrum_has_requested_bolometric_flux(effective_temperature):
    atmosphere = gray_hydrogen_atmosphere(
        effective_temperature, 8.0, n_depth=48
    )
    wavelength = np.geomspace(1.0, 1.0e8, 5_000)
    spectrum = synthesize_gray_spectrum(
        atmosphere, wavelength, backend="python"
    )
    target = STEFAN_BOLTZMANN * effective_temperature**4
    np.testing.assert_allclose(spectrum.bolometric_flux, target, rtol=2.0e-5)


def test_opacity_ratio_changes_spectral_formation_depth():
    atmosphere = gray_hydrogen_atmosphere(12_000.0, 8.0)
    wavelength = np.array([3_000.0, 6_000.0])
    gray = synthesize_gray_spectrum(atmosphere, wavelength, backend="python")
    non_gray = synthesize_gray_spectrum(
        atmosphere, wavelength, opacity_ratio=[0.1, 10.0], backend="python"
    )
    assert non_gray.surface_flux_lambda[0] > gray.surface_flux_lambda[0]
    assert non_gray.surface_flux_lambda[1] < gray.surface_flux_lambda[1]


def test_balmer_synthesis_can_disable_neutral_self_broadening():
    atmosphere = gray_hydrogen_atmosphere(6_000.0, 8.0, n_depth=24)
    wavelength = np.linspace(6540.0, 6590.0, 101)
    broadened = synthesize_balmer_spectrum(atmosphere, wavelength)
    stark_only = synthesize_balmer_spectrum(
        atmosphere,
        wavelength,
        include_balmer_self_broadening=False,
    )

    assert broadened.metadata["balmer_self_broadening"].startswith("Barklem")
    assert "Ali-Griem" in broadened.metadata["balmer_self_broadening"]
    assert stark_only.metadata["balmer_self_broadening"] == "disabled"
    assert np.max(
        np.abs(
            broadened.surface_flux_lambda - stark_only.surface_flux_lambda
        )
    ) > 0.0


def test_balmer_synthesis_exposes_impact_validity_fraction():
    atmosphere = gray_hydrogen_atmosphere(6_000.0, 8.0, n_depth=24)
    wavelength = np.linspace(6520.0, 6610.0, 181)
    tabulated_range = synthesize_balmer_spectrum(atmosphere, wavelength)
    strict_range = synthesize_balmer_spectrum(
        atmosphere,
        wavelength,
        balmer_self_broadening_impact_validity_fraction=0.2,
    )

    assert (
        strict_range.metadata[
            "balmer_self_broadening_impact_validity_fraction"
        ]
        == 0.2
    )
    assert np.max(
        np.abs(
            tabulated_range.surface_flux_lambda
            - strict_range.surface_flux_lambda
        )
    ) > 0.0


def test_balmer_synthesis_exposes_ali_griem_control():
    atmosphere = gray_hydrogen_atmosphere(6_000.0, 8.0, n_depth=24)
    wavelength = np.linspace(6520.0, 6610.0, 181)
    barklem = synthesize_balmer_spectrum(atmosphere, wavelength)
    ali_griem = synthesize_balmer_spectrum(
        atmosphere,
        wavelength,
        balmer_self_broadening_prescription="ali-griem",
    )

    assert ali_griem.metadata["balmer_self_broadening_prescription"] == "ali-griem"
    assert ali_griem.metadata["balmer_self_broadening"].startswith("Ali-Griem")
    assert np.max(
        np.abs(barklem.surface_flux_lambda - ali_griem.surface_flux_lambda)
    ) > 0.0


def test_balmer_synthesis_exposes_allard_2008_halpha_control():
    atmosphere = gray_hydrogen_atmosphere(7_000.0, 8.0, n_depth=24)
    wavelength = np.linspace(6520.0, 6610.0, 181)
    barklem = synthesize_balmer_spectrum(atmosphere, wavelength)
    allard = synthesize_balmer_spectrum(
        atmosphere,
        wavelength,
        balmer_self_broadening_prescription="allard-2008",
    )

    assert allard.metadata["balmer_self_broadening_prescription"] == "allard-2008"
    assert allard.metadata["balmer_self_broadening"].startswith("Allard et al. 2008")
    assert np.max(
        np.abs(barklem.surface_flux_lambda - allard.surface_flux_lambda)
    ) > 0.0


def test_optical_formal_spectrum_can_include_series_pseudocontinuum_without_lyman():
    atmosphere = gray_hydrogen_atmosphere(15_000.0, 8.5, n_depth=16)
    wavelength = np.linspace(3_500.0, 3_900.0, 81)
    ordinary = synthesize_balmer_spectrum(atmosphere, wavelength)
    dissolved = synthesize_balmer_spectrum(
        atmosphere,
        wavelength,
        include_lyman=False,
        include_series_pseudocontinuum=True,
    )
    assert (
        dissolved.metadata["dissolved_level_pseudocontinuum"]
        == "series-wide DAM/Q-MHD Lyman--Brackett approximation"
    )
    assert np.max(
        np.abs(dissolved.surface_flux_lambda - ordinary.surface_flux_lambda)
    ) > 0.0


def test_formal_spectrum_accepts_precomputed_lyman_opacity():
    atmosphere = gray_hydrogen_atmosphere(14_000.0, 8.0, n_depth=16)
    wavelength = np.linspace(1_180.0, 1_300.0, 121)
    ordinary = synthesize_balmer_spectrum(
        atmosphere,
        wavelength,
        include_lyman=True,
        include_neutral_lyman_alpha_wing=False,
    )
    supplied = synthesize_balmer_spectrum(
        atmosphere,
        wavelength,
        include_lyman=True,
        include_neutral_lyman_alpha_wing=False,
        precomputed_lyman_opacity=np.zeros(
            (wavelength.size, atmosphere.n_depth)
        ),
    )
    assert np.max(
        np.abs(ordinary.surface_flux_lambda - supplied.surface_flux_lambda)
    ) > 0.0

    with pytest.raises(ValueError, match="precomputed_lyman_opacity"):
        synthesize_balmer_spectrum(
            atmosphere,
            wavelength,
            include_lyman=True,
            precomputed_lyman_opacity=np.zeros((wavelength.size, 2)),
        )
