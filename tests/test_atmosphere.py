import numpy as np
import pytest

from wd_spectra import (
    gray_hydrogen_atmosphere,
    gray_hydrogen_helium_atmosphere,
    hydrogen_continuum_atmosphere,
    radiative_equilibrium_helium_atmosphere,
)
from wd_spectra.atmosphere import (
    _smooth_radiative_temperature_correction,
    _solve_bracketed_log_root,
)


def test_gray_atmosphere_obeys_structure_equations():
    atmosphere = gray_hydrogen_atmosphere(
        12_000.0, 8.0, rosseland_opacity=0.2
    )
    tau = atmosphere.rosseland_optical_depth

    np.testing.assert_allclose(
        (atmosphere.temperature / atmosphere.effective_temperature) ** 4,
        0.75 * (tau + 2.0 / 3.0),
        rtol=2e-15,
    )
    np.testing.assert_allclose(
        atmosphere.gas_pressure,
        atmosphere.gravity * atmosphere.column_mass,
        rtol=2e-15,
    )
    np.testing.assert_allclose(atmosphere.column_mass, tau / 0.2)
    assert np.all(np.diff(tau) > 0.0)
    assert np.all(np.diff(atmosphere.gas_pressure) > 0.0)


def test_safeguarded_secant_root_matches_bisection_with_few_evaluations():
    calls = 0

    def residual(value: float) -> float:
        nonlocal calls
        calls += 1
        return value + 0.04 * np.exp(value) - 2.0

    root = _solve_bracketed_log_root(residual, -20.0, 20.0)
    lower, upper = -20.0, 20.0
    for _ in range(70):
        midpoint = 0.5 * (lower + upper)
        if midpoint + 0.04 * np.exp(midpoint) - 2.0 < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    reference = 0.5 * (lower + upper)
    np.testing.assert_allclose(root, reference, atol=2.0e-11)
    assert calls < 35


def test_smoothed_radiative_correction_does_not_leak_into_convection():
    correction = np.array([0.0, 0.04, 0.04, -0.04, -0.04, 0.0])
    convective_weight = np.array([0.0, 0.0, 1.0, 1.0, 0.5, 0.0])

    smoothed = _smooth_radiative_temperature_correction(
        correction, convective_weight
    )

    np.testing.assert_allclose(smoothed[2:4], 0.0)
    assert abs(smoothed[4]) <= 0.5 * abs(correction[4])


@pytest.mark.parametrize("damping", [0.0, -0.1, 1.01, np.nan])
def test_helium_temperature_correction_damping_must_be_bounded(damping):
    with pytest.raises(ValueError, match="temperature_correction_damping"):
        radiative_equilibrium_helium_atmosphere(
            12_000.0,
            8.0,
            stark_table=None,
            temperature_correction_damping=damping,
        )


@pytest.mark.parametrize("damping", [0.0, -0.1, 1.01, np.nan])
def test_helium_convective_correction_damping_must_be_bounded(damping):
    with pytest.raises(ValueError, match="convective_correction_damping"):
        radiative_equilibrium_helium_atmosphere(
            12_000.0,
            8.0,
            stark_table=None,
            convective_correction_damping=damping,
        )


@pytest.mark.parametrize("tolerance", [0.0, -0.1, np.nan])
def test_helium_convective_gradient_tolerance_must_be_positive(tolerance):
    with pytest.raises(ValueError, match="convective_gradient_tolerance"):
        radiative_equilibrium_helium_atmosphere(
            12_000.0,
            8.0,
            stark_table=None,
            convective_gradient_tolerance=tolerance,
        )


@pytest.mark.parametrize("tolerance", [0.0, -0.1, np.nan])
def test_helium_convective_flux_tolerance_must_be_positive(tolerance):
    with pytest.raises(ValueError, match="convective_flux_tolerance"):
        radiative_equilibrium_helium_atmosphere(
            12_000.0,
            8.0,
            stark_table=None,
            convective_flux_tolerance=tolerance,
        )


@pytest.mark.parametrize("order", [0, 7])
def test_mixed_structure_balmer_quadrature_order_must_be_bounded(order):
    with pytest.raises(ValueError, match="quadrature_order"):
        radiative_equilibrium_helium_atmosphere(
            12_000.0,
            8.0,
            stark_table=None,
            hydrogen_self_broadening_quadrature_order=order,
        )


def test_helium_consecutive_convergence_count_must_be_positive():
    with pytest.raises(ValueError, match="consecutive_convergence_iterations"):
        radiative_equilibrium_helium_atmosphere(
            12_000.0,
            8.0,
            stark_table=None,
            consecutive_convergence_iterations=0,
        )


def test_continuum_atmosphere_caches_consistent_hm_populations():
    atmosphere = hydrogen_continuum_atmosphere(12_000.0, 8.0, n_depth=12)
    state = atmosphere.hydrogen_lte_state
    assert state is not None
    assert state.level_population_density is not None
    assert state.level_occupation_probability is not None
    np.testing.assert_allclose(
        np.sum(state.level_population_density, axis=-1),
        atmosphere.neutral_h_density,
        rtol=3.0e-15,
    )


def test_gray_mixed_atmosphere_caches_both_species_and_total_density():
    atmosphere = gray_hydrogen_helium_atmosphere(
        20_000.0, 8.0, -2.0, n_depth=8
    )
    hydrogen = atmosphere.hydrogen_lte_state
    helium = atmosphere.helium_lte_state
    assert hydrogen is not None
    assert helium is not None
    np.testing.assert_allclose(
        hydrogen.hydrogen_nuclei_density / helium.helium_nuclei_density,
        1.0e-2,
        rtol=2.0e-14,
    )
    np.testing.assert_allclose(
        atmosphere.mass_density,
        hydrogen.mass_density + helium.mass_density,
        rtol=2.0e-15,
    )
    np.testing.assert_allclose(
        atmosphere.electron_density,
        hydrogen.proton_density
        + helium.singly_ionized_he_density
        + 2.0 * helium.doubly_ionized_he_density,
        rtol=2.0e-11,
    )
    assert atmosphere.metadata["composition"] == "homogeneous-hydrogen-helium"
