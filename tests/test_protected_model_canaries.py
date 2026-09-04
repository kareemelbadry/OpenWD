"""Slow end-to-end checks for the solver behavior protected by v0.1.1."""

import warnings

import numpy as np
import pytest

from wd_spectra.models import (
    AtmosphereConvergenceWarning,
    DAConfig,
    DBConfig,
    compute_da,
    compute_db,
)


pytestmark = pytest.mark.canary

_MINIMAL_FORMAL_WAVELENGTH = np.array([4_000.0, 5_000.0])


@pytest.mark.parametrize(
    "effective_temperature,maximum_expected_iterations",
    [(10_000.0, 30), (22_000.0, 60)],
)
def test_protected_db_cold_starts_converge_without_fallback(
    effective_temperature,
    maximum_expected_iterations,
):
    with warnings.catch_warnings():
        warnings.simplefilter("error", AtmosphereConvergenceWarning)
        result = compute_db(
            DBConfig(
                effective_temperature=effective_temperature,
                quality="production",
            ),
            _MINIMAL_FORMAL_WAVELENGTH,
        )

    metadata = result.atmosphere.metadata
    assert result.atmosphere.n_depth == 80
    assert metadata["radiative_equilibrium_converged"]
    assert metadata["maximum_all_depth_total_flux_residual"] < 3.0e-3
    assert (
        metadata["radiative_equilibrium_maximum_log_temperature_correction"]
        < 3.0e-4
    )
    assert metadata["radiative_equilibrium_iterations"] <= (
        maximum_expected_iterations
    )
    assert not metadata["initial_temperature_was_supplied"]
    assert result.metadata["atmosphere_convergence_status"] == "converged"
    assert not result.metadata["checkpoint_matches_model_request"]


def test_standard_db_22000_enters_exact_flux_verification():
    with warnings.catch_warnings():
        warnings.simplefilter("error", AtmosphereConvergenceWarning)
        result = compute_db(
            DBConfig(effective_temperature=22_000.0, quality="standard"),
            _MINIMAL_FORMAL_WAVELENGTH,
        )

    metadata = result.atmosphere.metadata
    assert result.atmosphere.n_depth == 40
    assert metadata["radiative_equilibrium_converged"]
    assert metadata["maximum_all_depth_total_flux_residual"] < 3.0e-3
    assert metadata["radiative_equilibrium_iterations"] <= 30
    assert metadata["nonlinear_solver_segments"][-1]["phase"] == (
        "formal-radiative-flux-completion"
    )
    assert metadata["nonlinear_solver_terminal_reason"] in (
        "initial-state-converged",
        "residual-and-step-converged",
    )
    assert not metadata["initial_temperature_was_supplied"]


@pytest.mark.parametrize(
    "effective_temperature,maximum_expected_iterations",
    [(5_000.0, 45), (20_000.0, 45)],
)
def test_protected_da_cold_starts_converge_without_fallback(
    effective_temperature,
    maximum_expected_iterations,
):
    with warnings.catch_warnings():
        warnings.simplefilter("error", AtmosphereConvergenceWarning)
        result = compute_da(
            DAConfig(
                effective_temperature=effective_temperature,
                quality="production",
            ),
            _MINIMAL_FORMAL_WAVELENGTH,
        )

    metadata = result.atmosphere.metadata
    assert result.atmosphere.n_depth == 100
    assert metadata["radiative_equilibrium_converged"]
    assert metadata["maximum_total_flux_residual"] < 3.0e-3
    assert (
        metadata["radiative_equilibrium_maximum_log_temperature_correction"]
        < 3.0e-4
    )
    assert metadata["radiative_equilibrium_iterations"] <= (
        maximum_expected_iterations
    )
    assert result.metadata["atmosphere_initialization"] == "gray"
    assert result.metadata["atmosphere_convergence_status"] == "converged"
