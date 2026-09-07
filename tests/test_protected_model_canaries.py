"""Slow end-to-end checks for the solver behavior protected by v0.1.1."""

import logging
import time
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


@pytest.fixture(autouse=True)
def _live_nonlinear_diagnostics(capsys, request):
    """Also show Jacobian work and rejected directions between accepted steps."""
    class LiveHandler(logging.Handler):
        def emit(self, record):
            with capsys.disabled():
                print(f"[{request.node.name}] {record.getMessage()}", flush=True)

    logger = logging.getLogger("wd_spectra.nonlinear")
    handler = LiveHandler()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def _progress_callback(label, capsys):
    """Emit live convergence telemetry even while pytest capture is active."""

    start = time.monotonic()
    with capsys.disabled():
        print(f"[{label}] starting cold atmosphere (no checkpoint)", flush=True)

    def report(iteration, _atmosphere, diagnostics):
        maximum_flux = diagnostics.get(
            "maximum_all_depth_total_flux_residual",
            diagnostics.get("maximum_total_flux_residual", np.nan),
        )
        message = (
            f"[{label}] elapsed={time.monotonic() - start:.1f}s "
            f"iteration={iteration} "
            f"phase={diagnostics.get('solver_phase', 'unknown')} "
            f"max_flux={float(maximum_flux):.6g} "
            "max_dlnT="
            f"{float(diagnostics.get('maximum_log_temperature_correction', np.nan)):.6g} "
            f"trust={float(diagnostics.get('trust_radius', np.nan)):.6g} "
            f"line_search={float(diagnostics.get('line_search_factor', np.nan)):.6g}"
        )
        # sys.__stdout__ still points at pytest's captured file descriptor.
        # Explicitly suspend capture so a stalled canary remains observable.
        with capsys.disabled():
            print(message, flush=True)

    return report


@pytest.mark.parametrize(
    "effective_temperature,maximum_expected_iterations",
    [(10_000.0, 30), (22_000.0, 60)],
)
def test_protected_db_cold_starts_converge_without_fallback(
    effective_temperature,
    maximum_expected_iterations,
    capsys,
):
    with warnings.catch_warnings():
        warnings.simplefilter("error", AtmosphereConvergenceWarning)
        result = compute_db(
            DBConfig(
                effective_temperature=effective_temperature,
                quality="production",
            ),
            _MINIMAL_FORMAL_WAVELENGTH,
            iteration_callback=_progress_callback(
                f"DB-{effective_temperature:g}K-production", capsys
            ),
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


def test_standard_db_22000_enters_exact_flux_verification(capsys):
    with warnings.catch_warnings():
        warnings.simplefilter("error", AtmosphereConvergenceWarning)
        result = compute_db(
            DBConfig(effective_temperature=22_000.0, quality="standard"),
            _MINIMAL_FORMAL_WAVELENGTH,
            iteration_callback=_progress_callback("DB-22000K-standard", capsys),
        )

    metadata = result.atmosphere.metadata
    assert result.atmosphere.n_depth == 40
    assert metadata["radiative_equilibrium_converged"]
    assert metadata["maximum_all_depth_total_flux_residual"] < 3.0e-3
    assert metadata["radiative_equilibrium_iterations"] <= 30
    assert metadata["radiative_equilibrium_maximum_log_temperature_correction"] < 3e-4
    assert metadata["nonlinear_solver_segments"][-1]["phase"] == (
        "formal-radiative-flux-completion"
    )
    assert metadata["nonlinear_solver_terminal_reason"] in (
        "initial-state-converged",
        "residual-and-step-converged",
        "stationary-residual-converged",
    )
    assert not metadata["initial_temperature_was_supplied"]


@pytest.mark.parametrize(
    "effective_temperature,maximum_expected_iterations",
    [(5_000.0, 45), (20_000.0, 45)],
)
def test_protected_da_cold_starts_converge_without_fallback(
    effective_temperature,
    maximum_expected_iterations,
    capsys,
):
    with warnings.catch_warnings():
        warnings.simplefilter("error", AtmosphereConvergenceWarning)
        result = compute_da(
            DAConfig(
                effective_temperature=effective_temperature,
                quality="production",
            ),
            _MINIMAL_FORMAL_WAVELENGTH,
            iteration_callback=_progress_callback(
                f"DA-{effective_temperature:g}K-production", capsys
            ),
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


@pytest.mark.parametrize(
    "effective_temperature,maximum_expected_iterations",
    [(3_000.0, 120), (4_000.0, 60)],
)
def test_ultracool_da_cold_starts_converge_with_exact_flux_verification(
    effective_temperature,
    maximum_expected_iterations,
    capsys,
):
    with warnings.catch_warnings():
        warnings.simplefilter("error", AtmosphereConvergenceWarning)
        result = compute_da(
            DAConfig(
                effective_temperature=effective_temperature,
                quality="production",
            ),
            _MINIMAL_FORMAL_WAVELENGTH,
            iteration_callback=_progress_callback(
                f"DA-{effective_temperature:g}K-production", capsys
            ),
        )

    metadata = result.atmosphere.metadata
    assert result.atmosphere.n_depth == 100
    assert metadata["radiative_equilibrium_converged"]
    assert metadata["maximum_all_depth_total_flux_residual"] < 2.0e-3
    assert (
        metadata["radiative_equilibrium_maximum_log_temperature_correction"]
        < 2.0e-4
    )
    assert metadata["radiative_equilibrium_iterations"] <= (
        maximum_expected_iterations
    )
    assert metadata["formal_flux_completion_used"]
    assert metadata["adiabatic_asymptotic_conditioning_enabled"]
    assert metadata["adiabatic_asymptotic_interfaces"] > 0
    assert metadata["initial_convective_gradient_projection_mode"] == (
        "interface-transport"
    )
    assert not metadata["initial_bolometric_rescaling_enabled"]
    assert not metadata["initial_temperature_was_supplied"]
    assert result.metadata["atmosphere_initialization"] == "gray"
    assert result.metadata["atmosphere_convergence_status"] == "converged"
