from copy import deepcopy
import numpy as np
import pytest
from wd_spectra._convergence import equilibrium_certificate, recorded_equilibrium_status


def measured():
    return dict(
        radiative_equilibrium_solver_converged=True,
        maximum_all_depth_total_flux_residual=1e-5,
        maximum_relative_cell_energy_balance_residual=1e-5,
        temperature_correction_measured=True,
        radiative_equilibrium_maximum_log_temperature_correction=1e-5,
        maximum_unrestricted_log_temperature_correction=1e-5,
        electron_scattering_source_final_maximum_relative_residual=1e-12,
        lower_boundary_absorption_escape_bound=1e-5,
    )


def test_complete_measured_evidence_is_required():
    data = measured()
    data["equilibrium_certificate"] = equilibrium_certificate(data)
    assert recorded_equilibrium_status(data) == "converged"
    assert (
        recorded_equilibrium_status({"radiative_equilibrium_converged": True})
        == "unknown"
    )
    assert not data["equilibrium_certificate"]["independent_grid_validation"]
    assert not data["equilibrium_certificate"]["full_physics_validation"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("maximum_all_depth_total_flux_residual", 0.01),
        ("maximum_relative_cell_energy_balance_residual", 0.24),
        ("temperature_correction_measured", False),
        ("maximum_unrestricted_log_temperature_correction", 0.04),
        ("electron_scattering_source_final_maximum_relative_residual", 1e-3),
        ("lower_boundary_absorption_escape_bound", 0.1),
        ("radiative_equilibrium_solver_converged", False),
        ("maximum_relative_cell_energy_balance_residual", np.nan),
        ("maximum_relative_cell_energy_balance_residual", None),
    ],
)
def test_one_failed_check_cannot_be_hidden_by_old_true_flag(key, value):
    data = measured()
    data["equilibrium_certificate"] = equilibrium_certificate(data)
    data["radiative_equilibrium_converged"] = True
    data[key] = value
    assert recorded_equilibrium_status(data) == "unconverged"


def test_initial_state_zero_is_not_a_measured_correction():
    data = measured()
    data.update(
        temperature_correction_measured=False,
        maximum_unrestricted_log_temperature_correction=None,
        radiative_equilibrium_maximum_log_temperature_correction=0.0,
    )
    report = equilibrium_certificate(data)
    assert not report["verified"]
    assert report["failures"] == ["temperature_stationarity"]


@pytest.mark.parametrize("checks", [None, [], {"all_depth_flux": None}])
def test_malformed_checkpoint_certificate_is_unknown(checks):
    assert (
        recorded_equilibrium_status(
            dict(equilibrium_certificate=dict(schema=1, checks=checks))
        )
        == "unknown"
    )


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, True])
def test_bad_certificate_tolerance_is_rejected(value):
    with pytest.raises(ValueError):
        equilibrium_certificate(measured(), flux_tolerance=value)
