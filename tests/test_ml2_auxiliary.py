import numpy as np
import pytest

from wd_spectra._ml2_auxiliary import (
    ml2_scaled_coefficients, ml2_auxiliary_from_gradient, ml2_auxiliary_compatibility,
    solve_auxiliary_ml2_experiment,
)
from wd_spectra.nonlinear import NonlinearEvaluation, solve_trust_region_newton


def test_signed_ml2_coordinate_preserves_flux_and_stable_branch():
    loss = np.geomspace(1e-6, 1e2, 7)
    coefficient = np.geomspace(1e6, 1e30, 7)
    target = 1e10
    y = np.array([-4., -0.3, 0., 0.2, 0.7, 1., 3.])
    a, b = ml2_scaled_coefficients(loss, coefficient, target)
    adiabatic = np.full(7, 0.4)
    gradient = adiabatic + a*y + b*y*abs(y)
    recovered = ml2_auxiliary_from_gradient(gradient, adiabatic, loss, coefficient, target)
    np.testing.assert_allclose(recovered, y, rtol=1e-5, atol=1e-12)
    excess = np.maximum(gradient - adiabatic, 0)
    element = 2 * excess / (loss + np.sqrt(loss**2 + 4*excess))
    np.testing.assert_allclose(coefficient * element**3 / target,
                               np.maximum(y, 0)**3, rtol=3e-5, atol=1e-12)


def test_ml2_auxiliary_tangent_includes_all_material_changes():
    n = 5
    rng = np.random.default_rng(441)
    state = rng.normal(size=n) * 0.02
    y = np.array([-2., -0.5, 0., 0.6, 1.2])
    response = rng.normal(size=(3, n, n)) * 0.01
    gradient_response = rng.normal(size=(n, n))
    def material(x):
        return tuple(base * np.exp(matrix @ x) for base, matrix in zip((0.4, 0.03, 1e14), response))
    coefficients = material(state)
    derivatives = tuple(c[:, None] * r for c, r in zip(coefficients, response))
    gradient = gradient_response @ state
    defect, tangent, y_tangent = ml2_auxiliary_compatibility(
        gradient, y, coefficients, derivatives, gradient_response, 1e10)
    for j in range(n):
        delta = np.eye(n)[j] * 1e-6
        plus = ml2_auxiliary_compatibility(gradient_response @ (state + delta), y,
                                          material(state + delta), None, None, 1e10)[0]
        minus = ml2_auxiliary_compatibility(gradient_response @ (state - delta), y,
                                           material(state - delta), None, None, 1e10)[0]
        np.testing.assert_allclose((plus - minus)/2e-6, tangent[:, j], atol=1e-9)
    plus = ml2_auxiliary_compatibility(gradient, y+1e-6, coefficients, None, None, 1e10)[0]
    minus = ml2_auxiliary_compatibility(gradient, y-1e-6, coefficients, None, None, 1e10)[0]
    np.testing.assert_allclose((plus-minus)/2e-6, y_tangent, atol=3e-9)


def test_zero_loss_zero_excess_is_finite():
    np.testing.assert_array_equal(
        ml2_auxiliary_from_gradient(np.array([.4]), np.array([.4]),
                                    np.array([0.]), np.array([1e20]), 1e10), [0.])


@pytest.mark.parametrize("physical_offset", [0., .1])
@pytest.mark.parametrize("local_scaling", [False, True])
def test_enlarged_system_tangent_and_independent_physical_gate(physical_offset, local_scaling):
    target = np.array([.2, .400001, .400003])
    adiabatic = np.full(3, .4)
    loss = np.full(3, 1e-3)
    coefficient = np.full(3, 1e8)
    def convection(state):
        y = ml2_auxiliary_from_gradient(state, adiabatic, loss, coefficient, 1.)
        return np.maximum(y, 0)**3
    target_convection = convection(target)
    def evaluate(state, need_jacobian):
        radiation = 1 - target_convection + .2*(state-target)
        actual_flux = radiation + convection(state) + physical_offset
        payload = {
            "energy_balance_is_physical_flux": True,
            "convection_transport": {"adiabatic_gradient": adiabatic,
                                     "ml2_radiative_loss": loss,
                                     "ml2_flux_coefficient": coefficient},
            "temperature_gradient": state,
            "radiative_flux_interface": radiation,
            "total_flux_interface": actual_flux,
            "radiative_cell_energy_defect": np.diff(radiation),
            "cell_energy_scale": np.array([1e-4, 10.]),
            "log_temperature_from_state": np.eye(3),
            "interface_gradient_operator": np.eye(3),
            "radiative_flux_log_temperature_jacobian": .2*np.eye(3),
            "ml2_coefficient_log_temperature_responses": (np.zeros((3, 3)),)*3,
        }
        return NonlinearEvaluation(actual_flux-1, None, payload)
    def audited_solver(initial, augmented, **options):
        base = augmented(initial, True)
        for j in range(len(initial)):
            delta = np.eye(len(initial))[j] * 1e-7
            numerical = (augmented(initial+delta, False).residual -
                         augmented(initial-delta, False).residual) / 2e-7
            np.testing.assert_allclose(numerical, base.jacobian[:, j], rtol=2e-8, atol=1e-5)
        return solve_trust_region_newton(initial, augmented, **options)
    result, metadata = solve_auxiliary_ml2_experiment(
        target, evaluate, audited_solver, 1., maximum_iterations=2,
        residual_tolerance=1e-6, finite_difference_fallback_step=None,
        local_energy_scaling=local_scaling)
    assert result.converged == (physical_offset == 0.)
    assert result.state.shape == (3,)
    assert metadata["energy_residual_maximum"] < 1e-8
    assert result.diagnostics.final_residual_maximum == pytest.approx(physical_offset, abs=1e-8)
