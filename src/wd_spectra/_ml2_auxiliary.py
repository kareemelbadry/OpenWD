"""Algebraic ML2 coordinates for controlled coupled-system experiments.

The signed element-velocity coordinate separates efficient-convection
stiffness from temperature differences. No material coefficient is modified.
"""

from __future__ import annotations

import numpy as np
from dataclasses import replace

from .nonlinear import (
    NonlinearEvaluation, RecoverableEvaluationError, nonlinear_result_metadata,
)
from ._energy_balance import locally_scaled_energy_rows


def ml2_scaled_coefficients(loss, coefficient, target_flux):
    """Return a,b with nabla-nabla_ad = a*y + b*y*abs(y).

    For y>=0, the original ML2 relation gives F_conv/F_star=y**3 exactly.
    The signed extension for y<0 parameterizes stable, zero-convection cells.
    It does not extend convective flux into those cells.
    """
    if target_flux <= 0 or not np.isfinite(target_flux):
        raise ValueError("auxiliary ML2 requires a positive finite stellar flux")
    if np.any(loss < 0) or np.any(~np.isfinite(loss)):
        raise ValueError("auxiliary ML2 requires finite nonnegative radiative loss")
    if np.any(coefficient <= 0) or np.any(~np.isfinite(coefficient)):
        raise ValueError("auxiliary ML2 requires positive finite flux coefficients")
    velocity_scale = np.cbrt(target_flux / coefficient)
    return loss * velocity_scale, velocity_scale**2


def ml2_auxiliary_from_gradient(gradient, adiabatic, loss, coefficient, target_flux):
    a, b = ml2_scaled_coefficients(loss, coefficient, target_flux)
    excess = np.asarray(gradient) - adiabatic
    magnitude = np.abs(excess)
    denominator = a + np.sqrt(a**2 + 4 * b * magnitude)
    root = np.divide(2 * magnitude, denominator, out=np.zeros_like(magnitude), where=denominator > 0)
    return np.sign(excess) * root


def ml2_auxiliary_compatibility(gradient, y, coefficients, responses, gradient_jacobian, target_flux):
    """Gradient compatibility and its exact material/coordinate tangent.

    Coefficients are (adiabatic gradient, radiative loss, flux coefficient).
    Responses, when supplied, differentiate those same arrays with respect
    to the temperature-state variables. The normalization is left to callers.
    """
    adiabatic, loss, coefficient = coefficients
    a, b = ml2_scaled_coefficients(loss, coefficient, target_flux)
    defect = gradient - adiabatic - a * y - b * y * np.abs(y)
    y_response = -a - 2 * b * np.abs(y)
    if responses is None:
        return defect, None, y_response
    ad_response, loss_response, coefficient_response = responses
    velocity_scale = np.cbrt(target_flux / coefficient)
    log_velocity_response = -coefficient_response / (3 * coefficient[:, None])
    a_response = loss_response * velocity_scale[:, None] + a[:, None] * log_velocity_response
    b_response = 2 * b[:, None] * log_velocity_response
    temperature_response = (
        gradient_jacobian - ad_response - y[:, None] * a_response
        - (y * np.abs(y))[:, None] * b_response
    )
    return defect, temperature_response, y_response


def solve_auxiliary_ml2_experiment(initial, evaluate, solver, target_flux,
                                   *, local_energy_scaling=False, **options):
    """Research-only enlarged system; never selected by production dispatch.

    Both the auxiliary closure and the flux evaluated from the actual T
    gradient must converge. Return the original-sized physical result plus
    separately labelled enlarged-system telemetry. No flux is overwritten.
    """
    n = len(initial)
    first = evaluate(initial, False)
    if not first.payload["energy_balance_is_physical_flux"]:
        raise ValueError("Auxiliary ML2 requires the full physical flux phase")
    if options.get("trial_projector") is not None:
        raise ValueError("Test auxiliary ML2 independently of trial projection")

    def coefficients(payload):
        transport = payload["convection_transport"]
        return tuple(np.asarray(transport[key])[1:] for key in (
            "adiabatic_gradient", "ml2_radiative_loss", "ml2_flux_coefficient"))

    adiabatic, loss, coefficient = coefficients(first.payload)
    y0 = ml2_auxiliary_from_gradient(
        first.payload["temperature_gradient"][1:], adiabatic, loss, coefficient, target_flux)
    a0, b0 = ml2_scaled_coefficients(loss, coefficient, target_flux)
    # Stable cells can have a very large signed y while carrying no flux.
    # Nondimensionalize those unknowns too: otherwise the generic least-squares
    # regularization suppresses a legitimate radiative temperature correction.
    # The fixed scale covers either F_star or the supplied gradient, whichever
    # demands a larger element coordinate. No empirical efficiency threshold.
    coordinate_scale = np.maximum(1., np.abs(y0))
    scale = a0*coordinate_scale + b0*coordinate_scale**2
    cell_scale = first.payload.get("cell_energy_scale")
    if local_energy_scaling:
        cell_scale = np.asarray(cell_scale) / target_flux
        # Validate once; this scale must not switch as the iterate changes.
        locally_scaled_energy_rows(np.zeros(n), cell_scale)

    def augmented_evaluate(state, need_jacobian):
        original = evaluate(state[:n], need_jacobian)
        payload = original.payload
        y = state[n:] * coordinate_scale
        material = coefficients(payload)
        responses = gradient_jacobian = None
        if need_jacobian:
            mapping = payload["log_temperature_from_state"]
            responses = tuple(response[1:] @ mapping for response in
                              payload["ml2_coefficient_log_temperature_responses"])
            gradient_jacobian = payload["interface_gradient_operator"][1:] @ mapping
        defect, tangent, y_tangent = ml2_auxiliary_compatibility(
            payload["temperature_gradient"][1:], y, material, responses,
            gradient_jacobian, target_flux)
        with np.errstate(over="ignore", invalid="ignore"):
            auxiliary_flux = np.maximum(y, 0)**3
        energy = np.asarray(payload["radiative_flux_interface"]) / target_flux - 1
        energy = energy.copy()
        energy[1:] += auxiliary_flux
        if local_energy_scaling:
            # Evaluate radiative divergence as absorption*(J-B), not the
            # subtraction of two almost identical thin-cell fluxes.
            cell_defect = (payload["radiative_cell_energy_defect"] / target_flux
                           + np.diff(np.concatenate(([0.], auxiliary_flux))))
            energy[:-1] -= cell_defect / cell_scale
        residual = np.concatenate((energy, defect / scale))
        if not np.all(np.isfinite(residual)):
            raise RecoverableEvaluationError("Nonfinite auxiliary ML2 trial")
        jacobian = None
        if need_jacobian:
            jacobian = np.zeros((2*n-1, 2*n-1))
            jacobian[:n, :n] = payload["radiative_flux_log_temperature_jacobian"] @ mapping / target_flux
            jacobian[np.arange(1, n), n + np.arange(n-1)] = 3*np.maximum(y, 0)**2 * coordinate_scale
            jacobian[n:, :n] = tangent / scale[:, None]
            jacobian[n + np.arange(n-1), n + np.arange(n-1)] = y_tangent * coordinate_scale / scale
            if local_energy_scaling:
                jacobian[:n] = locally_scaled_energy_rows(jacobian[:n], cell_scale)
        return NonlinearEvaluation(residual, jacobian, payload)

    original_measure = options.get("step_measure")
    options["step_measure"] = lambda old, new: (
        np.max(np.abs(new[:n] - old[:n])) if original_measure is None else
        original_measure(old[:n], new[:n]))
    original_convergence = options.get("convergence_test")
    tolerance = options.get("residual_tolerance", 2e-3)
    def converged(state, evaluation, step):
        actual = np.asarray(evaluation.payload["total_flux_interface"]) / target_flux - 1
        if np.max(np.abs(actual)) > tolerance:
            return False
        physical = NonlinearEvaluation(actual, None, evaluation.payload)
        return original_convergence is None or original_convergence(state[:n], physical, step)
    options["convergence_test"] = converged
    original_callback = options.get("callback")
    if original_callback is not None:
        options["callback"] = lambda record, state, evaluation: original_callback(record, state[:n], evaluation)
    result = solver(np.concatenate((initial, y0 / coordinate_scale)), augmented_evaluate, **options)
    metadata = nonlinear_result_metadata(result)
    metadata["auxiliary_coordinate_scaling"] = "fixed max(stellar-flux velocity, initial absolute velocity)"
    metadata["local_energy_scaling"] = local_energy_scaling
    metadata["energy_residual_maximum"] = float(np.max(np.abs(result.evaluation.residual[:n])))
    metadata["scaled_compatibility_residual_maximum"] = float(np.max(np.abs(result.evaluation.residual[n:])))
    final = evaluate(result.state[:n], False)
    physical_residual = final.residual
    diagnostics = replace(result.diagnostics,
                          final_residual_rms=float(np.sqrt(np.mean(physical_residual**2))),
                          final_residual_maximum=float(np.max(np.abs(physical_residual))),
                          final_worst_residual_index=int(np.argmax(np.abs(physical_residual))))
    return replace(result, state=result.state[:n], evaluation=final, diagnostics=diagnostics), metadata
