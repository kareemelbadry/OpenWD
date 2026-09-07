"""Local thermal conditioning followed by, never substituted for, equilibrium.

The artificial time scale damps poorly initialized thermal modes. Every trial
uses the original material, transfer and ML2 equations. It is not a physical
evolution calculation, and this module makes no convergence claim.
"""

import logging
import numpy as np
from .constants import STEFAN_BOLTZMANN
from .nonlinear import NonlinearEvaluation, RecoverableEvaluationError

_LOGGER = logging.getLogger(__name__)


def equilibrated_direction(matrix, residual):
    """Unpenalized square solve with row and column equilibration."""
    row = np.max(abs(matrix), axis=1)
    if np.any(row == 0):
        raise np.linalg.LinAlgError("unconstrained energy equation")
    scaled = matrix / row[:, None]
    column = np.max(abs(scaled), axis=0)
    if np.any(column == 0):
        raise np.linalg.LinAlgError("unconstrained temperature variable")
    return np.linalg.solve(scaled / column[None, :], -residual / row) / column


def frozen_energy_evaluator(evaluate):
    """Hold positive equation weights fixed within each Newton linearization.

    Removing derivatives of these numerical weights improves globalization;
    the actual, currently normalized energy is still independently certified.
    The full material/radiation/convection derivative is retained.
    """
    scale = None

    def weighted(state, need_jacobian):
        nonlocal scale
        ev = evaluate(state, need_jacobian)
        p = ev.payload
        target = STEFAN_BOLTZMANN * p["atmosphere"].effective_temperature ** 4
        if scale is None or need_jacobian:
            scale = np.asarray(p["cell_energy_scale"]).copy()
        defect = p["radiative_cell_energy_defect"] + np.diff(
            p["convective_flux_interface"]
        )
        residual = p["total_flux_interface"] / target - 1
        residual[:-1] -= defect / scale
        jacobian = None
        if need_jacobian:
            jacobian = (
                p["radiative_flux_log_temperature_jacobian"]
                + p["convective_flux_log_temperature_jacobian"]
            ) / target
            jacobian[:-1] -= (
                p["cell_energy_log_temperature_jacobian"] / scale[:, None]
            )
            jacobian = jacobian @ p["log_temperature_from_state"]
        return NonlinearEvaluation(residual, jacobian, p)

    return weighted


def thermal_condition(
    initial,
    evaluate,
    *,
    maximum_sweeps=40,
    local_tolerance=3e-3,
    maximum_temperature_step=0.04,
    callback=None,
):
    """Take bounded linearly implicit thermal steps, then return to static solve.

    Cell heating advances ln(T); excess bottom flux cools the bottom thermal
    reservoir. Both artificial inertias vanish from the stationary equations.
    Temporal-defect control chooses the step, not Teff or a convection mask.
    """
    state = np.asarray(initial).copy()
    step = 1.0
    history = []
    reason = "sweep-budget"
    for iteration in range(maximum_sweeps):
        _LOGGER.info(
            "Thermal conditioning %d/%d: rebuilding actual material/transfer tangent",
            iteration + 1,
            maximum_sweeps,
        )
        base = evaluate(state, True)
        p = base.payload
        if (
            np.max(abs(p["cell_energy_balance_relative_residual"]))
            < local_tolerance
        ):
            reason = "local-balance-handoff"
            break
        target = STEFAN_BOLTZMANN * p["atmosphere"].effective_temperature ** 4
        scale = p["cell_energy_scale"].copy()
        mapping = p["log_temperature_from_state"]

        def thermal_rows(payload):
            energy = (
                payload["radiative_cell_energy_defect"]
                + np.diff(payload["convective_flux_interface"])
            ) / scale
            return np.r_[
                energy, payload["total_flux_interface"][-1] / target - 1
            ]

        residual = thermal_rows(p)
        jacobian = np.vstack(
            (
                p["cell_energy_log_temperature_jacobian"] / scale[:, None],
                (
                    p["radiative_flux_log_temperature_jacobian"][-1]
                    + p["convective_flux_log_temperature_jacobian"][-1]
                )
                / target,
            )
        )
        accepted = False
        # The representability check is the real lower bound; this finite
        # ceiling prevents pathological evaluation loops on invalid inputs.
        for attempt in range(64):
            matrix = jacobian.copy()
            nodes = np.arange(len(state) - 1)
            matrix[nodes, nodes] -= 1 / step
            matrix[-1, -1] += 1 / step
            try:
                delta = equilibrated_direction(matrix, residual)
            except np.linalg.LinAlgError:
                step *= 0.5
                continue
            if (
                not np.all(np.isfinite(delta))
                or np.max(abs(delta)) > maximum_temperature_step
            ):
                step *= 0.5
                continue
            candidate = state + np.linalg.solve(mapping, delta)
            if np.array_equal(mapping @ candidate, mapping @ state):
                reason = "temperature-resolution-handoff"
                break
            try:
                trial = evaluate(candidate, False)
            except RecoverableEvaluationError:
                step *= 0.5
                continue
            temporal = thermal_rows(trial.payload)
            temporal[:-1] -= delta[:-1] / step
            temporal[-1] += delta[-1] / step
            error = float(
                np.max(abs(temporal)) / max(np.max(abs(residual)), 1e-12)
            )
            if not np.isfinite(error) or error > 0.25:
                step *= 0.5
                continue
            state = candidate
            record = dict(
                iteration=iteration + 1,
                pseudo_time_step=step,
                temporal_defect=error,
                maximum_log_temperature_change=float(np.max(abs(delta))),
                maximum_local_energy=float(
                    np.max(
                        abs(
                            trial.payload[
                                "cell_energy_balance_relative_residual"
                            ]
                        )
                    )
                ),
                maximum_flux_error=float(
                    np.max(
                        abs(trial.payload["total_flux_interface"] / target - 1)
                    )
                ),
                initialization_only=True,
                converged=False,
            )
            history.append(record)
            _LOGGER.info(
                "Thermal conditioning: local energy %.6g, flux %.6g, dlnT %.6g, dt %.6g",
                record["maximum_local_energy"],
                record["maximum_flux_error"],
                record["maximum_log_temperature_change"],
                step,
            )
            if callback is not None:
                callback(record, state, trial)
            if error < 0.0625:
                step *= 2
            accepted = True
            break
        if not accepted:
            if reason == "sweep-budget":
                reason = "no-admissible-thermal-step"
            break
    return state, dict(
        initialization_only=True,
        converged=False,
        termination=reason,
        iterations=len(history),
        history=history,
    )
