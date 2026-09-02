"""Reusable safeguarded nonlinear solves for atmosphere structure.

The atmosphere modules supply physical residuals and approximate Jacobians;
this module owns convergence, scaling, trust-region step control, and Broyden
updates.  Keeping those policies here prevents individual spectral types from
accumulating their own damping constants and restart heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
Payload = TypeVar("Payload")


@dataclass(frozen=True)
class NonlinearEvaluation(Generic[Payload]):
    """Residual, optional Jacobian, and caller-owned physical state."""

    residual: FloatArray
    jacobian: FloatArray | None
    payload: Payload


@dataclass(frozen=True)
class NonlinearIteration:
    """One accepted trust-region iteration."""

    iteration: int
    residual_rms: float
    residual_maximum: float
    maximum_step: float
    trust_radius: float
    line_search_factor: float
    jacobian_recomputed: bool


@dataclass(frozen=True)
class NonlinearResult(Generic[Payload]):
    """Final nonlinear state and convergence history."""

    state: FloatArray
    evaluation: NonlinearEvaluation[Payload]
    converged: bool
    iterations: int
    history: tuple[NonlinearIteration, ...]


def _validated_evaluation(
    evaluation: NonlinearEvaluation[Payload],
    state_size: int,
    *,
    require_jacobian: bool,
) -> NonlinearEvaluation[Payload]:
    residual = np.asarray(evaluation.residual, dtype=np.float64)
    if residual.shape != (state_size,) or np.any(~np.isfinite(residual)):
        raise ValueError("nonlinear residual must be a finite state-sized vector")
    jacobian = evaluation.jacobian
    if require_jacobian and jacobian is None:
        raise ValueError("a Jacobian was requested but not returned")
    if jacobian is not None:
        jacobian = np.asarray(jacobian, dtype=np.float64)
        if jacobian.shape != (state_size, state_size):
            raise ValueError("nonlinear Jacobian must be square and state-sized")
        if np.any(~np.isfinite(jacobian)):
            raise ValueError("nonlinear Jacobian must be finite")
    return NonlinearEvaluation(residual, jacobian, evaluation.payload)


def _residual_merit(residual: FloatArray) -> float:
    """Balance global progress against a localized failed depth cell."""

    rms = float(np.sqrt(np.mean(residual**2)))
    maximum = float(np.max(np.abs(residual)))
    return rms + 0.25 * maximum


def solve_trust_region_newton(
    initial_state: FloatArray,
    evaluate: Callable[[FloatArray, bool], NonlinearEvaluation[Payload]],
    *,
    maximum_iterations: int = 60,
    residual_tolerance: float = 2.0e-3,
    step_tolerance: float = 2.0e-4,
    initial_trust_radius: float = 0.04,
    maximum_trust_radius: float = 0.12,
    minimum_trust_radius: float = 1.0e-5,
    jacobian_refresh_interval: int = 4,
    minimum_line_search_factor: float = 1.0 / 64.0,
    finite_difference_fallback_step: float | None = 1.0e-4,
    callback: Callable[
        [NonlinearIteration, FloatArray, NonlinearEvaluation[Payload]], None
    ]
    | None = None,
    convergence_test: Callable[
        [FloatArray, NonlinearEvaluation[Payload], float], bool
    ]
    | None = None,
    acceptance_test: Callable[
        [NonlinearEvaluation[Payload], NonlinearEvaluation[Payload]], bool
    ]
    | None = None,
    step_measure: Callable[[FloatArray, FloatArray], float] | None = None,
    stationary_completion_iterations: int | None = None,
) -> NonlinearResult[Payload]:
    """Solve a square nonlinear system with automatic globalization.

    The caller controls the physics through ``evaluate(state, jacobian)``.
    Newton steps are row-scaled and solved in a least-squares sense, making a
    singular provisional atmosphere recoverable.  A trust radius limits the
    largest state change; backtracking accepts only reductions of a merit
    function containing both RMS and maximum residuals.  Between exact or
    approximate Jacobian refreshes, a rank-one Broyden update communicates
    the observed opacity/EOS response without extra formal transfer solves.
    ``stationary_completion_iterations`` is reserved for explicitly
    approximate warm-start phases: it permits a caller to move on after that
    many consecutive accepted physical steps fall below ``step_tolerance``,
    even though the provisional residual is not the final equation root.
    """

    state = np.asarray(initial_state, dtype=np.float64).copy()
    if state.ndim != 1 or state.size < 2 or np.any(~np.isfinite(state)):
        raise ValueError("initial_state must be a finite one-dimensional vector")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    if residual_tolerance <= 0.0 or step_tolerance <= 0.0:
        raise ValueError("nonlinear tolerances must be positive")
    if not 0.0 < minimum_line_search_factor <= 1.0:
        raise ValueError("minimum_line_search_factor must lie in (0, 1]")
    if not 0.0 < minimum_trust_radius <= initial_trust_radius:
        raise ValueError("trust radii must be positive and ordered")
    if maximum_trust_radius < initial_trust_radius:
        raise ValueError("maximum_trust_radius must exceed the initial value")
    if jacobian_refresh_interval < 1:
        raise ValueError("jacobian_refresh_interval must be positive")
    if finite_difference_fallback_step is not None and (
        not np.isfinite(finite_difference_fallback_step)
        or finite_difference_fallback_step <= 0.0
    ):
        raise ValueError(
            "finite_difference_fallback_step must be positive or None"
        )
    if (
        stationary_completion_iterations is not None
        and stationary_completion_iterations < 1
    ):
        raise ValueError(
            "stationary_completion_iterations must be positive or None"
        )

    def finite_difference_jacobian(
        current_state: FloatArray,
        current_evaluation: NonlinearEvaluation[Payload],
    ) -> FloatArray:
        assert finite_difference_fallback_step is not None
        numerical = np.empty((state.size, state.size), dtype=np.float64)
        for column in range(state.size):
            # The caller chooses state variables with useful numerical units
            # (atmospheres use ln T), so this is an absolute state increment.
            # Scaling by the value itself would make a log-state perturbation
            # depend spuriously on the choice of dimensional zero point.
            step = finite_difference_fallback_step
            trial_state = current_state.copy()
            trial_state[column] += step
            trial = _validated_evaluation(
                evaluate(trial_state, False),
                state.size,
                require_jacobian=False,
            )
            numerical[:, column] = (
                trial.residual - current_evaluation.residual
            ) / step
        return numerical

    # A restart may already satisfy the requested physical tolerances.  Test
    # that state with one residual evaluation before paying for a Jacobian;
    # in atmosphere problems the latter can require one expensive opacity
    # and transfer derivative per depth point.  Passing a zero step is the
    # mathematically appropriate stationarity test for an unchanged restart.
    evaluation = _validated_evaluation(
        evaluate(state, False), state.size, require_jacobian=False
    )
    initial_residual_maximum = float(np.max(np.abs(evaluation.residual)))
    if (
        initial_residual_maximum < residual_tolerance
        and (
            convergence_test is None
            or convergence_test(state, evaluation, 0.0)
        )
    ):
        return NonlinearResult(state, evaluation, True, 0, ())

    evaluation = _validated_evaluation(
        evaluate(state, True), state.size, require_jacobian=True
    )
    assert evaluation.jacobian is not None
    jacobian = evaluation.jacobian.copy()
    trust_radius = float(initial_trust_radius)
    history: list[NonlinearIteration] = []
    last_step_maximum = np.inf
    numerical_fallback_used_at_state = False
    rejected_secant_repairs_at_state = 0
    analytic_restart_used_at_state = False

    def stationary_result_if_converged(
        iteration: int,
    ) -> NonlinearResult[Payload] | None:
        """Accept an accurate residual after sub-tolerance steps stagnate."""

        stationary_step = min(trust_radius, minimum_trust_radius)
        residual = evaluation.residual
        converged = bool(
            np.max(np.abs(residual)) < residual_tolerance
            and stationary_step < step_tolerance
            and (
                convergence_test is None
                or convergence_test(state, evaluation, stationary_step)
            )
        )
        if not converged:
            return None
        record = NonlinearIteration(
            iteration=iteration,
            residual_rms=float(np.sqrt(np.mean(residual**2))),
            residual_maximum=float(np.max(np.abs(residual))),
            maximum_step=stationary_step,
            trust_radius=trust_radius,
            line_search_factor=0.0,
            jacobian_recomputed=False,
        )
        history.append(record)
        if callback is not None:
            callback(record, state.copy(), evaluation)
        return NonlinearResult(
            state, evaluation, True, iteration, tuple(history)
        )

    for iteration in range(1, maximum_iterations + 1):
        residual = evaluation.residual
        residual_maximum = float(np.max(np.abs(residual)))
        if (
            residual_maximum < residual_tolerance
            and last_step_maximum < step_tolerance
            and (
                convergence_test is None
                or convergence_test(
                    state, evaluation, last_step_maximum
                )
            )
        ):
            return NonlinearResult(
                state,
                evaluation,
                True,
                iteration - 1,
                tuple(history),
            )
        if (
            stationary_completion_iterations is not None
            and len(history) >= stationary_completion_iterations
            and all(
                record.maximum_step < step_tolerance
                for record in history[-stationary_completion_iterations:]
            )
            and (
                convergence_test is None
                or convergence_test(state, evaluation, last_step_maximum)
            )
        ):
            # Some callers use this solver for an explicitly approximate
            # warm-start phase before imposing the final equations.  Once
            # repeated accepted physical changes are below the requested
            # structural tolerance, continuing to chase a nonzero residual
            # of that provisional system only consumes expensive opacity and
            # transfer evaluations.  The option is disabled for ordinary
            # root solves; atmosphere preconditioners opt in explicitly.
            return NonlinearResult(
                state,
                evaluation,
                True,
                iteration - 1,
                tuple(history),
            )

        row_scale = np.maximum(
            np.max(np.abs(jacobian), axis=1),
            np.finfo(np.float64).tiny,
        )
        scaled_jacobian = jacobian / row_scale[:, np.newaxis]
        scaled_residual = residual / row_scale
        # A small diagonal Tikhonov term selects a smooth finite step when a
        # provisional convection boundary makes the local Jacobian singular.
        regularization = 1.0e-8
        augmented_matrix = np.vstack(
            (
                scaled_jacobian,
                np.sqrt(regularization) * np.eye(state.size),
            )
        )
        augmented_rhs = np.concatenate(
            (-scaled_residual, np.zeros(state.size))
        )
        step = np.linalg.lstsq(
            augmented_matrix, augmented_rhs, rcond=1.0e-10
        )[0]
        step_maximum = (
            float(np.max(np.abs(step)))
            if step_measure is None
            else float(step_measure(state, state + step))
        )
        if not np.isfinite(step_maximum) or step_maximum < 0.0:
            raise ValueError(
                "step_measure must return a finite non-negative value"
            )
        if step_maximum > trust_radius:
            step *= trust_radius / step_maximum

        old_state = state
        old_evaluation = evaluation
        old_merit = _residual_merit(old_evaluation.residual)
        factor = 1.0
        accepted = False
        trial_state = old_state
        trial = old_evaluation
        while factor >= minimum_line_search_factor:
            trial_state = old_state + factor * step
            trial = _validated_evaluation(
                evaluate(trial_state, False),
                state.size,
                require_jacobian=False,
            )
            if (
                _residual_merit(trial.residual) < old_merit
                and (
                    acceptance_test is None
                    or acceptance_test(old_evaluation, trial)
                )
            ):
                state = trial_state
                evaluation = trial
                accepted = True
                break
            factor *= 0.5

        if not accepted:
            # The backtracking evaluations have measured a true directional
            # derivative even though none of their steps lowered the merit.
            # Retain that information instead of immediately rebuilding the
            # same approximate Jacobian and proposing the same uphill step.
            # Successive rejected directions build an inexpensive low-rank
            # recovery for large systems where a complete finite-difference
            # Jacobian would require one transfer solve per depth point.
            sampled_step = trial_state - old_state
            sampled_denominator = float(sampled_step @ sampled_step)
            if sampled_denominator > np.finfo(np.float64).tiny:
                sampled_residual_change = (
                    trial.residual - old_evaluation.residual
                )
                jacobian += np.outer(
                    sampled_residual_change - jacobian @ sampled_step,
                    sampled_step,
                ) / sampled_denominator
                rejected_secant_repairs_at_state += 1
                trust_radius *= 0.5
                if trust_radius < minimum_trust_radius:
                    stationary = stationary_result_if_converged(iteration)
                    if stationary is not None:
                        return stationary
                    return NonlinearResult(
                        state,
                        evaluation,
                        False,
                        iteration - 1,
                        tuple(history),
                    )
                if rejected_secant_repairs_at_state < 2:
                    continue
            # Large atmosphere systems deliberately disable a column-by-
            # column finite-difference Jacobian: one such fallback would
            # require a complete opacity/transfer solve for every depth
            # point.  Previously that also made the branch above retain
            # rejected rank-one repairs until the trust radius collapsed.
            # A fresh analytic tangent is much cheaper and is equivalent to
            # restarting from the last accepted atmosphere with no stale
            # Broyden history.  Permit one such restart at each physical
            # state; if it still fails, the ordinary trust collapse below
            # remains the bounded failure mode.
            if (
                finite_difference_fallback_step is None
                and not analytic_restart_used_at_state
            ):
                evaluation = _validated_evaluation(
                    evaluate(state, True),
                    state.size,
                    require_jacobian=True,
                )
                assert evaluation.jacobian is not None
                jacobian = evaluation.jacobian.copy()
                trust_radius = max(trust_radius, initial_trust_radius)
                rejected_secant_repairs_at_state = 0
                analytic_restart_used_at_state = True
                continue
            # A frozen-opacity or other approximate physics block can point
            # uphill after a large EOS/opacity change.  Before making the
            # trust region arbitrarily small, measure the complete residual
            # derivative once at this state.  This is deliberately a recovery
            # path rather than the normal iteration cost.
            if (
                finite_difference_fallback_step is not None
                and not numerical_fallback_used_at_state
            ):
                jacobian = finite_difference_jacobian(state, evaluation)
                numerical_fallback_used_at_state = True
                continue
            trust_radius *= 0.25
            if trust_radius < minimum_trust_radius:
                stationary = stationary_result_if_converged(iteration)
                if stationary is not None:
                    return stationary
                return NonlinearResult(
                    state,
                    evaluation,
                    False,
                    iteration - 1,
                    tuple(history),
                )
            # A rejected linearization is stale; rebuild it at the unchanged
            # physical state before proposing another direction.
            evaluation = _validated_evaluation(
                evaluate(state, True), state.size, require_jacobian=True
            )
            assert evaluation.jacobian is not None
            jacobian = evaluation.jacobian.copy()
            numerical_fallback_used_at_state = False
            rejected_secant_repairs_at_state = 0
            continue

        accepted_step = state - old_state
        last_step_maximum = (
            float(np.max(np.abs(accepted_step)))
            if step_measure is None
            else float(step_measure(old_state, state))
        )
        if not np.isfinite(last_step_maximum) or last_step_maximum < 0.0:
            raise ValueError("step_measure must return a finite non-negative value")
        numerical_fallback_used_at_state = False
        rejected_secant_repairs_at_state = 0
        analytic_restart_used_at_state = False
        actual_reduction = old_merit - _residual_merit(evaluation.residual)
        linear_residual = old_evaluation.residual + jacobian @ accepted_step
        predicted_reduction = old_merit - _residual_merit(linear_residual)
        agreement = actual_reduction / max(
            predicted_reduction, np.finfo(np.float64).tiny
        )
        # Backtracking has already established that an accepted step lowers
        # the true nonlinear merit.  A poor model ratio requests a Jacobian
        # refresh below, but should not by itself collapse the trust radius;
        # doing both makes a systematically approximate Kantorovich block
        # creep forward in minimum-sized steps even while every full step is
        # successful.
        if factor < 0.5:
            trust_radius = max(minimum_trust_radius, 0.5 * trust_radius)
        elif factor == 1.0:
            # Backtracking has directly verified this direction.  Permit a
            # modest recovery from an overly small trust region even while
            # the approximate Jacobian is still learning the local response.
            growth = 1.5 if agreement > 0.75 else 1.25
            trust_radius = min(maximum_trust_radius, growth * trust_radius)

        # A heavily backtracked step is still a measured, downhill secant and
        # is often the most useful information available for an approximate
        # atmosphere tangent.  The trust radius has already been reduced
        # above.  Rebuilding solely because ``factor < 0.5`` discarded that
        # information and made expensive opacity-aware Jacobians repeat after
        # every accepted step.  Refresh on the scheduled cadence or genuinely
        # poor predicted/actual agreement; otherwise let Broyden learn from
        # the accepted nonlinear response below.
        refresh = (
            iteration % jacobian_refresh_interval == 0
            or agreement < 0.1
        )
        if refresh:
            evaluation = _validated_evaluation(
                evaluate(state, True), state.size, require_jacobian=True
            )
            assert evaluation.jacobian is not None
            jacobian = evaluation.jacobian.copy()
            # An analytic refresh restores the global transfer/EOS block but
            # must not discard the just-measured directional derivative.
            denominator = float(accepted_step @ accepted_step)
            if denominator > np.finfo(np.float64).tiny:
                residual_change = (
                    evaluation.residual - old_evaluation.residual
                )
                jacobian += np.outer(
                    residual_change - jacobian @ accepted_step,
                    accepted_step,
                ) / denominator
        else:
            residual_change = evaluation.residual - old_evaluation.residual
            denominator = float(accepted_step @ accepted_step)
            if denominator > np.finfo(np.float64).tiny:
                jacobian += np.outer(
                    residual_change - jacobian @ accepted_step,
                    accepted_step,
                ) / denominator

        record = NonlinearIteration(
            iteration=iteration,
            residual_rms=float(np.sqrt(np.mean(evaluation.residual**2))),
            residual_maximum=float(np.max(np.abs(evaluation.residual))),
            maximum_step=last_step_maximum,
            trust_radius=trust_radius,
            line_search_factor=factor,
            jacobian_recomputed=refresh,
        )
        history.append(record)
        if callback is not None:
            callback(record, state.copy(), evaluation)

    final_converged = bool(
        np.max(np.abs(evaluation.residual)) < residual_tolerance
        and last_step_maximum < step_tolerance
        and (
            convergence_test is None
            or convergence_test(state, evaluation, last_step_maximum)
        )
    )
    return NonlinearResult(
        state,
        evaluation,
        final_converged,
        maximum_iterations,
        tuple(history),
    )
