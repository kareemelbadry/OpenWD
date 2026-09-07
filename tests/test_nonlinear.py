import numpy as np
import pytest

from wd_spectra.nonlinear import (
    NonlinearEvaluation,
    RecoverableEvaluationError,
    solve_trust_region_newton,
)


def test_alternative_step_builder_cannot_bypass_physical_trust_limit():
    def evaluate(state, need_jacobian):
        return NonlinearEvaluation(state-1., np.eye(2) if need_jacobian else None, None)
    result = solve_trust_region_newton(
        np.zeros(2), evaluate, maximum_iterations=1,
        step_builder=lambda state, ev, jac, radius: np.full(2, 100.),
        initial_trust_radius=.04, step_measure=lambda old, new: 2*np.max(abs(new-old)))
    np.testing.assert_allclose(result.state, [.02, .02], atol=1e-15)
    assert not result.converged


@pytest.mark.parametrize("penalty", [-1., np.inf, np.nan])
def test_linear_regularization_must_be_finite_nonnegative(penalty):
    with pytest.raises(ValueError, match="linear_regularization"):
        solve_trust_region_newton(np.zeros(2), None, linear_regularization=penalty)


def test_unpenalized_newton_retains_weak_resolved_direction_and_trust_limit():
    matrix = np.array([[1., 1.], [1., 1.+1e-5]])
    target = np.array([.01, -.01])
    def evaluate(state, need_jacobian):
        return NonlinearEvaluation(matrix@(state-target),
                                   matrix if need_jacobian else None, None)
    # The weak direction is resolvable, but the default regularization damps it.
    unpenalized = solve_trust_region_newton(np.zeros(2), evaluate,
        linear_regularization=0., maximum_iterations=1,
        allow_initial_convergence=False, residual_tolerance=1e-12)
    np.testing.assert_allclose(unpenalized.state, target, atol=1e-11)
    bounded = solve_trust_region_newton(np.zeros(2), evaluate,
        linear_regularization=0., maximum_iterations=1,
        allow_initial_convergence=False, residual_tolerance=1e-12,
        initial_trust_radius=.001, maximum_trust_radius=.001)
    assert np.max(abs(bounded.state)) <= .001000000001
    assert not bounded.converged


@pytest.mark.parametrize("direction", [np.ones(3), np.array([np.nan, 0.])])
def test_alternative_step_builder_rejects_invalid_directions(direction):
    def evaluate(state, need_jacobian):
        return NonlinearEvaluation(state-1., np.eye(2) if need_jacobian else None, None)
    with pytest.raises(ValueError, match="step_builder must return"):
        solve_trust_region_newton(np.zeros(2), evaluate, maximum_iterations=1,
                                  step_builder=lambda *args: direction)


def test_trust_region_newton_solves_coupled_nonlinear_system():
    target = np.asarray([2.0, 3.0])

    def evaluate(state, need_jacobian):
        residual = np.asarray(
            [
                state[0] ** 2 + 0.2 * state[1] - target[0] ** 2 - 0.6,
                np.exp(state[1]) + 0.1 * state[0] - np.exp(target[1]) - 0.2,
            ]
        )
        jacobian = None
        if need_jacobian:
            jacobian = np.asarray(
                [[2.0 * state[0], 0.2], [0.1, np.exp(state[1])]]
            )
        return NonlinearEvaluation(residual, jacobian, state.copy())

    result = solve_trust_region_newton(
        np.asarray([1.4, 2.5]),
        evaluate,
        maximum_iterations=80,
        residual_tolerance=1.0e-10,
        step_tolerance=1.0e-9,
        initial_trust_radius=0.3,
        maximum_trust_radius=0.8,
    )
    assert result.converged
    np.testing.assert_allclose(result.state, target, rtol=0.0, atol=2.0e-9)
    assert result.history
    assert result.history[-1].residual_maximum < 1.0e-10
    assert result.diagnostics.terminal_reason == "residual-and-step-converged"
    assert result.diagnostics.accepted_iterations == len(result.history)
    assert result.diagnostics.residual_evaluations >= (
        result.diagnostics.jacobian_evaluations
    )


def test_trust_region_newton_rejects_invalid_jacobian():
    def evaluate(state, need_jacobian):
        return NonlinearEvaluation(
            residual=np.ones_like(state),
            jacobian=np.ones((state.size, state.size + 1)),
            payload=None,
        )

    with pytest.raises(ValueError, match="square and state-sized"):
        solve_trust_region_newton(np.ones(3), evaluate)


def test_projected_trials_are_the_states_evaluated_and_reported():
    accepted = []
    def evaluate(state, need_jacobian):
        return NonlinearEvaluation(state-np.ones(2), np.eye(2) if need_jacobian else None,
                                   state.copy())
    def report(record, state, evaluation):
        np.testing.assert_array_equal(state, evaluation.payload)
        accepted.append((state.copy(), record.maximum_step))
    result = solve_trust_region_newton(
        np.zeros(2), evaluate, maximum_iterations=60,
        initial_trust_radius=.3, maximum_trust_radius=.3,
        residual_tolerance=1e-8, step_tolerance=1e-8,
        trial_projector=lambda old, trial: old+.5*(trial-old), callback=report)
    assert result.converged
    previous = np.zeros(2)
    for state, recorded_step in accepted:
        assert recorded_step == pytest.approx(np.max(abs(state-previous)))
        assert recorded_step <= .3
        previous = state
    np.testing.assert_allclose(result.state, np.ones(2), atol=1e-8)


def test_projection_cannot_escape_trust_region_or_fabricate_convergence():
    def evaluate(state, need_jacobian):
        return NonlinearEvaluation(state-np.ones(2), np.eye(2) if need_jacobian else None, None)
    result = solve_trust_region_newton(
        np.zeros(2), evaluate, maximum_iterations=2,
        initial_trust_radius=.1, maximum_trust_radius=.1,
        trial_projector=lambda old, trial: np.ones(2),
        finite_difference_fallback_step=None)
    assert not result.converged
    assert result.diagnostics.infeasible_trial_rejections > 0
    np.testing.assert_array_equal(result.state, np.zeros(2))
    with pytest.raises(ValueError, match="original shape"):
        solve_trust_region_newton(np.zeros(2), evaluate,
                                 trial_projector=lambda old, trial: np.ones(3))


def test_projection_enforces_the_callers_physical_step_measure():
    def evaluate(state, need_jacobian):
        return NonlinearEvaluation(
            state - np.ones(2), np.eye(2) if need_jacobian else None, None
        )

    result = solve_trust_region_newton(
        np.zeros(2), evaluate, maximum_iterations=2,
        initial_trust_radius=0.1, maximum_trust_radius=0.1,
        step_measure=lambda old, new: 100.0 * np.max(np.abs(new - old)),
        trial_projector=lambda old, trial: old + 0.02,
        finite_difference_fallback_step=None,
    )
    # Small in raw variable units, but twenty times the physical trust radius.
    assert not result.converged
    assert result.diagnostics.infeasible_trial_rejections > 0
    np.testing.assert_array_equal(result.state, np.zeros(2))


def test_identity_projection_preserves_the_uncorrected_path_exactly():
    def evaluate(state, need_jacobian):
        return NonlinearEvaluation(
            state**2 - np.array([4.0, 9.0]),
            np.diag(2.0 * state) if need_jacobian else None, None,
        )

    initial = np.array([1.0, 2.0])
    baseline = solve_trust_region_newton(initial, evaluate)
    identity = solve_trust_region_newton(
        initial, evaluate, trial_projector=lambda old, trial: trial
    )
    assert baseline.converged and identity.converged
    np.testing.assert_array_equal(identity.state, baseline.state)
    assert identity.history == baseline.history
    assert identity.diagnostics.residual_evaluations == baseline.diagnostics.residual_evaluations


def test_finite_difference_fallback_recovers_from_uphill_jacobian():
    def evaluate(state, need_jacobian):
        residual = np.asarray(
            [state[0] ** 2 - 4.0, np.exp(state[1]) - 2.0]
        )
        # Deliberately reverse the proposed Newton direction.  The generic
        # recovery path must measure the full residual rather than requiring
        # a caller-specific damping tweak.
        jacobian = None
        if need_jacobian:
            jacobian = -np.asarray(
                [[2.0 * state[0], 0.0], [0.0, np.exp(state[1])]]
            )
        return NonlinearEvaluation(residual, jacobian, None)

    result = solve_trust_region_newton(
        np.asarray([1.5, 0.3]),
        evaluate,
        maximum_iterations=40,
        residual_tolerance=1.0e-9,
        step_tolerance=1.0e-8,
        initial_trust_radius=0.4,
        maximum_trust_radius=0.8,
        jacobian_refresh_interval=20,
    )
    assert result.converged
    np.testing.assert_allclose(
        result.state, [2.0, np.log(2.0)], rtol=0.0, atol=2.0e-8
    )


def test_rejected_secant_updates_recover_without_full_finite_difference():
    target = np.asarray([2.0, np.log(2.0)])

    def evaluate(state, need_jacobian):
        residual = np.asarray(
            [state[0] ** 2 - 4.0, np.exp(state[1]) - 2.0]
        )
        jacobian = None
        if need_jacobian:
            jacobian = -np.asarray(
                [[2.0 * state[0], 0.0], [0.0, np.exp(state[1])]]
            )
        return NonlinearEvaluation(residual, jacobian, None)

    result = solve_trust_region_newton(
        np.asarray([1.5, 0.3]),
        evaluate,
        maximum_iterations=80,
        residual_tolerance=1.0e-9,
        step_tolerance=1.0e-8,
        initial_trust_radius=0.4,
        maximum_trust_radius=0.8,
        jacobian_refresh_interval=20,
        finite_difference_fallback_step=None,
    )

    assert result.converged
    np.testing.assert_allclose(result.state, target, rtol=0.0, atol=2.0e-8)


def test_large_system_rebuilds_analytic_tangent_after_rejected_repairs():
    analytic_calls = 0
    residual_calls = 0
    target = np.asarray([1.0, 2.0, 3.0])

    def evaluate(state, need_jacobian):
        nonlocal analytic_calls, residual_calls
        residual = state - target
        if need_jacobian:
            analytic_calls += 1
            # Emulate a stale Broyden-patched atmosphere tangent on the first
            # call; the fresh tangent at the unchanged state is correct.
            jacobian = -np.eye(3) if analytic_calls == 1 else np.eye(3)
        else:
            residual_calls += 1
            jacobian = None
        return NonlinearEvaluation(residual, jacobian, None)

    result = solve_trust_region_newton(
        np.zeros(3),
        evaluate,
        maximum_iterations=40,
        residual_tolerance=1.0e-10,
        step_tolerance=2.0e-9,
        initial_trust_radius=0.2,
        maximum_trust_radius=0.8,
        finite_difference_fallback_step=None,
    )

    assert result.converged
    np.testing.assert_allclose(result.state, target, rtol=0.0, atol=1.0e-10)
    assert analytic_calls >= 2
    # The recovery is one analytic rebuild, not dozens of seven-point
    # backtracking sequences through an expensive atmosphere opacity block.
    assert residual_calls < 30


def test_trust_radius_uses_caller_physical_step_measure():
    target = np.asarray([1.0, 1.0])
    physical_transform = np.asarray([[1.0, 0.0], [1.0, 1.0]])

    def evaluate(state, need_jacobian):
        residual = state - target
        jacobian = np.eye(2) if need_jacobian else None
        return NonlinearEvaluation(residual, jacobian, None)

    result = solve_trust_region_newton(
        np.zeros(2),
        evaluate,
        maximum_iterations=40,
        residual_tolerance=1.0e-10,
        step_tolerance=1.0e-9,
        initial_trust_radius=0.1,
        maximum_trust_radius=0.3,
        step_measure=lambda old, new: float(
            np.max(np.abs(physical_transform @ (new - old)))
        ),
    )

    assert result.converged
    assert result.history[0].maximum_step <= 0.1 * (1.0 + 1.0e-12)


def test_acceptance_test_backtracks_a_secondary_physical_guardrail():
    def evaluate(state, need_jacobian):
        residual = state - 1.0
        jacobian = np.eye(2) if need_jacobian else None
        return NonlinearEvaluation(residual, jacobian, state.copy())

    result = solve_trust_region_newton(
        np.zeros(2),
        evaluate,
        maximum_iterations=1,
        initial_trust_radius=0.4,
        maximum_trust_radius=0.4,
        acceptance_test=lambda _old, trial: np.max(trial.payload) <= 0.3,
    )

    np.testing.assert_allclose(result.state, [0.2, 0.2])
    assert result.history[0].line_search_factor == pytest.approx(0.5)
    diagnostics = result.diagnostics
    assert diagnostics.terminal_reason == "maximum-iterations-exhausted"
    assert diagnostics.accepted_iterations == 1
    assert diagnostics.rejected_trial_evaluations == 1
    assert diagnostics.acceptance_test_rejections == 1
    assert diagnostics.merit_rejections == 0
    assert diagnostics.rejected_trial_line_search_factors == (1.0,)
    assert result.history[0].rejected_trial_evaluations == 1
    assert result.history[0].worst_residual_index == 0


def test_recoverable_trial_error_backtracks_to_physical_domain():
    def evaluate(state, need_jacobian):
        if state[0] > 1.5:
            raise RecoverableEvaluationError("outside physical domain")
        residual = state - 1.0
        # Make the unbounded Newton proposal twice as large as the root.  The
        # full trial is infeasible and the half-step lands exactly at it.
        jacobian = 0.5 * np.eye(2) if need_jacobian else None
        return NonlinearEvaluation(residual, jacobian, None)

    result = solve_trust_region_newton(
        np.zeros(2),
        evaluate,
        maximum_iterations=2,
        residual_tolerance=1.0e-10,
        step_tolerance=1.1,
        initial_trust_radius=2.0,
        maximum_trust_radius=2.0,
    )

    assert result.converged
    np.testing.assert_allclose(result.state, np.ones(2))
    assert result.history[0].line_search_factor == pytest.approx(0.5)
    assert result.diagnostics.rejected_trial_evaluations == 1
    assert result.diagnostics.infeasible_trial_rejections == 1
    assert result.diagnostics.merit_rejections == 0
    assert result.diagnostics.rejected_trial_residual_maxima == (np.inf,)
    import json
    from wd_spectra.nonlinear import nonlinear_result_metadata
    exported = nonlinear_result_metadata(result)
    assert exported["rejected_trial_residual_maxima"] == (None,)
    restored = json.loads(json.dumps(exported, allow_nan=False))
    assert restored["infeasible_trial_rejections"] == 1
    assert restored["rejected_trial_residual_maxima"] == [None]
    assert result.diagnostics.rejected_trial_worst_residual_indices == (-1,)


def test_trial_value_error_is_not_misclassified_as_recoverable():
    def evaluate(state, need_jacobian):
        if state[0] > 0.0:
            raise ValueError("broken trial evaluation")
        return NonlinearEvaluation(
            state - 1.0,
            np.eye(2) if need_jacobian else None,
            None,
        )

    with pytest.raises(ValueError, match="broken trial evaluation"):
        solve_trust_region_newton(np.zeros(2), evaluate)


def test_backtracked_downhill_step_retains_its_broyden_secant():
    jacobian_calls = 0

    def evaluate(state, need_jacobian):
        nonlocal jacobian_calls
        jacobian_calls += int(need_jacobian)
        residual = state - 1.0
        jacobian = np.eye(2) if need_jacobian else None
        return NonlinearEvaluation(residual, jacobian, state.copy())

    result = solve_trust_region_newton(
        np.zeros(2),
        evaluate,
        maximum_iterations=1,
        initial_trust_radius=0.4,
        maximum_trust_radius=0.4,
        jacobian_refresh_interval=20,
        acceptance_test=lambda old, trial: (
            np.max(trial.payload - old.payload) <= 0.06
        ),
    )

    assert result.history[0].line_search_factor == pytest.approx(0.125)
    assert not result.history[0].jacobian_recomputed
    assert jacobian_calls == 1


def test_acceptable_initial_residual_returns_without_building_jacobian():
    jacobian_calls = 0

    def evaluate(state, need_jacobian):
        nonlocal jacobian_calls
        jacobian_calls += int(need_jacobian)
        return NonlinearEvaluation(
            residual=np.full_like(state, 1.0e-6),
            jacobian=np.eye(state.size) if need_jacobian else None,
            payload=None,
        )

    result = solve_trust_region_newton(
        np.zeros(2),
        evaluate,
        maximum_iterations=40,
        residual_tolerance=1.0e-3,
        step_tolerance=2.0e-4,
        initial_trust_radius=0.04,
        maximum_trust_radius=0.12,
        minimum_trust_radius=1.0e-5,
        finite_difference_fallback_step=None,
    )

    assert result.converged
    assert result.iterations == 0
    assert not result.history
    assert jacobian_calls == 0
    assert result.diagnostics.terminal_reason == "initial-state-converged"
    assert result.diagnostics.residual_evaluations == 1
    assert result.diagnostics.jacobian_evaluations == 0


def test_unfinished_restart_must_compute_a_temperature_correction():
    calls = []
    def evaluate(state, need_jacobian):
        calls.append(need_jacobian)
        return NonlinearEvaluation(1e-3*state, 1e-3*np.eye(2) if need_jacobian else None, None)
    result = solve_trust_region_newton(
        np.ones(2), evaluate, maximum_iterations=1,
        residual_tolerance=.003, step_tolerance=.0003,
        allow_initial_convergence=False, finite_difference_fallback_step=None)
    assert True in calls
    assert not result.converged
    assert result.history[-1].maximum_step > .0003
    assert result.diagnostics.terminal_reason != "initial-state-converged"


def test_verified_stationary_direction_can_complete_a_strict_restart():
    def evaluate(state, need_jacobian):
        return NonlinearEvaluation(state, np.eye(2) if need_jacobian else None, None)
    result = solve_trust_region_newton(
        np.zeros(2), evaluate, allow_initial_convergence=False,
        finite_difference_fallback_step=None)
    assert result.converged
    assert result.diagnostics.jacobian_evaluations > 0
    assert result.diagnostics.terminal_reason == "stationary-residual-converged"
    assert result.history[-1].maximum_step == 0.0


def test_trust_collapse_does_not_manufacture_a_small_correction():
    def evaluate(state, need_jacobian):
        return NonlinearEvaluation(np.full(2, 1e-4),
            np.eye(2)*1e-8 if need_jacobian else None, None)
    result = solve_trust_region_newton(np.zeros(2), evaluate,
        allow_initial_convergence=False, finite_difference_fallback_step=None,
        maximum_iterations=100, residual_tolerance=.003, step_tolerance=.0003)
    assert not result.converged
    assert result.diagnostics.terminal_reason == 'trust-region-collapsed'
    assert not result.history


def test_stationary_warm_start_can_complete_before_reaching_root():
    def evaluate(state, need_jacobian):
        residual = state - 1.0
        jacobian = np.eye(2) if need_jacobian else None
        return NonlinearEvaluation(residual, jacobian, None)

    result = solve_trust_region_newton(
        np.zeros(2),
        evaluate,
        maximum_iterations=20,
        residual_tolerance=1.0e-10,
        step_tolerance=2.0e-3,
        initial_trust_radius=1.0e-3,
        maximum_trust_radius=1.0e-3,
        stationary_completion_iterations=2,
    )

    assert result.converged
    assert result.iterations == 2
    assert result.history[-1].residual_maximum > 0.9
    assert all(record.maximum_step < 2.0e-3 for record in result.history)
    assert (
        result.diagnostics.terminal_reason
        == "stationary-warm-start-complete"
    )


def test_telemetry_records_bounded_trust_region_collapse():
    def evaluate(state, need_jacobian):
        return NonlinearEvaluation(
            residual=np.ones_like(state),
            jacobian=np.eye(state.size) if need_jacobian else None,
            payload=None,
        )

    result = solve_trust_region_newton(
        np.zeros(2),
        evaluate,
        maximum_iterations=40,
        minimum_trust_radius=1.0e-3,
        finite_difference_fallback_step=None,
    )

    diagnostics = result.diagnostics
    assert not result.converged
    assert diagnostics.terminal_reason == "trust-region-collapsed"
    assert diagnostics.accepted_iterations == 0
    assert diagnostics.rejected_directions > 0
    assert diagnostics.rejected_trial_evaluations > 0
    assert diagnostics.merit_rejections == (
        diagnostics.rejected_trial_evaluations
    )
    assert diagnostics.analytic_restarts > 0
    assert diagnostics.smallest_trust_radius < 1.0e-3
    assert len(diagnostics.rejected_trial_worst_residual_indices) == (
        diagnostics.rejected_trial_evaluations
    )
