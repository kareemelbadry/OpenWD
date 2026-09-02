import numpy as np
import pytest

from wd_spectra.nonlinear import NonlinearEvaluation, solve_trust_region_newton


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


def test_trust_region_newton_rejects_invalid_jacobian():
    def evaluate(state, need_jacobian):
        return NonlinearEvaluation(
            residual=np.ones_like(state),
            jacobian=np.ones((state.size, state.size + 1)),
            payload=None,
        )

    with pytest.raises(ValueError, match="square and state-sized"):
        solve_trust_region_newton(np.ones(3), evaluate)


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
