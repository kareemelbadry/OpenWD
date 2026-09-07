import numpy as np
import pytest
from wd_spectra.nonlinear import (
    NonlinearEvaluation, RecoverableEvaluationError, solve_trust_region_newton,
)


def test_derivative_probe_uses_admissible_side_of_same_equations():
    outside = []
    def evaluate(state, need_jacobian):
        if np.any(state > 1.):
            outside.append(state.copy())
            raise RecoverableEvaluationError("material table upper boundary")
        return NonlinearEvaluation(state - .8, -np.eye(2) if need_jacobian else None, None)
    result = solve_trust_region_newton(np.ones(2), evaluate,
        maximum_iterations=40, initial_trust_radius=.4, maximum_trust_radius=.8,
        residual_tolerance=1e-10, step_tolerance=1e-9,
        jacobian_refresh_interval=20)
    assert result.converged
    assert result.diagnostics.finite_difference_jacobian_rebuilds >= 1
    assert outside
    np.testing.assert_allclose(result.state, [.8, .8], atol=1e-10)


def test_no_admissible_probe_returns_last_state_as_unconverged():
    def evaluate(state, need_jacobian):
        if np.any(state != 1.):
            raise RecoverableEvaluationError("isolated admissible state")
        return NonlinearEvaluation(np.ones(2), np.eye(2) if need_jacobian else None, None)
    result = solve_trust_region_newton(np.ones(2), evaluate, maximum_iterations=10)
    assert not result.converged
    assert result.diagnostics.terminal_reason == "finite-difference-domain-exhausted"
    np.testing.assert_array_equal(result.state, np.ones(2))


def test_programming_error_in_probe_is_not_hidden():
    def evaluate(state, need_jacobian):
        if np.any(state > 1.) and np.max(state-1.) <= 1.01e-4:
            raise TypeError("opacity implementation error")
        if np.any(state > 1.):
            raise RecoverableEvaluationError("material domain")
        return NonlinearEvaluation(state - .8, -np.eye(2) if need_jacobian else None, None)
    with pytest.raises(TypeError, match="implementation error"):
        solve_trust_region_newton(np.ones(2), evaluate, maximum_iterations=10)
