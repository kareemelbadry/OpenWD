import numpy as np
import pytest

from wd_spectra._material_response import temperature_response_probes
from wd_spectra.nonlinear import RecoverableEvaluationError


@pytest.mark.parametrize("centered", [False, True])
def test_interior_stencil_and_evaluation_order_are_unchanged(centered):
    calls = []
    def evaluate(offset):
        calls.append(offset)
        return np.exp(offset)
    hot, cold, h = temperature_response_probes(evaluate, 2e-4, centered=centered)
    assert calls == ([2e-4, -2e-4] if centered else [2e-4])
    assert hot == np.exp(2e-4)
    assert cold == (np.exp(-2e-4) if centered else None)
    assert h == 2e-4


@pytest.mark.parametrize("valid_side", [-1, 1])
def test_material_boundary_uses_one_sided_same_model_derivative(valid_side):
    def evaluate(offset):
        if offset*valid_side < 0:
            raise RecoverableEvaluationError("physical table boundary")
        return np.exp(offset)
    hot, cold, h = temperature_response_probes(evaluate, 2e-4, centered=True)
    assert cold is None
    assert h == valid_side*2e-4
    np.testing.assert_allclose((hot-1)/h, 1., rtol=1.1e-4)


def test_no_valid_probe_does_not_substitute_values():
    def evaluate(offset):
        raise RecoverableEvaluationError("no admissible probe")
    with pytest.raises(RecoverableEvaluationError, match="no admissible"):
        temperature_response_probes(evaluate, 2e-4, centered=True)


def test_callback_programming_errors_propagate():
    def evaluate(offset):
        raise TypeError("bad opacity callback")
    with pytest.raises(TypeError, match="bad opacity"):
        temperature_response_probes(evaluate, 2e-4, centered=True)
