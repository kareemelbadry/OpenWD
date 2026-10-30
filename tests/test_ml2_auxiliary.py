import numpy as np

from wd_spectra._ml2_auxiliary import (
    ml2_scaled_coefficients, ml2_auxiliary_from_gradient, ml2_auxiliary_compatibility,
)


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
