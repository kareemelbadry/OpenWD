"""ML2 numerical identities across efficient and inefficient limits."""
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

from wd_spectra import convection


@pytest.mark.parametrize("excess,loss", [(1e-3, 1e12), (1e-20, 1.),
                                       (1., 1e200), (1., 1e-200),
                                       (1e-20, 0.), (0., 0.), (0., 1e200)])
def test_ml2_root_matches_high_precision_rational_identity(excess, loss):
    with localcontext() as context:
        context.prec = 80
        e, b = Decimal(str(excess)), Decimal(str(loss))
        denominator = (b*b/4+e).sqrt()+b/2
        expected = float(e/denominator) if denominator else 0.
    actual, root = convection._ml2_contrast_and_root(np.array([excess]), np.array([loss]))
    assert np.isfinite(root[0])
    assert actual[0] == pytest.approx(expected, rel=3e-15, abs=0.)


def test_all_flux_paths_and_tangent_retain_small_nonzero_convection(monkeypatch):
    adiabatic, loss, coefficient = np.array([.4]), np.array([1e12]), np.array([1e50])
    coefficients = (adiabatic, loss, coefficient)
    monkeypatch.setattr(convection, "_ml2_local_coefficients_from_thermodynamics",
                        lambda *args: coefficients)
    monkeypatch.setattr(convection, "_ml2_local_coefficients", lambda *args: coefficients)
    atmosphere = SimpleNamespace(temperature=np.array([5000.]))
    gradient = np.array([.401])
    excess = gradient - adiabatic
    contrast, root = convection._ml2_contrast_and_root(excess, loss)
    expected = coefficient * contrast**3
    assert expected[0] > 0.
    # Demonstrate the old subtraction loses the entire root in this limit.
    np.testing.assert_array_equal(np.sqrt(.25*loss**2+excess)-.5*loss, [0.])
    kwargs = dict(specific_heat_constant_pressure=np.ones(1),
                  density_temperature_derivative=np.ones(1),
                  adiabatic_temperature_gradient=adiabatic)
    flux = convection.ml2_convective_flux_for_gradient_from_thermodynamics(
        atmosphere, np.ones(1), gradient, **kwargs)
    derivative = convection.ml2_convective_flux_gradient_derivative_from_thermodynamics(
        atmosphere, np.ones(1), gradient, **kwargs)
    other_flux = convection.ml2_convective_flux_for_gradient(atmosphere, np.ones(1), gradient)
    np.testing.assert_allclose(flux, expected, rtol=2e-15)
    np.testing.assert_array_equal(other_flux, flux)
    np.testing.assert_allclose(derivative, 3*coefficient*contrast**2/(2*root), rtol=2e-15)
    step = 1e-7
    plus = convection.ml2_convective_flux_for_gradient_from_thermodynamics(
        atmosphere, np.ones(1), gradient+step, **kwargs)
    minus = convection.ml2_convective_flux_for_gradient_from_thermodynamics(
        atmosphere, np.ones(1), gradient-step, **kwargs)
    np.testing.assert_allclose((plus-minus)/(2*step), derivative, rtol=1e-7)
