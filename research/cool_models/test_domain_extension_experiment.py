import numpy as np
import pytest
from wd_spectra._compat import trapezoid
from wd_spectra.spectrum import planck_lambda_angstrom
from domain_extension_experiment import extend_domain


def test_isothermal_extension_preserves_nodes_and_reaches_analytic_screen():
    wave = np.geomspace(100, 1e7, 600)
    t = np.array([5000., 5000.])
    p = np.array([1., 2.])
    target = trapezoid(np.pi*planck_lambda_angstrom(wave, 5000.), wave)
    requested = 1e-3
    result = extend_domain(p, t, p.copy(), np.full(wave.size, 2.), wave, 1., target,
                           lambda pressure, temperature: (0., 1., np.ones(wave.size)),
                           escape_tolerance=requested)
    new_p, new_t, new_tau, meta = result
    np.testing.assert_array_equal(new_p[:2], p)
    np.testing.assert_array_equal(new_t[:2], t)
    np.testing.assert_array_equal(new_tau[:2], p)
    np.testing.assert_allclose(new_p[-1], -np.log(requested), rtol=2e-5)
    np.testing.assert_allclose(new_tau, new_p, rtol=2e-5)
    assert meta['appended_nodes'] == len(new_p)-len(p)
    assert meta['seed_absorption_escape_bound'] == pytest.approx(requested)
    assert np.min(np.diff(np.log(new_p))) > .4


def test_extension_never_silently_returns_insufficient_domain():
    wave = np.geomspace(100, 1e7, 40)
    with pytest.raises(RuntimeError, match='pressure budget'):
        extend_domain([1, 2], [5000, 5000], [1, 2], np.zeros(wave.size), wave, 1., 1.,
                      lambda pressure, temperature: (0., 1., np.zeros(wave.size)),
                      maximum_pressure_factor=2.)
