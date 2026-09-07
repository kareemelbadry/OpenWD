import numpy as np
import pytest
from wd_spectra._compat import trapezoid
from wd_spectra.opacity import optical_depth_from_mass_opacity
from wd_spectra.radiative_transfer import coherent_scattering_feautrier_field
from stable_feautrier_experiment import stable_field, stable_response


@pytest.mark.parametrize('fraction', [0., .8, .999])
def test_stable_blocks_reproduce_existing_well_resolved_equations(fraction):
    tau = np.geomspace(1e-3, 100, 50)
    planck = np.broadcast_to(1.+tau*.2, (2, len(tau))).copy()
    old_s, old = coherent_scattering_feautrier_field(tau, planck, 1.-fraction, fraction)
    new_s, new = stable_field(tau, planck, 1.-fraction, fraction)
    np.testing.assert_allclose(new_s, old_s, rtol=1e-9)
    np.testing.assert_allclose(new.mean_intensity, old.mean_intensity, rtol=1e-9)
    np.testing.assert_allclose(new.interface_flux, old.interface_flux, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize('surface_tau', [1e-3, 1e-12])
def test_temperature_and_opacity_response_finite_difference(surface_tau):
    mass = np.geomspace(surface_tau, 100., 24)
    wave = np.array([2000., 5000., 15000.])
    base_b = np.broadcast_to(1+mass*.2, (3, len(mass))).copy()
    base_abs = np.full_like(base_b, .1)
    base_scat = np.full_like(base_b, .9)
    exponent_b, exponent_a, exponent_s = 4., 1.3, -.2
    def field(x):
        b = base_b*np.exp(exponent_b*x)
        a = base_abs*np.exp(exponent_a*x)
        s = base_scat*np.exp(exponent_s*x)
        tau = optical_depth_from_mass_opacity(mass, a+s)
        source, result = stable_field(tau, b, a, s)
        return source, result, (tau, b, a, s)
    source, base, (tau, b, a, s) = field(np.zeros(len(mass)))
    deps = (exponent_a-exponent_s)*a*s/(a+s)**2
    direct = a/(a+s)*exponent_b*b+deps*(b-base.mean_intensity)
    flux_j, mean_j, _ = stable_response(tau, wave, source, direct, exponent_b*b,
                                       s/(a+s), mass, exponent_a*a+exponent_s*s)
    for index in [0, 8, 20, 23]:
        dx = np.eye(len(mass))[index]*1e-5
        _, plus, _ = field(dx)
        _, minus, _ = field(-dx)
        measured_mean = (plus.mean_intensity-minus.mean_intensity)/(2e-5)
        measured_flux = trapezoid((plus.interface_flux-minus.interface_flux)/(2e-5), wave, axis=0)
        np.testing.assert_allclose(mean_j[:, :, index], measured_mean, rtol=3e-5, atol=2e-7)
        np.testing.assert_allclose(flux_j[:, index], measured_flux, rtol=5e-5, atol=2e-3)


def test_thin_cells_against_70_digit_discrete_reference():
    mp = pytest.importorskip('mpmath')
    tau = np.geomspace(1e-12, 10., 60)
    planck = np.ones((2, len(tau)))
    _, new = stable_field(tau, planck, 1., 0., n_angle=1)
    _, old = coherent_scattering_feautrier_field(tau, planck, 1., 0., n_angle=1)
    with mp.workdps(70):
        expanded = [mp.mpf(0)]+[mp.mpf(float(t)) for t in tau]
        h = [expanded[i+1]-expanded[i] for i in range(len(tau))]
        n = len(expanded)
        matrix, rhs = mp.matrix(n), mp.matrix(n, 1)
        mu = mp.mpf('.5')
        matrix[0, 0], matrix[0, 1] = 1+mu/h[0], -mu/h[0]
        for i in range(1, n-1):
            a = 2*mu**2/(h[i-1]*(h[i-1]+h[i]))
            c = 2*mu**2/(h[i]*(h[i-1]+h[i]))
            matrix[i, i-1], matrix[i, i], matrix[i, i+1] = -a, 1+a+c, -c
            rhs[i] = 1
        matrix[n-1, n-1], rhs[n-1] = 1, 1
        u = mp.lu_solve(matrix, rhs)
        expected = np.array([float(u[i]) for i in range(1, n)])
        expected_flux = np.array([float(4*mp.pi*mu**2*(u[i+1]-u[i])/h[i]) for i in range(n-1)])
    np.testing.assert_allclose(new.mean_intensity[0], expected, rtol=3e-14)
    np.testing.assert_allclose(new.interface_flux[0], expected_flux, rtol=3e-12, atol=1e-14)
    # Ensure this regression actually exposes the original loss of precision.
    assert np.max(abs(old.mean_intensity[0]-expected)) > 1e-7
