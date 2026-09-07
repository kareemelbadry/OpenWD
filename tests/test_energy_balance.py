import numpy as np
import pytest

from wd_spectra._energy_balance import (
    cell_conservation_rows, discrete_cell_energy_balance, transport_conditioned_rows,
    locally_scaled_energy_rows,
    discrete_radiative_cell_energy_balance,
    integrated_radiative_cell_energy_state_response,
)
from wd_spectra.nonlinear import NonlinearEvaluation, solve_trust_region_newton


def test_cell_energy_rows_are_invertible_and_preserve_all_fluxes():
    rng = np.random.default_rng(918)
    defect = rng.normal(size=40)
    local = cell_conservation_rows(defect)
    recovered = np.empty_like(defect)
    recovered[-1] = local[-1]
    for i in range(defect.size - 2, -1, -1):
        recovered[i] = recovered[i + 1] - local[i]
    np.testing.assert_allclose(recovered, defect, atol=2e-15)
    np.testing.assert_array_equal(cell_conservation_rows(np.ones(4)), [0, 0, 0, 1])


def test_cell_energy_jacobian_differentiates_the_same_equations():
    state = np.array([0.7, 1.1, 1.4, 1.8])
    coupling = np.tril(np.ones((4, 4)))
    physical = coupling @ state**3 - np.arange(4)
    jacobian = coupling @ np.diag(3 * state**2)
    local_jacobian = cell_conservation_rows(jacobian)
    for j in range(4):
        delta = np.eye(4)[j] * 1e-5
        plus = cell_conservation_rows(coupling @ (state + delta)**3 - np.arange(4))
        minus = cell_conservation_rows(coupling @ (state - delta)**3 - np.arange(4))
        np.testing.assert_allclose((plus - minus) / 2e-5, local_jacobian[:, j], atol=3e-9)
    np.testing.assert_array_equal(physical, coupling @ state**3 - np.arange(4))


def test_local_rows_cannot_hide_a_wrong_boundary_flux():
    def evaluate(state, need_jacobian):
        flux_defect = np.full(4, 0.1)
        return NonlinearEvaluation(
            cell_conservation_rows(flux_defect),
            np.zeros((4, 4)) if need_jacobian else None, flux_defect,
        )
    result = solve_trust_region_newton(np.ones(4), evaluate, maximum_iterations=2,
                                      finite_difference_fallback_step=None)
    assert not result.converged
    assert result.evaluation.residual[-1] == 0.1


@pytest.mark.parametrize("values", [np.ones(1), np.ones((2, 2, 2))])
def test_cell_rows_reject_invalid_shapes(values):
    with pytest.raises(ValueError, match="vector or matrix"):
        cell_conservation_rows(values)


def test_cell_rows_resolve_nearly_duplicate_transport_equations():
    n = 12
    matrix = np.ones((n, n)) / n + np.diag(np.geomspace(1e-7, 0.01, n))
    target = np.linspace(0.8, 1.2, n)
    def evaluate(state, need_jacobian):
        return NonlinearEvaluation(
            cell_conservation_rows(matrix @ (state - target)),
            cell_conservation_rows(matrix) if need_jacobian else None, None,
        )
    result = solve_trust_region_newton(
        np.zeros(n), evaluate, maximum_iterations=20,
        residual_tolerance=1e-11, step_tolerance=1e-8,
        finite_difference_fallback_step=None,
    )
    assert result.converged
    np.testing.assert_allclose(result.state, target, atol=2e-12)


def test_transport_conditioning_preserves_the_root_and_thin_thick_limits():
    rng = np.random.default_rng(20)
    n = 30
    width = np.geomspace(1e-9, 1e4, n - 1)
    transform = transport_conditioned_rows(np.eye(n), width)
    np.testing.assert_array_equal(np.diag(transform), np.ones(n))
    defect = rng.normal(size=n)
    local = transport_conditioned_rows(defect, width)
    np.testing.assert_allclose(np.linalg.solve(transform, local), defect, atol=3e-15)
    np.testing.assert_allclose(local[0], defect[0] - defect[1], atol=2e-9)
    np.testing.assert_allclose(local[-2], defect[-2], atol=2e-4)
    assert local[-1] == defect[-1]


def test_local_scaling_is_invertible_and_measures_thin_cell_heating():
    scale = np.array([1e-9, .01, 1., 1e4])
    matrix = locally_scaled_energy_rows(np.eye(5), scale)
    assert np.all(np.diag(matrix) > 0)
    defect = np.array([.1, .2, .3, .4, .5])
    np.testing.assert_allclose(np.linalg.solve(matrix, matrix @ defect), defect, atol=1e-15)
    np.testing.assert_allclose(locally_scaled_energy_rows(np.ones(5), scale), np.ones(5))
    thin = np.array([0., 1e-10, 1e-10, 1e-10, 1e-10])
    assert locally_scaled_energy_rows(thin, scale)[0] == pytest.approx(-.1)


@pytest.mark.parametrize("epsilon", [1.0, 0.01])
def test_local_energy_exchange_matches_feautrier_flux_divergence(epsilon):
    from wd_spectra._compat import trapezoid
    from wd_spectra.radiative_transfer import coherent_scattering_feautrier_field
    wave = np.geomspace(500, 100000, 50)
    depth = np.geomspace(1e-4, 30, 20)
    tau = depth[None, :] * (wave[:, None] / 1000)**0.4
    planck = np.exp(-wave[:, None] / 20000) * (1 + depth[None, :])
    source, field = coherent_scattering_feautrier_field(
        tau, planck, np.full_like(tau, epsilon), np.full_like(tau, 1 - epsilon), n_angle=3
    )
    convective = np.linspace(0, 500, depth.size)**2
    defect, scale = discrete_cell_energy_balance(
        wave, tau, planck, field.mean_intensity, np.full_like(tau, epsilon), convective
    )
    expected = np.diff(trapezoid(field.interface_flux, wave, axis=0) + convective)
    np.testing.assert_allclose(defect, expected, rtol=1e-5, atol=1e-5)
    assert np.all(scale > 0)


@pytest.mark.parametrize("scattering_fraction", [0., .98, .999999])
def test_direct_cell_energy_tangent_includes_opacity_and_scattering(scattering_fraction):
    from wd_spectra.opacity import optical_depth_from_mass_opacity
    from wd_spectra.radiative_transfer import (
        coherent_scattering_feautrier_field,
        integrated_coherent_scattering_feautrier_state_response,
    )

    wave = np.geomspace(1000., 50000., 17)
    mass = np.geomspace(1e-6, 30., 7)
    depth = np.arange(mass.size)
    planck = np.exp(-wave[:, None]/50000.)*(1+.2*depth[None, :])
    extinction = .3*(1+wave[:, None]/10000.)*(1+.03*depth[None, :])
    absorption = (1-scattering_fraction)*extinction
    scattering = scattering_fraction*extinction
    extinction = absorption + scattering
    db, da, ds = .4*planck, .15*absorption, -.08*scattering
    tau = optical_depth_from_mass_opacity(mass, extinction)
    source, field = coherent_scattering_feautrier_field(
        tau, planck, absorption, scattering, n_angle=3)
    flux_jac, energy_jac, emission_jac = integrated_radiative_cell_energy_state_response(
        wave, tau, mass, planck, field.mean_intensity, source,
        absorption, scattering, db, da, ds, n_angle=3, wavelength_chunk_size=3)
    direct = (da*planck + absorption*db + ds*field.mean_intensity
              - (da+ds)*source)/extinction
    plain_flux, _, _ = integrated_coherent_scattering_feautrier_state_response(
        tau, wave, source, direct, db, scattering/extinction, mass, da+ds,
        n_angle=3, wavelength_chunk_size=3, return_auxiliary_response=False)
    # The chunk consumer changes neither the transfer solve nor its flux tangent.
    np.testing.assert_array_equal(flux_jac, plain_flux)
    _, scale = discrete_radiative_cell_energy_balance(
        wave, tau, planck, field.mean_intensity, absorption/extinction)
    h = .01
    for k in range(mass.size):
        values = []
        for sign in (-1., 1.):
            direction = np.eye(mass.size)[k]
            bp = planck + sign*h*db*direction
            ap = absorption + sign*h*da*direction
            sp = scattering + sign*h*ds*direction
            tp = optical_depth_from_mass_opacity(mass, ap+sp)
            _, fp = coherent_scattering_feautrier_field(tp, bp, ap, sp, n_angle=3)
            values.append(discrete_radiative_cell_energy_balance(
                wave, tp, bp, fp.mean_intensity, ap/(ap+sp)))
        finite_difference = (values[1][0]-values[0][0])/(2*h)
        np.testing.assert_allclose(energy_jac[:, k]/scale, finite_difference/scale,
                                   rtol=3e-3, atol=2e-5)
        emission_difference = (values[1][1]-values[0][1])/(2*h)
        np.testing.assert_allclose(emission_jac[:, k]/scale, emission_difference/scale,
                                   rtol=3e-5, atol=2e-7)


def test_mean_response_stream_is_read_only_and_wavelength_bounded():
    from wd_spectra.radiative_transfer import integrated_coherent_scattering_feautrier_state_response
    wave = np.linspace(1000., 10000., 7)
    mass = np.geomspace(.001, 10., 5)
    tau = np.broadcast_to(mass, (7, 5)).copy()
    source = np.ones_like(tau)
    chunks = []
    def consume(start, stop, mean_response):
        assert mean_response.shape == (stop-start, 5, 5)
        assert stop-start <= 2
        assert not mean_response.flags.writeable
        chunks.append((start, stop))
    _, mean, sources = integrated_coherent_scattering_feautrier_state_response(
        tau, wave, source, source, source, np.zeros_like(source), mass,
        np.zeros_like(source), wavelength_chunk_size=2,
        return_auxiliary_response=False, mean_response_consumer=consume)
    assert chunks == [(0, 2), (2, 4), (4, 6), (6, 7)]
    assert mean is None and sources is None
