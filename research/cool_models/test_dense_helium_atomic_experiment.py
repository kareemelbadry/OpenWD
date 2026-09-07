"""Algebra/safety tests do not validate the underlying approximate pair model."""
from types import SimpleNamespace
import numpy as np
import pytest
from wd_spectra.constants import BOLTZMANN, HELIUM_MASS
from dense_helium_atomic_experiment import (
    AtomicDenseEOS, TABLE_VERSION, atomic_ion_fraction, atomic_collision_continuum,
    atomic_dense_experiment,
)


class AnalyticBulk:
    """Only a unit-test double; never substituted for REOS in an atmosphere."""
    def evaluate(self, p, t):
        p, t = np.broadcast_arrays(p, t)
        return p*HELIUM_MASS/(BOLTZMANN*t), 1.5*BOLTZMANN*t/HELIUM_MASS, p < 1e15


@pytest.fixture
def model(tmp_path):
    t, rho = np.geomspace(3000, 17000, 5), np.geomspace(1e-12, 1.6, 8)
    mu = np.zeros((len(t), len(rho), 2))
    mu[..., 0], mu[..., 1] = rho, -2*rho
    path = tmp_path/'test-table.npz'
    np.savez(path, temperature=t, density=rho, mu=mu, version=TABLE_VERSION)
    return AtomicDenseEOS(AnalyticBulk(), path)


def test_atomic_equilibrium_algebra_and_saturation():
    log_s = np.linspace(-600., 8., 100)
    x = atomic_ion_fraction(log_s)
    np.testing.assert_allclose(2*np.log(x)-np.log1p(-x), log_s, atol=2e-12)
    with np.errstate(over='raise', invalid='raise'):
        x = atomic_ion_fraction([-1e5, 1e5])
    np.testing.assert_array_equal(x, [0., 1.])


def test_charge_nuclei_levels_and_same_shift_in_mass_action(model):
    from wd_spectra import eos
    from dense_helium_fluid_experiment import EV
    t = np.array([4000., 6000., 8000.])
    p = np.array([1e9, 1e10, 1e9])
    state = model.lte(t, p)
    n0, ne = state.neutral_he_density, state.electron_density
    np.testing.assert_array_equal(ne, state.singly_ionized_he_density)
    np.testing.assert_allclose(n0+ne, state.helium_nuclei_density, rtol=1e-15)
    np.testing.assert_allclose(state.neutral_level_population_density.sum(axis=-1), n0, rtol=1e-15)
    expected = (np.log(4/state.neutral_partition_function)
        +1.5*np.log(2*np.pi*eos.ELECTRON_MASS*BOLTZMANN*t/eos.PLANCK**2)
        -(eos.HELIUM_FIRST_IONIZATION_ENERGY
          +EV*model.ionization_shift_ev(t,state.mass_density))/(BOLTZMANN*t))
    np.testing.assert_allclose(2*np.log(ne)-np.log(n0), expected, atol=3e-14)


def test_bulk_derivatives_and_fail_closed_domain(model):
    t, p = np.array([4000., 6000.]), np.array([1e9, 1e10])
    thermal = model.thermodynamics(t, p)
    np.testing.assert_allclose(thermal.specific_heat_constant_pressure,
                              2.5*BOLTZMANN/HELIUM_MASS, rtol=1e-11)
    np.testing.assert_allclose(thermal.density_temperature_derivative, 1., rtol=1e-11)
    np.testing.assert_allclose(thermal.adiabatic_temperature_gradient, .4, rtol=1e-11)
    with pytest.raises(ValueError, match='REOS3 domain'):
        model.lte(5000., 1e16)
    with pytest.raises(ValueError, match='HNC table domain'):
        model.lte(2900., 1e10)
    from dense_helium_limits import DenseHeliumDomainError
    with pytest.raises(DenseHeliumDomainError, match='trace-ion'):
        model.lte(15000., 1e4)


def test_invalid_initial_state_is_fatal_but_invalid_trial_can_be_shortened(model):
    from wd_spectra.nonlinear import NonlinearEvaluation, solve_trust_region_newton
    from dense_helium_limits import DenseHeliumDomainError
    rejected = []
    def evaluate(x, jacobian):
        try:
            model.lte(float(np.exp(x[0])), 1e4)
        except DenseHeliumDomainError:
            rejected.append(float(np.exp(x[0])))
            raise
        return NonlinearEvaluation(x-np.array([np.log(10000.), 0.]),
                                   np.eye(2) if jacobian else None, {})
    with pytest.raises(DenseHeliumDomainError, match='trace-ion'):
        solve_trust_region_newton(np.array([np.log(15000.), 0.]), evaluate, maximum_iterations=2)
    np.testing.assert_allclose(rejected, [15000.])
    rejected.clear()
    first = True
    def oversized_first_direction(x, evaluation, jacobian, radius):
        nonlocal first
        if first:
            first = False
            return np.array([.45, 0.])
        return -evaluation.residual
    result = solve_trust_region_newton(np.array([np.log(9500.), 0.]), evaluate,
        step_builder=oversized_first_direction, initial_trust_radius=.5,
        maximum_trust_radius=.5, maximum_iterations=20, residual_tolerance=1e-8,
        step_tolerance=1e-8)
    assert rejected, "the test must actually exercise the trace-domain rejection"
    assert result.converged
    np.testing.assert_allclose(np.exp(result.state[0]), 10000., rtol=1e-8)


def test_no_bound_free_donors_are_invented_for_atomic_model():
    from wd_spectra import helium_molecular as m
    wave, t = m._WAVELENGTH_FREE_FREE[:, None], m._TEMPERATURE[None, :]
    np.testing.assert_allclose(atomic_collision_continuum(wave, t), m._FREE_FREE, atol=0)
    np.testing.assert_array_equal(atomic_collision_continuum([1., 1e9], 5000), 0.)


def test_scoped_experiment_restores_production_even_after_failure(model):
    from wd_spectra import eos, helium
    original_eos, original_opacity = eos.hummer_mihalas_helium_lte, helium.helium_dimer_ion_continuum_coefficient
    runner = SimpleNamespace(hummer_mihalas_helium_lte=original_eos,
        hummer_mihalas_helium_thermodynamics=eos.hummer_mihalas_helium_thermodynamics,
        compute_db=lambda *a, **k: None, save_model_result=lambda *a, **k: None,
        helium_continuum_atmosphere=lambda *a, **k: None, run=lambda *a, **k: None,
        radiative_equilibrium_helium_atmosphere=lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        with atomic_dense_experiment(runner, model):
            assert eos.hummer_mihalas_helium_lte == model.lte
            raise RuntimeError('intentional test exception')
    assert eos.hummer_mihalas_helium_lte is original_eos
    assert helium.helium_dimer_ion_continuum_coefficient is original_opacity
