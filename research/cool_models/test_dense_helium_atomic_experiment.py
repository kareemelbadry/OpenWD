"""Algebra/safety tests do not validate the underlying approximate pair model."""
from types import SimpleNamespace
import numpy as np
import pytest
from wd_spectra.constants import BOLTZMANN, HELIUM_MASS
from wd_spectra._cool.dense_helium_scope import AtomicDenseEOS, TABLE_VERSION, atomic_dense_experiment


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


def test_bulk_derivatives_and_fail_closed_domain(model):
    t, p = np.array([4000., 6000.]), np.array([1e9, 1e10])
    thermal = model.thermodynamics(t, p)
    np.testing.assert_allclose(thermal.specific_heat_constant_pressure,
                              2.5*BOLTZMANN/HELIUM_MASS, rtol=1e-11)
    np.testing.assert_allclose(thermal.density_temperature_derivative, 1., rtol=1e-11)
    np.testing.assert_allclose(thermal.adiabatic_temperature_gradient, .4, rtol=1e-11)
    with pytest.raises(ValueError, match='REOS3 domain'):
        model.bulk_state(5000., 1e16)
    with pytest.raises(ValueError, match='HNC table domain'):
        model.excess_potentials(2900., 1e-3)


def test_scoped_experiment_restores_production_even_after_failure(model):
    from wd_spectra import eos, helium
    original_eos, original_opacity = eos.hummer_mihalas_helium_lte, helium.helium_continuum_mass_absorption_coefficient
    model.lte = lambda *a, **k: None
    runner = SimpleNamespace(hummer_mihalas_helium_lte=original_eos,
        hummer_mihalas_helium_thermodynamics=eos.hummer_mihalas_helium_thermodynamics,
        helium_continuum_mass_absorption_coefficient=original_opacity,
        compute_db=lambda *a, **k: None, save_model_result=lambda *a, **k: None,
        helium_continuum_atmosphere=lambda *a, **k: None, run=lambda *a, **k: None,
        radiative_equilibrium_helium_atmosphere=lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        with atomic_dense_experiment(runner, model,
                                     opacity_factory=lambda original: lambda *a, **k: original(*a, **k)):
            assert eos.hummer_mihalas_helium_lte == model.lte
            assert helium.helium_continuum_mass_absorption_coefficient is not original_opacity
            raise RuntimeError('intentional test exception')
    assert eos.hummer_mihalas_helium_lte is original_eos
    assert helium.helium_continuum_mass_absorption_coefficient is original_opacity
