import numpy as np
import pytest
from wd_spectra import eos, helium_molecular
from wd_spectra.constants import BOLTZMANN
from helium_dimer_eos_experiment import (
    dissociation_equilibrium, helium_dimer_lte, helium_dimer_thermodynamics,
    helium_dimer_continuum,
    helium_dimer_atmosphere_experiment,
)


def test_equilibrium_nodes_positive_extrapolation_and_derivative():
    k, _ = dissociation_equilibrium(helium_molecular._TEMPERATURE)
    np.testing.assert_allclose(k, helium_molecular._EQUILIBRIUM_CONSTANT, rtol=2e-14)
    t = np.geomspace(2000., 1e5, 200)
    k, derivative = dissociation_equilibrium(t)
    assert np.all(k > 0) and np.all(np.diff(k) > 0)
    eps = 1e-5
    measured = np.log(dissociation_equilibrium(t*np.exp(eps))[0]
                      /dissociation_equilibrium(t*np.exp(-eps))[0])/(2*eps)
    np.testing.assert_allclose(derivative, measured, rtol=2e-8)


def test_opacity_reproduces_table_nodes_and_uses_same_equilibrium():
    t = helium_molecular._TEMPERATURE[None, :]
    wave = np.geomspace(100., 1e7, 80)[:, None]
    np.testing.assert_allclose(helium_dimer_continuum(wave, t),
        helium_molecular.helium_dimer_ion_continuum_coefficient(wave, t), rtol=2e-14)
    t = np.array([3000., 5000., 9000.])
    result = helium_dimer_lte(t, np.full(3, 1e10))
    k, _ = dissociation_equilibrium(t)
    n0, n1 = result.atomic.neutral_he_density, result.atomic.singly_ionized_he_density
    np.testing.assert_allclose(n0*n1/k, result.molecular_ion_density, rtol=2e-14)
    assert np.all(np.isfinite(helium_dimer_continuum(wave, t)))


def test_pressure_nuclei_charge_and_molecular_equilibrium():
    temperature = np.array([3000., 4200., 5000., 8000., 22000., 50000.])[:, None]
    pressure = np.array([1e4, 1e8, 1e12, 1e14])[None, :]
    result = helium_dimer_lte(temperature, pressure)
    state, dimer = result.atomic, result.molecular_ion_density
    n0, n1, n2, ne = (state.neutral_he_density, state.singly_ionized_he_density,
                      state.doubly_ionized_he_density, state.electron_density)
    np.testing.assert_allclose(ne, n1+2*n2+dimer, rtol=1e-10)
    np.testing.assert_allclose(state.helium_nuclei_density, n0+n1+n2+2*dimer, rtol=2e-14)
    np.testing.assert_allclose((n0+n1+n2+dimer+ne)*BOLTZMANN*temperature,
                              np.broadcast_to(pressure, ne.shape), rtol=2e-13)
    np.testing.assert_allclose(n0*n1/dimer,
        np.broadcast_to(dissociation_equilibrium(temperature)[0], ne.shape), rtol=2e-13)


def test_extreme_ionization_uses_normalized_fractions_without_overflow():
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        t = np.array([1e5, 1e6, 1e7])
        p = np.array([1e-100, 1e-20, 1e4])
        result = helium_dimer_lte(t, p)
        state = result.atomic
        assert np.all(np.isfinite(state.electron_density))
        np.testing.assert_allclose(state.electron_density,
            state.singly_ionized_he_density+2*state.doubly_ionized_he_density
            +result.molecular_ion_density, rtol=1e-10)


def test_atomic_limit_reproduces_original_eos_and_enthalpy():
    t = np.array([5000., 8000., 22000.])
    p = np.array([1e6, 1e9, 1e5])
    modified = helium_dimer_lte(t, p, include_dimer=False)
    original = eos.hummer_mihalas_helium_lte(t, p, correlated_microfields=True)
    np.testing.assert_allclose(modified.atomic.mass_density, original.mass_density, rtol=2e-13)
    np.testing.assert_allclose(modified.atomic.electron_density, original.electron_density, rtol=2e-13)
    thermo = helium_dimer_thermodynamics(t, p, include_dimer=False)
    old_thermo = eos.hummer_mihalas_helium_thermodynamics(t, p, correlated_microfields=True)
    np.testing.assert_allclose(thermo.specific_heat_constant_pressure,
                              old_thermo.specific_heat_constant_pressure, rtol=2e-8)


def test_thermal_derivatives_remain_finite_and_smooth_at_table_nodes():
    t = np.concatenate([helium_molecular._TEMPERATURE*(1+x) for x in [-1e-5, 0, 1e-5]])
    thermo = helium_dimer_thermodynamics(t, np.full_like(t, 1e10))
    assert np.all(np.isfinite(thermo.specific_heat_constant_pressure))
    assert np.all(thermo.specific_heat_constant_pressure > 0)
    assert np.all(thermo.adiabatic_temperature_gradient > 0)
    cp = thermo.specific_heat_constant_pressure.reshape(3, -1)
    assert np.max(abs(cp[2]/cp[0]-1)) < .01


def test_experiment_restores_defaults_and_cannot_export_a_production_checkpoint(tmp_path):
    from types import SimpleNamespace
    from wd_spectra import atmosphere, helium
    from wd_spectra.models import load_atmosphere_checkpoint
    original = eos.hummer_mihalas_helium_lte
    original_opacity = helium.helium_dimer_ion_continuum_coefficient
    runner = SimpleNamespace(hummer_mihalas_helium_lte=original,
        hummer_mihalas_helium_thermodynamics=eos.hummer_mihalas_helium_thermodynamics,
        save_model_result=lambda *a: None)
    t, p = np.array([5000., 6000.]), np.array([1e8, 1e10])
    with helium_dimer_atmosphere_experiment(runner):
        state = runner.hummer_mihalas_helium_lte(t, p)
        assert state.electron_density[0] > original(t, p).electron_density[0]
        a = atmosphere.Atmosphere(5000., 8., np.array([.01, 10.]), p/1e8, t, p,
            state.mass_density, np.zeros(2), np.zeros(2), state.electron_density,
            {}, helium_lte_state=state)
        result = SimpleNamespace(atmosphere=a, metadata={},
            spectrum=SimpleNamespace(wavelength_angstrom=np.array([4000., 5000.]),
                                     surface_flux_lambda=np.ones(2)))
        runner.save_model_result(result, tmp_path)
    assert eos.hummer_mihalas_helium_lte is original
    assert atmosphere.hummer_mihalas_helium_lte is original
    assert helium.helium_dimer_ion_continuum_coefficient is original_opacity
    assert not (tmp_path/'atmosphere.npz').exists()
    with pytest.raises(KeyError, match='required arrays'):
        load_atmosphere_checkpoint(tmp_path/'experimental-dimer-structure.npz', 5000., 8., 'helium')
