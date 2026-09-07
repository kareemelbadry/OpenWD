import numpy as np
import pytest
from dense_helium_chemical_equilibrium import ExcessPotentials, solve_mass_action, solve_chemistry
from dense_helium_atomic_experiment import atomic_ion_fraction


def test_full_mass_action_charge_and_nuclei_over_extreme_molecular_ratios():
    n = np.full((9, 7), 1e22)
    s = np.geomspace(1e-15, 1e27, 9)[:, None]
    k = np.geomspace(1e-5, 1e50, 7)[None, :]
    (n0,n1,nd,ne), charge_error, nuclei_error = solve_mass_action(np.log(n),np.log(s),np.log(k))
    assert charge_error < 1e-12 and nuclei_error < 1e-12
    np.testing.assert_allclose(n0+n1+2*nd, n, rtol=1e-12)
    np.testing.assert_allclose(n1+nd, ne, rtol=1e-12)
    np.testing.assert_allclose(np.log(n1)+np.log(ne)-np.log(n0),np.broadcast_to(np.log(s),n.shape),atol=5e-14)
    np.testing.assert_allclose(np.log(n0)+np.log(n1)-np.log(nd),np.broadcast_to(np.log(k),n.shape),atol=5e-14)


def test_molecular_free_limit_recovers_exact_atomic_solution():
    logs = np.linspace(-100, 20, 40)
    (n0,n1,nd,ne), *_ = solve_mass_action(0., logs, 600.)
    np.testing.assert_allclose(ne, atomic_ion_fraction(logs), rtol=2e-14)
    assert nd.max() < 1e-200


def test_chemical_potentials_obey_nuclear_reference_invariance():
    original = ExcessPotentials(.3, -.8, 1.2, .6, 'synthetic algebra test')
    shifted = ExcessPotentials(7.3, 6.2, 15.2, .6, 'synthetic algebra test')
    np.testing.assert_allclose(original.reaction_shifts(), shifted.reaction_shifts(), atol=2e-15)
    a = solve_chemistry(6000., .1, original)
    b = solve_chemistry(6000., .1, shifted)
    np.testing.assert_allclose(a.electron, b.electron, rtol=3e-14)


def test_no_missing_dimer_potential_or_unmarked_source_allowed():
    with pytest.raises(ValueError, match='all four'):
        ExcessPotentials(0., 0., None, .5, 'incomplete').reaction_shifts()
    with pytest.raises(ValueError, match='all four'):
        ExcessPotentials(0., 0., 0., .5, '').reaction_shifts()
    with pytest.raises(ValueError, match='tabulated'):
        solve_chemistry(3000., .1, ExcessPotentials(0., 0., 0., 0., 'unit test'))


def test_electron_insertion_suppresses_ionization_not_enhances_it():
    original = solve_chemistry(6000., .1, ExcessPotentials(0., 0., 0., 0., 'unit test'))
    extra_electron_energy = solve_chemistry(6000., .1, ExcessPotentials(0., 0., 0., 1., 'unit test'))
    assert extra_electron_energy.electron < original.electron
