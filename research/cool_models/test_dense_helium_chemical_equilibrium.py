import numpy as np
import pytest
from wd_spectra.constants import HELIUM_MASS
from dense_helium_chemical_equilibrium import ExcessPotentials, solve_solute_chemistry


def test_chemical_potentials_obey_nuclear_reference_invariance():
    original = ExcessPotentials(.3, -.8, 1.2, .6, 'synthetic algebra test')
    shifted = ExcessPotentials(7.3, 6.2, 15.2, .6, 'synthetic algebra test')
    np.testing.assert_allclose(original.reaction_shifts(), shifted.reaction_shifts(), atol=2e-15)
    a = solve_solute_chemistry(6000., .1, original, 0.)
    b = solve_solute_chemistry(6000., .1, shifted, 0.)
    np.testing.assert_allclose(a.electron, b.electron, rtol=3e-14)


def test_no_missing_dimer_potential_or_unmarked_source_allowed():
    with pytest.raises(ValueError, match='all four'):
        ExcessPotentials(0., 0., None, .5, 'incomplete').reaction_shifts()
    with pytest.raises(ValueError, match='all four'):
        ExcessPotentials(0., 0., 0., .5, '').reaction_shifts()


def test_electron_insertion_suppresses_ionization_not_enhances_it():
    original = solve_solute_chemistry(6000., .1, ExcessPotentials(0., 0., 0., 0., 'unit test'), 0.)
    extra_electron_energy = solve_solute_chemistry(6000., .1, ExcessPotentials(0., 0., 0., 1., 'unit test'), 0.)
    assert extra_electron_energy.electron < original.electron


def test_solute_chemistry_conserves_and_reduces_to_atomic_without_bound_pairs():
    potentials = ExcessPotentials(.3, -.8, 1.2, .6, 'synthetic algebra test')
    rho = np.geomspace(1e-6, 1., 7)
    free = solve_solute_chemistry(6000., rho, potentials, 0.)
    # Without bound pairs the solute balance is the atomic Saha law n_i^2 = S (n - n_i).
    np.testing.assert_allclose(free.electron**2/(rho/HELIUM_MASS-free.electron),
                               np.exp(free.log_ionization_constant), rtol=1e-12)
    assert np.all(free.molecular_ion == 0)
    bound = solve_solute_chemistry(6000., rho, potentials, 3*rho)
    np.testing.assert_allclose(bound.electron, free.electron, rtol=1e-12)
    np.testing.assert_allclose(bound.molecular_ion/bound.electron, -np.expm1(-3*rho), rtol=1e-12)
    nuclei = rho/HELIUM_MASS
    np.testing.assert_allclose(bound.neutral+bound.atomic_ion+2*bound.molecular_ion, nuclei, rtol=1e-14)
    assert bound.maximum_relative_charge_residual < 1e-14
