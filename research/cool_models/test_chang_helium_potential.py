import numpy as np
from scipy.optimize import minimize_scalar
from chang_helium_potential import (
    dimer_ground_hartree,trimer_ground_hartree,molecular_ion_neutral_pair_ev,HARTREE_EV,
)


def test_independently_published_dimer_and_trimer_minima():
    dimer=minimize_scalar(dimer_ground_hartree,bounds=(1.9,2.2),method='bounded')
    trimer=minimize_scalar(lambda r:trimer_ground_hartree([r,r,2*r]),bounds=(2.2,2.5),method='bounded')
    np.testing.assert_allclose(dimer.x,2.046179,atol=2e-6)
    np.testing.assert_allclose(-dimer.fun*HARTREE_EV,2.452,atol=.0005)
    np.testing.assert_allclose(trimer.x,2.340,atol=5e-6)
    np.testing.assert_allclose((dimer.fun-trimer.fun)*HARTREE_EV,.1751,atol=3e-5)


def test_permutation_symmetry_and_polarization_limit():
    r=np.array([2.1,3.6,5.])
    expected=trimer_ground_hartree(r)
    import itertools
    for order in itertools.permutations(range(3)):
        assert trimer_ground_hartree(r[list(order)]) == expected
    np.testing.assert_allclose(dimer_ground_hartree(100)*100**4,-1.3793/2,rtol=1e-13)


def test_spherical_average_converges_in_relevant_pair_separations():
    r=np.geomspace(.8,20,50)
    first=molecular_ion_neutral_pair_ev(r,n_angle=128)
    second=molecular_ion_neutral_pair_ev(r,n_angle=256)
    np.testing.assert_allclose(first,second,atol=2e-4)
