"""Research-data tests; no download or production-default substitution."""
import numpy as np
import pytest

from wd_spectra import molecules, eos
from wd_spectra.constants import BOLTZMANN
import state_sum_h2_experiment as experiment

pytestmark = pytest.mark.skipif(
    not experiment.DATA.is_file(), reason="Optional public RACPPK state data absent"
)


def test_partition_and_energy_obey_the_same_thermodynamic_potential():
    t = np.geomspace(50.0, 100000.0, 120)
    h = 1e-5
    q, energy, cv = experiment.moments(t)
    dlogq = (
        np.log(experiment.partition(t * np.exp(h)))
        - np.log(experiment.partition(t * np.exp(-h)))
    ) / (2 * h)
    denergy = (experiment.energy(t * np.exp(h)) - experiment.energy(t * np.exp(-h))) / (
        2 * h * t
    )
    np.testing.assert_allclose(energy, BOLTZMANN * t * dlogq, rtol=2e-9)
    np.testing.assert_allclose(cv, denergy, rtol=2e-9)
    assert np.all(q > 0) and np.all(energy >= 0) and np.all(cv >= 0)


def test_low_temperature_limit_and_no_negative_heat_capacity():
    t = np.geomspace(1.0, 100000.0, 1000)
    q, energy, cv = experiment.moments(t)
    assert q[0] == 0.25
    assert np.all(np.diff(q) >= 0)
    assert np.all(energy >= 0) and np.all(cv >= 0)
    assert q[-1] < np.sum(experiment.states()[1])


def test_independent_exomol_partition_table():
    path = experiment.DATA.with_name("1H2__RACPPK.pf")
    if not path.is_file():
        pytest.skip("Optional independently published partition table absent")
    table = np.loadtxt(path)
    # The published PF has four decimal places. The additional ~6e-7
    # relative difference at high T is consistent with older hc/k constants;
    # it is not corrected or fitted away in the state sum.
    np.testing.assert_allclose(
        4 * experiment.partition(table[:, 0]), table[:, 1], atol=5.1e-5, rtol=1e-6
    )


def test_patches_are_scoped_and_restore_both_eos_imports():
    original_q = molecules.molecular_hydrogen_internal_partition_function
    original_e = molecules.molecular_hydrogen_rovibrational_energy
    original_eos_e = eos.molecular_hydrogen_rovibrational_energy
    with experiment.state_sum_h2_experiment():
        assert (
            molecules.molecular_hydrogen_internal_partition_function
            is experiment.partition
        )
        assert molecules.molecular_hydrogen_rovibrational_energy is experiment.energy
        assert eos.molecular_hydrogen_rovibrational_energy is experiment.energy
    assert molecules.molecular_hydrogen_internal_partition_function is original_q
    assert molecules.molecular_hydrogen_rovibrational_energy is original_e
    assert eos.molecular_hydrogen_rovibrational_energy is original_eos_e


@pytest.mark.parametrize("t", [0.0, -1.0, np.nan, np.inf])
def test_invalid_temperature_is_not_clamped(t):
    with pytest.raises(ValueError, match="finite positive"):
        experiment.moments(t)
