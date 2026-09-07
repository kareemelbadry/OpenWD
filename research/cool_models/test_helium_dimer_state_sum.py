import numpy as np
import pytest
from helium_dimer_state_sum import calculate_bound_states
from wd_spectra import helium_molecular as molecular


def test_bound_state_grid_box_and_published_equilibrium_comparison():
    standard=calculate_bound_states()
    fine=calculate_bound_states(points=32767,extent=40.)
    wide=calculate_bound_states(points=32767,extent=60.)
    t=np.array([1001.,2000.,3000.,4200.,6300.,8400.,12600.,16800.])
    assert np.all(standard.rotational_quantum_number % 2 == 1)
    assert np.all(standard.energy_ev < 0)
    np.testing.assert_allclose(standard.equilibrium(t)[0],fine.equilibrium(t)[0],rtol=1e-4)
    np.testing.assert_allclose(wide.equilibrium(t)[0],fine.equilibrium(t)[0],rtol=1e-4)
    # This is a different published potential. Check agreement, do not fit
    # levels or rescale the partition function to Stancil's table.
    ratio=standard.equilibrium(molecular._TEMPERATURE[:5])[0]/molecular._EQUILIBRIUM_CONSTANT[:5]
    assert np.max(abs(ratio-1)) < .05
    eps=1e-5
    derivative=np.log(standard.equilibrium(t*np.exp(eps))[0]/
                      standard.equilibrium(t*np.exp(-eps))[0])/(2*eps)
    np.testing.assert_allclose(derivative,standard.equilibrium(t)[1],rtol=1e-8)
    with pytest.raises(ValueError,match='domain'):
        standard.equilibrium(999)
