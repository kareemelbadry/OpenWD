from types import SimpleNamespace
import numpy as np
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.nonlinear import NonlinearEvaluation
from frozen_energy_linearization import frozen_energy_evaluator


def test_weights_change_only_at_new_linearizations_and_tangent_matches_trials():
    target=STEFAN_BOLTZMANN*5000**4;n=4
    matrix=np.eye(n)+.1
    def evaluate(state,need):
        f=1+matrix@state
        p=dict(atmosphere=SimpleNamespace(effective_temperature=5000.),
            energy_balance_is_physical_flux=True,
            cell_energy_scale=target*np.exp(100*state[:-1]),
            radiative_cell_energy_defect=target*np.diff(f),
            convective_flux_interface=np.zeros(n),total_flux_interface=target*f,
            radiative_flux_log_temperature_jacobian=target*matrix,
            convective_flux_log_temperature_jacobian=np.zeros((n,n)),
            cell_energy_log_temperature_jacobian=target*np.diff(matrix,axis=0),
            radiative_cell_energy_log_temperature_jacobian=target*np.diff(matrix,axis=0),
            log_temperature_from_state=np.eye(n))
        return NonlinearEvaluation(f-1,matrix if need else None,p)
    weighted=frozen_energy_evaluator(evaluate)
    x=np.array([.01,-.02,.03,.04]);base=weighted(x,True);h=1e-6
    measured=np.column_stack([(weighted(x+h*d,False).residual-weighted(x-h*d,False).residual)/(2*h)
        for d in np.eye(n)])
    np.testing.assert_allclose(base.jacobian,measured,rtol=1e-9,atol=1e-9)
    old=base.payload['diagnostic_fixed_cell_scale'].copy()
    moved=x+.01
    np.testing.assert_array_equal(weighted(moved,False).payload['diagnostic_fixed_cell_scale'],old)
    newer=weighted(moved,True)
    assert np.all(newer.payload['diagnostic_fixed_cell_scale']>old)
    np.testing.assert_array_equal(base.payload['diagnostic_fixed_cell_scale'],old)
    root=weighted(np.zeros(n),False)
    np.testing.assert_array_equal(root.residual,np.zeros(n))
