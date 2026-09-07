from types import SimpleNamespace
import numpy as np
from wd_spectra import adaptive_structure as adaptive
from wd_spectra.nonlinear import NonlinearEvaluation
from wd_spectra.constants import STEFAN_BOLTZMANN
from exact_dense_tangent_experiment import replace_material_tangent


def test_material_replacement_updates_flux_energy_and_scale_without_state_change():
    n=4;gradient=np.array([0.,.45,.46,.5]);target=STEFAN_BOLTZMANN*5000**4
    coefficients=(np.full(n,.4),np.full(n,.01),np.full(n,target*100))
    zero=tuple(np.zeros((n,n)) for _ in range(3))
    new=(np.eye(n)*.02,np.eye(n)*.03,np.eye(n)*target)
    mapping=np.tril(np.ones((n,n)))
    payload=dict(atmosphere=SimpleNamespace(effective_temperature=5000.),
        convection_transport=dict(zip(('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient'),coefficients)),
        temperature_gradient=gradient,ml2_coefficient_log_temperature_responses=zero,
        log_temperature_from_state=mapping,cell_energy_log_temperature_jacobian=np.ones((n-1,n)),
        cell_energy_scale_log_temperature_jacobian=np.full((n-1,n),2.))
    original=NonlinearEvaluation(np.arange(n,dtype=float),np.eye(n),payload)
    result=replace_material_tangent(original,new)
    correction=adaptive._ml2_flux_coefficient_response(gradient,coefficients,new);correction[0]=0
    np.testing.assert_allclose(result.jacobian,np.eye(n)+correction@mapping/target)
    np.testing.assert_allclose(result.payload['cell_energy_log_temperature_jacobian'],1+np.diff(correction,axis=0))
    np.testing.assert_allclose(result.payload['cell_energy_scale_log_temperature_jacobian'],2+correction[:-1]+correction[1:])
    assert result.residual is original.residual
    assert result.payload['atmosphere'] is original.payload['atmosphere']
    np.testing.assert_array_equal(payload['cell_energy_log_temperature_jacobian'],np.ones((n-1,n)))
