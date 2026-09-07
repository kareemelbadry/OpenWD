import numpy as np
import pytest
from full_coupled_dense_experiment import current_augmented_energy,current_compatibility
from test_inverse_ml2_proposal import energy_evaluation
from wd_spectra.constants import STEFAN_BOLTZMANN


@pytest.mark.parametrize('normalization',[None,np.array([.2,.5,2.])])
def test_full_current_energy_tangent_in_original_state_and_velocity_coordinates(normalization):
    p=energy_evaluation(True).payload
    target=STEFAN_BOLTZMANN*8000.**4
    mapping=p['log_temperature_from_state']
    scale=np.array([2.,1.,1.])
    y0=np.array([-.3,.8,1.1])
    p['thermal_cell_emission']*=10
    p['thermal_cell_emission_log_temperature_jacobian']*=10
    log_cj=p['thermal_cell_emission_log_temperature_jacobian']/p['thermal_cell_emission'][:,None]
    def at(z):
        dt=mapping@z[:4]
        c=p['thermal_cell_emission']*np.exp(log_cj@dt)
        actual={**p,
            'radiative_flux_interface':p['radiative_flux_interface']+p['radiative_flux_log_temperature_jacobian']@dt,
            'radiative_cell_energy_defect':p['radiative_cell_energy_defect']+p['radiative_cell_energy_log_temperature_jacobian']@dt,
            'thermal_cell_emission':c,
            'thermal_cell_emission_log_temperature_jacobian':c[:,None]*log_cj}
        return current_augmented_energy(actual,y0+scale*z[4:],scale,target,mapping,normalization=normalization)
    z=np.array([.01,-.02,.01,.03,-.1,.04,.02])
    h=1e-6
    fd=np.column_stack([(at(z+h*d)[0]-at(z-h*d)[0])/(2*h) for d in np.eye(7)])
    np.testing.assert_allclose(at(z)[1],fd,rtol=2e-8,atol=1e-9)


@pytest.mark.parametrize('thermal_scaling',[False,True])
@pytest.mark.parametrize('normalization',[None,np.array([.03,.2,.1])])
def test_full_current_compatibility_tangent(thermal_scaling,normalization):
    p=energy_evaluation(True).payload
    mapping=p['log_temperature_from_state']
    target=STEFAN_BOLTZMANN*8000.**4
    scale=np.array([2.,1.,1.])
    y0=np.array([-.3,.8,1.1])
    log_cj=p['thermal_cell_emission_log_temperature_jacobian']/p['thermal_cell_emission'][:,None]
    keys=('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient')
    def at(z):
        dt=mapping@z[:4]
        coef,responses=p['diagnostic_material_model'](dt)
        c=p['thermal_cell_emission']*np.exp(log_cj@dt)
        actual={**p,'convection_transport':dict(zip(keys,coef)),
            'ml2_coefficient_log_temperature_responses':responses,
            'temperature_gradient':p['temperature_gradient']+p['interface_gradient_operator']@dt,
            'thermal_cell_emission':c,'thermal_cell_emission_log_temperature_jacobian':c[:,None]*log_cj}
        return current_compatibility(actual,y0+scale*z[4:],scale,target,mapping,
            thermal_scaling=thermal_scaling,normalization=normalization)
    z=np.array([.01,-.02,.01,.03,-.1,.04,.02]);h=1e-6
    fd=np.column_stack([(at(z+h*d)[0]-at(z-h*d)[0])/(2*h) for d in np.eye(7)])
    np.testing.assert_allclose(at(z)[1],fd,rtol=3e-8,atol=1e-9)


@pytest.mark.parametrize('secant',[False,True])
def test_bounded_schur_satisfies_linear_closure_and_avoids_giant_null_step(secant):
    from types import SimpleNamespace
    from full_coupled_dense_experiment import bounded_schur_step
    mapping=np.array([[1.,0.,0.],[1.,1.,0.],[1.,1.,1.]])
    ct=np.array([[1.,-1.,0.],[0.,1.,-1.]])
    ey=np.array([[.1,.2],[.3,.1],[.1,.1]])
    energy=np.diag([1.,1e-8,1.])
    cy=-np.eye(2)+(np.array([[.01,.02],[.03,.06]]) if secant else 0.)
    j=np.block([[(energy+ey@np.linalg.solve(cy,ct))@mapping,ey],[ct@mapping,cy]])
    residual=np.array([.1,.01,-.1,0.,0.])
    direction=bounded_schur_step(SimpleNamespace(residual=residual),j,.04,3,mapping)
    assert np.max(abs(mapping@direction[:3]))<=.04*(1+1e-13)
    np.testing.assert_allclose((j@direction+residual)[3:],0.,atol=1e-14)
    assert np.linalg.norm((j@direction+residual)[:3])<np.linalg.norm(residual[:3])
    assert np.max(abs(mapping@np.linalg.solve(j,-residual)[:3]))>1e5
