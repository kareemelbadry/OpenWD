import numpy as np
import pytest
from functools import partial
from augmented_ml2_proposal import augmented_model,augmented_step
from inverse_ml2_proposal import newton_inverse_step
from test_inverse_ml2_proposal import energy_evaluation
from solver_step_experiments import nonlinear_convection_model


@pytest.mark.parametrize('cell_only',[False,True])
@pytest.mark.parametrize('positive',[False,True])
@pytest.mark.parametrize('current_scale,thermal_scale',[(False,False),(True,False),(True,True)])
def test_augmented_energy_and_compatibility_tangent(cell_only,positive,current_scale,thermal_scale):
    ev=energy_evaluation(True)
    # Ensure a physically positive heating rate for this synthetic column.
    ev.payload['thermal_cell_emission']*=10
    ev.payload['thermal_cell_emission_log_temperature_jacobian']*=10
    ev.payload['diagnostic_positive_radiative_rates']=positive
    model=augmented_model(ev,cell_only=cell_only,velocity_scaled_compatibility=current_scale,
        thermal_scaled_compatibility=thermal_scale)
    direct=nonlinear_convection_model(ev,np.ones(3),direct_energy=True,current_energy=True,cell_only=cell_only)
    np.testing.assert_allclose(model(np.zeros(7))[0][:4],direct(np.zeros(4))[0],atol=1e-14)
    np.testing.assert_allclose(model(np.zeros(7))[0][4:],0.,atol=1e-14)
    z=np.array([.01,-.02,.03,-.01,.1,-.03,.02])
    h=1e-6
    fd=np.column_stack([(model(z+h*d)[0]-model(z-h*d)[0])/(2*h) for d in np.eye(7)])
    np.testing.assert_allclose(model(z)[1],fd,rtol=1e-7,atol=1e-9)


@pytest.mark.parametrize('stepper',[augmented_step,newton_inverse_step,
    partial(augmented_step,project_compatibility=True),partial(augmented_step,equilibrate=True)])
def test_coupled_step_temperature_bound(stepper):
    ev=energy_evaluation(True)
    step=stepper(np.zeros(4),ev,.04,300)
    dt=ev.payload['log_temperature_from_state']@step
    assert np.max(abs(dt)) <= .04*(1+1e-12)
    direct=nonlinear_convection_model(ev,np.ones(3),direct_energy=True,current_energy=True)
    assert np.linalg.norm(direct(dt)[0]) < np.linalg.norm(direct(np.zeros(4))[0])


@pytest.mark.parametrize('time_step',[.01,1.,100.])
@pytest.mark.parametrize('relax_bottom',[False,True])
def test_augmented_pseudo_time_tangent(time_step,relax_bottom):
    ev=energy_evaluation(True)
    ev.payload['thermal_cell_emission']*=10
    ev.payload['thermal_cell_emission_log_temperature_jacobian']*=10
    ev.payload['diagnostic_positive_radiative_rates']=True
    ev.payload['cell_energy_scale']=ev.payload['thermal_cell_emission'].copy()
    model=augmented_model(ev,velocity_scaled_compatibility=True,
        thermal_scaled_compatibility=True,pseudo_time_step=time_step,pseudo_relax_bottom=relax_bottom)
    z=np.array([.01,-.02,.03,-.01,.1,-.03,.02]);h=1e-6
    fd=np.column_stack([(model(z+h*d)[0]-model(z-h*d)[0])/(2*h) for d in np.eye(7)])
    np.testing.assert_allclose(model(z)[1],fd,rtol=1e-7,atol=1e-9)


def test_local_material_tangent_preserves_base_and_its_own_derivative():
    from tangent_ml2_materials import TangentML2Materials
    p=energy_evaluation(True).payload
    model=TangentML2Materials(p)
    base,derivative=model(np.zeros(4))
    for key,value in zip(('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient'),base):
        np.testing.assert_allclose(value,p['convection_transport'][key],rtol=1e-15)
    z=np.array([.01,-.02,.03,-.01]);h=1e-6
    values,jac=model(z)
    for i in range(3):
        fd=np.column_stack([(model(z+h*d)[0][i]-model(z-h*d)[0][i])/(2*h) for d in np.eye(4)])
        np.testing.assert_allclose(jac[i],fd,rtol=1e-7,atol=1e-9)
