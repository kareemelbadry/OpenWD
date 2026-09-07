from types import SimpleNamespace
import numpy as np
import pytest
from inverse_ml2_proposal import inverse_model,constrained_inverse_step
from solver_step_experiments import nonlinear_convection_model
from test_solver_step_experiments import trial_evaluation
from wd_spectra.constants import STEFAN_BOLTZMANN


def energy_evaluation(exact=False):
    ev=trial_evaluation(); p=ev.payload
    p['atmosphere'].n_depth=4
    target=STEFAN_BOLTZMANN*8000**4
    p['radiative_cell_energy_defect']=np.diff(p['radiative_flux_interface'])
    p['radiative_cell_energy_log_temperature_jacobian']=np.diff(p['radiative_flux_log_temperature_jacobian'],axis=0)
    p['thermal_cell_emission']=target*np.array([.1,.3,20.])
    p['thermal_cell_emission_log_temperature_jacobian']=p['thermal_cell_emission'][:,None]*np.eye(3,4)*.7
    if exact:
        tr=p['convection_transport']
        a,b,c=(tr[k] for k in ('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient'))
        aj,bj,cj=p['ml2_coefficient_log_temperature_responses']
        def material(dt):
            coef=(a+aj@dt+.2*dt**2,b*np.exp((bj/b[:,None])@dt),c*np.exp((cj/c[:,None])@dt))
            jac=(aj+np.diag(.4*dt),coef[1][:,None]*(bj/b[:,None]),coef[2][:,None]*(cj/c[:,None]))
            return coef,jac
        class Exact:
            evaluate=staticmethod(material)
            __call__=staticmethod(material)
        p['diagnostic_material_model']=Exact()
    return ev


@pytest.mark.parametrize('exact',[False,True])
@pytest.mark.parametrize('cell_only',[False,True])
def test_inverse_eliminates_compatibility_without_changing_energy_equations(exact,cell_only):
    ev=energy_evaluation(exact)
    inverse=inverse_model(ev,cell_only=cell_only)
    direct=nonlinear_convection_model(ev,np.ones(3),direct_energy=True,current_energy=True,cell_only=cell_only)
    z=np.array([.01,-.02,.01,-.01])
    r,j,dt,dtj=inverse(z)
    np.testing.assert_allclose(r,direct(dt)[0],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(j,direct(dt)[1]@dtj,rtol=1e-11,atol=1e-12)
    h=1e-6
    fd=np.column_stack([(inverse(z+h*d)[0]-inverse(z-h*d)[0])/(2*h) for d in np.eye(4)])
    np.testing.assert_allclose(j,fd,rtol=1e-7,atol=1e-9)


def test_constrained_inverse_satisfies_true_temperature_box_without_rescaling():
    ev=energy_evaluation(True)
    step=constrained_inverse_step(np.zeros(4),ev,.04,200)
    delta=ev.payload['log_temperature_from_state']@step
    assert np.max(abs(delta)) <= .04*(1+1e-8)
    direct=nonlinear_convection_model(ev,np.ones(3),direct_energy=True,current_energy=True)
    assert np.linalg.norm(direct(delta)[0]) < np.linalg.norm(direct(np.zeros(4))[0])
