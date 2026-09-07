"""Rotating a proposal must not exclude an admissible uniform temperature step."""
import numpy as np
from wd_spectra.nonlinear import NonlinearEvaluation
import solver_step_experiments as module


def test_uniform_mode_uses_nodal_not_rotated_coordinate_bound(monkeypatch):
    n=40
    uniform=np.ones((n,n))/n
    jac=uniform+1e3*(np.eye(n)-uniform)
    desired=np.full(n,.02)
    def model(*args,**kwargs):
        return lambda delta:(jac@(delta-desired),jac)
    monkeypatch.setattr(module,'nonlinear_convection_model',model)
    ev=NonlinearEvaluation(-jac@desired,jac,dict(log_temperature_from_state=np.eye(n),
                                               diagnostic_fixed_cell_scale=np.ones(n-1)))
    step=module.make_step(np.zeros(n),ev,jac,.04,'nonlinear-convection-current-energy',
                          inner_scaling='svd',inner_max_evaluations=1000)
    np.testing.assert_allclose(step,desired,rtol=0,atol=1e-7)
    assert np.max(abs(step))<.04


def test_nonlinear_proposal_uses_driver_measured_secant(monkeypatch):
    # At an unchanged physical state the analytic payload is unchanged,
    # but a rejected trial has measured twice the original slope. The next
    # proposal must use that information rather than repeat the first step.
    n=3;mapping=np.array([[1.,0,0],[1.,1.,0],[1.,1.,1.]])
    residual=np.full(n,.02)
    def model(*args,**kwargs):
        return lambda delta:(residual+delta+delta**2,np.eye(n)+np.diag(2*delta))
    monkeypatch.setattr(module,'nonlinear_convection_model',model)
    ev=NonlinearEvaluation(residual,mapping,dict(log_temperature_from_state=mapping,
                                               diagnostic_measured_proposals=True))
    raw=module.make_step(np.zeros(n),ev,mapping,.04,'nonlinear-convection',
                        inner_scaling='svd',inner_max_evaluations=1000)
    repaired=module.make_step(np.zeros(n),ev,2*mapping,.04,'nonlinear-convection',
                             inner_scaling='svd',inner_max_evaluations=1000)
    dt=mapping@repaired
    np.testing.assert_allclose(residual+2*dt+dt**2,0.,atol=1e-9)
    assert np.max(abs(dt)) < .6*np.max(abs(mapping@raw))
