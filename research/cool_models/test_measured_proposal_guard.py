import numpy as np
from wd_spectra.nonlinear import NonlinearEvaluation,solve_trust_region_newton
from measured_proposal_guard import require_measured_proposal


def test_accepted_trust_limited_step_is_not_a_stationarity_certificate():
    def evaluate(state,jacobian):
        return NonlinearEvaluation(1e-4*(state-10.),np.eye(2)*1e-4 if jacobian else None,{})
    settings=dict(step_builder=lambda state,ev,jac,radius:np.full(2,radius),
        maximum_iterations=2,allow_initial_convergence=False,residual_tolerance=.003,
        step_tolerance=.01,initial_trust_radius=.001,maximum_trust_radius=.002)
    old=solve_trust_region_newton(np.zeros(2),evaluate,**settings)
    guarded=solve_trust_region_newton(np.zeros(2),evaluate,**require_measured_proposal(settings))
    assert old.converged
    assert not guarded.converged
    assert guarded.state[0]<.01


def test_genuine_small_correction_passes_and_physical_gate_is_retained():
    def evaluate(state,jacobian):
        return NonlinearEvaluation(state-1e-5,np.eye(2) if jacobian else None,{})
    settings=dict(step_builder=lambda state,ev,jac,radius:-ev.residual,
        maximum_iterations=2,step_tolerance=2e-4,initial_trust_radius=.04)
    result=solve_trust_region_newton(np.zeros(2),evaluate,**require_measured_proposal(settings))
    assert result.converged
    settings['convergence_test']=lambda state,ev,step:False
    refused=solve_trust_region_newton(np.zeros(2),evaluate,**require_measured_proposal(settings))
    assert not refused.converged


def test_tiny_unsolved_inner_proposal_cannot_certify_stationarity():
    def evaluate(state,jacobian):
        return NonlinearEvaluation(state-1e-5,np.eye(2) if jacobian else None,{})
    def proposal(state,ev,jac,radius):
        ev.payload['diagnostic_inner_proposal_root_solved']=False
        return np.zeros_like(state)
    settings=dict(step_builder=proposal,maximum_iterations=2,
        step_tolerance=2e-4,initial_trust_radius=.04)
    result=solve_trust_region_newton(np.zeros(2),evaluate,**require_measured_proposal(settings))
    assert not result.converged
