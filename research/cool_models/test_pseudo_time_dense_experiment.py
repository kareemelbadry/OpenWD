import numpy as np
import pytest
from pseudo_time_dense_experiment import pseudo_direction,physical_rows


def test_stable_heating_and_algebraic_boundary_sign():
    residual=np.array([2.,-3.,.1]);jacobian=np.diag([-4.,-5.,2.])
    result=pseudo_direction(residual,jacobian,.5)
    np.testing.assert_allclose(result,[2./6.,-3./7.,-.05])
    np.testing.assert_array_equal(jacobian,np.diag([-4.,-5.,2.]))


def test_can_cross_static_merit_barrier_without_changing_root():
    # A stable cold root with a positive-slope heating curve at the start:
    # ordinary Newton initially heats in the WRONG thermal direction.
    x=0.;initial=None;maximum=0.
    for _ in range(120):
        q=-(x+.5)*((x-.4)**2+.001)
        dq=-((x-.4)**2+.001)-2*(x+.5)*(x-.4)
        if initial is None:initial=abs(q)
        maximum=max(maximum,abs(q))
        delta=pseudo_direction(np.array([q,0.]),np.diag([dq,1.]),.25)[0]
        assert delta<=1e-14
        x+=delta
    assert maximum>initial
    np.testing.assert_allclose(x,-.5,atol=1e-8)


def test_boundary_reservoir_has_correct_thermal_sign_and_vanishing_small_step():
    residual=np.array([2.,-3.,.1]);jacobian=np.diag([-4.,-5.,2.])
    result=pseudo_direction(residual,jacobian,.5,relax_boundary=True)
    np.testing.assert_allclose(result,[2./6.,-3./7.,-.025])
    # Unlike an algebraic Newton boundary, every increment vanishes with dt,
    # including when the static boundary tangent is identically zero.
    jacobian[-1,-1]=0.
    for dt in (1e-4,1e-8):
        result=pseudo_direction(residual,jacobian,dt,relax_boundary=True)
        np.testing.assert_allclose(result[-1],-.1*dt)
        assert result[0]>0 and result[1]<0


def test_rows_use_actual_flux_and_frozen_transport_scale():
    p=dict(radiative_cell_energy_defect=np.array([2.,3.]),
        convective_flux_interface=np.array([0.,5.,7.]),total_flux_interface=np.array([8.,9.,11.]),
        cell_energy_log_temperature_jacobian=np.arange(6).reshape(2,3),
        radiative_flux_log_temperature_jacobian=np.ones((3,3)),
        radiative_cell_energy_log_temperature_jacobian=np.arange(6).reshape(2,3)-1.)
    r,j=physical_rows(p,np.array([2.,5.]),10.,True)
    np.testing.assert_allclose(r,[3.5,1.,.1])
    np.testing.assert_allclose(j[-1],[.3,.3,.3])


@pytest.mark.parametrize('domain_limited',[False,True])
def test_local_handoff_still_requires_the_static_solver(tmp_path,monkeypatch,domain_limited):
    from wd_spectra import adaptive_structure as adaptive
    from wd_spectra.nonlinear import NonlinearEvaluation
    from pseudo_time_dense_experiment import pseudo_time_initializer
    calls=[];sentinel=object()
    def static(initial,evaluate,**options):
        calls.append((initial.copy(),options.copy()))
        return sentinel
    monkeypatch.setattr(adaptive,'solve_trust_region_newton',static)
    initial=np.array([1.,2.])
    def evaluate(state,jac):
        return NonlinearEvaluation(np.ones(2),np.eye(2) if jac else None,
            dict(energy_balance_is_physical_flux=True,
                 material_temperature_response_domain_limited=domain_limited,
                 cell_energy_balance_relative_residual=np.array([.5 if domain_limited else 1e-5])))
    with pseudo_time_initializer(30,tmp_path/'telemetry.jsonl',until_local_balance=True):
        result=adaptive.solve_trust_region_newton(initial,evaluate,residual_tolerance=.003,
            allow_initial_convergence=True)
    assert result is sentinel
    assert len(calls)==1
    np.testing.assert_array_equal(calls[0][0],initial)
    assert calls[0][1]['allow_initial_convergence'] is False
    assert adaptive.solve_trust_region_newton is static
