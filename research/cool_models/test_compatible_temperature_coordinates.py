import numpy as np
import pytest
from compatible_temperature_coordinates import CompatibleTemperatures
from wd_spectra._ml2_auxiliary import ml2_auxiliary_from_gradient


def materials(logt,derivative=False):
    n=len(logt)
    mean=np.eye(n)*.5+np.eye(n,k=-1)*.5;mean[0,0]=1.
    t=mean@logt
    fields=(.3+.01*(t-8.),np.exp(-4+.1*(t-8.)),np.exp(25+.3*(t-8.)))
    if not derivative:return fields
    return fields,(.01*mean,fields[1][:,None]*.1*mean,fields[2][:,None]*.3*mean)


def test_identity_and_implicit_tangent_include_finite_materials():
    logt=np.array([8.,8.2,8.65,9.05])
    p=np.exp(np.arange(4)+10.)
    coords=CompatibleTemperatures(logt,p,materials,1e8)
    recovered,mapping,defect=coords.decode(coords.initial)
    np.testing.assert_allclose(recovered,logt,atol=1e-12)
    assert defect<1e-7
    h=1e-5
    measured=np.column_stack([(coords.decode(coords.initial+h*d)[0]-
        coords.decode(coords.initial-h*d)[0])/(2*h) for d in np.eye(4)])
    np.testing.assert_allclose(mapping,measured,rtol=3e-6,atol=1e-9)


def test_actual_gradient_flux_matches_coordinate_and_its_tangent():
    p=np.exp(np.arange(4)+10.)
    coords=CompatibleTemperatures(np.array([8.,8.2,8.65,9.05]),p,materials,1e8)
    q=coords.initial+np.array([.01,.001,-.002,.002])
    def flux(q):
        logt,_,_=coords.decode(q)
        y=ml2_auxiliary_from_gradient(np.diff(logt),*(v[1:] for v in materials(logt)),1e8)
        return np.r_[0.,np.maximum(y,0.)**3]*1e8
    np.testing.assert_allclose(flux(q),np.r_[0.,np.maximum(q[1:]*coords.scale,0.)**3]*1e8,
        atol=10.,rtol=1e-10)
    h=1e-5
    measured=np.column_stack([(flux(q+h*d)-flux(q-h*d))/(2*h) for d in np.eye(4)])
    np.testing.assert_allclose(coords.convective_tangent(q),measured,rtol=3e-6,atol=2.)


def test_physical_energy_chain_rule_uses_actual_gradient_flux():
    from compatible_coordinate_solve import energy_tangent,coordinate_energy
    target=1e8
    reference=np.array([8.,8.2,8.65,9.05])
    coords=CompatibleTemperatures(reference,np.exp(np.arange(4)+10.),materials,target)
    q=coords.initial+np.array([.01,.001,-.002,.002])
    def evaluate(q):
        logt,mapping,_=coords.decode(q)
        x=logt-reference
        y=ml2_auxiliary_from_gradient(np.diff(logt),*(v[1:] for v in materials(logt)),target)
        convection=np.r_[0.,np.maximum(y,0.)**3]*target
        radiation=target*(.7+.2*x)
        thermal=target*np.exp(x[:-1])
        exchange=target*.1*np.sin(x[:-1])
        scale=thermal+convection[:-1]+convection[1:]
        payload=dict(radiative_flux_log_temperature_jacobian=target*.2*np.eye(4),
            radiative_cell_energy_log_temperature_jacobian=target*.1*np.cos(x[:-1])[:,None]*np.eye(4)[:-1],
            thermal_cell_emission_log_temperature_jacobian=thermal[:,None]*np.eye(4)[:-1],
            radiative_cell_energy_defect=exchange,convective_flux_interface=convection,cell_energy_scale=scale)
        payload.update(radiative_flux_interface=radiation,thermal_cell_emission=thermal)
        values=(radiation+convection)/target-1.
        values[:-1]-=(exchange+np.diff(convection))/scale
        compatible_flux=np.r_[0.,np.maximum(q[1:]*coords.scale,0.)**3]*target
        alternative,_=coordinate_energy(payload,compatible_flux,mapping,coords.convective_tangent(q),target)
        np.testing.assert_allclose(alternative,values,rtol=1e-10,atol=1e-7)
        np.testing.assert_array_equal(payload['convective_flux_interface'],convection)
        return values,energy_tangent(payload,mapping,coords.convective_tangent(q),target)
    _,jac=evaluate(q)
    h=1e-5
    measured=np.column_stack([(evaluate(q+h*d)[0]-evaluate(q-h*d)[0])/(2*h) for d in np.eye(4)])
    np.testing.assert_allclose(jac,measured,rtol=3e-6,atol=1e-8)


@pytest.mark.parametrize('pressure',[[1.],[1.,1.],[2.,1.],[0.,1.],[1.,np.nan]])
def test_invalid_physical_coordinate_grid_rejected(pressure):
    with pytest.raises(ValueError):CompatibleTemperatures(np.array([8.,8.1]),pressure,materials,1e8)


def test_coordinate_proposal_bounds_physical_temperature_before_decoding():
    from compatible_coordinate_solve import linear_trust_fraction
    mapping=np.array([[1.,0.],[1.,1e-5]])
    direction=np.array([100.,-2e6])
    fraction=linear_trust_fraction(mapping,direction,.04)
    assert fraction<1.
    np.testing.assert_allclose(np.max(abs(mapping@(fraction*direction))),.04)
    assert linear_trust_fraction(mapping,np.zeros(2),.04)==1.
    assert linear_trust_fraction(mapping,np.array([1e-5,0.]),.04)==1.


def test_physical_box_does_not_throttle_resolved_mode_by_weak_mode():
    from compatible_coordinate_solve import physical_box_direction
    jac=np.diag([1.,1e-8]);residual=np.array([.01,1e-4])
    direction,root=physical_box_direction(jac,residual,np.eye(2),.04)
    assert not root
    np.testing.assert_allclose(direction[0],-.01,rtol=1e-12)
    assert np.max(abs(direction))<=.04
    assert np.linalg.norm(jac@direction+residual)<.02*np.linalg.norm(residual)


def test_physical_box_preserves_unconstrained_newton_root_step():
    from compatible_coordinate_solve import physical_box_direction
    jac=np.array([[2.,1.],[.5,3.]])
    residual=np.array([1e-5,2e-5])
    mapping=np.array([[1.,0.],[1.,.1]])
    direction,root=physical_box_direction(jac,residual,mapping,.04)
    assert root
    np.testing.assert_allclose(jac@direction,-residual,atol=1e-18)


def test_matching_merit_accepts_squared_residual_descent_without_changing_default():
    from wd_spectra import nonlinear
    from compatible_coordinate_solve import compatible_coordinate_solver,least_squares_merit
    before=np.array([1.,1.]);after=np.array([1.05,.94])
    default=nonlinear._residual_merit
    assert default(after)>default(before)
    assert least_squares_merit(after)<least_squares_merit(before)
    with compatible_coordinate_solver({}):
        assert nonlinear._residual_merit is least_squares_merit
    assert nonlinear._residual_merit is default
