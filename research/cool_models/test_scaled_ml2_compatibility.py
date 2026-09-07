import numpy as np
from scaled_ml2_compatibility import scaled_compatibility


def test_current_compatibility_scale_differentiated_on_both_branches_and_onset():
    loss=np.array([.002,.01,.02]);coef=np.array([1e8,1e9,1e8]);ad=np.array([.4,.35,.4])
    def at(z):
        dt=z[:3];y=z[3:]
        coefficients=(ad+.03*dt,loss*np.exp(.4*dt),coef*np.exp(-.2*dt))
        responses=(.03*np.eye(3),np.diag(.4*coefficients[1]),np.diag(-.2*coefficients[2]))
        r,t,yj=scaled_compatibility(np.array([.2,.36,.5])+dt,y,coefficients,responses,np.eye(3),1.)
        return r,np.hstack((t,np.diag(yj)))
    z=np.array([.01,-.02,.01,-20.,0.,.7])
    h=1e-6
    def difference(h):
        return np.column_stack([(at(z+h*d)[0]-at(z-h*d)[0])/(2*h) for d in np.eye(6)])
    # The scale is C1 at onset, so a centered stencil has an O(h) term
    # there. Richardson cancellation tests the limiting derivative rather
    # than relaxing the tolerance on that known one-sided curvature term.
    fd=2*difference(h/2)-difference(h)
    np.testing.assert_allclose(at(z)[1],fd,rtol=1e-7,atol=1e-5)
    # A convective auxiliary state cannot hide a huge excess behind a
    # normalization inherited from the seed's initially stable gradient.
    r,_,_=scaled_compatibility(np.array([.41]),np.array([1.]),
        (np.array([.4]),np.array([.001]),np.array([1e9])),None,None,1.)
    assert r[0] > 1000


def test_local_emission_norm_does_not_hide_tiny_but_thermally_important_convection():
    coefficients=(np.array([.4]),np.array([1000.]),np.array([1.]))
    old,*_=scaled_compatibility(np.array([.5]),np.array([0.]),coefficients,None,None,1.)
    new,*_=scaled_compatibility(np.array([.5]),np.array([0.]),coefficients,None,None,1.,
        thermal_scale=np.array([1e-12]))
    assert abs(old[0])<.00011
    assert abs(new[0])>.99


def test_thermal_norm_tangent_on_stable_onset_and_convective_branches():
    def at(z):
        dt=z[:3];y=z[3:]
        ad=np.full(3,.4);loss=.01*np.exp(.4*dt);coef=1e8*np.exp(-.2*dt)
        coefficients=(ad+.03*dt,loss,coef)
        responses=(.03*np.eye(3),np.diag(.4*loss),np.diag(-.2*coef))
        thermal=np.array([1e-8,.1,1e3])*np.exp(2*dt)
        r,t,yj=scaled_compatibility(np.array([.2,.36,.5])+dt,y,coefficients,responses,np.eye(3),1.,
            thermal_scale=thermal,thermal_scale_jacobian=np.diag(2*thermal))
        return r,np.hstack((t,np.diag(yj)))
    z=np.array([.01,-.02,.01,-20.,0.,.7]);h=1e-6
    def difference(h):return np.column_stack([(at(z+h*d)[0]-at(z-h*d)[0])/(2*h) for d in np.eye(6)])
    fd=2*difference(h/2)-difference(h)
    np.testing.assert_allclose(at(z)[1],fd,rtol=1e-7,atol=1e-5)
