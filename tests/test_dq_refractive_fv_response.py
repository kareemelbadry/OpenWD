import numpy as np
import pytest
from wd_spectra._dq.dq_refractive_finite_volume import solve
from wd_spectra._dq.dq_refractive_fv_response import integrated_response


@pytest.mark.parametrize('shape',['monotonic','interior_peak','two_peaks'])
@pytest.mark.parametrize('optical_scale',[1.,1e-4])
def test_geometry_and_scattering_tangent_matches_full_finite_difference(shape,optical_scale):
    mass=np.geomspace(1e-4,12,12);wave=np.array([4000.,5000.,7000.])
    t=np.linspace(.8,2.,len(mass));r=np.array([.8,1.,1.1])[:,None]
    profile=1-np.exp(-mass)
    if shape=='interior_peak':profile*=np.exp(-mass/5.)
    if shape=='two_peaks':
        profile=np.exp(-np.log(mass/.02)**2)+1.4*np.exp(-np.log(mass/2.)**2)
    def state(x):
        return (optical_scale*.3*r*x**.7,optical_scale*.5*r*x**-.3,r*x**4,
            1+.12*r*profile*x**-.2)
    base=state(t);a,s,b,n=base
    derivatives=(.7*a,-.3*s,4*b,-.2*(n-1))
    analytic=integrated_response(mass,wave,*base,*derivatives,n_angle=6)
    expected=[np.empty_like(x) for x in analytic];h=2e-5
    for j in range(len(mass)):
        outputs=[]
        for sign in (1.,-1.):
            v=t.copy();v[j]*=np.exp(sign*h)
            f=solve(mass,*state(v),n_angle=6)
            outputs.append([np.trapz(f[key],wave,axis=0) for key in
                ('flux','cell_heating','cell_thermal_emission')])
        for e,high,low in zip(expected,*outputs):e[:,j]=(high-low)/(2*h)
    for value,exact in zip(analytic,expected):
        np.testing.assert_allclose(value,exact,rtol=2e-5,atol=np.max(abs(exact))*1e-8)
    np.testing.assert_allclose(np.diff(analytic[0],axis=0),analytic[1],rtol=2e-9,atol=1e-7)


def test_nonrefractive_planck_tangent_is_linear_operator():
    mass=np.geomspace(1e-5,20,16);wave=np.array([3000.,6000.])
    a=np.ones((2,len(mass)));s=a*.5;b=1+mass[None,:]*a;n=a.copy()
    zero=np.zeros_like(a)
    jac=integrated_response(mass,wave,a,s,b,n,zero,zero,b,zero,n_angle=4)
    result=solve(mass,a,s,b,n,n_angle=4)
    for value,key in zip(jac,('flux','cell_heating','cell_thermal_emission')):
        np.testing.assert_allclose(value.sum(axis=1),np.trapz(result[key],wave,axis=0),rtol=2e-10,atol=2e-8)
