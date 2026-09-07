"""Independent conservation and tangent tests for the opt-in mass operator."""
import numpy as np
import pytest
from wd_spectra._compat import trapezoid
from wd_spectra.opacity import optical_depth_from_mass_opacity
from wd_spectra._stable_feautrier import cancellation_safe_field
from wd_spectra._mass_feautrier import mass_field, mass_energy, mass_energy_response


@pytest.mark.parametrize('chunk',[0,-1,True,1.5])
def test_invalid_chunks_are_rejected(chunk):
    with pytest.raises(ValueError,match='chunk_size'):
        mass_field(np.array([[.1,1.]]),np.ones((1,2)),1.,0.,column_mass=np.array([.1,1.]),
            wavelength_chunk_size=chunk)


@pytest.mark.parametrize('mass',[[0.,1.],[1.,1.],[2.,1.],[1.,np.nan]])
def test_invalid_mass_grid_is_rejected(mass):
    with pytest.raises(ValueError,match='column mass'):
        mass_field(np.array([[.1,1.]]),np.ones((1,2)),1.,0.,column_mass=np.array(mass))


@pytest.mark.parametrize('epsilon', [1., .1, 1e-6])
def test_constant_extinction_reproduces_optical_depth_discretization(epsilon):
    mass=np.geomspace(1e-12,100.,40)
    b=np.broadcast_to(1.+mass*.2,(3,len(mass))).copy()
    ext=np.broadcast_to(np.array([.2,1.,5.])[:,None],b.shape)
    tau=optical_depth_from_mass_opacity(mass,ext)
    old_source,old=cancellation_safe_field(tau,b,epsilon*ext,(1-epsilon)*ext)
    source,new=mass_field(tau,b,epsilon*ext,(1-epsilon)*ext,column_mass=mass)
    np.testing.assert_allclose(source,old_source,rtol=3e-13)
    np.testing.assert_allclose(new.interface_flux,old.interface_flux,rtol=3e-12,atol=1e-12)


@pytest.mark.parametrize('epsilon', [1., .1, 1e-6])
def test_variable_extinction_physical_energy_is_flux_divergence(epsilon):
    mass=np.geomspace(1e-8,100.,30)
    wave=np.array([1000.,2500.,9000.])
    ext=(.2+np.exp(2*np.sin(np.arange(30))))[None,:]*np.array([.2,1.,5.])[:,None]
    b=np.broadcast_to(1+mass*.1,ext.shape)
    tau=optical_depth_from_mass_opacity(mass,ext)
    _,field=mass_field(tau,b,epsilon*ext,(1-epsilon)*ext,column_mass=mass)
    energy,_=mass_energy(wave,mass,b,field.mean_intensity,epsilon*ext)
    divergence=np.diff(trapezoid(field.interface_flux,wave,axis=0))
    np.testing.assert_allclose(energy,divergence,rtol=2e-8,atol=2e-8)


@pytest.mark.parametrize('epsilon', [1., .1, 1e-6])
def test_variable_extinction_complete_material_response(epsilon):
    mass=np.geomspace(1e-10,100.,24);wave=np.array([1000.,4000.,20000.])
    ext=(.5+np.exp(np.sin(np.arange(24))))[None,:]*np.array([.3,1.,3.])[:,None]
    base_b=np.broadcast_to(1+mass*.2,ext.shape)
    def evaluate(x):
        b=base_b*np.exp(4*x);a=epsilon*ext*np.exp(1.3*x);s=(1-epsilon)*ext*np.exp(-.2*x)
        tau=optical_depth_from_mass_opacity(mass,a+s)
        source,field=mass_field(tau,b,a,s,column_mass=mass)
        energy,cooling=mass_energy(wave,mass,b,field.mean_intensity,a)
        return (trapezoid(field.interface_flux,wave,axis=0),energy,cooling),(tau,b,a,s,source,field)
    _,(tau,b,a,s,source,field)=evaluate(np.zeros(24))
    fj,ej,cj=mass_energy_response(wave,tau,mass,b,field.mean_intensity,source,a,s,4*b,1.3*a,-.2*s)
    # Physical local cooling must not acquire neighbouring-opacity derivatives.
    offdiag=cj.copy();offdiag[np.arange(23),np.arange(23)]=0
    assert np.count_nonzero(offdiag)==0
    np.testing.assert_allclose(ej,np.diff(fj,axis=0),rtol=3e-7,atol=2e-7)
    for i in (0,7,16,23):
        # Near conservative scattering a 1e-5 perturbation of a thin node
        # subtracts almost identical global fields. A measured step-size
        # study resolves the tangent at 1e-3 (O(h^2) truncation < 5e-6).
        h=1e-3
        dx=np.eye(24)[i]*h
        plus,_=evaluate(dx);minus,_=evaluate(-dx)
        for analytic,p,m in zip((fj,ej,cj),plus,minus):
            fd=(p-m)/(2*h)
            np.testing.assert_allclose(analytic[:,i],fd,rtol=3e-5,atol=2e-5)


def test_isothermal_semi_infinite_solution_converges_with_mass_resolution():
    # Analytic emergent flux is pi B, independent of smooth opacity changes.
    errors=[]
    for n in (40,80,160):
        mass=np.geomspace(1e-8,100.,n)
        absorption=(1+9*mass/(1+mass))[None,:]
        tau=optical_depth_from_mass_opacity(mass,absorption)
        _,field=mass_field(tau,np.ones((1,n)),absorption,0.,column_mass=mass,n_angle=8)
        errors.append(abs(field.interface_flux[0,0]/np.pi-1))
    assert errors[1]<errors[0]/3
    assert errors[2]<errors[1]/3


def test_variable_mass_volume_against_high_precision_matrix():
    mp=pytest.importorskip('mpmath')
    mass=np.geomspace(1e-10,30.,20)
    ext=(1+9*mass/(1+mass))[None,:];b=(1+.2*mass)[None,:]
    a=.01*ext;s=.99*ext;tau=optical_depth_from_mass_opacity(mass,ext)
    _,field=mass_field(tau,b,a,s,column_mass=mass,n_angle=1)
    with mp.workdps(70):
        m=[mp.mpf(float(v)) for v in mass];t=[mp.mpf(float(v)) for v in tau[0]]
        h=[t[0]]+[t[i]-t[i-1] for i in range(1,len(t))]
        mh=[m[0]]+[m[i]-m[i-1] for i in range(1,len(m))]
        n=len(m);matrix=mp.matrix(n+1);rhs=mp.matrix(n+1,1);mu=mp.mpf('.5')
        matrix[0,0]=1+mu/h[0];matrix[0,1]=-mu/h[0]
        for i in range(n-1):
            volume=mp.mpf(float(ext[0,i]))*.5*(mh[i]+mh[i+1])
            left=mu**2/(h[i]*volume);right=mu**2/(h[i+1]*volume)
            matrix[i+1,i]=-left;matrix[i+1,i+1]=left+right+mp.mpf('.01');matrix[i+1,i+2]=-right
            rhs[i+1]=mp.mpf('.01')*mp.mpf(float(b[0,i]))
        matrix[n,n]=1;rhs[n]=float(b[0,-1])
        u=mp.lu_solve(matrix,rhs)
        mean=np.array([float(u[i+1]) for i in range(n)])
        flux=np.array([float(4*mp.pi*mu**2*(u[i+1]-u[i])/h[i]) for i in range(n)])
    np.testing.assert_allclose(field.mean_intensity[0],mean,rtol=3e-13)
    np.testing.assert_allclose(field.interface_flux[0],flux,rtol=3e-12,atol=1e-12)
