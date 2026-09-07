from dataclasses import replace
import numpy as np
import pytest
from wd_spectra.dense_eos import HeliumREOS3Table
from wd_spectra.constants import BOLTZMANN,HELIUM_MASS
from smooth_reos3_experiment import SmoothREOS3
from dense_helium_materials import DenseExactMaterials
from convective_consistency_experiment import MaterialCoefficients
from test_discrete_dense_transport_seed import Column


def test_exact_dense_materials_include_finite_eos_and_interface_derivatives():
    # Analytic repulsive pressure makes a genuinely nonideal test table.
    temps=np.array([1000.,3000.,6000.,10000.,20000.,30000.])
    density=np.geomspace(1e-10,100.,100)
    gas=BOLTZMANN/HELIUM_MASS
    p=tuple(density*gas*t+1e12*density**2 for t in temps)
    u=tuple(1.5*gas*t/1e10+100*density for t in temps)
    table=HeliumREOS3Table(temps,tuple(density for _ in temps),p,u)
    bulk=SmoothREOS3(table,3000.,17000.)
    pressure=np.geomspace(1e8,1e12,8)
    t=np.linspace(4200.,5300.,8)
    def at(temp):
        rho=bulk.fields(pressure,temp)[0]
        return Column(temp,pressure,np.geomspace(1e-8,100.,8),rho)
    exact=MaterialCoefficients(at,lambda a:bulk.thermodynamics(a.temperature,a.gas_pressure),
        lambda a:.01*(a.temperature/5000.)**2*a.mass_density**.3,1.)
    model=DenseExactMaterials(exact,np.log(t),.1,bulk=bulk)
    dt=.02*np.sin(np.arange(8))
    values,j=model(dt)
    # Accepted material evaluation uses the same EOS. Only opacity is a
    # tangent model; density, Cp, Q and adiabatic gradient are exact here.
    fields,_=model.fields(dt)
    actual=exact.fields(np.log(t)+dt)[1]
    np.testing.assert_allclose(fields[[0,2,3,4]],actual[[0,2,3,4]],rtol=1e-13)
    h=1e-6
    for k in range(3):
        fd=np.column_stack([(model(dt+h*d)[0][k]-model(dt-h*d)[0][k])/(2*h)
                            for d in np.eye(8)])
        scale=np.maximum(np.max(abs(fd),axis=1),1e-100)
        np.testing.assert_allclose(j[k]/scale[:,None],fd/scale[:,None],rtol=1e-7,atol=2e-8)
    with pytest.raises(ValueError,match='trust region'):
        model(np.full(8,.2))

    # Regression: a finite material step straddles a PCHIP knot while the
    # actual Newton tangent must use the local one-sided polynomial.
    near=np.full(8,np.log(6000.)+2e-5)
    knot_model=DenseExactMaterials(exact,near,.04,bulk=bulk)
    values,tangent=knot_model(np.zeros(8))
    analytic=tangent[0]@np.ones(8)
    measured=(exact(near+1e-6)[0]-exact(near-1e-6)[0])/(2e-6)
    straddling=(exact(near+2e-4)[0]-exact(near-2e-4)[0])/(4e-4)
    np.testing.assert_allclose(analytic,measured,rtol=2e-6,atol=1e-8)
    assert np.max(abs(straddling-measured))>1e-3*np.max(abs(measured))
