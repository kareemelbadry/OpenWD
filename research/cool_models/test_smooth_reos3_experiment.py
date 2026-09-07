import numpy as np
import pytest
from wd_spectra.dense_eos import HeliumREOS3Table
from wd_spectra.constants import BOLTZMANN, HELIUM_MASS
from smooth_reos3_experiment import SmoothREOS3


def test_dilute_analytic_eos_and_thermal_derivatives():
    temperatures=np.array([1000.,3000.,6000.,10000.,20000.,30000.])
    rho=np.geomspace(1e-10,100,60)
    p=tuple(rho*BOLTZMANN*t/HELIUM_MASS for t in temperatures)
    u=tuple(np.full_like(rho,1.5*BOLTZMANN*t/HELIUM_MASS/1e10) for t in temperatures)
    table=HeliumREOS3Table(temperatures,tuple(rho for t in temperatures),p,u)
    model=SmoothREOS3(table,3000.,17000.)
    t=np.array([3000.,5000.,6000.,7000.,10000.,15000.])
    pressure=np.geomspace(1e5,1e12,len(t))
    density,energy,inside=model.evaluate(pressure,t)
    np.testing.assert_allclose(density,pressure*HELIUM_MASS/(BOLTZMANN*t),rtol=1e-13)
    np.testing.assert_allclose(energy,1.5*BOLTZMANN*t/HELIUM_MASS,rtol=1e-13)
    a=model.thermodynamics(t,pressure)
    np.testing.assert_allclose(a.specific_heat_constant_pressure,2.5*BOLTZMANN/HELIUM_MASS,rtol=1e-13)
    np.testing.assert_allclose(a.density_temperature_derivative,1.,rtol=1e-13)
    np.testing.assert_allclose(a.adiabatic_temperature_gradient,.4,rtol=1e-13)
    with pytest.raises(ValueError,match='common-support'):
        model.evaluate(1e-4,5000.)
    fields,slopes=model.fixed_pressure(pressure)(t)
    np.testing.assert_allclose(slopes[0],-density,rtol=1e-13)
    np.testing.assert_allclose(slopes[1],energy,rtol=1e-13)
    np.testing.assert_allclose(slopes[2]/fields[2],0.,atol=1e-13)
    np.testing.assert_allclose(slopes[3],0.,atol=1e-13)
