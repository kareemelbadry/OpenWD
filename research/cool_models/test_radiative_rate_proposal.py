from types import SimpleNamespace
import numpy as np
import pytest
from radiative_rate_proposal import radiative_rates
from wd_spectra.constants import STEFAN_BOLTZMANN


def rates_payload(heating_slope,cooling_slope):
    target=STEFAN_BOLTZMANN*5000.**4
    return dict(atmosphere=SimpleNamespace(effective_temperature=5000.),
        radiative_cell_energy_defect=np.array([.5*target]),
        radiative_cell_energy_log_temperature_jacobian=np.array([[target*(1.5*heating_slope-cooling_slope)]]),
        thermal_cell_emission=np.array([target]),
        thermal_cell_emission_log_temperature_jacobian=np.array([[target*cooling_slope]]),
        diagnostic_positive_radiative_rates=True)


def test_common_stiff_opacity_cannot_manufacture_energy_root():
    p=rates_payload(40.,40.)
    for delta in (-.1,-.025,0.,.1):
        exchange,j,thermal,tj=radiative_rates(p,np.array([delta]))
        np.testing.assert_allclose(exchange/thermal,.5,rtol=1e-14)
        np.testing.assert_allclose(j,40*exchange[:,None],rtol=1e-14)
    p['diagnostic_positive_radiative_rates']=False
    # Demonstrates the old linear numerator's spurious zero even though
    # heating/cooling is 1.5 at every temperature in this analytic example.
    assert radiative_rates(p,np.array([-.025]))[0][0] == 0.


def test_positive_rate_zero_and_first_derivative_are_consistent():
    p=rates_payload(40.,44.)
    for delta in (0.,-.02,.04):
        d=np.array([delta]);e,j,c,cj=radiative_rates(p,d)
        h=1e-7
        fd=(radiative_rates(p,d+h)[0]-radiative_rates(p,d-h)[0])/(2*h)
        np.testing.assert_allclose(j[:,0],fd,rtol=1e-8)
    root=np.array([np.log(1.5)/4.])
    np.testing.assert_allclose(radiative_rates(p,root)[0],0.,atol=1e-12)
    e,j,_,_=radiative_rates(p,np.zeros(1))
    target=STEFAN_BOLTZMANN*5000.**4
    np.testing.assert_allclose(e,p['radiative_cell_energy_defect']/target,rtol=1e-14)
    np.testing.assert_allclose(j,p['radiative_cell_energy_log_temperature_jacobian']/target,rtol=1e-14)
