from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from test_discrete_dense_transport_seed import Column
from wd_spectra.constants import STEFAN_BOLTZMANN
import exact_convective_trial_projection as module


@pytest.mark.parametrize('coefficient_scale',[1e-7,1e8])
def test_projection_checks_actual_flux_and_retains_inefficient_trials(monkeypatch,coefficient_scale):
    target=STEFAN_BOLTZMANN*5000**4
    pressure=np.geomspace(1e8,1e9,20)
    seed=Column(4000*(pressure/pressure[0])**.4,pressure,np.ones(20),pressure/1e12)
    def at(teff,p,t,tau):return Column(t,p,tau,p/1e12,effective_temperature=teff)
    class Material:
        def __init__(self,at,*unused):self.at=at
        def fields(self,lt):return self.at(np.exp(lt)),None
        def assemble(self,a,f):
            return np.full(a.n_depth,.2),np.full(a.n_depth,1e-3),target*coefficient_scale*(a.temperature/5000)**2
    monkeypatch.setattr(module,'MaterialCoefficients',Material)
    result,info=module.project_trial(seed,SimpleNamespace(atmosphere_at=at),
        dict(thermodynamics=None,rosseland_opacity=None,mixing_length_alpha=1.,
             with_temperature=lambda t:replace(seed,temperature=t)))
    assert result.temperature[0]==seed.temperature[0]
    if coefficient_scale<1:
        assert result is seed
        assert info['corrected_interfaces']==0
    else:
        assert info['corrected_interfaces']==19
        assert info['maximum_actual_convective_flux_ratio']<=1+1e-6
        assert np.all(result.temperature<=seed.temperature*(1+1e-13))
