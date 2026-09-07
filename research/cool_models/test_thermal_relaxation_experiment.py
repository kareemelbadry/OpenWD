import numpy as np
from thermal_relaxation_experiment import bracketed_thermal_update,local_exchange
from wd_spectra._energy_balance import discrete_radiative_cell_energy_balance
from wd_spectra.opacity import optical_depth_from_mass_opacity


def test_exact_base_cell_energy():
    wave=np.geomspace(100,1e5,12);mass=np.geomspace(1e-4,20,7)
    rng=np.random.default_rng(27)
    a=np.exp(rng.normal(size=(12,7)));s=a*.3
    b=np.exp(rng.normal(size=(12,7)));j=b*.94
    tau=optical_depth_from_mass_opacity(mass,a+s)
    exchange,cooling=discrete_radiative_cell_energy_balance(wave,tau,b,j,a/(a+s))
    np.testing.assert_allclose(local_exchange(wave,mass,a+s,j,a,s,b),exchange/cooling,rtol=1e-13)


def test_heating_cooling_roots_and_inactive_cell():
    logt=np.array([5.,5.,5.]);root=np.array([5.4,4.6,4.8])
    result,found,missing=bracketed_thermal_update(logt,lambda x:root-x,
        np.array([True,True,False]),2.,8.)
    np.testing.assert_allclose(result,[5.12,4.88,5.])
    np.testing.assert_array_equal(found,[True,True,False])
    assert not np.any(missing)


def test_no_domain_boundary_substituted_for_root():
    result,found,missing=bracketed_thermal_update(np.array([5.]),lambda x:np.ones(1),
        np.ones(1,bool),4.,5.2)
    np.testing.assert_array_equal(result,[5.])
    assert not found[0] and missing[0]


def test_first_root_in_thermal_direction():
    def exchange(x):return -(x-5.05)*(x-5.3)*(x-5.5)
    result,found,missing=bracketed_thermal_update(np.array([5.]),exchange,
        np.ones(1,bool),4.,6.)
    np.testing.assert_allclose(result,[5.05],atol=1e-8)
    assert found[0] and not missing[0]


def test_saturated_root_does_not_keep_changing_during_refinement():
    history=[]
    def exchange(x):
        history.append(x.copy())
        return np.array([5.63,5.045])-x
    result,found,missing=bracketed_thermal_update(np.array([5.,5.]),exchange,
        np.ones(2,bool),4.,6.)
    np.testing.assert_allclose(result,[5.12,5.045],atol=1e-8)
    np.testing.assert_allclose(np.array(history)[-8:,0],5.12,rtol=0,atol=0)
    assert np.all(found) and not np.any(missing)


def test_depth_local_cache_matches_full_evaluation():
    from dataclasses import replace
    from test_discrete_dense_transport_seed import Column
    from thermal_relaxation_experiment import LocalOpacityCache
    p=np.geomspace(1e6,1e9,8);t=np.linspace(3000,6000,8)
    a=Column(t,p,np.geomspace(.1,10,8),p/1e12)
    def absorption(a):return a.temperature[None,:]**2*np.array([1.,2.,3.])[:,None]/a.gas_pressure
    def scattering(a):return np.ones((3,a.n_depth))*.12
    options=dict(with_temperature=lambda t:replace(a,temperature=t),
        true_absorption=absorption,scattering_opacity=scattering)
    initial=absorption(a)
    cache=LocalOpacityCache(a,options,initial,scattering(a))
    lt=np.log(t);lt[[1,4]]+=.02
    actual,_=cache(lt)
    np.testing.assert_allclose(actual,absorption(options['with_temperature'](np.exp(lt))),rtol=1e-14)
    np.testing.assert_array_equal(initial,absorption(a))
