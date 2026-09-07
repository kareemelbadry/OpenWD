import numpy as np
from wd_spectra.stark import default_lyman_stark_table,StarkLine
from physical_detuning_stark_experiment import physical_detuning_line,physical_detuning_lyman


def test_detuning_interpolation_reproduces_density_temperature_table_nodes():
    line=default_lyman_stark_table()[(1,2)];converted=physical_detuning_line(line)
    center=1215.6713;w=center+np.r_[0.,np.geomspace(1e-5,100,1000)]
    for logne in line.log_electron_density[::4]:
        for logt in line.log_temperature[::2]:
            np.testing.assert_allclose(converted.wavelength_profile(w,center,10.**logt,10.**logne),
                line.wavelength_profile(w,center,10.**logt,10.**logne),rtol=2e-12,atol=1e-20)


def test_fixed_physical_profile_does_not_acquire_density_dependence():
    # A physical power law represented at different field scales must stay
    # invariant under interpolation, even in the core-to-wing transition.
    # Construct the same piecewise-log profile on a common physical grid.
    ne=np.array([10.,13.]);temperature=np.array([3.,4.])
    a=np.linspace(-8,8,161);f=np.log10(1.25e-9)+(2/3)*ne
    p=np.array([[np.maximum(-3*(a+field),-4.)+field for _ in temperature] for field in f])
    line=StarkLine(1,2,a,ne,temperature,p,np.zeros((2,2),int))
    converted=physical_detuning_line(line)
    w=1000.+np.geomspace(.01,10,30)
    low=converted.wavelength_profile(w,1000.,5000,1e10)
    high=converted.wavelength_profile(w,1000.,5000,1e13)
    middle=converted.wavelength_profile(w,1000.,5000,1e11)
    np.testing.assert_allclose(low,high,rtol=2e-13,atol=1e-15)
    np.testing.assert_allclose(middle,low,rtol=2e-13,atol=1e-15)
    # Log interpolation is exactly the weighted physical corner profiles.
    np.testing.assert_allclose(np.log(middle),(2*np.log(low)+np.log(high))/3,atol=2e-14)


def test_lyman_scope_restores_original_profile():
    line=default_lyman_stark_table()[(1,2)];w=np.array([1215.6213,1215.6713,1215.7213])
    before=line.wavelength_profile(w,1215.6713,4000,1e4)
    with physical_detuning_lyman():
        after=line.wavelength_profile(w,1215.6713,4000,1e4)
        edge=line.wavelength_profile(w,1215.6713,4000,1e10)
    np.testing.assert_array_equal(after,edge)
    np.testing.assert_array_equal(line.wavelength_profile(w,1215.6713,4000,1e4),before)
    assert not np.array_equal(after,before)
