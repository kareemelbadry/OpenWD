import numpy as np
import pytest
from wd_spectra import helium
from heminus_join_experiment import (coefficient_with_smooth_join,heminus_join_scope,
    active_policy,HARD_JOIN,SMOOTH_JOIN,validate_continuation_policy)


def test_join_preserves_explicit_sources_and_unchanged_domains():
    original=helium.helium_minus_free_free_coefficient
    wave=np.geomspace(.01,100.,101)[:,None];temp=np.array([1000,1400,3000,5000,8000,10080,16000])[None,:]
    for mode in ('john1968','john1994'):
        np.testing.assert_array_equal(coefficient_with_smooth_join(original,wave,temp,prescription=mode),
                                      original(wave,temp,prescription=mode))
    expected=original(wave,temp)
    actual=coefficient_with_smooth_join(original,wave,temp)
    unchanged=(wave<.5063)|(wave>=1)|(temp<1400)|(temp>10080)
    np.testing.assert_array_equal(actual[unchanged],expected[unchanged])


def test_join_is_bounded_positive_and_monotone_in_wavelength():
    original=helium.helium_minus_free_free_coefficient
    wave=np.geomspace(.5063,1.,1001)[:,None];temp=np.geomspace(1400,10080,101)[None,:]
    actual=coefficient_with_smooth_join(original,wave,temp)
    old=original(wave,temp,prescription='john1968');new=original(wave,temp,prescription='john1994')
    assert np.all(actual>=np.minimum(old,new)*(1-2e-15))
    assert np.all(actual<=np.maximum(old,new)*(1+2e-15))
    assert np.all(actual>0) and np.all(np.diff(actual,axis=0)>0)


@pytest.mark.parametrize('edge',[.5063,1.])
def test_value_and_first_derivative_match_at_join_boundaries(edge):
    original=helium.helium_minus_free_free_coefficient
    temp=np.array([1400.,3000.,5000.,8000.,10080.]);h=1e-7
    values=coefficient_with_smooth_join(original,edge*np.exp(np.array([-h,0.,h]))[:,None],temp)
    left=(values[1]-values[0])/h;right=(values[2]-values[1])/h
    np.testing.assert_allclose(left,right,rtol=3e-6,atol=1e-34)
    assert np.max(abs(values[2]/values[0]-1))<1e-6


def test_scope_is_explicit_and_restored_after_exception():
    original=helium.helium_minus_free_free_coefficient
    with pytest.raises(RuntimeError):
        with heminus_join_scope():
            assert active_policy()==SMOOTH_JOIN
            assert helium.helium_minus_free_free_coefficient(.5063,5000)==original(.5063,5000,prescription='john1968')
            raise RuntimeError('probe')
    assert active_policy()==HARD_JOIN and helium.helium_minus_free_free_coefficient is original


def test_resume_cannot_silently_change_back_to_the_old_opacity():
    assert validate_continuation_policy(HARD_JOIN,SMOOTH_JOIN)
    assert not validate_continuation_policy(SMOOTH_JOIN,SMOOTH_JOIN)
    with pytest.raises(ValueError,match='no silent opacity reversion'):
        validate_continuation_policy(SMOOTH_JOIN,HARD_JOIN)
