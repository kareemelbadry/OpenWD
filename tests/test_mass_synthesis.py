"""Explicit conservative synthesis does not alter the established default."""
import numpy as np
import pytest
from wd_spectra import gray_helium_atmosphere
from wd_spectra.spectrum import synthesize_helium_spectrum,planck_lambda_angstrom


def test_explicit_mass_synthesis_and_default_isolation(monkeypatch):
    from wd_spectra import _mass_feautrier as mass
    a=gray_helium_atmosphere(8000.,8.,n_depth=12)
    wave=np.geomspace(1000.,1e5,60)
    options=dict(stark_table=None,include_lines=False,include_uv_resonance_lines=False,
                 include_helium_ii_lines=False,n_angle=4)
    before=synthesize_helium_spectrum(a,wave,**options)
    seen=[];original=mass.mass_field
    def record(tau,source,absorption,scattering,**kwargs):
        result=original(tau,source,absorption,scattering,**kwargs)
        seen.append((source.copy(),scattering.copy(),kwargs,result))
        return result
    monkeypatch.setattr(mass,'mass_field',record)
    changed=synthesize_helium_spectrum(a,wave,transfer_discretization='column-mass',**options)
    assert len(seen)==2
    np.testing.assert_array_equal(seen[0][0],planck_lambda_angstrom(wave[:,None],a.temperature[None,:]))
    np.testing.assert_array_equal(seen[1][0],seen[0][3][0])
    assert not np.any(seen[1][1])
    np.testing.assert_array_equal(changed.surface_flux_lambda,seen[0][3][1].interface_flux[:,0])
    assert changed.metadata['independent_radiation_scaled_source_error']<1e-10
    after=synthesize_helium_spectrum(a,wave,**options)
    assert len(seen)==2
    np.testing.assert_array_equal(before.surface_flux_lambda,after.surface_flux_lambda)
    assert before.metadata==after.metadata


def test_invalid_synthesis_discretization_fails_before_opacity():
    with pytest.raises(ValueError,match='transfer_discretization'):
        synthesize_helium_spectrum(None,[4000.,5000.],stark_table=None,
                                  transfer_discretization='guess')


def test_failed_independent_mass_source_closure_is_not_reported_success(monkeypatch):
    from wd_spectra import _mass_feautrier as mass
    from dataclasses import replace
    a=gray_helium_atmosphere(8000.,8.,n_depth=8)
    original=mass.mass_field;calls=0
    def wrong(*args,**kwargs):
        nonlocal calls
        calls+=1
        source,field=original(*args,**kwargs)
        if calls==2:field=replace(field,mean_intensity=field.mean_intensity*2)
        return source,field
    monkeypatch.setattr(mass,'mass_field',wrong)
    with pytest.raises(RuntimeError,match='independent source closure'):
        synthesize_helium_spectrum(a,np.geomspace(1000.,1e5,30),stark_table=None,
            include_lines=False,include_uv_resonance_lines=False,include_helium_ii_lines=False,
            transfer_discretization='column-mass')
