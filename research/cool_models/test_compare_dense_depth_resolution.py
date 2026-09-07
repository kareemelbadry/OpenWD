import numpy as np
from wd_spectra._compat import trapezoid
from compare_dense_depth_resolution import compare_spectra


def test_identical_integrals_do_not_hide_shape_error():
    wave=np.linspace(1000,60000,1000)
    coarse=np.ones(len(wave))
    fine=1+.05*np.sin(2*np.pi*(wave-wave[0])/(wave[-1]-wave[0]))
    result=compare_spectra(wave,coarse,wave,fine,trapezoid(coarse,wave))
    assert abs(result['coarse_flux_ratio']-result['fine_flux_ratio'])<1e-14
    assert not result['independent_depth_resolution_verified']
    assert result['integrated_absolute_spectral_change_over_target']>.03


def test_equal_spectra_pass():
    wave=np.linspace(1000,60000,1000);flux=np.ones(len(wave))
    result=compare_spectra(wave,flux,wave,flux,trapezoid(flux,wave))
    assert result['independent_depth_resolution_verified']
