import numpy as np
import pytest
from wd_spectra import atmosphere
from extended_thermal_wavelength_experiment import extend_grid,extended_quadrature


def test_existing_nodes_are_bitwise_preserved():
    old=np.geomspace(100,1e5,600)
    new=extend_grid(old,1e8)
    np.testing.assert_array_equal(new[:len(old)],old)
    assert new[-1]==1e8
    assert np.max(np.diff(np.log(new[len(old)-1:])))<=np.log(old[-1]/old[-2])*(1+1e-11)


def test_extension_is_scoped_and_restored_after_error():
    original=atmosphere.np
    with pytest.raises(RuntimeError):
        with extended_quadrature(1e8):
            assert atmosphere.np.geomspace(100,1e5,600)[-1]==1e8
            np.testing.assert_array_equal(atmosphere.np.geomspace(1,100,10),np.geomspace(1,100,10))
            assert np.geomspace(100,1e5,600)[-1]==1e5
            raise RuntimeError('probe')
    assert atmosphere.np is original


def test_invalid_extent_rejected():
    with pytest.raises(ValueError):extend_grid(np.geomspace(100,1e5,600),1e5)
