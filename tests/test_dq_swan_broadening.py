"""He impact broadening of resolved Swan lines (empirical opt-in, off by default)."""
import numpy as np
import pytest

from wd_spectra._compat import trapezoid

from wd_spectra.carbon_molecular import read_c2_cross_section_table
from wd_spectra.models.dq import DQConfig, validate_config
from wd_spectra._dq.data import data_root
from wd_spectra._dq.swan_strength_cache import CachedSwan
from wd_spectra._dq.dq_swan_impact_broadening import (
    ImpactBroadenedSwan, impact_half_width, weisskopf_detuning)


@pytest.fixture(scope='module')
def swans():
    table = read_c2_cross_section_table(data_root()/'c2-8states-r15000.npz')
    kwargs = dict(branches=data_root()/'swan-completed.npz', cache_grids=8, strength_cache_bytes=64*1024**2)
    return CachedSwan(table, **kwargs), ImpactBroadenedSwan(table, **kwargs)


def _smooth(w, f, fwhm=10.):
    s = fwhm/2.3548
    k = np.exp(-.5*((w[:, None]-w[None, :])/s)**2)
    return (k @ f)/k.sum(axis=1)


def test_zero_density_reproduces_unbroadened_swan(swans):
    plain, broad = swans
    w = np.linspace(4600., 5250., 2601)
    a = plain.cell_average(w, 5500., -3., 2.)
    b = broad.broadened_cell_average(w, 5500., -3., 2., 0.0)
    # Same lines and strengths; only the deposit/FFT quadrature differs.
    assert np.max(abs(_smooth(w, b) - _smooth(w, a)))/np.max(_smooth(w, a)) < 1e-2


def test_dense_helium_flattens_cores_and_removes_truncated_strength(swans):
    plain, broad = swans
    w = np.linspace(4400., 5300., 3601)
    n = 1e22
    a = plain.cell_average(w, 6000., -20., -10.)
    b = broad.broadened_cell_average(w, 6000., -20., -10., n)
    assert np.max(b) < np.max(a)
    # The Weisskopf truncation omits the Lorentz area beyond |dnu| = v/rho_W:
    # the kept fraction is (2/pi) atan(W/gamma) (documented, not conserved).
    gamma, limit = impact_half_width(6000., n), weisskopf_detuning(6000., n)
    assert 0 < gamma and 0 < limit and np.isfinite(limit)
    ratio = trapezoid(b, w)/trapezoid(a, w)
    kept = 2/np.pi*np.arctan(limit/gamma)
    assert ratio <= 1.0 + 1e-3
    assert abs(ratio - kept) < 0.1


def test_broadening_is_opt_in_and_validated():
    config = DQConfig()
    assert config.swan_helium_broadening == 'none'
    validate_config(DQConfig(swan_helium_broadening='exomol-weisskopf'))
    with pytest.raises(ValueError, match='swan_helium_broadening'):
        validate_config(DQConfig(swan_helium_broadening='lorentz'))
