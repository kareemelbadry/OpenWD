"""DQ He- free-free below John's 0.5063-micron edge (partial-wave by default, as in DZ)."""
import numpy as np
import pytest
from wd_spectra import gray_helium_atmosphere
from wd_spectra.models.dq import DQConfig, validate_config
from wd_spectra._dq.continuum_batches import helium_continuum_absorption
from wd_spectra._dq.runtime import make_material


def test_partial_wave_is_the_default_and_validated():
    assert DQConfig().helium_minus_free_free == 'partial-wave'
    validate_config(DQConfig(helium_minus_free_free='automatic'))
    with pytest.raises(ValueError, match='helium_minus_free_free'):
        validate_config(DQConfig(helium_minus_free_free='john1968'))


def test_material_carries_the_selection(tmp_path):
    for choice in ('partial-wave', 'automatic'):
        material = make_material(DQConfig(6000., 8., -6., helium_minus_free_free=choice))
        assert material.config.helium_minus_free_free == choice


def test_partial_wave_removes_the_edge_step_and_lowers_blue_opacity():
    a = gray_helium_atmosphere(6000., 8., n_depth=12)
    wave = np.array([4500., 5062.9, 5063.1, 6000.])
    automatic = helium_continuum_absorption(a, wave)
    partial = helium_continuum_absorption(a, wave, prescription='partial-wave')
    # Identical on John's (1994) table; below its edge lower and continuous
    # inside his temperature range (hotter deep layers keep the Carbon fit).
    np.testing.assert_array_equal(partial[2:], automatic[2:])
    cool = a.temperature <= 10080.
    assert np.all(partial[0, cool] < automatic[0, cool])
    np.testing.assert_array_equal(partial[0, ~cool], automatic[0, ~cool])
    edge = np.abs(partial[1, cool]/partial[2, cool]-1)
    assert np.max(edge) < 0.01 < np.max(np.abs(automatic[1, cool]/automatic[2, cool]-1))
