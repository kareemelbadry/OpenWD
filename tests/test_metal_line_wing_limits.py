"""Optional physical support limits for ordinary metal line wings."""
import numpy as np
import pytest

from wd_spectra import metals
from wd_spectra.constants import BOLTZMANN, LIGHT_SPEED, PLANCK


def _batch(depths=3):
    wave = np.linspace(1500.0, 1800.0, 3001)
    center = np.array([1657.0, 1700.5])
    strength = np.array([2.0e-2, 5.0e-3])
    sigma = np.full((2, depths), 0.02)
    gamma = np.array([[5.0, 20.0, 60.0], [2.0, 8.0, 30.0]])[:, :depths]
    window = np.zeros(2)
    population = np.ones((2, depths))
    return wave, center, strength, sigma, gamma, window, population


def _accumulate(*args, compiled=True, **kwargs):
    wave = args[0]
    out = np.zeros((len(wave), args[3].shape[1]))
    saved = metals._rt
    try:
        if not compiled:
            metals._rt = None
        metals._accumulate_lte_metal_line_profiles(*args, out, **kwargs)
    finally:
        metals._rt = saved
    return out


@pytest.mark.skipif(metals._rt is None, reason="compiled kernel unavailable")
def test_default_call_is_unchanged_and_kernels_agree():
    args = _batch()
    reference = _accumulate(*args)
    explicit = _accumulate(*args, maximum_half_window=None, red_wing_temperature=None)
    np.testing.assert_array_equal(reference, explicit)
    temperature = np.array([4000.0, 6000.0, 9000.0])
    limit = np.full((2, 3), 7.5)
    compiled = _accumulate(*args, maximum_half_window=limit, red_wing_temperature=temperature)
    python = _accumulate(*args, maximum_half_window=limit, red_wing_temperature=temperature,
                         compiled=False)
    np.testing.assert_allclose(compiled, python, rtol=1e-12, atol=0.0)


def test_red_wing_boltzmann_factor_only_on_the_red_side():
    wave, center, strength, sigma, gamma, window, population = _batch()
    one = (wave, center[:1], strength[:1], sigma[:1], gamma[:1], window[:1], population[:1])
    temperature = np.array([4000.0, 6000.0, 9000.0])
    plain = _accumulate(*one)
    bounded = _accumulate(*one, red_wing_temperature=temperature)
    detuning = LIGHT_SPEED / (wave * 1e-8) - LIGHT_SPEED / (center[0] * 1e-8)
    expected = np.where(detuning[:, None] < 0,
                        np.exp(PLANCK * detuning[:, None] / (BOLTZMANN * temperature[None, :])), 1.0)
    inside = plain > 0
    np.testing.assert_allclose(bounded[inside], (plain * expected)[inside], rtol=1e-12)
    blue = wave < center[0]
    np.testing.assert_array_equal(bounded[blue], plain[blue])


def test_support_limit_truncates_but_keeps_doppler_core():
    wave, center, strength, sigma, gamma, window, population = _batch()
    limit = np.full((2, 3), 3.0)
    limited = _accumulate(wave, center, strength, sigma, gamma, window, population,
                          maximum_half_window=limit)
    far = (np.abs(wave - center[0]) > 3.0 + 1e-9) & (np.abs(wave - center[1]) > 3.0 + 1e-9)
    assert np.all(limited[far] == 0.0)
    tiny = np.full((2, 3), 1e-6)
    core = _accumulate(wave, center, strength, sigma, gamma, window, population,
                       maximum_half_window=tiny)
    near = np.abs(wave - center[0]) <= 10 * sigma[0, 0] - 1e-9
    assert np.all(core[near, 0] > 0.0)
