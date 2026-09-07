from types import SimpleNamespace

import numpy as np
import pytest

from wd_spectra import helium, mixture, opacity


@pytest.mark.parametrize("module,mean_name,continuum_name", [
    (opacity, "rosseland_mean_hydrogen_continuum_opacity",
     "hydrogen_continuum_mass_absorption_coefficient"),
    (helium, "rosseland_mean_helium_continuum_opacity",
     "helium_continuum_mass_absorption_coefficient"),
    (mixture, "rosseland_mean_hydrogen_helium_continuum_opacity",
     "hydrogen_helium_continuum_mass_absorption_coefficient"),
])
@pytest.mark.parametrize("has_edge", [False, True])
def test_fixed_quadrature_preserves_continuum_provider_and_smoothness(
    monkeypatch, module, mean_name, continuum_name, has_edge
):
    wave = np.geomspace(500.0, 100000.0, 300)
    calls = []

    def continuum(atmosphere, wavelength, **kwargs):
        np.testing.assert_array_equal(wavelength, wave)
        calls.append(atmosphere.temperature.copy())
        values = np.where(wavelength < 3000.0, 50.0, 5.0) if has_edge else (
            np.full(wavelength.size, 5.0)
        )
        return values[:, None]

    monkeypatch.setattr(module, continuum_name, continuum)
    mean = getattr(module, mean_name)

    def evaluate(log_change):
        atmosphere = SimpleNamespace(
            temperature=np.array([12000.0*np.exp(log_change)]), n_depth=1,
            hydrogen_lte_state=object(), helium_lte_state=object(),
        )
        return mean(atmosphere, wavelength_angstrom=wave)[0]

    base = evaluate(0.0)
    small = (evaluate(1e-6) - evaluate(-1e-6)) / 2e-6
    large = (evaluate(1e-5) - evaluate(-1e-5)) / 2e-5
    assert len(calls) == 5
    np.testing.assert_allclose(small, large, rtol=1e-7, atol=1e-8)
    if not has_edge:
        assert base == pytest.approx(5.0)
