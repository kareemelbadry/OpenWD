"""Spectrum-only regression. A saved atmosphere NEVER seeds the cold canary."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from wd_spectra import gray_helium_atmosphere
from wd_spectra.models.common import ModelData
from wd_spectra.models.dq import DQConfig
from wd_spectra._dq.runtime import make_material, numerical_policy
from wd_spectra._dq.provenance import digest


@pytest.mark.spectral
def test_completed_default_retains_j1311_absolute_flux(tmp_path):
    """Full optical grid from the pre-promotion, independently checked model."""
    root = Path(__file__).parent/'data/dq_regressions'
    manifest = json.loads((root/'completed-manifest.json').read_text())
    fixture = root/'j1311-completed-fixed.npz'
    assert digest(fixture) == manifest['fixture_sha256']
    config = DQConfig(**manifest['parameters'])
    with np.load(fixture, allow_pickle=False) as z:
        seed = gray_helium_atmosphere(config.effective_temperature, config.logg,
                                     n_depth=len(z['temperature']))
        state = replace(seed, **{k: z[k].copy() for k in
            ('temperature', 'gas_pressure', 'column_mass', 'rosseland_optical_depth')})
        wave, expected = z['wavelength'].copy(), z['flux'].copy()
    with numerical_policy(tmp_path):
        material = make_material(config, ModelData.default(), tmp_path)
        state = material.chemistry(state)[0]
        actual = material.spectrum(state, wave, manifest['n_angle'])
    assert len(wave) == 30000
    # Cross-platform tolerance; no normalization or parameter fitting.
    np.testing.assert_allclose(actual.surface_flux_lambda, expected, rtol=3e-6, atol=0.)
