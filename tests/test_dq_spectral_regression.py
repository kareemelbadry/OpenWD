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
def test_refractive_spectrum_retains_j1235_absolute_flux(tmp_path):
    root=Path(__file__).parent/'data/dq_regressions'
    manifest=json.loads((root/'manifest.json').read_text())
    assert digest(root/'j1235-fixed.npz') == manifest['fixture_sha256']
    config=DQConfig(**manifest['parameters'])
    with np.load(root/'j1235-fixed.npz',allow_pickle=False) as z:
        seed=gray_helium_atmosphere(config.effective_temperature,config.logg,n_depth=len(z['temperature']))
        state=replace(seed,**{k:z[k].copy() for k in
            ('temperature','gas_pressure','column_mass','rosseland_optical_depth')})
        wave,expected=z['wavelength'].copy(),z['flux'].copy()
    with numerical_policy(tmp_path):
        material=make_material(config,ModelData.default(),tmp_path)
        actual=material.spectrum(state,wave,3)
    assert actual.metadata['refraction']
    # No normalization, interpolation or observational parameter fitting.
    np.testing.assert_allclose(actual.surface_flux_lambda,expected,rtol=3e-6,atol=0.)
