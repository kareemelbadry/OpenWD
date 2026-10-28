"""A fresh public sdB calculation against the separate research-driver recipe.

The control contains only a spectrum; it is loaded after the cold calculation
and never supplies a structure, population or solver starting state.
"""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from wd_spectra import SdBConfig, run_model


CONTROL = Path(__file__).parent / 'data/sdb_regressions/hd4539-hhe.npz'


@pytest.mark.canary
def test_hd4539_public_cold_start_and_research_spectrum(tmp_path):
    config = SdBConfig()
    output = tmp_path / 'hd4539-cold'
    run = run_model(config, output, require_convergence=True)
    assert run.convergence_verified and not run.selection.experimental
    request = json.loads((output / 'model-run.json').read_text())
    assert request['cold_start'] is True and request['initial_checkpoint'] is None
    assert request['request'] == asdict(config)
    saved = json.loads((output / 'metadata.json').read_text())
    metadata = saved['model_metadata']
    assert metadata['cold_start'] is True and metadata['experimental'] is False
    assert metadata['structure_convergence_status'] == 'converged'
    assert metadata['nlte_populations_converged'] is True
    assert metadata['nlte_population_change'] < config.population_tolerance
    # This is the LTE structure certificate; the NLTE step holds that
    # structure fixed and does not claim coupled NLTE radiative equilibrium.
    certificate = saved['atmosphere_metadata']['equilibrium_certificate']
    assert certificate['verified'] is True
    for name, tolerance in dict(all_depth_flux=3e-3, local_energy=3e-3,
                                temperature_stationarity=3e-4, source_closure=1e-6,
                                boundary_screening=3e-3).items():
        check = certificate['checks'][name]
        assert check['tolerance'] == tolerance
        assert check['measured'] is True and check['passed'] is True
        assert check['value'] < tolerance
    spectrum = run.spectrum
    assert spectrum.wavelength_angstrom.size == 124000
    assert np.all(np.isfinite(spectrum.surface_flux_lambda))
    assert np.all(spectrum.surface_flux_lambda > 0)
    assert np.all(np.isfinite(spectrum.continuum_flux_lambda))
    assert np.all(spectrum.continuum_flux_lambda > 0)
    assert np.max(spectrum.surface_flux_lambda / spectrum.continuum_flux_lambda) <= 1.000001
    with np.load(CONTROL, allow_pickle=False) as reference:
        np.testing.assert_array_equal(spectrum.wavelength_angstrom, reference['wavelength'])
        # The existing research recipe and the new public adapter must agree
        # in absolute flux to 0.1%, without normalization or a fitted scale.
        np.testing.assert_allclose(spectrum.surface_flux_lambda, reference['flux'],
                                   rtol=1e-3, atol=0)
