"""Public sdB adapter: run_model dispatch, saved spectrum and continuum.

The LTE structure is replaced by a small gray H/He atmosphere and the NLTE atom
by a three-level hydrogen fixture, so these check the adapter contract, not a
converged sdB model.
"""
import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from wd_spectra import SdBConfig, compute_sdb, run_model, select_physics
from wd_spectra.atmosphere import gray_hydrogen_helium_atmosphere
from wd_spectra.models import sdb
from wd_spectra.models.common import AtmosphereConvergenceWarning, ModelResult
from wd_spectra.spectrum import Spectrum
from test_hot_public import public_data  # noqa: F401  (fixture)


@pytest.fixture
def gray_structure(monkeypatch):
    calls = {}

    def structure(config, wavelength, **kwargs):
        calls['config'], calls['kwargs'] = config, kwargs
        atmosphere = gray_hydrogen_helium_atmosphere(
            config.effective_temperature, config.logg, config.log_hydrogen_to_helium, n_depth=8)
        return ModelResult('DAB', atmosphere, Spectrum(wavelength, np.ones_like(wavelength), {}),
                           config, {'atmosphere_convergence_status': 'converged'})

    monkeypatch.setattr(sdb, 'compute_dab', structure)
    return calls


def test_run_model_dispatches_sdb_and_saves_the_continuum(public_data, gray_structure, tmp_path):  # noqa: F811
    config = SdBConfig(effective_temperature=28000., logg=5.6, log_helium_to_hydrogen=-2.0,
                       hydrogen_levels=3, population_maximum_iterations=2)
    assert select_physics(config, data=public_data).workflow == 'sdb'
    assert not select_physics(config, data=public_data).experimental
    wave = np.linspace(4000., 7000., 31)
    with pytest.warns(AtmosphereConvergenceWarning):
        run = run_model(config, tmp_path / 'run', data=public_data, wavelength=wave)
    # The H/He ratio reaches the LTE structure with the DAB sign convention.
    assert gray_structure['config'].log_hydrogen_to_helium == 2.0
    assert gray_structure['kwargs']['initial_atmosphere'] is None
    assert run.selection.workflow == 'sdb' and not run.convergence_verified
    spectrum = run.spectrum
    np.testing.assert_array_equal(spectrum.wavelength_angstrom, wave)
    assert np.all(spectrum.surface_flux_lambda > 0)
    assert np.all(spectrum.continuum_flux_lambda > 0)
    assert np.loadtxt(run.spectrum_path).shape == (wave.size, 3)
    saved = json.loads((run.output_directory / 'metadata.json').read_text())
    assert saved['spectral_type'] == 'sdB'
    assert saved['model_metadata']['metals'] is False
    assert saved['model_metadata']['experimental'] is False
    assert saved['model_metadata']['structure_convergence_status'] == 'converged'
    assert 'pseudo-continuum' in saved['spectrum_metadata']['continuum_flux']
    # The line-free continuum differs from the flux at the Balmer lines.
    halpha = np.argmin(abs(wave - 6564.6))
    assert spectrum.surface_flux_lambda[halpha] != spectrum.continuum_flux_lambda[halpha]


@pytest.mark.parametrize('change', [
    dict(quality='quick'), dict(helium_i_atom='10'), dict(hydrogen_levels=40),
    dict(log_helium_to_hydrogen=float('nan')), dict(effective_temperature=-1.),
])
def test_invalid_sdb_configuration_fails_before_any_model(change):
    with pytest.raises(ValueError):
        compute_sdb(SdBConfig(**change))


def test_sdb_public_models_are_cold_starts():
    with pytest.raises(ValueError, match='cold start'):
        compute_sdb(SdBConfig(), initial_atmosphere=object())


def test_hd4539_control_retains_the_independent_driver_provenance():
    directory = Path(__file__).parent / 'data/sdb_regressions'
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert hashlib.sha256((directory / 'hd4539-hhe.npz').read_bytes()).hexdigest() == manifest['control_sha256']
    assert hashlib.sha256((directory / 'hd4539-fixed-state.npz').read_bytes()).hexdigest() == manifest['fixed_state_sha256']
    assert manifest['lte_convergence_status'] == 'converged'
    assert manifest['nlte_populations_converged'] is True
    assert manifest['nlte_population_change'] < 1e-4
    assert manifest['absolute_flux_rtol'] == 1e-3
    assert manifest['hydrogen_levels'] == 16 and manifest['helium_i_atom'] == '14'
    assert manifest['helium_conservation_row'] == 'dominant'
    assert manifest['accelerated_lambda'] is True
