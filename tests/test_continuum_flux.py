"""Line-free continuum flux returned beside every run_model spectrum."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from wd_spectra import (
    Spectrum,
    StokesSpectrum,
    gray_helium_atmosphere,
    gray_hydrogen_atmosphere,
    synthesize_balmer_spectrum,
    synthesize_helium_spectrum,
    synthesize_hydrogen_spectrum,
)
from wd_spectra.models import ModelData
from wd_spectra.models.automatic import ModelRun
from wd_spectra.models.common import ModelResult, save_model_result


def test_spectrum_continuum_is_optional_and_stokes_keeps_its_positional_order():
    wave = np.array([4000.0, 5000.0])
    assert Spectrum(wave, wave, {}).continuum_flux_lambda is None
    stokes = StokesSpectrum(wave, wave, {}, wave + 1, wave + 2, wave + 3)
    assert stokes.continuum_flux_lambda is None
    np.testing.assert_array_equal(stokes.surface_flux_v_lambda, wave + 3)
    stokes = StokesSpectrum(wave, wave, {}, wave + 1, wave + 2, wave + 3, wave + 4)
    np.testing.assert_array_equal(stokes.continuum_flux_lambda, wave + 4)
    copied = replace(stokes, metadata={"a": 1})
    np.testing.assert_array_equal(copied.surface_flux_q_lambda, wave + 1)
    np.testing.assert_array_equal(copied.continuum_flux_lambda, wave + 4)


@pytest.mark.parametrize(
    "options",
    [{}, {"transfer_discretization": "formal-pchip"},
     {"transfer_depth_refinement": 3}, {"emergent_ray_mu": 0.5}],
)
def test_balmer_continuum_is_the_same_transfer_without_line_opacity(options):
    atmosphere = gray_hydrogen_atmosphere(12_000.0, 8.0, n_depth=24)
    wave = np.linspace(4700.0, 5000.0, 61)
    spectrum = synthesize_balmer_spectrum(atmosphere, wave, **options)
    lineless = synthesize_balmer_spectrum(
        atmosphere, wave,
        precomputed_balmer_opacity=np.zeros((wave.size, atmosphere.n_depth)),
        **options,
    )
    np.testing.assert_allclose(
        spectrum.continuum_flux_lambda, lineless.surface_flux_lambda, rtol=1e-12
    )
    assert np.all(spectrum.surface_flux_lambda <= spectrum.continuum_flux_lambda)
    assert "lines" in spectrum.metadata["continuum_flux"]


def test_hydrogen_continuum_omits_lyman_and_keeps_pseudocontinuum():
    atmosphere = gray_hydrogen_atmosphere(15_000.0, 8.0, n_depth=24)
    wave = np.geomspace(1000.0, 9000.0, 300)
    plain = synthesize_hydrogen_spectrum(atmosphere, wave)
    balmer_only = synthesize_balmer_spectrum(atmosphere, wave)
    np.testing.assert_array_equal(
        plain.continuum_flux_lambda, balmer_only.continuum_flux_lambda
    )
    assert np.any(plain.surface_flux_lambda != balmer_only.surface_flux_lambda)
    # Dissolved-level absorption belongs to the continuum (TLUSTY convention).
    pseudo = synthesize_hydrogen_spectrum(
        atmosphere, wave, include_series_pseudocontinuum=True
    )
    pseudo_only = synthesize_balmer_spectrum(
        atmosphere, wave, include_series_pseudocontinuum=True,
        precomputed_balmer_opacity=np.zeros((wave.size, atmosphere.n_depth)),
    )
    np.testing.assert_allclose(
        pseudo.continuum_flux_lambda, pseudo_only.surface_flux_lambda, rtol=1e-12
    )
    edge = (wave > 3650.0) & (wave < 3750.0)
    assert np.all(pseudo.continuum_flux_lambda[edge] < plain.continuum_flux_lambda[edge])
    assert "pseudo-continuum" in pseudo.metadata["continuum_flux"]


@pytest.mark.parametrize("discretization", ["formal-linear", "column-mass"])
def test_helium_continuum_is_independent_of_lines(discretization):
    data = ModelData.default()
    atmosphere = gray_helium_atmosphere(15_000.0, 8.0, n_depth=24)
    wave = np.linspace(3800.0, 7200.0, 400)
    options = dict(stark_table=data.helium_i_stark,
                   transfer_discretization=discretization)
    spectrum = synthesize_helium_spectrum(atmosphere, wave, **options)
    lineless = synthesize_helium_spectrum(atmosphere, wave, include_lines=False, **options)
    np.testing.assert_array_equal(
        spectrum.continuum_flux_lambda, lineless.continuum_flux_lambda
    )
    assert np.all(spectrum.surface_flux_lambda <= spectrum.continuum_flux_lambda * (1 + 1e-12))
    line = np.argmin(abs(wave - 4472.7))
    assert spectrum.surface_flux_lambda[line] < 0.97 * spectrum.continuum_flux_lambda[line]
    # Without lines the flux is the continuum, pseudo-continuum included.
    np.testing.assert_allclose(
        lineless.surface_flux_lambda, spectrum.continuum_flux_lambda, rtol=1e-12
    )


def test_saved_spectrum_has_continuum_column_and_round_trips(tmp_path):
    atmosphere = gray_hydrogen_atmosphere(12_000.0, 8.0, n_depth=16)
    wave = np.linspace(4700.0, 5000.0, 31)
    spectrum = synthesize_balmer_spectrum(atmosphere, wave)
    save_model_result(ModelResult("DA", atmosphere, spectrum, None, {}), tmp_path / "run")
    path = tmp_path / "run" / "spectrum.txt"
    header = path.read_text().splitlines()[0]
    assert header.endswith("continuum_flux_erg_s-1_cm-2_A-1")
    values = np.loadtxt(path)
    assert values.shape == (wave.size, 3)
    selection = SimpleNamespace(workflow="da")
    run = ModelRun(tmp_path / "run", selection, path, False)
    np.testing.assert_allclose(run.spectrum.continuum_flux_lambda,
                               spectrum.continuum_flux_lambda, rtol=1e-15)
    legacy = tmp_path / "legacy.txt"
    np.savetxt(legacy, values[:, :2])
    assert ModelRun(tmp_path, selection, legacy, False).spectrum.continuum_flux_lambda is None


def test_run_metadata_records_the_continuum_definition(tmp_path):
    atmosphere = gray_hydrogen_atmosphere(12_000.0, 8.0, n_depth=16)
    spectrum = synthesize_balmer_spectrum(atmosphere, np.linspace(4700.0, 5000.0, 11))
    save_model_result(ModelResult("DA", atmosphere, spectrum, None, {}), tmp_path / "run")
    saved = json.loads((tmp_path / "run" / "metadata.json").read_text())
    assert saved["spectrum_metadata"]["continuum_flux"] == spectrum.metadata["continuum_flux"]
