"""Physical source closure and fixed-structure provenance, not iteration counts."""

from dataclasses import replace

import numpy as np
import pytest

from wd_spectra._spectrum_source import solve_spectrum_source
from wd_spectra._stable_feautrier import cancellation_safe_scalar_field
from wd_spectra.atmosphere import gray_helium_atmosphere
from wd_spectra.models import (
    DBConfig,
    ModelData,
    ModelResult,
    compute_db,
    load_atmosphere_checkpoint,
    save_model_result,
    AtmosphereConvergenceWarning,
)
from wd_spectra.models.common import (
    atmosphere_with_model_request_fingerprint,
    model_request_fingerprint,
)
from wd_spectra.spectrum import Spectrum
from wd_spectra._linear_scattering import (
    linear_lambda_operator,
    linear_scattering_source,
)
from wd_spectra.radiative_transfer import radiation_field


def test_linear_lambda_matches_independent_basis_sweeps():
    tau = np.broadcast_to(np.geomspace(1e-8, 100.0, 17), (3, 17)).copy()
    tau *= np.array([0.1, 1.0, 3.0])[:, None]
    lam = linear_lambda_operator(tau, n_angle=4)
    basis = np.broadcast_to(np.eye(17), (3, 17, 17)).reshape(51, 17).copy()
    expected = radiation_field(
        np.repeat(tau, 17, axis=0), basis, n_angle=4
    ).mean_intensity
    np.testing.assert_allclose(
        lam, expected.reshape(3, 17, 17).transpose(0, 2, 1), rtol=2e-13, atol=2e-15
    )


@pytest.mark.parametrize("epsilon", [1.0, 0.1, 1e-6])
def test_linear_source_is_checked_even_without_ca_scattering(epsilon):
    tau = np.broadcast_to(np.geomspace(1e-8, 100.0, 24), (2, 24)).copy()
    b = 1 + tau**0.3
    a = np.full_like(tau, epsilon)
    s = 1 - a
    source, _, metadata = solve_spectrum_source(
        tau,
        b,
        a,
        s,
        wavelength=np.array([4000.0, 5000.0]),
        n_angle=4,
        discretization="formal-linear",
    )
    independent = radiation_field(tau, source, n_angle=4)
    np.testing.assert_allclose(
        source, a * b + s * independent.mean_intensity, rtol=1e-10, atol=1e-12
    )
    assert metadata["source_converged"]
    assert metadata["independent_radiation_scaled_source_error"] < 1e-10


@pytest.mark.parametrize("epsilon", [1.0, 0.1, 1e-6])
def test_direct_source_closes_including_strong_scattering(epsilon):
    wave = np.array([2000.0, 5000.0, 15000.0])
    tau = np.broadcast_to(np.geomspace(1e-6, 100.0, 40), (3, 40)).copy()
    planck = 1.0 + tau**0.25
    a, s = np.full_like(tau, epsilon), np.full_like(tau, 1.0 - epsilon)
    source, field, metadata = solve_spectrum_source(
        tau, planck, a, s, wavelength=wave, n_angle=4
    )
    independent = cancellation_safe_scalar_field(tau, source, n_angle=4)
    np.testing.assert_allclose(
        source, a * planck + s * independent.mean_intensity, rtol=1e-10, atol=1e-12
    )
    assert np.all(field.interface_flux[:, 0] > 0.0)
    assert metadata["source_converged"]
    assert metadata["independent_radiation_scaled_source_error"] < 1e-10


def test_failed_independent_source_closure_cannot_claim_success(monkeypatch):
    import wd_spectra._spectrum_source as module

    original = module.cancellation_safe_field
    calls = []

    def corrupted(*args, **kwargs):
        source, field = original(*args, **kwargs)
        calls.append(True)
        if len(calls) == 2:
            field = replace(field, mean_intensity=field.mean_intensity * 1.1)
        return source, field

    monkeypatch.setattr(module, "cancellation_safe_field", corrupted)
    tau = np.broadcast_to(np.geomspace(0.001, 100.0, 20), (2, 20)).copy()
    with pytest.raises(RuntimeError, match="independent source closure"):
        solve_spectrum_source(
            tau,
            np.ones_like(tau),
            0.1 * np.ones_like(tau),
            0.9 * np.ones_like(tau),
            wavelength=np.array([4000.0, 5000.0]),
            n_angle=3,
        )


def test_wrong_fixed_request_warns_without_mutating_checkpoint(monkeypatch):
    from wd_spectra.models import stellar

    original_config = DBConfig(effective_temperature=10000.0)
    data = ModelData.default()
    at = gray_helium_atmosphere(10000.0, 8.0, n_depth=8)
    at = replace(at, metadata={"radiative_equilibrium_converged": True})
    at = atmosphere_with_model_request_fingerprint(
        at, model_request_fingerprint("DB", original_config, data)
    )
    monkeypatch.setattr(stellar, "_helium_tables", lambda data: (None, None))
    monkeypatch.setattr(
        stellar,
        "synthesize_helium_spectrum",
        lambda *args, **kwargs: Spectrum(np.array([4000.0, 5000.0]), np.ones(2), {}),
    )
    with pytest.warns(
        AtmosphereConvergenceWarning, match="checkpoint request mismatch"
    ):
        result = compute_db(
            DBConfig(effective_temperature=20000.0),
            [4000.0, 5000.0],
            initial_atmosphere=at,
            relax_atmosphere=False,
        )
    assert result.metadata["atmosphere_convergence_status"] != "converged"
    assert result.atmosphere.effective_temperature == 10000.0
    assert at.metadata["radiative_equilibrium_converged"] is True
    assert "fixed_synthesis_request_verified" not in at.metadata


@pytest.mark.parametrize("teff,logg", [(20000.0, 8.0), (10000.0, 7.5)])
def test_changed_checkpoint_parameters_invalidate_claim(tmp_path, teff, logg):
    at = gray_helium_atmosphere(10000.0, 8.0, n_depth=8)
    at = replace(at, metadata={"radiative_equilibrium_converged": True})
    config = DBConfig(effective_temperature=10000.0)
    at = atmosphere_with_model_request_fingerprint(
        at, model_request_fingerprint("DB", config, ModelData.default())
    )
    path = (
        save_model_result(
            ModelResult(
                "DB",
                at,
                Spectrum(np.array([4000.0, 5000.0]), np.ones(2), {}),
                config,
                {},
            ),
            tmp_path / "saved",
        )
        / "atmosphere.npz"
    )
    loaded = load_atmosphere_checkpoint(path, teff, logg, "helium")
    assert not loaded.metadata["radiative_equilibrium_converged"]
    assert "model_request_fingerprint" not in loaded.metadata
    np.testing.assert_array_equal(loaded.temperature, at.temperature)
