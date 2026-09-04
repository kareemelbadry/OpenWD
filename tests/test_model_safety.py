from dataclasses import replace
import warnings

import numpy as np
import pytest

from wd_spectra.atmosphere import Atmosphere
from wd_spectra.models import (
    AtmosphereConvergenceWarning,
    DBConfig,
    DZConfig,
    ModelData,
    ModelResult,
    atmosphere_convergence_status,
    load_atmosphere_checkpoint,
    save_model_result,
)
from wd_spectra.models import stellar
from wd_spectra.models.common import (
    atmosphere_matches_model_request,
    atmosphere_with_model_request_fingerprint,
    model_request_fingerprint,
    warn_if_atmosphere_not_converged,
)
from wd_spectra.spectrum import Spectrum


def _atmosphere(metadata=None):
    depth = np.array([1.0e-6, 1.0e-2, 1.0])
    return Atmosphere(
        effective_temperature=10_000.0,
        logg=8.0,
        rosseland_optical_depth=depth,
        column_mass=depth,
        temperature=np.array([8_000.0, 9_000.0, 11_000.0]),
        gas_pressure=np.array([1.0e2, 1.0e5, 1.0e8]),
        mass_density=np.array([1.0e-8, 1.0e-5, 1.0e-2]),
        neutral_h_density=np.zeros(3),
        proton_density=np.zeros(3),
        electron_density=np.full(3, 1.0e10),
        metadata={} if metadata is None else dict(metadata),
    )


def _spectrum():
    return Spectrum(
        wavelength_angstrom=np.array([4_000.0, 5_000.0]),
        surface_flux_lambda=np.array([1.0, 2.0]),
        metadata={},
    )


def test_model_request_fingerprint_is_order_independent_and_request_scoped(
    tmp_path,
):
    data = ModelData(tmp_path)
    first = model_request_fingerprint(
        "DZ",
        DZConfig(abundances={"Ca": -8.0, "Mg": -7.0}),
        data,
    )
    reordered = model_request_fingerprint(
        "DZ",
        DZConfig(abundances={"Mg": -7.0, "Ca": -8.0}),
        data,
    )
    changed = model_request_fingerprint(
        "DZ",
        DZConfig(abundances={"Ca": -8.1, "Mg": -7.0}),
        data,
    )

    assert first["sha256"] == reordered["sha256"]
    assert first["sha256"] != changed["sha256"]


def test_only_an_exact_fingerprint_authorizes_direct_resume(tmp_path):
    data = ModelData(tmp_path)
    requested = model_request_fingerprint("DB", DBConfig(), data)
    other = model_request_fingerprint(
        "DB", DBConfig(mixing_length_alpha=0.8), data
    )
    checkpoint = atmosphere_with_model_request_fingerprint(
        _atmosphere({"checkpoint_composition_verified": True}), requested
    )

    assert atmosphere_matches_model_request(checkpoint, requested)
    assert not atmosphere_matches_model_request(checkpoint, other)
    tampered = replace(
        checkpoint,
        metadata={
            **checkpoint.metadata,
            "model_request_fingerprint": {
                **requested,
                "data_root": "/different/data",
            },
        },
    )
    assert not atmosphere_matches_model_request(tampered, requested)
    assert not atmosphere_matches_model_request(
        _atmosphere({"checkpoint_composition_verified": True}), requested
    )


def test_unconverged_and_unknown_atmospheres_warn_without_blocking():
    unconverged = _atmosphere(
        {
            "radiative_equilibrium_converged": False,
            "maximum_all_depth_total_flux_residual": 0.12,
        }
    )
    with pytest.warns(
        AtmosphereConvergenceWarning,
        match="did not converge.*maximum all-depth flux residual=1.200e-01",
    ):
        assert (
            warn_if_atmosphere_not_converged(unconverged, "DB")
            == "unconverged"
        )

    with pytest.warns(AtmosphereConvergenceWarning, match="does not record"):
        assert warn_if_atmosphere_not_converged(_atmosphere(), "DB") == "unknown"


def test_converged_atmosphere_does_not_warn():
    atmosphere = _atmosphere({"radiative_equilibrium_converged": True})
    with warnings.catch_warnings():
        warnings.simplefilter("error", AtmosphereConvergenceWarning)
        assert warn_if_atmosphere_not_converged(atmosphere, "DB") == "converged"
    assert atmosphere_convergence_status(atmosphere) == "converged"


def test_checkpoint_round_trip_preserves_request_fingerprint(tmp_path):
    data = ModelData(tmp_path)
    fingerprint = model_request_fingerprint("DB", DBConfig(), data)
    atmosphere = atmosphere_with_model_request_fingerprint(
        _atmosphere({"radiative_equilibrium_converged": True}), fingerprint
    )
    result = ModelResult("DB", atmosphere, _spectrum(), DBConfig(), {})

    output = save_model_result(result, tmp_path / "saved")
    restored = load_atmosphere_checkpoint(
        output / "atmosphere.npz",
        10_000.0,
        8.0,
        "helium",
    )

    assert atmosphere_matches_model_request(restored, fingerprint)
    assert restored.metadata["radiative_equilibrium_converged"] is True


def test_compute_db_warns_and_returns_an_unconverged_exploratory_result(
    monkeypatch, tmp_path
):
    captured = {}
    partial = _atmosphere({"radiative_equilibrium_converged": False})
    monkeypatch.setattr(stellar, "_helium_tables", lambda data: (object(), object()))

    def fake_relaxation(*args, **kwargs):
        captured.update(kwargs)
        return partial

    monkeypatch.setattr(
        stellar, "radiative_equilibrium_helium_atmosphere", fake_relaxation
    )
    monkeypatch.setattr(stellar, "synthesize_helium_spectrum", lambda *a, **k: _spectrum())

    with pytest.warns(AtmosphereConvergenceWarning):
        result = stellar.compute_db(
            DBConfig(quality="quick"),
            np.array([4_000.0, 5_000.0]),
            data=ModelData(tmp_path),
        )

    assert result.atmosphere.metadata["radiative_equilibrium_converged"] is False
    assert result.metadata["atmosphere_convergence_status"] == "unconverged"
    assert captured["resume_supplied_structure_in_formal_flux_phase"] is False
    assert captured["initial_gas_pressure"] is None
    assert captured["initial_rosseland_optical_depth"] is None


def test_compute_db_exact_fingerprint_controls_direct_resume(monkeypatch, tmp_path):
    data = ModelData(tmp_path)
    config = DBConfig(quality="quick")
    fingerprint = model_request_fingerprint("DB", config, data)
    checkpoint = atmosphere_with_model_request_fingerprint(
        _atmosphere({"radiative_equilibrium_converged": True}), fingerprint
    )
    captured = {}
    monkeypatch.setattr(stellar, "_helium_tables", lambda data: (object(), object()))

    def fake_relaxation(*args, **kwargs):
        captured.update(kwargs)
        return replace(
            checkpoint,
            metadata={
                **checkpoint.metadata,
                "radiative_equilibrium_converged": True,
            },
        )

    monkeypatch.setattr(
        stellar, "radiative_equilibrium_helium_atmosphere", fake_relaxation
    )
    monkeypatch.setattr(stellar, "synthesize_helium_spectrum", lambda *a, **k: _spectrum())

    stellar.compute_db(
        config,
        np.array([4_000.0, 5_000.0]),
        data=data,
        initial_atmosphere=checkpoint,
    )

    assert captured["resume_supplied_structure_in_formal_flux_phase"] is True
    np.testing.assert_array_equal(
        captured["initial_gas_pressure"], checkpoint.gas_pressure
    )
    np.testing.assert_array_equal(
        captured["initial_rosseland_optical_depth"],
        checkpoint.rosseland_optical_depth,
    )
