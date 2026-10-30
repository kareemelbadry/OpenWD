"""J1637 PR regression coverage; full cold convergence remains a separate test."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from wd_spectra import D6Config
from wd_spectra.models.common import AtmosphereConvergenceWarning
from d6_regression_support import (
    ARRAYS, METRICS, STEPS, assert_trajectory_matches, atomic_input_scope,
    fixed_spectrum, trajectory,
)

DIRECTORY = Path(__file__).parent / "data/d6_regressions/warm_v5_excited_parent_o_i"
SEED = DIRECTORY.with_name(DIRECTORY.name + "-seed.npz")


def test_frozen_j1637_seed_certificate_and_reference_integrity():
    manifest = json.loads((DIRECTORY / "manifest.json").read_text())
    path = DIRECTORY / "j1637.npz"
    assert manifest["schema"] == 1
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["output_sha256"]
    assert hashlib.sha256(SEED.read_bytes()).hexdigest() == manifest["seed_sha256"]
    assert manifest["config"] == json.loads(json.dumps(asdict(D6Config())))
    assert manifest["steps"] == STEPS and manifest["metrics"] == list(METRICS)
    assert manifest["cold_start_qualification"] is False
    assert manifest["source_request"]["config"] == manifest["config"]
    certificate = manifest["source_cold_certificate"]
    assert certificate["verified"] and not certificate["failures"]
    for key in certificate["required_checks"]:
        check = certificate["checks"][key]
        assert check["measured"] and check["passed"]
        assert 0 <= check["value"] < check["tolerance"]
    with np.load(path, allow_pickle=False) as saved:
        with np.load(SEED, allow_pickle=False) as seed:
            for key in seed.files:
                np.testing.assert_array_equal(seed[key], saved[key])
        assert saved["base_temperature"].shape == (48,)
        for key in ARRAYS:
            assert saved[key].shape == (STEPS, 48)
            assert np.all(np.isfinite(saved[key])) and np.all(saved[key] > 0)
        assert saved["metrics"].shape == (STEPS, len(METRICS))
        assert np.all(np.isfinite(saved["metrics"]))
        perturbation = 0.015 * np.exp(-((np.log10(saved["optical_depth"]) + 2) / 1)**2)
        np.testing.assert_allclose(saved["initial_temperature"],
            saved["base_temperature"] * np.exp(perturbation), rtol=1e-14)
        assert np.max(np.abs(saved["temperature"][0] / saved["initial_temperature"] - 1)) > 1e-4


@pytest.mark.canary
def test_j1637_two_step_warm_trajectory():
    with atomic_input_scope():
        with np.load(SEED, allow_pickle=False) as seed:
            records = trajectory(seed["column_mass"], seed["optical_depth"],
                                 seed["initial_temperature"])
    with np.load(DIRECTORY / "j1637.npz", allow_pickle=False) as saved:
        assert_trajectory_matches(records, saved)


@pytest.mark.spectral
def test_j1637_fixed_atmosphere_spectrum(monkeypatch):
    import wd_spectra.models.d6 as public

    def forbidden_solve(*args, **kwargs):
        pytest.fail("fixed-atmosphere D6 spectrum started an atmosphere solve")

    monkeypatch.setattr(public, "radiative_equilibrium_d6_atmosphere", forbidden_solve)
    with np.load(DIRECTORY / "j1637.npz", allow_pickle=False) as saved:
        with atomic_input_scope(), pytest.warns(AtmosphereConvergenceWarning):
            spectrum = fixed_spectrum(saved["column_mass"], saved["optical_depth"],
                                      saved["base_temperature"], saved["wavelength"])
        flux = spectrum.surface_flux_lambda
        assert np.all(np.isfinite(flux)) and np.all(flux > 0)
        np.testing.assert_allclose(flux, saved["surface_flux"], rtol=2e-6,
                                   atol=1e-12 * np.max(saved["surface_flux"]))
        assert spectrum.metadata["source_converged"]
        assert spectrum.metadata["independent_radiation_scaled_source_error"] < 1e-10
        assert spectrum.metadata["maximum_metal_lines"] == 25_000
        assert spectrum.metadata["transfer_discretization"] == "formal-linear"
