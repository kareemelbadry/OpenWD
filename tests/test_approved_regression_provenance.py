"""Approved outputs are frozen independently of the test that compares them."""
import hashlib
import json
from pathlib import Path

import pytest
from wd_spectra.models.common import _MODEL_PHYSICS_REVISION

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode,count", [("fixed", 12), ("cold", 8)])
def test_reviewed_baselines_and_historical_inputs_are_unchanged(mode, count):
    directory = ROOT / "tests/data/approved_regressions" / mode
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["physics_revision"] == _MODEL_PHYSICS_REVISION
    assert len(manifest["records"]) == count
    assert len({record["case"] for record in manifest["records"]}) == count
    for record in manifest["records"]:
        frozen = directory / (record["case"] + ".npz")
        historical = ROOT / record["historical_input"]
        assert hashlib.sha256(frozen.read_bytes()).hexdigest() == record["output_sha256"]
        assert hashlib.sha256(historical.read_bytes()).hexdigest() == record["historical_sha256"]
        assert record["source_error"] < 1e-10
        assert record["fresh_equilibrium_not_claimed"]
        if mode == "cold":
            assert record["source_public_cold_start"]
            certificate = record["source_equilibrium_certificate"]
            assert certificate["verified"]
            assert set(certificate["checks"]) == {
                "all_depth_flux", "local_energy", "temperature_stationarity",
                "source_closure", "boundary_screening",
            }
            assert all(check["passed"] for check in certificate["checks"].values())
        assert record["transfer_discretization"] == (
            "formal-pchip" if record["case"].startswith("da-") else "formal-linear"
        )
