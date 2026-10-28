"""Combined hydrogen controls retain both independently qualified predecessors."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
APPROVED = ROOT / "tests/data/approved_regressions"
DIRECTORY = APPROVED / "hydrogen_combined_2026_10_28"
MANIFEST = json.loads((DIRECTORY / "manifest.json").read_text())
PREVIOUS = json.loads((APPROVED / "hydrogen_dissolution_2026_10_27/manifest.json").read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_combined_controls_cover_every_dissolution_case():
    assert set(MANIFEST["cases"]) == set(PREVIOUS["cases"])
    assert len(MANIFEST["cases"]) == 25
    assert MANIFEST["combined_pr52_head"] == "a10c59f6c0a110a140d143b1ec51f0ce7852c9e7"
    assert MANIFEST["combined_main"] == "175d5cbdf57a418944c7b9e5bd9b6c7812b43e41"


@pytest.mark.parametrize("name", sorted(PREVIOUS["cases"]))
def test_combined_control_preserves_inputs_and_counterfactual_approvals(name):
    record = MANIFEST["cases"][name]
    assert digest(DIRECTORY / record["fixture"]) == record["fixture_sha256"]
    assert record["source_identity_sha256"] == MANIFEST["source_identity_sha256"]
    assert set(record["recoveries"]) == {"main54", "pr52", "pre_both"}
    assert record["tolerances_changed"] is False
    assert record["atmosphere_reconverged"] is False
    assert record["cold_start_qualification"] is False
    revision = record["request_fingerprint"]["physics_revision"]
    assert "montreal-h-radius-ali-griem-2026-10-27" in revision
    assert "hydrogen-full-dissolution-2026-10-27" in revision
    assert not record["config"].get("mean_3d_line_profiles", False)
    for path, expected in record["input_sha256"].items():
        assert digest(ROOT / path) == expected
    with np.load(DIRECTORY / record["fixture"], allow_pickle=False) as current:
        assert np.all(np.isfinite(current["surface_flux"]))
        assert np.all(current["surface_flux"] > 0.0)
        for recovery in record["recoveries"].values():
            assert recovery["passed"] is True
            reference = ROOT / recovery["reference"]
            assert digest(reference) == recovery["reference_sha256"]
            with np.load(reference, allow_pickle=False) as previous:
                np.testing.assert_array_equal(current["wavelength"], previous["wavelength"])
