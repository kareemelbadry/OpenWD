"""Versioned full-dissolution controls retain their exact inputs and predecessors."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'tests/data/approved_regressions/hydrogen_dissolution_2026_10_27'
MANIFEST = json.loads((DIRECTORY / 'manifest.json').read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('case', sorted(MANIFEST['cases']))
def test_dissolution_controls_retain_their_input_and_approval_chain(case):
    record = MANIFEST['cases'][case]
    source = ROOT / record['structure_input']
    previous = ROOT / record['previous_approved']
    output = DIRECTORY / 'fixed' / (case + '.npz')
    assert digest(source) == record['structure_sha256']
    assert digest(previous) == record['previous_sha256']
    assert digest(output) == record['fixture_sha256']
    assert record['tolerances_changed'] is False
    assert record['cold_start_qualification'] is False
    assert record['atmosphere_reconverged'] is False
    assert record['request_fingerprint']['physics_revision'].endswith(
        'hydrogen-full-dissolution-2026-10-27')
    with np.load(output, allow_pickle=False) as current, np.load(previous, allow_pickle=False) as prior:
        np.testing.assert_array_equal(current['wavelength'], prior['wavelength'])
        flux = current['surface_flux']
        assert np.all(np.isfinite(flux)) and np.all(flux > 0.)
        old = next(prior[name] for name in ('surface_flux', 'original_surface_flux', 'flux')
                   if name in prior.files)
        maximum = float(np.max(np.abs(flux / old - 1.)))
        assert maximum == record['change_from_previous']['maximum_relative_change']
