"""Shipped dense-He He- ff table: provenance, domain and physical limits."""
import json
from pathlib import Path

import numpy as np
import pytest

from wd_spectra.constants import HELIUM_MASS
from wd_spectra.dense_helium_continuum import DenseHeliumCorrectionTable

PATH = Path(__file__).resolve().parents[1]/"src/wd_spectra/data/dq/correction.npz"


@pytest.fixture(scope="module")
def raw():
    with np.load(PATH, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def test_closure_and_dielectric_provenance(raw):
    meta = json.loads(str(raw["metadata_json"]))
    assert "Rogers-Young" in meta["fluid"] and meta["parameters_fitted"] is False
    assert meta["dielectric_dressing"] == "propagation"
    # Default factor is the bare correlation ratio; Blouin's 1/|eps|^2 is
    # reconstructible from the stored dielectric, not applied.
    np.testing.assert_array_equal(raw["factor"], raw["correlation_ratio"])
    eps = raw["dielectric"]
    assert np.all(eps >= 1) and np.all(eps[:, 0] == 1) and eps.max() < 2.5
    # Self-consistency was enforced at every tabulated state.
    np.testing.assert_allclose(raw["structure_zero"], raw["structure_zero_virial"], rtol=5e-3)


def test_domain_reaches_dense_cool_dq_photospheres():
    table = DenseHeliumCorrectionTable(PATH)
    top = 1e-3*np.expm1(table.axes[1][-1])
    assert top > 2.9   # J1803 requested 2.55 g/cm3 against the old 2.51 limit
    value = table.correction(np.array([5000.]), np.array([4500.]), np.array([2.55/HELIUM_MASS]))
    assert 0 < value[0, 0] < 1


def test_correction_tracks_long_wavelength_structure_factor(raw):
    # In cool DQ layers, as hbar*omega/kT -> 0 the thermal Born weight samples
    # k*a_i < 1, so the ratio approaches S(0) from above (hotter layers sample
    # larger k); it rises with photon energy.
    ratio, s0 = raw["correlation_ratio"], raw["structure_zero"]
    cool = np.exp(raw["log_temperature"]) < 6200
    dense = s0[cool] < .5
    longest = ratio[cool][:, :, -1][dense]
    assert np.all(longest >= s0[cool][dense]*(1-1e-3))
    assert np.all(longest <= s0[cool][dense]*1.25)
    assert np.all(np.diff(ratio[:, 1:, :], axis=2) <= 1e-12)   # decreasing with wavelength
    assert np.all(np.diff(ratio, axis=1) < 0)                  # more correlated when denser
    assert np.all(np.diff(ratio[:, 1:, :], axis=0) > 0)        # less correlated when hotter
