"""Offline re-capture of fixed-structure references after the 2026-11-10 audit fixes.

Two fixes move fixed-atmosphere spectra:

* hydrogen Stark x neutral-impact Lorentz convolution: the far wing is now
  resolved (DA, DAB and trace-H DZ cases, at most 0.26%);
* the Rohrmann RWA n=7, l=6 photoionization coefficient (public DAH default,
  at most 8e-5), and the 12C mass in the C2 dissociation constant (DQ J1311,
  at most 1.3%).

Each reference is re-synthesized on its unchanged frozen structure and
wavelength grid exactly as its test does, and written as soon as it is
finished.  The J1311 manifest gains a ``reapprovals`` entry with the previous
hash and the band changes.  Never run by tests or CI.

Usage: PYTHONPATH=src python research/capture_audit_fix_controls.py [CASE ...]
"""
from __future__ import annotations

from dataclasses import replace
import datetime
import json
from pathlib import Path
import sys
import tempfile
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from wd_spectra import _rt  # noqa: E402
from wd_spectra._compat import trapezoid  # noqa: E402
from wd_spectra.models import (  # noqa: E402
    DABConfig,
    DAConfig,
    DBConfig,
    DZConfig,
    compute_da,
    compute_dab,
    compute_db,
    compute_dz,
    load_atmosphere_checkpoint,
)
from wd_spectra.models.common import config_from_stored  # noqa: E402

assert _rt is not None, "build the compiled kernel first (tools_local/build_rt.py)"

APPROVED = ROOT / "tests/data/approved_regressions/current"
SPECTRAL = ROOT / "tests/data/spectral_regressions"
DQ = ROOT / "tests/data/dq_regressions"
SPECTRAL_CASES = ("da-5000", "da-20000", "dab-9000", "dab-20000", "dz-pg1225", "dz-j0738")
DAH_CASES = ("j1018+0111", "j1254+5612", "j1351+5419", "j2247+1456")
DQ_BANDS = ((3500, 4000), (4000, 4500), (4500, 5500), (5500, 7000))


def _write(path: Path, **arrays) -> float:
    """Write a compressed npz like the originals; return max relative change."""

    with np.load(path) as old:
        name = "surface_flux" if "surface_flux" in old else "flux"
        change = float(np.max(np.abs(arrays[name] / old[name] - 1.0)))
    np.savez_compressed(path, **arrays)
    return change


def spectral(case: str) -> float:
    # Same configuration and checkpoint loading as
    # tests/test_spectral_regressions.py.
    path = SPECTRAL / (case + ".npz")
    with np.load(path) as saved:
        kind = str(saved["spectral_type"])
        stored = json.loads(str(saved["config_json"]))
        if kind == "DB":
            stored.setdefault("transfer_discretization", "optical-depth")
        if kind == "DZ":
            stored.pop("strong_line_atomic_data", None)
        config = config_from_stored(
            {"DA": DAConfig, "DB": DBConfig, "DAB": DABConfig, "DZ": DZConfig}[kind],
            stored,
        )
        wave = saved["wavelength"].copy()
    with np.load(APPROVED / (case + ".npz")) as approved:
        np.testing.assert_array_equal(wave, approved["wavelength"])
    molecular = kind == "DA" and config.effective_temperature <= 12000
    atmosphere = load_atmosphere_checkpoint(
        path,
        config.effective_temperature,
        config.logg,
        {"DA": "hydrogen", "DB": "helium", "DAB": "mixed", "DZ": "helium"}[kind],
        include_molecules=molecular,
        include_negative_hydrogen=molecular,
        log_hydrogen_to_helium=getattr(config, "log_hydrogen_to_helium", None),
        trihydrogen_ion_partition_model="neale-tennyson-1995" if kind == "DA" else None,
    )
    compute = {"DA": compute_da, "DB": compute_db, "DAB": compute_dab, "DZ": compute_dz}[kind]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute(config, wave, initial_atmosphere=atmosphere, relax_atmosphere=False)
    return _write(
        APPROVED / (case + ".npz"),
        wavelength=wave,
        surface_flux=result.spectrum.surface_flux_lambda,
    )


def dah(key: str) -> float:
    from test_dah_paper import default_config, frozen_atmosphere
    from wd_spectra import compute_dah

    path = APPROVED / f"dah-default-{key}.npz"
    with np.load(path) as reference:
        wave = reference["wavelength"].copy()
    config = default_config(key)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_dah(
            config, wave, initial_atmosphere=frozen_atmosphere(key, config),
            relax_atmosphere=False,
        )
    return _write(path, wavelength=wave, surface_flux=result.spectrum.surface_flux_lambda)


def dq_j1311() -> float:
    # Same synthesis as tests/test_dq_spectral_regression.py.
    from wd_spectra import gray_helium_atmosphere
    from wd_spectra._dq.provenance import digest
    from wd_spectra._dq.runtime import make_material
    from wd_spectra.models.common import ModelData
    from wd_spectra.models.dq import DQConfig

    manifest_path = DQ / "completed-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    fixture = DQ / "j1311-completed-fixed.npz"
    previous = digest(fixture)
    assert previous == manifest["fixture_sha256"]
    config = DQConfig(**manifest["parameters"])
    with np.load(fixture, allow_pickle=False) as z:
        stored = {name: z[name].copy() for name in z.files}
    seed = gray_helium_atmosphere(
        config.effective_temperature, config.logg, n_depth=len(stored["temperature"])
    )
    state = replace(seed, **{name: stored[name].copy() for name in (
        "temperature", "gas_pressure", "column_mass", "rosseland_optical_depth")})
    with tempfile.TemporaryDirectory() as scratch:
        material = make_material(config, ModelData.default(), Path(scratch))
        state = material.chemistry(state)[0]
        flux = material.spectrum(state, stored["wavelength"], manifest["n_angle"]).surface_flux_lambda
    wave, old = stored["wavelength"], stored["flux"]
    bands = {}
    for lo, hi in DQ_BANDS:
        take = (wave >= lo) & (wave <= hi)
        bands[f"{lo}-{hi}"] = float(
            trapezoid(flux[take] - old[take], wave[take]) / trapezoid(old[take], wave[take])
        )
    stored["flux"] = flux
    change = _write(fixture, **stored)
    manifest["fixture_sha256"] = digest(fixture)
    manifest["reapprovals"].append({
        "date": datetime.date.today().isoformat(),
        "previous_fixture_sha256": previous,
        "physics_revision": "audit-fixes-2026-11-10",
        "reason": (
            "C2 dissociation constant uses the 12C2 reduced mass m(12C)/2 = 6 u "
            "instead of 6 hydrogen masses (K was 1.18% too large), so n(C2) "
            "rises by 1.18% where carbon is mostly atomic and the Swan bands "
            "deepen. The carbon mass in "
            "the density alone changes nothing here. Structure arrays unchanged."
        ),
        "band_fractional_changes": bands,
        "max_relative_change": change,
    })
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return change


def main(selected: list[str]) -> None:
    jobs = (
        [(case, lambda c=case: spectral(c)) for case in SPECTRAL_CASES]
        + [(f"dah-default-{key}", lambda k=key: dah(k)) for key in DAH_CASES]
        + [("dq-j1311", dq_j1311)]
    )
    for name, job in jobs:
        if selected and name not in selected:
            continue
        print(f"{name}: max relative change {job():.3e}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
