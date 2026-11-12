"""Measure (and, with --write, re-capture) fixed-structure references after the 2026-11-11 audit fixes.

Three fixes can move fixed-atmosphere spectra:

* hydrogen Stark tables: the log-bilinear (ne, T) interpolation is rescaled to
  the bilinear weighting of the corner-profile areas (all hydrogen cases);
* the hydrogenic Kramers excited-level cross section carries the level's
  Gaunt factor (hot NLTE metal continua; no fixed-structure control here);
* J-resolved TLUSTY ground terms are measured from the lowest J level (Si II
  edges in D6 and DZ).

Each reference is re-synthesized on its unchanged frozen structure and
wavelength grid exactly as its test does.  Without ``--write`` nothing is
written; the script prints the maximum relative change over the whole grid,
in significant flux (lambda F > 1% of its maximum, the tests' 0.1% region),
and of the integrated flux.  Never run by tests or CI.

Usage: PYTHONPATH=src python research/capture_stark_area_controls.py [--write] [CASE ...]
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "tests"))

import capture_audit_fix_controls as base  # noqa: E402
from wd_spectra._compat import trapezoid  # noqa: E402

APPROVED = base.APPROVED
WRITE = False
SPECTRAL_CASES = (
    "da-3000", "da-4000", "da-5000", "da-20000", "db-22000",
    "dab-9000", "dab-20000", "dz-pg1225", "dz-j0738",
)


def _report(path: Path, **arrays) -> str:
    with np.load(path) as old:
        name = "surface_flux" if "surface_flux" in old else "flux"
        before, wave = old[name], old["wavelength"]
    after = arrays[name]
    ratio = np.abs(after / before - 1.0)
    significant = wave * before > 0.01 * np.max(wave * before)
    integral = trapezoid(after, wave) / trapezoid(before, wave) - 1.0
    worst = float(wave[np.argmax(np.where(significant, ratio, 0.0))])
    if WRITE:
        np.savez_compressed(path, **arrays)
    return (
        f"max {np.max(ratio):.2e}  significant {np.max(ratio[significant]):.2e}"
        f" at {worst:.1f} A  integral {integral:+.2e}"
        + ("  [written]" if WRITE else "")
    )


base._write = _report


def dah_paper(key: str) -> str:
    from test_dah_paper import frozen_atmosphere, paper_config
    from wd_spectra import compute_dah

    path = APPROVED / f"dah-paper-{key}.npz"
    with np.load(path) as reference:
        wave = reference["wavelength"].copy()
    config = paper_config(key)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_dah(config, wave, initial_atmosphere=frozen_atmosphere(key, config),
                             relax_atmosphere=False)
    return _report(path, wavelength=wave, surface_flux=result.spectrum.surface_flux_lambda)


def daz(case: str) -> str:
    # Same synthesis as tests/test_daz_regressions.py.
    from wd_spectra import DAZConfig, compute_daz
    from wd_spectra.models import load_atmosphere_checkpoint
    from wd_spectra.models.common import config_from_stored

    controls = ROOT / "tests/data/daz_regressions"
    with np.load(controls / (case + ".npz")) as saved:
        stored = json.loads(str(saved["config_json"]))
        stored.pop("strong_line_atomic_data", None)
        config = config_from_stored(DAZConfig, stored)
        wave = saved["wavelength"].copy()
    path = APPROVED / ("daz-" + case + ".npz")
    with np.load(path) as approved:
        np.testing.assert_array_equal(wave, approved["wavelength"])
    molecules = config.effective_temperature <= 12000
    atmosphere = load_atmosphere_checkpoint(
        controls / (case + ".npz"), config.effective_temperature, config.logg, "hydrogen",
        include_molecules=molecules, include_negative_hydrogen=molecules,
        trihydrogen_ion_partition_model="neale-tennyson-1995",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_daz(config, wave, initial_atmosphere=atmosphere, relax_atmosphere=False)
    return _report(path, wavelength=wave, surface_flux=result.spectrum.surface_flux_lambda)


def public_da(case: str) -> str:
    # Same synthesis as tests/test_da_public_spectrum.py.
    from wd_spectra import DAConfig, compute_da
    from wd_spectra.models import load_atmosphere_checkpoint
    from wd_spectra.models.common import config_from_stored

    source = ROOT / "tests/data/da_cubic_regressions" / (case + ".npz")
    with np.load(source) as saved:
        config = config_from_stored(DAConfig, json.loads(str(saved["config_json"])))
        wave = saved["wavelength"].copy()
    path = APPROVED / ("public-" + case + ".npz")
    with np.load(path) as approved:
        np.testing.assert_array_equal(wave, approved["wavelength"])
    molecular = config.effective_temperature <= 12000
    state = load_atmosphere_checkpoint(
        source, config.effective_temperature, config.logg, "hydrogen",
        include_molecules=molecular, include_negative_hydrogen=molecular,
        trihydrogen_ion_partition_model="neale-tennyson-1995",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_da(config, wave, initial_atmosphere=state, relax_atmosphere=False)
    return _report(path, wavelength=wave, surface_flux=result.spectrum.surface_flux_lambda)


def main(arguments: list[str]) -> None:
    global WRITE
    WRITE = "--write" in arguments
    selected = [a for a in arguments if a != "--write"]
    from test_dah_paper import PAPER_KEYS, TARGETS

    jobs = (
        [(case, lambda c=case: base.spectral(c)) for case in SPECTRAL_CASES]
        + [("public-da-12000-public", lambda: public_da("da-12000-public"))]
        + [(f"daz-{c}", lambda c=c: daz(c)) for c in ("g149_28", "galex1931")]
        + [(f"dah-default-{k}", lambda k=k: base.dah(k)) for k in TARGETS]
        + [(f"dah-paper-{k}", lambda k=k: dah_paper(k)) for k in PAPER_KEYS]
    )
    for name, job in jobs:
        if selected and name not in selected:
            continue
        print(f"{name}: {job()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
