"""Capture the DB, DAB and DZ fixed-spectrum controls for the infrared He I lines.

The helium synthesis now includes HELIUM_I_IMPACT_LINES (He I 10830, 20581
and the n = 3 series, wd_spectra.helium_impact) by default.  The frozen
structures, configurations and wavelength grids are unchanged, and the
structure solver does not use the new lines by default.  For each case the
script first proves that synthesis with the new lines switched off reproduces
the current approved control at the test tolerance, then synthesizes the
control with the current default.  No atmosphere is re-relaxed; these are not
cold-start qualifications.

The script refuses an existing output directory and is never run by tests or
CI. Copy the written files into tests/data/approved_regressions/current/ and
record manifest.json in the pull request.

usage: python research/capture_he_i_infrared_controls.py OUTDIR
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "MKL_NUM_THREADS", "OPENWD_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import wd_spectra.models.stellar as stellar  # noqa: E402
import wd_spectra.spectrum as spectrum  # noqa: E402
from wd_spectra._compat import trapezoid  # noqa: E402
from wd_spectra.models import (  # noqa: E402
    AtmosphereConvergenceWarning,
    DABConfig,
    DBConfig,
    DZConfig,
    compute_dab,
    compute_db,
    compute_dz,
    load_atmosphere_checkpoint,
)
from wd_spectra.models.common import config_from_stored  # noqa: E402

DATA = ROOT / "tests/data"
CURRENT = DATA / "approved_regressions/current"
CASES = ("db-22000", "dab-9000", "dab-20000", "dz-pg1225", "dz-j0738")
SOURCES = (
    "src/wd_spectra/helium_impact.py",
    "src/wd_spectra/spectrum.py",
    "src/wd_spectra/atmosphere.py",
    "src/wd_spectra/models/stellar.py",
    "research/capture_he_i_infrared_controls.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthesize(path, *, impact_lines):
    with np.load(path) as saved:
        kind = str(saved["spectral_type"])
        stored = json.loads(str(saved["config_json"]))
        wave = saved["wavelength"]
    # As in tests/test_spectral_regressions.py.
    if kind == "DB":
        stored.setdefault("transfer_discretization", "optical-depth")
    if kind == "DZ":
        stored.pop("strong_line_atomic_data", None)
    config_type, compute, composition = {
        "DB": (DBConfig, compute_db, "helium"),
        "DAB": (DABConfig, compute_dab, "mixed"),
        "DZ": (DZConfig, compute_dz, "helium"),
    }[kind]
    config = config_from_stored(config_type, stored)
    atmosphere = load_atmosphere_checkpoint(
        path, config.effective_temperature, config.logg, composition,
        include_molecules=False, include_negative_hydrogen=False,
        log_hydrogen_to_helium=getattr(config, "log_hydrogen_to_helium", None),
        trihydrogen_ion_partition_model=None,
    )
    original = spectrum.synthesize_helium_spectrum
    original_stellar = stellar.synthesize_helium_spectrum
    if not impact_lines:
        switched = functools.partial(original, include_helium_i_impact_lines=False)
        spectrum.synthesize_helium_spectrum = switched
        stellar.synthesize_helium_spectrum = switched
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AtmosphereConvergenceWarning)
            result = compute(config, wave, initial_atmosphere=atmosphere, relax_atmosphere=False)
    finally:
        spectrum.synthesize_helium_spectrum = original
        stellar.synthesize_helium_spectrum = original_stellar
    flux = result.spectrum.surface_flux_lambda
    assert np.all(np.isfinite(flux)) and np.all(flux > 0)
    assert result.spectrum.metadata["source_converged"]
    return kind, wave, flux


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = {}
    for case in CASES:
        path = DATA / "spectral_regressions" / (case + ".npz")
        with np.load(CURRENT / (case + ".npz")) as approved:
            old = approved["surface_flux"]
            approved_wave = approved["wavelength"]
        # 1. The new lines switched off reproduce the current control.
        kind, wave, previous = synthesize(path, impact_lines=False)
        np.testing.assert_array_equal(wave, approved_wave)
        np.testing.assert_allclose(previous, old, rtol=2e-6, atol=1e-12 * np.max(old))
        # 2. The same structure, configuration and grid with the current default.
        _, _, new = synthesize(path, impact_lines=True)
        target = out / (case + ".npz")
        np.savez_compressed(target, wavelength=wave, surface_flux=new)
        significant = wave * old > 0.01 * np.max(wave * old)
        ratio = new[significant] / old[significant]
        optical = wave < 7300.0
        bands = {}
        for lo, hi in ((1150, 3000), (3000, 4000), (4000, 7000), (7000, 300000)):
            take = (wave >= lo) & (wave <= hi)
            if np.count_nonzero(take) >= 2:
                bands[f"{lo}-{hi}"] = float(
                    trapezoid(new[take] - old[take], wave[take]) / trapezoid(old[take], wave[take])
                )
        worst = int(np.argmax(np.abs(np.log(ratio))))
        records[case] = dict(
            family=kind,
            frozen_structure=str(path.relative_to(ROOT)),
            frozen_structure_sha256=sha256(path),
            previous_control_sha256=sha256(CURRENT / (case + ".npz")),
            new_control_sha256=sha256(target),
            wavelength_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
            previous_reproduced_at_test_tolerance=True,
            atmosphere_reconverged=False,
            significant_pixels=int(np.count_nonzero(significant)),
            maximum_relative_change=float(ratio[worst] - 1.0),
            maximum_change_wavelength_angstrom=float(wave[significant][worst]),
            maximum_relative_change_below_7300_angstrom=float(
                np.max(np.abs(new[optical] / old[optical] - 1.0))
            ),
            pixels_changed_over_0p1_percent=int(np.count_nonzero(np.abs(ratio - 1.0) > 1e-3)),
            integrated_relative_change=bands,
        )
        print(case, json.dumps(records[case]["integrated_relative_change"]),
              f"max {records[case]['maximum_relative_change']:+.4f} at "
              f"{records[case]['maximum_change_wavelength_angstrom']:.2f} A; "
              f"<7300 A {records[case]['maximum_relative_change_below_7300_angstrom']:.2e}",
              flush=True)
    manifest = dict(
        reason="helium synthesis includes the infrared He I lines (HELIUM_I_IMPACT_LINES) by default",
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                  text=True).stdout.strip(),
        git_dirty=bool(subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=ROOT,
                                      capture_output=True, text=True).stdout.strip()),
        source_sha256={name: sha256(ROOT / name) for name in SOURCES},
        python=platform.python_version(),
        numpy=np.__version__,
        cases=records,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
