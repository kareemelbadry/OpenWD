"""Move the DB 22000 cold-canary references by the infrared He I lines alone.

The infrared He I lines (wd_spectra.helium_impact) enter the helium synthesis
but not the default structure solve, so a cold DB start converges to the same
atmosphere as before and only its spectrum changes.  For each case this script
reads a ``ratio.npz`` written for that canary model (a local cold start of the
same configuration on the canary formal grid, synthesized with and without the
new lines on its converged structure), checks that the local structure matches
the approved reference at the canary tolerance, and writes a reference whose
flux is the approved flux times the with/without ratio.  Wavelength, gas
pressure and column mass are copied unchanged from the approved reference.

The script refuses an existing output directory and is never run by tests or
CI.  Copy the written files into
tests/data/approved_regressions/he_i_infrared_2026_11_27/cold/.

usage: python research/capture_he_i_infrared_cold_controls.py OUTDIR \\
           db-22000=PATH/ratio.npz db-22000-standard=PATH/ratio.npz
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wd_spectra._compat import trapezoid  # noqa: E402
APPROVED = ROOT / "tests/data/approved_regressions/cold"
SOURCES = (
    "src/wd_spectra/helium_impact.py",
    "src/wd_spectra/spectrum.py",
    "research/capture_he_i_infrared_cold_controls.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = {}
    for argument in sys.argv[2:]:
        case, ratio_path = argument.split("=", 1)
        ratio_path = Path(ratio_path).resolve()
        previous = APPROVED / (case + ".npz")
        with np.load(previous) as old, np.load(ratio_path) as local:
            wave = old["wavelength"]
            np.testing.assert_array_equal(wave, local["wavelength"])
            # The structure is not moved by the new lines: the local cold start
            # matches the approved one at the canary's structure tolerance.
            for name in ("gas_pressure", "column_mass"):
                np.testing.assert_allclose(
                    local[name][: len(old[name])], old[name], rtol=2e-5, atol=0.0
                )
            ratio = local["with_lines"] / local["without_lines"]
            assert np.all(np.isfinite(ratio)) and np.all(ratio > 0.0) and np.all(ratio <= 1.0 + 1e-12)
            flux = old["surface_flux"] * ratio
            target = out / (case + ".npz")
            np.savez_compressed(
                target, wavelength=wave, surface_flux=flux,
                gas_pressure=old["gas_pressure"], column_mass=old["column_mass"],
            )
            bands = {}
            for lo, hi in ((1150, 3000), (3500, 7000), (7000, 300000)):
                take = (wave >= lo) & (wave <= hi)
                bands[f"{lo}-{hi}"] = float(
                    trapezoid(flux[take] - old["surface_flux"][take], wave[take])
                    / trapezoid(old["surface_flux"][take], wave[take])
                )
            records[case] = dict(
                previous_control=str(previous.relative_to(ROOT)),
                previous_control_sha256=sha256(previous),
                new_control_sha256=sha256(target),
                local_ratio_sha256=sha256(ratio_path),
                structure_matches_previous_at_rtol_2e_5=True,
                minimum_ratio=float(np.min(ratio)),
                minimum_ratio_wavelength_angstrom=float(wave[np.argmin(ratio)]),
                maximum_change_below_7300_angstrom=float(np.max(1.0 - ratio[wave < 7300.0])),
                integrated_relative_change=bands,
            )
            print(case, json.dumps(records[case]), flush=True)
    manifest = dict(
        reason="helium synthesis includes the infrared He I lines (HELIUM_I_IMPACT_LINES) by default",
        capture="approved flux times the local with/without-lines ratio on the same converged cold structure",
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                  text=True).stdout.strip(),
        source_sha256={name: sha256(ROOT / name) for name in SOURCES},
        python=platform.python_version(),
        numpy=np.__version__,
        cases=records,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
