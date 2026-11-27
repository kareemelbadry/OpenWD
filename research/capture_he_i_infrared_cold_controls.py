"""Capture DB cold-canary references for the infrared He I lines.

The infrared He I lines (wd_spectra.helium_impact) now enter the DB structure
and spectrum.  Each case reads a ``ratio.npz`` written by a local cold start of
the canary model (the configuration and formal wavelength grid of
tests/test_protected_model_canaries.py) whose equilibrium certificate was
verified, and stores its surface flux, gas pressure and column mass as the new
reference.  ``ratio.npz`` also holds the flux re-synthesized on the same
structure without the new lines, so the manifest separates the line opacity
itself from the structural response.

The script refuses an existing output directory and is never run by tests or
CI.  Copy the written files into
tests/data/approved_regressions/he_i_infrared_2026_11_27/cold/.

usage: python research/capture_he_i_infrared_cold_controls.py OUTDIR \\
           CASE=PATH/ratio.npz [CASE=PATH/ratio.npz ...]
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

REVIEWED = (
    ROOT / "tests/data/approved_regressions/heh_mixed_heminus_2026_11_19/cold",
    ROOT / "tests/data/approved_regressions/cold",
)
SOURCES = (
    "src/wd_spectra/helium_impact.py",
    "src/wd_spectra/spectrum.py",
    "src/wd_spectra/atmosphere.py",
    "src/wd_spectra/models/stellar.py",
    "research/capture_he_i_infrared_cold_controls.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bands(wave, new, old):
    result = {}
    for lo, hi in ((1150, 3000), (3500, 7000), (7000, 300000)):
        take = (wave >= lo) & (wave <= hi)
        result[f"{lo}-{hi}"] = float(
            trapezoid(new[take] - old[take], wave[take]) / trapezoid(old[take], wave[take])
        )
    return result


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = {}
    for argument in sys.argv[2:]:
        case, ratio_path = argument.split("=", 1)
        ratio_path = Path(ratio_path).resolve()
        done = json.loads((ratio_path.parent / "model.done").read_text())
        assert done["verified"] and done["status"] == "converged", done
        previous = next(folder / (case + ".npz") for folder in REVIEWED
                        if (folder / (case + ".npz")).exists())
        with np.load(previous) as old, np.load(ratio_path) as local:
            wave = old["wavelength"]
            np.testing.assert_array_equal(wave, local["wavelength"])
            flux = local["with_lines"]
            target = out / (case + ".npz")
            np.savez_compressed(
                target, wavelength=wave, surface_flux=flux,
                gas_pressure=local["gas_pressure"], column_mass=local["column_mass"],
            )
            n = min(len(old["gas_pressure"]), len(local["gas_pressure"]))
            records[case] = dict(
                previous_control=str(previous.relative_to(ROOT)),
                previous_control_sha256=sha256(previous),
                new_control_sha256=sha256(target),
                local_run_sha256=sha256(ratio_path),
                local_run=done,
                maximum_gas_pressure_change=float(
                    np.max(np.abs(local["gas_pressure"][:n] / old["gas_pressure"][:n] - 1.0))
                ),
                maximum_column_mass_change=float(
                    np.max(np.abs(local["column_mass"][:n] / old["column_mass"][:n] - 1.0))
                ),
                maximum_flux_change_below_7300_angstrom=float(
                    np.max(np.abs(flux[wave < 7300.0] / old["surface_flux"][wave < 7300.0] - 1.0))
                ),
                integrated_relative_change=bands(wave, flux, old["surface_flux"]),
                integrated_change_from_line_opacity_alone=bands(
                    wave, local["with_lines"], local["without_lines"]
                ),
            )
            print(case, json.dumps(records[case]), flush=True)
    manifest = dict(
        reason=("infrared He I lines (HELIUM_I_IMPACT_LINES) in the DB structure opacity "
                "and spectrum"),
        capture="local verified cold start of each canary model on the canary formal grid",
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                  text=True).stdout.strip(),
        source_sha256={name: sha256(ROOT / name) for name in SOURCES},
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        cases=records,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
