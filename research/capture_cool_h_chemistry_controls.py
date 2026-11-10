"""Offline capture of the controls moved by the 2026-11-10 cool-hydrogen physics.

That revision divides the Neale & Tennyson H3+ partition function by its eight
proton-spin states, uses the 0 K H3+ dissociation energy, normalizes the
H--H2 Lyalpha red wing to Kowalski & Saumon (2006) with a continued red tail,
and adds H2- free-free and H2--H CIA.  It moves only models with H2/H3+.

``fixed`` re-synthesizes the named ``current/`` fixed-spectrum controls on their
unchanged frozen structures (``capture_release_controls.capture``) and replaces
them in place.  ``cold`` takes completed public cold starts from the protected
canaries (OPENWD_TEST_ARTIFACTS), which ``capture`` verifies, writes them to a
new reviewed directory and records the change from the previous reviewed cold
references.  Tolerances are unchanged.  Never run by tests or CI.

Usage:
  PYTHONPATH=src python research/capture_cool_h_chemistry_controls.py fixed CASE [CASE ...]
  PYTHONPATH=src python research/capture_cool_h_chemistry_controls.py cold OUTPUT SOURCES.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from capture_release_controls import capture, digest, source_identity  # noqa: E402
from capture_montreal_h_radius_cold_controls import change  # noqa: E402

CURRENT = ROOT / "tests/data/approved_regressions/current"
PREVIOUS_COLD = ROOT / "tests/data/approved_regressions/montreal_h_radius_2026_10_27/cold"


def _manifest(records):
    return dict(
        description=__doc__, date="2026-11-10",
        reference_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_identity=source_identity(), python=platform.python_version(),
        numpy=np.__version__, cases=records)


def fixed(cases):
    records = {}
    for case in cases:
        previous = CURRENT / (case + ".npz")
        with np.load(previous) as old:
            old_wave, old_flux = old["wavelength"].copy(), old["surface_flux"].copy()
        record = capture(case, "fixed", CURRENT, {})
        with np.load(CURRENT / (case + ".npz")) as new:
            np.testing.assert_array_equal(new["wavelength"], old_wave)
            flux = new["surface_flux"]
        important = old_wave * old_flux > 0.01 * np.max(old_wave * old_flux)
        record["max_significant_change_from_previous_control"] = float(
            np.max(abs(flux[important] / old_flux[important] - 1.0)))
        records[case] = record
        print(case, record["max_significant_change_from_previous_control"], flush=True)
    return records


def cold(output, sources):
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    for case, source in sources.items():
        record = capture(case, "cold", output, {case: source})
        previous = PREVIOUS_COLD / (case + ".npz")
        record["previous_cold_reference"] = str(previous.relative_to(ROOT))
        record["previous_cold_reference_sha256"] = digest(previous)
        record["change_from_previous"] = change(
            output / (case + ".npz"), previous, record["config"]["effective_temperature"])
        record["tolerances_changed"] = False
        records[case] = record
    (output / "manifest.json").write_text(
        json.dumps(_manifest(records), indent=2, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("fixed")
    f.add_argument("cases", nargs="+")
    c = sub.add_parser("cold")
    c.add_argument("output", type=Path)
    c.add_argument("sources", type=Path)
    args = parser.parse_args()
    if args.mode == "fixed":
        records = fixed(args.cases)
        print(json.dumps({k: v.get("max_significant_change_from_previous_control")
                          for k, v in records.items()}, indent=2))
    else:
        cold(args.output, json.loads(args.sources.read_text()))


if __name__ == "__main__":
    main()
