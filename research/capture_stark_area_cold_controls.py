"""Capture the protected cold-start controls moved by the 2026-11-11 Stark-table fix.

The hydrogen Stark-table interpolation in electron density and temperature is
now rescaled to the bilinear weighting of the corner-profile areas.  In the
first full validation of that change only ``da-5000`` moved beyond its cold
tolerance: its structure still matched the previous reviewed reference to
2e-5, and its Halpha core deepened by 0.37%.

``capture`` (from capture_release_controls) loads a completed public cold start
saved by the protected canary (OPENWD_TEST_ARTIFACTS), verifies its equilibrium
certificate and cold-start flags, and re-synthesizes the default spectrum on
that structure.  This writes them to a new reviewed directory and records the
change from the previous reviewed cold reference.  Tolerances are unchanged.
Never run by tests or CI.

Usage:
  PYTHONPATH=src python research/capture_stark_area_cold_controls.py OUTPUT SOURCES.json
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

PREVIOUS_COLD = ROOT / "tests/data/approved_regressions/cool_h_chemistry_2026_11_10/cold"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("sources", type=Path,
                        help="JSON mapping case to a verified cold-start atmosphere.npz")
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    records = {}
    for case, source in sources.items():
        record = capture(case, "cold", args.output, {case: source})
        previous = PREVIOUS_COLD / (case + ".npz")
        record["previous_cold_reference"] = str(previous.relative_to(ROOT))
        record["previous_cold_reference_sha256"] = digest(previous)
        record["change_from_previous"] = change(
            args.output / (case + ".npz"), previous, record["config"]["effective_temperature"])
        record["tolerances_changed"] = False
        records[case] = record
    manifest = dict(
        description=__doc__, date="2026-11-12",
        reference_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_identity=source_identity(), python=platform.python_version(),
        numpy=np.__version__, cases=records)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({k: v["change_from_previous"] for k, v in records.items()}, indent=2, default=str))


if __name__ == "__main__":
    main()
