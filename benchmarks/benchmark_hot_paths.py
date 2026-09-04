"""Reproducible timings for the production hydrogen-profile hot path."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import wd_spectra.opacity as opacity_module
from wd_spectra import gray_hydrogen_atmosphere
from wd_spectra.opacity import BALMER_LINES, balmer_mass_absorption_coefficient


def timed_balmer_opacity(*, full: bool, python_reference: bool) -> dict[str, object]:
    """Time a fixed, bundled-data Balmer-opacity calculation."""

    if full:
        wavelength = np.unique(
            np.concatenate(
                (
                    np.geomspace(900.0, 30_000.0, 2500),
                    np.arange(3400.0, 7500.001, 0.25),
                )
            )
        )
        atmosphere = gray_hydrogen_atmosphere(20_000.0, 8.0, n_depth=40)
        lines = BALMER_LINES
        quadrature_order = 128
    else:
        wavelength = np.linspace(3800.0, 7000.0, 2001)
        atmosphere = gray_hydrogen_atmosphere(20_000.0, 8.0, n_depth=12)
        lines = BALMER_LINES[:3]
        quadrature_order = 32

    compiled = opacity_module._rt
    if python_reference:
        opacity_module._rt = None
    try:
        start = time.perf_counter()
        opacity = balmer_mass_absorption_coefficient(
            atmosphere,
            wavelength,
            lines=lines,
            self_broadening_quadrature_order=quadrature_order,
            self_broadening_truncation_closure="stark-core",
        )
        elapsed = time.perf_counter() - start
    finally:
        opacity_module._rt = compiled
    return {
        "mode": "full" if full else "short",
        "backend": "python-reference" if python_reference else "auto",
        "elapsed_seconds": elapsed,
        "wavelength_points": int(wavelength.size),
        "depth_points": int(atmosphere.n_depth),
        "balmer_lines": len(lines),
        "quadrature_order": quadrature_order,
        "openwd_num_threads": os.environ.get("OPENWD_NUM_THREADS", "automatic"),
        "opacity_sum": float(np.sum(opacity)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--python-reference", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            timed_balmer_opacity(
                full=args.full,
                python_reference=args.python_reference,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
