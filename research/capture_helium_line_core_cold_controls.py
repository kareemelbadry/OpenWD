"""Capture DB cold-canary references after resolving the LTE He line cores.

The LTE helium structure grid now resolves the He II and He I resonance Doppler
cores (``atmosphere._resolved_line_core_wavelengths``).  Each case reads the
model a CI cold-canary job saved as its failure candidate (artifact
``cold-<case>``: ``cold/<case>/diagnostics/<case>/{atmosphere.npz,
spectrum.txt, metadata.json}``).  It checks that the candidate's equilibrium
certificate is verified and that its spectrum is on the reference wavelength
grid, then stores its surface flux, gas pressure and column mass as the new
reference.  The manifest records the change from the previous reviewed control.

The script refuses an existing output directory and is never run by tests or
CI.  Copy the written files into
tests/data/approved_regressions/helium_line_cores_2026_11_27/cold/.

usage: python research/capture_helium_line_core_cold_controls.py OUTDIR RUN_URL \\
           CASE=ARTIFACT_DIR [CASE=ARTIFACT_DIR ...]
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
    ROOT / "tests/data/approved_regressions/he_i_infrared_2026_11_27/cold",
    ROOT / "tests/data/approved_regressions/heh_mixed_heminus_2026_11_19/cold",
    ROOT / "tests/data/approved_regressions/cold",
)
SOURCES = (
    "src/wd_spectra/atmosphere.py",
    "research/capture_helium_line_core_cold_controls.py",
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


def main(argv):
    out, run_url, *pairs = argv
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        reason="He II and He I resonance Doppler cores resolved in the LTE helium structure grid",
        capture="CI cold-canary failure candidate (Linux) with a verified equilibrium certificate",
        ci_run=run_url,
        git_commit=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip(),
        source_sha256={name: sha256(ROOT / name) for name in SOURCES},
        platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
        cases={},
    )
    for pair in pairs:
        case, directory = pair.split("=", 1)
        model = Path(directory) / "cold" / case / "diagnostics" / case
        metadata = json.loads((model / "metadata.json").read_text())
        certificate = metadata["atmosphere_metadata"]["equilibrium_certificate"]
        if not certificate.get("verified") or certificate.get("failures"):
            raise SystemExit(f"{case}: candidate certificate is not verified")
        previous = next(folder / f"{case}.npz" for folder in REVIEWED if (folder / f"{case}.npz").exists())
        spectrum = np.loadtxt(model / "spectrum.txt")
        with np.load(model / "atmosphere.npz") as atmosphere, np.load(previous) as old:
            # spectrum.txt is text: compare the grid to 1e-12, then keep the reference's own array.
            if spectrum.shape[0] != old["wavelength"].size or not np.allclose(
                    spectrum[:, 0], old["wavelength"], rtol=1e-12, atol=0.0):
                raise SystemExit(f"{case}: candidate spectrum is not on the reference grid")
            new = dict(wavelength=old["wavelength"].copy(), surface_flux=spectrum[:, 1].copy(),
                       gas_pressure=atmosphere["gas_pressure"].copy(),
                       column_mass=atmosphere["column_mass"].copy())
            change = dict(
                maximum_gas_pressure_change=float(np.max(np.abs(new["gas_pressure"][: old["gas_pressure"].size]
                                                                  / old["gas_pressure"] - 1))),
                maximum_column_mass_change=float(np.max(np.abs(new["column_mass"][: old["column_mass"].size]
                                                                 / old["column_mass"] - 1))),
                maximum_flux_change_below_7300_angstrom=float(np.max(np.abs(
                    (new["surface_flux"] / old["surface_flux"] - 1)[old["wavelength"] < 7300]))),
                integrated_relative_change=bands(old["wavelength"], new["surface_flux"], old["surface_flux"]),
            )
        target = out / f"{case}.npz"
        np.savez(target, **new)
        manifest["cases"][case] = dict(
            previous_control=str(previous.relative_to(ROOT)), previous_control_sha256=sha256(previous),
            new_control_sha256=sha256(target), candidate_atmosphere_sha256=sha256(model / "atmosphere.npz"),
            candidate_spectrum_sha256=sha256(model / "spectrum.txt"),
            n_depth=int(new["column_mass"].size), **change,
        )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
