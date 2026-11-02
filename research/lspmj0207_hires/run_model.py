#!/usr/bin/env python3
"""OpenWD DAZ model of LSPM J0207+3331 at the Le Bourdais & Dufour (2025) parameters.

Parameters (ApJ, arXiv:2510.08676, Tables 2-3), nothing fitted:
  Teff = 5910 K, log g = 8.11, hydrogen host (log He/H < -1; pure H adopted).
  Detected abundances, log10 N(Z)/N(H). Sr (-10.85) is included only with
  --with-sr. Upper limits (He, Li, C, O, K, V) are omitted.

Mirrors research/cool_dz_paper/run_model.py of the OpenWD checkout: default
DAZConfig physics, cold start, compiled transfer kernel required.  The spectrum
is saved on a 0.02 A grid over 3000-9000 A (vacuum wavelengths, surface F_lambda).

usage: python run_model.py OUTDIR [--quality standard] [--threads 8] [--structure-lines N] [--formal-lines N]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("outdir", type=Path)
parser.add_argument("--quality", default="standard")
parser.add_argument("--threads", default="8")
parser.add_argument("--structure-lines", type=int, default=None,
                    help="DAZConfig.structure_maximum_metal_lines (default: synthesis list, 20,000)")
parser.add_argument("--formal-lines", type=int, default=None,
                    help="DAZConfig.formal_maximum_metal_lines (default 20,000)")
parser.add_argument("--with-sr", action="store_true", help="include Sr at the paper abundance")
parser.add_argument("--strong-line-data", default="stout", choices=["stout", "nist-asd"],
                    help="DAZConfig.strong_line_atomic_data")
args = parser.parse_args()

os.environ["OPENWD_NUM_THREADS"] = args.threads
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np  # noqa: E402

import wd_spectra  # noqa: E402
import wd_spectra.metals  # noqa: E402
from wd_spectra import DAZConfig, compute_daz  # noqa: E402
from wd_spectra.models.common import default_wavelength_grid, save_model_result  # noqa: E402

assert wd_spectra.metals._rt is not None, "build the C kernel (wd_spectra._rt) before running"

TEFF = 5910.0
LOGG = 8.11
ABUNDANCES = {  # log10 N(Z)/N(H), Le Bourdais & Dufour (2025) Table 3, detections only
    "Na": -7.98, "Mg": -6.41, "Al": -7.28, "Si": -6.37, "Ca": -7.38, "Ti": -8.74,
    "Cr": -7.83, "Mn": -8.10, "Fe": -6.20, "Co": -8.44, "Ni": -7.48, "Cu": -9.21,
}
SR = -10.85  # Le Bourdais & Dufour (2025) Table 3; used with --with-sr


def main():
    out = args.outdir
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(wd_spectra.__file__).resolve().parents[2]
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    wavelength = np.unique(np.r_[
        default_wavelength_grid(900.0, 30000.0, continuum_points=2500, optical_step=1.0),
        np.arange(3000.0, 9000.0, 0.02)])
    abundances = dict(ABUNDANCES, **({"Sr": SR} if args.with_sr else {}))
    config = DAZConfig(effective_temperature=TEFF, logg=LOGG, abundances=abundances, quality=args.quality,
                       strong_line_atomic_data=args.strong_line_data,
                       structure_maximum_metal_lines=args.structure_lines,
                       formal_maximum_metal_lines=args.formal_lines)
    inputs = {"object": "LSPM J0207+3331", "teff": TEFF, "logg": LOGG, "host": "H",
              "abundances_log_n_z_over_n_h": abundances, "omitted": ({} if args.with_sr else {"Sr": SR}),
              "strong_line_atomic_data": args.strong_line_data,
              "source": "Le Bourdais & Dufour (2025), arXiv:2510.08676, Tables 2-3",
              "quality": args.quality, "structure_maximum_metal_lines": args.structure_lines,
              "formal_maximum_metal_lines": args.formal_lines, "openwd_version": wd_spectra.__version__,
              "openwd_commit": commit, "openwd_local_changes": bool(dirty),
              "openwd_num_threads": args.threads, "host_name": os.uname().nodename}
    (out / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    print(json.dumps(inputs), flush=True)
    start = time.time()

    def record(iteration, atmosphere, info):
        values = {k: v for k, v in dict(info).items() if isinstance(v, (int, float, str, bool)) or v is None}
        line = json.dumps({"iteration": int(iteration), "wall_s": round(time.time() - start, 1), **values}, default=str)
        with open(out / "iterations.jsonl", "a") as handle:
            handle.write(line + "\n")
        print(line[:300], flush=True)

    result = compute_daz(config, wavelength, iteration_callback=record)
    save_model_result(result, out)
    np.savez_compressed(out / "spectrum_fine.npz", wavelength=result.spectrum.wavelength_angstrom,
                        flux=result.spectrum.surface_flux_lambda,
                        continuum=result.spectrum.continuum_flux_lambda)
    certificate = result.atmosphere.metadata.get("equilibrium_certificate", {})
    summary = {"wall_s": round(time.time() - start, 1),
               "verified": bool(certificate.get("verified")) if isinstance(certificate, dict) else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if not summary["verified"]:
        sys.exit("convergence certificate NOT verified")


if __name__ == "__main__":
    main()
