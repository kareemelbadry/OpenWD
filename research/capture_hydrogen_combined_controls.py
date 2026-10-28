"""Offline fixed-state capture of the combined PR #52 and PR #54 physics.

Every case first computes the intended combination, then independently
recovers the immutable PR #54, standalone PR #52, and pre-both spectra at
their existing tolerances. All structures are freshly reloaded inside the
appropriate research scope. No atmosphere is iterated, no flux is fitted or
rescaled, and no historical reference is overwritten. Tests never run this
script. These outputs do not certify cold convergence or observational truth.

PYTHONPATH=src python research/capture_hydrogen_combined_controls.py OUTPUT --jobs 4
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

import numpy as np
import scipy

from wd_spectra import compiled_backend_available
from wd_spectra.models import AtmosphereConvergenceWarning
from wd_spectra.models.common import ModelData, model_request_fingerprint
from capture_hydrogen_dissolution_controls import cases, difference
from hydrogen_dissolution_control_legacy import previous_hydrogen_dissolution_cutoff
from montreal_h_radius_control_legacy import previous_hydrogen_neutral_defaults
from hydrogen_integration_control_support import ROOT, DISSOLUTION, main_control_path

sys.path.insert(0, str(ROOT / "tests"))
from test_da_public_spectrum import assert_spectrum_preserved


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    paths = sorted((ROOT / "src/wd_spectra").rglob("*.py"))
    paths += sorted((ROOT / "src/wd_spectra").glob("_rt*.so"))
    paths += [ROOT / name for name in (
        "research/capture_hydrogen_combined_controls.py",
        "research/capture_hydrogen_dissolution_controls.py",
        "research/hydrogen_dissolution_control_legacy.py",
        "research/montreal_h_radius_control_legacy.py",
        "research/hydrogen_integration_control_support.py",
        "tests/test_da_public_spectrum.py",
    )]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return fingerprint, hashes


def read_reference(path, wavelength):
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(wavelength, saved["wavelength"])
        flux = next(saved[k].copy() for k in ("surface_flux", "original_surface_flux", "flux")
                    if k in saved.files)
    return flux


def assert_recovered(name, case, flux, expected):
    if name.startswith("public-"):
        assert_spectrum_preserved(case["wavelength"], flux, expected)
    elif name.startswith("daz-"):
        take = case["wavelength"] * expected > 0.01 * np.max(case["wavelength"] * expected)
        np.testing.assert_allclose(flux[take], expected[take], rtol=1e-3, atol=0.0)
    elif name.startswith("dah-"):
        np.testing.assert_allclose(flux, expected, rtol=2e-8, atol=0.0)
    else:
        np.testing.assert_allclose(flux, expected, rtol=2e-6, atol=1e-12 * np.max(expected))


def capture_one(arguments):
    name, output, expected_identity = arguments
    output = Path(output)
    assert compiled_backend_available(), "compiled acceleration is required"
    assert source_identity()[0] == expected_identity, "capture inputs changed"
    record_path = output / "records" / (name + ".json")
    if record_path.exists():
        record = json.loads(record_path.read_text())
        assert record["source_identity_sha256"] == expected_identity
        assert digest(output / record["fixture"]) == record["fixture_sha256"]
        for p, value in record["input_sha256"].items():
            assert digest(ROOT / p) == value, p
        return name, record

    started = time.monotonic()
    build = dict(cases())[name]
    spectra, recoveries, inputs = {}, {}, {}
    current_case = None
    for phase in ("combined", "main54", "pr52", "pre_both"):
        print(f"{name}: {phase} START", flush=True)
        with ExitStack() as stack:
            if phase in {"pr52", "pre_both"}:
                stack.enter_context(previous_hydrogen_neutral_defaults())
            if phase in {"main54", "pre_both"}:
                stack.enter_context(previous_hydrogen_dissolution_cutoff())
            # Radius/default changes also affect the checkpoint's reconstructed EOS.
            case = build()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AtmosphereConvergenceWarning)
                result = case["run"]()
            flux = result.spectrum.surface_flux_lambda.copy()
            assert np.all(np.isfinite(flux)) and np.all(flux > 0.0)
            spectra[phase] = flux
            if phase == "combined":
                current_case = case
            else:
                reference = (main_control_path(name) if phase == "main54" else
                             DISSOLUTION / "fixed" / (name + ".npz") if phase == "pr52" else
                             case["previous"])
                expected = read_reference(reference, case["wavelength"])
                assert_recovered(name, case, flux, expected)
                recoveries[phase] = dict(
                    reference=str(reference.relative_to(ROOT)),
                    reference_sha256=digest(reference),
                    difference=difference(case["wavelength"], flux, expected),
                    tolerance=case["rtol"], passed=True,
                )
                inputs[str(reference.relative_to(ROOT))] = digest(reference)
        print(f"{name}: {phase} PASSED ({time.monotonic() - started:.1f}s)", flush=True)

    assert source_identity()[0] == expected_identity, "capture inputs changed"
    inputs[str(current_case["structure"].relative_to(ROOT))] = digest(current_case["structure"])
    fixture = Path("fixed") / (name + ".npz")
    if (output / fixture).exists():
        raise FileExistsError(f"unrecorded capture output already exists: {fixture}")
    np.savez_compressed(output / fixture, wavelength=current_case["wavelength"],
                        surface_flux=spectra["combined"])
    config = current_case["config"]
    record = dict(
        fixture=str(fixture), fixture_sha256=digest(output / fixture),
        source_identity_sha256=expected_identity, input_sha256=inputs,
        spectral_type=current_case["kind"],
        structure_input=str(current_case["structure"].relative_to(ROOT)),
        config=json.loads(json.dumps(asdict(config), default=list)),
        request_fingerprint=model_request_fingerprint(current_case["kind"], config, ModelData.default()),
        recoveries=recoveries,
        change_from_main=difference(current_case["wavelength"], spectra["combined"], spectra["main54"]),
        change_from_pr52=difference(current_case["wavelength"], spectra["combined"], spectra["pr52"]),
        nonadditive_interaction=difference(
            current_case["wavelength"],
            spectra["combined"] - spectra["main54"] - spectra["pr52"] + 2.0 * spectra["pre_both"],
            spectra["pre_both"],
        ),
        elapsed_seconds=time.monotonic() - started,
        tolerances_changed=False, atmosphere_reconverged=False, cold_start_qualification=False,
    )
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return name, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    names = list(dict(cases()))
    if args.case:
        if set(args.case) - set(names):
            parser.error("unknown capture case")
        names = [n for n in names if n in args.case]
    for group in ("fixed", "records"):
        (args.output / group).mkdir(parents=True, exist_ok=True)
    identity, sources = source_identity()
    records = {}
    # A fresh interpreter per case prevents research-scope state crossing cases.
    context = multiprocessing.get_context("spawn")
    with context.Pool(args.jobs, maxtasksperchild=1) as pool:
        arguments = [(name, str(args.output.resolve()), identity) for name in names]
        for name, record in pool.imap_unordered(capture_one, arguments):
            records[name] = record
            print(f"{name}: RECORDED {len(records)}/{len(names)}", flush=True)
    assert source_identity()[0] == identity, "capture inputs changed"
    manifest = dict(
        description=__doc__, date="2026-10-28", source_identity_sha256=identity,
        reference_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256=sources, python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        combined_pr52_head="a10c59f6c0a110a140d143b1ec51f0ce7852c9e7",
        combined_main="175d5cbdf57a418944c7b9e5bd9b6c7812b43e41",
        cases={name: records[name] for name in names},
    )
    suffix = "manifest.json" if len(names) == len(dict(cases())) else "manifest.partial.json"
    (args.output / suffix).write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
