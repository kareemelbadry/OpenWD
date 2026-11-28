"""Capture the dab-20000 cold-start reference after the mixed H/He IR changes.

Two changes of 2026-11-27 move the DAB 20000/8 cold start: the infrared He I
lines (wd_spectra.helium_impact) in its structure and spectrum, and the
Paschen/Brackett wing sampling of the mixed H/He structure. RUN is the output
directory of ``research/validate_public_cold_start.py dab-20000 RUN`` (the CI
command) from a clean tree; its certificate must be verified. The flux, gas
pressure and column mass of that fresh model become the new reference.

The manifest separates the parts: the same structure re-synthesized without
the infrared He I lines (line opacity alone), and, if CANDIDATE is given, the
CI model of the branch before the sampling fix (its validation directory).

The script refuses an existing output directory and is never run by tests or
CI. Copy the written files into
tests/data/approved_regressions/paschen_brackett_wings_2026_11_27/cold/.

usage: python research/capture_paschen_brackett_cold_controls.py OUTDIR RUN [CANDIDATE]
"""
from __future__ import annotations

import functools
import hashlib
import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import wd_spectra.metals as _metals  # noqa: E402
import wd_spectra.models.stellar as _stellar  # noqa: E402
import wd_spectra.spectrum as _spectrum  # noqa: E402
from wd_spectra import DABConfig, compute_dab  # noqa: E402
from wd_spectra._compat import trapezoid  # noqa: E402
from wd_spectra.models.common import (  # noqa: E402
    AtmosphereConvergenceWarning, load_atmosphere_checkpoint)

PREVIOUS = ROOT / "tests/data/approved_regressions/cold/dab-20000.npz"
SOURCES = (
    "src/wd_spectra/atmosphere.py",
    "src/wd_spectra/helium_impact.py",
    "src/wd_spectra/spectrum.py",
    "src/wd_spectra/models/stellar.py",
    "research/validate_public_cold_start.py",
    "research/capture_paschen_brackett_cold_controls.py",
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


def model(directory: Path):
    validation = json.loads((directory / "validation.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    values = np.loadtxt(directory / "spectrum.txt")
    with np.load(directory / "atmosphere.npz") as saved:
        nodes = {name: saved[name] for name in ("gas_pressure", "column_mass", "temperature")}
    return validation, metadata, values[:, 0], values[:, 1], nodes


def changes(wave, new, old, new_nodes, old_nodes):
    n = min(len(new_nodes["gas_pressure"]), len(old_nodes["gas_pressure"]))
    return dict(
        maximum_relative_flux_change=float(np.max(np.abs(new / old - 1.0))),
        wavelength_of_maximum_angstrom=float(wave[np.argmax(np.abs(new / old - 1.0))]),
        integrated_relative_change=bands(wave, new, old),
        maximum_gas_pressure_change=float(np.max(np.abs(
            new_nodes["gas_pressure"][:n] / old_nodes["gas_pressure"][:n] - 1.0))),
        maximum_column_mass_change=float(np.max(np.abs(
            new_nodes["column_mass"][:n] / old_nodes["column_mass"][:n] - 1.0))),
    )


def main() -> None:
    assert _metals._rt is not None, "compiled _rt kernel not loaded"
    out = Path(sys.argv[1]).resolve()
    run = Path(sys.argv[2]).resolve()
    candidate = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else None
    out.mkdir(parents=True, exist_ok=False)
    validation, metadata, wave, flux, nodes = model(run)
    assert validation["cold_start"] is True and validation["certificate"]["verified"] is True
    assert metadata["model_metadata"]["atmosphere_convergence_status"] == "converged"
    # The validator's DAB 20000 request: DABConfig(effective_temperature=20000, quality="production").
    config = DABConfig(effective_temperature=20000.0, quality="production")
    assert metadata["spectral_type"] == "DAB"
    for name in ("effective_temperature", "logg", "log_hydrogen_to_helium", "quality"):
        assert metadata["config"][name] == getattr(config, name), name
    with np.load(PREVIOUS) as previous:
        old = {name: previous[name] for name in previous.files}
    # spectrum.txt round-trips the wavelengths to the last bit; keep the exact grid
    # (it equals the validator's requested grid).
    np.testing.assert_allclose(wave, old["wavelength"], rtol=1e-14, atol=0)
    wave = old["wavelength"]

    # Same structure, with and without the infrared He I lines in the spectrum.
    # As research/capture_he_i_infrared_controls.py reloads a DAB structure.
    atmosphere = load_atmosphere_checkpoint(
        run / "atmosphere.npz", config.effective_temperature, config.logg, "mixed",
        include_molecules=False, include_negative_hydrogen=False,
        log_hydrogen_to_helium=config.log_hydrogen_to_helium,
        trihydrogen_ion_partition_model=None,
    )
    original, original_stellar = _spectrum.synthesize_helium_spectrum, _stellar.synthesize_helium_spectrum

    def synthesis(include_lines):
        switched = functools.partial(original, include_helium_i_impact_lines=include_lines)
        _spectrum.synthesize_helium_spectrum = _stellar.synthesize_helium_spectrum = switched
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AtmosphereConvergenceWarning)
                result = compute_dab(config, wave, initial_atmosphere=atmosphere,
                                     relax_atmosphere=False)
        finally:
            _spectrum.synthesize_helium_spectrum = original
            _stellar.synthesize_helium_spectrum = original_stellar
        assert result.spectrum.metadata["source_converged"]
        return result.spectrum.surface_flux_lambda

    with_lines, without_lines = synthesis(True), synthesis(False)
    resynthesis_error = float(np.max(np.abs(with_lines / flux - 1.0)))
    assert resynthesis_error < 1e-6, resynthesis_error

    target = out / "dab-20000.npz"
    np.savez_compressed(target, wavelength=wave, surface_flux=flux,
                        gas_pressure=nodes["gas_pressure"], column_mass=nodes["column_mass"])
    record = dict(
        previous_control=str(PREVIOUS.relative_to(ROOT)),
        previous_control_sha256=sha256(PREVIOUS),
        new_control_sha256=sha256(target),
        local_run=str(run),
        local_run_validation=validation,
        resynthesis_reproduces_run=resynthesis_error,
        change_from_previous=changes(wave, flux, old["surface_flux"], nodes, old),
        infrared_he_i_line_opacity_alone=bands(wave, with_lines, without_lines),
        structure_change_without_infrared_he_i_spectrum_lines=bands(
            wave, without_lines, old["surface_flux"]),
    )
    if candidate is not None:
        cand_validation, _, cand_wave, cand_flux, cand_nodes = model(candidate)
        np.testing.assert_allclose(cand_wave, wave, rtol=1e-14, atol=0)
        record["branch_before_sampling_fix"] = dict(
            source=str(candidate),
            validation_bands=cand_validation["bands"],
            change_from_previous=changes(wave, cand_flux, old["surface_flux"], cand_nodes, old),
            sampling_fix_change=changes(wave, flux, cand_flux, nodes, cand_nodes),
            maximum_temperature_change_from_sampling_fix=float(np.max(np.abs(
                nodes["temperature"] / cand_nodes["temperature"] - 1.0)))
            if nodes["temperature"].shape == cand_nodes["temperature"].shape else None,
        )
    manifest = dict(
        reason=("infrared He I lines in the DAB structure and spectrum, and Paschen/Brackett "
                "wing sampling (+-300 A) of the mixed H/He structure"),
        capture="local verified cold start by research/validate_public_cold_start.py (CI command)",
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                  text=True).stdout.strip(),
        source_changes=subprocess.run(["git", "diff", "--name-only"], cwd=ROOT, capture_output=True,
                                      text=True).stdout.split(),
        source_sha256={name: sha256(ROOT / name) for name in SOURCES},
        platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
        cases={"dab-20000": record},
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(record, indent=1), flush=True)


if __name__ == "__main__":
    main()
