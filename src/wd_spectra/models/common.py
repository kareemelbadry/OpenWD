"""Shared public interfaces for one-shot white-dwarf models.

The low-level :mod:`wd_spectra` modules intentionally expose individual EOS,
opacity, transfer, and atmosphere operations.  This module defines the small
stable surface used by the spectral-type presets: a data manifest, a quality
level, and a uniform result that can be saved without importing validation
scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..atmosphere import Atmosphere
from ..eos import (
    hummer_mihalas_helium_lte,
    hummer_mihalas_hydrogen_helium_lte,
    hummer_mihalas_hydrogen_lte,
)
from ..spectrum import Spectrum
FloatArray = NDArray[np.float64]
Quality = Literal["quick", "standard", "production"]
AtmosphereComposition = Literal["hydrogen", "helium", "mixed"]
ConvergenceStatus = Literal["converged", "unconverged", "unknown"]

_MODEL_REQUEST_FINGERPRINT_SCHEMA = 1
_MODEL_PHYSICS_REVISION = "openwd-0.1.3-solver-telemetry"


class AtmosphereConvergenceWarning(RuntimeWarning):
    """A spectrum is being returned from an unverified atmosphere."""


@dataclass(frozen=True)
class NumericalResolution:
    """Numerical work budget associated with a public quality level."""

    n_depth: int
    n_continuum: int
    maximum_iterations: int
    n_angle: int


_RESOLUTION = {
    "quick": NumericalResolution(20, 80, 2, 2),
    "standard": NumericalResolution(40, 300, 120, 3),
    "production": NumericalResolution(80, 600, 500, 4),
}


def numerical_resolution(quality: Quality) -> NumericalResolution:
    """Return numerical settings without altering the requested physics."""

    try:
        return _RESOLUTION[quality]
    except KeyError as exc:  # pragma: no cover - protects untyped callers
        raise ValueError(f"unknown quality {quality!r}") from exc


@dataclass(frozen=True)
class ModelData:
    """Locations of the external atomic/profile data used by model presets.

    ``root`` is the repository or installed data workspace.  Set the
    ``WD_SPECTRA_DATA`` environment variable when calling the package away
    from the source checkout.
    """

    root: Path

    @classmethod
    def default(cls, root: str | Path | None = None) -> "ModelData":
        if root is None:
            root = os.environ.get("OPENWD_DATA")
        if root is None:
            root = files("wd_spectra").joinpath("data/runtime")
        return cls(Path(root).expanduser().resolve())

    @property
    def cache(self) -> Path:
        bundled = self.root / "cache"
        return bundled if bundled.is_dir() else self.root / ".cache"

    @property
    def allard_lyman(self) -> Path:
        return self.root / "allard_data"

    @property
    def allard_tlusty205(self) -> Path:
        """Checksum-pinned fixed Allard tables distributed with TLUSTY205."""

        return self.cache / "allard-tlusty205"

    @property
    def helium_i_stark(self) -> Path:
        return self.cache / "helium-stark/Beauchamp25_LD.txt"

    @property
    def helium_ii_stark(self) -> Path:
        return self.cache / "helium-stark/he2prf.dat"

    @property
    def stout(self) -> Path:
        return self.cache / "metal-opacity/atomic-line-list"

    @property
    def verner_photoionization(self) -> Path:
        return self.cache / "metal-opacity/verner-photoionization.dat"

    @property
    def barklem_neutral_h_broadening(self) -> Path:
        """ABO line-by-line neutral-H widths for metal transitions."""

        return self.cache / "metal-opacity/barklem-neutral-h-broadening.dat"

    @property
    def mg_he_red_wing(self) -> Path:
        return self.cache / "metal-opacity/mg-he-2852-red-wing.dat"

    @property
    def mg_i_he_density_profiles(self) -> Path:
        return (
            self.cache
            / "metal-opacity/unified-metal-he/mgi-2852-density-6000K.dat"
        )

    @property
    def ca_i_he_density_profiles(self) -> Path:
        return (
            self.cache
            / "metal-opacity/unified-metal-he/cai-4227-density-4000K.dat"
        )

    @property
    def ca_i_he_temperature_profiles(self) -> Path:
        return (
            self.cache
            / "metal-opacity/unified-metal-he/cai-4227-temperature-nhe5e21.dat"
        )

    @property
    def ca_ii_chianti_collisions(self) -> Path:
        """CHIANTI Ca II effective electron collision strengths."""

        return self.cache / "chianti/ca_2/ca_2.scups"

    @property
    def helium_reos3(self) -> Path:
        """Becker et al. (2014) ab-initio helium EOS table."""

        return self.cache / "dense-eos/he-reos3-table3.dat"

    @property
    def nist_asd_strong(self) -> Path:
        """Checksum-pinned NIST ASD strong-line query directory."""

        return self.cache / "nist-asd-strong"

    @property
    def h2_h2_cia(self) -> Path:
        return (
            self.cache
            / "molecular-opacity/CIA_Borysow_H2H2_0060-7000K_0.6-500um.dat"
        )

    def require(self, *paths: Path, fetch_command: str | None = None) -> None:
        missing = [path for path in paths if not path.is_file() and not path.is_dir()]
        if not missing:
            return
        detail = "\n".join(f"  - {path}" for path in missing)
        hint = (
            "\nReinstall OpenWD to restore the bundled data, or point "
            "OPENWD_DATA/ModelData at a complete external data directory."
        )
        raise FileNotFoundError(f"required model data are missing:\n{detail}{hint}")


@dataclass(frozen=True)
class ModelResult:
    """Atmosphere, rest-frame spectrum, and provenance from one calculation."""

    spectral_type: str
    atmosphere: Atmosphere
    spectrum: Spectrum
    config: object
    metadata: Mapping[str, Any]
    population_state: object | None = None


def model_request_fingerprint(
    spectral_type: str,
    config: object,
    data: ModelData,
) -> dict[str, object]:
    """Return a stable identity for one solver-relevant public request.

    The data root is deliberately part of the identity.  Moving a checkpoint
    to another data installation therefore degrades it to a warm start rather
    than claiming an exact same-physics resume.  The physics revision must be
    changed whenever solver equations or bundled physical data change.
    """

    request = {
        "schema": _MODEL_REQUEST_FINGERPRINT_SCHEMA,
        "physics_revision": _MODEL_PHYSICS_REVISION,
        "spectral_type": str(spectral_type),
        "config": _jsonable(config),
        "data_root": str(data.root.resolve()),
    }
    serialized = json.dumps(
        request,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        **request,
        "sha256": hashlib.sha256(serialized).hexdigest(),
    }


def atmosphere_matches_model_request(
    atmosphere: Atmosphere | None,
    fingerprint: Mapping[str, object],
) -> bool:
    """Return whether an atmosphere proves an exact request identity."""

    if atmosphere is None:
        return False
    recorded = atmosphere.metadata.get("model_request_fingerprint")
    if not isinstance(recorded, Mapping):
        return False
    return dict(recorded) == dict(fingerprint)


def atmosphere_with_model_request_fingerprint(
    atmosphere: Atmosphere,
    fingerprint: Mapping[str, object],
) -> Atmosphere:
    """Attach immutable request provenance without mutating caller metadata."""

    return replace(
        atmosphere,
        metadata={
            **atmosphere.metadata,
            "model_request_fingerprint": dict(fingerprint),
        },
    )


def atmosphere_convergence_status(
    atmosphere: Atmosphere,
) -> ConvergenceStatus:
    """Classify whether a structure records successful self-consistency."""

    if atmosphere.metadata.get(
        "mean_3d_temperature_differential_is_equilibrium_model"
    ) is False:
        return "unconverged"
    value = atmosphere.metadata.get("radiative_equilibrium_converged")
    if isinstance(value, (bool, np.bool_)):
        return "converged" if bool(value) else "unconverged"
    return "unknown"


def warn_if_atmosphere_not_converged(
    atmosphere: Atmosphere,
    spectral_type: str,
) -> ConvergenceStatus:
    """Warn, but do not prevent exploratory synthesis from a partial model."""

    status = atmosphere_convergence_status(atmosphere)
    if status == "converged":
        return status
    if status == "unconverged":
        detail = "records that radiative/convective equilibrium did not converge"
    else:
        detail = "does not record a verified atmosphere-convergence status"
    metrics = []
    residual = atmosphere.metadata.get(
        "maximum_all_depth_total_flux_residual",
        atmosphere.metadata.get("maximum_total_flux_residual"),
    )
    if isinstance(
        residual, (int, float, np.integer, np.floating)
    ) and np.isfinite(residual):
        metrics.append(f"maximum all-depth flux residual={float(residual):.3e}")
    correction = atmosphere.metadata.get(
        "radiative_equilibrium_maximum_log_temperature_correction"
    )
    if isinstance(
        correction, (int, float, np.integer, np.floating)
    ) and np.isfinite(correction):
        metrics.append(
            "maximum log-temperature correction="
            f"{float(correction):.3e}"
        )
    terminal_reason = atmosphere.metadata.get(
        "nonlinear_solver_terminal_reason"
    )
    if isinstance(terminal_reason, str):
        metrics.append(f"nonlinear terminal reason={terminal_reason}")
    metric_text = f" ({', '.join(metrics)})" if metrics else ""
    warnings.warn(
        f"Returning a {spectral_type} spectrum from an atmosphere that {detail}"
        f"{metric_text}. The result is suitable for explicit exploratory work, "
        "but must not be treated as a converged model.",
        AtmosphereConvergenceWarning,
        stacklevel=3,
    )
    return status


def default_wavelength_grid(
    lower: float = 900.0,
    upper: float = 30_000.0,
    *,
    continuum_points: int = 2500,
    optical_step: float = 0.25,
) -> FloatArray:
    """Broad UV--IR grid with a resolved optical line region."""

    if not 0.0 < lower < upper:
        raise ValueError("wavelength bounds must be positive and increasing")
    broad = np.geomspace(lower, upper, continuum_points)
    optical_lower = max(lower, 3400.0)
    optical_upper = min(upper, 7500.0)
    optical = (
        np.arange(optical_lower, optical_upper + 0.5 * optical_step, optical_step)
        if optical_lower < optical_upper
        else np.empty(0)
    )
    return np.unique(np.r_[broad, optical])


def validate_wavelength(wavelength: ArrayLike | None) -> FloatArray:
    result = default_wavelength_grid() if wavelength is None else np.asarray(
        wavelength, dtype=np.float64
    )
    if result.ndim != 1 or result.size < 2:
        raise ValueError("wavelength must be a one-dimensional array")
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("wavelength must contain finite positive values")
    if np.any(np.diff(result) <= 0.0):
        raise ValueError("wavelength must increase strictly")
    return np.ascontiguousarray(result)


def load_atmosphere_checkpoint(
    path: str | Path,
    effective_temperature: float,
    logg: float,
    composition: AtmosphereComposition,
    *,
    log_hydrogen_to_helium: float | None = None,
    include_molecules: bool = False,
    include_negative_hydrogen: bool = False,
    trihydrogen_ion_partition_model: str | None = None,
) -> Atmosphere:
    """Restore a validation/one-shot atmosphere from the common NPZ schemas.

    The historical validation files differ mainly in whether their thermal
    arrays carry a ``thermal_`` prefix.  EOS populations are always rebuilt
    from the requested current composition rather than trusted from a stale
    checkpoint.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"atmosphere checkpoint does not exist: {source}")
    stored_metadata: dict[str, object] = {}
    with np.load(source) as saved:
        def required(*names: str) -> FloatArray:
            for name in names:
                if name in saved:
                    return np.asarray(saved[name], dtype=np.float64)
            raise KeyError(
                f"{source} has none of the required arrays {', '.join(names)}"
            )

        temperature = required("thermal_temperature", "temperature", "T")
        pressure = required("thermal_gas_pressure", "gas_pressure", "P")
        column_mass = required("column_mass", "m")
        if "rosseland_optical_depth" in saved:
            optical_depth = np.asarray(
                saved["rosseland_optical_depth"], dtype=np.float64
            )
        elif "tau" in saved:
            optical_depth = np.asarray(saved["tau"], dtype=np.float64)
        else:
            optical_depth = np.geomspace(1.0e-8, 1.0e2, temperature.size)
        saved_mass_density = (
            np.asarray(saved["thermal_mass_density"], dtype=np.float64)
            if "thermal_mass_density" in saved
            else np.asarray(saved["mass_density"], dtype=np.float64)
            if "mass_density" in saved else None
        )
        saved_electron_density = (
            np.asarray(saved["thermal_electron_density"], dtype=np.float64)
            if "thermal_electron_density" in saved
            else np.asarray(saved["electron_density"], dtype=np.float64)
            if "electron_density" in saved else None
        )
        if "atmosphere_metadata_json" in saved:
            try:
                serialized_metadata = str(
                    np.asarray(saved["atmosphere_metadata_json"]).item()
                )
                parsed_metadata = json.loads(serialized_metadata)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{source} contains invalid atmosphere metadata"
                ) from exc
            if not isinstance(parsed_metadata, dict):
                raise ValueError(
                    f"{source} atmosphere metadata must be a JSON object"
                )
            stored_metadata = parsed_metadata
    metadata = {
        **stored_metadata,
        "model": "restored one-shot atmosphere",
        "source_checkpoint": str(source.resolve()),
        "checkpoint_composition": composition,
    }
    zeros = np.zeros_like(temperature)
    if composition == "hydrogen":
        state = hummer_mihalas_hydrogen_lte(
            temperature,
            pressure,
            correlated_microfields=True,
            include_molecules=include_molecules,
            include_negative_hydrogen=include_negative_hydrogen,
            trihydrogen_ion_partition_model=(
                trihydrogen_ion_partition_model
            ),
        )
        return Atmosphere(
            effective_temperature,
            logg,
            optical_depth,
            column_mass,
            temperature,
            pressure,
            state.mass_density,
            state.neutral_h_density,
            state.proton_density,
            state.electron_density,
            metadata,
            hydrogen_lte_state=state,
        )
    if composition == "helium":
        state = hummer_mihalas_helium_lte(
            temperature,
            pressure,
            neutral_radius_scale=0.5,
            correlated_microfields=True,
        )
        return Atmosphere(
            effective_temperature,
            logg,
            optical_depth,
            column_mass,
            temperature,
            pressure,
            state.mass_density,
            zeros,
            zeros,
            state.electron_density,
            metadata,
            helium_lte_state=state,
        )
    if composition == "mixed":
        if log_hydrogen_to_helium is None:
            raise ValueError("mixed checkpoints require log_hydrogen_to_helium")
        state = hummer_mihalas_hydrogen_helium_lte(
            temperature,
            pressure,
            log_hydrogen_to_helium,
            hydrogen_neutral_radius_scale=0.5,
            helium_neutral_radius_scale=0.5,
            correlated_microfields=True,
        )
        hydrogen = state.hydrogen_lte_state
        return Atmosphere(
            effective_temperature,
            logg,
            optical_depth,
            column_mass,
            temperature,
            pressure,
            state.mass_density,
            hydrogen.neutral_h_density,
            hydrogen.proton_density,
            state.electron_density,
            metadata,
            hydrogen_lte_state=hydrogen,
            helium_lte_state=state.helium_lte_state,
        )
    raise ValueError(f"unknown atmosphere composition {composition!r}")


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return {
            item.name: _jsonable(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def save_model_result(result: ModelResult, output: str | Path) -> Path:
    """Write the common one-shot artifact set and return its directory."""

    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    spectrum = result.spectrum
    atmosphere = result.atmosphere
    np.savetxt(
        directory / "spectrum.txt",
        np.column_stack((spectrum.wavelength_angstrom, spectrum.surface_flux_lambda)),
        header="vacuum_wavelength_angstrom surface_flux_erg_s-1_cm-2_A-1",
    )
    np.savez_compressed(
        directory / "atmosphere.npz",
        effective_temperature=atmosphere.effective_temperature,
        logg=atmosphere.logg,
        rosseland_optical_depth=atmosphere.rosseland_optical_depth,
        column_mass=atmosphere.column_mass,
        temperature=atmosphere.temperature,
        gas_pressure=atmosphere.gas_pressure,
        mass_density=atmosphere.mass_density,
        electron_density=atmosphere.electron_density,
        atmosphere_metadata_json=np.asarray(
            json.dumps(
                _jsonable(atmosphere.metadata),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
    )
    record = {
        "schema": 1,
        "spectral_type": result.spectral_type,
        "config": _jsonable(result.config),
        "model_metadata": _jsonable(result.metadata),
        "atmosphere_metadata": _jsonable(atmosphere.metadata),
        "spectrum_metadata": _jsonable(spectrum.metadata),
    }
    (directory / "metadata.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return directory
