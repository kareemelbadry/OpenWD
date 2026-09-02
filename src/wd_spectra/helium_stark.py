"""Stark-broadened neutral-helium line profiles.

The reader supports the CC BY 4.0 tables released with Tremblay et al.
(2026, ApJ 1000, 253; Zenodo 10.5281/zenodo.18722143).  The same layout is
used for their corrected Beauchamp semi-analytic profiles and new computer
simulations.  Profiles are normalized per Angstrom and already include
thermal Doppler convolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .stark import _bracket

try:  # Optional acceleration built by setup.py.
    from . import _rt
except ImportError:  # pragma: no cover - exercised in source-only installs
    _rt = None


FloatArray = NDArray[np.float64]
HELIUM_STARK_ZENODO_RECORD = "https://zenodo.org/records/18722143"
HELIUM_STARK_FILENAMES = (
    "Beauchamp25_LD.txt",
    "Beauchamp25_NLD.txt",
    "Tremblay26.txt",
)
HELIUM_STARK_SHA256 = {
    "Beauchamp25_LD.txt": "790da75f779434215afdfa3147d147c6c5117322577e658a613d04aed4a6ebfa",
    "Beauchamp25_NLD.txt": "a4650f8b0b9635f16dd32c7d69a61dda787bf6b49a148fae0c19e82b49f80f58",
    "Tremblay26.txt": "732547fc9aeb3581692145172901cc6502302bcd3611d2df907338d2473a82d0",
}

_TEMPERATURE = np.asarray([10_000.0, 20_000.0, 40_000.0])
_LOG_TEMPERATURE = np.log10(_TEMPERATURE)
_LOG_ELECTRON_DENSITY = np.concatenate(
    (np.arange(13.0, 18.0, 0.5), [np.log10(6.0e17)])
)
_BLOCK_HEADER = re.compile(
    r"^\s*(\d+)\s+LAMBDA\s+([0-9.]+)\s*$", re.IGNORECASE
)


@dataclass(frozen=True)
class HeliumStarkLine:
    """One asymmetric, area-normalized He I profile grid."""

    nominal_wavelength_angstrom: float
    wavelength_offset_angstrom: FloatArray
    log_electron_density: FloatArray
    log_temperature: FloatArray
    log_profile_per_angstrom: FloatArray

    def wavelength_profile(
        self,
        wavelength_angstrom: ArrayLike,
        line_center_angstrom: float,
        temperature: float,
        electron_density: float,
        *,
        lorentz_hwhm_angstrom: float = 0.0,
    ) -> FloatArray:
        """Interpolate the normalized profile at one thermodynamic state.

        A positive ``lorentz_hwhm_angstrom`` convolves the tabulated
        Stark-plus-Doppler profile with a Lorentzian.  This is how the
        Montreal calculations add neutral-particle and radiative broadening
        to the released tables.  The convolution is integrated on the native
        irregular table grid before evaluation at the requested wavelengths.
        """

        wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
        if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
            raise ValueError("wavelength must contain finite positive values")
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if not np.isfinite(electron_density) or electron_density <= 0.0:
            raise ValueError("electron_density must be finite and positive")
        if (
            not np.isfinite(lorentz_hwhm_angstrom)
            or lorentz_hwhm_angstrom < 0.0
        ):
            raise ValueError("lorentz_hwhm_angstrom must be finite and non-negative")

        log_ne = float(
            np.clip(
                np.log10(electron_density),
                self.log_electron_density[0],
                self.log_electron_density[-1],
            )
        )
        log_t = float(
            np.clip(
                np.log10(temperature),
                self.log_temperature[0],
                self.log_temperature[-1],
            )
        )
        ne_lower, ne_fraction = _bracket(self.log_electron_density, log_ne)
        t_lower, t_fraction = _bracket(self.log_temperature, log_t)
        local_log_profile = np.zeros_like(self.wavelength_offset_angstrom)
        for ne_index, ne_weight in (
            (ne_lower, 1.0 - ne_fraction),
            (ne_lower + 1, ne_fraction),
        ):
            for t_index, t_weight in (
                (t_lower, 1.0 - t_fraction),
                (t_lower + 1, t_fraction),
            ):
                local_log_profile += (
                    ne_weight
                    * t_weight
                    * self.log_profile_per_angstrom[ne_index, t_index]
                )
        local_profile = np.power(10.0, local_log_profile)
        normalization = np.trapz(local_profile, self.wavelength_offset_angstrom)
        if normalization > 0.0:
            local_profile /= normalization
        offset = wavelength - line_center_angstrom
        if lorentz_hwhm_angstrom > 0.0:
            native = self.wavelength_offset_angstrom
            if _rt is not None and hasattr(
                _rt, "piecewise_linear_lorentz_convolution"
            ):
                return np.asarray(
                    _rt.piecewise_linear_lorentz_convolution(
                        np.ascontiguousarray(native),
                        np.ascontiguousarray(local_profile),
                        np.ascontiguousarray(offset),
                        float(lorentz_hwhm_angstrom),
                    ),
                    dtype=np.float64,
                )
            spacing = np.diff(native)
            # Integrate the Lorentz convolution analytically over the
            # piecewise-linear representation of the native profile.  Direct
            # quadrature is badly conditioned when a narrow Lorentz kernel is
            # smaller than the irregular table spacing.
            left = native[:-1][np.newaxis, :] - offset[:, np.newaxis]
            right = native[1:][np.newaxis, :] - offset[:, np.newaxis]
            gamma = lorentz_hwhm_angstrom
            angle_integral = (
                np.arctan(right / gamma) - np.arctan(left / gamma)
            ) / np.pi
            first_moment = gamma / (2.0 * np.pi) * np.log(
                (right**2 + gamma**2) / (left**2 + gamma**2)
            )
            slope = np.diff(local_profile) / spacing
            segment = (
                local_profile[:-1][np.newaxis, :] * angle_integral
                + slope[np.newaxis, :]
                * (
                    first_moment
                    + (offset[:, np.newaxis] - native[:-1][np.newaxis, :])
                    * angle_integral
                )
            )
            return np.maximum(np.sum(segment, axis=1), 0.0)
        return np.interp(
            offset,
            self.wavelength_offset_angstrom,
            local_profile,
            left=0.0,
            right=0.0,
        )


@dataclass(frozen=True)
class HeliumStarkTable:
    """Collection of He I Stark profiles indexed by nominal wavelength."""

    lines: dict[int, HeliumStarkLine]
    source_path: Path

    def __getitem__(self, nominal_wavelength_angstrom: int) -> HeliumStarkLine:
        return self.lines[nominal_wavelength_angstrom]


def _read_numeric_values(
    text_lines: list[str], cursor: int, count: int, *, label: str
) -> tuple[FloatArray, int]:
    values: list[float] = []
    while len(values) < count:
        if cursor >= len(text_lines):
            raise ValueError(f"unexpected end of file while reading {label}")
        tokens = text_lines[cursor].split()
        cursor += 1
        try:
            values.extend(float(token) for token in tokens)
        except ValueError as error:
            raise ValueError(f"non-numeric value while reading {label}") from error
    if len(values) != count:
        raise ValueError(f"too many values in final line of {label}")
    return np.asarray(values, dtype=np.float64), cursor


def read_helium_stark_table(path: str | Path) -> HeliumStarkTable:
    """Read a Tremblay/Beauchamp 2025-2026 neutral-helium table."""

    path = Path(path)
    text_lines = path.read_text(encoding="ascii").splitlines()
    cursor = 0
    profiles: dict[int, HeliumStarkLine] = {}
    while cursor < len(text_lines):
        if not text_lines[cursor].strip():
            cursor += 1
            continue
        match = _BLOCK_HEADER.fullmatch(text_lines[cursor])
        if match is None:
            raise ValueError(
                f"invalid helium Stark-table header in {path}: "
                f"{text_lines[cursor]!r}"
            )
        cursor += 1
        point_count = int(match.group(1))
        nominal = float(match.group(2))
        if point_count < 3:
            raise ValueError(f"invalid wavelength count for He I {nominal:g}")
        offset, cursor = _read_numeric_values(
            text_lines,
            cursor,
            point_count,
            label=f"He I {nominal:g} wavelength offsets",
        )
        value_count = (
            _LOG_ELECTRON_DENSITY.size * _LOG_TEMPERATURE.size * point_count
        )
        values, cursor = _read_numeric_values(
            text_lines,
            cursor,
            value_count,
            label=f"He I {nominal:g} profiles",
        )
        profile = values.reshape(
            _LOG_ELECTRON_DENSITY.size,
            _LOG_TEMPERATURE.size,
            point_count,
        )
        if (
            np.any(~np.isfinite(offset))
            or np.any(np.diff(offset) <= 0.0)
            or np.any(~np.isfinite(profile))
            or np.any(profile < 0.0)
        ):
            raise ValueError(f"invalid profile data for He I {nominal:g}")
        key = int(round(nominal))
        if key in profiles:
            raise ValueError(f"duplicate He I profile near {key} Angstrom")
        tiny = np.finfo(np.float64).tiny
        profiles[key] = HeliumStarkLine(
            nominal_wavelength_angstrom=nominal,
            wavelength_offset_angstrom=offset,
            log_electron_density=_LOG_ELECTRON_DENSITY.copy(),
            log_temperature=_LOG_TEMPERATURE.copy(),
            log_profile_per_angstrom=np.log10(np.maximum(profile, tiny)),
        )
    if len(profiles) != 36:
        raise ValueError(f"expected 36 He I profiles in {path}, found {len(profiles)}")
    return HeliumStarkTable(lines=profiles, source_path=path)
