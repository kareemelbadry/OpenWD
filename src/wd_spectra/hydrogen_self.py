"""Barklem--Piskunov--O'Mara neutral-H Balmer profile tables.

The official HLINPROF distribution stores its self-broadening profiles in a
Fortran sequential-unformatted file.  This reader is intentionally small and
independent of a Fortran compiler.  It supports both the little-endian
``bpo_self.grid.DEC`` and big-endian ``bpo_self.grid.SUN`` deliveries.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import struct

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

BARKLEM_SELF_PROFILE_URL = (
    "https://raw.githubusercontent.com/barklem/hlinop/master/"
    "bpo_self.grid.DEC"
)
BARKLEM_SELF_PROFILE_SHA256 = (
    "37b5b652154ebaea93bd0827b0743d35a3a1308079eac0ca4300d4b3705352e0"
)


def _trapezoid_weights(coordinate: FloatArray) -> FloatArray:
    weights = np.empty_like(coordinate)
    weights[0] = 0.5 * (coordinate[1] - coordinate[0])
    weights[-1] = 0.5 * (coordinate[-1] - coordinate[-2])
    weights[1:-1] = 0.5 * (coordinate[2:] - coordinate[:-2])
    return weights


@dataclass(frozen=True)
class BarklemSelfBroadeningTable:
    """Full BPO Halpha--Hgamma self-broadening kernels.

    ``profile_per_angstrom`` is ordered as transition, neutral-H density,
    temperature, detuning.  Each source profile is area-normalized on the
    table's common, logarithmically refined wavelength-offset grid.  The
    slight 2--3 per cent quadrature error in the original 101-point delivery
    is removed on input so convolution conserves oscillator strength.
    """

    lower_level: IntArray
    upper_level: IntArray
    neutral_h_density: FloatArray
    temperature: FloatArray
    wavelength_offset_angstrom: FloatArray
    profile_per_angstrom: FloatArray
    source: str

    def __post_init__(self) -> None:
        n_line = self.lower_level.size
        expected = (
            n_line,
            self.neutral_h_density.size,
            self.temperature.size,
            self.wavelength_offset_angstrom.size,
        )
        if self.upper_level.shape != (n_line,):
            raise ValueError("Barklem transition arrays have inconsistent shapes")
        if self.profile_per_angstrom.shape != expected:
            raise ValueError("Barklem profile table has an inconsistent shape")
        if (
            np.any(np.diff(self.neutral_h_density) <= 0.0)
            or np.any(np.diff(self.temperature) <= 0.0)
            or np.any(np.diff(self.wavelength_offset_angstrom) <= 0.0)
        ):
            raise ValueError("Barklem table coordinates must be increasing")
        if (
            np.any(self.neutral_h_density <= 0.0)
            or np.any(self.temperature <= 0.0)
            or np.any(~np.isfinite(self.profile_per_angstrom))
            or np.any(self.profile_per_angstrom < 0.0)
        ):
            raise ValueError("Barklem table contains non-physical values")

    @property
    def transitions(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            zip(self.lower_level.tolist(), self.upper_level.tolist())
        )

    def contains_state(
        self,
        transition: tuple[int, int],
        temperature: float,
        neutral_h_density: float,
    ) -> bool:
        return (
            transition in self.transitions
            and self.temperature[0] <= temperature <= self.temperature[-1]
            and self.neutral_h_density[0]
            <= neutral_h_density
            <= self.neutral_h_density[-1]
        )

    def profile_at_state(
        self,
        transition: tuple[int, int],
        temperature: float,
        neutral_h_density: float,
    ) -> FloatArray:
        """Bilinearly interpolate one normalized kernel in log state.

        HLINPROF performs the same interpolation in log temperature, log
        perturber density, and log profile.  Extrapolation is deliberately
        rejected so callers can use the distribution's documented p--d
        fallback outside its 2500--20000 K and 1e14--1e19 cm^-3 domain.
        """

        if not self.contains_state(
            transition, float(temperature), float(neutral_h_density)
        ):
            raise ValueError("state lies outside the Barklem self-profile table")
        line_index = self.transitions.index(transition)
        log_temperature = np.log10(self.temperature)
        log_density = np.log10(self.neutral_h_density)
        target_temperature = np.log10(float(temperature))
        target_density = np.log10(float(neutral_h_density))
        upper_temperature = int(
            np.clip(
                np.searchsorted(log_temperature, target_temperature, side="left"),
                1,
                log_temperature.size - 1,
            )
        )
        upper_density = int(
            np.clip(
                np.searchsorted(log_density, target_density, side="left"),
                1,
                log_density.size - 1,
            )
        )
        temperature_weight = (
            (target_temperature - log_temperature[upper_temperature - 1])
            / (
                log_temperature[upper_temperature]
                - log_temperature[upper_temperature - 1]
            )
        )
        density_weight = (
            (target_density - log_density[upper_density - 1])
            / (log_density[upper_density] - log_density[upper_density - 1])
        )
        corner = self.profile_per_angstrom[
            line_index,
            upper_density - 1 : upper_density + 1,
            upper_temperature - 1 : upper_temperature + 1,
            :,
        ]
        log_corner = np.log(np.maximum(corner, np.finfo(np.float64).tiny))
        low_temperature = (
            (1.0 - density_weight) * log_corner[0, 0]
            + density_weight * log_corner[1, 0]
        )
        high_temperature = (
            (1.0 - density_weight) * log_corner[0, 1]
            + density_weight * log_corner[1, 1]
        )
        profile = np.exp(
            (1.0 - temperature_weight) * low_temperature
            + temperature_weight * high_temperature
        )
        weights = _trapezoid_weights(self.wavelength_offset_angstrom)
        normalization = float(np.sum(weights * profile))
        if not np.isfinite(normalization) or normalization <= 0.0:
            raise RuntimeError("interpolated Barklem profile lost normalization")
        return np.asarray(profile / normalization, dtype=np.float64)


def _read_fortran_records(path: Path, endian: str) -> list[bytes]:
    records: list[bytes] = []
    marker = struct.Struct(endian + "i")
    with path.open("rb") as stream:
        while True:
            prefix = stream.read(marker.size)
            if not prefix:
                break
            if len(prefix) != marker.size:
                raise ValueError(f"truncated Fortran record marker in {path}")
            length = marker.unpack(prefix)[0]
            if length < 0 or length > 100_000_000:
                raise ValueError(f"invalid Fortran record length in {path}")
            payload = stream.read(length)
            suffix = stream.read(marker.size)
            if len(payload) != length or len(suffix) != marker.size:
                raise ValueError(f"truncated Fortran record in {path}")
            if marker.unpack(suffix)[0] != length:
                raise ValueError(f"mismatched Fortran record markers in {path}")
            records.append(payload)
    return records


def _detect_endian(path: Path) -> str:
    with path.open("rb") as stream:
        prefix = stream.read(4)
    if len(prefix) != 4:
        raise ValueError(f"empty Barklem profile table: {path}")
    if struct.unpack("<i", prefix)[0] == 4:
        return "<"
    if struct.unpack(">i", prefix)[0] == 4:
        return ">"
    raise ValueError(f"unrecognized Fortran record byte order in {path}")


@lru_cache(maxsize=4)
def read_barklem_self_broadening_table(
    path: str | Path,
) -> BarklemSelfBroadeningTable:
    """Read an official HLINPROF ``bpo_self.grid`` delivery."""

    source = Path(path).expanduser().resolve()
    endian = _detect_endian(source)
    records = _read_fortran_records(source, endian)
    cursor = 0

    def take() -> bytes:
        nonlocal cursor
        if cursor >= len(records):
            raise ValueError(f"incomplete Barklem profile table: {source}")
        result = records[cursor]
        cursor += 1
        return result

    def scalar_int() -> int:
        payload = take()
        if len(payload) != 4:
            raise ValueError(f"invalid integer record in {source}")
        return struct.unpack(endian + "i", payload)[0]

    n_line = scalar_int()
    transitions = np.frombuffer(take(), dtype=endian + "i4").astype(np.int64)
    if transitions.size != 2 * n_line:
        raise ValueError(f"invalid transition record in {source}")
    transitions = transitions.reshape(n_line, 2)
    n_density = scalar_int()
    density = np.frombuffer(take(), dtype=endian + "f8").astype(np.float64)
    if density.size != n_density:
        raise ValueError(f"invalid density grid in {source}")
    n_temperature = scalar_int()
    temperature = np.frombuffer(take(), dtype=endian + "f8").astype(np.float64)
    if temperature.size != n_temperature:
        raise ValueError(f"invalid temperature grid in {source}")
    n_profile = scalar_int()
    detuning = np.empty((n_line, n_density, n_temperature, n_profile))
    profile = np.empty_like(detuning)
    pair = struct.Struct(endian + "dd")
    for line_index in range(n_line):
        for density_index in range(n_density):
            for temperature_index in range(n_temperature):
                for profile_index in range(n_profile):
                    payload = take()
                    if len(payload) != pair.size:
                        raise ValueError(f"invalid profile record in {source}")
                    (
                        detuning[
                            line_index,
                            density_index,
                            temperature_index,
                            profile_index,
                        ],
                        profile[
                            line_index,
                            density_index,
                            temperature_index,
                            profile_index,
                        ],
                    ) = pair.unpack(payload)
    if cursor != len(records):
        raise ValueError(f"unexpected trailing records in {source}")
    common_detuning = detuning[0, 0, 0]
    if not np.allclose(detuning, common_detuning, rtol=0.0, atol=1.0e-12):
        raise ValueError("Barklem delivery does not use one common detuning grid")
    weights = _trapezoid_weights(common_detuning)
    normalizations = np.sum(profile * weights, axis=-1, keepdims=True)
    if np.any(~np.isfinite(normalizations)) or np.any(normalizations <= 0.0):
        raise ValueError(f"invalid profile normalization in {source}")
    profile = profile / normalizations
    return BarklemSelfBroadeningTable(
        lower_level=np.asarray(transitions[:, 0], dtype=np.int64),
        upper_level=np.asarray(transitions[:, 1], dtype=np.int64),
        neutral_h_density=np.asarray(density, dtype=np.float64),
        temperature=np.asarray(temperature, dtype=np.float64),
        wavelength_offset_angstrom=np.asarray(common_detuning, dtype=np.float64),
        profile_per_angstrom=np.asarray(profile, dtype=np.float64),
        source=str(source),
    )


__all__ = [
    "BARKLEM_SELF_PROFILE_SHA256",
    "BARKLEM_SELF_PROFILE_URL",
    "BarklemSelfBroadeningTable",
    "read_barklem_self_broadening_table",
]
