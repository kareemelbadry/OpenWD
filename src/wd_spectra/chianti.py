"""Small, dependency-free readers for CHIANTI collision-strength fits.

CHIANTI ``.scups`` files store Burgess--Tully scaled effective electron
collision strengths.  Only the compact interpolation machinery lives here;
ion-specific statistical-equilibrium or line-source calculations belong in
their respective physics modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


CHIANTI_CA_II_SCUPS_URL = (
    "https://sohoftp.nascom.nasa.gov/solarsoft/packages/chianti/dbase/"
    "ca/ca_2/ca_2.scups"
)
CHIANTI_CA_II_SCUPS_SHA256 = (
    "6982803dbde3f35f27b2589425c6af0c0f921a071dbc0edcf396efedb7a3a110"
)


def _natural_cubic_spline_value(
    x: FloatArray, y: FloatArray, query: float
) -> float:
    """Evaluate the natural cubic spline used by the CHIANTI readers."""

    coordinate = np.asarray(x, dtype=np.float64)
    value = np.asarray(y, dtype=np.float64)
    if (
        coordinate.ndim != 1
        or value.shape != coordinate.shape
        or coordinate.size < 2
        or np.any(~np.isfinite(coordinate))
        or np.any(~np.isfinite(value))
        or np.any(np.diff(coordinate) <= 0.0)
    ):
        raise ValueError("cubic-spline nodes must be finite and increasing")
    second = np.zeros_like(value)
    work = np.zeros_like(value)
    for index in range(1, coordinate.size - 1):
        fraction = (
            (coordinate[index] - coordinate[index - 1])
            / (coordinate[index + 1] - coordinate[index - 1])
        )
        pivot = fraction * second[index - 1] + 2.0
        second[index] = (fraction - 1.0) / pivot
        slope_change = (
            (value[index + 1] - value[index])
            / (coordinate[index + 1] - coordinate[index])
            - (value[index] - value[index - 1])
            / (coordinate[index] - coordinate[index - 1])
        )
        work[index] = (
            6.0 * slope_change
            / (coordinate[index + 1] - coordinate[index - 1])
            - fraction * work[index - 1]
        ) / pivot
    for index in range(coordinate.size - 2, -1, -1):
        second[index] = second[index] * second[index + 1] + work[index]
    upper = int(np.searchsorted(coordinate, query, side="right"))
    upper = int(np.clip(upper, 1, coordinate.size - 1))
    lower = upper - 1
    width = coordinate[upper] - coordinate[lower]
    a = (coordinate[upper] - query) / width
    b = (query - coordinate[lower]) / width
    return float(
        a * value[lower]
        + b * value[upper]
        + ((a**3 - a) * second[lower] + (b**3 - b) * second[upper])
        * width**2
        / 6.0
    )


@dataclass(frozen=True)
class ChiantiScaledCollisionComponent:
    """One Burgess--Tully scaled CHIANTI effective-collision-strength fit."""

    transition_energy_rydberg: float
    transition_type: int
    scaling_parameter: float
    scaled_temperature: tuple[float, ...]
    scaled_upsilon: tuple[float, ...]

    def effective_collision_strength(self, temperature: float) -> float:
        """De-scale the CHIANTI spline at one electron temperature."""

        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        scaled_temperature = np.asarray(self.scaled_temperature)
        scaled_upsilon = np.asarray(self.scaled_upsilon)
        if (
            scaled_temperature.size < 2
            or scaled_temperature.size != scaled_upsilon.size
        ):
            raise ValueError("CHIANTI scaled collision arrays are inconsistent")
        kte = float(temperature) / self.transition_energy_rydberg / 1.57888e5
        if self.transition_type in (1, 4):
            query = 1.0 - np.log(self.scaling_parameter) / np.log(
                kte + self.scaling_parameter
            )
        elif self.transition_type in (2, 3, 5, 6):
            query = kte / (kte + self.scaling_parameter)
        else:
            raise ValueError(
                f"unsupported CHIANTI transition type {self.transition_type}"
            )
        query = float(np.clip(query, scaled_temperature[0], scaled_temperature[-1]))
        scaled = _natural_cubic_spline_value(
            scaled_temperature, scaled_upsilon, query
        )
        if self.transition_type == 1:
            value = scaled * np.log(kte + np.e)
        elif self.transition_type == 2:
            value = scaled
        elif self.transition_type == 3:
            value = scaled / (kte + 1.0)
        elif self.transition_type == 4:
            value = scaled * np.log(kte + self.scaling_parameter)
        elif self.transition_type == 5:
            value = scaled / max(kte, 1.0e-300)
        else:
            value = 10.0**scaled
        return float(max(value, 0.0))


@lru_cache(maxsize=16)
def _read_chianti_scaled_collision_components_cached(
    path: str,
) -> Mapping[tuple[int, int], ChiantiScaledCollisionComponent]:
    lines = Path(path).read_text(encoding="ascii").splitlines()
    records: dict[tuple[int, int], ChiantiScaledCollisionComponent] = {}
    cursor = 0
    while cursor < len(lines):
        header = lines[cursor].strip()
        cursor += 1
        if not header or header.startswith("%"):
            continue
        if header == "-1":
            break
        fields = header.split()
        if len(fields) < 8 or cursor + 1 >= len(lines):
            raise ValueError(f"malformed CHIANTI SCUPS header in {path}: {header}")
        try:
            lower_index = int(fields[0])
            upper_index = int(fields[1])
            energy = float(fields[2])
            node_count = int(fields[5])
            transition_type = int(fields[6])
            scaling_parameter = float(fields[7])
            scaled_temperature = tuple(float(v) for v in lines[cursor].split())
            scaled_upsilon = tuple(float(v) for v in lines[cursor + 1].split())
        except ValueError as error:
            raise ValueError(
                f"malformed CHIANTI SCUPS record in {path}: {header}"
            ) from error
        cursor += 2
        if (
            energy <= 0.0
            or node_count < 2
            or len(scaled_temperature) != node_count
            or len(scaled_upsilon) != node_count
        ):
            raise ValueError(f"incomplete CHIANTI SCUPS record in {path}: {header}")
        key = (lower_index, upper_index)
        if key in records:
            raise ValueError(f"duplicate CHIANTI SCUPS transition {key} in {path}")
        records[key] = ChiantiScaledCollisionComponent(
            transition_energy_rydberg=energy,
            transition_type=transition_type,
            scaling_parameter=scaling_parameter,
            scaled_temperature=scaled_temperature,
            scaled_upsilon=scaled_upsilon,
        )
    if not records:
        raise ValueError(f"{path} contains no CHIANTI collision records")
    return MappingProxyType(records)


def read_chianti_scaled_collision_components(
    path: str | Path,
) -> Mapping[tuple[int, int], ChiantiScaledCollisionComponent]:
    """Read all fine-level fits in one CHIANTI ``.scups`` file."""

    return _read_chianti_scaled_collision_components_cached(str(Path(path).resolve()))
