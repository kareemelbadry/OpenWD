"""Electron heat conduction in white-dwarf envelopes.

This module interpolates the public Potekhin/Cassisi electron-conductivity
table.  It supplies microscopic transport coefficients, not an atmosphere
grid: temperature, density, ionization, and the atmospheric stratification
remain outputs of the atmosphere and EOS solvers.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import STEFAN_BOLTZMANN


FloatArray = NDArray[np.float64]
_TABLE_PATH = (
    Path(__file__).with_name("data")
    / "conduction"
    / "potekhin_condtab21wd.dat"
)
_N_TEMPERATURE = 19
_N_DENSITY = 64


@lru_cache(maxsize=1)
def _read_potekhin_hydrogen_conductivity_table(
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Read the Z=1 block of the 2021 Ioffe ``condtab21wd`` table."""

    with _TABLE_PATH.open("r", encoding="ascii") as stream:
        version = stream.readline().strip()
        rows = [
            [float(token) for token in stream.readline().split()]
            for _ in range(_N_DENSITY + 1)
        ]
    data = np.asarray(rows, dtype=np.float64)
    if (
        "conduct21.f" not in version
        or data.shape != (_N_DENSITY + 1, _N_TEMPERATURE + 1)
        or not np.isclose(data[0, 0], 1.0)
        or np.any(~np.isfinite(data))
    ):
        raise ValueError(f"unexpected Potekhin conductivity table in {_TABLE_PATH}")
    log_temperature = data[0, 1:]
    log_density = data[1:, 0]
    log_conductivity = data[1:, 1:]
    if (
        np.any(np.diff(log_temperature) <= 0.0)
        or np.any(np.diff(log_density) <= 0.0)
        or np.any(log_conductivity <= 0.0)
    ):
        raise ValueError(f"invalid Potekhin conductivity grid in {_TABLE_PATH}")
    return log_temperature, log_density, log_conductivity


def _potekhin_cubic_interpolate(
    grid: FloatArray,
    values: FloatArray,
    coordinate: float,
) -> float:
    """Reproduce ``CINTERP3`` from Potekhin's public ``condint.f``."""

    interval = int(np.searchsorted(grid, coordinate, side="right") - 1)
    interval = min(max(interval, 0), grid.size - 2)
    x0 = grid[interval]
    x1 = grid[interval + 1]
    spacing = x1 - x0
    fraction = (coordinate - x0) / spacing
    v0 = values[interval]
    v1 = values[interval + 1]

    if interval > 0:
        left_spacing = x0 - grid[interval - 1]
        left_derivative = (
            (v1 - v0) / spacing**2
            + (v0 - values[interval - 1]) / left_spacing**2
        ) / (1.0 / spacing + 1.0 / left_spacing)
    if interval < grid.size - 2:
        right_spacing = grid[interval + 2] - x1
        right_derivative = (
            (v1 - v0) / spacing**2
            + (values[interval + 2] - v1) / right_spacing**2
        ) / (1.0 / spacing + 1.0 / right_spacing)

    offset = coordinate - x0
    if 0 < interval < grid.size - 2:
        quadratic = (
            3.0 * (v1 - v0)
            - spacing * (right_derivative + 2.0 * left_derivative)
        )
        cubic = (
            spacing * (left_derivative + right_derivative)
            - 2.0 * (v1 - v0)
        )
        return float(
            v0
            + left_derivative * offset
            + quadratic * fraction**2
            + cubic * fraction**3
        )
    if interval == 0:
        quadratic = v0 - v1 + right_derivative * spacing
        return float(
            v1
            - right_derivative * (spacing - offset)
            + quadratic * (1.0 - fraction) ** 2
        )
    quadratic = v1 - v0 - left_derivative * spacing
    return float(
        v0 + left_derivative * offset + quadratic * fraction**2
    )


def hydrogen_electron_thermal_conductivity(
    temperature: ArrayLike,
    mass_density: ArrayLike,
) -> FloatArray:
    r"""Return the pure-H electron conductivity in cgs units.

    The result has units erg cm\ :sup:`-1` s\ :sup:`-1` K\ :sup:`-1`.
    The public 2021 ``condtab21wd`` table combines the arbitrary-degeneracy
    Potekhin/Cassisi treatment with the weakly damped Blouin et al. (2020)
    correction appropriate to white-dwarf H/He envelopes.  Values outside
    its published domain are returned as zero instead of extrapolated.

    The source table assumes a fully ionized plasma.  In a cool DA its formal
    low-temperature entries are only estimates, but conduction is immaterial
    where the atmosphere is predominantly neutral and radiative transport is
    overwhelmingly larger.
    """

    temperature_array, density_array = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(mass_density, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(temperature_array))
        or np.any(temperature_array <= 0.0)
        or np.any(~np.isfinite(density_array))
        or np.any(density_array <= 0.0)
    ):
        raise ValueError("temperature and mass_density must be finite and positive")
    log_temperature_grid, log_density_grid, table = (
        _read_potekhin_hydrogen_conductivity_table()
    )
    log_temperature = np.log10(temperature_array)
    log_density = np.log10(density_array)
    inside = (
        (log_temperature >= log_temperature_grid[0])
        & (log_temperature <= log_temperature_grid[-1])
        & (log_density >= log_density_grid[0])
        & (log_density <= log_density_grid[-1])
    )
    result = np.zeros(temperature_array.shape, dtype=np.float64)
    flat_result = result.ravel()
    flat_temperature = log_temperature.ravel()
    flat_density = log_density.ravel()
    for index in np.flatnonzero(inside.ravel()):
        density_interpolated = np.asarray(
            [
                _potekhin_cubic_interpolate(
                    log_density_grid,
                    table[:, temperature_index],
                    float(flat_density[index]),
                )
                for temperature_index in range(log_temperature_grid.size)
            ],
            dtype=np.float64,
        )
        log_value = _potekhin_cubic_interpolate(
            log_temperature_grid,
            density_interpolated,
            float(flat_temperature[index]),
        )
        flat_result[index] = 10.0**log_value
    return result


def conductive_opacity_from_thermal_conductivity(
    temperature: ArrayLike,
    mass_density: ArrayLike,
    thermal_conductivity: ArrayLike,
) -> FloatArray:
    r"""Return the equivalent conductive opacity in cm2 g-1.

    It is defined so that radiative and conductive diffusion combine through
    ``1/kappa_total = 1/kappa_rad + 1/kappa_cond``.  A zero conductivity maps
    to infinite conductive opacity.
    """

    temperature_array, density_array, conductivity_array = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(mass_density, dtype=np.float64),
        np.asarray(thermal_conductivity, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(temperature_array))
        or np.any(temperature_array <= 0.0)
        or np.any(~np.isfinite(density_array))
        or np.any(density_array <= 0.0)
        or np.any(~np.isfinite(conductivity_array))
        or np.any(conductivity_array < 0.0)
    ):
        raise ValueError("temperature and density must be positive; conductivity non-negative")
    numerator = 16.0 * STEFAN_BOLTZMANN * temperature_array**3
    denominator = 3.0 * density_array * conductivity_array
    return np.divide(
        numerator,
        denominator,
        out=np.full(temperature_array.shape, np.inf),
        where=conductivity_array > 0.0,
    )


__all__ = [
    "conductive_opacity_from_thermal_conductivity",
    "hydrogen_electron_thermal_conductivity",
]
