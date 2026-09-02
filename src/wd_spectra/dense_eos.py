"""Tabulated dense-fluid equations of state.

The helium table distributed with Becker et al. (2014) gives mass density,
temperature, pressure, and specific internal energy.  Atmosphere calculations
most naturally require the inverse relation ``rho(P, T)``.  This module keeps
that interpolation separate from the chemical-picture ionization solver: the
tabulated EOS supplies the bulk density and caloric quantities, while the
level populations are solved at the resulting helium-nucleus density.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
GPA_TO_DYNE_CM2 = 1.0e10
KJ_G_TO_ERG_G = 1.0e10
HELIUM_REOS3_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/215/21/table3.dat"
HELIUM_REOS3_SHA256 = (
    "f67c76b03df5be453f4bb96f57c9cf822f997a83114d551c313208aecbd745a1"
)


@dataclass(frozen=True)
class HeliumREOS3Table:
    """Inverse ``(P,T)`` representation of the He-REOS.3 table.

    Interpolation is linear in ``log(P)``, ``log(rho)``, and ``log(T)``.
    The tabulated specific internal energy is interpolated linearly in energy
    after the pressure interpolation.  Requests outside the rectangular
    pressure/temperature support are reported through ``inside_domain`` so a
    caller can retain its dilute-gas EOS rather than extrapolating the dense
    table.
    """

    temperature_grid: FloatArray
    density_by_temperature: tuple[FloatArray, ...]
    pressure_by_temperature: tuple[FloatArray, ...]
    internal_energy_by_temperature: tuple[FloatArray, ...]
    source: str = "Becker et al. (2014) He-REOS.3, VizieR J/ApJS/215/21 table 3"

    def __post_init__(self) -> None:
        temperature = np.asarray(self.temperature_grid, dtype=np.float64)
        if temperature.ndim != 1 or temperature.size < 2:
            raise ValueError("REOS temperature grid must be one-dimensional")
        if np.any(~np.isfinite(temperature)) or np.any(np.diff(temperature) <= 0.0):
            raise ValueError("REOS temperatures must be finite and increasing")
        if not (
            len(self.density_by_temperature)
            == len(self.pressure_by_temperature)
            == len(self.internal_energy_by_temperature)
            == temperature.size
        ):
            raise ValueError("REOS isotherm arrays do not match temperature grid")
        for density, pressure, energy in zip(
            self.density_by_temperature,
            self.pressure_by_temperature,
            self.internal_energy_by_temperature,
        ):
            density = np.asarray(density, dtype=np.float64)
            pressure = np.asarray(pressure, dtype=np.float64)
            energy = np.asarray(energy, dtype=np.float64)
            if density.ndim != 1 or density.size < 2:
                raise ValueError("each REOS isotherm must have at least two points")
            if density.shape != pressure.shape or density.shape != energy.shape:
                raise ValueError("REOS density, pressure, and energy shapes differ")
            if (
                np.any(~np.isfinite(density))
                or np.any(~np.isfinite(pressure))
                or np.any(~np.isfinite(energy))
                or np.any(density <= 0.0)
                or np.any(pressure <= 0.0)
                or np.any(np.diff(density) <= 0.0)
                or np.any(np.diff(pressure) <= 0.0)
            ):
                raise ValueError("REOS isotherms must be finite, positive, and monotone")

    def _at_isotherm(
        self,
        index: int,
        log_pressure: FloatArray,
    ) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
        pressure = self.pressure_by_temperature[index]
        density = self.density_by_temperature[index]
        energy = self.internal_energy_by_temperature[index]
        log_grid_pressure = np.log(pressure)
        inside = (
            (log_pressure >= log_grid_pressure[0])
            & (log_pressure <= log_grid_pressure[-1])
        )
        clipped = np.clip(log_pressure, log_grid_pressure[0], log_grid_pressure[-1])
        return (
            np.interp(clipped, log_grid_pressure, np.log(density)),
            np.interp(clipped, log_grid_pressure, energy),
            inside,
        )

    def evaluate(
        self,
        gas_pressure: ArrayLike,
        temperature: ArrayLike,
    ) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
        """Return density [g cm^-3], internal energy [erg g^-1], and domain mask."""

        pressure, temp = np.broadcast_arrays(
            np.asarray(gas_pressure, dtype=np.float64),
            np.asarray(temperature, dtype=np.float64),
        )
        if (
            np.any(~np.isfinite(pressure))
            or np.any(pressure <= 0.0)
            or np.any(~np.isfinite(temp))
            or np.any(temp <= 0.0)
        ):
            raise ValueError("pressure and temperature must be finite and positive")
        shape = pressure.shape
        flat_pressure = pressure.ravel()
        flat_temperature = temp.ravel()
        log_pressure = np.log(flat_pressure)
        log_temperature = np.log(flat_temperature)
        log_grid_temperature = np.log(self.temperature_grid)
        upper = np.searchsorted(log_grid_temperature, log_temperature, side="right")
        upper = np.clip(upper, 1, self.temperature_grid.size - 1)
        lower = upper - 1
        temperature_weight = (
            (log_temperature - log_grid_temperature[lower])
            / (log_grid_temperature[upper] - log_grid_temperature[lower])
        )
        result_log_density = np.empty_like(flat_pressure)
        result_energy = np.empty_like(flat_pressure)
        inside = (
            (flat_temperature >= self.temperature_grid[0])
            & (flat_temperature <= self.temperature_grid[-1])
        )
        for lower_index in np.unique(lower):
            selected = lower == lower_index
            upper_index = lower_index + 1
            lower_density, lower_energy, lower_inside = self._at_isotherm(
                int(lower_index), log_pressure[selected]
            )
            upper_density, upper_energy, upper_inside = self._at_isotherm(
                int(upper_index), log_pressure[selected]
            )
            weight = temperature_weight[selected]
            result_log_density[selected] = (
                (1.0 - weight) * lower_density + weight * upper_density
            )
            result_energy[selected] = (
                (1.0 - weight) * lower_energy + weight * upper_energy
            )
            local_inside = lower_inside & upper_inside
            local_inside = np.where(weight <= 1.0e-14, lower_inside, local_inside)
            local_inside = np.where(weight >= 1.0 - 1.0e-14, upper_inside, local_inside)
            inside[selected] &= local_inside
        return (
            np.exp(result_log_density).reshape(shape),
            (result_energy * KJ_G_TO_ERG_G).reshape(shape),
            inside.reshape(shape),
        )


def read_helium_reos3_table(path: str | Path) -> HeliumREOS3Table:
    """Read Becker et al. (2014) VizieR table 3.

    The four whitespace-separated columns are density [g cm^-3], temperature
    [K], pressure [GPa], and specific internal energy [kJ g^-1].
    """

    source_path = Path(path)
    values = np.loadtxt(source_path, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"{source_path} is not a four-column He-REOS.3 table")
    density, temperature, pressure_gpa, energy = values.T
    temperature_grid = np.unique(temperature)
    density_rows: list[FloatArray] = []
    pressure_rows: list[FloatArray] = []
    energy_rows: list[FloatArray] = []
    for local_temperature in temperature_grid:
        selected = temperature == local_temperature
        density_rows.append(np.asarray(density[selected]))
        pressure_rows.append(np.asarray(pressure_gpa[selected] * GPA_TO_DYNE_CM2))
        energy_rows.append(np.asarray(energy[selected]))
    return HeliumREOS3Table(
        np.asarray(temperature_grid),
        tuple(density_rows),
        tuple(pressure_rows),
        tuple(energy_rows),
    )
