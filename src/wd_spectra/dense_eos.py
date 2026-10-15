"""Tabulated dense-fluid equations of state.

The helium table distributed with Becker et al. (2014) gives mass density,
temperature, pressure, and specific internal energy.  Atmosphere calculations
most naturally require the inverse relation ``rho(P, T)``.  This module keeps
that interpolation separate from the chemical-picture ionization solver: the
tabulated EOS supplies the bulk density and caloric quantities, while the
level populations are solved at the resulting helium-nucleus density.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator

from .constants import BOLTZMANN, HELIUM_MASS


FloatArray = NDArray[np.float64]
GPA_TO_DYNE_CM2 = 1.0e10
KJ_G_TO_ERG_G = 1.0e10
HELIUM_REOS3_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/215/21/table3.dat"
HELIUM_REOS3_SHA256 = (
    "f67c76b03df5be453f4bb96f57c9cf822f997a83114d551c313208aecbd745a1"
)


def _pchip_slope(left, right, h_left, h_right):
    """Fritsch-Carlson weighted-harmonic node slope (zero at extrema)."""

    w1, w2 = 2 * h_right + h_left, h_right + 2 * h_left
    same = left * right > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (w1 + w2) / (w1 / left + w2 / right)
    return np.where(same, slope, 0.0)


@dataclass(frozen=True)
class HeliumREOS3Table:
    """Inverse ``(P,T)`` representation of the He-REOS.3 table.

    ``interpolation="log-linear"`` (the historical default) is linear in
    ``log(P)``, ``log(rho)``, and ``log(T)``, with the specific internal energy
    linear in ``log(T)``.  The table isotherms are 1.7-2x apart in T, so that
    interpolant does not reproduce the ideal-gas limit between them: c_P and
    nabla_ad are wrong by up to ~40% even at low density and jump at every
    isotherm (e.g. nabla_ad 0.46 -> 0.35 across 6000 K at 1e8 dyn cm^-2).

    ``interpolation="ideal-excess"`` uses the same tabulated values but
    interpolates only the non-ideal excess of each isotherm,
    ``ln(rho/rho_ideal)`` and ``u - 3kT/(2 m_He)``, with PCHIP in ``log(P)``
    and a C1 cubic Hermite in ``log(T)``. It reproduces every tabulated point
    and the exact ideal monatomic gas, and gives continuous analytic c_P, Q
    and nabla_ad (``thermodynamics``). Below the lowest tabulated density it
    continues the dilute (ideal-gas) endpoint. Requests outside the rectangular
    pressure/temperature support are reported through ``inside_domain`` so a
    caller can retain its dilute-gas EOS rather than extrapolating the dense
    table.
    """

    temperature_grid: FloatArray
    density_by_temperature: tuple[FloatArray, ...]
    pressure_by_temperature: tuple[FloatArray, ...]
    internal_energy_by_temperature: tuple[FloatArray, ...]
    source: str = "Becker et al. (2014) He-REOS.3, VizieR J/ApJS/215/21 table 3"
    interpolation: str = "log-linear"

    def __post_init__(self) -> None:
        if self.interpolation not in ("log-linear", "ideal-excess"):
            raise ValueError("REOS interpolation must be 'log-linear' or 'ideal-excess'")
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
        if self.interpolation == "ideal-excess":
            phi, psi, _, _, inside = self._excess(log_pressure, log_temperature)
            density = flat_pressure * HELIUM_MASS / (BOLTZMANN * flat_temperature) * np.exp(phi)
            energy = 1.5 * BOLTZMANN * flat_temperature / HELIUM_MASS + psi
            return density.reshape(shape), energy.reshape(shape), inside.reshape(shape)
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

    def with_interpolation(self, interpolation: str) -> "HeliumREOS3Table":
        return replace(self, interpolation=interpolation)

    def _excess_isotherms(self):
        cached = self.__dict__.get("_excess_cache")
        if cached is None:
            cached = []
            for temperature, density, pressure, energy in zip(
                self.temperature_grid,
                self.density_by_temperature,
                self.pressure_by_temperature,
                self.internal_energy_by_temperature,
            ):
                log_pressure = np.log(pressure)
                ideal_log_density = np.log(
                    pressure * HELIUM_MASS / (BOLTZMANN * temperature)
                )
                ideal_energy = 1.5 * BOLTZMANN * temperature / HELIUM_MASS
                cached.append((
                    log_pressure[0],
                    log_pressure[-1],
                    PchipInterpolator(log_pressure, np.log(density) - ideal_log_density),
                    PchipInterpolator(
                        log_pressure, np.asarray(energy) * KJ_G_TO_ERG_G - ideal_energy
                    ),
                ))
            object.__setattr__(self, "_excess_cache", cached)
        return cached

    def _excess(self, log_pressure: FloatArray, log_temperature: FloatArray):
        """Hermite excess (phi, psi) and their log-T derivatives at fixed P."""

        grid = np.log(self.temperature_grid)
        n = grid.size
        isotherms = self._excess_isotherms()
        lower = np.clip(np.searchsorted(grid, log_temperature, side="right") - 1, 0, n - 2)
        phi = np.empty_like(log_pressure)
        psi = np.empty_like(log_pressure)
        dphi = np.empty_like(log_pressure)
        dpsi = np.empty_like(log_pressure)
        inside = (log_temperature >= grid[0]) & (log_temperature <= grid[-1])

        def values(k, x):
            low, high, phi_curve, psi_curve = isotherms[k]
            # Below the lowest tabulated density (1e-10 g cm^-3) the excess is
            # held at its dilute endpoint value: zero to 1e-6 for T <= 6000 K,
            # where it is exact; it omits only the growth of thermal He
            # ionization at still lower density for T >~ 1e4 K. Higher
            # pressures than tabulated remain outside the domain.
            ok = x <= high
            clipped = np.clip(x, low, high)
            return phi_curve(clipped), psi_curve(clipped), ok

        for i in np.unique(lower):
            sel = lower == i
            x = log_pressure[sel]
            y = log_temperature[sel]
            f0, g0, ok0 = values(i, x)
            f1, g1, ok1 = values(i + 1, x)
            h = grid[i + 1] - grid[i]
            secant_f, secant_g = (f1 - f0) / h, (g1 - g0) / h
            # Monotone (Fritsch-Carlson/PCHIP) node slopes where the
            # neighbouring isotherm covers this pressure, else the secant.
            # This keeps the tabulated monotonicity of the excess energy
            # (no negative c_P from cubic overshoot near ionization).
            d0f, d0g = secant_f.copy(), secant_g.copy()
            if i > 0:
                fm, gm, okm = values(i - 1, x)
                hm = grid[i] - grid[i - 1]
                d0f = np.where(okm, _pchip_slope((f0 - fm) / hm, secant_f, hm, h), d0f)
                d0g = np.where(okm, _pchip_slope((g0 - gm) / hm, secant_g, hm, h), d0g)
            d1f, d1g = secant_f.copy(), secant_g.copy()
            if i + 2 < n:
                fp, gp, okp = values(i + 2, x)
                hp = grid[i + 2] - grid[i + 1]
                d1f = np.where(okp, _pchip_slope(secant_f, (fp - f1) / hp, h, hp), d1f)
                d1g = np.where(okp, _pchip_slope(secant_g, (gp - g1) / hp, h, hp), d1g)
            t = (y - grid[i]) / h
            h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
            h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
            e00, e10 = (6 * t**2 - 6 * t) / h, 3 * t**2 - 4 * t + 1
            e01, e11 = (-6 * t**2 + 6 * t) / h, 3 * t**2 - 2 * t
            phi[sel] = h00 * f0 + h10 * h * d0f + h01 * f1 + h11 * h * d1f
            psi[sel] = h00 * g0 + h10 * h * d0g + h01 * g1 + h11 * h * d1g
            dphi[sel] = e00 * f0 + e10 * d0f + e01 * f1 + e11 * d1f
            dpsi[sel] = e00 * g0 + e10 * d0g + e01 * g1 + e11 * d1g
            inside[sel] &= np.where(t <= 1e-14, ok0, np.where(t >= 1 - 1e-14, ok1, ok0 & ok1))
        return phi, psi, dphi, dpsi, inside

    def thermodynamics(
        self, gas_pressure: ArrayLike, temperature: ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
        """Analytic ``(rho, c_P, Q, nabla_ad, inside)`` of the ideal-excess EOS.

        ``Q = -dln(rho)/dln(T)`` at fixed P. Only available for the smooth
        interpolation; the log-linear interpolant has no continuous derivative.
        """

        if self.interpolation != "ideal-excess":
            raise ValueError("analytic REOS thermodynamics require interpolation='ideal-excess'")
        pressure, temp = np.broadcast_arrays(
            np.asarray(gas_pressure, dtype=np.float64),
            np.asarray(temperature, dtype=np.float64),
        )
        shape = pressure.shape
        p, t = pressure.ravel(), temp.ravel()
        phi, _, dphi, dpsi, inside = self._excess(np.log(p), np.log(t))
        density = p * HELIUM_MASS / (BOLTZMANN * t) * np.exp(phi)
        expansion = 1.0 - dphi
        specific_heat = (
            1.5 * BOLTZMANN / HELIUM_MASS + dpsi / t + p * expansion / (density * t)
        )
        gradient = p * expansion / (density * t * specific_heat)
        return tuple(a.reshape(shape) for a in (density, specific_heat, expansion, gradient, inside))


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
