"""Schönning--Butler He II Stark-profile tables distributed by SYNSPEC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
ThermodynamicInterpolation = Literal[
    "synspec-quadratic", "log-bilinear", "series-adaptive"
]
HELIUM_II_STARK_URL = (
    "https://tlusty.oca.eu/tlusty/Synspec49/data/he2prf.dat"
)
HELIUM_II_STARK_SHA256 = (
    "e015e448be5b7448d498571f9475a8e6c6cbf32fcac36ef0369d8831afb3199f"
)


@dataclass(frozen=True)
class HeliumIIStarkLine:
    """One tabulated He II profile, stored as ``log10(phi_nu / Hz^-1)``."""

    lower_level: int
    upper_level: int
    wavelength_offset_angstrom: FloatArray
    log_temperature: FloatArray
    log_electron_density: FloatArray
    log_profile_per_hz: FloatArray
    thermodynamic_interpolation: ThermodynamicInterpolation = "synspec-quadratic"

    def frequency_profile(
        self,
        wavelength_angstrom: ArrayLike,
        line_center_angstrom: float,
        temperature: float,
        electron_density: float,
    ) -> FloatArray:
        """Interpolate the symmetric profile per Hz as SYNSPEC does.

        The Schönning--Butler file tabulates positive wavelength displacement
        and logarithmic profiles.  SYNSPEC's ``INTHE2`` routine uses
        three-point quadratic interpolation in ``log10(T)`` and ``log10(Ne)``
        and linear interpolation in ``log10(abs(delta wavelength))``.  Outside
        the tabulated electron-density range it switches to its analytic
        Doppler-plus-Holtsmark approximation; that continuation is reproduced
        here as well.
        """

        wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
        if (
            np.any(~np.isfinite(wavelength))
            or np.any(wavelength <= 0.0)
            or not np.isfinite(line_center_angstrom)
            or line_center_angstrom <= 0.0
            or not np.isfinite(temperature)
            or temperature <= 0.0
            or not np.isfinite(electron_density)
            or electron_density <= 0.0
        ):
            raise ValueError("profile inputs must be finite and positive")
        log_t = float(np.log10(temperature))
        log_ne = float(np.log10(electron_density))
        requested_interpolation = self.thermodynamic_interpolation
        interpolation = requested_interpolation
        high_pickering_adaptive = (
            requested_interpolation == "series-adaptive"
            and self.lower_level == 4
            and self.upper_level >= 8
        )
        sparse_high_pickering = (
            high_pickering_adaptive
            and self.log_electron_density[-1] <= 16.0 + 1.0e-12
        )
        if requested_interpolation == "series-adaptive":
            # Quadratic interpolation best preserves the low-member core and
            # wings, while the sparse high-Pickering tables develop large
            # quadratic overshoots.  Use the monotone log-bilinear surface for
            # the crowded 4->n, n>=8 profiles only.  This choice applies only
            # *inside* the table: beyond its density range SYNSPEC uses the
            # analytic Doppler--Holtsmark continuation.  Clamping the sparse
            # n>=8 tables at their upper boundary (usually Ne=1e16 cm^-3)
            # produces severely over-strong Pickering lines in white dwarfs.
            interpolation = (
                "log-bilinear"
                if high_pickering_adaptive
                else "synspec-quadratic"
            )
        high_density_blend_fraction = 0.0
        if sparse_high_pickering and log_ne > self.log_electron_density[-1]:
            # The n>=9 Pickering tables stop one decade earlier than all of
            # the lower-series tables (log Ne=16 instead of 17).  SYNSPEC's
            # hard switch to STARKA at 1.01*log(Ne_max) makes the profile jump
            # by factors of several.  Retain the tabulated boundary shape and
            # join it continuously, in log profile, to the analytic high-
            # density limit over the missing 16--17 decade.  This is the
            # density interval that the adjacent 4->8 table actually covers.
            blend_coordinate = float(np.clip(
                log_ne - self.log_electron_density[-1], 0.0, 1.0
            ))
            high_density_blend_fraction = (
                blend_coordinate**2 * (3.0 - 2.0 * blend_coordinate)
            )
        outside_density_table = (
            log_ne < 0.99 * self.log_electron_density[0]
            or log_ne > 1.01 * self.log_electron_density[-1]
        )
        use_analytic = outside_density_table and (
            interpolation == "synspec-quadratic" or high_pickering_adaptive
        )
        if sparse_high_pickering:
            use_analytic = (
                log_ne < 0.99 * self.log_electron_density[0]
                or high_density_blend_fraction >= 1.0
            )
        if (
            not use_analytic
            and log_t > 1.01 * self.log_temperature[-1]
            and (interpolation == "synspec-quadratic" or high_pickering_adaptive)
        ):
            _, beta_doppler = _synspec_helium_ii_stark_scales(
                self,
                line_center_angstrom,
                temperature,
                electron_density,
            )
            use_analytic = beta_doppler > 10.0
        if (
            not use_analytic
            and interpolation == "synspec-quadratic"
        ):
            temperature_profile = _quadratic_interpolate(
                self.log_temperature,
                np.moveaxis(self.log_profile_per_hz, 1, 0),
                log_t,
            )
            local = _quadratic_interpolate(
                self.log_electron_density,
                temperature_profile,
                log_ne,
            )
        elif interpolation == "log-bilinear":
            clipped_log_t = float(np.clip(
                log_t, self.log_temperature[0], self.log_temperature[-1]
            ))
            clipped_log_ne = float(np.clip(
                log_ne,
                self.log_electron_density[0],
                self.log_electron_density[-1],
            ))
            t_lower, t_fraction = _bracket(
                self.log_temperature, clipped_log_t
            )
            ne_lower, ne_fraction = _bracket(
                self.log_electron_density, clipped_log_ne
            )
            local = np.zeros(
                self.wavelength_offset_angstrom.size, dtype=np.float64
            )
            for ne_index, ne_weight in (
                (ne_lower, 1.0 - ne_fraction),
                (ne_lower + 1, ne_fraction),
            ):
                for t_index, t_weight in (
                    (t_lower, 1.0 - t_fraction),
                    (t_lower + 1, t_fraction),
                ):
                    local += (
                        ne_weight
                        * t_weight
                        * self.log_profile_per_hz[ne_index, t_index]
                    )
            if high_density_blend_fraction > 0.0:
                analytic = np.log10(
                    _synspec_helium_ii_analytic_profile(
                        self,
                        line_center_angstrom,
                        temperature,
                        electron_density,
                    )
                )
                local = (
                    (1.0 - high_density_blend_fraction) * local
                    + high_density_blend_fraction * analytic
                )
        if use_analytic:
            local = np.log10(
                _synspec_helium_ii_analytic_profile(
                    self,
                    line_center_angstrom,
                    temperature,
                    electron_density,
                )
            )
        offset = np.abs(wavelength - line_center_angstrom)
        log_offset_grid = np.log10(
            np.maximum(self.wavelength_offset_angstrom, 1.0e-4)
        )
        query = np.log10(np.maximum(offset, 1.0e-4))
        result = _linear_interpolate_with_extrapolation(
            log_offset_grid, local, query
        )
        return np.power(10.0, result)


@dataclass(frozen=True)
class HeliumIIStarkTable:
    lines: dict[tuple[int, int], HeliumIIStarkLine]
    source_path: Path

    def __getitem__(self, transition: tuple[int, int]) -> HeliumIIStarkLine:
        return self.lines[transition]


def _bracket(grid: FloatArray, value: float) -> tuple[int, float]:
    upper = int(np.searchsorted(grid, value, side="right"))
    lower = min(max(upper - 1, 0), grid.size - 2)
    fraction = (value - grid[lower]) / (grid[lower + 1] - grid[lower])
    return lower, float(np.clip(fraction, 0.0, 1.0))


def _quadratic_interpolate(
    grid: FloatArray, values: FloatArray, value: float
) -> FloatArray:
    """Three-point Lagrange interpolation matching SYNSPEC's ``YINT`` use."""

    if grid.size < 3 or values.shape[0] != grid.size:
        raise ValueError("quadratic interpolation requires at least three grid points")
    lower = int(np.searchsorted(grid, value, side="left")) - 1
    start = min(max(lower, 0), grid.size - 3)
    selected_grid = grid[start : start + 3]
    selected_values = values[start : start + 3]
    weights = np.ones(3, dtype=np.float64)
    for index in range(3):
        for other in range(3):
            if other != index:
                weights[index] *= (
                    (value - selected_grid[other])
                    / (selected_grid[index] - selected_grid[other])
                )
    return np.tensordot(weights, selected_values, axes=(0, 0))


def _linear_interpolate_with_extrapolation(
    grid: FloatArray, values: FloatArray, query: FloatArray
) -> FloatArray:
    """Piecewise-linear interpolation with the end slopes continued."""

    upper = np.searchsorted(grid, query, side="left")
    lower = np.clip(upper - 1, 0, grid.size - 2)
    fraction = (query - grid[lower]) / (grid[lower + 1] - grid[lower])
    return values[lower] + fraction * (values[lower + 1] - values[lower])


def _synspec_helium_ii_stark_scales(
    line: HeliumIIStarkLine,
    line_center_angstrom: float,
    temperature: float,
    electron_density: float,
) -> tuple[float, float]:
    """Return SYNSPEC's wavelength scale ``FXK`` and Doppler width in beta."""

    offset = max(float(line.wavelength_offset_angstrom[-1]), 1.0e-4)
    x_c_log = (
        float(line.log_profile_per_hz[0, 0, -1])
        + 2.5 * np.log10(offset)
        + 31.831
        - float(line.log_electron_density[0])
        - 2.0 * np.log10(line_center_angstrom)
    )
    x_k = 10.0 ** ((2.0 / 3.0) * (x_c_log - 0.176))
    f_x_k = 1.25e-9 * electron_density ** (2.0 / 3.0) * x_k
    doppler = (
        1.0e8
        / line_center_angstrom
        * np.sqrt(4.12e7 * temperature)
    )
    d_beta = (
        line_center_angstrom**2 / 2.997925e18 / f_x_k
    )
    return float(f_x_k), float(d_beta * doppler)


def _synspec_helium_ii_analytic_profile(
    line: HeliumIIStarkLine,
    line_center_angstrom: float,
    temperature: float,
    electron_density: float,
) -> FloatArray:
    """SYNSPEC ``STARKA`` continuation evaluated on a table's offset grid."""

    f_x_k, beta_doppler = _synspec_helium_ii_stark_scales(
        line, line_center_angstrom, temperature, electron_density
    )
    d_beta = (
        line_center_angstrom**2 / 2.997925e18 / f_x_k
    )
    beta = np.maximum(line.wavelength_offset_angstrom, 1.0e-4) / f_x_k
    auxiliary = 1.5 * np.log(beta_doppler) - 0.978
    division = 0.0
    if beta_doppler >= 5.821:
        if auxiliary >= 1.26:
            division = np.sqrt(auxiliary) * (
                1.0
                + 1.25 * np.log(auxiliary) / (4.0 * auxiliary - 5.0)
            )
        else:
            division = np.sqrt(0.28 + auxiliary)
        for _ in range(5):
            next_division = division * (
                1.0
                - (
                    division**2 - 2.5 * np.log(division) - auxiliary
                )
                / (2.0 * division**2 - 2.5)
            )
            if abs(next_division - division) <= 1.0e-4:
                division = next_division
                break
            division = next_division

    result = np.empty_like(beta)
    if auxiliary > 1.26:
        core = beta / beta_doppler <= division
        result[core] = (
            0.5641895
            * np.exp(-(beta[core] / beta_doppler) ** 2)
            / beta_doppler
        )
        result[~core] = 1.5 * beta[~core] ** -2.5
    else:
        inner = beta <= 1.52
        middle = (beta > 1.52) & (beta < 8.325)
        outer = ~(inner | middle)
        result[inner] = 0.07966 / 2.0
        logarithm = np.log(beta[middle])
        result[middle] = (
            0.07209481
            / 2.0
            * np.exp((-0.5758228 * logarithm + 0.4796232) * logarithm)
        )
        result[outer] = 1.5 * beta[outer] ** -2.5
    return np.maximum(result * d_beta, np.finfo(np.float64).tiny)


def _consume_values(
    lines: list[str], cursor: int, count: int, initial: list[str] | None = None
) -> tuple[FloatArray, int]:
    tokens = [] if initial is None else list(initial)
    while len(tokens) < count:
        if cursor >= len(lines):
            raise ValueError("truncated He II Stark table")
        tokens.extend(lines[cursor].split())
        cursor += 1
    if len(tokens) != count:
        raise ValueError("unexpected extra values in He II Stark-table row")
    values = np.asarray(tokens, dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("non-finite He II Stark-table value")
    return values, cursor


def read_helium_ii_stark_table(
    path: str | Path,
    *,
    thermodynamic_interpolation: ThermodynamicInterpolation = "synspec-quadratic",
) -> HeliumIIStarkTable:
    """Read SYNSPEC's public ``he2prf.dat`` profile file.

    ``synspec-quadratic`` reproduces ``INTHE2``.  ``log-bilinear`` retains a
    controlled comparison mode.  ``series-adaptive`` keeps the quadratic
    interpolation for low members and uses the monotone bilinear surface for
    the sparse 4-to-n tables at n>=8, where quadratic overshoot is severe.  For
    the n>=9 tables, which stop at log Ne=16, it smoothly joins the tabulated
    boundary to the analytic continuation over the missing 16--17 decade.
    """

    if thermodynamic_interpolation not in (
        "synspec-quadratic",
        "log-bilinear",
        "series-adaptive",
    ):
        raise ValueError("unknown He II thermodynamic interpolation")

    source = Path(path)
    lines = source.read_text(encoding="ascii").splitlines()
    cursor = 0
    profiles: dict[tuple[int, int], HeliumIIStarkLine] = {}
    marker = re.compile(
        r"^\s*HEII\s+HE2\s*(\d+)\s+HE2\s*(\d+)\s+0\s*$"
    )
    while cursor < len(lines):
        match = marker.match(lines[cursor])
        cursor += 1
        if match is None:
            continue
        lower, upper = int(match.group(1)), int(match.group(2))
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        header = lines[cursor].split()
        cursor += 1
        n_offset = int(header[0])
        offset, cursor = _consume_values(
            lines, cursor, n_offset, initial=header[1:]
        )
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        temperature_header = lines[cursor].split()
        cursor += 1
        if temperature_header[0] != "T":
            raise ValueError("missing He II temperature grid")
        n_temperature = int(temperature_header[1])
        log_temperature, cursor = _consume_values(
            lines, cursor, n_temperature, initial=temperature_header[2:]
        )
        electron_header = lines[cursor].split()
        cursor += 1
        if electron_header[0] != "E":
            raise ValueError("missing He II electron-density grid")
        n_electron = int(electron_header[1])
        log_electron, cursor = _consume_values(
            lines, cursor, n_electron, initial=electron_header[2:]
        )
        values, cursor = _consume_values(
            lines, cursor, n_electron * n_temperature * n_offset
        )
        profiles[(lower, upper)] = HeliumIIStarkLine(
            lower,
            upper,
            offset,
            log_temperature,
            log_electron,
            values.reshape(n_electron, n_temperature, n_offset),
            thermodynamic_interpolation,
        )
    if len(profiles) != 19:
        raise ValueError(
            f"expected 19 He II profiles in {source}, found {len(profiles)}"
        )
    return HeliumIIStarkTable(profiles, source)
