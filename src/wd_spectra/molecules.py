"""Molecular-hydrogen equilibrium constants and ultraviolet cross sections."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import (
    BOLTZMANN,
    ELECTRON_MASS,
    HYDROGEN_MASS,
    LIGHT_SPEED,
    PI,
    PLANCK,
)


FloatArray = NDArray[np.float64]

BORYSOW_H2_H2_CIA_FILENAME = (
    "CIA_Borysow_H2H2_0060-7000K_0.6-500um.dat"
)
BORYSOW_H2_H2_CIA_SHA256 = (
    "a7f342a635e8c7092bb4d5ef30aca50f9de1a54e047b706f48d6c59237010fe2"
)
BORYSOW_H2_H2_CIA_URL = (
    "https://raw.githubusercontent.com/pcubillos/pyratbay/master/"
    "pyratbay/data/CIA/"
    + BORYSOW_H2_H2_CIA_FILENAME
)

# Loschmidt number used by the tabulated cm^-1 amagat^-2 coefficients.
LOSCHMIDT_NUMBER_DENSITY = 2.686_780_111e19


@dataclass(frozen=True)
class H2H2CollisionInducedAbsorptionTable:
    """Temperature-dependent H2--H2 CIA coefficients.

    ``absorption_coefficient`` has the native Borysow-table unit
    cm^-1 amagat^-2.  The opacity layer converts it to inverse length using
    ``(n(H2) / n_amagat)^2`` before dividing by mass density.
    """

    wavenumber_cm_inverse: FloatArray
    temperature_K: FloatArray
    absorption_coefficient: FloatArray
    source_path: Path
    extrapolate_high_wavenumber_tail: bool = False

    def __post_init__(self) -> None:
        wavenumber = np.asarray(self.wavenumber_cm_inverse, dtype=np.float64)
        temperature = np.asarray(self.temperature_K, dtype=np.float64)
        coefficient = np.asarray(self.absorption_coefficient, dtype=np.float64)
        if wavenumber.ndim != 1 or wavenumber.size < 2:
            raise ValueError("wavenumber_cm_inverse must be a non-trivial 1D grid")
        if temperature.ndim != 1 or temperature.size < 2:
            raise ValueError("temperature_K must be a non-trivial 1D grid")
        if np.any(~np.isfinite(wavenumber)) or np.any(np.diff(wavenumber) <= 0.0):
            raise ValueError("wavenumber_cm_inverse must be finite and increasing")
        if np.any(~np.isfinite(temperature)) or np.any(np.diff(temperature) <= 0.0):
            raise ValueError("temperature_K must be finite and increasing")
        if coefficient.shape != (wavenumber.size, temperature.size):
            raise ValueError(
                "absorption_coefficient must have shape (wavenumber, temperature)"
            )
        if np.any(~np.isfinite(coefficient)) or np.any(coefficient <= 0.0):
            raise ValueError("absorption_coefficient must be finite and positive")
        if self.extrapolate_high_wavenumber_tail and np.any(
            coefficient[-1] >= coefficient[-2]
        ):
            raise ValueError("CIA tail extrapolation requires declining terminal coefficients")

    def coefficient_for_wavelength_temperature(
        self,
        wavelength_angstrom: ArrayLike,
        temperature_K: ArrayLike,
    ) -> FloatArray:
        """Return bilinearly interpolated coefficients in cm^-1 amagat^-2.

        The result has shape ``(wavelength, temperature)``. Below the lowest
        tabulated wavenumber it is zero. With a declining high-wavenumber tail
        explicitly enabled, continue the terminal log-slope; otherwise values
        above the table are zero. This continuation is an extrapolation, not
        new opacity data. Temperatures remain clamped to the table range.
        """

        wavelength = np.atleast_1d(
            np.asarray(wavelength_angstrom, dtype=np.float64)
        )
        temperature = np.atleast_1d(np.asarray(temperature_K, dtype=np.float64))
        if wavelength.ndim != 1 or np.any(~np.isfinite(wavelength)) or np.any(
            wavelength <= 0.0
        ):
            raise ValueError("wavelength_angstrom must be a finite positive 1D array")
        if temperature.ndim != 1 or np.any(~np.isfinite(temperature)) or np.any(
            temperature <= 0.0
        ):
            raise ValueError("temperature_K must be a finite positive 1D array")

        wavenumber = 1.0e8 / wavelength
        result = np.zeros((wavelength.size, temperature.size), dtype=np.float64)
        inside = wavenumber >= self.wavenumber_cm_inverse[0]
        if not self.extrapolate_high_wavenumber_tail:
            inside &= wavenumber <= self.wavenumber_cm_inverse[-1]
        if not np.any(inside):
            return result

        sampled_wavenumber = wavenumber[inside]
        wavenumber_index = np.searchsorted(
            self.wavenumber_cm_inverse, sampled_wavenumber, side="right"
        ) - 1
        wavenumber_index = np.clip(
            wavenumber_index, 0, self.wavenumber_cm_inverse.size - 2
        )
        wavenumber_fraction = (
            sampled_wavenumber - self.wavenumber_cm_inverse[wavenumber_index]
        ) / (
            self.wavenumber_cm_inverse[wavenumber_index + 1]
            - self.wavenumber_cm_inverse[wavenumber_index]
        )

        sampled_temperature = np.clip(
            temperature, self.temperature_K[0], self.temperature_K[-1]
        )
        temperature_index = np.searchsorted(
            self.temperature_K, sampled_temperature, side="right"
        ) - 1
        temperature_index = np.clip(
            temperature_index, 0, self.temperature_K.size - 2
        )
        temperature_fraction = (
            sampled_temperature - self.temperature_K[temperature_index]
        ) / (
            self.temperature_K[temperature_index + 1]
            - self.temperature_K[temperature_index]
        )

        logarithm = np.log10(self.absorption_coefficient)
        lower_temperature = (
            (1.0 - wavenumber_fraction[:, np.newaxis])
            * logarithm[
                wavenumber_index[:, np.newaxis],
                temperature_index[np.newaxis, :],
            ]
            + wavenumber_fraction[:, np.newaxis]
            * logarithm[
                (wavenumber_index + 1)[:, np.newaxis],
                temperature_index[np.newaxis, :],
            ]
        )
        upper_temperature = (
            (1.0 - wavenumber_fraction[:, np.newaxis])
            * logarithm[
                wavenumber_index[:, np.newaxis],
                (temperature_index + 1)[np.newaxis, :],
            ]
            + wavenumber_fraction[:, np.newaxis]
            * logarithm[
                (wavenumber_index + 1)[:, np.newaxis],
                (temperature_index + 1)[np.newaxis, :],
            ]
        )
        interpolated = (
            (1.0 - temperature_fraction[np.newaxis, :]) * lower_temperature
            + temperature_fraction[np.newaxis, :] * upper_temperature
        )
        result[inside] = 10.0**interpolated
        return result


def read_borysow_h2_h2_cia_table(
    path: str | Path,
) -> H2H2CollisionInducedAbsorptionTable:
    """Read the merged Pyrat Bay/Borysow H2--H2 CIA text table."""

    source_path = Path(path)
    section: str | None = None
    temperatures: list[float] = []
    rows: list[list[float]] = []
    for raw_line in source_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            section = line.upper()
            continue
        if section == "@TEMPERATURES":
            temperatures.extend(float(value) for value in line.split())
        elif section == "@DATA":
            rows.append([float(value) for value in line.split()])

    if len(temperatures) < 2 or len(rows) < 2:
        raise ValueError(f"incomplete H2-H2 CIA table: {source_path}")
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(temperatures) + 1:
        raise ValueError(
            f"H2-H2 CIA rows in {source_path} do not match the temperature grid"
        )
    return H2H2CollisionInducedAbsorptionTable(
        wavenumber_cm_inverse=values[:, 0],
        temperature_K=np.asarray(temperatures, dtype=np.float64),
        absorption_coefficient=values[:, 1:],
        source_path=source_path,
        # The production Borysow merge ends at 16480 cm^-1 (6068 A) on a
        # declining tail in every temperature column. Continue its measured
        # terminal log-slope, with no fitted taper scale. A different input
        # table that ends on a rising/flat branch remains explicitly limited
        # to its tabulated domain; an increasing exponential is not justified.
        extrapolate_high_wavenumber_tail=bool(
            np.all(values[-1, 1:] < values[-2, 1:])
        ),
    )

# Ground-state dissociation energies.  The H2+ value refers to H + H+.
H2_DISSOCIATION_ENERGY = 4.4781 * 1.602176634e-12
H2_PLUS_DISSOCIATION_ENERGY = 2.6507 * 1.602176634e-12
H2_H_CIA_FILENAME = "h2_h_cia_hitran2011.cia.xz"
# SHA-256 of the decompressed HITRAN file H2-H_2011.cia.
H2_H_CIA_SHA256 = (
    "0f68b263e20d99e25704a95eac489c1b58b37984974638fc8bcf9bfe86fe9f18"
)


def _molecular_data_path(name: str) -> Path:
    return Path(__file__).with_name("data") / "molecular" / name


@lru_cache(maxsize=1)
def _bell_h2_minus_free_free_table() -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return Bell (1980) H2- ff as (theta, log10 lambda_A, log10 k).

    ``k`` is in cm^4 dyn^-1 per H2 molecule per unit electron pressure.
    """

    path = _molecular_data_path("h2minus_ff_bell1980.csv")
    rows = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    wavelength = np.asarray([float(v) for v in rows[0].split(",")[1:]])
    values = np.asarray(
        [[float(v) for v in row.split(",")] for row in rows[1:]],
        dtype=np.float64,
    )
    theta = values[:, 0]
    coefficient = values[:, 1:].T * 1.0e-26
    order = np.argsort(wavelength)
    result = (
        np.ascontiguousarray(theta),
        np.ascontiguousarray(np.log10(wavelength[order])),
        np.ascontiguousarray(np.log10(coefficient[order])),
    )
    for array in result:
        array.setflags(write=False)
    return result


def h2_minus_free_free_coefficient_per_electron_pressure(
    wavelength_angstrom: ArrayLike, temperature: ArrayLike
) -> FloatArray:
    r"""Return Bell (1980) H2- free-free absorption in cm^4 dyn^-1.

    The result has shape ``(wavelength, temperature)``.  Multiplying it by the
    electron pressure and the H2 number density gives an absorption
    coefficient in cm^-1, with stimulated emission already included, exactly
    as for the John (1988) H- free-free term.  Bell's table spans
    0.35--15.2 micron and theta = 5040/T = 0.5--3.6 (1400--10080 K).  The
    logarithm is interpolated bilinearly in (theta, log lambda).  Outside the
    wavelength range it follows the terminal log-log slope of each end (the
    table is close to the free-free lambda^2 law there); temperatures are held
    at the table edges, where H2 is either negligible (hot) or below the
    present model range (cold).
    """

    wavelength = np.atleast_1d(np.asarray(wavelength_angstrom, dtype=np.float64))
    temperature = np.atleast_1d(_positive_temperature(temperature))
    if wavelength.ndim != 1 or np.any(~np.isfinite(wavelength)) or np.any(
        wavelength <= 0.0
    ):
        raise ValueError("wavelength_angstrom must be a finite positive 1D array")
    theta_grid, log_wavelength_grid, log_coefficient = (
        _bell_h2_minus_free_free_table()
    )
    theta = np.clip(5040.0 / temperature, theta_grid[0], theta_grid[-1])
    # Interpolate in theta first: (n_lambda_table, n_temperature).
    theta_index = np.clip(
        np.searchsorted(theta_grid, theta, side="right") - 1,
        0,
        theta_grid.size - 2,
    )
    theta_fraction = (theta - theta_grid[theta_index]) / (
        theta_grid[theta_index + 1] - theta_grid[theta_index]
    )
    at_temperature = (
        (1.0 - theta_fraction) * log_coefficient[:, theta_index]
        + theta_fraction * log_coefficient[:, theta_index + 1]
    )
    log_wavelength = np.log10(wavelength)
    wavelength_index = np.clip(
        np.searchsorted(log_wavelength_grid, log_wavelength, side="right") - 1,
        0,
        log_wavelength_grid.size - 2,
    )
    wavelength_fraction = (
        log_wavelength - log_wavelength_grid[wavelength_index]
    ) / (
        log_wavelength_grid[wavelength_index + 1]
        - log_wavelength_grid[wavelength_index]
    )
    interpolated = (
        (1.0 - wavelength_fraction[:, np.newaxis])
        * at_temperature[wavelength_index]
        + wavelength_fraction[:, np.newaxis]
        * at_temperature[wavelength_index + 1]
    )
    return 10.0**interpolated


@dataclass(frozen=True)
class H2HCollisionInducedAbsorptionTable:
    """HITRAN H2--H collision-induced absorption (Gustafsson & Frommhold 2003).

    ``absorption_coefficient`` is the binary coefficient in cm^5 molecule^-2,
    shape ``(wavenumber, temperature)``; multiplying by n(H2) n(H) gives
    cm^-1.  HITRAN's binary coefficients are net absorption, so no separate
    stimulated-emission factor is applied.
    """

    wavenumber_cm_inverse: FloatArray
    temperature_K: FloatArray
    absorption_coefficient: FloatArray

    def coefficient_for_wavelength_temperature(
        self, wavelength_angstrom: ArrayLike, temperature_K: ArrayLike
    ) -> FloatArray:
        """Return interpolated coefficients, shape ``(wavelength, temperature)``.

        The logarithm is interpolated linearly in wavenumber and temperature.
        The calculation covers 1000--2500 K and 100--10000 cm^-1 only.  Hotter
        layers use the 2500 K column (photospheres of 4000--6000 K DA models
        lie above the published range; the coefficient grows with temperature,
        so this underestimates rather than invents opacity).  Above 10000 cm^-1
        (blueward of 1 micron) every column declines, and its terminal
        log-slope is continued as for the Borysow H2--H2 tail; below
        100 cm^-1 the coefficient is zero.
        """

        wavelength = np.atleast_1d(np.asarray(wavelength_angstrom, dtype=np.float64))
        temperature = np.atleast_1d(np.asarray(temperature_K, dtype=np.float64))
        if wavelength.ndim != 1 or np.any(~np.isfinite(wavelength)) or np.any(
            wavelength <= 0.0
        ):
            raise ValueError("wavelength_angstrom must be a finite positive 1D array")
        temperature = np.clip(
            temperature, self.temperature_K[0], self.temperature_K[-1]
        )
        wavenumber = 1.0e8 / wavelength
        grid = self.wavenumber_cm_inverse
        log_coefficient = np.log10(self.absorption_coefficient)
        temperature_index = np.clip(
            np.searchsorted(self.temperature_K, temperature, side="right") - 1,
            0,
            self.temperature_K.size - 2,
        )
        temperature_fraction = (
            temperature - self.temperature_K[temperature_index]
        ) / (
            self.temperature_K[temperature_index + 1]
            - self.temperature_K[temperature_index]
        )
        at_temperature = (
            (1.0 - temperature_fraction) * log_coefficient[:, temperature_index]
            + temperature_fraction * log_coefficient[:, temperature_index + 1]
        )
        result = np.zeros((wavelength.size, temperature.size), dtype=np.float64)
        inside = wavenumber >= grid[0]
        if not np.any(inside):
            return result
        sampled = wavenumber[inside]
        index = np.clip(np.searchsorted(grid, sampled, side="right") - 1, 0, grid.size - 2)
        fraction = (sampled - grid[index]) / (grid[index + 1] - grid[index])
        result[inside] = 10.0 ** (
            (1.0 - fraction[:, np.newaxis]) * at_temperature[index]
            + fraction[:, np.newaxis] * at_temperature[index + 1]
        )
        return result


@lru_cache(maxsize=1)
def h2_h_cia_table() -> H2HCollisionInducedAbsorptionTable:
    """Load the bundled HITRAN 2011 H2--H CIA file after checking its hash."""

    import hashlib
    import lzma

    raw = lzma.decompress(_molecular_data_path(H2_H_CIA_FILENAME).read_bytes())
    if hashlib.sha256(raw).hexdigest() != H2_H_CIA_SHA256:
        raise ValueError("bundled H2-H CIA table failed its checksum")
    lines = raw.decode("ascii").splitlines()
    temperatures: list[float] = []
    columns: list[FloatArray] = []
    wavenumber: FloatArray | None = None
    position = 0
    while position < len(lines):
        header = lines[position].split()
        if not header:
            position += 1
            continue
        count = int(header[3])
        temperatures.append(float(header[4]))
        block = np.asarray(
            [line.split() for line in lines[position + 1 : position + 1 + count]],
            dtype=np.float64,
        )
        if wavenumber is None:
            wavenumber = block[:, 0]
        elif not np.array_equal(wavenumber, block[:, 0]):
            raise ValueError("H2-H CIA blocks use different wavenumber grids")
        columns.append(block[:, 1])
        position += 1 + count
    assert wavenumber is not None
    coefficient = np.column_stack(columns)
    if np.any(coefficient <= 0.0) or np.any(np.diff(temperatures) <= 0.0):
        raise ValueError("unexpected H2-H CIA table contents")
    if np.any(coefficient[-1] >= coefficient[-2]):
        raise ValueError("H2-H CIA tail must decline for its continuation")
    # Continue every column past its last point along its terminal log-slope
    # by appending one far point (1000 Angstrom, where it is already ~70 dex
    # down); log-linear interpolation then reproduces the extrapolation
    # without a special case.
    far = 1.0e5
    slope = (np.log10(coefficient[-1]) - np.log10(coefficient[-2])) / (
        wavenumber[-1] - wavenumber[-2]
    )
    far_values = 10.0 ** (np.log10(coefficient[-1]) + slope * (far - wavenumber[-1]))
    wavenumber = np.r_[wavenumber, far]
    coefficient = np.vstack([coefficient, far_values])
    return H2HCollisionInducedAbsorptionTable(
        wavenumber_cm_inverse=wavenumber,
        temperature_K=np.asarray(temperatures, dtype=np.float64),
        absorption_coefficient=coefficient,
    )


# Zero-kelvin dissociation energy of H3+ into H2(v=0, J=0) + H+, measured
# from the lowest Pauli-allowed H3+ level (J=1, K=1).  This is the energy
# zero of the Neale & Tennyson (1995) partition function used below.  The
# former 4.377 eV is the 298 K proton affinity of H2 (an enthalpy), not the
# 0 K energy the equilibrium constant needs.
#
# Mizus et al. (2019, arXiv:1812.06035, Sec. 3) give the
# Born--Oppenheimer value from the vibrationless J=0 ground state,
# D0 = De - ZPE(H3+) + ZPE(H2) = 37195.3 - 4362.76 + 2179.3
# = 35011.8 cm^-1, and the non-BO correction to it is below 1 cm^-1.  The
# J=1, K=1 level lies 64.21 cm^-1 above J=0, which lowers the dissociation
# energy from that level to 34947.6 cm^-1.  Their quoted total, 35076 cm^-1,
# equals 35011.8 + 64.21: the Pauli offset there was added to, rather than
# subtracted from, the dissociation energy.  Converting the 298 K proton
# affinity (NIST, 422.3 kJ/mol) to 0 K with ideal-gas enthalpies gives an
# independent ~4.32 eV, consistent with the value adopted here.
H3_PLUS_DISSOCIATION_ENERGY_CM_INVERSE = 35011.8 - 64.21
H3_PLUS_DISSOCIATION_ENERGY = (
    H3_PLUS_DISSOCIATION_ENERGY_CM_INVERSE * PLANCK * LIGHT_SPEED
)
# The Neale & Tennyson (1995) H3+ partition function counts every one of the
# 2^3 proton-spin states (ExoMol's NT readme; para levels carry g_ns = 2 and
# ortho levels g_ns = 4).  The Barklem--Collet H2 partition function and the
# atomic H/H+ terms of this EOS divide the nuclear-spin degeneracy out
# (Q(H2) -> 1/4 as T -> 0).  A consistent H3+ equilibrium therefore uses
# Q_NT / 8.  Kilic et al. (2026) report and correct the same mismatch in the
# Montreal codes.
H3_PLUS_NUCLEAR_SPIN_STATES = 8.0
# Electron affinity for H + e <-> H-.
H_MINUS_DETACHMENT_ENERGY = 0.754195 * 1.602176634e-12

# Characteristic rigid-rotor and harmonic-oscillator temperatures of the
# H2+ X 2Sigma_g+ ground state.
H2_PLUS_ROTATIONAL_TEMPERATURE = 43.1
H2_PLUS_VIBRATIONAL_TEMPERATURE = 3340.0


def _positive_temperature(temperature: ArrayLike) -> FloatArray:
    values = np.asarray(temperature, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("temperature must contain finite positive values")
    return values


def _equilibrium_rotational_partition(
    temperature: FloatArray, rotational_temperature: float
) -> FloatArray:
    """Return the equilibrium ortho/para rotational partition function.

    Even rotational levels carry para-H nuclear-spin weight one and odd
    levels carry ortho-H weight three.  Summing the levels explicitly avoids
    relying on the high-temperature approximation in cool surface layers.
    """

    level = np.arange(0, 81, dtype=np.float64)
    spin_weight = np.where(level.astype(np.int64) % 2 == 0, 1.0, 3.0)
    energy_over_k = rotational_temperature * level * (level + 1.0)
    return np.sum(
        spin_weight
        * (2.0 * level + 1.0)
        * np.exp(-energy_over_k / temperature[..., np.newaxis]),
        axis=-1,
    )


@lru_cache(maxsize=1)
def _molecular_hydrogen_partition_table() -> tuple[FloatArray, FloatArray]:
    """Load the Barklem--Collet H2 partition-function table."""

    path = (
        Path(__file__).with_name("data")
        / "molecular"
        / "barklem_collet_h2_partition.csv"
    )
    values = np.loadtxt(path, delimiter=",", comments="#")
    temperature = np.ascontiguousarray(values[:, 0])
    partition = np.ascontiguousarray(values[:, 1])
    temperature.setflags(write=False)
    partition.setflags(write=False)
    return temperature, partition


def molecular_hydrogen_internal_partition_function(
    temperature: ArrayLike,
) -> FloatArray:
    """Return the Barklem--Collet equilibrium H2 partition function.

    Their direct state sum includes the available rovibrational and
    electronic levels and uses the same division by the nuclear statistical
    weight as this EOS.  Logarithmic interpolation is used over the published
    near-zero to 10,000 K grid.  The upper endpoint is retained above that
    range, where H2 is already negligible in the present atmosphere models.
    """

    temperature = _positive_temperature(temperature)
    table_temperature, table_partition = _molecular_hydrogen_partition_table()
    return np.exp(
        np.interp(
            np.log(temperature),
            np.log(table_temperature),
            np.log(table_partition),
        )
    )


def molecular_hydrogen_ion_internal_partition_function(
    temperature: ArrayLike,
) -> FloatArray:
    """Return the ground-electronic-state H2+ internal partition function."""

    temperature = _positive_temperature(temperature)
    rotation = _equilibrium_rotational_partition(
        temperature, H2_PLUS_ROTATIONAL_TEMPERATURE
    )
    vibration = 1.0 / -np.expm1(
        -H2_PLUS_VIBRATIONAL_TEMPERATURE / temperature
    )
    # 0.5 = 2 / 4: the X 2Sigma_g+ electronic spin degeneracy (two) times
    # the division by the 2^2 proton-spin states used throughout this EOS
    # (the ortho/para weights 1 and 3 above sum to those four states).  It is
    # not an error for "degeneracy two"; see H3_PLUS_NUCLEAR_SPIN_STATES.
    return 0.5 * rotation * vibration


def trihydrogen_ion_internal_partition_function(
    temperature: ArrayLike,
    *,
    model: str = "neale-tennyson-1995",
) -> FloatArray:
    r"""Return the H3+ internal partition function.

    The Neale & Tennyson (1995) fit explicitly accounts for bound levels up
    to the H3+ dissociation limit and is valid to about one per cent over
    500--8000 K.  Its energy zero is the lowest allowed rotational state.
    Evaluation of the published sixth-order polynomial is retained through
    10,000 K; the endpoint is held fixed at higher temperature, where H2 and
    H3+ are already negligible in the present DA calculations.

    This returns the published value, which includes all eight proton-spin
    states.  The chemical equilibrium divides it by
    ``H3_PLUS_NUCLEAR_SPIN_STATES`` to match the H2 convention.

    The much smaller partition function advocated by Kylänpää & Rantala
    (2011), and the possibility of unaccounted negative molecular ions, make
    H3+ chemical equilibrium an active uncertainty.  Callers must therefore
    request this named model explicitly rather than receiving an undocumented
    scale factor.
    """

    values = _positive_temperature(temperature)
    if model != "neale-tennyson-1995":
        raise ValueError(
            "trihydrogen-ion partition model must be "
            "'neale-tennyson-1995'"
        )
    logarithmic_temperature = np.log10(np.clip(values, 500.0, 10_000.0))
    coefficients = np.asarray(
        [
            78.6233962485680706,
            -134.822002886523251,
            88.4482694968956480,
            -25.9274134010262429,
            2.60233376654769222,
            0.224167420795110400,
            -0.0452550693680233290,
        ],
        dtype=np.float64,
    )
    log10_partition = np.polynomial.polynomial.polyval(
        logarithmic_temperature, coefficients
    )
    return 10.0**log10_partition


def trihydrogen_ion_dissociation_constant(
    temperature: ArrayLike,
    *,
    partition_model: str = "neale-tennyson-1995",
) -> FloatArray:
    r"""Return ``n(H2) n(H+) / n(H3+)`` in cm^-3.

    This is the ideal translational equilibrium for dissociation into H2 and
    a proton.  The reduced mass is 2/3 of a hydrogen-atom mass.  The H2
    partition function uses the same Barklem--Collet convention as the rest
    of the molecular EOS, with the proton-spin degeneracy divided out; the
    Neale & Tennyson H3+ value is divided by its eight proton-spin states to
    match.
    """

    temperature = _positive_temperature(temperature)
    reduced_mass = (2.0 / 3.0) * HYDROGEN_MASS
    translation = (
        2.0 * PI * reduced_mass * BOLTZMANN * temperature / PLANCK**2
    ) ** 1.5
    logarithm = (
        np.log(translation)
        + np.log(molecular_hydrogen_internal_partition_function(temperature))
        - np.log(
            trihydrogen_ion_internal_partition_function(
                temperature, model=partition_model
            )
            / H3_PLUS_NUCLEAR_SPIN_STATES
        )
        - H3_PLUS_DISSOCIATION_ENERGY / (BOLTZMANN * temperature)
    )
    limits = np.log([np.finfo(np.float64).tiny, np.finfo(np.float64).max])
    return np.exp(np.clip(logarithm, limits[0], limits[1]))


def negative_hydrogen_ionization_constant(
    temperature: ArrayLike,
    *,
    atomic_internal_partition_function: ArrayLike = 1.0,
) -> FloatArray:
    r"""Return ``n(H) n(e) / n(H-)`` in cm^-3.

    H- has a singlet ground state.  The factor four is the free-electron spin
    degeneracy times the absolute H I electronic partition ``2 U_H`` used by
    the occupation-probability EOS.
    """

    temperature, atomic_partition = np.broadcast_arrays(
        _positive_temperature(temperature),
        np.asarray(atomic_internal_partition_function, dtype=np.float64),
    )
    if np.any(~np.isfinite(atomic_partition)) or np.any(atomic_partition <= 0.0):
        raise ValueError("atomic_internal_partition_function must be positive")
    translation = (
        2.0 * PI * ELECTRON_MASS * BOLTZMANN * temperature / PLANCK**2
    ) ** 1.5
    logarithm = (
        np.log(translation)
        + np.log(4.0 * atomic_partition)
        - H_MINUS_DETACHMENT_ENERGY / (BOLTZMANN * temperature)
    )
    limits = np.log([np.finfo(np.float64).tiny, np.finfo(np.float64).max])
    return np.exp(np.clip(logarithm, limits[0], limits[1]))


def trihydrogen_ion_rovibrational_energy(
    temperature: ArrayLike,
    *,
    partition_model: str = "neale-tennyson-1995",
) -> FloatArray:
    r"""Return H3+ thermal internal energy per ion in erg.

    For a partition function ``Q``, the excitation energy is
    ``k T d ln(Q) / d ln(T)``.  A centered logarithmic derivative of the
    published analytic fit keeps the caloric and chemical EOS consistent.
    """

    temperature = _positive_temperature(temperature)
    epsilon = 1.0e-4
    lower = trihydrogen_ion_internal_partition_function(
        temperature * np.exp(-epsilon), model=partition_model
    )
    upper = trihydrogen_ion_internal_partition_function(
        temperature * np.exp(epsilon), model=partition_model
    )
    derivative = (np.log(upper) - np.log(lower)) / (2.0 * epsilon)
    return BOLTZMANN * temperature * derivative


def molecular_hydrogen_dissociation_constant(
    temperature: ArrayLike,
    *,
    atomic_internal_partition_function: ArrayLike = 1.0,
) -> FloatArray:
    r"""Return ``n(H)**2 / n(H2)`` in cm^-3.

    The atomic partition supplied by the HM EOS is ground-state-relative, so
    the absolute electronic partition of H is ``2 U_H``.  Nuclear-spin
    weights are retained in the equilibrium ortho/para molecular sum.
    """

    temperature, atomic_partition = np.broadcast_arrays(
        _positive_temperature(temperature),
        np.asarray(atomic_internal_partition_function, dtype=np.float64),
    )
    if np.any(~np.isfinite(atomic_partition)) or np.any(atomic_partition <= 0.0):
        raise ValueError("atomic_internal_partition_function must be positive")
    reduced_mass = 0.5 * HYDROGEN_MASS
    translation = (
        2.0 * PI * reduced_mass * BOLTZMANN * temperature / PLANCK**2
    ) ** 1.5
    logarithm = (
        np.log(translation)
        + 2.0 * np.log(2.0 * atomic_partition)
        - np.log(molecular_hydrogen_internal_partition_function(temperature))
        - H2_DISSOCIATION_ENERGY / (BOLTZMANN * temperature)
    )
    limits = np.log([np.finfo(np.float64).tiny, np.finfo(np.float64).max])
    return np.exp(np.clip(logarithm, limits[0], limits[1]))


def molecular_hydrogen_ion_dissociation_constant(
    temperature: ArrayLike,
    *,
    atomic_internal_partition_function: ArrayLike = 1.0,
) -> FloatArray:
    r"""Return ``n(H) n(H+) / n(H2+)`` in cm^-3."""

    temperature, atomic_partition = np.broadcast_arrays(
        _positive_temperature(temperature),
        np.asarray(atomic_internal_partition_function, dtype=np.float64),
    )
    if np.any(~np.isfinite(atomic_partition)) or np.any(atomic_partition <= 0.0):
        raise ValueError("atomic_internal_partition_function must be positive")
    reduced_mass = 0.5 * HYDROGEN_MASS
    translation = (
        2.0 * PI * reduced_mass * BOLTZMANN * temperature / PLANCK**2
    ) ** 1.5
    logarithm = (
        np.log(translation)
        + np.log(2.0 * atomic_partition)
        - np.log(
            molecular_hydrogen_ion_internal_partition_function(temperature)
        )
        - H2_PLUS_DISSOCIATION_ENERGY / (BOLTZMANN * temperature)
    )
    limits = np.log([np.finfo(np.float64).tiny, np.finfo(np.float64).max])
    return np.exp(np.clip(logarithm, limits[0], limits[1]))


def _molecular_rovibrational_energy_per_particle(
    temperature: ArrayLike,
    rotational_temperature: float,
    vibrational_temperature: float,
) -> FloatArray:
    """Return thermal rotational plus vibrational energy in erg."""

    temperature = _positive_temperature(temperature)
    level = np.arange(0, 81, dtype=np.float64)
    spin_weight = np.where(level.astype(np.int64) % 2 == 0, 1.0, 3.0)
    energy_over_k = rotational_temperature * level * (level + 1.0)
    weights = (
        spin_weight
        * (2.0 * level + 1.0)
        * np.exp(-energy_over_k / temperature[..., np.newaxis])
    )
    rotational_energy = BOLTZMANN * np.sum(
        weights * energy_over_k, axis=-1
    ) / np.sum(weights, axis=-1)
    vibrational_energy = (
        BOLTZMANN
        * vibrational_temperature
        / np.expm1(vibrational_temperature / temperature)
    )
    return rotational_energy + vibrational_energy


def molecular_hydrogen_rovibrational_energy(
    temperature: ArrayLike,
) -> FloatArray:
    """Return H2 internal excitation energy per molecule in erg.

    The thermodynamic identity ``E = kT d(ln Q)/d(ln T)`` keeps the energy
    consistent with the Barklem--Collet partition function used in chemical
    equilibrium.  Slopes are interpolated on the same logarithmic grid.
    """

    temperature = _positive_temperature(temperature)
    table_temperature, table_partition = _molecular_hydrogen_partition_table()
    log_temperature = np.log(table_temperature)
    log_partition = np.log(table_partition)
    logarithmic_slope = np.gradient(log_partition, log_temperature)
    sampled_slope = np.interp(
        np.log(temperature),
        log_temperature,
        logarithmic_slope,
        left=0.0,
        right=logarithmic_slope[-1],
    )
    return BOLTZMANN * temperature * sampled_slope


def molecular_hydrogen_ion_rovibrational_energy(
    temperature: ArrayLike,
) -> FloatArray:
    """Return thermal H2+ rovibrational energy per ion in erg."""

    return _molecular_rovibrational_energy_per_particle(
        temperature,
        H2_PLUS_ROTATIONAL_TEMPERATURE,
        H2_PLUS_VIBRATIONAL_TEMPERATURE,
    )


@lru_cache(maxsize=2)
def _photoabsorption_table(species: str) -> tuple[FloatArray, FloatArray]:
    filename = {
        "h2": "h2_photoabsorption_1nm.csv",
        "h2plus": "h2plus_photodissociation.csv",
    }[species]
    path = Path(__file__).with_name("data") / "molecular" / filename
    values = np.loadtxt(path, delimiter=",", comments="#")
    wavelength_angstrom = np.ascontiguousarray(values[:, 0] * 10.0)
    cross_section = np.ascontiguousarray(values[:, 1])
    wavelength_angstrom.setflags(write=False)
    cross_section.setflags(write=False)
    return wavelength_angstrom, cross_section


def molecular_hydrogen_photoabsorption_cross_section(
    wavelength_angstrom: ArrayLike,
) -> FloatArray:
    """Return the 1-nm-binned Leiden H2 photoabsorption cross section."""

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_angstrom must be finite and positive")
    table_wavelength, table_cross_section = _photoabsorption_table("h2")
    return np.interp(
        wavelength, table_wavelength, table_cross_section, left=0.0, right=0.0
    )


def molecular_hydrogen_ion_photoabsorption_cross_section(
    wavelength_angstrom: ArrayLike,
) -> FloatArray:
    """Return the Leiden v=0 H2+ photodissociation cross section."""

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_angstrom must be finite and positive")
    table_wavelength, table_cross_section = _photoabsorption_table("h2plus")
    return np.interp(
        wavelength, table_wavelength, table_cross_section, left=0.0, right=0.0
    )
