"""H2--He CIA from the Abel et al. HITRAN table (native units cm5)."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
from itertools import islice
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class H2HeCIATable:
    wavenumber: np.ndarray
    temperature: np.ndarray
    coefficient_cm5: np.ndarray
    source_path: Path
    source_sha256: str | None = None
    _log_coefficient: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        w, t, c = self.wavenumber, self.temperature, self.coefficient_cm5
        if (
            w.ndim != 1
            or t.ndim != 1
            or min(w.size, t.size) < 2
            or c.shape != (w.size, t.size)
            or np.any(~np.isfinite(w))
            or np.any(~np.isfinite(t))
            or np.any(~np.isfinite(c))
            or np.any(np.diff(w) <= 0)
            or np.any(np.diff(t) <= 0)
            or np.any(c < 0)
            or w[0] <= 0
            or t[0] <= 0
        ):
            raise ValueError("Invalid H2-He CIA grid")
        object.__setattr__(
            self, "_log_coefficient", np.log(np.maximum(c, np.finfo(float).tiny))
        )

    def evaluate(self, wavelength, temperature):
        """Bilinear log interpolation; exact-zero cells use linear interpolation.

        Temperature holds at the native endpoints. Above the frequency edge,
        only a measured declining terminal log slope is continued. This is
        explicit extrapolation, not a new fitted blue opacity. Below the
        lowest frequency, coefficients are zero. CIA already includes the
        induced absorption/emission convention; do not add a stimulated factor.
        """
        wave = np.atleast_1d(np.asarray(wavelength, float))
        t = np.atleast_1d(np.asarray(temperature, float))
        if (
            wave.ndim != 1
            or t.ndim != 1
            or np.any(~np.isfinite(wave))
            or np.any(wave <= 0)
            or np.any(~np.isfinite(t))
            or np.any(t <= 0)
        ):
            raise ValueError(
                "CIA requires positive one-dimensional wavelength and temperature"
            )
        wn = 1e8 / wave
        i = np.clip(
            np.searchsorted(self.wavenumber, wn, side="right") - 1,
            0,
            self.wavenumber.size - 2,
        )
        tc = np.clip(t, self.temperature[0], self.temperature[-1])
        j = np.clip(
            np.searchsorted(self.temperature, tc, side="right") - 1,
            0,
            self.temperature.size - 2,
        )
        a = ((wn - self.wavenumber[i]) / (self.wavenumber[i + 1] - self.wavenumber[i]))[
            :, None
        ]
        b = (
            (tc - self.temperature[j]) / (self.temperature[j + 1] - self.temperature[j])
        )[None, :]

        def interp(matrix):
            lo = (1 - b) * matrix[i[:, None], j[None, :]] + b * matrix[
                i[:, None], j[None, :] + 1
            ]
            hi = (1 - b) * matrix[i[:, None] + 1, j[None, :]] + b * matrix[
                i[:, None] + 1, j[None, :] + 1
            ]
            return lo + a * (hi - lo), lo, hi

        value, lo, hi = interp(self._log_coefficient)
        # Mask unsupported/rising tails BEFORE exponentiating.
        outside = wn > self.wavenumber[-1]
        valid = (wn[:, None] >= self.wavenumber[0]) & (~outside[:, None] | (hi < lo))
        result = np.exp(np.minimum(np.where(valid, value, -np.inf), 700.0))
        zeros = self.coefficient_cm5[i[:, None], j[None, :]] == 0
        zeros |= self.coefficient_cm5[i[:, None] + 1, j[None, :]] == 0
        zeros |= self.coefficient_cm5[i[:, None], j[None, :] + 1] == 0
        zeros |= self.coefficient_cm5[i[:, None] + 1, j[None, :] + 1] == 0
        linear, _, _ = interp(self.coefficient_cm5)
        result = np.where(zeros & ~outside[:, None], np.maximum(linear, 0), result)
        result[wn < self.wavenumber[0]] = 0
        return result


def read_hitran_h2_he_cia(path):
    """Strict packed HITRAN reader; no amagat-unit conversion is needed."""
    path = Path(path)
    columns = []
    temperatures = []
    grid = None
    digest = hashlib.sha256()
    with path.open("r", encoding="ascii", newline="") as stream:
        for header in stream:
            digest.update(header.encode("ascii"))
            if not header.strip():
                continue
            parts = header.split()
            if parts[0] != "H2-He" or len(parts) < 5:
                raise ValueError("Expected an H2-He HITRAN header")
            count = int(parts[3])
            temperature = float(parts[4])
            lines = list(islice(stream, count))
            digest.update("".join(lines).encode("ascii"))
            block = np.loadtxt(lines)
            if block.shape != (count, 2):
                raise ValueError("Truncated HITRAN CIA block")
            if grid is None:
                grid = block[:, 0].copy()
            if not np.array_equal(grid, block[:, 0]):
                raise ValueError("HITRAN blocks must share one grid")
            if grid[0] != float(parts[1]) or grid[-1] != float(parts[2]):
                raise ValueError("CIA header bounds disagree with its data")
            columns.append(block[:, 1])
            temperatures.append(temperature)
    if grid is None:
        raise ValueError("Empty HITRAN CIA file")
    return H2HeCIATable(
        grid,
        np.asarray(temperatures),
        np.column_stack(columns),
        path,
        digest.hexdigest(),
    )


def h2_he_cia_mass_absorption_coefficient(atmosphere, wavelength, table):
    h = atmosphere.hydrogen_lte_state
    he = atmosphere.helium_lte_state
    if h is None or he is None or h.molecular_hydrogen_density is None:
        raise ValueError("H2-He CIA requires a molecular mixed-composition state")
    return (
        table.evaluate(wavelength, atmosphere.temperature)
        * h.molecular_hydrogen_density[None, :]
        * he.neutral_he_density[None, :]
        / atmosphere.mass_density[None, :]
    )


@dataclass(frozen=True)
class MolecularHHePhysics:
    """Pass the same molecular chemistry and opacity to all model consumers."""

    h2_he_table: H2HeCIATable
    h2_h2_table: object
    include_cia: bool = True
    include_neutral_lyman_wing: bool = True

    def lte(self, *args, **kwargs):
        from ._mixed_molecules import molecular_hydrogen_helium_lte

        return molecular_hydrogen_helium_lte(*args, **kwargs)

    def thermodynamics(self, *args, **kwargs):
        from ._mixed_molecules import molecular_hydrogen_helium_thermodynamics

        return molecular_hydrogen_helium_thermodynamics(*args, **kwargs)

    def hydrogen_opacity(
        self, atmosphere, wavelength, *, unified_allard_table=None, **kwargs
    ):
        from .opacity import (
            hydrogen_continuum_mass_absorption_coefficient,
            lyman_alpha_neutral_hydrogen_wing_mass_absorption_coefficient,
        )

        kwargs.pop("include_molecular_absorption", None)
        result = hydrogen_continuum_mass_absorption_coefficient(
            atmosphere,
            wavelength,
            include_molecular_absorption=True,
            h2_h2_cia_table=self.h2_h2_table if self.include_cia else None,
            **kwargs
        )
        if self.include_cia:
            result += h2_he_cia_mass_absorption_coefficient(
                atmosphere, wavelength, self.h2_he_table
            )
        if self.include_neutral_lyman_wing:
            result += lyman_alpha_neutral_hydrogen_wing_mass_absorption_coefficient(
                atmosphere,
                wavelength,
                allard_table=(
                    None
                    if unified_allard_table is None
                    else unified_allard_table.lines.get((1, 2))
                ),
            )
        return result
