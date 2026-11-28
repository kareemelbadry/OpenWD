"""Public H2db hydrogen transitions for magnetic white-dwarf spectra.

Schimeczek & Wunner's H2db tables provide field-dependent transition energies
and dipole strengths.  This reader identifies the Balmer transitions from
their field-free lower-state and transition energies, preserving the original
symmetry labels encoded in the directory names.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from functools import lru_cache
import math
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import (
    BOLTZMANN,
    HYDROGEN_IONIZATION_ENERGY,
    LIGHT_SPEED,
    PLANCK,
)


FloatArray = NDArray[np.float64]

H2DB_REFERENCE_FIELD_MEGAGAUSS = 4_701.03
H2DB_DATASET_DOI = "10.18419/darus-2118"
# Several public H2db tracks (all 2p m=+1 -> upper m=0) end where the upper
# state crosses the lower one and the transition energy reaches zero.  Beyond
# the last row such a branch absorbs, if at all, beyond 9 microns, from a
# lower state that is already several eV above the tightly bound 2p state.
# Its optical/IR oscillator strength is therefore zero.  Branches ending at a
# larger photon energy still raise rather than extrapolating incomplete data.
H2DB_VANISHING_BRANCH_ENERGY_RYDBERG = 1.0e-2
# H2db energies are infinite-mass Rydbergs.  Energy differences are converted
# with the reduced-mass hydrogen Rydberg used by the ordinary EOS and line
# centers, so the zero-field limit is exact.
_RYDBERG_ENERGY_ERG = HYDROGEN_IONIZATION_ENERGY
# m_e/m_p (CODATA 2018) for the finite-nuclear-mass component positions.
ELECTRON_PROTON_MASS_RATIO = 1.0 / 1_836.152_673_43


def _finite_mass_balmer_terms(
    field_megagauss: float | FloatArray,
) -> tuple[float | FloatArray, float | FloatArray]:
    """Return the H2db lookup field and proton cyclotron energy in reduced Rydbergs."""
    mass_ratio = ELECTRON_PROTON_MASS_RATIO
    beta = field_megagauss / H2DB_REFERENCE_FIELD_MEGAGAUSS
    return (
        field_megagauss * (1.0 + mass_ratio) ** 2,
        4.0 * beta * mass_ratio * (1.0 + mass_ratio),
    )


@dataclass(frozen=True)
class H2dbEnergyTrack:
    """One spin-down, non-positive-``m`` stationary H2db state."""

    source_path: Path
    principal_quantum_number: int
    orbital_quantum_number: int
    absolute_magnetic_quantum_number: int
    z_parity: int
    longitudinal_excitation: int
    beta: FloatArray
    energy_rydberg: FloatArray

    def energy_at_field(self, field_strength_megagauss: float) -> float:
        """Interpolate the stationary-state energy in Rydbergs."""

        field = float(field_strength_megagauss)
        if not np.isfinite(field) or field < 0.0:
            raise ValueError("field_strength_megagauss must be finite and nonnegative")
        beta = field / H2DB_REFERENCE_FIELD_MEGAGAUSS
        if not self.beta[0] <= beta <= self.beta[-1]:
            raise ValueError(
                f"B={field:g} MG lies outside H2db range "
                f"{self.beta[0] * H2DB_REFERENCE_FIELD_MEGAGAUSS:g}--"
                f"{self.beta[-1] * H2DB_REFERENCE_FIELD_MEGAGAUSS:g} MG for "
                f"{self.source_path.name}"
            )
        return float(np.interp(beta, self.beta, self.energy_rydberg))


@dataclass(frozen=True)
class H2dbEnergyDatabase:
    """Stationary H2db energies indexed by their zero-field ``n,l,|m|``."""

    root: Path
    tracks: dict[tuple[int, int, int], H2dbEnergyTrack]
    _shell_excitation_cache: dict[
        tuple[int, float, int], tuple[FloatArray, ...]
    ] = dataclass_field(default_factory=dict, compare=False, repr=False)

    @property
    def maximum_complete_principal_quantum_number(self) -> int:
        """Largest shell for which all ``n**2`` spatial states are present."""

        maximum = 0
        for principal in sorted({key[0] for key in self.tracks}):
            expected = {
                (principal, orbital, absolute_m)
                for orbital in range(principal)
                for absolute_m in range(orbital + 1)
            }
            if expected.issubset(self.tracks):
                maximum = principal
            else:
                break
        return maximum

    def spin_down_energy_rydberg(
        self,
        principal_quantum_number: int,
        orbital_quantum_number: int,
        magnetic_quantum_number: int,
        field_strength_megagauss: float,
    ) -> float:
        """Return the spin-down energy for either sign of ``m``.

        H2db stores only ``m <= 0``.  Equation (21) of Rohrmann (2026)
        gives the positive-``m`` partner by adding ``4 m beta``.
        """

        principal = int(principal_quantum_number)
        orbital = int(orbital_quantum_number)
        magnetic = int(magnetic_quantum_number)
        if principal < 1 or not 0 <= orbital < principal or abs(magnetic) > orbital:
            raise ValueError("invalid hydrogen quantum numbers")
        if float(field_strength_megagauss) == 0.0:
            return -1.0 / principal**2
        track = self.tracks.get((principal, orbital, abs(magnetic)))
        if track is None:
            raise KeyError(
                f"H2db has no n={principal}, l={orbital}, |m|={abs(magnetic)} track"
            )
        energy = track.energy_at_field(field_strength_megagauss)
        if magnetic > 0:
            beta = float(field_strength_megagauss) / H2DB_REFERENCE_FIELD_MEGAGAUSS
            energy += 4.0 * magnetic * beta
        return energy

    def energy_rydberg(
        self,
        principal_quantum_number: int,
        orbital_quantum_number: int,
        magnetic_quantum_number: int,
        spin_projection: float,
        field_strength_megagauss: float,
    ) -> float:
        """Return a stationary-state energy for ``m_s = +/- 1/2``."""

        spin = float(spin_projection)
        if spin not in {-0.5, 0.5}:
            raise ValueError("spin_projection must be -0.5 or +0.5")
        energy = self.spin_down_energy_rydberg(
            principal_quantum_number,
            orbital_quantum_number,
            magnetic_quantum_number,
            field_strength_megagauss,
        )
        if spin > 0.0:
            beta = float(field_strength_megagauss) / H2DB_REFERENCE_FIELD_MEGAGAUSS
            energy += 4.0 * beta
        return energy

    def ground_state_binding_energy_erg(
        self, field_strength_megagauss: float
    ) -> float:
        """Return the stationary ground-state binding energy."""

        field = float(field_strength_megagauss)
        if not np.isfinite(field) or field < 0.0:
            raise ValueError("field_strength_megagauss must be finite and nonnegative")
        if field == 0.0:
            return float(HYDROGEN_IONIZATION_ENERGY)
        energy = self.energy_rydberg(1, 0, 0, -0.5, field)
        if energy >= 0.0:
            raise ValueError("H2db ground state is not bound at the requested field")
        return -energy * _RYDBERG_ENERGY_ERG

    def shell_boltzmann_weight(
        self,
        maximum_level: int,
        field_strength_megagauss: float,
        temperature: FloatArray,
        *,
        maximum_magnetic_level: int = 17,
    ) -> FloatArray:
        """Return complete centered-state Boltzmann sums for H shells.

        Every spatial ``(n,l,m)`` state is included through ``n=17``: exact
        H2db tracks are used where available, with the Vera-Rueda--Rohrmann
        centered-state fits completing the missing high-``|m|`` and high-
        ``nu`` states.  Above that limit a complete ``n**2`` hydrogenic shell is
        retained relative to the field-dependent ground state.  This closure
        is preferable to treating the incomplete Balmer transition list as a
        level inventory, which silently omits an increasing fraction of each
        high shell.

        Electron spin is excluded from both the bound-state and continuum
        statistical weights, matching the convention used by the ordinary
        hydrogen EOS.  Finite-pseudomomentum and decentered states remain
        outside this stationary-atom partition function.
        """

        if maximum_level < 1:
            raise ValueError("maximum_level must be positive")
        if maximum_magnetic_level < 1 or maximum_magnetic_level > 17:
            raise ValueError("maximum_magnetic_level must lie between 1 and 17")
        temperature = np.asarray(temperature, dtype=np.float64)
        if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
            raise ValueError("temperature must be finite and positive")
        field = float(field_strength_megagauss)
        if not np.isfinite(field) or field < 0.0:
            raise ValueError("field_strength_megagauss must be finite and nonnegative")
        if field == 0.0:
            level = np.arange(1, maximum_level + 1, dtype=np.float64)
            excitation = HYDROGEN_IONIZATION_ENERGY * (1.0 - 1.0 / level**2)
            return level.reshape((1,) * temperature.ndim + (-1,)) ** 2 * np.exp(
                -excitation.reshape((1,) * temperature.ndim + (-1,))
                / (BOLTZMANN * temperature[..., np.newaxis])
            )

        cache_key = (int(maximum_level), field, int(maximum_magnetic_level))
        excitation_by_shell = self._shell_excitation_cache.get(cache_key)
        if excitation_by_shell is None:
            ground = self.energy_rydberg(1, 0, 0, -0.5, field)
            shell_values: list[FloatArray] = []
            for principal in range(1, maximum_level + 1):
                if principal <= maximum_magnetic_level:
                    energies = [
                        self.energy_rydberg_with_analytic_fallback(
                            principal, orbital, magnetic, -0.5, field
                        )
                        for orbital in range(principal)
                        for magnetic in range(-orbital, orbital + 1)
                    ]
                    excitation = np.asarray(energies, dtype=np.float64) - ground
                else:
                    hydrogenic = -1.0 / principal**2
                    excitation = np.full(
                        principal**2, hydrogenic - ground, dtype=np.float64
                    )
                if np.any(excitation < -1.0e-10):
                    raise ValueError("magnetic shell contains a state below the ground state")
                shell_values.append(np.maximum(excitation, 0.0))
            excitation_by_shell = tuple(shell_values)
            self._shell_excitation_cache[cache_key] = excitation_by_shell

        output = np.empty(temperature.shape + (maximum_level,), dtype=np.float64)
        inverse_k_temperature = 1.0 / (
            BOLTZMANN * temperature[..., np.newaxis]
        )
        for index, excitation_rydberg in enumerate(excitation_by_shell):
            output[..., index] = np.sum(
                np.exp(
                    -excitation_rydberg.reshape(
                        (1,) * temperature.ndim + (-1,)
                    )
                    * _RYDBERG_ENERGY_ERG
                    * inverse_k_temperature
                ),
                axis=-1,
            )
        return output

    def energy_rydberg_with_analytic_fallback(
        self,
        principal_quantum_number: int,
        orbital_quantum_number: int,
        magnetic_quantum_number: int,
        spin_projection: float,
        field_strength_megagauss: float,
    ) -> float:
        """Return an H2db energy, completing missing high-``|m|`` states.

        The public H2db archive contains every spatial state through ``n=5``
        but becomes progressively incomplete at higher principal quantum
        number.  Vera-Rueda & Rohrmann (2020), Appendix A, give centered-state
        fits for arbitrary ``|m|`` and longitudinal quantum number ``nu``;
        those fits complete the missing states.  Exact H2db tracks always
        take precedence.
        """

        principal = int(principal_quantum_number)
        orbital = int(orbital_quantum_number)
        magnetic = int(magnetic_quantum_number)
        spin = float(spin_projection)
        if principal < 1 or not 0 <= orbital < principal or abs(magnetic) > orbital:
            raise ValueError("invalid hydrogen quantum numbers")
        if spin not in {-0.5, 0.5}:
            raise ValueError("spin_projection must be -0.5 or +0.5")
        field = float(field_strength_megagauss)
        if not np.isfinite(field) or field < 0.0:
            raise ValueError("field_strength_megagauss must be finite and nonnegative")
        if field == 0.0:
            return -1.0 / principal**2
        track = self.tracks.get((principal, orbital, abs(magnetic)))
        if track is not None:
            return self.energy_rydberg(
                principal, orbital, magnetic, spin, field
            )
        longitudinal = _longitudinal_quantum_number(
            principal, orbital, abs(magnetic)
        )
        base = _vera_rueda_rohrmann_centered_energy_rydberg(
            abs(magnetic), longitudinal, field
        )
        beta = field / H2DB_REFERENCE_FIELD_MEGAGAUSS
        if magnetic > 0:
            base += 4.0 * magnetic * beta
        if spin > 0.0:
            base += 4.0 * beta
        return base

    def photoionization_threshold_with_analytic_fallback_rydberg(
        self,
        principal_quantum_number: int,
        orbital_quantum_number: int,
        magnetic_quantum_number: int,
        field_strength_megagauss: float,
        polarization: int,
    ) -> float:
        """Return the RWA threshold, completing missing high-``|m|`` states."""

        if polarization not in {-1, 0, 1}:
            raise ValueError("polarization must be -1, 0, or +1")
        magnetic = int(magnetic_quantum_number)
        base = self.energy_rydberg_with_analytic_fallback(
            principal_quantum_number,
            orbital_quantum_number,
            -abs(magnetic),
            -0.5,
            field_strength_megagauss,
        )
        beta = float(field_strength_megagauss) / H2DB_REFERENCE_FIELD_MEGAGAUSS
        threshold = -base
        if polarization == 1 and magnetic >= 0:
            threshold += 4.0 * beta
        elif polarization == -1 and magnetic >= 1:
            threshold -= 4.0 * beta
        return max(0.0, threshold)


def _zero_field_labels(
    absolute_magnetic_quantum_number: int,
    z_parity: int,
    excitation_index: int,
) -> tuple[int, int, int]:
    """Map H2db path labels to ``n,l,nu`` using Vera Rueda & Rohrmann."""

    absolute_m = int(absolute_magnetic_quantum_number)
    index = int(excitation_index)
    if absolute_m < 0 or z_parity not in {-1, 1} or index < 1:
        raise ValueError("invalid H2db state labels")
    longitudinal = 2 * (index - 1) if z_parity == 1 else 2 * index - 1
    if longitudinal % 2 == 0:
        principal = math.floor(
            1.0
            + 2.0 * longitudinal / (1.0 + math.sqrt(2.0 * longitudinal + 1.0))
            + 1.0e-12
        ) + absolute_m
        if (principal - absolute_m) % 2:
            orbital = (
                0.5 * ((principal - absolute_m + 1) ** 2 - 4)
                - longitudinal
                + absolute_m
            )
        else:
            orbital = (
                0.5 * ((principal - absolute_m + 1) ** 2 - 5)
                - longitudinal
                + absolute_m
            )
    else:
        principal = math.floor(
            2.0
            + (2.0 * longitudinal - 2.0)
            / (1.0 + math.sqrt(2.0 * longitudinal - 1.0))
            + 1.0e-12
        ) + absolute_m
        if (principal - absolute_m) % 2:
            orbital = (
                0.5 * ((principal - absolute_m) ** 2 - 1)
                - longitudinal
                + absolute_m
            )
        else:
            orbital = (
                0.5 * (principal - absolute_m) ** 2
                - longitudinal
                + absolute_m
            )
    rounded_orbital = int(round(orbital))
    if (
        abs(orbital - rounded_orbital) > 1.0e-8
        or not absolute_m <= rounded_orbital < principal
        or (-1) ** (rounded_orbital - absolute_m) != z_parity
    ):
        raise ValueError("inconsistent H2db zero-field state correspondence")
    return int(principal), rounded_orbital, longitudinal


@lru_cache(maxsize=None)
def _longitudinal_quantum_number(
    principal_quantum_number: int,
    orbital_quantum_number: int,
    absolute_magnetic_quantum_number: int,
) -> int:
    """Invert the exact zero-/high-field state correspondence."""

    principal = int(principal_quantum_number)
    orbital = int(orbital_quantum_number)
    absolute_m = int(absolute_magnetic_quantum_number)
    if principal < 1 or not 0 <= absolute_m <= orbital < principal:
        raise ValueError("invalid hydrogen quantum numbers")
    parity = (-1) ** (orbital - absolute_m)
    for excitation_index in range(1, principal**2 + 2):
        candidate = _zero_field_labels(absolute_m, parity, excitation_index)
        if candidate[:2] == (principal, orbital):
            return candidate[2]
        if candidate[0] > principal:
            break
    raise ValueError("could not invert the magnetic hydrogen state correspondence")


_CENTERED_ENERGY_PARAMETERS = {
    0: {
        "x_a": (-0.851584, -2.90213, 1.01555),
        "x_b": (0.786224, -2.28335, 0.937692),
        "epsilon_a": (0.0950091, -1.97412, 1.00523),
        "epsilon_b": (0.573409, -1.54066, 0.977581),
        "epsilon_c": (1.26974, -0.378015, 0.910852),
        "epsilon_prime_a": (0.170505, 0.0516550, 0.692991),
    },
    1: {
        "x_b": (-0.666302, -1.50237, 1.17845),
        "epsilon_b": (-0.361037, -0.980935, 1.22078),
        "epsilon_prime_b": (0.213743, 0.223000, 0.882388),
    },
    2: {
        "x_b": (0.0528777, -2.38204, 0.960364),
        "epsilon_b": (-0.452254, -1.00281, 1.23880),
        "epsilon_prime_b": (0.119340, 0.296234, 1.03199),
    },
    3: {
        "x_b": (-0.710984, -1.78597, 1.16795),
        "epsilon_b": (-0.790709, -0.784790, 1.36181),
        "epsilon_prime_b": (0.117903, 0.260062, 1.11196),
    },
}


def _magnetic_fit_parameter(
    coefficients: tuple[float, float, float], absolute_m: int
) -> float:
    b0, b1, b2 = coefficients
    return float(b0 + b1 * np.log10(1.0 + absolute_m) ** b2)


def _vera_rueda_rohrmann_centered_energy_rydberg(
    absolute_magnetic_quantum_number: int,
    longitudinal_quantum_number: int,
    field_strength_megagauss: float,
) -> float:
    """Evaluate the 2020 centered-state energy fits for arbitrary ``nu``.

    This is the infinite-nuclear-mass stationary energy used to extend the
    Schimeczek--Wunner tracks.  The fit is exact in the zero-field limit and
    was calibrated over ``-4 <= log10(beta) <= 3``.  The derivative-matching
    denominator in the published ``nu=1,2`` expression is written here as
    ``epsilon_n-epsilon_nu``; this follows directly by differentiating their
    Eq. A.7 and reproduces the numerical H2db tracks, whereas the printed
    ``epsilon_n-1`` does not.
    """

    absolute_m = int(absolute_magnetic_quantum_number)
    longitudinal = int(longitudinal_quantum_number)
    field = float(field_strength_megagauss)
    if absolute_m < 0 or longitudinal < 0:
        raise ValueError("magnetic quantum numbers must be nonnegative")
    if not np.isfinite(field) or field < 0.0:
        raise ValueError("field_strength_megagauss must be finite and nonnegative")
    parity = 1 if longitudinal % 2 == 0 else -1
    excitation_index = (
        longitudinal // 2 + 1
        if parity == 1
        else (longitudinal + 1) // 2
    )
    principal, _orbital, recovered_longitudinal = _zero_field_labels(
        absolute_m, parity, excitation_index
    )
    if recovered_longitudinal != longitudinal:
        raise ValueError("inconsistent centered-state quantum-number mapping")
    if field == 0.0:
        return -1.0 / principal**2
    beta = field / H2DB_REFERENCE_FIELD_MEGAGAUSS
    x = float(np.log10(beta))
    epsilon_n = -2.0 * np.log10(principal)
    parameters = _CENTERED_ENERGY_PARAMETERS.get(longitudinal)

    if longitudinal == 0:
        x_a = _magnetic_fit_parameter(parameters["x_a"], absolute_m)
        x_b = _magnetic_fit_parameter(parameters["x_b"], absolute_m)
        epsilon_a = _magnetic_fit_parameter(
            parameters["epsilon_a"], absolute_m
        )
        epsilon_b = _magnetic_fit_parameter(
            parameters["epsilon_b"], absolute_m
        )
        epsilon_c = _magnetic_fit_parameter(
            parameters["epsilon_c"], absolute_m
        )
        epsilon_prime_a = _magnetic_fit_parameter(
            parameters["epsilon_prime_a"], absolute_m
        )
        a1 = (epsilon_n - epsilon_a) / (epsilon_a - 1.0)
        a2 = (
            -epsilon_prime_a
            * (1.0 + a1) ** 2
            / ((epsilon_n - 1.0) * a1)
        )
        if x < x_a:
            epsilon = 1.0 + (epsilon_n - 1.0) / (
                1.0
                + a1
                * np.exp(a2 * (x - x_a - 0.1 * (x - x_a) ** 2))
            )
        elif x < x_b:
            epsilon = epsilon_a + (epsilon_b - epsilon_a) * (
                (x - x_a) / (x_b - x_a)
            ) ** 1.22
        else:
            epsilon = epsilon_b + (epsilon_c - epsilon_b) * (
                (x - x_b) / (3.0 - x_b)
            ) ** 0.92
        return -10.0**epsilon

    epsilon_nu = -2.0 * np.log10(max(1, (longitudinal + 1) // 2))
    if longitudinal >= 4:
        return _vera_rueda_rohrmann_high_nu_energy_rydberg(
            absolute_m,
            longitudinal,
            x,
            epsilon_n,
            epsilon_nu,
        )
    assert parameters is not None
    x_b = _magnetic_fit_parameter(parameters["x_b"], absolute_m)
    epsilon_b = _magnetic_fit_parameter(parameters["epsilon_b"], absolute_m)
    epsilon_prime_b = _magnetic_fit_parameter(
        parameters["epsilon_prime_b"], absolute_m
    )
    a1 = (epsilon_n - epsilon_b) / (epsilon_b - epsilon_nu)
    a2 = (
        -epsilon_prime_b
        * (1.0 + a1) ** 2
        / ((epsilon_n - epsilon_nu) * a1)
    )
    if longitudinal == 1:
        delta = {1: 0.20, 2: 0.22, 3: 0.24}.get(absolute_m, 0.26)
        if x > x_b:
            delta = 0.0
    elif longitudinal == 2:
        delta = 0.20
    else:
        delta = {0: 0.0, 1: 0.08, 2: 0.13, 3: 0.15, 4: 0.165, 5: 0.17}.get(
            absolute_m, 0.17
        )
    low_field_epsilon = epsilon_nu + (epsilon_n - epsilon_nu) / (
        1.0
        + a1
        * np.exp(a2 * (x - x_b) * (1.0 - delta * (x - x_b)))
    )
    if longitudinal == 1 or x <= x_b:
        return -10.0**low_field_epsilon

    if longitudinal == 3:
        q = 2.5
        epsilon = epsilon_b + (
            (epsilon_nu - epsilon_b)
            * (x - x_b)
            / (
                ((epsilon_nu - epsilon_b) / epsilon_prime_b) ** q
                + (x - x_b) ** q
            )
            ** (1.0 / q)
        )
        return -10.0**epsilon

    m_star = min(absolute_m, 4)
    xi = -0.0125 + 0.030456 * np.log10(1.0 + m_star) ** 1.134
    epsilon = epsilon_b + (2.0 / np.pi) * epsilon_b * np.arctan(
        np.pi
        * epsilon_prime_b
        * (x - x_b)
        / (2.0 * epsilon_b)
        * (1.0 + xi * (x - x_b))
    )
    return -10.0**epsilon


def _high_nu_fit_coefficients(
    longitudinal_quantum_number: int,
) -> dict[str, tuple[float, float, float]]:
    """Return Appendix-A coefficients for centered states with ``nu>=4``."""

    nu = int(longitudinal_quantum_number)
    if nu < 4:
        raise ValueError("high-nu coefficients require nu>=4")
    even = nu % 2 == 0
    delta = np.log10(nu) - np.log10(4 if even else 5)
    integer_part = np.floor(0.5 * (nu + 0.5))
    t_value = 2.0 * np.sqrt(integer_part) + (2.0 if even else -2.0)
    tau = t_value - np.floor(t_value)
    if even:
        b0 = {
            "x_a": -1.1 * tau - 1.154902 - 2.087178 * delta**1.082710,
            "x_b": -0.522879,
            "epsilon_a": -0.68 - 1.176143 * delta**0.8685913,
            "epsilon_b": -0.8867395 - 1.744739 * delta**1.095173,
            "y_a": 0.02 - 0.034 * tau + 0.2 / (nu**1.1 + 3.0),
            "y_b": 0.3780437 * nu**-0.9572978,
        }
        b1 = {
            "x_a": -0.01890508,
            "x_b": -0.95 - 1.1 * nu**-0.4,
            "epsilon_a": -0.2 - 1.1 * nu**-0.4,
            "epsilon_b": -2.487767 * nu**-0.9652760,
            "y_a": 0.1438085,
            "y_b": 0.8265754 * nu**-0.9347425,
        }
        b2 = {
            "x_a": 0.5904491,
            "x_b": 0.85 + 1.1 * nu**-0.4,
            "epsilon_a": 0.6 + 0.8 * delta**0.4,
            "epsilon_b": 1.209001,
            "y_a": 1.596943,
            "y_b": 1.114659,
        }
    else:
        b0 = {
            "x_a": -tau - 1.18 - 2.312886 * delta**0.7737455,
            "x_b": -1.154902,
            "epsilon_a": -0.9558838 - 1.069160 * delta**0.8065575,
            "epsilon_b": -1.12 - 1.707775 * delta**1.119483,
            "y_a": 0.013 - 0.034 * tau + 0.2 / (nu**0.74 + 6.0),
            "y_b": 0.3480917 * nu**-0.9508739,
        }
        b1 = {
            "x_a": 0.1044253,
            "x_b": -0.1 - 2.0 * nu**-0.4,
            "epsilon_a": -0.2 - 0.9 * nu**-0.4,
            "epsilon_b": -2.713701 * nu**-1.000845,
            "y_a": 0.1457385,
            "y_b": 1.0286911 * nu**-0.9818393,
        }
        b2 = {
            "x_a": 0.7094884,
            "x_b": 0.90 + 1.5 * nu**-0.4,
            "epsilon_a": 0.7 + 0.73 * delta**0.4,
            "epsilon_b": 1.251972,
            "y_a": 1.603306,
            "y_b": 1.181565,
        }
    return {name: (b0[name], b1[name], b2[name]) for name in b0}


def _vera_rueda_rohrmann_high_nu_energy_rydberg(
    absolute_magnetic_quantum_number: int,
    longitudinal_quantum_number: int,
    log_beta: float,
    epsilon_n: float,
    epsilon_nu: float,
) -> float:
    """Evaluate the Vera-Rueda--Rohrmann ``nu>=4`` centered-state fit.

    The published end branches and their fitted join slopes are used
    directly.  A cubic Hermite segment connects the two joins.  This is
    algebraically stable and enforces the four published endpoint
    constraints; it avoids a typographical ambiguity in the printed middle
    branch of their Eq. A.12.
    """

    absolute_m = int(absolute_magnetic_quantum_number)
    nu = int(longitudinal_quantum_number)
    parameters = _high_nu_fit_coefficients(nu)
    values = {
        name: _magnetic_fit_parameter(coefficients, absolute_m)
        for name, coefficients in parameters.items()
    }
    x_a = values["x_a"]
    x_b = values["x_b"]
    epsilon_a = values["epsilon_a"]
    epsilon_b = values["epsilon_b"]
    y_a = max(values["y_a"], np.finfo(float).eps)
    y_b = max(values["y_b"], np.finfo(float).eps)
    if x_a > x_b:
        midpoint_x = 0.5 * (x_a + x_b)
        midpoint_epsilon = 0.5 * (epsilon_a + epsilon_b)
        x_a = x_b = midpoint_x
        epsilon_a = epsilon_b = midpoint_epsilon

    x = float(log_beta)
    if x <= x_a:
        c_a = (epsilon_a - epsilon_n) / y_a
        epsilon = epsilon_a + (
            (epsilon_a - epsilon_n)
            * (x - x_a)
            / np.sqrt(c_a**2 + (x - x_a) ** 2)
        )
    elif x >= x_b:
        c_b = (epsilon_nu - epsilon_b) / y_b
        epsilon = epsilon_b + (
            (epsilon_nu - epsilon_b)
            * (x - x_b)
            / np.sqrt(c_b**2 + (x - x_b) ** 2)
        )
    elif x_b == x_a:
        epsilon = 0.5 * (epsilon_a + epsilon_b)
    else:
        fraction = (x - x_a) / (x_b - x_a)
        spacing = x_b - x_a
        h00 = 2.0 * fraction**3 - 3.0 * fraction**2 + 1.0
        h10 = fraction**3 - 2.0 * fraction**2 + fraction
        h01 = -2.0 * fraction**3 + 3.0 * fraction**2
        h11 = fraction**3 - fraction**2
        epsilon = (
            h00 * epsilon_a
            + h10 * spacing * y_a
            + h01 * epsilon_b
            + h11 * spacing * y_b
        )
    return -10.0**epsilon


@dataclass(frozen=True)
class H2dbTransition:
    """One field-dependent H2db transition and its symmetry labels."""

    source_path: Path
    lower_magnetic_quantum_number: int
    upper_magnetic_quantum_number: int
    lower_z_parity: int
    upper_z_parity: int
    lower_excitation_index: int
    upper_excitation_index: int
    balmer_upper_level: int
    beta: FloatArray
    transition_energy_rydberg: FloatArray
    dipole_strength: FloatArray
    initial_state_energy_rydberg: FloatArray

    @property
    def delta_m(self) -> int:
        return (
            self.upper_magnetic_quantum_number
            - self.lower_magnetic_quantum_number
        )

    def values_at_field(
        self, field_strength_megagauss: float
    ) -> tuple[float, float, float]:
        """Interpolate energy, dipole strength, and lower energy linearly."""

        beta = float(field_strength_megagauss) / H2DB_REFERENCE_FIELD_MEGAGAUSS
        if beta < self.beta[0]:
            raise ValueError(
                f"B={field_strength_megagauss:g} MG lies outside H2db range "
                f"{self.beta[0] * H2DB_REFERENCE_FIELD_MEGAGAUSS:g}--"
                f"{self.beta[-1] * H2DB_REFERENCE_FIELD_MEGAGAUSS:g} MG for "
                f"{self.source_path.name}"
            )
        if beta > self.beta[-1]:
            # The public archive terminates two Halpha state tracks as their
            # transition energies approach zero.  At that point their line
            # centers have left the UV/optical/IR by orders of magnitude.  A
            # zero oscillator strength is the conservative optical closure;
            # clamping or linear extrapolation would retain a spurious remote
            # line or drive its photon energy negative.  All normally ending
            # branches continue to raise, so this cannot silently broaden the
            # usable field range of incomplete atomic data.
            final_energy = float(self.transition_energy_rydberg[-1])
            if 0.0 < final_energy <= H2DB_VANISHING_BRANCH_ENERGY_RYDBERG:
                return (
                    final_energy,
                    0.0,
                    float(self.initial_state_energy_rydberg[-1]),
                )
            raise ValueError(
                f"B={field_strength_megagauss:g} MG lies outside H2db range "
                f"{self.beta[0] * H2DB_REFERENCE_FIELD_MEGAGAUSS:g}--"
                f"{self.beta[-1] * H2DB_REFERENCE_FIELD_MEGAGAUSS:g} MG for "
                f"{self.source_path.name}"
            )
        energy = float(np.interp(beta, self.beta, self.transition_energy_rydberg))
        initial = float(
            np.interp(beta, self.beta, self.initial_state_energy_rydberg)
        )
        finite_strength = np.isfinite(self.dipole_strength)
        if np.count_nonzero(finite_strength) < 2:
            strength = 0.0
        else:
            strength = float(
                np.interp(
                    beta,
                    self.beta[finite_strength],
                    self.dipole_strength[finite_strength],
                )
            )
        return energy, max(0.0, strength), initial


@dataclass(frozen=True)
class H2dbBalmerComponents:
    """Magnetic components and normalized depth-dependent line strengths."""

    wavelength_angstrom: FloatArray
    normalized_strength: FloatArray
    line_strength_scale: FloatArray
    delta_m: NDArray[np.int64]
    # H2db (lower m, z-parity, nu, upper m, z-parity, nu), physical sign of m.
    labels: NDArray[np.int64] | None = None


@dataclass(frozen=True)
class H2dbIntrashellComponents:
    """n=2 -> 2 components: absolute oscillator strengths and lower-state shares."""

    wavelength_angstrom: FloatArray
    oscillator_strength: FloatArray
    lower_state_fraction: FloatArray
    delta_m: NDArray[np.int64]


@dataclass(frozen=True)
class H2dbTransitionDatabase:
    """Balmer subset of an extracted H2db transition archive."""

    root: Path
    transitions_by_upper_level: dict[int, tuple[H2dbTransition, ...]]
    # 2s0 -> 2p+1, 2p-1 -> 2s0 and 2p0 -> 2s0 (empty for an older subset).
    intrashell_transitions: tuple[H2dbTransition, ...] = ()

    def _lower_state_energies(self, field_strength_megagauss: float) -> dict[tuple[int, int, int], float]:
        """Energy (Ry) of every n=2 substate that starts a Balmer branch."""

        energies: dict[tuple[int, int, int], float] = {}
        for available_transitions in self.transitions_by_upper_level.values():
            for transition in available_transitions:
                key = (
                    transition.lower_magnetic_quantum_number,
                    transition.lower_z_parity,
                    transition.lower_excitation_index,
                )
                if key not in energies:
                    energies[key] = transition.values_at_field(field_strength_megagauss)[2]
        return energies

    def intrashell_components(
        self,
        field_strength_megagauss: float,
        temperature: FloatArray,
        *,
        finite_nuclear_mass: bool = False,
    ) -> H2dbIntrashellComponents:
        """Evaluate the n=2 -> 2 components, which exist only in a field.

        They have no field-free line to redistribute, so they carry absolute
        oscillator strengths: per (isotropically averaged) transition between
        two substates f = E d / 3 with E in Ry and the H2db dipole strength d
        in a0^2 (the H2db Halpha sums reproduce f(2s-3p) = 0.4349 and
        f(2p-3s) + f(2p-3d) = 0.7094). ``lower_state_fraction`` is the LTE
        share of the n=2 shell in the lower substate, from the same four H2db
        substate energies as the Balmer components. Positions use the
        package's hydrogen Rydberg as the Balmer default does, or the
        finite-mass form of :meth:`balmer_components`. Below the table floor
        the components lie beyond 20 microns and are omitted (empty result).
        """

        temperature = np.asarray(temperature, dtype=np.float64)
        if temperature.ndim != 1 or np.any(temperature <= 0.0):
            raise ValueError("temperature must be a positive one-dimensional array")
        transitions = self.intrashell_transitions
        floor = max(
            t.beta[0]
            for group in (*self.transitions_by_upper_level.values(), transitions)
            for t in group
        ) * H2DB_REFERENCE_FIELD_MEGAGAUSS
        if not transitions or field_strength_megagauss < floor:
            empty = np.zeros(0)
            return H2dbIntrashellComponents(
                empty, empty, np.zeros((0, temperature.size)), np.zeros(0, dtype=np.int64)
            )
        values = [t.values_at_field(field_strength_megagauss) for t in transitions]
        energy = np.asarray([item[0] for item in values])
        dipole = np.asarray([item[1] for item in values])
        delta_m = np.asarray([t.delta_m for t in transitions], dtype=np.int64)
        line_energy = energy
        if finite_nuclear_mass:
            # As in balmer_components (see _finite_mass_balmer_terms).
            table_field, proton_cyclotron = _finite_mass_balmer_terms(field_strength_megagauss)
            line_energy = np.asarray([t.values_at_field(table_field)[0] for t in transitions]) \
                - proton_cyclotron * delta_m
        wavelength = 1.0e8 * PLANCK * LIGHT_SPEED / (line_energy * _RYDBERG_ENERGY_ERG)

        lower = self._lower_state_energies(field_strength_megagauss)
        lower_energy = np.asarray(list(lower.values()))
        weight = np.exp(
            -(lower_energy - np.min(lower_energy))[:, np.newaxis]
            * _RYDBERG_ENERGY_ERG / (BOLTZMANN * temperature[np.newaxis, :])
        )
        share = dict(zip(lower, weight / np.sum(weight, axis=0)))
        fraction = np.asarray([
            share[(t.lower_magnetic_quantum_number, t.lower_z_parity, t.lower_excitation_index)]
            for t in transitions
        ])
        return H2dbIntrashellComponents(
            wavelength, np.maximum(0.0, energy * dipole) / 3.0, fraction, delta_m
        )

    def balmer_components(
        self,
        upper_level: int,
        rest_wavelength_angstrom: float,
        field_strength_megagauss: float,
        temperature: FloatArray,
        *,
        field_angle_deg: float | None = None,
        minimum_relative_strength: float = 1.0e-5,
        finite_nuclear_mass: bool = False,
    ) -> H2dbBalmerComponents:
        """Evaluate one Balmer line's H2db components at a local field.

        H2db tabulates dipole strengths rather than the exact aggregate
        field-free oscillator-strength convention used by the atmosphere
        opacity.  We therefore redistribute the validated zero-field line
        opacity among H2db components. Relative strengths sum to unity, but
        ``line_strength_scale`` retains the field-dependent total oscillator
        strength per n=2 atom. Relative substate populations use H2db energies.
        Below the complete table's lower field bound, the analytic normal
        triplet supplies the weak-field continuation without extrapolating
        quantized energy tracks. The disk can still use strong-field EOS and
        continuum physics consistently in these weak surface cells.

        ``finite_nuclear_mass=True`` places the components of an atom at rest
        with a finite proton mass: Delta E = mu Delta E_inf(beta/mu^2)
        - Delta m hbar omega_cp, mu = 1/(1 + m_e/m_p), hbar omega_cp =
        (m_e/m_p) hbar omega_c (Wunner et al. 1985, A&A 149, 102, Eq. 1;
        Pavlov & Meszaros 1993, ApJ 416, 752, Eq. 2.12a). H2db is the
        infinite-mass problem; the default scales it to the reduced-mass line
        centre only. The -Delta m hbar omega_cp term moves sigma- blueward and
        sigma+ redward (-4.6 A for the Halpha 2s0-3p-1 stationary component at
        196 MG); the beta/mu^2 term adds 2 (m_e/m_p) B dlambda/dB, zero at
        stationary points. In the linear Zeeman regime the two cancel at first
        order in m_e/m_p. Strengths and populations keep the tabulated field.
        """

        transitions = self.transitions_by_upper_level.get(int(upper_level), ())
        if not transitions:
            raise KeyError(f"H2db contains no Balmer n=2->{upper_level} transitions")
        if field_strength_megagauss <= 0.0:
            raise ValueError("H2db components require a positive field")
        temperature = np.asarray(temperature, dtype=np.float64)
        if temperature.ndim != 1 or np.any(temperature <= 0.0):
            raise ValueError("temperature must be a positive one-dimensional array")
        # The lower-state partition also reads tracks from other Balmer
        # members, so use the complete database's floor, not this line's.
        minimum_field = max(
            t.beta[0] for group in self.transitions_by_upper_level.values() for t in group
        ) * H2DB_REFERENCE_FIELD_MEGAGAUSS
        # Finite nuclear mass (see docstring): read the infinite-mass tracks
        # at beta/mu^2 and subtract Delta m hbar omega_cp; both terms are
        # divided by mu because the rest wavelength is the reduced-mass one.
        table_field, proton_cyclotron = (
            _finite_mass_balmer_terms(field_strength_megagauss)
            if finite_nuclear_mass else (field_strength_megagauss, 0.0)
        )
        if field_strength_megagauss < minimum_field:
            # Delta E/Ry = 2 beta delta_m, with exact field-free Balmer
            # energy; use the caller's reduced-mass wavelength convention.
            delta_m = np.array([-1, 0, 1], dtype=np.int64)
            table_beta = table_field / H2DB_REFERENCE_FIELD_MEGAGAUSS
            rest_energy = 0.25 - 1.0 / int(upper_level)**2
            wavelength = rest_wavelength_angstrom * rest_energy / (
                rest_energy + 2.0 * table_beta * delta_m - proton_cyclotron * delta_m
            )
            strengths = np.full((3, temperature.size), 1.0 / 3.0)
            if field_angle_deg is not None:
                angle = float(field_angle_deg)
                if not np.isfinite(angle) or not 0.0 <= angle <= 180.0:
                    raise ValueError("field_angle_deg must lie between 0 and 180 degrees")
                cosine_squared = np.cos(np.deg2rad(angle))**2
                strengths *= np.where(delta_m[:, None] == 0,
                                      1.5 * (1.0 - cosine_squared),
                                      0.75 * (1.0 + cosine_squared))
            return H2dbBalmerComponents(wavelength, strengths, np.ones_like(temperature), delta_m)
        values = [
            transition.values_at_field(field_strength_megagauss)
            for transition in transitions
        ]
        energy = np.asarray([item[0] for item in values])
        dipole = np.asarray([item[1] for item in values])
        initial = np.asarray([item[2] for item in values])
        # Exact infinite-mass field-free Balmer energy of every branch.  The
        # first tabulated row lies at beta up to 1e-5 and already carries a
        # linear Zeeman shift of up to 2e-5 Ry (about 1 A at Halpha).
        zero_energy = np.full(len(transitions), 0.25 - 1.0 / int(upper_level) ** 2)
        delta_m = np.asarray([transition.delta_m for transition in transitions])
        labels = np.asarray([
            (t.lower_magnetic_quantum_number, t.lower_z_parity, t.lower_excitation_index,
             t.upper_magnetic_quantum_number, t.upper_z_parity, t.upper_excitation_index)
            for t in transitions
        ], dtype=np.int64)

        # Scaling each branch to the package's (reduced-mass) vacuum line
        # center removes the infinite-nuclear-mass offset of H2db while
        # retaining all nonlinear magnetic displacements.
        line_energy = energy
        if finite_nuclear_mass:
            line_energy = np.asarray([
                transition.values_at_field(table_field)[0] for transition in transitions
            ]) - proton_cyclotron * delta_m
        wavelength = float(rest_wavelength_angstrom) * zero_energy / line_energy
        oscillator_proxy = np.maximum(0.0, energy * dipole)
        # The field-free reference: exact energies, degenerate n=2 substates,
        # and the dipole strength of the lowest tabulated field (the dipole
        # strength is quadratic in beta near zero field).
        reference_energy = zero_energy
        reference_dipole = np.asarray(
            [transition.dipole_strength[0] for transition in transitions]
        )
        reference_initial = np.full(len(transitions), -0.25)
        reference_oscillator_proxy = np.maximum(
            0.0,
            reference_energy * np.nan_to_num(reference_dipole, nan=0.0),
        )
        if field_angle_deg is not None:
            angle = float(field_angle_deg)
            if not np.isfinite(angle) or not 0.0 <= angle <= 180.0:
                raise ValueError("field_angle_deg must lie between 0 and 180 degrees")
            cosine_squared = np.cos(np.deg2rad(angle)) ** 2
            pi_factor = 1.5 * (1.0 - cosine_squared)
            sigma_factor = 0.75 * (1.0 + cosine_squared)
            angular_factor = np.where(delta_m == 0, pi_factor, sigma_factor)
            oscillator_proxy *= angular_factor
            reference_oscillator_proxy *= angular_factor

        full_excitation = initial - np.min(initial)
        full_boltzmann = np.exp(
            -full_excitation[:, np.newaxis]
            * _RYDBERG_ENERGY_ERG
            / (BOLTZMANN * temperature[np.newaxis, :])
        )
        full_normalization = np.sum(
            oscillator_proxy[:, np.newaxis] * full_boltzmann, axis=0
        )
        reference_excitation = reference_initial - np.min(reference_initial)
        reference_boltzmann = np.exp(
            -reference_excitation[:, np.newaxis]
            * _RYDBERG_ENERGY_ERG
            / (BOLTZMANN * temperature[np.newaxis, :])
        )
        reference_normalization = np.sum(
            reference_oscillator_proxy[:, np.newaxis] * reference_boltzmann,
            axis=0,
        )
        if np.any(full_normalization <= 0.0) or np.any(
            reference_normalization <= 0.0
        ):
            raise ValueError("H2db total component strengths vanish")

        # ``lower_population`` in the opacity layer is the *total* n=2 shell
        # population.  Convert the Boltzmann-weighted sum of oscillator
        # strengths above into an average per n=2 atom.  The former code used
        # only the numerator ratio and therefore implicitly increased or
        # decreased the number of absorbers when the four lower substates
        # split.  This partition normalization becomes order unity already at
        # 50 MG and is essential at several hundred MG.
        lower_state_current = self._lower_state_energies(field_strength_megagauss)
        lower_state_reference = dict.fromkeys(lower_state_current, -0.25)
        current_lower_energy = np.asarray(
            list(lower_state_current.values()), dtype=np.float64
        )
        reference_lower_energy = np.asarray(
            list(lower_state_reference.values()), dtype=np.float64
        )
        current_lower_weight = np.exp(
                -(current_lower_energy - np.min(current_lower_energy))[:, np.newaxis]
                * _RYDBERG_ENERGY_ERG
                / (BOLTZMANN * temperature[np.newaxis, :])
        )
        current_lower_partition = np.sum(current_lower_weight, axis=0)
        reference_lower_partition = np.sum(
            np.exp(
                -(
                    reference_lower_energy - np.min(reference_lower_energy)
                )[:, np.newaxis]
                * _RYDBERG_ENERGY_ERG
                / (BOLTZMANN * temperature[np.newaxis, :])
            ),
            axis=0,
        )
        line_strength_scale = (
            full_normalization
            / current_lower_partition
            * reference_lower_partition
            / reference_normalization
        )

        keep = oscillator_proxy >= minimum_relative_strength * np.max(
            oscillator_proxy
        )
        wavelength = wavelength[keep]
        oscillator_proxy = oscillator_proxy[keep]
        initial = initial[keep]
        delta_m = delta_m[keep]
        labels = labels[keep]
        # A constant shift of every lower-state energy cancels.  More tightly
        # bound substates receive the correct larger LTE population.
        excitation = initial - np.min(initial)
        boltzmann = np.exp(
            -excitation[:, np.newaxis]
            * _RYDBERG_ENERGY_ERG
            / (BOLTZMANN * temperature[np.newaxis, :])
        )
        strength = oscillator_proxy[:, np.newaxis] * boltzmann
        normalization = np.sum(strength, axis=0)
        if np.any(normalization <= 0.0):
            raise ValueError("H2db component strengths vanish at the requested field")
        strength /= normalization[np.newaxis, :]

        # H2db supplies absolute dipole strengths, not only component ratios.
        # Magnetic mixing transfers oscillator strength between field-free
        # Balmer manifolds. Retaining the ratio to the tabulated near-zero-
        # field sum restores that information while calibrating the absolute
        # line opacity to the validated nonmagnetic calculation.
        return H2dbBalmerComponents(
            wavelength.astype(np.float64),
            strength.astype(np.float64),
            line_strength_scale.astype(np.float64),
            delta_m.astype(np.int64),
            labels,
        )


def _archive_label(labels: NDArray[np.int16]) -> Path:
    lower_m, upper_m, lower_parity, upper_parity, lower_nu, upper_nu = (
        int(value) for value in labels[:6]
    )
    return Path(
        f"h2db/m_{lower_m:+d}_to_{upper_m:+d}/"
        f"pi_z_{lower_parity:+d}_to_{upper_parity:+d}/"
        f"nu_{lower_nu}_to_{upper_nu}"
    )


@lru_cache(maxsize=4)
def _read_h2db_cached(path_string: str) -> H2dbTransitionDatabase:
    """Read the Balmer subset and apply the physical state conventions.

    Two conventions of the public transition archive are corrected here.

    1. Its directory label ``m`` has the opposite sign to the energy archive
       (and to the Rohrmann/Vera-Rueda convention used by the EOS and the RWA
       continuum): its lower ``m=+1`` state is the tightly bound 2p state that
       the energy archive stores as ``m=-1``.  We store ``m_phys = -m_label``,
       so ``delta_m = +1`` is the blue-shifted (``q=+1``) sigma component, as
       in the linear-Zeeman branch.
    2. Delta m = 0 transitions are stored once per ``|m|``, for the tightly
       bound ``m_phys<0`` lower state only.  The mirrored ``m_phys>0`` partner
       has the same spatial wavefunctions, so its transition energy and
       dipole strength are identical, and both states are shifted by
       ``+4|m| beta`` Ry.  Omitting it removed ~1/6 of the field-free pi
       strength (pi fraction 0.275 instead of 1/3) and induced spurious
       linear dichroism.
    """

    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"H2db subset does not exist: {path}")
    with np.load(path) as values:
        labels = np.asarray(values["transition_labels"])
        offsets = np.asarray(values["transition_offsets"])
        beta = np.asarray(values["transition_beta"], dtype=np.float64)
        energy = np.asarray(values["transition_energy_rydberg"], dtype=np.float64)
        dipole = np.asarray(values["transition_dipole_strength"], dtype=np.float64)
        initial = np.asarray(
            values["transition_initial_energy_rydberg"], dtype=np.float64
        )
    transitions: dict[int, list[H2dbTransition]] = {}
    intrashell: list[H2dbTransition] = []
    for index, label in enumerate(labels):
        rows = slice(int(offsets[index]), int(offsets[index + 1]))
        lower_m = -int(label[0])
        upper_m = -int(label[1])
        upper_level = int(label[6])
        track_beta = np.ascontiguousarray(beta[rows])
        base = dict(
            source_path=_archive_label(label),
            lower_z_parity=int(label[2]),
            upper_z_parity=int(label[3]),
            lower_excitation_index=int(label[4]),
            upper_excitation_index=int(label[5]),
            balmer_upper_level=upper_level,
            beta=track_beta,
            transition_energy_rydberg=np.ascontiguousarray(energy[rows]),
            dipole_strength=np.ascontiguousarray(dipole[rows]),
        )
        if label.size > 7:
            # Paschen/Brackett subset: key (lower level, upper level).
            upper_level = (int(label[7]), upper_level)
        elif upper_level == 2:
            # n=2 -> 2 (subset label 2): no Delta m = 0 mirror is needed, the
            # only pi branch is 2p0 -> 2s0.
            intrashell.append(H2dbTransition(
                lower_magnetic_quantum_number=lower_m,
                upper_magnetic_quantum_number=upper_m,
                initial_state_energy_rydberg=np.ascontiguousarray(initial[rows]),
                **base,
            ))
            continue
        transitions.setdefault(upper_level, []).append(
            H2dbTransition(
                lower_magnetic_quantum_number=lower_m,
                upper_magnetic_quantum_number=upper_m,
                initial_state_energy_rydberg=np.ascontiguousarray(initial[rows]),
                **base,
            )
        )
        if lower_m == upper_m and lower_m != 0:
            if lower_m > 0:
                raise ValueError(
                    "H2db archive unexpectedly stores a loosely bound pi branch"
                )
            transitions[upper_level].append(
                H2dbTransition(
                    lower_magnetic_quantum_number=-lower_m,
                    upper_magnetic_quantum_number=-upper_m,
                    initial_state_energy_rydberg=np.ascontiguousarray(
                        initial[rows] + 4.0 * abs(lower_m) * track_beta
                    ),
                    **base,
                )
            )
    if labels.shape[1] > 7:
        return {key: tuple(items) for key, items in sorted(transitions.items())}  # type: ignore[return-value]
    if not all(level in transitions for level in range(3, 7)):
        raise ValueError("H2db subset is missing one or more of Halpha through Hdelta")
    return H2dbTransitionDatabase(
        path,
        {level: tuple(items) for level, items in sorted(transitions.items())},
        tuple(intrashell),
    )


def read_h2db_transition_database(
    path: str | Path,
) -> H2dbTransitionDatabase:
    """Read and cache the Balmer transitions of the bundled H2db subset."""

    return _read_h2db_cached(str(Path(path).expanduser().resolve()))


def read_h2db_paschen_brackett_transitions(
    path: str | Path,
) -> dict[tuple[int, int], tuple[H2dbTransition, ...]]:
    """Read the n=3, 4 subset (``tools/build_h2db_subset.py --paschen-brackett``).

    Keys are (lower, upper) principal quantum numbers of the zero-field line;
    the state conventions are those of the Balmer subset.
    """

    return _read_h2db_cached(str(Path(path).expanduser().resolve()))  # type: ignore[return-value]


@lru_cache(maxsize=4)
def _read_h2db_energies_cached(path_string: str) -> H2dbEnergyDatabase:
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"H2db subset does not exist: {path}")
    with np.load(path) as values:
        labels = np.asarray(values["energy_labels"])
        offsets = np.asarray(values["energy_offsets"])
        beta = np.asarray(values["energy_beta"], dtype=np.float64)
        energy = np.asarray(values["energy_rydberg"], dtype=np.float64)
    tracks: dict[tuple[int, int, int], H2dbEnergyTrack] = {}
    for index, label in enumerate(labels):
        stored_m, z_parity, excitation_index = (int(value) for value in label)
        if stored_m > 0:
            raise ValueError("unexpected positive-m H2db energy track")
        principal, orbital, longitudinal = _zero_field_labels(
            abs(stored_m), z_parity, excitation_index
        )
        rows = slice(int(offsets[index]), int(offsets[index + 1]))
        track_beta = np.ascontiguousarray(beta[rows])
        if track_beta.size < 2 or np.any(np.diff(track_beta) <= 0.0):
            raise ValueError("invalid H2db energy grid")
        key = (principal, orbital, abs(stored_m))
        if key in tracks:
            raise ValueError(f"duplicate H2db energy state {key}")
        tracks[key] = H2dbEnergyTrack(
            Path(f"h2db/m_{stored_m}/pi_z_{z_parity:+d}/nu_{excitation_index}"),
            principal,
            orbital,
            abs(stored_m),
            z_parity,
            longitudinal,
            track_beta,
            np.ascontiguousarray(energy[rows]),
        )
    database = H2dbEnergyDatabase(path, tracks)
    if database.maximum_complete_principal_quantum_number < 5:
        raise ValueError("H2db energy subset is incomplete below n=5")
    return database


@dataclass(frozen=True)
class MotionalStarkTable:
    """Delta alpha = alpha_up - alpha_low [atomic units] of H2db components, and
    exact F-averaged profiles where second order fails.

    Built by ``tools/build_motional_stark_table.py``; Delta alpha is NaN where
    the second-order motional shift does not apply (see there). ``profiles``
    holds per label the quantiles [A from the F = 0 line] of the blue and red
    parts ([field, temperature, 2 * len(profile_levels)]) and their strength
    shares relative to the F = 0 line ([field, temperature, 2]); NaN where the
    component has no exact profile.
    """

    field_megagauss: FloatArray
    delta_alpha: dict[tuple[int, ...], FloatArray]
    profile_field_megagauss: FloatArray = dataclass_field(default_factory=lambda: np.empty(0))
    profile_temperature_kelvin: FloatArray = dataclass_field(default_factory=lambda: np.empty(0))
    profile_levels: FloatArray = dataclass_field(default_factory=lambda: np.empty(0))
    profiles: dict[tuple[int, ...], tuple[FloatArray, FloatArray]] = dataclass_field(default_factory=dict)

    def delta_alpha_at(self, labels: ArrayLike, field_strength_megagauss: float) -> float:
        """Return tabulated nodes or interpolate between two valid neighbours."""

        values = self.delta_alpha.get(tuple(int(x) for x in labels))
        grid = self.field_megagauss
        if values is None or not grid[0] <= field_strength_megagauss <= grid[-1]:
            return math.nan
        upper = int(np.searchsorted(grid, field_strength_megagauss))
        if grid[upper] == field_strength_megagauss:
            return float(values[upper])
        index = upper - 1
        weight = (field_strength_megagauss - grid[index]) / (grid[index + 1] - grid[index])
        return float((1.0 - weight) * values[index] + weight * values[index + 1])

    def profile_at(
        self, labels: ArrayLike, field_strength_megagauss: float, temperature: FloatArray
    ) -> tuple[FloatArray, FloatArray] | None:
        """Exact profile per temperature: knots [A from the F = 0 line] and the
        cumulative strength there (relative to the F = 0 line), [depth, knot],
        piecewise linear between knots. Between field nodes the quantiles and
        shares are linear in B (a core that moves with B moves); between
        temperature nodes the cumulative distributions are blended linearly (T
        clamped to the nodes): the builder's lines sit at fixed F^2 cells and T
        only reweights them, so a profile between temperature nodes is a
        mixture of the nodes' profiles. On a node only that node counts; a
        single node without a profile (the builder drops a node when the lower
        state mixes in some F cells) is bridged by its two neighbours. None
        outside the table or next to two or more nodes without a profile."""

        entry = self.profiles.get(tuple(int(x) for x in labels))
        grid = self.profile_field_megagauss
        if entry is None or not grid[0] <= field_strength_megagauss <= grid[-1]:
            return None
        knots, shares = entry
        index = min(int(np.searchsorted(grid, field_strength_megagauss, side="right")) - 1, grid.size - 2)
        weight = (field_strength_megagauss - grid[index]) / (grid[index + 1] - grid[index])
        lo, hi = (index + 1 if weight == 1.0 else index), (index if weight == 0.0 else index + 1)  # on a node: that node only
        finite = np.isfinite(shares).all(axis=(1, 2))
        below, above = np.flatnonzero(finite[:lo + 1]), hi + np.flatnonzero(finite[hi:])
        if below.size == 0 or above.size == 0 or above[0] - below[-1] > 2:
            return None
        lo, hi = below[-1], above[0]
        weight = 0.0 if hi == lo else (field_strength_megagauss - grid[lo]) / (grid[hi] - grid[lo])
        knots = (1.0 - weight) * knots[lo] + weight * knots[hi]  # [temperature node, knot]
        shares = (1.0 - weight) * shares[lo] + weight * shares[hi]
        levels = self.profile_levels
        cumulative = np.concatenate((shares[:, :1] * levels, shares[:, :1] + shares[:, 1:] * levels), axis=-1)
        knots = knots + 1.0e-9 * np.arange(knots.shape[-1])  # strictly increasing for np.interp
        nodes = self.profile_temperature_kelvin
        t = np.clip(np.asarray(temperature, dtype=np.float64), nodes[0], nodes[-1])
        j = np.clip(np.searchsorted(nodes, t, side="right") - 1, 0, nodes.size - 2)
        u = (t - nodes[j]) / (nodes[j + 1] - nodes[j])
        out_knots = np.sort(np.concatenate((knots[j], knots[j + 1]), axis=1), axis=1)
        out_cumulative = np.empty_like(out_knots)
        for d in range(t.size):
            out_cumulative[d] = ((1.0 - u[d]) * np.interp(out_knots[d], knots[j[d]], cumulative[j[d]])
                                 + u[d] * np.interp(out_knots[d], knots[j[d] + 1], cumulative[j[d] + 1]))
        return out_knots, out_cumulative


@lru_cache(maxsize=2)
def _read_motional_stark_cached(path_string: str) -> MotionalStarkTable:
    with np.load(path_string) as values:
        knots = np.concatenate((values["profile_blue_angstrom"], values["profile_red_angstrom"]), axis=-1)
        shares = np.stack((values["profile_blue_share"], values["profile_red_share"]), axis=-1)
        labels = [tuple(int(x) for x in label) for label in values["labels"]]
        return MotionalStarkTable(
            np.asarray(values["field_megagauss"], dtype=np.float64),
            {label: np.asarray(row, dtype=np.float64) for label, row in zip(labels, values["delta_alpha"])},
            np.asarray(values["profile_field_megagauss"], dtype=np.float64),
            np.asarray(values["profile_temperature_kelvin"], dtype=np.float64),
            np.asarray(values["profile_quantile_levels"], dtype=np.float64),
            {label: (np.asarray(k, dtype=np.float64), np.asarray(s, dtype=np.float64))
             for label, k, s in zip(labels, knots, shares) if np.isfinite(s).any()},
        )


def read_motional_stark_table(path: str | Path) -> MotionalStarkTable:
    """Read and cache the bundled motional Stark table."""
    return _read_motional_stark_cached(str(Path(path).expanduser().resolve()))


def read_h2db_energy_database(path: str | Path) -> H2dbEnergyDatabase:
    """Read and cache the stationary-state energies of the bundled subset."""

    return _read_h2db_energies_cached(str(Path(path).expanduser().resolve()))


__all__ = [
    "H2DB_DATASET_DOI",
    "H2DB_REFERENCE_FIELD_MEGAGAUSS",
    "H2DB_VANISHING_BRANCH_ENERGY_RYDBERG",
    "H2dbBalmerComponents",
    "H2dbEnergyDatabase",
    "H2dbEnergyTrack",
    "H2dbIntrashellComponents",
    "H2dbTransition",
    "H2dbTransitionDatabase",
    "MotionalStarkTable",
    "read_h2db_energy_database",
    "read_h2db_paschen_brackett_transitions",
    "read_motional_stark_table",
    "read_h2db_transition_database",
]
