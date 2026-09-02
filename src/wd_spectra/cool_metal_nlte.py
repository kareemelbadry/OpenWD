"""Approximate non-LTE source functions for cool-star resonance lines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .atmosphere import Atmosphere
from .chianti import read_chianti_scaled_collision_components
from .constants import BOLTZMANN, LIGHT_SPEED, PLANCK
from .metals import AtomicDatabase


FloatArray = NDArray[np.float64]
_EFFECTIVE_COLLISION_RATE_CONSTANT = 8.629e-6


@dataclass(frozen=True)
class ResonanceScatteringProbability:
    """Depth-dependent resonant re-emission probability for one line."""

    probability: FloatArray
    resonant_einstein_a: float
    total_radiative_rate: float
    source: str


def ca_ii_resonance_scattering_probabilities(
    atmosphere: Atmosphere,
    atomic_database: AtomicDatabase,
    collision_strength_path: str | Path,
) -> Mapping[tuple[int, int], ResonanceScatteringProbability]:
    r"""Return physical H/K scattering probabilities at every depth.

    The upper ``4p ^2P^o`` level can resonantly decay to the ground state,
    radiatively branch into the infrared triplet, or be collisionally
    transferred to another Ca II level.  In this reduced-atom treatment the
    first channel is coherent line scattering and the latter channels destroy
    an H/K photon.  Electron rates use the R-matrix effective collision
    strengths supplied in the CHIANTI Ca II ``.scups`` file,

    ``q_ij = 8.629e-6 Upsilon/(g_i sqrt(T)) exp(-Delta E/kT)``

    for upward transitions, with the exponential omitted for downward ones.
    Ca II populations and extinction remain LTE, so this is a fixed-structure
    source-function treatment rather than a full multilevel NLTE solve.
    """

    ion = atomic_database.ions.get(("Ca", 1))
    if ion is None:
        raise ValueError("atomic database does not contain Ca II")
    levels = {level.index: level for level in ion.levels}
    collisions = read_chianti_scaled_collision_components(
        collision_strength_path
    )
    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    electron_density = np.asarray(atmosphere.electron_density, dtype=np.float64)
    if np.any(temperature <= 0.0) or np.any(electron_density < 0.0):
        raise ValueError("atmosphere temperatures and electron densities are invalid")

    radiative_rate_by_upper: dict[int, float] = {}
    for transition in ion.transitions:
        radiative_rate_by_upper[transition.upper_index] = (
            radiative_rate_by_upper.get(transition.upper_index, 0.0)
            + transition.einstein_a
        )

    resonance = tuple(
        transition
        for transition in ion.transitions
        if transition.lower_index == 1
        and 3920.0 < transition.wavelength_vacuum_angstrom < 3990.0
    )
    if len(resonance) != 2:
        raise ValueError("Ca II atomic data must contain both H and K transitions")

    result: dict[tuple[int, int], ResonanceScatteringProbability] = {}
    for transition in resonance:
        upper = levels[transition.upper_index]
        collisional_exit_coefficient = np.zeros_like(temperature)
        for (lower_index, upper_index), fit in collisions.items():
            if transition.upper_index not in (lower_index, upper_index):
                continue
            other_index = (
                upper_index
                if lower_index == transition.upper_index
                else lower_index
            )
            other = levels.get(other_index)
            if other is None:
                continue
            upsilon = np.asarray(
                [fit.effective_collision_strength(value) for value in temperature],
                dtype=np.float64,
            )
            coefficient = (
                _EFFECTIVE_COLLISION_RATE_CONSTANT
                * upsilon
                / (upper.statistical_weight * np.sqrt(temperature))
            )
            energy_difference = (
                other.energy_wavenumber - upper.energy_wavenumber
            ) * PLANCK * LIGHT_SPEED
            if energy_difference > 0.0:
                coefficient *= np.exp(
                    -energy_difference / (BOLTZMANN * temperature)
                )
            collisional_exit_coefficient += coefficient
        collisional_rate = electron_density * collisional_exit_coefficient
        total_radiative_rate = radiative_rate_by_upper[transition.upper_index]
        probability = transition.einstein_a / (
            total_radiative_rate + collisional_rate
        )
        result[(transition.lower_index, transition.upper_index)] = (
            ResonanceScatteringProbability(
                probability=np.clip(probability, 0.0, 1.0),
                resonant_einstein_a=transition.einstein_a,
                total_radiative_rate=total_radiative_rate,
                source=(
                    "CHIANTI Ca II electron collision strengths; radiative "
                    "branching from the supplied atomic database"
                ),
            )
        )
    return MappingProxyType(result)
