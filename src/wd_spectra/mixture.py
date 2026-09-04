"""Opacity helpers for homogeneous warm hydrogen--helium atmospheres."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._compat import trapezoid
from .atmosphere import Atmosphere
from .constants import BOLTZMANN, LIGHT_SPEED, PLANCK
from .eos import HeliumLTEState
from .helium import (
    helium_continuum_mass_absorption_coefficient,
    helium_rayleigh_scattering_mass_coefficient,
)
from .opacity import (
    _slice_hydrogen_lte_state,
    electron_scattering_mass_coefficient,
    hydrogen_continuum_mass_absorption_coefficient,
    hydrogen_rayleigh_scattering_mass_coefficient,
)


FloatArray = NDArray[np.float64]


def hydrogen_helium_continuum_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    include_electron_scattering: bool = True,
    include_rayleigh_scattering: bool = True,
    include_helium_dimer_ion: bool = True,
    include_helium_three_body_cia: bool = True,
    include_rydberg_bound_free: bool = True,
) -> FloatArray:
    """Return the summed atomic H/He continuum in cm2 g-1.

    Electron scattering is evaluated once using the shared electron density.
    Molecular H/He opacity is intentionally outside this warm-atmosphere
    milestone.
    """

    if atmosphere.hydrogen_lte_state is None or atmosphere.helium_lte_state is None:
        raise ValueError("both hydrogen_lte_state and helium_lte_state are required")
    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    opacity = hydrogen_continuum_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        include_electron_scattering=False,
        include_rayleigh_scattering=False,
        include_molecular_absorption=False,
    )
    opacity += helium_continuum_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        include_electron_scattering=False,
        include_rayleigh_scattering=False,
        include_helium_dimer_ion=include_helium_dimer_ion,
        include_helium_three_body_cia=include_helium_three_body_cia,
        include_rydberg_bound_free=include_rydberg_bound_free,
    )
    if include_electron_scattering:
        opacity += electron_scattering_mass_coefficient(atmosphere)[np.newaxis, :]
    if include_rayleigh_scattering:
        opacity += hydrogen_rayleigh_scattering_mass_coefficient(
            atmosphere, wavelength
        )
        opacity += helium_rayleigh_scattering_mass_coefficient(
            atmosphere, wavelength
        )
    return np.maximum(opacity, np.finfo(np.float64).tiny)


def _slice_helium_lte_state(
    state: HeliumLTEState, depth: int
) -> HeliumLTEState:
    return HeliumLTEState(
        **{
            field: getattr(state, field)[depth : depth + 1]
            if isinstance(getattr(state, field), np.ndarray)
            else getattr(state, field)
            for field in state.__dataclass_fields__
        }
    )


def rosseland_mean_hydrogen_helium_continuum_opacity(
    atmosphere: Atmosphere,
    *,
    n_frequency: int = 240,
    include_helium_dimer_ion: bool = True,
    include_helium_three_body_cia: bool = True,
    include_rydberg_bound_free: bool = True,
) -> FloatArray:
    """Return the Rosseland mean of the summed warm H/He continuum."""

    hydrogen = atmosphere.hydrogen_lte_state
    helium = atmosphere.helium_lte_state
    if hydrogen is None or helium is None:
        raise ValueError("a homogeneous H/He Atmosphere is required")
    if n_frequency < 40:
        raise ValueError("n_frequency must be at least 40")
    dimensionless_frequency = np.geomspace(0.1, 30.0, n_frequency)
    exponential = np.exp(dimensionless_frequency)
    weight = (
        dimensionless_frequency**4
        * exponential
        / np.expm1(dimensionless_frequency) ** 2
    )
    result = np.empty(atmosphere.n_depth, dtype=np.float64)
    for depth in range(atmosphere.n_depth):
        wavelength = (
            PLANCK
            * LIGHT_SPEED
            / (
                BOLTZMANN
                * atmosphere.temperature[depth]
                * dimensionless_frequency
            )
            * 1.0e8
        )
        order = np.argsort(wavelength)
        point = Atmosphere(
            effective_temperature=atmosphere.effective_temperature,
            logg=atmosphere.logg,
            rosseland_optical_depth=atmosphere.rosseland_optical_depth[
                depth : depth + 1
            ],
            column_mass=atmosphere.column_mass[depth : depth + 1],
            temperature=atmosphere.temperature[depth : depth + 1],
            gas_pressure=atmosphere.gas_pressure[depth : depth + 1],
            mass_density=atmosphere.mass_density[depth : depth + 1],
            neutral_h_density=atmosphere.neutral_h_density[depth : depth + 1],
            proton_density=atmosphere.proton_density[depth : depth + 1],
            electron_density=atmosphere.electron_density[depth : depth + 1],
            metadata=atmosphere.metadata,
            hydrogen_lte_state=_slice_hydrogen_lte_state(hydrogen, depth),
            helium_lte_state=_slice_helium_lte_state(helium, depth),
        )
        opacity = hydrogen_helium_continuum_mass_absorption_coefficient(
            point,
            wavelength[order],
            include_helium_dimer_ion=include_helium_dimer_ion,
            include_helium_three_body_cia=include_helium_three_body_cia,
            include_rydberg_bound_free=include_rydberg_bound_free,
        )[:, 0]
        inverse_mean = trapezoid(
            weight[order] / np.maximum(opacity, 1.0e-30),
            dimensionless_frequency[order],
        ) / trapezoid(weight[order], dimensionless_frequency[order])
        result[depth] = 1.0 / inverse_mean
    return result
