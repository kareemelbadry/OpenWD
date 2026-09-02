"""Published structure diagnostics for cool hydrogen atmospheres.

This module deliberately separates literature-controlled mean-3D structure
experiments from the production one-dimensional atmosphere solver.  It does
not turn a 1D atmosphere into a hydrodynamical calculation; it provides a
reproducible way to test the temperature-structure difference shown by
Tremblay et al. (2013, A&A 552, A13, Fig. 7).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .atmosphere import Atmosphere, _solve_bracketed_log_root
from .eos import hummer_mihalas_hydrogen_lte


FloatArray = NDArray[np.float64]


# Digitized from the vector paths of Tremblay et al. (2013), Fig. 7.  The
# plotted 3D and 1D curves are shifted by the same vertical amount for each
# model, so their difference is unaffected by the presentation offsets.  The
# paper supplies log(g)=8 simulations only.  Values are rounded to the nearest
# kelvin, which is more precise than the line width in the published figure.
TREMBLAY_2013_LOG_ROSSELAND_DEPTH = np.asarray(
    [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0,
     -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
    dtype=np.float64,
)
TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K = np.asarray(
    [5_997.0, 7_012.0, 8_032.0], dtype=np.float64
)
TREMBLAY_2013_MEAN_3D_MINUS_1D_TEMPERATURE_K = np.asarray(
    [
        [-641, -533, -359, -199, -8, 71, -2, -38, -38, -35, -45, -66, -66, -94],
        [-1057, -925, -755, -532, -253, 7, 182, -7, -79, -79, -65, -81, -89, -93],
        [-1529, -1359, -1076, -736, -372, -10, 227, -4, -112, -58, -18, -10, -28, -19],
    ],
    dtype=np.float64,
)


def tremblay_2013_mean_3d_temperature_difference(
    effective_temperature: float,
    rosseland_optical_depth: ArrayLike,
) -> FloatArray:
    """Return the published mean-3D minus 1D temperature difference.

    The interpolation is limited to the 5997--8032 K, log(g)=8 portion of
    Fig. 7 that brackets the present very-cool DA validation sample.  It is a
    digitized diagnostic, not a substitute for a CO5BOLD structure grid.
    """

    teff = float(effective_temperature)
    tau = np.asarray(rosseland_optical_depth, dtype=np.float64)
    if not np.isfinite(teff):
        raise ValueError("effective_temperature must be finite")
    if (
        teff < TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K[0]
        or teff > TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K[-1]
    ):
        raise ValueError(
            "the digitized Tremblay et al. (2013) cool-DA differential "
            "covers 5997--8032 K"
        )
    if np.any(~np.isfinite(tau)) or np.any(tau <= 0.0):
        raise ValueError("rosseland_optical_depth must be finite and positive")

    log_tau = np.log10(tau)
    depth_curves = np.vstack(
        [
            np.interp(
                log_tau,
                TREMBLAY_2013_LOG_ROSSELAND_DEPTH,
                row,
                left=row[0],
                right=row[-1],
            )
            for row in TREMBLAY_2013_MEAN_3D_MINUS_1D_TEMPERATURE_K
        ]
    )
    upper = int(
        np.searchsorted(
            TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K, teff, side="right"
        )
    )
    upper = min(max(upper, 1), TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K.size - 1)
    lower = upper - 1
    fraction = (
        (teff - TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K[lower])
        / (
            TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K[upper]
            - TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K[lower]
        )
    )
    return np.asarray(
        (1.0 - fraction) * depth_curves[lower] + fraction * depth_curves[upper],
        dtype=np.float64,
    )


def atmosphere_with_tremblay_2013_mean_3d_temperature_difference(
    atmosphere: Atmosphere,
    *,
    neutral_radius_scale: float | None = None,
    maximum_logg_offset: float = 0.4,
) -> Atmosphere:
    """Apply the Fig. 7 temperature differential and recompute the H EOS.

    Gas pressure, column mass, and the original Rosseland coordinate remain
    fixed.  This is the same kind of differential experiment used in the
    paper, but it lacks the published mean-3D pressure structure and is not
    guaranteed to satisfy radiative equilibrium.  The guard against large
    gravity offsets prevents silent use far from the sole published log(g)=8
    sequence.
    """

    if atmosphere.hydrogen_lte_state is None:
        raise ValueError("a pure-hydrogen LTE atmosphere is required")
    if abs(atmosphere.logg - 8.0) > float(maximum_logg_offset):
        raise ValueError(
            "the Tremblay et al. (2013) structure differential is only "
            "published at log(g)=8"
        )
    difference = tremblay_2013_mean_3d_temperature_difference(
        atmosphere.effective_temperature,
        atmosphere.rosseland_optical_depth,
    )
    temperature = atmosphere.temperature + difference
    if np.any(temperature <= 0.0):
        raise RuntimeError("the mean-3D temperature differential is unphysical")
    original = atmosphere.hydrogen_lte_state
    radius_scale = (
        original.neutral_radius_scale
        if neutral_radius_scale is None
        else float(neutral_radius_scale)
    )
    maximum_level = (
        original.level_population_density.shape[-1]
        if original.level_population_density is not None
        else 40
    )
    state = hummer_mihalas_hydrogen_lte(
        temperature,
        atmosphere.gas_pressure,
        maximum_level=maximum_level,
        neutral_radius_scale=radius_scale,
        correlated_microfields=(original.microfield_model == "qmhd"),
        include_molecules=(original.molecular_hydrogen_density is not None),
        include_negative_hydrogen=(
            original.negative_hydrogen_density is not None
        ),
        trihydrogen_ion_partition_model=(
            original.trihydrogen_ion_partition_model
        ),
    )
    return replace(
        atmosphere,
        temperature=temperature,
        mass_density=state.mass_density,
        neutral_h_density=state.neutral_h_density,
        proton_density=state.proton_density,
        electron_density=state.electron_density,
        hydrogen_lte_state=state,
        metadata={
            **atmosphere.metadata,
            "mean_3d_temperature_differential": (
                "Tremblay et al. 2013 Fig. 7 vector digitization"
            ),
            "mean_3d_temperature_differential_reference_logg": 8.0,
            "mean_3d_temperature_differential_preserves_pressure": True,
            "mean_3d_temperature_differential_is_equilibrium_model": False,
            "mean_3d_temperature_differential_neutral_radius_scale": (
                radius_scale
            ),
        },
    )


def hydrostatic_atmosphere_with_tremblay_2013_mean_3d_temperature_difference(
    atmosphere: Atmosphere,
    *,
    h2_h2_cia_table=None,
    maximum_logg_offset: float = 0.4,
) -> Atmosphere:
    """Apply the published T difference and reintegrate hydrostatic balance.

    Unlike :func:`atmosphere_with_tremblay_2013_mean_3d_temperature_difference`,
    this variant treats the published Rosseland-depth coordinate as fixed and
    solves ``dP/dtau_R = g/kappa_R`` with the corrected temperature and the
    implemented continuum Rosseland opacity.  It remains a mean-3D
    differential diagnostic because the paper's pressure structure and
    turbulent pressure are unavailable.
    """

    from .opacity import rosseland_mean_hydrogen_continuum_opacity

    fixed_pressure = (
        atmosphere_with_tremblay_2013_mean_3d_temperature_difference(
            atmosphere,
            maximum_logg_offset=maximum_logg_offset,
        )
    )
    temperature = fixed_pressure.temperature
    tau = atmosphere.rosseland_optical_depth
    gravity = atmosphere.gravity
    original = atmosphere.hydrogen_lte_state
    assert original is not None
    correlated_microfields = original.microfield_model == "qmhd"
    include_molecules = original.molecular_hydrogen_density is not None
    maximum_level = (
        original.level_population_density.shape[-1]
        if original.level_population_density is not None
        else 40
    )

    def opacity_at(temp: float, pressure: float) -> float:
        state = hummer_mihalas_hydrogen_lte(
            np.asarray([temp]),
            np.asarray([pressure]),
            maximum_level=maximum_level,
            neutral_radius_scale=original.neutral_radius_scale,
            correlated_microfields=correlated_microfields,
            include_molecules=include_molecules,
            include_negative_hydrogen=(
                original.negative_hydrogen_density is not None
            ),
            trihydrogen_ion_partition_model=(
                original.trihydrogen_ion_partition_model
            ),
        )
        point = Atmosphere(
            effective_temperature=atmosphere.effective_temperature,
            logg=atmosphere.logg,
            rosseland_optical_depth=np.asarray([1.0]),
            column_mass=np.asarray([pressure / gravity]),
            temperature=np.asarray([temp]),
            gas_pressure=np.asarray([pressure]),
            mass_density=np.atleast_1d(state.mass_density),
            neutral_h_density=np.atleast_1d(state.neutral_h_density),
            proton_density=np.atleast_1d(state.proton_density),
            electron_density=np.atleast_1d(state.electron_density),
            metadata={},
            hydrogen_lte_state=state,
        )
        return float(
            rosseland_mean_hydrogen_continuum_opacity(
                point,
                h2_h2_cia_table=h2_h2_cia_table,
            )[0]
        )

    def pressure_increment(
        previous_pressure: float,
        delta_tau: float,
        midpoint_temperature: float,
    ) -> float:
        target = gravity * delta_tau

        def residual(log_increment: float) -> float:
            increment = np.exp(log_increment)
            midpoint_pressure = previous_pressure + 0.5 * increment
            return (
                log_increment
                + np.log(opacity_at(midpoint_temperature, midpoint_pressure))
                - np.log(target)
            )

        lower = np.log(max(target / 1.0e12, 1.0e-20))
        upper = np.log(max(target / 1.0e-12, 1.0e-19))
        return float(
            np.exp(_solve_bracketed_log_root(residual, lower, upper))
        )

    pressure = np.empty_like(tau)
    pressure[0] = pressure_increment(0.0, float(tau[0]), float(temperature[0]))
    for index in range(1, tau.size):
        pressure[index] = pressure[index - 1] + pressure_increment(
            float(pressure[index - 1]),
            float(tau[index] - tau[index - 1]),
            float(0.5 * (temperature[index] + temperature[index - 1])),
        )
    state = hummer_mihalas_hydrogen_lte(
        temperature,
        pressure,
        maximum_level=maximum_level,
        neutral_radius_scale=original.neutral_radius_scale,
        correlated_microfields=correlated_microfields,
        include_molecules=include_molecules,
        include_negative_hydrogen=(
            original.negative_hydrogen_density is not None
        ),
        trihydrogen_ion_partition_model=(
            original.trihydrogen_ion_partition_model
        ),
    )
    return replace(
        fixed_pressure,
        gas_pressure=pressure,
        column_mass=pressure / gravity,
        mass_density=state.mass_density,
        neutral_h_density=state.neutral_h_density,
        proton_density=state.proton_density,
        electron_density=state.electron_density,
        hydrogen_lte_state=state,
        metadata={
            **fixed_pressure.metadata,
            "mean_3d_temperature_differential_preserves_pressure": False,
            "mean_3d_temperature_differential_hydrostatic_reintegration": (
                "dP/dtau_R=g/kappa_R; continuum Rosseland mean"
            ),
            "mean_3d_temperature_differential_turbulent_pressure": False,
        },
    )


__all__ = [
    "TREMBLAY_2013_EFFECTIVE_TEMPERATURE_K",
    "TREMBLAY_2013_MEAN_3D_MINUS_1D_TEMPERATURE_K",
    "TREMBLAY_2013_LOG_ROSSELAND_DEPTH",
    "atmosphere_with_tremblay_2013_mean_3d_temperature_difference",
    "hydrostatic_atmosphere_with_tremblay_2013_mean_3d_temperature_difference",
    "tremblay_2013_mean_3d_temperature_difference",
]
