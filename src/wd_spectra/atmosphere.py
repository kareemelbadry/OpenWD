"""One-dimensional white-dwarf atmosphere structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .eos import (
    HeliumLTEState,
    HydrogenHeliumLTEState,
    HydrogenLTEState,
    hummer_mihalas_helium_lte,
    hummer_mihalas_helium_lte_with_reos3,
    hummer_mihalas_hydrogen_helium_lte,
    hummer_mihalas_hydrogen_lte,
    ideal_hydrogen_lte,
)

if TYPE_CHECKING:
    from .dense_eos import HeliumREOS3Table
    from .d6 import TOPbasePhotoionizationDatabase
    from .metals import (
        AtomicDatabase,
        CaIHeProfileTable,
        CaIIHeProfileTable,
        MgHeRedWingTable,
        MgIIHeProfileTable,
        VernerPhotoionizationDatabase,
    )
    from .molecules import H2H2CollisionInducedAbsorptionTable
    from .hydrogen_self import BarklemSelfBroadeningTable
    from .jackson_lyman import JacksonLymanProfileTable
    from .quasimolecular import AllardUnifiedLymanTable


FloatArray = NDArray[np.float64]


def _metal_line_opacity_sampling_grid(
    line_centers_angstrom: FloatArray,
    *,
    maximum_wing_sampled_lines: int = 1_000,
) -> tuple[FloatArray, int]:
    """Return a bounded transfer grid for a ranked structural line list.

    ``selected_metal_lines`` returns lines from strongest to weakest.  Five
    samples are retained for the strongest lines, whose resolved cores and
    near wings materially affect the flux integral.  The remaining line
    forest receives a compact three-point core stencil.  Keeping bracketing
    samples is important: a single isolated line-center point would acquire
    an arbitrary trapezoidal wavelength weight set by the next unrelated
    transition.  This bounds the grid growth without changing the selected
    opacity contributors or assigning weak lines artificially broad bins.
    """

    centers = np.asarray(line_centers_angstrom, dtype=np.float64)
    if centers.ndim != 1 or np.any(~np.isfinite(centers)):
        raise ValueError("metal line centers must be a finite one-dimensional array")
    if maximum_wing_sampled_lines < 0:
        raise ValueError("maximum_wing_sampled_lines must be non-negative")
    if centers.size == 0:
        return centers.copy(), 0
    n_wing = min(int(maximum_wing_sampled_lines), int(centers.size))
    samples: list[FloatArray] = []
    if n_wing:
        offsets = np.asarray((-2.0, -0.5, 0.0, 0.5, 2.0))
        samples.append((centers[:n_wing, np.newaxis] + offsets).ravel())
    if n_wing < centers.size:
        core_offsets = np.asarray((-0.05, 0.0, 0.05))
        samples.append(
            (centers[n_wing:, np.newaxis] + core_offsets).ravel()
        )
    return np.unique(np.concatenate(samples)), n_wing


def _solve_bracketed_log_root(
    residual: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    residual_tolerance: float = 2.0e-11,
    interval_tolerance: float = 2.0e-11,
    maximum_iterations: int = 40,
) -> float:
    """Solve a monotone scalar residual with safeguarded secant steps.

    The hydrostatic equation is best conditioned in the logarithm of the
    pressure increment.  Its residual is nearly linear there, so bisection's
    fixed 45 opacity evaluations are unnecessarily expensive.  Secant steps
    converge rapidly, while the retained sign-changing bracket makes the
    solve as robust as the former bisection.
    """

    function = residual  # Keep the hot-loop calls below compact.
    f_lower = float(function(lower))
    while f_lower > 0.0:
        lower -= np.log(10.0)
        f_lower = float(function(lower))
    f_upper = float(function(upper))
    while f_upper < 0.0:
        upper += np.log(10.0)
        f_upper = float(function(upper))

    replaced_side = 0
    for _ in range(maximum_iterations):
        width = upper - lower
        if width <= interval_tolerance:
            return 0.5 * (lower + upper)
        candidate = upper - f_upper * width / (f_upper - f_lower)
        if not lower < candidate < upper:
            candidate = 0.5 * (lower + upper)
        f_candidate = float(function(candidate))
        if abs(f_candidate) <= residual_tolerance:
            return candidate
        if f_candidate < 0.0:
            lower, f_lower = candidate, f_candidate
            if replaced_side < 0:
                f_upper *= 0.5
            replaced_side = -1
        else:
            upper, f_upper = candidate, f_candidate
            if replaced_side > 0:
                f_lower *= 0.5
            replaced_side = 1
    return 0.5 * (lower + upper)


def _smooth_radiative_temperature_correction(
    correction: FloatArray,
    convective_weight: FloatArray,
) -> FloatArray:
    """Smooth a radiative correction without leaking into convection."""

    radiative = correction * (1.0 - convective_weight)
    smoothed = radiative.copy()
    smoothed[1:-1] = (
        0.25 * radiative[:-2]
        + 0.5 * radiative[1:-1]
        + 0.25 * radiative[2:]
    )
    # The stencil crosses the radiative/convective boundary, so the mask has
    # to be applied both before and after smoothing.
    smoothed *= 1.0 - convective_weight
    return smoothed


def _node_values_on_upper_interfaces(
    values: FloatArray,
    *,
    surface_value: float | None = None,
) -> FloatArray:
    """Map depth-node values to the upper cell interfaces."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or np.any(~np.isfinite(array)):
        raise ValueError("values must be a finite one-dimensional depth array")
    result = np.empty_like(array)
    result[0] = array[0] if surface_value is None else float(surface_value)
    result[1:] = 0.5 * (array[:-1] + array[1:])
    return result


def _upper_interface_values_on_nodes(values: FloatArray) -> FloatArray:
    """Map upper-interface values back to the atmosphere depth nodes."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or np.any(~np.isfinite(array)):
        raise ValueError("values must be a finite one-dimensional depth array")
    result = np.empty_like(array)
    result[:-1] = 0.5 * (array[:-1] + array[1:])
    result[-1] = array[-1]
    return result


@dataclass(frozen=True)
class Atmosphere:
    """A plane-parallel atmosphere sampled from the surface inward.

    All quantities use cgs units.  The optical-depth and column-mass arrays
    must increase inward.
    """

    effective_temperature: float
    logg: float
    rosseland_optical_depth: FloatArray
    column_mass: FloatArray
    temperature: FloatArray
    gas_pressure: FloatArray
    mass_density: FloatArray
    neutral_h_density: FloatArray
    proton_density: FloatArray
    electron_density: FloatArray
    metadata: dict[str, object]
    hydrogen_lte_state: HydrogenLTEState | None = None
    helium_lte_state: HeliumLTEState | None = None

    @property
    def gravity(self) -> float:
        """Surface gravity in cm s^-2."""

        return 10.0**self.logg

    @property
    def n_depth(self) -> int:
        return int(self.temperature.size)


def _atmosphere_from_hydrogen_helium_state(
    effective_temperature: float,
    logg: float,
    rosseland_optical_depth: FloatArray,
    column_mass: FloatArray,
    temperature: FloatArray,
    gas_pressure: FloatArray,
    state: HydrogenHeliumLTEState,
    metadata: dict[str, object],
) -> Atmosphere:
    """Build an :class:`Atmosphere` from one shared mixed-composition EOS."""

    hydrogen = state.hydrogen_lte_state
    return Atmosphere(
        effective_temperature=float(effective_temperature),
        logg=float(logg),
        rosseland_optical_depth=rosseland_optical_depth,
        column_mass=column_mass,
        temperature=temperature,
        gas_pressure=gas_pressure,
        mass_density=state.mass_density,
        neutral_h_density=hydrogen.neutral_h_density,
        proton_density=hydrogen.proton_density,
        electron_density=state.electron_density,
        metadata=metadata,
        hydrogen_lte_state=hydrogen,
        helium_lte_state=state.helium_lte_state,
    )


def gray_hydrogen_atmosphere(
    effective_temperature: float,
    logg: float,
    *,
    n_depth: int = 120,
    tau_min: float = 1.0e-6,
    tau_max: float = 1.0e3,
    rosseland_opacity: float = 0.1,
    hopf_constant: float = 2.0 / 3.0,
    correlated_microfields: bool = False,
    include_molecules: bool = False,
    include_negative_hydrogen: bool = False,
    trihydrogen_ion_partition_model: str | None = None,
) -> Atmosphere:
    r"""Construct an LTE, hydrostatic Eddington-gray pure-H atmosphere.

    The structure obeys

    .. math:: T^4 = \frac{3}{4}T_\mathrm{eff}^4(\tau_R + q)

    and ``P_g = g tau_R / kappa_R``.  Constant Rosseland opacity is a
    controlled approximation for solver validation, not yet a realistic DA
    atmosphere.
    """

    if not np.isfinite(effective_temperature) or effective_temperature <= 0.0:
        raise ValueError("effective_temperature must be finite and positive")
    if not np.isfinite(logg):
        raise ValueError("logg must be finite")
    if n_depth < 3:
        raise ValueError("n_depth must be at least 3")
    if not 0.0 < tau_min < tau_max:
        raise ValueError("require 0 < tau_min < tau_max")
    if not np.isfinite(rosseland_opacity) or rosseland_opacity <= 0.0:
        raise ValueError("rosseland_opacity must be finite and positive")
    if not np.isfinite(hopf_constant) or hopf_constant <= 0.0:
        raise ValueError("hopf_constant must be finite and positive")

    tau = np.geomspace(tau_min, tau_max, n_depth, dtype=np.float64)
    temperature = effective_temperature * (
        0.75 * (tau + hopf_constant)
    ) ** 0.25
    column_mass = tau / rosseland_opacity
    gas_pressure = 10.0**logg * column_mass
    eos = hummer_mihalas_hydrogen_lte(
        temperature,
        gas_pressure,
        correlated_microfields=correlated_microfields,
        include_molecules=include_molecules,
        include_negative_hydrogen=include_negative_hydrogen,
        trihydrogen_ion_partition_model=trihydrogen_ion_partition_model,
    )

    return Atmosphere(
        effective_temperature=float(effective_temperature),
        logg=float(logg),
        rosseland_optical_depth=tau,
        column_mass=column_mass,
        temperature=temperature,
        gas_pressure=gas_pressure,
        mass_density=eos.mass_density,
        neutral_h_density=eos.neutral_h_density,
        proton_density=eos.proton_density,
        electron_density=eos.electron_density,
        metadata={
            "model": "eddington-gray",
            "composition": "pure-hydrogen",
            "eos": (
                "q-mhd-correlated-occupation-probability"
                if correlated_microfields
                else "hummer-mihalas-occupation-probability"
            ),
            "rosseland_opacity_cm2_g": float(rosseland_opacity),
            "hopf_constant": float(hopf_constant),
            "includes_molecular_equilibrium": bool(include_molecules),
        },
        hydrogen_lte_state=eos,
    )


def hydrogen_continuum_atmosphere(
    effective_temperature: float,
    logg: float,
    *,
    n_depth: int = 100,
    tau_min: float = 1.0e-6,
    tau_max: float = 1.0e2,
    hopf_constant: float = 2.0 / 3.0,
    correlated_microfields: bool = False,
    include_molecules: bool = False,
    include_negative_hydrogen: bool = False,
    trihydrogen_ion_partition_model: str | None = None,
    h2_h2_cia_table: H2H2CollisionInducedAbsorptionTable | None = None,
) -> Atmosphere:
    r"""Construct a gray-temperature atmosphere with a physical pressure scale.

    The Eddington temperature law is retained, but hydrostatic equilibrium is
    integrated using a local Rosseland mean computed from the implemented LTE
    hydrogen continuum.  This removes the arbitrary constant opacity from the
    density structure while stopping short of non-gray radiative equilibrium.
    """

    # Reuse the gray constructor for input validation and the temperature grid.
    seed = gray_hydrogen_atmosphere(
        effective_temperature,
        logg,
        n_depth=n_depth,
        tau_min=tau_min,
        tau_max=tau_max,
        rosseland_opacity=1.0,
        hopf_constant=hopf_constant,
        correlated_microfields=correlated_microfields,
        include_molecules=include_molecules,
        include_negative_hydrogen=include_negative_hydrogen,
        trihydrogen_ion_partition_model=trihydrogen_ion_partition_model,
    )
    from .opacity import rosseland_mean_hydrogen_continuum_opacity

    gravity = seed.gravity
    tau = seed.rosseland_optical_depth
    temperature = seed.temperature

    def opacity_at(temp: float, pressure: float) -> float:
        eos = hummer_mihalas_hydrogen_lte(
            np.array([temp]),
            np.array([pressure]),
            correlated_microfields=correlated_microfields,
            include_molecules=include_molecules,
            include_negative_hydrogen=include_negative_hydrogen,
            trihydrogen_ion_partition_model=trihydrogen_ion_partition_model,
        )
        point = Atmosphere(
            effective_temperature=float(effective_temperature),
            logg=float(logg),
            rosseland_optical_depth=np.array([1.0]),
            column_mass=np.array([pressure / gravity]),
            temperature=np.array([temp]),
            gas_pressure=np.array([pressure]),
            mass_density=np.atleast_1d(eos.mass_density),
            neutral_h_density=np.atleast_1d(eos.neutral_h_density),
            proton_density=np.atleast_1d(eos.proton_density),
            electron_density=np.atleast_1d(eos.electron_density),
            metadata={},
            hydrogen_lte_state=eos,
        )
        return float(
            rosseland_mean_hydrogen_continuum_opacity(
                point, h2_h2_cia_table=h2_h2_cia_table
            )[0]
        )

    def solve_log_increment(
        previous_pressure: float, delta_tau: float, midpoint_temperature: float
    ) -> float:
        target = gravity * delta_tau

        def residual(log_increment: float) -> float:
            increment = np.exp(log_increment)
            midpoint_pressure = previous_pressure + 0.5 * increment
            return log_increment + np.log(opacity_at(midpoint_temperature, midpoint_pressure)) - np.log(target)

        lower = np.log(max(target / 1.0e12, 1.0e-20))
        upper = np.log(max(target / 1.0e-12, 1.0e-19))
        return float(np.exp(_solve_bracketed_log_root(residual, lower, upper)))

    # The surface pressure is the integral across the first optical-depth bin.
    gas_pressure = np.empty_like(tau)
    gas_pressure[0] = solve_log_increment(0.0, float(tau[0]), float(temperature[0]))
    for index in range(1, tau.size):
        gas_pressure[index] = gas_pressure[index - 1] + solve_log_increment(
            float(gas_pressure[index - 1]),
            float(tau[index] - tau[index - 1]),
            float(0.5 * (temperature[index] + temperature[index - 1])),
        )

    column_mass = gas_pressure / gravity
    eos = hummer_mihalas_hydrogen_lte(
        temperature,
        gas_pressure,
        correlated_microfields=correlated_microfields,
        include_molecules=include_molecules,
        include_negative_hydrogen=include_negative_hydrogen,
        trihydrogen_ion_partition_model=trihydrogen_ion_partition_model,
    )
    provisional = Atmosphere(
        effective_temperature=float(effective_temperature),
        logg=float(logg),
        rosseland_optical_depth=tau,
        column_mass=column_mass,
        temperature=temperature,
        gas_pressure=gas_pressure,
        mass_density=eos.mass_density,
        neutral_h_density=eos.neutral_h_density,
        proton_density=eos.proton_density,
        electron_density=eos.electron_density,
        metadata={},
        hydrogen_lte_state=eos,
    )
    rosseland_opacity = rosseland_mean_hydrogen_continuum_opacity(
        provisional, h2_h2_cia_table=h2_h2_cia_table
    )
    return Atmosphere(
        effective_temperature=provisional.effective_temperature,
        logg=provisional.logg,
        rosseland_optical_depth=tau,
        column_mass=column_mass,
        temperature=temperature,
        gas_pressure=gas_pressure,
        mass_density=eos.mass_density,
        neutral_h_density=eos.neutral_h_density,
        proton_density=eos.proton_density,
        electron_density=eos.electron_density,
        metadata={
            "model": "eddington-gray-temperature/hydrogen-continuum-hydrostatic",
            "composition": "pure-hydrogen",
            "eos": (
                "q-mhd-correlated-occupation-probability"
                if correlated_microfields
                else "hummer-mihalas-occupation-probability"
            ),
            "rosseland_opacity_cm2_g": rosseland_opacity,
            "hopf_constant": float(hopf_constant),
            "includes_molecular_equilibrium": bool(include_molecules),
            "includes_h2_h2_collision_induced_absorption": bool(
                include_molecules and h2_h2_cia_table is not None
            ),
        },
        hydrogen_lte_state=eos,
    )


def gray_helium_atmosphere(
    effective_temperature: float,
    logg: float,
    *,
    n_depth: int = 120,
    tau_min: float = 1.0e-8,
    tau_max: float = 1.0e3,
    rosseland_opacity: float = 0.1,
    hopf_constant: float = 2.0 / 3.0,
    correlated_microfields: bool = True,
    neutral_radius_scale: float = 0.5,
    helium_reos3_table: HeliumREOS3Table | None = None,
) -> Atmosphere:
    """Construct an LTE Eddington-gray, hydrostatic pure-He atmosphere."""

    if not np.isfinite(effective_temperature) or effective_temperature <= 0.0:
        raise ValueError("effective_temperature must be finite and positive")
    if not np.isfinite(logg):
        raise ValueError("logg must be finite")
    if n_depth < 3:
        raise ValueError("n_depth must be at least 3")
    if not 0.0 < tau_min < tau_max:
        raise ValueError("require 0 < tau_min < tau_max")
    if not np.isfinite(rosseland_opacity) or rosseland_opacity <= 0.0:
        raise ValueError("rosseland_opacity must be finite and positive")
    if not np.isfinite(hopf_constant) or hopf_constant <= 0.0:
        raise ValueError("hopf_constant must be finite and positive")
    tau = np.geomspace(tau_min, tau_max, n_depth)
    temperature = effective_temperature * (0.75 * (tau + hopf_constant)) ** 0.25
    column_mass = tau / rosseland_opacity
    pressure = 10.0**logg * column_mass
    eos_function = (
        hummer_mihalas_helium_lte
        if helium_reos3_table is None
        else hummer_mihalas_helium_lte_with_reos3
    )
    eos = eos_function(
        temperature,
        pressure,
        *((helium_reos3_table,) if helium_reos3_table is not None else ()),
        neutral_radius_scale=neutral_radius_scale,
        correlated_microfields=correlated_microfields,
    )
    return Atmosphere(
        float(effective_temperature), float(logg), tau, column_mass,
        temperature, pressure, eos.mass_density,
        np.zeros(n_depth), np.zeros(n_depth), eos.electron_density,
        {
            "model": "eddington-gray",
            "composition": "pure-helium",
            "eos": "q-mhd-helium-occupation-probability" if correlated_microfields else "hummer-mihalas-helium-occupation-probability",
            "rosseland_opacity_cm2_g": float(rosseland_opacity),
            "hopf_constant": float(hopf_constant),
            "helium_neutral_radius_scale": float(neutral_radius_scale),
            "bulk_helium_eos": (
                "ideal chemical-picture pressure closure"
                if helium_reos3_table is None
                else helium_reos3_table.source
            ),
        },
        helium_lte_state=eos,
    )


def gray_hydrogen_helium_atmosphere(
    effective_temperature: float,
    logg: float,
    log_hydrogen_to_helium: float,
    *,
    n_depth: int = 120,
    tau_min: float = 1.0e-8,
    tau_max: float = 1.0e3,
    rosseland_opacity: float = 0.1,
    hopf_constant: float = 2.0 / 3.0,
    correlated_microfields: bool = True,
    hydrogen_neutral_radius_scale: float = 0.5,
    helium_neutral_radius_scale: float = 0.5,
) -> Atmosphere:
    """Construct a homogeneous atomic H/He Eddington-gray atmosphere."""

    if not np.isfinite(effective_temperature) or effective_temperature <= 0.0:
        raise ValueError("effective_temperature must be finite and positive")
    if not np.isfinite(logg) or not np.isfinite(log_hydrogen_to_helium):
        raise ValueError("logg and log_hydrogen_to_helium must be finite")
    if n_depth < 3:
        raise ValueError("n_depth must be at least 3")
    if not 0.0 < tau_min < tau_max:
        raise ValueError("require 0 < tau_min < tau_max")
    if not np.isfinite(rosseland_opacity) or rosseland_opacity <= 0.0:
        raise ValueError("rosseland_opacity must be finite and positive")
    if not np.isfinite(hopf_constant) or hopf_constant <= 0.0:
        raise ValueError("hopf_constant must be finite and positive")
    tau = np.geomspace(tau_min, tau_max, n_depth)
    temperature = effective_temperature * (
        0.75 * (tau + hopf_constant)
    ) ** 0.25
    column_mass = tau / rosseland_opacity
    pressure = 10.0**logg * column_mass
    state = hummer_mihalas_hydrogen_helium_lte(
        temperature,
        pressure,
        log_hydrogen_to_helium,
        hydrogen_neutral_radius_scale=hydrogen_neutral_radius_scale,
        helium_neutral_radius_scale=helium_neutral_radius_scale,
        correlated_microfields=correlated_microfields,
    )
    return _atmosphere_from_hydrogen_helium_state(
        effective_temperature,
        logg,
        tau,
        column_mass,
        temperature,
        pressure,
        state,
        {
            "model": "eddington-gray",
            "composition": "homogeneous-hydrogen-helium",
            "log_hydrogen_to_helium": float(log_hydrogen_to_helium),
            "eos": (
                "q-mhd-hydrogen-helium-occupation-probability"
                if correlated_microfields
                else "hummer-mihalas-hydrogen-helium-occupation-probability"
            ),
            "rosseland_opacity_cm2_g": float(rosseland_opacity),
            "hopf_constant": float(hopf_constant),
            "hydrogen_neutral_radius_scale": float(
                hydrogen_neutral_radius_scale
            ),
            "helium_neutral_radius_scale": float(helium_neutral_radius_scale),
            "mixed_molecular_chemistry": False,
        },
    )


def hydrogen_helium_continuum_atmosphere(
    effective_temperature: float,
    logg: float,
    log_hydrogen_to_helium: float,
    *,
    n_depth: int = 100,
    tau_min: float = 1.0e-8,
    tau_max: float = 1.0e2,
    hopf_constant: float = 2.0 / 3.0,
    correlated_microfields: bool = True,
    hydrogen_neutral_radius_scale: float = 0.5,
    helium_neutral_radius_scale: float = 0.5,
    rosseland_frequency_points: int = 160,
    include_helium_dimer_ion: bool = True,
    include_helium_three_body_cia: bool = True,
    include_rydberg_bound_free: bool = True,
) -> Atmosphere:
    """Gray-temperature mixed atmosphere with a physical pressure scale."""

    seed = gray_hydrogen_helium_atmosphere(
        effective_temperature,
        logg,
        log_hydrogen_to_helium,
        n_depth=n_depth,
        tau_min=tau_min,
        tau_max=tau_max,
        rosseland_opacity=1.0,
        hopf_constant=hopf_constant,
        correlated_microfields=correlated_microfields,
        hydrogen_neutral_radius_scale=hydrogen_neutral_radius_scale,
        helium_neutral_radius_scale=helium_neutral_radius_scale,
    )
    from .mixture import rosseland_mean_hydrogen_helium_continuum_opacity

    gravity = seed.gravity
    tau = seed.rosseland_optical_depth
    temperature = seed.temperature

    def point_at(temp: float, pressure: float) -> Atmosphere:
        state = hummer_mihalas_hydrogen_helium_lte(
            np.asarray([temp]),
            np.asarray([pressure]),
            log_hydrogen_to_helium,
            hydrogen_neutral_radius_scale=hydrogen_neutral_radius_scale,
            helium_neutral_radius_scale=helium_neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )
        return _atmosphere_from_hydrogen_helium_state(
            effective_temperature,
            logg,
            np.ones(1),
            np.asarray([pressure / gravity]),
            np.asarray([temp]),
            np.asarray([pressure]),
            state,
            {},
        )

    def opacity_at(temp: float, pressure: float) -> float:
        return float(
            rosseland_mean_hydrogen_helium_continuum_opacity(
                point_at(temp, pressure),
                n_frequency=rosseland_frequency_points,
                include_helium_dimer_ion=include_helium_dimer_ion,
                include_helium_three_body_cia=include_helium_three_body_cia,
                include_rydberg_bound_free=include_rydberg_bound_free,
            )[0]
        )

    def solve_increment(previous: float, delta_tau: float, temp: float) -> float:
        target = gravity * delta_tau

        def residual(log_increment: float) -> float:
            increment = np.exp(log_increment)
            opacity = opacity_at(temp, previous + 0.5 * increment)
            return log_increment + np.log(opacity) - np.log(target)

        return float(
            np.exp(
                _solve_bracketed_log_root(
                    residual,
                    np.log(max(target / 1.0e12, 1.0e-20)),
                    np.log(max(target / 1.0e-12, 1.0e-19)),
                )
            )
        )

    pressure = np.empty_like(tau)
    pressure[0] = solve_increment(0.0, float(tau[0]), float(temperature[0]))
    for depth in range(1, n_depth):
        pressure[depth] = pressure[depth - 1] + solve_increment(
            float(pressure[depth - 1]),
            float(tau[depth] - tau[depth - 1]),
            float(0.5 * (temperature[depth] + temperature[depth - 1])),
        )
    column_mass = pressure / gravity
    state = hummer_mihalas_hydrogen_helium_lte(
        temperature,
        pressure,
        log_hydrogen_to_helium,
        hydrogen_neutral_radius_scale=hydrogen_neutral_radius_scale,
        helium_neutral_radius_scale=helium_neutral_radius_scale,
        correlated_microfields=correlated_microfields,
    )
    provisional = _atmosphere_from_hydrogen_helium_state(
        effective_temperature,
        logg,
        tau,
        column_mass,
        temperature,
        pressure,
        state,
        {},
    )
    rosseland = rosseland_mean_hydrogen_helium_continuum_opacity(
        provisional,
        n_frequency=rosseland_frequency_points,
        include_helium_dimer_ion=include_helium_dimer_ion,
        include_helium_three_body_cia=include_helium_three_body_cia,
        include_rydberg_bound_free=include_rydberg_bound_free,
    )
    return _atmosphere_from_hydrogen_helium_state(
        effective_temperature,
        logg,
        tau,
        column_mass,
        temperature,
        pressure,
        state,
        {
            "model": "eddington-gray-temperature/hydrogen-helium-continuum-hydrostatic",
            "composition": "homogeneous-hydrogen-helium",
            "log_hydrogen_to_helium": float(log_hydrogen_to_helium),
            "eos": (
                "q-mhd-hydrogen-helium-occupation-probability"
                if correlated_microfields
                else "hummer-mihalas-hydrogen-helium-occupation-probability"
            ),
            "rosseland_opacity_cm2_g": rosseland,
            "hopf_constant": float(hopf_constant),
            "hydrogen_neutral_radius_scale": float(
                hydrogen_neutral_radius_scale
            ),
            "helium_neutral_radius_scale": float(helium_neutral_radius_scale),
            "helium_dimer_ion_continuum": bool(include_helium_dimer_ion),
            "helium_three_body_collision_induced_absorption": bool(
                include_helium_three_body_cia
            ),
            "helium_rydberg_bound_free": bool(include_rydberg_bound_free),
            "helium_minus_free_free": (
                "John-1994 inside tabulated T/lambda domain; "
                "Carbon-1969 fit to John-1968 elsewhere"
            ),
            "mixed_molecular_chemistry": False,
        },
    )


def helium_continuum_atmosphere(
    effective_temperature: float,
    logg: float,
    *,
    n_depth: int = 100,
    tau_min: float = 1.0e-8,
    tau_max: float = 1.0e2,
    hopf_constant: float = 2.0 / 3.0,
    correlated_microfields: bool = True,
    neutral_radius_scale: float = 0.5,
    rosseland_frequency_points: int = 160,
    include_helium_dimer_ion: bool = True,
    include_helium_three_body_cia: bool = True,
    include_rydberg_bound_free: bool = True,
    metal_database: AtomicDatabase | None = None,
    metal_abundances: Mapping[str, float] | None = None,
    log_hydrogen_abundance: float | None = None,
    include_dense_helium_metal_ionization: bool = True,
    helium_reos3_table: HeliumREOS3Table | None = None,
) -> Atmosphere:
    """Gray-temperature He atmosphere with a physical pressure scale.

    When metal data and abundances are supplied, metal electron donation and
    the resulting He-minus/free-free continuum are included during the
    hydrostatic integration.  Metal lines are intentionally omitted from this
    Rosseland-mean seed; their structural blanketing belongs in the subsequent
    non-gray radiative-equilibrium calculation.
    """

    if (metal_database is None) != (metal_abundances is None):
        raise ValueError("metal_database and metal_abundances must be supplied together")
    if log_hydrogen_abundance is not None and metal_database is None:
        return hydrogen_helium_continuum_atmosphere(
            effective_temperature,
            logg,
            log_hydrogen_abundance,
            n_depth=n_depth,
            tau_min=tau_min,
            tau_max=tau_max,
            hopf_constant=hopf_constant,
            correlated_microfields=correlated_microfields,
            helium_neutral_radius_scale=neutral_radius_scale,
            rosseland_frequency_points=rosseland_frequency_points,
            include_helium_dimer_ion=include_helium_dimer_ion,
            include_helium_three_body_cia=include_helium_three_body_cia,
            include_rydberg_bound_free=include_rydberg_bound_free,
        )

    seed = gray_helium_atmosphere(
        effective_temperature, logg, n_depth=n_depth, tau_min=tau_min,
        tau_max=tau_max, rosseland_opacity=1.0, hopf_constant=hopf_constant,
        correlated_microfields=correlated_microfields,
        neutral_radius_scale=neutral_radius_scale,
        helium_reos3_table=helium_reos3_table,
    )
    from .helium import rosseland_mean_helium_continuum_opacity

    gravity = seed.gravity
    tau = seed.rosseland_optical_depth
    temperature = seed.temperature

    def opacity_at(temp: float, pressure: float) -> float:
        eos = (
            hummer_mihalas_helium_lte(
                np.asarray([temp]), np.asarray([pressure]),
                neutral_radius_scale=neutral_radius_scale,
                correlated_microfields=correlated_microfields,
            )
            if helium_reos3_table is None
            else hummer_mihalas_helium_lte_with_reos3(
                np.asarray([temp]), np.asarray([pressure]), helium_reos3_table,
                neutral_radius_scale=neutral_radius_scale,
                correlated_microfields=correlated_microfields,
            )
        )
        point = Atmosphere(
            float(effective_temperature), float(logg), np.ones(1),
            np.asarray([pressure / gravity]), np.asarray([temp]),
            np.asarray([pressure]), eos.mass_density, np.zeros(1), np.zeros(1),
            eos.electron_density, {}, helium_lte_state=eos,
        )
        if metal_database is not None and metal_abundances is not None:
            from .metals import atmosphere_with_metal_electrons, metal_lte_state

            metal_state = metal_lte_state(
                point,
                metal_database,
                metal_abundances,
                reference_species="He",
                include_dense_helium_ionization=(
                    include_dense_helium_metal_ionization
                ),
                log_hydrogen_abundance=log_hydrogen_abundance,
            )
            point = atmosphere_with_metal_electrons(point, metal_state)
        return float(
            rosseland_mean_helium_continuum_opacity(
                point,
                n_frequency=rosseland_frequency_points,
                include_helium_dimer_ion=include_helium_dimer_ion,
                include_helium_three_body_cia=include_helium_three_body_cia,
                include_rydberg_bound_free=include_rydberg_bound_free,
            )[0]
        )

    def solve_increment(previous: float, delta_tau: float, temp: float) -> float:
        target = gravity * delta_tau

        def residual(log_increment: float) -> float:
            increment = np.exp(log_increment)
            opacity = opacity_at(temp, previous + 0.5 * increment)
            return log_increment + np.log(opacity) - np.log(target)

        return float(np.exp(_solve_bracketed_log_root(
            residual,
            np.log(max(target / 1.0e12, 1.0e-20)),
            np.log(max(target / 1.0e-12, 1.0e-19)),
        )))

    pressure = np.empty_like(tau)
    pressure[0] = solve_increment(0.0, float(tau[0]), float(temperature[0]))
    for depth in range(1, n_depth):
        pressure[depth] = pressure[depth - 1] + solve_increment(
            float(pressure[depth - 1]), float(tau[depth] - tau[depth - 1]),
            float(0.5 * (temperature[depth] + temperature[depth - 1])),
        )
    column_mass = pressure / gravity
    eos = (
        hummer_mihalas_helium_lte(
            temperature, pressure, neutral_radius_scale=neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )
        if helium_reos3_table is None
        else hummer_mihalas_helium_lte_with_reos3(
            temperature, pressure, helium_reos3_table,
            neutral_radius_scale=neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )
    )
    provisional = Atmosphere(
        float(effective_temperature), float(logg), tau, column_mass,
        temperature, pressure, eos.mass_density, np.zeros(n_depth),
        np.zeros(n_depth), eos.electron_density, {}, helium_lte_state=eos,
    )
    metal_state = None
    if metal_database is not None and metal_abundances is not None:
        from .metals import atmosphere_with_metal_electrons, metal_lte_state

        metal_state = metal_lte_state(
            provisional,
            metal_database,
            metal_abundances,
            reference_species="He",
            include_dense_helium_ionization=include_dense_helium_metal_ionization,
            log_hydrogen_abundance=log_hydrogen_abundance,
        )
        provisional = atmosphere_with_metal_electrons(provisional, metal_state)
    rosseland = rosseland_mean_helium_continuum_opacity(
        provisional,
        n_frequency=rosseland_frequency_points,
        include_helium_dimer_ion=include_helium_dimer_ion,
        include_helium_three_body_cia=include_helium_three_body_cia,
        include_rydberg_bound_free=include_rydberg_bound_free,
    )
    return Atmosphere(
        provisional.effective_temperature, provisional.logg, tau, column_mass,
        temperature, pressure, provisional.mass_density,
        provisional.neutral_h_density, provisional.proton_density,
        provisional.electron_density,
        {
            "model": "eddington-gray-temperature/helium-continuum-hydrostatic",
            "composition": (
                "metal-polluted-helium" if metal_state is not None else "pure-helium"
            ),
            "eos": "q-mhd-helium-occupation-probability" if correlated_microfields else "hummer-mihalas-helium-occupation-probability",
            "rosseland_opacity_cm2_g": rosseland,
            "hopf_constant": float(hopf_constant),
            "helium_neutral_radius_scale": float(neutral_radius_scale),
            "helium_dimer_ion_continuum": bool(include_helium_dimer_ion),
            "helium_three_body_collision_induced_absorption": bool(
                include_helium_three_body_cia
            ),
            "helium_rydberg_bound_free": bool(include_rydberg_bound_free),
            "bulk_helium_eos": (
                "ideal chemical-picture pressure closure"
                if helium_reos3_table is None
                else helium_reos3_table.source
            ),
            "helium_minus_free_free": (
                "John-1994 inside tabulated T/lambda domain; "
                "Carbon-1969 fit to John-1968 elsewhere"
            ),
            "metal_abundances": (
                dict(metal_state.log_number_abundance)
                if metal_state is not None else {}
            ),
            "log_hydrogen_abundance": log_hydrogen_abundance,
            "metal_electron_feedback": (
                "hydrostatic continuum opacity and charge neutrality"
                if metal_state is not None else "disabled"
            ),
        },
        hydrogen_lte_state=provisional.hydrogen_lte_state,
        helium_lte_state=provisional.helium_lte_state,
    )


def radiative_equilibrium_hydrogen_atmosphere(
    effective_temperature: float,
    logg: float,
    *,
    n_depth: int = 100,
    tau_min: float = 1.0e-10,
    max_iterations: int = 500,
    temperature_tolerance: float = 2.0e-4,
    flux_tolerance: float = 2.0e-3,
    n_continuum_wavelength: int = 600,
    include_balmer_lines: bool = True,
    include_paschen_lines: bool = True,
    include_brackett_lines: bool = True,
    include_balmer_self_broadening: bool = True,
    balmer_self_broadening_quadrature_order: int = 32,
    balmer_self_broadening_prescription: str = "barklem",
    balmer_self_broadening_truncation_closure: str = "renormalize",
    barklem_self_table: BarklemSelfBroadeningTable | None = None,
    include_lyman_lines: bool = True,
    resolve_balmer_line_cores: bool = True,
    balmer_line_core_step_angstrom: float = 0.10,
    resolve_lyman_line_cores: bool | None = None,
    include_series_pseudocontinuum: bool = False,
    correlated_microfields: bool = False,
    include_molecules: bool = False,
    include_negative_hydrogen: bool = False,
    trihydrogen_ion_partition_model: str | None = None,
    h2_h2_cia_table: H2H2CollisionInducedAbsorptionTable | None = None,
    include_neutral_lyman_alpha_wing: bool = True,
    unified_allard_table: AllardUnifiedLymanTable | None = None,
    jackson_lyman_table: JacksonLymanProfileTable | None = None,
    mixing_length_alpha: float | None = 0.7,
    convective_correction_damping: float = 1.0,
    convective_flux_tolerance: float = 2.0e-2,
    n_angle: int = 3,
    initial_temperature: FloatArray | None = None,
    initial_column_mass: FloatArray | None = None,
    initial_atmosphere: Atmosphere | None = None,
    maximum_radiative_correction_rosseland_depth: float = 20.0,
    apply_global_flux_correction: bool = True,
    radiative_correction_damping: float = 0.5,
    hydrogen_eos_function: Callable[
        [FloatArray, FloatArray], HydrogenLTEState
    ]
    | None = None,
    balmer_opacity_function: Callable[[Atmosphere, FloatArray], FloatArray]
    | None = None,
    continuum_opacity_function: Callable[[Atmosphere, FloatArray], FloatArray]
    | None = None,
    metal_database: AtomicDatabase | None = None,
    metal_abundances: Mapping[str, float] | None = None,
    metal_photoionization_database: VernerPhotoionizationDatabase | None = None,
    metal_topbase_photoionization_database: (
        TOPbasePhotoionizationDatabase | None
    ) = None,
    include_metal_lines: bool = True,
    minimum_metal_oscillator_strength: float = 1.0e-2,
    maximum_metal_lines: int | None = 1_000,
    iteration_callback: Callable[
        [int, Atmosphere, Mapping[str, float | int | bool]], None
    ]
    | None = None,
    structure_solver: Literal["lambda", "adaptive-newton"] = "lambda",
) -> Atmosphere:
    """Relax an H-dominated atmosphere toward non-gray energy equilibrium.

    Hydrostatic pressure remains exact on a fixed column-mass grid. With
    ``structure_solver="adaptive-newton"``, the unknowns are surface
    ``ln(T)`` and the layer logarithmic temperature gradients. A conservative
    Feautrier radiative flux and interface ML2 flux are solved together.  A
    local ML2-gradient phase first preconditions the nearly adiabatic interior;
    if necessary, the same-grid solve then restores the exact formal-flux
    residual in radiative layers.  The tangent Feautrier operator includes
    both source-function and opacity/optical-depth motion, with trust-region
    and backtracking globalization. With ``structure_solver="lambda"``, the older reference branch
    instead corrects node temperatures from the integral absorption balance
    ``integral kappa_abs (J-B) dlambda = 0`` plus a global flux correction.
    Both branches require the directly evaluated total flux to satisfy the
    requested tolerance. The default ``alpha=0.7`` matches the
    current Koester DA calibration; pass ``None`` for a strictly radiative
    control model.  The Lyman through Brackett series are included in the
    structural opacity by default.  The radiative-equilibrium
    grid extends to Rosseland optical depth 1e-10 by default because a grid
    beginning at 1e-6 can already be optically thick in the Lyman continuum
    of cool, high-gravity models.  Narrow Balmer-core points are included in
    the structure grid because omitting them produces an artificially warm
    upper atmosphere and broad, shallow optical cores.  Fine Lyman-core
    sampling is selected automatically at ``Teff >= 30000 K``.  In that
    regime it removes a warm bias from the Balmer-core-forming layers without
    changing the deeper line wings.  It remains off by default below 30000 K
    because the present strict-LTE solver can develop a neutral-opacity/
    cooling runaway in cool outer layers.  Pass an explicit boolean to
    override the automatic choice.

    When ``metal_database`` and ``metal_abundances`` are supplied, metal
    electron donation, bound--free opacity, and the selected structural metal
    lines are recomputed at every trial structure. The H chemical equilibrium
    is reclosed at the shared electron density. Metals remain trace in the
    pressure and thermodynamic derivatives used by ML2.

    If supplied, ``iteration_callback`` receives the updated atmosphere and a
    compact convergence record after every accepted correction. It can be
    used to write resumable checkpoints or inspect spectra during long cool-DA
    relaxations without changing the numerical iteration.  A caller may
    provide ``hydrogen_eos_function(T, P)`` to replace the ordinary HM/Q-MHD
    chemical-equilibrium solve while retaining the same hydrostatic and
    radiative-equilibrium machinery; the DAH module uses this controlled hook
    for its stationary-state magnetic EOS experiment.  Likewise,
    ``balmer_opacity_function(atmosphere, wavelength)`` can replace the
    zero-field structural Balmer opacity while leaving the remaining series
    and continuum controls unchanged.  ``continuum_opacity_function`` is the
    corresponding controlled hook for replacing the thermal continuum.  It
    must return the complete true-absorption continuum (not scattering), so a
    magnetic implementation can replace H I bound-free opacity without
    losing the otherwise validated free-free and molecular terms.
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if structure_solver not in ("lambda", "adaptive-newton"):
        raise ValueError(
            "structure_solver must be 'lambda' or 'adaptive-newton'"
        )
    if temperature_tolerance <= 0.0 or flux_tolerance <= 0.0:
        raise ValueError("convergence tolerances must be positive")
    if n_continuum_wavelength < 80:
        raise ValueError("n_continuum_wavelength must be at least 80")
    if n_angle < 1:
        raise ValueError("n_angle must be positive")
    if (metal_database is None) != (metal_abundances is None):
        raise ValueError(
            "metal_database and metal_abundances must be supplied together"
        )
    if (
        metal_photoionization_database is not None
        or metal_topbase_photoionization_database is not None
    ) and metal_database is None:
        raise ValueError(
            "metal photoionization requires metal_database and abundances"
        )
    if minimum_metal_oscillator_strength <= 0.0:
        raise ValueError("minimum_metal_oscillator_strength must be positive")
    if maximum_metal_lines is not None and maximum_metal_lines < 1:
        raise ValueError("maximum_metal_lines must be positive or None")
    if balmer_self_broadening_quadrature_order < 8:
        raise ValueError(
            "balmer_self_broadening_quadrature_order must be at least 8"
        )
    if (
        not np.isfinite(balmer_line_core_step_angstrom)
        or balmer_line_core_step_angstrom <= 0.0
    ):
        raise ValueError(
            "balmer_line_core_step_angstrom must be finite and positive"
        )
    if mixing_length_alpha is not None and (
        not np.isfinite(mixing_length_alpha) or mixing_length_alpha <= 0.0
    ):
        raise ValueError("mixing_length_alpha must be finite and positive")
    if (
        not np.isfinite(convective_correction_damping)
        or not 0.0 < convective_correction_damping <= 1.0
    ):
        raise ValueError("convective_correction_damping must be in (0, 1]")
    if (
        not np.isfinite(convective_flux_tolerance)
        or convective_flux_tolerance <= 0.0
    ):
        raise ValueError("convective_flux_tolerance must be finite and positive")
    if maximum_radiative_correction_rosseland_depth <= 0.0:
        raise ValueError(
            "maximum radiative-correction Rosseland depth must be positive"
        )
    if (
        not np.isfinite(radiative_correction_damping)
        or not 0.0 < radiative_correction_damping <= 1.0
    ):
        raise ValueError("radiative_correction_damping must be in (0, 1]")
    if initial_atmosphere is not None and (
        initial_temperature is not None or initial_column_mass is not None
    ):
        raise ValueError(
            "initial_atmosphere cannot be combined with initial temperature/grid"
        )
    resolved_lyman_line_cores = (
        effective_temperature >= 30_000.0
        if resolve_lyman_line_cores is None
        else bool(resolve_lyman_line_cores)
    )

    from .constants import (
        HYDROGEN_IONIZATION_ENERGY,
        LIGHT_SPEED,
        PLANCK,
        STEFAN_BOLTZMANN,
    )
    from .opacity import (
        BALMER_LINES,
        BRACKETT_LINES,
        LYMAN_LINES,
        PASCHEN_LINES,
        balmer_mass_absorption_coefficient,
        brackett_mass_absorption_coefficient,
        electron_scattering_mass_coefficient,
        hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient,
        hydrogen_continuum_mass_absorption_coefficient,
        hydrogen_rayleigh_scattering_mass_coefficient,
        lyman_alpha_neutral_hydrogen_wing_mass_absorption_coefficient,
        optical_depth_from_mass_opacity,
        paschen_mass_absorption_coefficient,
        lyman_mass_absorption_coefficient,
        rosseland_mean_hydrogen_continuum_opacity,
    )
    from .radiative_transfer import feautrier_radiation_field
    from .spectrum import planck_lambda_angstrom
    if mixing_length_alpha is not None:
        from .convection import (
            ml2_convective_flux_gradient_derivative_from_thermodynamics,
            ml2_convective_flux_for_gradient_from_thermodynamics,
            ml2_temperature_gradient_for_total_flux_from_thermodynamics,
        )

    if initial_atmosphere is None:
        seed = hydrogen_continuum_atmosphere(
            effective_temperature,
            logg,
            n_depth=n_depth,
            tau_min=tau_min,
            correlated_microfields=correlated_microfields,
            include_molecules=include_molecules,
            include_negative_hydrogen=include_negative_hydrogen,
            trihydrogen_ion_partition_model=trihydrogen_ion_partition_model,
            h2_h2_cia_table=h2_h2_cia_table,
        )
    else:
        if initial_atmosphere.hydrogen_lte_state is None:
            raise ValueError("initial_atmosphere must be a pure-H LTE structure")
        if not np.isclose(
            initial_atmosphere.effective_temperature,
            effective_temperature,
            rtol=0.0,
            atol=1.0e-8,
        ) or not np.isclose(
            initial_atmosphere.logg, logg, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                "initial_atmosphere must match effective temperature and log(g)"
            )
        seed = initial_atmosphere
    relaxation_depth = seed.rosseland_optical_depth.copy()
    wavelength = np.geomspace(
        100.0, 100_000.0, n_continuum_wavelength, dtype=np.float64
    )
    if include_balmer_lines:
        wavelength = np.unique(
            np.concatenate((wavelength, np.arange(3500.0, 7000.1, 10.0)))
        )
        # Line cores control the optically thin temperature through
        # radiative-equilibrium cooling.  A 10-A wing mesh alone entirely
        # misses their sub-Angstrom opacity peaks.
        if resolve_balmer_line_cores:
            wavelength = np.unique(
                np.concatenate(
                    (
                        wavelength,
                        *(
                            line.wavelength_vacuum_angstrom
                            + np.arange(
                                -5.0,
                                5.0 + 0.5 * balmer_line_core_step_angstrom,
                                balmer_line_core_step_angstrom,
                            )
                            for line in BALMER_LINES[:4]
                        ),
                    )
                )
            )
    infrared_line_offsets = np.asarray(
        (
            -300.0,
            -150.0,
            -75.0,
            -30.0,
            -10.0,
            -3.0,
            -1.0,
            0.0,
            1.0,
            3.0,
            10.0,
            30.0,
            75.0,
            150.0,
            300.0,
        ),
        dtype=np.float64,
    )
    if include_paschen_lines:
        wavelength = np.unique(
            np.concatenate(
                (
                    wavelength,
                    *(
                        line.wavelength_vacuum_angstrom
                        + infrared_line_offsets
                        for line in PASCHEN_LINES
                    ),
                )
            )
        )
    if include_brackett_lines:
        wavelength = np.unique(
            np.concatenate(
                (
                    wavelength,
                    *(
                        line.wavelength_vacuum_angstrom
                        + infrared_line_offsets
                        for line in BRACKETT_LINES
                    ),
                )
            )
        )
    if include_lyman_lines:
        wavelength = np.unique(
            np.concatenate(
                (
                    wavelength,
                    np.arange(900.0, 1250.1, 1.0),
                    np.arange(1255.0, 3000.1, 5.0),
                )
            )
        )
        if resolved_lyman_line_cores:
            wavelength = np.unique(
                np.concatenate(
                    (
                        wavelength,
                        *(
                            line.wavelength_vacuum_angstrom
                            + np.arange(-3.0, 3.0001, 0.05)
                            for line in LYMAN_LINES[:3]
                        ),
                    )
                )
            )
    metal_wing_sampled_lines = 0
    if metal_database is not None and metal_abundances is not None:
        from .metals import metal_lte_state, selected_metal_lines

        selection_state = metal_lte_state(
            seed,
            metal_database,
            metal_abundances,
            reference_species="H",
            include_dense_helium_ionization=False,
        )
        ion_stage_weight = {
            (element, charge): float(np.max(
                populations[charge]
                / np.maximum(
                    selection_state.element_number_density[element],
                    np.finfo(np.float64).tiny,
                )
            ))
            for element, populations in selection_state.ion_number_density.items()
            for charge in range(populations.shape[0])
        }
        structure_lines = selected_metal_lines(
            metal_database,
            metal_abundances,
            100.0,
            100_000.0,
            minimum_metal_oscillator_strength,
            maximum_metal_lines,
            effective_temperature,
            ion_stage_weight,
            flux_weighted=True,
        ) if include_metal_lines else []
        metal_centers = np.asarray(
            [line.wavelength_vacuum_angstrom for _, line in structure_lines]
        )
        if metal_centers.size:
            metal_grid, metal_wing_sampled_lines = (
                _metal_line_opacity_sampling_grid(metal_centers)
            )
            wavelength = np.unique(np.concatenate((
                wavelength,
                metal_grid,
            )))
    if metal_photoionization_database is not None:
        metal_edge_grid = np.asarray([
            PLANCK * LIGHT_SPEED
            / (fit.threshold_energy_ev * 1.602_176_634e-12)
            * 1.0e8
            for fit in metal_photoionization_database.fits.values()
            if metal_abundances is not None and fit.element in metal_abundances
        ])
        if metal_edge_grid.size:
            wavelength = np.unique(np.concatenate((
                wavelength,
                (
                    metal_edge_grid[:, np.newaxis]
                    * (1.0 + np.asarray((-1.0e-4, 1.0e-4)))
                ).ravel(),
            )))
    if metal_topbase_photoionization_database is not None:
        topbase_edge_grid = np.asarray([
            PLANCK * LIGHT_SPEED
            / (section.threshold_energy_ev * 1.602_176_634e-12)
            * 1.0e8
            for section in metal_topbase_photoionization_database.sections
            if metal_abundances is not None and section.element in metal_abundances
        ])
        if topbase_edge_grid.size:
            wavelength = np.unique(np.concatenate((
                wavelength,
                (
                    topbase_edge_grid[:, np.newaxis]
                    * (1.0 + np.asarray((-1.0e-4, 1.0e-4)))
                ).ravel(),
            )))
    target_flux = STEFAN_BOLTZMANN * effective_temperature**4
    if initial_temperature is None:
        temperature = seed.temperature.copy()
    else:
        temperature = np.asarray(initial_temperature, dtype=np.float64).copy()
        if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
            raise ValueError(
                "initial_temperature must contain finite positive values"
            )
        if initial_column_mass is not None:
            initial_mass = np.asarray(initial_column_mass, dtype=np.float64)
            if (
                initial_mass.shape != temperature.shape
                or np.any(~np.isfinite(initial_mass))
                or np.any(initial_mass <= 0.0)
                or np.any(np.diff(initial_mass) <= 0.0)
            ):
                raise ValueError(
                    "initial_column_mass must be positive, increasing, and match initial_temperature"
                )
            temperature = np.interp(
                np.log(seed.column_mass),
                np.log(initial_mass),
                temperature,
                left=temperature[0],
                right=temperature[-1],
            )
        elif temperature.shape != seed.temperature.shape:
            raise ValueError(
                "initial_temperature must have one value per depth when initial_column_mass is omitted"
            )
    flux_ratio = np.inf
    maximum_correction = np.inf
    maximum_radiative_correction = np.inf
    maximum_convective_correction = np.inf
    maximum_correction_depth_index = 0
    maximum_total_flux_residual = (
        np.inf if mixing_length_alpha is not None else 0.0
    )
    maximum_total_flux_residual_depth_index = -1
    maximum_total_flux_residual_rosseland_depth = np.nan
    signed_maximum_total_flux_residual = np.nan

    convective_gradient_operator: FloatArray | None = None
    if mixing_length_alpha is not None:
        log_pressure_grid = np.log(seed.gas_pressure)
        convective_gradient_operator = np.empty((seed.n_depth, seed.n_depth))
        for column in range(seed.n_depth):
            basis = np.zeros(seed.n_depth)
            basis[column] = 1.0
            convective_gradient_operator[:, column] = np.gradient(
                basis, log_pressure_grid, edge_order=2
            )
    convective_damping_backtrack = 1.0
    convective_flux_improvement_streak = 0
    previous_maximum_total_flux_residual = np.inf

    def with_temperature(values: FloatArray) -> Atmosphere:
        eos = (
            hummer_mihalas_hydrogen_lte(
                values,
                seed.gas_pressure,
                correlated_microfields=correlated_microfields,
                include_molecules=include_molecules,
                include_negative_hydrogen=include_negative_hydrogen,
                trihydrogen_ion_partition_model=(
                    trihydrogen_ion_partition_model
                ),
            )
            if hydrogen_eos_function is None
            else hydrogen_eos_function(values, seed.gas_pressure)
        )
        result = Atmosphere(
            effective_temperature=seed.effective_temperature,
            logg=seed.logg,
            rosseland_optical_depth=relaxation_depth,
            column_mass=seed.column_mass,
            temperature=values,
            gas_pressure=seed.gas_pressure,
            mass_density=eos.mass_density,
            neutral_h_density=eos.neutral_h_density,
            proton_density=eos.proton_density,
            electron_density=eos.electron_density,
            metadata=seed.metadata,
            hydrogen_lte_state=eos,
        )
        if metal_database is not None and metal_abundances is not None:
            from .metals import atmosphere_with_metal_electrons, metal_lte_state

            metal_state = metal_lte_state(
                result,
                metal_database,
                metal_abundances,
                reference_species="H",
                include_dense_helium_ionization=False,
            )
            result = atmosphere_with_metal_electrons(result, metal_state)
        return result

    hydrogen_structure_absorption_cache: dict[str, object] = {}

    def true_absorption(current_atmosphere: Atmosphere) -> FloatArray:
        """Evaluate every thermal opacity used by the structure solver."""

        absorption = (
            hydrogen_continuum_mass_absorption_coefficient(
                current_atmosphere,
                wavelength,
                include_electron_scattering=False,
                include_rayleigh_scattering=False,
                h2_h2_cia_table=h2_h2_cia_table,
            )
            if continuum_opacity_function is None
            else continuum_opacity_function(current_atmosphere, wavelength)
        )
        if absorption.shape != (wavelength.size, current_atmosphere.n_depth):
            raise ValueError(
                "continuum_opacity_function must return shape "
                "(wavelength, depth)"
            )
        if np.any(~np.isfinite(absorption)) or np.any(absorption < 0.0):
            raise ValueError(
                "continuum_opacity_function returned non-finite or negative opacity"
            )
        if include_balmer_lines:
            absorption += (
                balmer_mass_absorption_coefficient(
                    current_atmosphere,
                    wavelength,
                    include_self_broadening=include_balmer_self_broadening,
                    self_broadening_quadrature_order=(
                        balmer_self_broadening_quadrature_order
                    ),
                    self_broadening_prescription=(
                        balmer_self_broadening_prescription
                    ),
                    self_broadening_truncation_closure=(
                        balmer_self_broadening_truncation_closure
                    ),
                    barklem_self_table=barklem_self_table,
                    profile_edge_optical_depth=1.0e-4,
                )
                if balmer_opacity_function is None
                else balmer_opacity_function(current_atmosphere, wavelength)
            )
        if include_paschen_lines:
            absorption += paschen_mass_absorption_coefficient(
                current_atmosphere, wavelength
            )
        if include_brackett_lines:
            absorption += brackett_mass_absorption_coefficient(
                current_atmosphere, wavelength
            )
        if include_lyman_lines:
            absorption += lyman_mass_absorption_coefficient(
                current_atmosphere,
                wavelength,
                unified_allard_table=unified_allard_table,
                jackson_lyman_table=jackson_lyman_table,
            )
            if include_series_pseudocontinuum:
                absorption += (
                    hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
                        current_atmosphere, wavelength
                    )
                )
            if include_neutral_lyman_alpha_wing:
                absorption += (
                    lyman_alpha_neutral_hydrogen_wing_mass_absorption_coefficient(
                        current_atmosphere,
                        wavelength,
                        allard_table=(
                            unified_allard_table.lines.get((1, 2))
                            if (
                                unified_allard_table is not None
                                and not (
                                    jackson_lyman_table is not None
                                    and (1, 2) in jackson_lyman_table.lines
                                )
                            )
                            else None
                        ),
                    )
                )
        if metal_database is not None and metal_abundances is not None:
            from .metals import (
                metal_bound_free_mass_absorption_coefficient,
                metal_line_mass_absorption_coefficient,
                metal_lte_state,
            )

            metal_state = metal_lte_state(
                current_atmosphere,
                metal_database,
                metal_abundances,
                reference_species="H",
                include_dense_helium_ionization=False,
            )
            if metal_photoionization_database is not None:
                absorption += metal_bound_free_mass_absorption_coefficient(
                    current_atmosphere,
                    wavelength,
                    metal_database,
                    metal_state,
                    metal_photoionization_database,
                    excluded_ions=(
                        ()
                        if metal_topbase_photoionization_database is None
                        else metal_topbase_photoionization_database.ion_stages
                    ),
                )
            if metal_topbase_photoionization_database is not None:
                from .d6 import topbase_bound_free_mass_absorption_coefficient

                absorption += topbase_bound_free_mass_absorption_coefficient(
                    current_atmosphere,
                    wavelength,
                    metal_state,
                    metal_topbase_photoionization_database,
                )
            if include_metal_lines:
                absorption += metal_line_mass_absorption_coefficient(
                    current_atmosphere,
                    wavelength,
                    metal_database,
                    metal_state,
                    minimum_oscillator_strength=(
                        minimum_metal_oscillator_strength
                    ),
                    maximum_lines=maximum_metal_lines,
                )
        hydrogen_structure_absorption_cache.clear()
        hydrogen_structure_absorption_cache.update(
            atmosphere=current_atmosphere,
            absorption=absorption,
        )
        return absorption

    if structure_solver == "adaptive-newton":
        from .adaptive_structure import rosseland_mean_from_opacity_grid
        from .eos import hummer_mihalas_hydrogen_thermodynamics
        from .nonlinear import NonlinearEvaluation, solve_trust_region_newton
        from .radiative_transfer import (
            integrated_feautrier_interface_state_response,
        )

        log_pressure = np.log(seed.gas_pressure)
        interface_gradient_operator = np.zeros(
            (seed.n_depth, seed.n_depth), dtype=np.float64
        )
        interface_depth = np.arange(1, seed.n_depth)
        interface_pressure_step = np.diff(log_pressure)
        interface_gradient_operator[
            interface_depth, interface_depth - 1
        ] = -1.0 / interface_pressure_step
        interface_gradient_operator[
            interface_depth, interface_depth
        ] = 1.0 / interface_pressure_step

        def positive_interface_values(values):
            array = np.asarray(values, dtype=np.float64)
            interface = np.empty_like(array)
            interface[0] = array[0]
            interface[1:] = np.sqrt(array[:-1] * array[1:])
            return interface

        def arithmetic_interface_values(values):
            array = np.asarray(values, dtype=np.float64)
            interface = np.empty_like(array)
            interface[0] = array[0]
            interface[1:] = 0.5 * (array[:-1] + array[1:])
            return interface

        use_physical_radiative_residual = False
        solver_phase = "convective-gradient-preconditioner"
        solver_iteration_offset = 0

        def convection_transport(
            current_atmosphere: Atmosphere,
            temperature_gradient: FloatArray,
            radiative_flux_interface: FloatArray | None = None,
        ) -> dict[str, FloatArray] | None:
            """Evaluate the local EOS/ML2 closure on cell interfaces."""

            if mixing_length_alpha is None:
                return None
            explicit_metal_opacity = (
                metal_database is not None
                and (
                    include_metal_lines
                    or metal_photoionization_database is not None
                    or metal_topbase_photoionization_database is not None
                )
            )
            if explicit_metal_opacity:
                if (
                    hydrogen_structure_absorption_cache.get("atmosphere")
                    is current_atmosphere
                ):
                    absorption = np.asarray(
                        hydrogen_structure_absorption_cache["absorption"],
                        dtype=np.float64,
                    )
                else:
                    absorption = true_absorption(current_atmosphere)
                scattering = (
                    electron_scattering_mass_coefficient(
                        current_atmosphere
                    )[np.newaxis, :]
                    + hydrogen_rayleigh_scattering_mass_coefficient(
                        current_atmosphere, wavelength
                    )
                )
                rosseland = rosseland_mean_from_opacity_grid(
                    wavelength,
                    absorption + scattering,
                    current_atmosphere.temperature,
                )
            else:
                rosseland = rosseland_mean_hydrogen_continuum_opacity(
                    current_atmosphere,
                    h2_h2_cia_table=h2_h2_cia_table,
                )
            thermodynamics = hummer_mihalas_hydrogen_thermodynamics(
                current_atmosphere.temperature,
                current_atmosphere.gas_pressure,
                correlated_microfields=correlated_microfields,
                include_molecules=include_molecules,
                include_negative_hydrogen=include_negative_hydrogen,
                trihydrogen_ion_partition_model=(
                    trihydrogen_ion_partition_model
                ),
                central_state=current_atmosphere.hydrogen_lte_state,
            )
            interface_atmosphere = Atmosphere(
                effective_temperature=current_atmosphere.effective_temperature,
                logg=current_atmosphere.logg,
                rosseland_optical_depth=positive_interface_values(
                    current_atmosphere.rosseland_optical_depth
                ),
                column_mass=positive_interface_values(
                    current_atmosphere.column_mass
                ),
                temperature=positive_interface_values(
                    current_atmosphere.temperature
                ),
                gas_pressure=positive_interface_values(
                    current_atmosphere.gas_pressure
                ),
                mass_density=positive_interface_values(
                    current_atmosphere.mass_density
                ),
                neutral_h_density=positive_interface_values(
                    current_atmosphere.neutral_h_density
                ),
                proton_density=positive_interface_values(
                    current_atmosphere.proton_density
                ),
                electron_density=positive_interface_values(
                    current_atmosphere.electron_density
                ),
                metadata=current_atmosphere.metadata,
            )
            rosseland_interface = positive_interface_values(rosseland)
            heat_capacity_interface = arithmetic_interface_values(
                thermodynamics.specific_heat_constant_pressure
            )
            expansion_interface = arithmetic_interface_values(
                thermodynamics.density_temperature_derivative
            )
            adiabatic_gradient_interface = arithmetic_interface_values(
                thermodynamics.adiabatic_temperature_gradient
            )
            convective_flux_interface = (
                ml2_convective_flux_for_gradient_from_thermodynamics(
                    interface_atmosphere,
                    rosseland_interface,
                    temperature_gradient,
                    heat_capacity_interface,
                    expansion_interface,
                    adiabatic_gradient_interface,
                    mixing_length_alpha=mixing_length_alpha,
                )
            )
            convective_flux_interface[0] = 0.0
            diffusion_radiative_coefficient = (
                16.0
                * STEFAN_BOLTZMANN
                * interface_atmosphere.gravity
                * interface_atmosphere.temperature**4
                / (
                    3.0
                    * rosseland_interface
                    * interface_atmosphere.gas_pressure
                )
            )
            formal_radiative_coefficient = diffusion_radiative_coefficient
            if radiative_flux_interface is not None:
                formal_radiative_coefficient = np.where(
                    (temperature_gradient > 1.0e-8)
                    & (radiative_flux_interface > 0.0),
                    radiative_flux_interface
                    / np.maximum(temperature_gradient, 1.0e-8),
                    diffusion_radiative_coefficient,
                )
            desired_transport_gradient = (
                ml2_temperature_gradient_for_total_flux_from_thermodynamics(
                    interface_atmosphere,
                    rosseland_interface,
                    np.full_like(current_atmosphere.temperature, target_flux),
                    heat_capacity_interface,
                    expansion_interface,
                    adiabatic_gradient_interface,
                    mixing_length_alpha=mixing_length_alpha,
                    radiative_flux_coefficient=formal_radiative_coefficient,
                )
            )
            gradient_step = 2.0e-5
            desired_convective_flux = (
                ml2_convective_flux_for_gradient_from_thermodynamics(
                    interface_atmosphere,
                    rosseland_interface,
                    desired_transport_gradient,
                    heat_capacity_interface,
                    expansion_interface,
                    adiabatic_gradient_interface,
                    mixing_length_alpha=mixing_length_alpha,
                )
            )
            hotter_gradient_convective_flux = (
                ml2_convective_flux_for_gradient_from_thermodynamics(
                    interface_atmosphere,
                    rosseland_interface,
                    desired_transport_gradient + gradient_step,
                    heat_capacity_interface,
                    expansion_interface,
                    adiabatic_gradient_interface,
                    mixing_length_alpha=mixing_length_alpha,
                )
            )
            convective_flux_gradient_derivative = (
                hotter_gradient_convective_flux - desired_convective_flux
            ) / gradient_step
            actual_convective_flux_gradient_derivative = (
                ml2_convective_flux_gradient_derivative_from_thermodynamics(
                    interface_atmosphere,
                    rosseland_interface,
                    temperature_gradient,
                    heat_capacity_interface,
                    expansion_interface,
                    adiabatic_gradient_interface,
                    mixing_length_alpha=mixing_length_alpha,
                )
            )
            actual_convective_flux_gradient_derivative[0] = 0.0
            return {
                "rosseland": rosseland,
                "adiabatic_gradient": adiabatic_gradient_interface,
                "convective_flux": convective_flux_interface,
                "desired_gradient": desired_transport_gradient,
                "desired_convective_flux": desired_convective_flux,
                "formal_radiative_coefficient": (
                    formal_radiative_coefficient
                ),
                "convective_flux_gradient_derivative": (
                    convective_flux_gradient_derivative
                ),
                "actual_convective_flux_gradient_derivative": (
                    actual_convective_flux_gradient_derivative
                ),
            }

        def evaluate_newton(
            log_temperature: FloatArray, need_jacobian: bool
        ) -> NonlinearEvaluation[dict[str, object]]:
            current_temperature = np.exp(log_temperature)
            current_atmosphere = with_temperature(current_temperature)
            absorption = true_absorption(current_atmosphere)
            scattering = (
                electron_scattering_mass_coefficient(current_atmosphere)[
                    np.newaxis, :
                ]
                + hydrogen_rayleigh_scattering_mass_coefficient(
                    current_atmosphere, wavelength
                )
            )
            extinction = absorption + scattering
            optical_depth = optical_depth_from_mass_opacity(
                current_atmosphere.column_mass, extinction
            )
            planck = planck_lambda_angstrom(
                wavelength[:, np.newaxis],
                current_temperature[np.newaxis, :],
            )
            source = planck.copy()
            for _ in range(4):
                field = feautrier_radiation_field(
                    optical_depth, source, n_angle=n_angle
                )
                source = (
                    absorption * planck
                    + scattering * field.mean_intensity
                ) / extinction
            field = feautrier_radiation_field(
                optical_depth, source, n_angle=n_angle
            )
            if field.interface_flux is None:  # pragma: no cover
                raise RuntimeError(
                    "Feautrier solver did not return interface fluxes"
                )
            radiative_flux_interface = np.trapz(
                field.interface_flux, wavelength, axis=0
            )

            temperature_gradient = np.empty_like(log_temperature)
            temperature_gradient[0] = 0.0
            temperature_gradient[1:] = (
                np.diff(log_temperature) / np.diff(log_pressure)
            )
            convective_flux = np.zeros_like(current_temperature)
            convective_flux_interface = np.zeros_like(current_temperature)
            rosseland_for_convection = None
            transport = convection_transport(
                current_atmosphere,
                temperature_gradient,
                radiative_flux_interface,
            )
            if transport is not None:
                rosseland_for_convection = transport["rosseland"]
                convective_flux_interface = transport["convective_flux"]
                convective_flux = _upper_interface_values_on_nodes(
                    convective_flux_interface
                )

            total_flux_interface = (
                radiative_flux_interface + convective_flux_interface
            )
            residual = total_flux_interface / target_flux - 1.0
            transport_gradient_scale = None
            gradient_preconditioned = np.zeros(
                seed.n_depth, dtype=bool
            )
            if transport is not None and not use_physical_radiative_residual:
                desired_transport_gradient = transport["desired_gradient"]
                adiabatic_gradient_interface = transport[
                    "adiabatic_gradient"
                ]
                transport_gradient_scale = np.maximum(
                    desired_transport_gradient,
                    adiabatic_gradient_interface,
                )
                # Use the well-conditioned local gradient equation only in
                # layers where the locally balanced solution genuinely
                # transports flux by convection.  A diffusion-gradient
                # surrogate is not equivalent to the formal flux equation in
                # optically thin radiative layers.
                gradient_preconditioned = (
                    transport["desired_convective_flux"] > 0.0
                )
                gradient_preconditioned[0] = False
                residual[gradient_preconditioned] = (
                    temperature_gradient[gradient_preconditioned]
                    - desired_transport_gradient[gradient_preconditioned]
                ) / transport_gradient_scale[gradient_preconditioned]
            jacobian = None
            if need_jacobian:
                logarithmic_step = 2.0e-4
                hotter_planck = planck_lambda_angstrom(
                    wavelength[:, np.newaxis],
                    (
                        current_temperature
                        * np.exp(logarithmic_step)
                    )[np.newaxis, :],
                )
                planck_derivative = (
                    hotter_planck - planck
                ) / logarithmic_step
                hotter_atmosphere = with_temperature(
                    current_temperature * np.exp(logarithmic_step)
                )
                hotter_absorption = true_absorption(hotter_atmosphere)
                hotter_scattering = (
                    electron_scattering_mass_coefficient(hotter_atmosphere)[
                        np.newaxis, :
                    ]
                    + hydrogen_rayleigh_scattering_mass_coefficient(
                        hotter_atmosphere, wavelength
                    )
                )
                absorption_derivative = (
                    hotter_absorption - absorption
                ) / logarithmic_step
                scattering_derivative = (
                    hotter_scattering - scattering
                ) / logarithmic_step
                extinction_derivative = (
                    absorption_derivative + scattering_derivative
                )
                # Differentiate the explicit thermal/scattering source at
                # fixed J.  The tangent Feautrier solve below supplies the
                # non-local radiation response; coherent scattering is weak
                # in the cool DA regime where opacity motion matters most.
                source_derivative = (
                    absorption_derivative * planck
                    + absorption * planck_derivative
                    + scattering_derivative * field.mean_intensity
                    - extinction_derivative * source
                ) / extinction
                radiative_flux_jacobian = (
                    integrated_feautrier_interface_state_response(
                        optical_depth,
                        wavelength,
                        source,
                        source_derivative,
                        current_atmosphere.column_mass,
                        extinction_derivative,
                        n_angle=n_angle,
                    )
                )
                jacobian = radiative_flux_jacobian / target_flux
                if (
                    mixing_length_alpha is not None
                    and transport is not None
                    and use_physical_radiative_residual
                ):
                    # The completion residual is the actual radiative plus
                    # convective flux.  Its Newton matrix must include the
                    # local ML2 response at the current gradient; omitting it
                    # makes cool convective DA models reach the right surface
                    # flux while stalling with a deep transport defect.
                    jacobian += (
                        transport[
                            "actual_convective_flux_gradient_derivative"
                        ][:, np.newaxis]
                        * interface_gradient_operator
                        / target_flux
                    )
                if (
                    mixing_length_alpha is not None
                    and rosseland_for_convection is not None
                    and transport_gradient_scale is not None
                ):
                    formal_coefficient = transport[
                        "formal_radiative_coefficient"
                    ]
                    convective_gradient_derivative = transport[
                        "convective_flux_gradient_derivative"
                    ]
                    safe_gradient = np.maximum(
                        temperature_gradient, 1.0e-8
                    )
                    formal_coefficient_derivative = np.zeros_like(
                        radiative_flux_jacobian
                    )
                    formal_branch = (
                        (temperature_gradient > 1.0e-8)
                        & (radiative_flux_interface > 0.0)
                    )
                    formal_coefficient_derivative[formal_branch] = (
                        radiative_flux_jacobian[formal_branch]
                        * safe_gradient[formal_branch, np.newaxis]
                        - radiative_flux_interface[
                            formal_branch, np.newaxis
                        ]
                        * interface_gradient_operator[formal_branch]
                    ) / safe_gradient[formal_branch, np.newaxis] ** 2
                    desired_gradient_derivative = (
                        -desired_transport_gradient[:, np.newaxis]
                        * formal_coefficient_derivative
                        / (
                            formal_coefficient
                            + convective_gradient_derivative
                        )[:, np.newaxis]
                    )
                    jacobian[gradient_preconditioned] = (
                        interface_gradient_operator[gradient_preconditioned]
                        - desired_gradient_derivative[
                            gradient_preconditioned
                        ]
                    ) / transport_gradient_scale[
                        gradient_preconditioned, np.newaxis
                    ]

            payload: dict[str, object] = {
                "atmosphere": current_atmosphere,
                "radiative_flux_interface": radiative_flux_interface,
                "convective_flux": convective_flux,
                "convective_flux_interface": convective_flux_interface,
                "total_flux_interface": total_flux_interface,
                "convection_transport": transport,
                "temperature_gradient": temperature_gradient,
            }
            return NonlinearEvaluation(residual, jacobian, payload)

        log_pressure_step = np.diff(log_pressure)
        log_temperature_from_structure_state = np.zeros(
            (seed.n_depth, seed.n_depth), dtype=np.float64
        )
        log_temperature_from_structure_state[:, 0] = 1.0
        for depth in range(1, seed.n_depth):
            log_temperature_from_structure_state[depth:, depth] = (
                log_pressure_step[depth - 1]
            )

        def structure_state_from_log_temperature(
            log_temperature: FloatArray,
        ) -> FloatArray:
            state = np.empty_like(log_temperature)
            state[0] = log_temperature[0]
            state[1:] = (
                np.diff(log_temperature) / log_pressure_step
            )
            return state

        def evaluate_structure_state(
            structure_state: FloatArray, need_jacobian: bool
        ) -> NonlinearEvaluation[dict[str, object]]:
            log_temperature = (
                log_temperature_from_structure_state @ structure_state
            )
            evaluation = evaluate_newton(log_temperature, need_jacobian)
            jacobian = evaluation.jacobian
            if jacobian is not None:
                jacobian = (
                    jacobian @ log_temperature_from_structure_state
                )
            return NonlinearEvaluation(
                evaluation.residual, jacobian, evaluation.payload
            )

        def report_newton_iteration(record, state, evaluation):
            if iteration_callback is None:
                return
            payload = evaluation.payload
            current_atmosphere = payload["atmosphere"]
            total_flux_interface = np.asarray(
                payload["total_flux_interface"], dtype=np.float64
            )
            radiative_flux_interface = np.asarray(
                payload["radiative_flux_interface"], dtype=np.float64
            )
            convective_flux_interface = np.asarray(
                payload["convective_flux_interface"], dtype=np.float64
            )
            flux_residual = total_flux_interface / target_flux - 1.0
            maximum_flux_depth = int(np.argmax(np.abs(flux_residual)))
            transport = payload["convection_transport"]
            desired_gradient = (
                np.asarray(transport["desired_gradient"], dtype=np.float64)
                if transport is not None
                else np.full(seed.n_depth, np.nan)
            )
            temperature_gradient = np.asarray(
                payload["temperature_gradient"], dtype=np.float64
            )
            iteration_callback(
                solver_iteration_offset + record.iteration,
                current_atmosphere,
                {
                    "flux_ratio": float(total_flux_interface[0] / target_flux),
                    "maximum_log_temperature_correction": (
                        record.maximum_step
                    ),
                    "maximum_correction_depth_index": int(
                        maximum_flux_depth
                    ),
                    "maximum_total_flux_residual": float(
                        np.max(np.abs(flux_residual))
                    ),
                    "bottom_radiative_flux_ratio": float(
                        radiative_flux_interface[-1] / target_flux
                    ),
                    "bottom_convective_flux_ratio": float(
                        convective_flux_interface[-1] / target_flux
                    ),
                    "maximum_flux_depth_temperature_gradient": float(
                        temperature_gradient[maximum_flux_depth]
                    ),
                    "maximum_flux_depth_desired_gradient": float(
                        desired_gradient[maximum_flux_depth]
                    ),
                    "trust_radius": record.trust_radius,
                    "line_search_factor": record.line_search_factor,
                    "jacobian_recomputed": record.jacobian_recomputed,
                    "solver_phase": solver_phase,
                    "converged": bool(
                        record.residual_maximum < flux_tolerance
                        and record.maximum_step < temperature_tolerance
                        and np.max(np.abs(flux_residual)) < flux_tolerance
                    ),
                },
            )

        def physical_flux_converged(state, evaluation, maximum_step):
            total_flux_interface = np.asarray(
                evaluation.payload["total_flux_interface"], dtype=np.float64
            )
            return bool(
                np.max(
                    np.abs(total_flux_interface / target_flux - 1.0)
                )
                < flux_tolerance
            )

        def phase_converged(state, evaluation, maximum_step):
            if not use_physical_radiative_residual:
                return True
            return physical_flux_converged(
                state, evaluation, maximum_step
            )

        initial_log_temperature = np.log(temperature)
        if mixing_length_alpha is not None:
            # Construct one physically motivated alternative to a gray seed.
            # A gray or regridded atmosphere can sit just below the ML2
            # stability boundary even where the locally transported target
            # flux requires convection.  Project such cells onto the ML2
            # branch, but retain the projection only when a complete formal
            # transfer evaluation lowers the nonlinear residual.
            initial_atmosphere_for_projection = with_temperature(temperature)
            initial_rosseland = (
                rosseland_mean_hydrogen_continuum_opacity(
                    initial_atmosphere_for_projection,
                    h2_h2_cia_table=h2_h2_cia_table,
                )
            )
            initial_thermodynamics = (
                hummer_mihalas_hydrogen_thermodynamics(
                    temperature,
                    initial_atmosphere_for_projection.gas_pressure,
                    correlated_microfields=correlated_microfields,
                    include_molecules=include_molecules,
                    include_negative_hydrogen=include_negative_hydrogen,
                    trihydrogen_ion_partition_model=(
                        trihydrogen_ion_partition_model
                    ),
                    central_state=(
                        initial_atmosphere_for_projection.hydrogen_lte_state
                    ),
                )
            )
            if initial_temperature is None:
                # Preserve the validated gray-atmosphere initialization: only
                # reduce gradients that are already formally unstable.
                initial_gradient = np.gradient(
                    initial_log_temperature, log_pressure, edge_order=2
                )
                ml2_transport_gradient = (
                    ml2_temperature_gradient_for_total_flux_from_thermodynamics(
                        initial_atmosphere_for_projection,
                        initial_rosseland,
                        np.full_like(temperature, target_flux),
                        initial_thermodynamics.specific_heat_constant_pressure,
                        initial_thermodynamics.density_temperature_derivative,
                        initial_thermodynamics.adiabatic_temperature_gradient,
                        mixing_length_alpha=mixing_length_alpha,
                    )
                )
                unstable = (
                    initial_gradient
                    > initial_thermodynamics.adiabatic_temperature_gradient
                )
                projected_gradient = np.where(
                    unstable,
                    np.minimum(initial_gradient, ml2_transport_gradient),
                    initial_gradient,
                )
                projected_log_temperature = np.empty_like(
                    initial_log_temperature
                )
                projected_log_temperature[0] = initial_log_temperature[0]
                for depth in range(1, seed.n_depth):
                    projected_log_temperature[depth] = (
                        projected_log_temperature[depth - 1]
                        + 0.5
                        * (
                            projected_gradient[depth - 1]
                            + projected_gradient[depth]
                        )
                        * (log_pressure[depth] - log_pressure[depth - 1])
                    )
            else:
                # A regridded converged atmosphere may fall infinitesimally
                # below the ML2 boundary because its new pressure mesh changes
                # the EOS.  Use interface-consistent gradients and permit the
                # initialization projection to enter a locally required
                # convective branch.
                initial_interface_atmosphere = Atmosphere(
                    effective_temperature=(
                        initial_atmosphere_for_projection.effective_temperature
                    ),
                    logg=initial_atmosphere_for_projection.logg,
                    rosseland_optical_depth=positive_interface_values(
                        initial_atmosphere_for_projection.rosseland_optical_depth
                    ),
                    column_mass=positive_interface_values(
                        initial_atmosphere_for_projection.column_mass
                    ),
                    temperature=positive_interface_values(
                        initial_atmosphere_for_projection.temperature
                    ),
                    gas_pressure=positive_interface_values(
                        initial_atmosphere_for_projection.gas_pressure
                    ),
                    mass_density=positive_interface_values(
                        initial_atmosphere_for_projection.mass_density
                    ),
                    neutral_h_density=positive_interface_values(
                        initial_atmosphere_for_projection.neutral_h_density
                    ),
                    proton_density=positive_interface_values(
                        initial_atmosphere_for_projection.proton_density
                    ),
                    electron_density=positive_interface_values(
                        initial_atmosphere_for_projection.electron_density
                    ),
                    metadata=initial_atmosphere_for_projection.metadata,
                )
                initial_gradient = np.empty_like(initial_log_temperature)
                initial_gradient[0] = 0.0
                initial_gradient[1:] = (
                    np.diff(initial_log_temperature) / np.diff(log_pressure)
                )
                initial_adiabatic_gradient_interface = (
                    arithmetic_interface_values(
                        initial_thermodynamics.adiabatic_temperature_gradient
                    )
                )
                ml2_transport_gradient = (
                    ml2_temperature_gradient_for_total_flux_from_thermodynamics(
                        initial_interface_atmosphere,
                        positive_interface_values(initial_rosseland),
                        np.full_like(temperature, target_flux),
                        arithmetic_interface_values(
                            initial_thermodynamics.specific_heat_constant_pressure
                        ),
                        arithmetic_interface_values(
                            initial_thermodynamics.density_temperature_derivative
                        ),
                        initial_adiabatic_gradient_interface,
                        mixing_length_alpha=mixing_length_alpha,
                    )
                )
                unstable = (
                    initial_gradient > initial_adiabatic_gradient_interface
                )
                convection_required = (
                    ml2_transport_gradient
                    > initial_adiabatic_gradient_interface
                )
                projected_gradient = np.where(
                    unstable,
                    np.minimum(initial_gradient, ml2_transport_gradient),
                    np.where(
                        convection_required,
                        ml2_transport_gradient,
                        initial_gradient,
                    ),
                )
                projected_log_temperature = np.empty_like(
                    initial_log_temperature
                )
                projected_log_temperature[0] = initial_log_temperature[0]
                for depth in range(1, seed.n_depth):
                    projected_log_temperature[depth] = (
                        projected_log_temperature[depth - 1]
                        + projected_gradient[depth]
                        * (log_pressure[depth] - log_pressure[depth - 1])
                    )
            ordinary_seed_evaluation = evaluate_newton(
                initial_log_temperature, False
            )
            projected_seed_evaluation = evaluate_newton(
                projected_log_temperature, False
            )

            def seed_merit(evaluation):
                residual = evaluation.residual
                return float(
                    np.sqrt(np.mean(residual**2))
                    + 0.25 * np.max(np.abs(residual))
                )

            if seed_merit(projected_seed_evaluation) < seed_merit(
                ordinary_seed_evaluation
            ):
                initial_log_temperature = projected_log_temperature

        nonlinear_options = dict(
            maximum_iterations=max_iterations,
            residual_tolerance=flux_tolerance,
            step_tolerance=temperature_tolerance,
            initial_trust_radius=0.04,
            maximum_trust_radius=0.12,
            jacobian_refresh_interval=4,
            # A complete numerical recovery is affordable only on a tiny
            # diagnostic grid.  Production grids use the opacity-aware
            # Feautrier/ML2 block and its Broyden updates.
            finite_difference_fallback_step=(
                5.0e-3 if seed.n_depth <= 12 else None
            ),
            callback=report_newton_iteration,
            convergence_test=phase_converged,
            step_measure=lambda old_state, new_state: float(
                np.max(
                    np.abs(
                        log_temperature_from_structure_state
                        @ (new_state - old_state)
                    )
                )
            ),
        )
        result = solve_trust_region_newton(
            structure_state_from_log_temperature(initial_log_temperature),
            evaluate_structure_state,
            **nonlinear_options,
        )
        preconditioner_iterations = result.iterations
        formal_flux_is_converged = physical_flux_converged(
            result.state, result.evaluation, 0.0
        )
        if mixing_length_alpha is not None and not formal_flux_is_converged:
            # The local ML2-gradient equations rapidly establish the nearly
            # adiabatic interior but are not valid radiative-transfer
            # equations outside convection.  Continue on the same grid and
            # from the same state with exact formal-flux residuals in those
            # layers; no parameter or damping policy changes at this switch.
            use_physical_radiative_residual = True
            solver_phase = "formal-radiative-flux-completion"
            solver_iteration_offset = preconditioner_iterations
            result = solve_trust_region_newton(
                result.state,
                evaluate_structure_state,
                **nonlinear_options,
            )
        total_solver_iterations = solver_iteration_offset + result.iterations
        final_payload = result.evaluation.payload
        final_atmosphere = final_payload["atmosphere"]
        assert isinstance(final_atmosphere, Atmosphere)
        total_flux_interface = np.asarray(
            final_payload["total_flux_interface"], dtype=np.float64
        )
        convective_flux = np.asarray(
            final_payload["convective_flux"], dtype=np.float64
        )
        radiative_flux_interface = np.asarray(
            final_payload["radiative_flux_interface"], dtype=np.float64
        )
        convective_flux_interface = np.asarray(
            final_payload["convective_flux_interface"], dtype=np.float64
        )
        explicit_final_metal_opacity = (
            metal_database is not None
            and (
                include_metal_lines
                or metal_photoionization_database is not None
                or metal_topbase_photoionization_database is not None
            )
        )
        if explicit_final_metal_opacity:
            final_absorption = true_absorption(final_atmosphere)
            final_scattering = (
                electron_scattering_mass_coefficient(final_atmosphere)[
                    np.newaxis, :
                ]
                + hydrogen_rayleigh_scattering_mass_coefficient(
                    final_atmosphere, wavelength
                )
            )
            rosseland_opacity = rosseland_mean_from_opacity_grid(
                wavelength,
                final_absorption + final_scattering,
                final_atmosphere.temperature,
            )
        else:
            rosseland_opacity = rosseland_mean_hydrogen_continuum_opacity(
                final_atmosphere, h2_h2_cia_table=h2_h2_cia_table
            )
        rosseland_depth = np.empty_like(final_atmosphere.column_mass)
        rosseland_depth[0] = (
            rosseland_opacity[0] * final_atmosphere.column_mass[0]
        )
        rosseland_depth[1:] = rosseland_depth[0] + np.cumsum(
            0.5
            * (rosseland_opacity[1:] + rosseland_opacity[:-1])
            * np.diff(final_atmosphere.column_mass)
        )
        final_residual = total_flux_interface / target_flux - 1.0
        final_step = (
            result.history[-1].maximum_step if result.history else np.inf
        )
        return Atmosphere(
            effective_temperature=final_atmosphere.effective_temperature,
            logg=final_atmosphere.logg,
            rosseland_optical_depth=rosseland_depth,
            column_mass=final_atmosphere.column_mass,
            temperature=final_atmosphere.temperature,
            gas_pressure=final_atmosphere.gas_pressure,
            mass_density=final_atmosphere.mass_density,
            neutral_h_density=final_atmosphere.neutral_h_density,
            proton_density=final_atmosphere.proton_density,
            electron_density=final_atmosphere.electron_density,
            metadata={
                "model": (
                    "non-gray-radiative-convective-equilibrium"
                    if mixing_length_alpha is not None
                    else "non-gray-radiative-equilibrium"
                ),
                "composition": (
                    "metal-polluted-hydrogen"
                    if metal_database is not None
                    else "pure-hydrogen"
                ),
                "eos": (
                    final_atmosphere.hydrogen_lte_state.chemical_model
                    if final_atmosphere.hydrogen_lte_state is not None
                    else "hummer-mihalas-occupation-probability"
                ),
                "rosseland_opacity_cm2_g": rosseland_opacity,
                "structure_solver": "adaptive-trust-region-newton",
                "structure_residual": (
                    "ML2-gradient preconditioner plus conservative formal "
                    "interface total flux in radiative layers"
                ),
                "structure_jacobian": (
                    "opacity-aware tangent Feautrier plus implicit ML2 "
                    "gradient response"
                ),
                "radiative_equilibrium_iterations": total_solver_iterations,
                "convective_preconditioner_iterations": (
                    preconditioner_iterations
                ),
                "formal_flux_completion_used": bool(
                    solver_iteration_offset > 0
                ),
                "radiative_equilibrium_converged": bool(
                    result.converged
                    and np.max(np.abs(final_residual)) < flux_tolerance
                ),
                "radiative_equilibrium_maximum_log_temperature_correction": (
                    float(final_step)
                ),
                "radiative_equilibrium_flux_ratio": float(
                    total_flux_interface[0] / target_flux
                ),
                "radiative_equilibrium_wavelength_points": int(
                    wavelength.size
                ),
                "radiative_equilibrium_seed_tau_min": float(tau_min),
                "maximum_total_flux_residual": float(
                    np.max(np.abs(final_residual))
                ),
                "maximum_total_flux_residual_depth_index": int(
                    np.argmax(np.abs(final_residual))
                ),
                "maximum_convective_flux_fraction": float(
                    np.max(convective_flux) / target_flux
                ),
                "radiative_flux_fraction_by_interface": (
                    radiative_flux_interface / target_flux
                ),
                "convective_flux_fraction_by_interface": (
                    convective_flux_interface / target_flux
                ),
                "total_flux_fraction_by_interface": (
                    total_flux_interface / target_flux
                ),
                "convection": (
                    "ML2-Bergeron-1992"
                    if mixing_length_alpha is not None
                    else "none"
                ),
                "mixing_length_alpha": mixing_length_alpha,
                "radiative_equilibrium_includes_balmer_lines": bool(
                    include_balmer_lines
                ),
                "radiative_equilibrium_balmer_profile_edge_optical_depth": (
                    1.0e-4 if include_balmer_lines else None
                ),
                "radiative_equilibrium_includes_paschen_lines": bool(
                    include_paschen_lines
                ),
                "radiative_equilibrium_includes_brackett_lines": bool(
                    include_brackett_lines
                ),
                "radiative_equilibrium_includes_lyman_lines": bool(
                    include_lyman_lines
                ),
                "radiative_equilibrium_includes_series_pseudocontinuum": bool(
                    include_lyman_lines and include_series_pseudocontinuum
                ),
                "radiative_equilibrium_includes_molecular_equilibrium_and_opacity": bool(
                    include_molecules
                ),
                "radiative_equilibrium_metal_opacity": bool(
                    metal_database is not None
                ),
                "metal_abundances": (
                    dict(metal_abundances)
                    if metal_abundances is not None else {}
                ),
                "metal_electron_feedback": (
                    "fixed-H-nuclei shared H/metal charge closure"
                    if metal_database is not None else "disabled"
                ),
                "hydrogen_metal_eos_solver": (
                    "depth-local Newton in log(ne) and log(H partition)"
                    if metal_database is not None else "disabled"
                ),
                "metal_thermodynamic_derivatives": (
                    "trace-metal approximation: Q-MHD hydrogen derivatives"
                    if metal_database is not None else "not applicable"
                ),
                "rosseland_opacity_includes_metal_bound_bound_and_bound_free": (
                    explicit_final_metal_opacity
                ),
                "radiative_equilibrium_includes_metal_lines": bool(
                    metal_database is not None and include_metal_lines
                ),
                "radiative_equilibrium_includes_metal_bound_free": bool(
                    metal_photoionization_database is not None
                    or metal_topbase_photoionization_database is not None
                ),
                "radiative_equilibrium_level_resolved_metal_bound_free": (
                    metal_topbase_photoionization_database.source
                    if metal_topbase_photoionization_database is not None
                    else "disabled"
                ),
                "radiative_equilibrium_minimum_metal_oscillator_strength": float(
                    minimum_metal_oscillator_strength
                ),
                "radiative_equilibrium_maximum_metal_lines": maximum_metal_lines,
                "radiative_equilibrium_wing_sampled_metal_lines": int(
                    metal_wing_sampled_lines
                ),
                "radiative_equilibrium_metal_line_selection": (
                    "abundance-gf-boltzmann-ion-fraction-Planck-flux at Teff"
                ),
                "radiative_equilibrium_includes_negative_hydrogen_charge_equilibrium": bool(
                    include_negative_hydrogen
                ),
                "radiative_equilibrium_h3plus_partition_model": (
                    trihydrogen_ion_partition_model
                ),
                "radiative_equilibrium_h_h2_lyman_alpha_temperature_dependence": (
                    "Sahu-et-al-2025 inverse-temperature correction to "
                    "Rohrmann-et-al-2011 6000-K profile"
                    if include_molecules and include_lyman_lines
                    else None
                ),
                "electron_scattering_source": (
                    "coherent-isotropic Lambda iteration"
                ),
            },
            hydrogen_lte_state=final_atmosphere.hydrogen_lte_state,
        )

    for iteration in range(1, max_iterations + 1):
        atmosphere = with_temperature(temperature)
        absorption = true_absorption(atmosphere)
        scattering = (
            electron_scattering_mass_coefficient(atmosphere)[np.newaxis, :]
            + hydrogen_rayleigh_scattering_mass_coefficient(
                atmosphere, wavelength
            )
        )
        total_extinction = absorption + scattering
        optical_depth = optical_depth_from_mass_opacity(
            atmosphere.column_mass, total_extinction
        )
        planck = planck_lambda_angstrom(
            wavelength[:, np.newaxis], temperature[np.newaxis, :]
        )

        # Coherent isotropic Thomson scattering.  It is normally a small
        # correction for these models, so a few Lambda sweeps suffice.
        source = planck.copy()
        for _ in range(4):
            field = feautrier_radiation_field(
                optical_depth, source, n_angle=n_angle
            )
            source = (
                absorption * planck + scattering * field.mean_intensity
            ) / total_extinction
        field = feautrier_radiation_field(
            optical_depth, source, n_angle=n_angle
        )
        if field.interface_flux is None:  # pragma: no cover - API invariant
            raise RuntimeError("Feautrier solver did not return interface fluxes")
        radiative_flux_interface = np.trapz(
            field.interface_flux, wavelength, axis=0
        )
        radiative_flux = _upper_interface_values_on_nodes(
            radiative_flux_interface
        )

        emitted = np.trapz(absorption * planck, wavelength, axis=0)
        absorbed = np.trapz(
            absorption * field.mean_intensity, wavelength, axis=0
        )
        active = (
            relaxation_depth
            < maximum_radiative_correction_rosseland_depth
        )
        convective_flux = np.zeros_like(temperature)
        required_convective_flux = np.zeros_like(temperature)
        convective = np.zeros_like(temperature, dtype=bool)
        convective_weight = np.zeros_like(temperature)
        convective_log_temperature_correction = np.zeros_like(temperature)
        if mixing_length_alpha is not None:
            rosseland_for_convection = (
                rosseland_mean_hydrogen_continuum_opacity(
                    atmosphere, h2_h2_cia_table=h2_h2_cia_table
                )
            )
            required_convective_flux = np.clip(
                target_flux - radiative_flux, 0.0, target_flux
            )
            actual_gradient = np.gradient(
                np.log(temperature),
                np.log(atmosphere.gas_pressure),
                edge_order=2,
            )
            from .eos import hummer_mihalas_hydrogen_thermodynamics

            thermodynamics = hummer_mihalas_hydrogen_thermodynamics(
                temperature,
                atmosphere.gas_pressure,
                correlated_microfields=correlated_microfields,
                include_molecules=include_molecules,
                central_state=atmosphere.hydrogen_lte_state,
            )
            diffusion_radiative_flux_coefficient = (
                16.0
                * STEFAN_BOLTZMANN
                * atmosphere.gravity
                * atmosphere.temperature**4
                / (
                    3.0
                    * rosseland_for_convection
                    * atmosphere.gas_pressure
                )
            )
            # The formal transfer solution gives a better local radiative
            # response near the photosphere, while diffusion is the robust
            # asymptotic coefficient at small/large optical depth.  Use the
            # former only where its linearization is positive and resolved.
            transfer_radiative_flux_coefficient = np.where(
                (
                    (relaxation_depth >= 0.1)
                    & (relaxation_depth <= 30.0)
                    & (actual_gradient > 1.0e-8)
                    & (radiative_flux > 0.0)
                ),
                radiative_flux / np.maximum(actual_gradient, 1.0e-8),
                diffusion_radiative_flux_coefficient,
            )
            desired_gradient = (
                ml2_temperature_gradient_for_total_flux_from_thermodynamics(
                    atmosphere,
                    rosseland_for_convection,
                    np.full_like(temperature, target_flux),
                    thermodynamics.specific_heat_constant_pressure,
                    thermodynamics.density_temperature_derivative,
                    thermodynamics.adiabatic_temperature_gradient,
                    mixing_length_alpha=mixing_length_alpha,
                    radiative_flux_coefficient=(
                        transfer_radiative_flux_coefficient
                    ),
                )
            )
            desired_convective_flux = (
                ml2_convective_flux_for_gradient_from_thermodynamics(
                    atmosphere,
                    rosseland_for_convection,
                    desired_gradient,
                    thermodynamics.specific_heat_constant_pressure,
                    thermodynamics.density_temperature_derivative,
                    thermodynamics.adiabatic_temperature_gradient,
                    mixing_length_alpha=mixing_length_alpha,
                )
            )
            required_convective_flux = desired_convective_flux
            current_convective_flux = (
                ml2_convective_flux_for_gradient_from_thermodynamics(
                    atmosphere,
                    rosseland_for_convection,
                    actual_gradient,
                    thermodynamics.specific_heat_constant_pressure,
                    thermodynamics.density_temperature_derivative,
                    thermodynamics.adiabatic_temperature_gradient,
                    mixing_length_alpha=mixing_length_alpha,
                )
            )

            # The Schwarzschild boundary follows from the gradient radiation
            # alone would require.  Taper both across that boundary and at the
            # optically thin/deep limits so a layer cannot chatter between a
            # full ML2 update and no update on successive iterations.
            pure_radiative_gradient = (
                target_flux / diffusion_radiative_flux_coefficient
            )
            relative_superadiabaticity = (
                pure_radiative_gradient
                / np.maximum(
                    thermodynamics.adiabatic_temperature_gradient,
                    np.finfo(np.float64).tiny,
                )
                - 1.0
            )
            convective_weight = np.clip(
                np.log10(
                    np.maximum(relaxation_depth, np.finfo(np.float64).tiny)
                )
                + 1.0,
                0.0,
                1.0,
            )
            convective_weight *= np.clip(
                np.log10(
                    100.0 / np.maximum(relaxation_depth, 1.0e-300)
                ),
                0.0,
                1.0,
            )
            convective_weight *= np.clip(
                relative_superadiabaticity / 0.05, 0.0, 1.0
            )
            convective = convective_weight > 0.0

            convective_flux_interface = _node_values_on_upper_interfaces(
                current_convective_flux, surface_value=0.0
            )
            convective_weight_interface = np.zeros_like(convective_weight)
            convective_weight_interface[1:] = np.minimum(
                convective_weight[:-1], convective_weight[1:]
            )
            relaxation_depth_interface = _node_values_on_upper_interfaces(
                relaxation_depth, surface_value=0.0
            )
            flux_balance_monitor = (
                (convective_weight_interface > 0.5)
                & (relaxation_depth_interface > 1.0e-3)
                & (relaxation_depth_interface < 10.0)
            )
            if np.any(flux_balance_monitor):
                monitored_depth = np.flatnonzero(flux_balance_monitor)
                monitored_residual = (
                    (
                        radiative_flux_interface[monitored_depth]
                        + convective_flux_interface[monitored_depth]
                    )
                    / target_flux
                    - 1.0
                )
                local_maximum = int(np.argmax(np.abs(monitored_residual)))
                maximum_total_flux_residual_depth_index = int(
                    monitored_depth[local_maximum]
                )
                signed_maximum_total_flux_residual = float(
                    monitored_residual[local_maximum]
                )
                maximum_total_flux_residual = abs(
                    signed_maximum_total_flux_residual
                )
                maximum_total_flux_residual_rosseland_depth = float(
                    relaxation_depth_interface[
                        maximum_total_flux_residual_depth_index
                    ]
                )
            else:
                maximum_total_flux_residual = 0.0
                maximum_total_flux_residual_depth_index = -1
                maximum_total_flux_residual_rosseland_depth = np.nan
                signed_maximum_total_flux_residual = np.nan
            if (
                signed_maximum_total_flux_residual < -0.2
                and maximum_total_flux_residual_rosseland_depth >= 1.0
            ):
                # A deep flux deficit is the well-conditioned diffusion
                # problem the coupled root solves most directly.  It can use
                # a full trial step; the measured-residual backtracking below
                # still reduces that ceiling immediately if opacity changes
                # invalidate the frozen local prediction.
                adaptive_convective_damping = 1.0
            elif maximum_total_flux_residual < 0.05:
                adaptive_convective_damping = 0.05
            elif maximum_total_flux_residual < 0.2:
                adaptive_convective_damping = 0.1
            elif maximum_total_flux_residual < 1.0:
                adaptive_convective_damping = 0.25
            elif maximum_total_flux_residual < 10.0:
                adaptive_convective_damping = 0.5
            else:
                adaptive_convective_damping = 1.0
            if np.isfinite(previous_maximum_total_flux_residual):
                if (
                    maximum_total_flux_residual
                    > 1.01 * previous_maximum_total_flux_residual
                ):
                    convective_damping_backtrack = max(
                        0.05,
                        0.5
                        * min(
                            convective_correction_damping,
                            adaptive_convective_damping,
                            convective_damping_backtrack,
                        ),
                    )
                    convective_flux_improvement_streak = 0
                elif (
                    maximum_total_flux_residual
                    < previous_maximum_total_flux_residual
                ):
                    convective_flux_improvement_streak += 1
                    if convective_flux_improvement_streak >= 3:
                        convective_damping_backtrack = min(
                            1.0, 1.25 * convective_damping_backtrack
                        )
                        convective_flux_improvement_streak = 0
                else:
                    convective_flux_improvement_streak = 0
            previous_maximum_total_flux_residual = (
                maximum_total_flux_residual
            )
            maximum_convective_correction_damping = min(
                convective_correction_damping,
                adaptive_convective_damping,
                convective_damping_backtrack,
            )

            gradient_correction = convective_weight * (
                desired_gradient - actual_gradient
            )
            if np.any(convective):
                anchor = max(int(np.flatnonzero(convective)[0]) - 1, 0)
                # Integrate the requested change in dln(T)/dln(P) inward from
                # the radiative surface.  This local construction does not
                # couple separated depth cells, unlike the former dense
                # least-squares inverse of np.gradient, and is the same
                # conservative update used by the D6 atmosphere solver.
                log_pressure = np.log(atmosphere.gas_pressure)
                convective_log_temperature_correction.fill(0.0)
                for depth in range(anchor + 1, atmosphere.n_depth):
                    cell_change = 0.5 * (
                        gradient_correction[depth - 1]
                        + gradient_correction[depth]
                    )
                    convective_log_temperature_correction[depth] = (
                        convective_log_temperature_correction[depth - 1]
                        + cell_change
                        * (log_pressure[depth] - log_pressure[depth - 1])
                    )
            maximum_raw_convective_correction = float(
                np.max(np.abs(convective_log_temperature_correction))
            )
            if maximum_raw_convective_correction > 0.04:
                convective_log_temperature_correction *= (
                    0.04 / maximum_raw_convective_correction
                )
        hotter_temperature = 1.001 * temperature
        hotter_planck = planck_lambda_angstrom(
            wavelength[:, np.newaxis],
            hotter_temperature[np.newaxis, :],
        )
        logarithmic_derivative = np.trapz(
            absorption * (hotter_planck - planck), wavelength, axis=0
        ) / 0.001
        correction = (absorbed - emitted) / np.maximum(
            logarithmic_derivative, np.finfo(np.float64).tiny
        )
        correction = np.where(active, np.clip(correction, -0.04, 0.04), 0.0)
        smoothed = _smooth_radiative_temperature_correction(
            correction, convective_weight
        )
        surface_flux = radiative_flux_interface[0]
        flux_ratio = float(surface_flux / target_flux)
        global_correction = float(
            np.clip(-0.25 * np.log(flux_ratio), -0.04, 0.04)
        )
        radiative_temperature_correction = (
            radiative_correction_damping * smoothed
        )
        effective_convective_correction_damping = (
            convective_correction_damping
        )
        if (
            mixing_length_alpha is not None
            and np.any(convective)
            and np.any(flux_balance_monitor)
        ):
            # Frozen-opacity line search for the ML2 update.  Unlike the old
            # convective-deficit iteration, this predicts *both* parts of the
            # transported flux: the formal radiative flux is linearized in
            # the local gradient and the nonlinear ML2 flux is reevaluated
            # exactly for every candidate.  This makes a large correction
            # back off before it can cross the coupled root.
            assert convective_gradient_operator is not None
            trial_damping = np.unique(
                np.maximum(
                    maximum_convective_correction_damping
                    * np.asarray((1.0, 0.5, 0.25, 0.125, 0.0625)),
                    0.01,
                )
            )
            radiative_gradient_step = (
                convective_gradient_operator
                @ radiative_temperature_correction
            )
            convective_gradient_step = (
                convective_gradient_operator
                @ convective_log_temperature_correction
            )
            trial_gradient = (
                actual_gradient[np.newaxis, :]
                + radiative_gradient_step[np.newaxis, :]
                + trial_damping[:, np.newaxis]
                * convective_gradient_step[np.newaxis, :]
            )
            trial_radiative_flux = (
                radiative_flux[np.newaxis, :]
                + transfer_radiative_flux_coefficient[np.newaxis, :]
                * (
                    trial_gradient
                    - actual_gradient[np.newaxis, :]
                )
            )
            trial_convective_flux = (
                ml2_convective_flux_for_gradient_from_thermodynamics(
                    atmosphere,
                    rosseland_for_convection,
                    trial_gradient,
                    thermodynamics.specific_heat_constant_pressure,
                    thermodynamics.density_temperature_derivative,
                    thermodynamics.adiabatic_temperature_gradient,
                    mixing_length_alpha=mixing_length_alpha,
                )
            )
            trial_radiative_flux_interface = np.empty_like(
                trial_radiative_flux
            )
            trial_radiative_flux_interface[:, 0] = (
                radiative_flux_interface[0]
            )
            trial_radiative_flux_interface[:, 1:] = 0.5 * (
                trial_radiative_flux[:, :-1]
                + trial_radiative_flux[:, 1:]
            )
            trial_convective_flux_interface = np.empty_like(
                trial_convective_flux
            )
            trial_convective_flux_interface[:, 0] = 0.0
            trial_convective_flux_interface[:, 1:] = 0.5 * (
                trial_convective_flux[:, :-1]
                + trial_convective_flux[:, 1:]
            )
            trial_total_flux_residual = np.max(
                np.abs(
                    (
                        trial_radiative_flux_interface[
                            :, flux_balance_monitor
                        ]
                        + trial_convective_flux_interface[
                            :, flux_balance_monitor
                        ]
                    )
                    / target_flux
                    - 1.0
                ),
                axis=1,
            )
            effective_convective_correction_damping = float(
                trial_damping[np.argmin(trial_total_flux_residual)]
            )
        convective_temperature_correction = (
            effective_convective_correction_damping
            * convective_log_temperature_correction
        )
        global_relaxation = 1.0 if np.any(convective) else 0.25
        total_temperature_correction = (
            radiative_temperature_correction
            + (
                global_relaxation * global_correction
                if apply_global_flux_correction
                else 0.0
            )
            + convective_temperature_correction
        )
        temperature *= np.exp(total_temperature_correction)
        correction_monitor = active | convective
        # Test convergence against the correction that was actually applied.
        # The previous radiative-only branch monitored the undamped raw
        # residual instead, even though the solver applies half of its
        # smoothed value.  That could leave a stable hot model marked
        # unconverged indefinitely while every temperature update was already
        # below the requested tolerance.
        maximum_correction = float(
            np.max(
                np.abs(total_temperature_correction[correction_monitor])
            )
        )
        maximum_radiative_correction = float(
            np.max(np.abs(radiative_temperature_correction[active]))
        )
        maximum_convective_correction = float(
            np.max(
                np.abs(convective_temperature_correction[correction_monitor])
            )
        )
        maximum_correction_depth_index = int(
            np.argmax(
                np.where(
                    correction_monitor,
                    np.abs(total_temperature_correction),
                    -1.0,
                )
            )
        )
        converged = bool(
            iteration >= 12
            and maximum_correction < temperature_tolerance
            and abs(flux_ratio - 1.0) < flux_tolerance
            and (
                mixing_length_alpha is None
                or maximum_total_flux_residual < convective_flux_tolerance
            )
        )
        if iteration_callback is not None:
            iteration_callback(
                iteration,
                with_temperature(temperature.copy()),
                {
                    "flux_ratio": flux_ratio,
                    "maximum_log_temperature_correction": maximum_correction,
                    "maximum_radiative_log_temperature_correction": (
                        maximum_radiative_correction
                    ),
                    "maximum_convective_log_temperature_correction": (
                        maximum_convective_correction
                    ),
                    "maximum_correction_depth_index": (
                        maximum_correction_depth_index
                    ),
                    "maximum_convective_flux_fraction": float(
                        np.max(required_convective_flux[convective]) / target_flux
                        if np.any(convective)
                        else 0.0
                    ),
                    "maximum_total_flux_residual": (
                        maximum_total_flux_residual
                    ),
                    "signed_maximum_total_flux_residual": (
                        signed_maximum_total_flux_residual
                    ),
                    "maximum_total_flux_residual_depth_index": (
                        maximum_total_flux_residual_depth_index
                    ),
                    "maximum_total_flux_residual_rosseland_depth": (
                        maximum_total_flux_residual_rosseland_depth
                    ),
                    "effective_convective_correction_damping": (
                        effective_convective_correction_damping
                    ),
                    "converged": converged,
                },
            )
        if converged:
            break

    atmosphere = with_temperature(temperature)
    rosseland_opacity = rosseland_mean_hydrogen_continuum_opacity(
        atmosphere, h2_h2_cia_table=h2_h2_cia_table
    )
    rosseland_depth = np.empty_like(atmosphere.column_mass)
    rosseland_depth[0] = rosseland_opacity[0] * atmosphere.column_mass[0]
    rosseland_depth[1:] = rosseland_depth[0] + np.cumsum(
        0.5
        * (rosseland_opacity[1:] + rosseland_opacity[:-1])
        * np.diff(atmosphere.column_mass)
    )
    return Atmosphere(
        effective_temperature=atmosphere.effective_temperature,
        logg=atmosphere.logg,
        rosseland_optical_depth=rosseland_depth,
        column_mass=atmosphere.column_mass,
        temperature=atmosphere.temperature,
        gas_pressure=atmosphere.gas_pressure,
        mass_density=atmosphere.mass_density,
        neutral_h_density=atmosphere.neutral_h_density,
        proton_density=atmosphere.proton_density,
        electron_density=atmosphere.electron_density,
        metadata={
            "model": (
                "non-gray-radiative-convective-equilibrium"
                if mixing_length_alpha is not None
                else "non-gray-radiative-equilibrium"
            ),
            "composition": (
                "metal-polluted-hydrogen"
                if metal_database is not None
                else "pure-hydrogen"
            ),
            "eos": (
                atmosphere.hydrogen_lte_state.chemical_model
                if atmosphere.hydrogen_lte_state is not None
                else (
                    "q-mhd-correlated-occupation-probability"
                    if correlated_microfields
                    else "hummer-mihalas-occupation-probability"
                )
            ),
            "rosseland_opacity_cm2_g": rosseland_opacity,
            "radiative_equilibrium_iterations": iteration,
            "radiative_equilibrium_converged": bool(
                maximum_correction < temperature_tolerance
                and abs(flux_ratio - 1.0) < flux_tolerance
                and (
                    mixing_length_alpha is None
                    or maximum_total_flux_residual
                    < convective_flux_tolerance
                )
            ),
            "radiative_equilibrium_maximum_log_temperature_correction": maximum_correction,
            "radiative_equilibrium_maximum_radiative_log_temperature_correction": maximum_radiative_correction,
            "radiative_equilibrium_maximum_convective_log_temperature_correction": maximum_convective_correction,
            "radiative_equilibrium_maximum_correction_depth_index": maximum_correction_depth_index,
            "radiative_equilibrium_flux_ratio": flux_ratio,
            "radiative_equilibrium_wavelength_points": int(wavelength.size),
            "radiative_equilibrium_seed_tau_min": float(tau_min),
            "radiative_equilibrium_maximum_correction_rosseland_depth": float(
                maximum_radiative_correction_rosseland_depth
            ),
            "radiative_equilibrium_applies_global_flux_correction": bool(
                apply_global_flux_correction
            ),
            "radiative_equilibrium_radiative_correction_damping": float(
                radiative_correction_damping
            ),
            "radiative_equilibrium_includes_balmer_lines": bool(include_balmer_lines),
            "radiative_equilibrium_custom_balmer_opacity": bool(
                balmer_opacity_function is not None
            ),
            "radiative_equilibrium_includes_paschen_lines": bool(
                include_paschen_lines
            ),
            "radiative_equilibrium_includes_brackett_lines": bool(
                include_brackett_lines
            ),
            "radiative_equilibrium_includes_balmer_self_broadening": bool(
                include_balmer_lines and include_balmer_self_broadening
            ),
            "radiative_equilibrium_balmer_self_broadening_quadrature_order": int(
                balmer_self_broadening_quadrature_order
            ),
            "radiative_equilibrium_balmer_profile_edge_optical_depth": (
                1.0e-4 if include_balmer_lines else None
            ),
            "radiative_equilibrium_balmer_self_broadening_prescription": (
                balmer_self_broadening_prescription
            ),
            "radiative_equilibrium_balmer_self_broadening_truncation_closure": (
                balmer_self_broadening_truncation_closure
            ),
            "radiative_equilibrium_includes_lyman_lines": bool(include_lyman_lines),
            "radiative_equilibrium_resolves_balmer_line_cores": bool(
                include_balmer_lines and resolve_balmer_line_cores
            ),
            "radiative_equilibrium_balmer_line_core_step_angstrom": float(
                balmer_line_core_step_angstrom
            ),
            "radiative_equilibrium_resolves_lyman_line_cores": bool(
                include_lyman_lines and resolved_lyman_line_cores
            ),
            "radiative_equilibrium_includes_series_pseudocontinuum": bool(
                include_lyman_lines and include_series_pseudocontinuum
            ),
            "radiative_equilibrium_includes_neutral_lyman_alpha_wing": bool(
                include_lyman_lines
                and include_neutral_lyman_alpha_wing
            ),
            "radiative_equilibrium_includes_allard_lyman_profiles": bool(
                include_lyman_lines and unified_allard_table is not None
            ),
            "allard_profile_policy": "effective-nearest; half-Stark additive",
            "radiative_equilibrium_includes_jackson_lyman_profiles": bool(
                include_lyman_lines and jackson_lyman_table is not None
            ),
            "jackson_profile_policy": (
                "replace Lyalpha/Lybeta charged Stark; depth-interpolate "
                "log(T), log(ne); Doppler-convolved"
            ),
            "radiative_equilibrium_includes_molecular_equilibrium_and_opacity": bool(
                include_molecules
            ),
            "radiative_equilibrium_metal_opacity": bool(
                metal_database is not None
            ),
            "metal_abundances": (
                dict(metal_abundances)
                if metal_abundances is not None else {}
            ),
            "metal_electron_feedback": (
                "fixed-H-nuclei shared H/metal charge closure"
                if metal_database is not None else "disabled"
            ),
            "hydrogen_metal_eos_solver": (
                "depth-local Newton in log(ne) and log(H partition)"
                if metal_database is not None else "disabled"
            ),
            "metal_thermodynamic_derivatives": (
                "trace-metal approximation: Q-MHD hydrogen derivatives"
                if metal_database is not None else "not applicable"
            ),
            "radiative_equilibrium_includes_metal_lines": bool(
                metal_database is not None and include_metal_lines
            ),
            "radiative_equilibrium_includes_metal_bound_free": bool(
                metal_photoionization_database is not None
                or metal_topbase_photoionization_database is not None
            ),
            "radiative_equilibrium_level_resolved_metal_bound_free": (
                metal_topbase_photoionization_database.source
                if metal_topbase_photoionization_database is not None
                else "disabled"
            ),
            "radiative_equilibrium_minimum_metal_oscillator_strength": float(
                minimum_metal_oscillator_strength
            ),
            "radiative_equilibrium_maximum_metal_lines": maximum_metal_lines,
            "radiative_equilibrium_wing_sampled_metal_lines": int(
                metal_wing_sampled_lines
            ),
            "radiative_equilibrium_includes_negative_hydrogen_charge_equilibrium": bool(
                include_negative_hydrogen
            ),
            "radiative_equilibrium_h3plus_partition_model": (
                trihydrogen_ion_partition_model
            ),
            "radiative_equilibrium_h_h2_lyman_alpha_temperature_dependence": (
                "Sahu-et-al-2025 inverse-temperature correction to "
                "Rohrmann-et-al-2011 6000-K profile"
                if include_molecules and include_lyman_lines
                else None
            ),
            "radiative_equilibrium_includes_h2_h2_collision_induced_absorption": bool(
                include_molecules and h2_h2_cia_table is not None
            ),
            "radiative_equilibrium_custom_hydrogen_eos": bool(
                hydrogen_eos_function is not None
            ),
            "convection": (
                "ML2-Bergeron-1992" if mixing_length_alpha is not None else "none"
            ),
            "mixing_length_alpha": mixing_length_alpha,
            "convective_correction_damping": float(
                convective_correction_damping
            ),
            "effective_convective_correction_damping": float(
                effective_convective_correction_damping
            ),
            "convective_transport_solver": (
                "coupled-radiative-plus-ML2-total-flux"
                if mixing_length_alpha is not None
                else "none"
            ),
            "convective_flux_tolerance": float(convective_flux_tolerance),
            "maximum_total_flux_residual": float(
                maximum_total_flux_residual
            ),
            "signed_maximum_total_flux_residual": float(
                signed_maximum_total_flux_residual
            ),
            "maximum_total_flux_residual_depth_index": int(
                maximum_total_flux_residual_depth_index
            ),
            "maximum_total_flux_residual_rosseland_depth": float(
                maximum_total_flux_residual_rosseland_depth
            ),
            "maximum_convective_flux_fraction": float(
                np.max(required_convective_flux[convective]) / target_flux
                if np.any(convective)
                else 0.0
            ),
            "convective_depth_points": int(np.count_nonzero(convective)),
            "electron_scattering_source": "coherent-isotropic Lambda iteration",
            "hydrogen_rayleigh_scattering": (
                "H I Rohrmann-Vera Rueda 2022 plus H2 Dalgarno-Williams "
                "1962 redward of Lyalpha"
            ),
        },
        hydrogen_lte_state=atmosphere.hydrogen_lte_state,
    )


def radiative_equilibrium_helium_atmosphere(
    effective_temperature: float,
    logg: float,
    *,
    stark_table: object,
    helium_ii_stark_table: object | None = None,
    n_depth: int = 80,
    tau_min: float = 1.0e-8,
    max_iterations: int = 300,
    structure_solver: Literal["adaptive-newton", "lambda"] = "lambda",
    temperature_tolerance: float = 3.0e-4,
    flux_tolerance: float = 3.0e-3,
    consecutive_convergence_iterations: int = 3,
    n_continuum_wavelength: int = 500,
    include_lines: bool = True,
    include_helium_ii_lines: bool = True,
    correlated_microfields: bool = True,
    neutral_radius_scale: float = 0.5,
    mixing_length_alpha: float | None = 1.25,
    include_absorption_temperature_derivative: bool = True,
    temperature_correction_damping: float = 1.0,
    convective_correction_damping: float = 1.0,
    convective_gradient_tolerance: float = 0.03,
    convective_flux_tolerance: float = 0.03,
    n_angle: int = 3,
    include_helium_dimer_ion: bool = True,
    include_helium_three_body_cia: bool = True,
    include_rydberg_bound_free: bool = True,
    neutral_line_broadening: Literal["none", "unsold", "montreal"] = "unsold",
    initial_temperature: FloatArray | None = None,
    initial_column_mass: FloatArray | None = None,
    initial_gas_pressure: FloatArray | None = None,
    initial_rosseland_optical_depth: FloatArray | None = None,
    metal_database: AtomicDatabase | None = None,
    metal_abundances: Mapping[str, float] | None = None,
    log_hydrogen_abundance: float | None = None,
    include_trace_hydrogen_lines: bool = True,
    include_hydrogen_self_broadening: bool = True,
    include_hydrogen_neutral_helium_broadening: bool = True,
    hydrogen_self_broadening_quadrature_order: int = 32,
    hydrogen_self_broadening_impact_validity_fraction: float = 1.0,
    hydrogen_self_broadening_prescription: str = "barklem",
    hydrogen_self_broadening_truncation_closure: str = "renormalize",
    include_hydrogen_series_pseudocontinuum: bool = False,
    unified_allard_table: AllardUnifiedLymanTable | None = None,
    allard_stark_weight: float = 0.5,
    include_dense_helium_metal_ionization: bool = True,
    include_metal_lines: bool = True,
    minimum_metal_oscillator_strength: float = 1.0e-2,
    maximum_metal_lines: int | None = 1_000,
    mg_he_red_wing_table: MgHeRedWingTable | None = None,
    mg_ii_he_profile_table: MgIIHeProfileTable | None = None,
    ca_i_he_profile_table: CaIHeProfileTable | None = None,
    ca_ii_he_profile_table: CaIIHeProfileTable | None = None,
    helium_reos3_table: HeliumREOS3Table | None = None,
    metal_photoionization_database: VernerPhotoionizationDatabase | None = None,
    metal_topbase_photoionization_database: (
        TOPbasePhotoionizationDatabase | None
    ) = None,
    resume_supplied_structure_in_formal_flux_phase: bool = True,
    iteration_callback: Callable[
        [int, Atmosphere, Mapping[str, object]], None
    ]
    | None = None,
) -> Atmosphere:
    """Relax an LTE He atmosphere toward non-gray radiative equilibrium.

    This is the helium counterpart of the transparent DA reference solver.
    Hydrostatic equilibrium is exact on a fixed column-mass grid.  The
    released line-dissolved Beauchamp profiles should normally be supplied as
    ``stark_table``.  The default ML2 mixing length of 1.25 is the standard
    Montreal/ATMO DB calibration; pass ``None`` for a radiative control model.
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if not isinstance(resume_supplied_structure_in_formal_flux_phase, bool):
        raise TypeError(
            "resume_supplied_structure_in_formal_flux_phase must be boolean"
        )
    if structure_solver not in ("adaptive-newton", "lambda"):
        raise ValueError(
            "structure_solver must be 'adaptive-newton' or 'lambda'"
        )
    if consecutive_convergence_iterations < 1:
        raise ValueError("consecutive_convergence_iterations must be positive")
    if n_continuum_wavelength < 80:
        raise ValueError("n_continuum_wavelength must be at least 80")
    if mixing_length_alpha is not None and (
        not np.isfinite(mixing_length_alpha) or mixing_length_alpha <= 0.0
    ):
        raise ValueError("mixing_length_alpha must be finite and positive")
    if (
        not np.isfinite(temperature_correction_damping)
        or temperature_correction_damping <= 0.0
        or temperature_correction_damping > 1.0
    ):
        raise ValueError(
            "temperature_correction_damping must be in the interval (0, 1]"
        )
    if (
        not np.isfinite(convective_correction_damping)
        or not 0.0 < convective_correction_damping <= 1.0
    ):
        raise ValueError("convective_correction_damping must be in (0, 1]")
    if (
        not np.isfinite(convective_gradient_tolerance)
        or convective_gradient_tolerance <= 0.0
    ):
        raise ValueError("convective_gradient_tolerance must be positive")
    if (
        not np.isfinite(convective_flux_tolerance)
        or convective_flux_tolerance <= 0.0
    ):
        raise ValueError("convective_flux_tolerance must be positive")
    if neutral_line_broadening not in ("none", "unsold", "montreal"):
        raise ValueError(
            "neutral_line_broadening must be 'none', 'unsold', or 'montreal'"
        )
    if hydrogen_self_broadening_quadrature_order < 8:
        raise ValueError(
            "hydrogen_self_broadening_quadrature_order must be at least 8"
        )
    if (
        not np.isfinite(allard_stark_weight)
        or not 0.0 <= allard_stark_weight <= 1.0
    ):
        raise ValueError("allard_stark_weight must lie in [0, 1]")
    if (metal_database is None) != (metal_abundances is None):
        raise ValueError("metal_database and metal_abundances must be supplied together")
    if (
        metal_photoionization_database is not None
        or metal_topbase_photoionization_database is not None
    ) and metal_database is None:
        raise ValueError("metal photoionization requires metal_database and abundances")
    homogeneous_mixture = (
        log_hydrogen_abundance is not None and metal_database is None
    )
    from .constants import (
        BOLTZMANN,
        ELECTRON_MASS,
        ELEMENTARY_CHARGE_ESU,
        HELIUM_MASS,
        HYDROGEN_IONIZATION_ENERGY,
        LIGHT_SPEED,
        PI,
        PLANCK,
        STEFAN_BOLTZMANN,
    )
    from .helium import (
        HELIUM_I_LINES,
        HELIUM_I_RESONANCE_LINES,
        HELIUM_II_LINES,
        _helium_ii_level_distribution,
        helium_continuum_mass_absorption_coefficient,
        helium_i_line_mass_absorption_coefficient,
        helium_i_resonance_line_mass_absorption_coefficient,
        helium_ii_line_mass_absorption_coefficient,
        helium_rayleigh_scattering_mass_coefficient,
        rosseland_mean_helium_continuum_opacity,
    )
    if homogeneous_mixture:
        from .mixture import rosseland_mean_hydrogen_helium_continuum_opacity
    from .opacity import (
        BALMER_LINES,
        BRACKETT_LINES,
        LYMAN_LINES,
        PASCHEN_LINES,
        balmer_mass_absorption_coefficient,
        brackett_mass_absorption_coefficient,
        electron_scattering_mass_coefficient,
        hydrogen_continuum_mass_absorption_coefficient,
        hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient,
        hydrogen_rayleigh_scattering_mass_coefficient,
        lyman_mass_absorption_coefficient,
        optical_depth_from_mass_opacity,
        paschen_mass_absorption_coefficient,
    )
    from .radiative_transfer import radiation_field
    from .spectrum import planck_lambda_angstrom
    if mixing_length_alpha is not None:
        from .convection import (
            ml2_convective_flux,
            ml2_convective_flux_for_gradient,
            ml2_temperature_gradient_for_flux,
        )

    restart_seed_requested = (
        initial_gas_pressure is not None
        or initial_rosseland_optical_depth is not None
    )
    if restart_seed_requested:
        if any(
            value is None for value in (
                initial_temperature,
                initial_column_mass,
                initial_gas_pressure,
                initial_rosseland_optical_depth,
            )
        ):
            raise ValueError(
                "a checkpoint restart requires temperature, column mass, "
                "gas pressure, and Rosseland optical depth"
            )
        restart_temperature = np.asarray(initial_temperature, dtype=np.float64)
        restart_mass = np.asarray(initial_column_mass, dtype=np.float64)
        restart_pressure = np.asarray(initial_gas_pressure, dtype=np.float64)
        restart_tau = np.asarray(
            initial_rosseland_optical_depth, dtype=np.float64
        )
        restart_shape = restart_temperature.shape
        if (
            restart_temperature.ndim != 1
            or restart_temperature.size < 3
            or any(
                value.shape != restart_shape
                for value in (
                    restart_mass, restart_pressure, restart_tau
                )
            )
        ):
            raise ValueError(
                "checkpoint arrays must be one dimensional and have the "
                "same length"
            )
        if (
            np.any(~np.isfinite(restart_temperature))
            or np.any(restart_temperature <= 0.0)
            or np.any(~np.isfinite(restart_mass))
            or np.any(restart_mass <= 0.0)
            or np.any(np.diff(restart_mass) <= 0.0)
            or np.any(~np.isfinite(restart_pressure))
            or np.any(restart_pressure <= 0.0)
            or np.any(~np.isfinite(restart_tau))
            or np.any(restart_tau <= 0.0)
            or np.any(np.diff(restart_tau) <= 0.0)
        ):
            raise ValueError("checkpoint arrays must be finite, positive, and increasing")
        restart_source_depth = int(restart_temperature.size)
        if restart_source_depth != n_depth:
            # A converged production checkpoint is often a better warm start
            # for a standard-resolution calculation than a lower-resolution
            # atmosphere with the same nominal parameters.  Resample every
            # hydrostatic coordinate together on log Rosseland depth; do not
            # silently discard pressure or column-mass provenance by using
            # only the supplied temperature array.
            source_log_tau = np.log(restart_tau)
            target_log_tau = np.linspace(
                source_log_tau[0], source_log_tau[-1], n_depth
            )
            restart_temperature = np.exp(np.interp(
                target_log_tau,
                source_log_tau,
                np.log(restart_temperature),
            ))
            restart_mass = np.exp(np.interp(
                target_log_tau,
                source_log_tau,
                np.log(restart_mass),
            ))
            restart_pressure = np.exp(np.interp(
                target_log_tau,
                source_log_tau,
                np.log(restart_pressure),
            ))
            restart_tau = np.exp(target_log_tau)
        restart_metadata = {
            "model": "checkpoint-restart-seed",
            "composition": (
                "homogeneous-hydrogen-helium"
                if homogeneous_mixture else "helium"
            ),
            "log_hydrogen_to_helium": (
                float(log_hydrogen_abundance)
                if homogeneous_mixture else None
            ),
            "checkpoint_skips_hydrostatic_seed": True,
            "checkpoint_source_depth_points": restart_source_depth,
            "checkpoint_resampled_to_depth_points": int(n_depth),
            "bulk_helium_eos": (
                "ideal chemical-picture pressure closure"
                if helium_reos3_table is None
                else helium_reos3_table.source
            ),
        }
        if homogeneous_mixture:
            assert log_hydrogen_abundance is not None
            mixed_restart_eos = hummer_mihalas_hydrogen_helium_lte(
                restart_temperature,
                restart_pressure,
                log_hydrogen_abundance,
                helium_neutral_radius_scale=neutral_radius_scale,
                correlated_microfields=correlated_microfields,
            )
            seed = _atmosphere_from_hydrogen_helium_state(
                effective_temperature,
                logg,
                restart_tau.copy(),
                restart_mass.copy(),
                restart_temperature.copy(),
                restart_pressure.copy(),
                mixed_restart_eos,
                restart_metadata,
            )
        else:
            restart_eos = (
                hummer_mihalas_helium_lte(
                    restart_temperature,
                    restart_pressure,
                    neutral_radius_scale=neutral_radius_scale,
                    correlated_microfields=correlated_microfields,
                )
                if helium_reos3_table is None
                else hummer_mihalas_helium_lte_with_reos3(
                    restart_temperature,
                    restart_pressure,
                    helium_reos3_table,
                    neutral_radius_scale=neutral_radius_scale,
                    correlated_microfields=correlated_microfields,
                )
            )
            seed = Atmosphere(
                float(effective_temperature),
                float(logg),
                restart_tau.copy(),
                restart_mass.copy(),
                restart_temperature.copy(),
                restart_pressure.copy(),
                restart_eos.mass_density,
                np.zeros(n_depth),
                np.zeros(n_depth),
                restart_eos.electron_density,
                restart_metadata,
                helium_lte_state=restart_eos,
            )
    else:
        seed = helium_continuum_atmosphere(
            effective_temperature, logg, n_depth=n_depth, tau_min=tau_min,
            correlated_microfields=correlated_microfields,
            neutral_radius_scale=neutral_radius_scale,
            include_helium_dimer_ion=include_helium_dimer_ion,
            include_helium_three_body_cia=include_helium_three_body_cia,
            include_rydberg_bound_free=include_rydberg_bound_free,
            helium_reos3_table=helium_reos3_table,
            metal_database=metal_database,
            metal_abundances=metal_abundances,
            log_hydrogen_abundance=log_hydrogen_abundance,
            include_dense_helium_metal_ionization=(
                include_dense_helium_metal_ionization
            ),
        )
    relaxation_depth = seed.rosseland_optical_depth.copy()
    structure_helium_i_lines = HELIUM_I_LINES
    structure_helium_ii_lines = (
        HELIUM_II_LINES if include_helium_ii_lines else ()
    )
    if include_lines:
        # Screen structurally irrelevant helium transitions using a strict
        # upper bound on line-centre optical depth above tau_R ~= 1.  Lines
        # below 1e-3 cannot alter a structure solved to 3e-3 flux accuracy;
        # final spectrum synthesis remains unscreened.
        helium_state = seed.helium_lte_state
        if helium_state is not None:
            first_below_photosphere = min(
                int(
                    np.searchsorted(
                        seed.rosseland_optical_depth, 1.0, side="right"
                    )
                )
                + 1,
                seed.n_depth,
            )
            thermal_velocity_fraction = np.sqrt(
                BOLTZMANN * 10_000.0 / HELIUM_MASS
            ) / LIGHT_SPEED
            integrated_cross_section = (
                PI * ELEMENTARY_CHARGE_ESU**2
                / (ELECTRON_MASS * LIGHT_SPEED)
            )
            retained_lines = []
            for line in HELIUM_I_LINES:
                center = line.wavelength_vacuum_angstrom
                center_cm = center * 1.0e-8
                doppler_peak_per_angstrom = 1.0 / (
                    np.sqrt(2.0 * PI)
                    * center
                    * thermal_velocity_fraction
                )
                stimulated = -np.expm1(
                    -PLANCK
                    * LIGHT_SPEED
                    / (center_cm * BOLTZMANN * seed.temperature)
                )
                upper_mass_opacity = (
                    integrated_cross_section
                    * line.absorption_oscillator_strength
                    * helium_state.neutral_level_population_density[
                        :, line.lower_term_index
                    ]
                    * stimulated
                    * doppler_peak_per_angstrom
                    * 1.0e8
                    * center_cm**2
                    / LIGHT_SPEED
                    / seed.mass_density
                )
                upper_optical_depth = np.trapz(
                    upper_mass_opacity[:first_below_photosphere],
                    seed.column_mass[:first_below_photosphere],
                )
                if upper_optical_depth >= 1.0e-3:
                    retained_lines.append(line)
            structure_helium_i_lines = tuple(retained_lines)
            if include_helium_ii_lines:
                maximum_helium_ii_level = max(
                    line.upper_principal_quantum_number
                    for line in HELIUM_II_LINES
                )
                helium_ii_population, _ = _helium_ii_level_distribution(
                    seed, maximum_helium_ii_level
                )
                retained_helium_ii_lines = []
                for line in HELIUM_II_LINES:
                    center = line.wavelength_vacuum_angstrom
                    center_cm = center * 1.0e-8
                    doppler_peak_per_angstrom = 1.0 / (
                        np.sqrt(2.0 * PI)
                        * center
                        * thermal_velocity_fraction
                    )
                    stimulated = -np.expm1(
                        -PLANCK
                        * LIGHT_SPEED
                        / (center_cm * BOLTZMANN * seed.temperature)
                    )
                    upper_mass_opacity = (
                        integrated_cross_section
                        * line.absorption_oscillator_strength
                        * helium_ii_population[
                            :, line.lower_principal_quantum_number - 1
                        ]
                        * stimulated
                        * doppler_peak_per_angstrom
                        * 1.0e8
                        * center_cm**2
                        / LIGHT_SPEED
                        / seed.mass_density
                    )
                    upper_optical_depth = np.trapz(
                        upper_mass_opacity[:first_below_photosphere],
                        seed.column_mass[:first_below_photosphere],
                    )
                    if upper_optical_depth >= 1.0e-3:
                        retained_helium_ii_lines.append(line)
                structure_helium_ii_lines = tuple(retained_helium_ii_lines)
    wavelength = np.geomspace(100.0, 100_000.0, n_continuum_wavelength)
    if include_lines:
        core_offsets = np.arange(-5.0, 5.0001, 0.2)
        line_grids = [
            *(
                line.wavelength_vacuum_angstrom + core_offsets
                for line in structure_helium_i_lines
            )
        ]
        if structure_helium_i_lines:
            line_grids.append(np.arange(2600.0, 7500.1, 10.0))
        uv_resonance_grid = np.arange(480.0, 700.0001, 0.5)
        uv_flux_fraction = float(
            PI
            * np.trapz(
                planck_lambda_angstrom(
                    uv_resonance_grid, effective_temperature
                ),
                uv_resonance_grid,
            )
            / (STEFAN_BOLTZMANN * effective_temperature**4)
        )
        if uv_flux_fraction >= 1.0e-8:
            line_grids.append(uv_resonance_grid)
        if structure_helium_ii_lines:
            line_grids.extend(
                line.wavelength_vacuum_angstrom + core_offsets
                for line in structure_helium_ii_lines
            )
        if line_grids:
            wavelength = np.unique(np.concatenate((wavelength, *line_grids)))
    metal_wing_sampled_lines = 0
    if (
        metal_database is not None
        and metal_abundances is not None
        and include_metal_lines
    ):
        from .metals import metal_lte_state, selected_metal_lines

        selection_state = metal_lte_state(
            seed,
            metal_database,
            metal_abundances,
            log_hydrogen_abundance=log_hydrogen_abundance,
            include_dense_helium_ionization=(
                include_dense_helium_metal_ionization
            ),
        )
        ion_stage_weight = {
            (element, charge): float(np.max(
                populations[charge]
                / np.maximum(
                    selection_state.element_number_density[element],
                    np.finfo(np.float64).tiny,
                )
            ))
            for element, populations in selection_state.ion_number_density.items()
            for charge in range(populations.shape[0])
        }

        structure_lines = selected_metal_lines(
            metal_database,
            metal_abundances,
            100.0,
            100_000.0,
            minimum_metal_oscillator_strength,
            maximum_metal_lines,
            effective_temperature,
            ion_stage_weight,
            flux_weighted=True,
        )
        metal_centers = np.asarray(
            [line.wavelength_vacuum_angstrom for _, line in structure_lines]
        )
        if metal_centers.size:
            metal_grid, metal_wing_sampled_lines = (
                _metal_line_opacity_sampling_grid(metal_centers)
            )
            wavelength = np.unique(np.concatenate((wavelength, metal_grid)))
    if log_hydrogen_abundance is not None and include_trace_hydrogen_lines:
        hydrogen_centers = np.asarray([
            line.wavelength_vacuum_angstrom
            for line in (*LYMAN_LINES, *BALMER_LINES, *PASCHEN_LINES, *BRACKETT_LINES)
        ])
        hydrogen_offsets = np.asarray([-5.0, -1.0, 0.0, 1.0, 5.0])
        hydrogen_grid = (
            hydrogen_centers[:, np.newaxis] + hydrogen_offsets
        ).ravel()
        wavelength = np.unique(np.concatenate((wavelength, hydrogen_grid)))
        if homogeneous_mixture:
            # H opacity controls the optically thin temperature in a DBA just
            # as it does in a DA.  Five samples per line miss the narrow cores
            # and severely under-resolve the broad Lyman blanketing.  Match
            # the mature hydrogen solver and explicitly bracket the first
            # four bound-free thresholds so quadrature never interpolates
            # across a discontinuity.
            hydrogen_core_offsets = np.arange(-5.0, 5.0001, 0.2)
            lyman_core_offsets = np.arange(-3.0, 3.0001, 0.05)
            lyman_limit = (
                PLANCK * LIGHT_SPEED / HYDROGEN_IONIZATION_ENERGY * 1.0e8
            )
            series_limits = lyman_limit * np.arange(1.0, 5.0) ** 2
            edge_samples = (
                series_limits[:, np.newaxis]
                * np.asarray((1.0 - 1.0e-4, 1.0 + 1.0e-4))[np.newaxis, :]
            ).ravel()
            wavelength = np.unique(np.concatenate((
                wavelength,
                np.arange(900.0, 1250.1, 1.0),
                np.arange(1255.0, 3000.1, 5.0),
                edge_samples,
                *(
                    line.wavelength_vacuum_angstrom + hydrogen_core_offsets
                    for line in BALMER_LINES[:4]
                ),
                *(
                    line.wavelength_vacuum_angstrom + lyman_core_offsets
                    for line in LYMAN_LINES[:3]
                ),
            )))
    if metal_photoionization_database is not None:
        edge_grid = np.asarray([
            PLANCK * LIGHT_SPEED / (fit.threshold_energy_ev * 1.602_176_634e-12)
            * 1.0e8
            for fit in metal_photoionization_database.fits.values()
            if fit.element in metal_abundances
        ])
        if edge_grid.size:
            edge_samples = (
                edge_grid[np.newaxis, :]
                * (1.0 + np.asarray([-1.0e-4, 1.0e-4]))[:, np.newaxis]
            ).ravel()
            wavelength = np.unique(np.concatenate((wavelength, edge_samples)))
    if metal_topbase_photoionization_database is not None:
        topbase_edge_grid = np.asarray([
            PLANCK * LIGHT_SPEED
            / (section.threshold_energy_ev * 1.602_176_634e-12)
            * 1.0e8
            for section in metal_topbase_photoionization_database.sections
            if section.element in metal_abundances
        ])
        if topbase_edge_grid.size:
            topbase_edge_samples = (
                topbase_edge_grid[np.newaxis, :]
                * (1.0 + np.asarray([-1.0e-4, 1.0e-4]))[:, np.newaxis]
            ).ravel()
            wavelength = np.unique(
                np.concatenate((wavelength, topbase_edge_samples))
            )
    # Broad unified metal profiles carry flux over hundreds of Angstroms.
    # Sample their actual vector grids in the structure solution, rather than
    # relying only on five conventional line-center points.  A deterministic
    # stride caps the transfer cost without discarding satellites or edges.
    unified_profile_grids: list[FloatArray] = []
    if mg_he_red_wing_table is not None:
        for local_wavelength in (
            mg_he_red_wing_table.wavelength_by_temperature.values()
        ):
            step = max(1, local_wavelength.size // 100)
            unified_profile_grids.append(local_wavelength[::step])
        if mg_he_red_wing_table.wavelength_by_density is not None:
            for local_wavelength in (
                mg_he_red_wing_table.wavelength_by_density.values()
            ):
                step = max(1, local_wavelength.size // 100)
                unified_profile_grids.append(local_wavelength[::step])
    if mg_ii_he_profile_table is not None:
        step = max(1, mg_ii_he_profile_table.wavelength_angstrom.size // 400)
        unified_profile_grids.append(mg_ii_he_profile_table.wavelength_angstrom[::step])
    if ca_i_he_profile_table is not None:
        for local_wavelength in ca_i_he_profile_table.wavelength_by_density.values():
            step = max(1, local_wavelength.size // 100)
            unified_profile_grids.append(local_wavelength[::step])
        if ca_i_he_profile_table.wavelength_by_temperature is not None:
            for local_wavelength in (
                ca_i_he_profile_table.wavelength_by_temperature.values()
            ):
                step = max(1, local_wavelength.size // 100)
                unified_profile_grids.append(local_wavelength[::step])
    if ca_ii_he_profile_table is not None:
        unified_profile_grids.append(ca_ii_he_profile_table.wavelength_angstrom)
    if unified_profile_grids:
        wavelength = np.unique(np.concatenate((wavelength, *unified_profile_grids)))
    target_flux = STEFAN_BOLTZMANN * effective_temperature**4
    if initial_temperature is None:
        temperature = seed.temperature.copy()
    else:
        supplied = np.asarray(initial_temperature, dtype=np.float64)
        if np.any(~np.isfinite(supplied)) or np.any(supplied <= 0.0):
            raise ValueError("initial_temperature must contain finite positive values")
        if initial_column_mass is not None:
            mass = np.asarray(initial_column_mass, dtype=np.float64)
            if mass.shape != supplied.shape or np.any(np.diff(mass) <= 0.0):
                raise ValueError("initial_column_mass must increase and match initial_temperature")
            temperature = np.interp(
                np.log(seed.column_mass), np.log(mass), supplied,
                left=supplied[0], right=supplied[-1],
            )
        elif supplied.shape == seed.temperature.shape:
            temperature = supplied.copy()
        else:
            raise ValueError("initial_temperature must have one value per depth")

    convective_gradient_operator: FloatArray | None = None
    if mixing_length_alpha is not None:
        log_pressure_grid = np.log(seed.gas_pressure)
        convective_gradient_operator = np.empty((n_depth, n_depth))
        for column in range(n_depth):
            basis = np.zeros(n_depth)
            basis[column] = 1.0
            convective_gradient_operator[:, column] = np.gradient(
                basis, log_pressure_grid, edge_order=2
            )

    def with_temperature(values: FloatArray) -> Atmosphere:
        if homogeneous_mixture:
            assert log_hydrogen_abundance is not None
            mixed_eos = hummer_mihalas_hydrogen_helium_lte(
                values,
                seed.gas_pressure,
                log_hydrogen_abundance,
                helium_neutral_radius_scale=neutral_radius_scale,
                correlated_microfields=correlated_microfields,
            )
            result = _atmosphere_from_hydrogen_helium_state(
                seed.effective_temperature,
                seed.logg,
                relaxation_depth,
                seed.column_mass,
                values,
                seed.gas_pressure,
                mixed_eos,
                seed.metadata,
            )
        else:
            eos = (
                hummer_mihalas_helium_lte(
                    values, seed.gas_pressure,
                    neutral_radius_scale=neutral_radius_scale,
                    correlated_microfields=correlated_microfields,
                )
                if helium_reos3_table is None
                else hummer_mihalas_helium_lte_with_reos3(
                    values, seed.gas_pressure, helium_reos3_table,
                    neutral_radius_scale=neutral_radius_scale,
                    correlated_microfields=correlated_microfields,
                )
            )
            result = Atmosphere(
                seed.effective_temperature, seed.logg, relaxation_depth,
                seed.column_mass, values, seed.gas_pressure, eos.mass_density,
                np.zeros(n_depth), np.zeros(n_depth), eos.electron_density,
                seed.metadata, helium_lte_state=eos,
            )
        if metal_database is not None and metal_abundances is not None:
            from .metals import atmosphere_with_metal_electrons, metal_lte_state

            metal_state = metal_lte_state(
                result,
                metal_database,
                metal_abundances,
                reference_species="He",
                include_dense_helium_ionization=(
                    include_dense_helium_metal_ionization
                ),
                log_hydrogen_abundance=log_hydrogen_abundance,
            )
            result = atmosphere_with_metal_electrons(result, metal_state)
        return result

    structure_absorption_cache: dict[str, object] = {}

    def true_absorption(current: Atmosphere) -> FloatArray:
        result = helium_continuum_mass_absorption_coefficient(
            current, wavelength, include_electron_scattering=False,
            include_rayleigh_scattering=False,
            include_helium_dimer_ion=include_helium_dimer_ion,
            include_helium_three_body_cia=include_helium_three_body_cia,
            include_rydberg_bound_free=include_rydberg_bound_free,
        )
        if include_lines:
            result += helium_i_line_mass_absorption_coefficient(
                current, wavelength, stark_table,
                lines=structure_helium_i_lines,
                include_occupation_probability=True,
                # This convolution is expensive but is required for flux
                # consistency in cool, dense DB atmospheres.  Callers can
                # explicitly select "none" for fast warm-star diagnostics.
                neutral_broadening=neutral_line_broadening,
            )
            result += helium_i_resonance_line_mass_absorption_coefficient(
                current,
                wavelength,
                include_occupation_probability=True,
            )
            if include_helium_ii_lines:
                result += helium_ii_line_mass_absorption_coefficient(
                    current,
                    wavelength,
                    lines=structure_helium_ii_lines,
                    stark_table=helium_ii_stark_table,
                    include_occupation_probability=True,
                )
        if metal_database is not None and metal_abundances is not None and (
            include_metal_lines
            or metal_photoionization_database is not None
            or metal_topbase_photoionization_database is not None
        ):
            from .metals import (
                metal_bound_free_mass_absorption_coefficient,
                metal_line_mass_absorption_coefficient,
                metal_lte_state,
            )

            metal_state = metal_lte_state(
                current,
                metal_database,
                metal_abundances,
                reference_species="He",
                include_dense_helium_ionization=(
                    include_dense_helium_metal_ionization
                ),
                log_hydrogen_abundance=log_hydrogen_abundance,
            )
            if metal_photoionization_database is not None:
                result += metal_bound_free_mass_absorption_coefficient(
                    current,
                    wavelength,
                    metal_database,
                    metal_state,
                    metal_photoionization_database,
                    excluded_ions=(
                        ()
                        if metal_topbase_photoionization_database is None
                        else metal_topbase_photoionization_database.ion_stages
                    ),
                )
            if metal_topbase_photoionization_database is not None:
                from .d6 import topbase_bound_free_mass_absorption_coefficient

                result += topbase_bound_free_mass_absorption_coefficient(
                    current,
                    wavelength,
                    metal_state,
                    metal_topbase_photoionization_database,
                )
            if include_metal_lines:
                result += metal_line_mass_absorption_coefficient(
                    current,
                    wavelength,
                    metal_database,
                    metal_state,
                    mg_he_red_wing_table=mg_he_red_wing_table,
                    mg_ii_he_profile_table=mg_ii_he_profile_table,
                    ca_i_he_profile_table=ca_i_he_profile_table,
                    ca_ii_he_profile_table=ca_ii_he_profile_table,
                    minimum_oscillator_strength=minimum_metal_oscillator_strength,
                    maximum_lines=maximum_metal_lines,
                )
        if current.hydrogen_lte_state is not None:
            result += hydrogen_continuum_mass_absorption_coefficient(
                current,
                wavelength,
                include_electron_scattering=False,
                include_rayleigh_scattering=False,
                include_molecular_absorption=False,
            )
            if include_trace_hydrogen_lines:
                result += balmer_mass_absorption_coefficient(
                    current,
                    wavelength,
                    include_self_broadening=include_hydrogen_self_broadening,
                    include_neutral_helium_broadening=(
                        include_hydrogen_neutral_helium_broadening
                    ),
                    self_broadening_quadrature_order=(
                        hydrogen_self_broadening_quadrature_order
                    ),
                    self_broadening_impact_validity_fraction=(
                        hydrogen_self_broadening_impact_validity_fraction
                    ),
                    self_broadening_prescription=(
                        hydrogen_self_broadening_prescription
                    ),
                    self_broadening_truncation_closure=(
                        hydrogen_self_broadening_truncation_closure
                    ),
                    profile_edge_optical_depth=1.0e-4,
                )
                result += lyman_mass_absorption_coefficient(
                    current,
                    wavelength,
                    unified_allard_table=unified_allard_table,
                    allard_stark_weight=allard_stark_weight,
                )
                result += paschen_mass_absorption_coefficient(current, wavelength)
                result += brackett_mass_absorption_coefficient(current, wavelength)
                if include_hydrogen_series_pseudocontinuum:
                    result += (
                        hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
                            current, wavelength
                        )
                    )
        structure_absorption_cache.clear()
        structure_absorption_cache.update(
            atmosphere=current,
            absorption=result,
        )
        return result

    if structure_solver == "adaptive-newton":
        from .adaptive_structure import (
            rosseland_mean_from_opacity_grid,
            solve_adaptive_lte_structure,
        )
        from .eos import (
            hummer_mihalas_helium_thermodynamics,
            hummer_mihalas_hydrogen_helium_thermodynamics,
        )

        def scattering_opacity(current: Atmosphere) -> FloatArray:
            scattering = (
                electron_scattering_mass_coefficient(current)[np.newaxis, :]
                + helium_rayleigh_scattering_mass_coefficient(
                    current, wavelength
                )
            )
            if current.hydrogen_lte_state is not None:
                scattering += hydrogen_rayleigh_scattering_mass_coefficient(
                    current, wavelength
                )
            return scattering

        def rosseland_opacity(current: Atmosphere) -> FloatArray:
            metal_opacity_is_explicit = (
                metal_database is not None
                and (
                    include_metal_lines
                    or metal_photoionization_database is not None
                    or metal_topbase_photoionization_database is not None
                )
            )
            if metal_opacity_is_explicit:
                if structure_absorption_cache.get("atmosphere") is current:
                    absorption = np.asarray(
                        structure_absorption_cache["absorption"],
                        dtype=np.float64,
                    )
                else:
                    absorption = true_absorption(current)
                return rosseland_mean_from_opacity_grid(
                    wavelength,
                    absorption + scattering_opacity(current),
                    current.temperature,
                )
            function = (
                rosseland_mean_hydrogen_helium_continuum_opacity
                if homogeneous_mixture
                else rosseland_mean_helium_continuum_opacity
            )
            return function(
                current,
                n_frequency=120,
                include_helium_dimer_ion=include_helium_dimer_ion,
                include_helium_three_body_cia=include_helium_three_body_cia,
                include_rydberg_bound_free=include_rydberg_bound_free,
            )

        def thermodynamics(current: Atmosphere) -> object:
            if homogeneous_mixture:
                assert log_hydrogen_abundance is not None
                return hummer_mihalas_hydrogen_helium_thermodynamics(
                    current.temperature,
                    current.gas_pressure,
                    log_hydrogen_abundance,
                    helium_neutral_radius_scale=neutral_radius_scale,
                    correlated_microfields=correlated_microfields,
                )
            return hummer_mihalas_helium_thermodynamics(
                current.temperature,
                current.gas_pressure,
                neutral_radius_scale=neutral_radius_scale,
                correlated_microfields=correlated_microfields,
                helium_reos3_table=helium_reos3_table,
            )

        adaptive_seed = with_temperature(temperature)
        return solve_adaptive_lte_structure(
            adaptive_seed,
            wavelength,
            with_temperature=with_temperature,
            true_absorption=true_absorption,
            scattering_opacity=scattering_opacity,
            rosseland_opacity=rosseland_opacity,
            thermodynamics=thermodynamics,
            mixing_length_alpha=mixing_length_alpha,
            max_iterations=max_iterations,
            temperature_tolerance=temperature_tolerance,
            flux_tolerance=flux_tolerance,
            n_angle=n_angle,
            initial_temperature_was_supplied=(
                initial_temperature is not None
                and resume_supplied_structure_in_formal_flux_phase
            ),
            iteration_callback=iteration_callback,
            metadata={
                **{
                    key: value
                    for key, value in seed.metadata.items()
                    if key not in ("model", "composition", "eos")
                },
                "composition": (
                    "metal-polluted-helium"
                    if metal_database is not None
                    else "homogeneous-hydrogen-helium"
                    if homogeneous_mixture
                    else "pure-helium"
                ),
                "eos": (
                    "trace-metal-charge-neutral-q-mhd-helium-occupation-probability"
                    if metal_database is not None
                    else "hummer-mihalas-hydrogen-helium-occupation-probability"
                    if homogeneous_mixture
                    else "q-mhd-helium-occupation-probability"
                    if correlated_microfields
                    else "hummer-mihalas-helium-occupation-probability"
                ),
                "helium_neutral_radius_scale": float(neutral_radius_scale),
                "radiative_equilibrium_includes_helium_lines": bool(
                    include_lines
                ),
                "radiative_equilibrium_retained_helium_i_lines": int(
                    len(structure_helium_i_lines)
                ),
                "radiative_equilibrium_retained_helium_ii_lines": int(
                    len(structure_helium_ii_lines)
                ),
                "radiative_equilibrium_neutral_helium_line_broadening": (
                    neutral_line_broadening if include_lines else "disabled"
                ),
                "helium_stark_profiles": (
                    "Tremblay-2026/Beauchamp-2025 explicit table"
                ),
                "helium_i_ground_resonance_broadening": (
                    "Dimitrijevic-Sahal-Brechot-1989 electron/He-II impact"
                ),
                "helium_ii_profiles": (
                    "Schoning-Butler/SYNSPEC table"
                    if include_lines
                    and include_helium_ii_lines
                    and helium_ii_stark_table is not None
                    else "hydrogenic Z^-5 transform of unified hydrogen tables"
                    if include_lines and include_helium_ii_lines
                    else "disabled"
                ),
                "helium_dimer_ion_continuum": bool(include_helium_dimer_ion),
                "helium_three_body_collision_induced_absorption": bool(
                    include_helium_three_body_cia
                ),
                "helium_rydberg_bound_free": bool(include_rydberg_bound_free),
                "helium_minus_free_free": (
                    "John-1994 inside tabulated T/lambda domain; "
                    "Carbon-1969 fit to John-1968 elsewhere"
                ),
                "radiative_equilibrium_hydrogen_abundance": (
                    float(log_hydrogen_abundance)
                    if log_hydrogen_abundance is not None
                    else None
                ),
                "radiative_equilibrium_balmer_profile_edge_optical_depth": (
                    1.0e-4 if include_trace_hydrogen_lines else None
                ),
                "radiative_equilibrium_metal_opacity": bool(
                    metal_database is not None
                ),
                "radiative_equilibrium_includes_metal_lines": bool(
                    metal_database is not None and include_metal_lines
                ),
                "radiative_equilibrium_includes_metal_bound_free": bool(
                    metal_photoionization_database is not None
                    or metal_topbase_photoionization_database is not None
                ),
                "radiative_equilibrium_level_resolved_metal_bound_free": (
                    metal_topbase_photoionization_database.source
                    if metal_topbase_photoionization_database is not None
                    else "disabled"
                ),
                "radiative_equilibrium_minimum_metal_oscillator_strength": float(
                    minimum_metal_oscillator_strength
                ),
                "radiative_equilibrium_maximum_metal_lines": maximum_metal_lines,
                "radiative_equilibrium_wing_sampled_metal_lines": int(
                    metal_wing_sampled_lines
                ),
                "radiative_equilibrium_metal_line_selection": (
                    "abundance-gf-boltzmann-ion-fraction-Planck-flux at Teff"
                ),
                "metal_abundances": (
                    dict(metal_abundances)
                    if metal_abundances is not None else {}
                ),
                "metal_electron_feedback": (
                    "charge-neutral EOS and continuum opacity"
                    if metal_database is not None else "disabled"
                ),
                "metal_thermodynamic_derivatives": (
                    "trace-metal approximation: Q-MHD helium derivatives"
                    if metal_database is not None else "not applicable"
                ),
                "rosseland_opacity_includes_metal_bound_bound_and_bound_free": (
                    metal_database is not None
                    and (
                        include_metal_lines
                        or metal_photoionization_database is not None
                        or metal_topbase_photoionization_database is not None
                    )
                ),
            },
        )

    flux_ratio = np.inf
    maximum_correction = np.inf
    maximum_correction_depth_index = 0
    last_evaluated_temperature = temperature.copy()
    last_flux_ratio = flux_ratio
    last_maximum_correction = maximum_correction
    last_maximum_correction_depth_index = maximum_correction_depth_index
    last_required_convective_flux = np.zeros(n_depth)
    last_convective = np.zeros(n_depth, dtype=bool)
    last_maximum_convective_gradient_residual = 0.0
    last_maximum_convective_gradient_residual_depth_index = 0
    last_maximum_total_flux_residual = 0.0
    converged = False
    required_convective_flux = np.zeros(n_depth)
    convective = np.zeros(n_depth, dtype=bool)
    maximum_convective_gradient_residual = 0.0
    maximum_convective_gradient_residual_depth_index = 0
    maximum_total_flux_residual = 0.0
    convergence_streak = 0
    flux_ratio_history: list[float] = []
    maximum_correction_history: list[float] = []
    maximum_convective_gradient_residual_history: list[float] = []
    maximum_convective_gradient_residual_depth_index_history: list[int] = []
    maximum_total_flux_residual_history: list[float] = []
    effective_convective_correction_damping_history: list[float] = []
    convective_damping_backtrack = 1.0
    convective_flux_improvement_streak = 0
    previous_maximum_total_flux_residual = np.inf
    best_total_flux_residual = np.inf
    best_transport_state: tuple[object, ...] | None = None
    for iteration in range(1, max_iterations + 1):
        atmosphere = with_temperature(temperature)
        absorption = true_absorption(atmosphere)
        scattering = (
            electron_scattering_mass_coefficient(atmosphere)[np.newaxis, :]
            + helium_rayleigh_scattering_mass_coefficient(atmosphere, wavelength)
        )
        if atmosphere.hydrogen_lte_state is not None:
            scattering += hydrogen_rayleigh_scattering_mass_coefficient(
                atmosphere, wavelength
            )
        total = absorption + scattering
        optical_depth = optical_depth_from_mass_opacity(atmosphere.column_mass, total)
        planck = planck_lambda_angstrom(
            wavelength[:, np.newaxis], temperature[np.newaxis, :]
        )
        source = planck.copy()
        for _ in range(4):
            field = radiation_field(optical_depth, source, n_angle=n_angle)
            source = (absorption * planck + scattering * field.mean_intensity) / total
        field = radiation_field(optical_depth, source, n_angle=n_angle)
        emitted = np.trapz(absorption * planck, wavelength, axis=0)
        absorbed = np.trapz(absorption * field.mean_intensity, wavelength, axis=0)
        active = relaxation_depth < 10.0
        convective_weight = np.zeros(n_depth)
        convective_log_temperature_correction = np.zeros(n_depth)
        if mixing_length_alpha is not None:
            rosseland_function = (
                rosseland_mean_hydrogen_helium_continuum_opacity
                if homogeneous_mixture
                else rosseland_mean_helium_continuum_opacity
            )
            rosseland_for_convection = rosseland_function(
                atmosphere,
                n_frequency=120,
                include_helium_dimer_ion=include_helium_dimer_ion,
                include_helium_three_body_cia=include_helium_three_body_cia,
                include_rydberg_bound_free=include_rydberg_bound_free,
            )
            radiative_flux = np.trapz(field.flux, wavelength, axis=0)
            required_convective_flux = np.clip(
                target_flux - radiative_flux, 0.0, target_flux
            )
            actual_gradient = np.gradient(
                np.log(temperature), np.log(atmosphere.gas_pressure),
                edge_order=2,
            )
            if homogeneous_mixture:
                from .eos import hummer_mihalas_hydrogen_helium_thermodynamics

                assert log_hydrogen_abundance is not None
                adiabatic_gradient = (
                    hummer_mihalas_hydrogen_helium_thermodynamics(
                        temperature,
                        atmosphere.gas_pressure,
                        log_hydrogen_abundance,
                        helium_neutral_radius_scale=neutral_radius_scale,
                        correlated_microfields=correlated_microfields,
                    ).adiabatic_temperature_gradient
                )
            else:
                from .eos import hummer_mihalas_helium_thermodynamics

                adiabatic_gradient = hummer_mihalas_helium_thermodynamics(
                    temperature, atmosphere.gas_pressure,
                    neutral_radius_scale=neutral_radius_scale,
                    correlated_microfields=correlated_microfields,
                    helium_reos3_table=helium_reos3_table,
                ).adiabatic_temperature_gradient
            ml2_gradient = ml2_temperature_gradient_for_flux(
                atmosphere, rosseland_for_convection,
                required_convective_flux,
                mixing_length_alpha=mixing_length_alpha,
            )
            current_convective_flux = ml2_convective_flux(
                atmosphere,
                rosseland_for_convection,
                mixing_length_alpha=mixing_length_alpha,
            )
            flux_balance_monitor = (
                (relaxation_depth > 1.0e-3)
                & (relaxation_depth < 10.0)
            )
            maximum_total_flux_residual = (
                float(
                    np.max(
                        np.abs(
                            (
                                radiative_flux[flux_balance_monitor]
                                + current_convective_flux[flux_balance_monitor]
                            )
                            / target_flux
                            - 1.0
                        )
                    )
                )
                if np.any(flux_balance_monitor)
                else 0.0
            )
            convective = (
                (required_convective_flux > 1.0e-4 * target_flux)
                & (ml2_gradient > adiabatic_gradient)
                & (relaxation_depth > 1.0e-3)
            )
            convective_weight[convective] = np.clip(
                required_convective_flux[convective]
                / (0.02 * target_flux),
                0.0,
                1.0,
            )
            # The radiative correction can stop at tau_R=10, but the
            # convective gradient needs a deeper buffer.  Cutting it at the
            # same cell leaves the centered derivative at the boundary tied
            # to an uncorrected neighbour and makes the measured mismatch
            # grow even while the interior improves.  Taper the ML2 update
            # smoothly to zero between tau_R=10 and 100; this also avoids the
            # poorly conditioned last cells of the finite atmosphere.
            convective_depth_weight = np.clip(
                np.log10(100.0 / np.maximum(relaxation_depth, 1.0e-300)),
                0.0,
                1.0,
            )
            convective_weight *= convective_depth_weight
            convective = convective_weight > 0.0
            maximum_convective_gradient_residual = float(
                np.max(
                    convective_weight
                    * np.abs(ml2_gradient - actual_gradient)
                )
            )
            maximum_convective_gradient_residual_depth_index = int(
                np.argmax(
                    convective_weight
                    * np.abs(ml2_gradient - actual_gradient)
                )
            )
            gradient_correction = convective_weight * (
                ml2_gradient - actual_gradient
            )
            log_pressure = np.log(atmosphere.gas_pressure)
            convective_log_temperature_correction.fill(0.0)
            if np.any(convective):
                # Anchor immediately above the first unstable layer and
                # integrate the required change in dlnT/dlnP inward.  This
                # preserves the already-converged radiative surface instead
                # of anchoring at the artificial bottom boundary and cooling
                # the whole photosphere while the deep adiabat is rebuilt.
                anchor = max(int(np.flatnonzero(convective)[0]) - 1, 0)
                for depth in range(anchor + 1, n_depth):
                    cell_correction = 0.5 * (
                        gradient_correction[depth - 1]
                        + gradient_correction[depth]
                    )
                    convective_log_temperature_correction[depth] = (
                        convective_log_temperature_correction[depth - 1]
                        + cell_correction
                        * (log_pressure[depth] - log_pressure[depth - 1])
                    )
                # np.gradient uses centered, non-uniform finite differences;
                # simply integrating point gradients with a trapezoid does
                # not invert that discrete operator.  At a sharp convective
                # boundary it can therefore improve one layer while driving
                # its neighbour in the wrong direction.  Solve the small
                # weighted linear problem for the actual discrete gradients,
                # with weak regularization toward the smooth integral above.
                assert convective_gradient_operator is not None
                fitted_depth = np.flatnonzero(convective)
                corrected_depth = np.arange(anchor + 1, n_depth)
                fit_weight = np.sqrt(convective_weight[fitted_depth])
                design = convective_gradient_operator[
                    np.ix_(fitted_depth, corrected_depth)
                ] * fit_weight[:, np.newaxis]
                target = (
                    gradient_correction[fitted_depth] * fit_weight
                )
                regularization = 1.0e-4
                augmented_design = np.vstack((
                    design,
                    np.sqrt(regularization)
                    * np.eye(corrected_depth.size),
                ))
                augmented_target = np.concatenate((
                    target,
                    np.sqrt(regularization)
                    * convective_log_temperature_correction[corrected_depth],
                ))
                convective_log_temperature_correction[corrected_depth] = (
                    np.linalg.lstsq(
                        augmented_design,
                        augmented_target,
                        rcond=None,
                    )[0]
                )
            # Preserve the *shape* of the integrated gradient correction.
            # Element-wise clipping turns a broad correction into two flat
            # plateaus and therefore changes only one cell at their boundary;
            # a model can then look numerically stationary while carrying no
            # convective flux over most of the unstable zone.  A single scale
            # factor limits the temperature step without destroying the
            # desired gradient throughout that zone.
            maximum_raw_convective_correction = float(
                np.max(np.abs(convective_log_temperature_correction))
            )
            if maximum_raw_convective_correction > 0.04:
                convective_log_temperature_correction *= (
                    0.04 / maximum_raw_convective_correction
                )

        else:
            maximum_convective_gradient_residual = 0.0
            maximum_convective_gradient_residual_depth_index = 0
            maximum_total_flux_residual = 0.0
        hotter_temperature = 1.001 * temperature
        hotter_planck = planck_lambda_angstrom(
            wavelength[:, np.newaxis], hotter_temperature[np.newaxis, :]
        )
        planck_derivative = np.trapz(
            absorption * (hotter_planck - planck), wavelength, axis=0
        ) / 0.001
        if include_absorption_temperature_derivative:
            hotter_absorption = true_absorption(with_temperature(hotter_temperature))
            residual = emitted - absorbed
            hotter_residual = np.trapz(
                hotter_absorption * (hotter_planck - field.mean_intensity),
                wavelength, axis=0,
            )
            full_derivative = (hotter_residual - residual) / 0.001
            derivative = np.where(
                np.isfinite(full_derivative) & (full_derivative > 0.0),
                full_derivative, planck_derivative,
            )
        else:
            derivative = planck_derivative
        correction = (absorbed - emitted) / np.maximum(derivative, np.finfo(np.float64).tiny)
        correction = np.where(active, np.clip(correction, -0.04, 0.04), 0.0)
        smoothed = _smooth_radiative_temperature_correction(
            correction, convective_weight
        )
        # Smoothing a radiative-equilibrium correction after masking the
        # convective layers leaks the neighbouring radiative update straight
        # back into those layers.  Near the top of a coarsely sampled
        # convection zone that small leaked gradient can oppose the ML2
        # update and make a heavily damped iteration walk in the wrong
        # direction.  Reapply the same smooth convective weight after the
        # stencil so radiative and convective gradient controllers remain
        # disjoint.
        surface_flux = np.trapz(field.flux[:, 0], wavelength)
        flux_ratio = float(surface_flux / target_flux)
        global_correction = float(np.clip(-0.25 * np.log(flux_ratio), -0.04, 0.04))
        adaptive_convective_damping = float(
            np.clip(
                maximum_convective_gradient_residual / 0.1,
                0.005,
                1.0,
            )
        )
        if maximum_total_flux_residual > 0.5:
            # A small gradient mismatch can still correspond to orders of
            # magnitude too much ML2 flux.  In that regime the analytic line
            # search, rather than the absolute gradient error, must set the
            # step size; otherwise cool trace-H burn-in takes hundreds of
            # iterations.
            adaptive_convective_damping = 1.0
        elif maximum_total_flux_residual < 0.1:
            adaptive_convective_damping = min(
                adaptive_convective_damping, 0.1
            )
        elif maximum_total_flux_residual < 0.5:
            adaptive_convective_damping = min(
                adaptive_convective_damping, 0.2
            )
        if np.isfinite(previous_maximum_total_flux_residual):
            if (
                maximum_total_flux_residual
                > 1.01 * previous_maximum_total_flux_residual
            ):
                current_damping_limit = min(
                    convective_correction_damping,
                    adaptive_convective_damping,
                    convective_damping_backtrack,
                )
                convective_damping_backtrack = max(
                    0.0005, 0.5 * current_damping_limit
                )
                convective_flux_improvement_streak = 0
            elif (
                maximum_total_flux_residual
                < previous_maximum_total_flux_residual
            ):
                convective_flux_improvement_streak += 1
                if convective_flux_improvement_streak >= 3:
                    convective_damping_backtrack = min(
                        1.0, 1.25 * convective_damping_backtrack
                    )
                    convective_flux_improvement_streak = 0
            else:
                convective_flux_improvement_streak = 0
        previous_maximum_total_flux_residual = maximum_total_flux_residual
        maximum_convective_correction_damping = min(
            convective_correction_damping,
            adaptive_convective_damping,
            convective_damping_backtrack,
        )
        effective_convective_correction_damping = (
            maximum_convective_correction_damping
        )
        if np.any(convective):
            # Choose the ML2 step with a cheap frozen-radiation line search.
            # The formal transfer and opacity are not repeated: only the
            # analytic local ML2 flux is evaluated for trial gradients.  This
            # is enough to resolve the alternating corrections of adjacent
            # coarse depth points and avoids many conservatively damped full
            # atmosphere iterations.
            assert convective_gradient_operator is not None
            trial_damping = np.unique(
                np.maximum(
                    maximum_convective_correction_damping
                    * np.array([1.0, 0.5, 0.25, 0.125, 0.0625]),
                    5.0e-4,
                )
            )
            radiative_gradient_step = (
                temperature_correction_damping
                * convective_gradient_operator @ (0.5 * smoothed)
            )
            convective_gradient_step = (
                temperature_correction_damping
                * convective_gradient_operator
                @ convective_log_temperature_correction
            )
            trial_gradient = (
                actual_gradient[np.newaxis, :]
                + radiative_gradient_step[np.newaxis, :]
                + trial_damping[:, np.newaxis]
                * convective_gradient_step[np.newaxis, :]
            )
            trial_convective_flux = ml2_convective_flux_for_gradient(
                atmosphere,
                rosseland_for_convection,
                trial_gradient,
                mixing_length_alpha=mixing_length_alpha,
            )
            trial_total_flux_residual = np.max(
                np.abs(
                    (
                        radiative_flux[np.newaxis, flux_balance_monitor]
                        + trial_convective_flux[:, flux_balance_monitor]
                    )
                    / target_flux
                    - 1.0
                ),
                axis=1,
            )
            effective_convective_correction_damping = float(
                trial_damping[np.argmin(trial_total_flux_residual)]
            )
        convective_correction = (
            effective_convective_correction_damping
            * convective_log_temperature_correction
        )
        global_relaxation = 1.0 if np.any(convective) else 0.25
        applied = (
            0.5 * smoothed
            + global_relaxation * global_correction
            + convective_correction
        )
        applied *= temperature_correction_damping
        correction_monitor = active | convective
        maximum_correction = float(
            np.max(np.abs(applied[correction_monitor]))
        )
        maximum_correction_depth_index = int(
            np.argmax(
                np.where(correction_monitor, np.abs(applied), -1.0)
            )
        )
        flux_ratio_history.append(flux_ratio)
        maximum_correction_history.append(maximum_correction)
        maximum_convective_gradient_residual_history.append(
            maximum_convective_gradient_residual
        )
        maximum_convective_gradient_residual_depth_index_history.append(
            maximum_convective_gradient_residual_depth_index
        )
        maximum_total_flux_residual_history.append(
            maximum_total_flux_residual
        )
        effective_convective_correction_damping_history.append(
            effective_convective_correction_damping
        )
        within_tolerance = (
            maximum_correction < temperature_tolerance
            and abs(flux_ratio - 1.0) < flux_tolerance
            and maximum_convective_gradient_residual
            < convective_gradient_tolerance
            and maximum_total_flux_residual < convective_flux_tolerance
        )
        convergence_streak = convergence_streak + 1 if within_tolerance else 0
        last_evaluated_temperature = temperature.copy()
        last_flux_ratio = flux_ratio
        last_maximum_correction = maximum_correction
        last_maximum_correction_depth_index = maximum_correction_depth_index
        last_required_convective_flux = required_convective_flux.copy()
        last_convective = convective.copy()
        last_maximum_convective_gradient_residual = (
            maximum_convective_gradient_residual
        )
        last_maximum_convective_gradient_residual_depth_index = (
            maximum_convective_gradient_residual_depth_index
        )
        last_maximum_total_flux_residual = maximum_total_flux_residual
        if (
            mixing_length_alpha is not None
            and maximum_total_flux_residual < best_total_flux_residual
        ):
            best_total_flux_residual = maximum_total_flux_residual
            best_transport_state = (
                temperature.copy(),
                flux_ratio,
                maximum_correction,
                maximum_correction_depth_index,
                required_convective_flux.copy(),
                convective.copy(),
                maximum_convective_gradient_residual,
                maximum_convective_gradient_residual_depth_index,
                maximum_total_flux_residual,
                iteration,
            )
        if convergence_streak >= consecutive_convergence_iterations:
            converged = True
            break

        temperature *= np.exp(applied)

    selected_iteration = iteration
    # For a capped convective run, retain the evaluated structure with the
    # smallest *direct total-flux* residual.  The old scalar temperature merit
    # could erase necessary burn-in; this physically targeted checkpoint does
    # the opposite, preserving progress when a later trial crosses to the
    # other side of a highly nonlinear ML2 boundary.  Radiative-only runs keep
    # the last evaluated iterate.
    if not converged:
        if best_transport_state is not None:
            (
                temperature,
                flux_ratio,
                maximum_correction,
                maximum_correction_depth_index,
                required_convective_flux,
                convective,
                maximum_convective_gradient_residual,
                maximum_convective_gradient_residual_depth_index,
                maximum_total_flux_residual,
                selected_iteration,
            ) = best_transport_state
        else:
            temperature = last_evaluated_temperature
            flux_ratio = last_flux_ratio
            maximum_correction = last_maximum_correction
            maximum_correction_depth_index = (
                last_maximum_correction_depth_index
            )
            required_convective_flux = last_required_convective_flux
            convective = last_convective
            maximum_convective_gradient_residual = (
                last_maximum_convective_gradient_residual
            )
            maximum_convective_gradient_residual_depth_index = (
                last_maximum_convective_gradient_residual_depth_index
            )
            maximum_total_flux_residual = last_maximum_total_flux_residual

    atmosphere = with_temperature(temperature)
    final_rosseland_function = (
        rosseland_mean_hydrogen_helium_continuum_opacity
        if homogeneous_mixture
        else rosseland_mean_helium_continuum_opacity
    )
    rosseland = final_rosseland_function(
        atmosphere,
        include_helium_dimer_ion=include_helium_dimer_ion,
        include_helium_three_body_cia=include_helium_three_body_cia,
        include_rydberg_bound_free=include_rydberg_bound_free,
    )
    rosseland_depth = np.empty_like(atmosphere.column_mass)
    rosseland_depth[0] = rosseland[0] * atmosphere.column_mass[0]
    rosseland_depth[1:] = rosseland_depth[0] + np.cumsum(
        0.5 * (rosseland[1:] + rosseland[:-1]) * np.diff(atmosphere.column_mass)
    )
    return Atmosphere(
        atmosphere.effective_temperature, atmosphere.logg, rosseland_depth,
        atmosphere.column_mass, atmosphere.temperature, atmosphere.gas_pressure,
        atmosphere.mass_density, atmosphere.neutral_h_density,
        atmosphere.proton_density, atmosphere.electron_density,
        {
            "model": "non-gray-radiative-convective-equilibrium" if mixing_length_alpha is not None else "non-gray-radiative-equilibrium",
            "composition": (
                "metal-polluted-helium"
                if metal_database is not None
                else "homogeneous-hydrogen-helium"
                if homogeneous_mixture
                else "pure-helium"
            ),
            "metal_abundances": (
                dict(metal_abundances) if metal_abundances is not None else {}
            ),
            "metal_electron_feedback": (
                "charge-neutral EOS and continuum opacity"
                if metal_database is not None else "disabled"
            ),
            "radiative_equilibrium_includes_metal_lines": bool(
                metal_database is not None and include_metal_lines
            ),
            "radiative_equilibrium_includes_metal_bound_free": bool(
                metal_photoionization_database is not None
                or metal_topbase_photoionization_database is not None
            ),
            "radiative_equilibrium_level_resolved_metal_bound_free": (
                metal_topbase_photoionization_database.source
                if metal_topbase_photoionization_database is not None
                else "disabled"
            ),
            "radiative_equilibrium_mg_ii_he_unified_profile": (
                mg_ii_he_profile_table.source
                if mg_ii_he_profile_table is not None else "disabled"
            ),
            "radiative_equilibrium_mg_i_he_unified_profile": (
                mg_he_red_wing_table.source
                if mg_he_red_wing_table is not None else "disabled"
            ),
            "radiative_equilibrium_ca_i_he_unified_profile": (
                ca_i_he_profile_table.source
                if ca_i_he_profile_table is not None else "disabled"
            ),
            "radiative_equilibrium_ca_ii_he_unified_profile": (
                ca_ii_he_profile_table.source
                if ca_ii_he_profile_table is not None else "disabled"
            ),
            "radiative_equilibrium_minimum_metal_oscillator_strength": float(
                minimum_metal_oscillator_strength
            ),
            "radiative_equilibrium_maximum_metal_lines": maximum_metal_lines,
            "radiative_equilibrium_wing_sampled_metal_lines": int(
                metal_wing_sampled_lines
            ),
            "radiative_equilibrium_metal_line_selection": (
                "abundance-gf-boltzmann-ion-fraction-Planck-flux at Teff"
            ),
            "log_hydrogen_abundance": log_hydrogen_abundance,
            "log_hydrogen_to_helium": (
                float(log_hydrogen_abundance)
                if homogeneous_mixture else None
            ),
            "checkpoint_restart_seed": bool(restart_seed_requested),
            "eos": (
                "q-mhd-hydrogen-helium-occupation-probability"
                if homogeneous_mixture and correlated_microfields
                else "hummer-mihalas-hydrogen-helium-occupation-probability"
                if homogeneous_mixture
                else "q-mhd-helium-occupation-probability"
                if correlated_microfields
                else "hummer-mihalas-helium-occupation-probability"
            ),
            "hydrogen_neutral_radius_scale": (
                atmosphere.hydrogen_lte_state.neutral_radius_scale
                if atmosphere.hydrogen_lte_state is not None else None
            ),
            "helium_neutral_radius_scale": float(neutral_radius_scale),
            "rosseland_opacity_cm2_g": rosseland,
            "radiative_equilibrium_iterations": iteration,
            "radiative_equilibrium_selected_iteration": int(
                selected_iteration
            ),
            "radiative_equilibrium_converged": converged,
            "radiative_equilibrium_maximum_log_temperature_correction": maximum_correction,
            "radiative_equilibrium_maximum_convective_gradient_residual": float(
                maximum_convective_gradient_residual
            ),
            "radiative_equilibrium_maximum_convective_gradient_residual_depth_index": int(
                maximum_convective_gradient_residual_depth_index
            ),
            "radiative_equilibrium_convective_gradient_tolerance": float(
                convective_gradient_tolerance
            ),
            "radiative_equilibrium_maximum_total_flux_residual": float(
                maximum_total_flux_residual
            ),
            "radiative_equilibrium_convective_flux_tolerance": float(
                convective_flux_tolerance
            ),
            "radiative_equilibrium_maximum_correction_depth_index": maximum_correction_depth_index,
            "radiative_equilibrium_flux_ratio": flux_ratio,
            "radiative_equilibrium_wavelength_points": int(wavelength.size),
            "radiative_equilibrium_temperature_correction_damping": float(
                temperature_correction_damping
            ),
            "radiative_equilibrium_absorption_temperature_derivative": bool(
                include_absorption_temperature_derivative
            ),
            "radiative_equilibrium_required_consecutive_convergence_iterations": int(
                consecutive_convergence_iterations
            ),
            "radiative_equilibrium_final_convergence_streak": int(
                convergence_streak
            ),
            "radiative_equilibrium_flux_ratio_history": tuple(
                flux_ratio_history
            ),
            "radiative_equilibrium_maximum_log_temperature_correction_history": tuple(
                maximum_correction_history
            ),
            "radiative_equilibrium_maximum_convective_gradient_residual_history": tuple(
                maximum_convective_gradient_residual_history
            ),
            "radiative_equilibrium_maximum_convective_gradient_residual_depth_index_history": tuple(
                maximum_convective_gradient_residual_depth_index_history
            ),
            "radiative_equilibrium_maximum_total_flux_residual_history": tuple(
                maximum_total_flux_residual_history
            ),
            "radiative_equilibrium_effective_convective_correction_damping_history": tuple(
                effective_convective_correction_damping_history
            ),
            "radiative_equilibrium_global_correction_depth_weight": "uniform",
            "radiative_equilibrium_includes_helium_lines": bool(include_lines),
            "radiative_equilibrium_hydrogen_self_broadening": bool(
                include_trace_hydrogen_lines
                and include_hydrogen_self_broadening
            ),
            "radiative_equilibrium_hydrogen_neutral_helium_broadening": bool(
                include_trace_hydrogen_lines
                and include_hydrogen_neutral_helium_broadening
            ),
            "radiative_equilibrium_hydrogen_self_broadening_quadrature_order": int(
                hydrogen_self_broadening_quadrature_order
            ),
            "radiative_equilibrium_balmer_profile_edge_optical_depth": (
                1.0e-4 if include_trace_hydrogen_lines else None
            ),
            "radiative_equilibrium_hydrogen_self_broadening_prescription": (
                hydrogen_self_broadening_prescription
            ),
            "radiative_equilibrium_hydrogen_self_broadening_truncation_closure": (
                hydrogen_self_broadening_truncation_closure
            ),
            "radiative_equilibrium_hydrogen_series_pseudocontinuum": bool(
                include_trace_hydrogen_lines
                and include_hydrogen_series_pseudocontinuum
            ),
            "radiative_equilibrium_includes_allard_lyman_profiles": bool(
                include_trace_hydrogen_lines and unified_allard_table is not None
            ),
            "radiative_equilibrium_allard_stark_weight": float(
                allard_stark_weight
            ),
            "radiative_equilibrium_neutral_helium_line_broadening": (
                neutral_line_broadening if include_lines else "disabled"
            ),
            "helium_stark_profiles": "Tremblay-2026/Beauchamp-2025 explicit table",
            "helium_i_ground_resonance_broadening": (
                "Dimitrijevic-Sahal-Brechot-1989 electron/He-II impact"
            ),
            "helium_ii_profiles": (
                "Schönning-Butler/SYNSPEC table"
                if include_lines
                and include_helium_ii_lines
                and helium_ii_stark_table is not None
                else "hydrogenic Z^-5 transform of unified hydrogen tables"
                if include_lines and include_helium_ii_lines
                else "disabled"
            ),
            "helium_dimer_ion_continuum": bool(include_helium_dimer_ion),
            "helium_three_body_collision_induced_absorption": bool(
                include_helium_three_body_cia
            ),
            "helium_rydberg_bound_free": bool(include_rydberg_bound_free),
            "helium_minus_free_free": (
                "John-1994 inside tabulated T/lambda domain; "
                "Carbon-1969 fit to John-1968 elsewhere"
            ),
            "convection": "ML2-Bergeron-1992" if mixing_length_alpha is not None else "none",
            "mixing_length_alpha": mixing_length_alpha,
            "convective_correction_damping": float(
                convective_correction_damping
            ),
            "maximum_convective_flux_fraction": float(
                np.max(required_convective_flux[convective]) / target_flux
                if np.any(convective) else 0.0
            ),
            "convective_depth_points": int(np.count_nonzero(convective)),
        },
        hydrogen_lte_state=atmosphere.hydrogen_lte_state,
        helium_lte_state=atmosphere.helium_lte_state,
    )


def radiative_equilibrium_hydrogen_helium_atmosphere(
    effective_temperature: float,
    logg: float,
    log_hydrogen_to_helium: float,
    *,
    stark_table: object,
    **kwargs: object,
) -> Atmosphere:
    """Relax a homogeneous warm H/He atmosphere to non-gray equilibrium.

    This named entry point makes the composition convention explicit while
    sharing the mature DB atmosphere iteration.  Keyword options are the same
    as :func:`radiative_equilibrium_helium_atmosphere`.
    """

    if "log_hydrogen_abundance" in kwargs:
        raise TypeError(
            "pass log_hydrogen_to_helium positionally, not "
            "log_hydrogen_abundance"
        )
    kwargs.setdefault("include_hydrogen_series_pseudocontinuum", True)
    # He II contributes less than 1e-12 of the local heating throughout the
    # tested 10 kK mixed structure, yet its tabulated profiles are expensive.
    # Retain it automatically once He ionization becomes structurally relevant.
    kwargs.setdefault("include_helium_ii_lines", effective_temperature >= 15_000.0)
    # A matched 10--30 kK mixed-composition control grid gives structures that
    # agree with the full finite-difference opacity derivative to better than
    # 1.3e-4 pointwise in temperature.  Avoiding the second full line-opacity
    # evaluation approximately halves every mixed-atmosphere iteration.
    kwargs.setdefault("include_absorption_temperature_derivative", False)
    return radiative_equilibrium_helium_atmosphere(
        effective_temperature,
        logg,
        stark_table=stark_table,
        log_hydrogen_abundance=log_hydrogen_to_helium,
        **kwargs,
    )
