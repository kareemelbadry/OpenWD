"""Composition-independent adaptive LTE atmosphere-structure solver.

The hydrogen and helium modules supply their own EOS and opacity closures;
this module owns the conservative radiative/convective flux residual, the
surface-temperature plus pressure-gradient variables, and the safeguarded
trust-region iteration.  Keeping those numerical choices here prevents each
spectral-type module from accumulating a separate collection of damping and
restart heuristics.
"""

from __future__ import annotations

from typing import Callable, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from ._compat import trapezoid
from .atmosphere import Atmosphere, _upper_interface_values_on_nodes
from .constants import (
    BOLTZMANN,
    LIGHT_SPEED,
    PLANCK,
    STEFAN_BOLTZMANN,
)
from .convection import (
    ml2_convective_flux_gradient_derivative_from_thermodynamics,
    ml2_convective_flux_for_gradient_from_thermodynamics,
    ml2_temperature_gradient_for_total_flux_from_thermodynamics,
)
from .nonlinear import (
    NonlinearEvaluation,
    nonlinear_result_metadata,
    solve_trust_region_newton,
)
from .opacity import optical_depth_from_mass_opacity
from .radiative_transfer import (
    emergent_flux,
    feautrier_radiation_field,
    integrated_emergent_flux_state_response,
    integrated_feautrier_interface_state_response,
)
from .spectrum import planck_lambda_angstrom


FloatArray = NDArray[np.float64]
IterationCallback = Callable[[int, Atmosphere, Mapping[str, object]], None]


def rosseland_mean_from_opacity_grid(
    wavelength_angstrom: FloatArray,
    extinction: FloatArray,
    temperature: FloatArray,
) -> FloatArray:
    """Evaluate a Rosseland mean on an atmosphere's opacity-sampling grid.

    This is used when an atmosphere contains important opacity that is not
    part of a host-only analytic continuum mean, notably metal bound-free and
    bound-bound opacity in a DZ.  The supplied grid must already resolve the
    relevant edges and sample the selected structural lines.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    opacity = np.asarray(extinction, dtype=np.float64)
    local_temperature = np.asarray(temperature, dtype=np.float64)
    expected = (wavelength.size, local_temperature.size)
    if wavelength.ndim != 1 or local_temperature.ndim != 1:
        raise ValueError("wavelength and temperature must be one dimensional")
    if opacity.shape != expected:
        raise ValueError("extinction must have shape (wavelength, depth)")
    if (
        np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
        or np.any(~np.isfinite(local_temperature))
        or np.any(local_temperature <= 0.0)
        or np.any(~np.isfinite(opacity))
        or np.any(opacity < 0.0)
    ):
        raise ValueError("Rosseland-mean inputs must be finite and physical")
    wavelength_cm = wavelength[:, np.newaxis] * 1.0e-8
    exponent = PLANCK * LIGHT_SPEED / (
        wavelength_cm * BOLTZMANN * local_temperature[np.newaxis, :]
    )
    exp_negative = np.exp(-np.clip(exponent, 0.0, 745.0))
    denominator = np.maximum(
        1.0 - exp_negative, np.finfo(np.float64).tiny
    )
    d_planck_d_temperature = (
        2.0 * PLANCK * LIGHT_SPEED**2 / wavelength_cm**5 * 1.0e-8
        * exp_negative / denominator**2
        * exponent / local_temperature[np.newaxis, :]
    )
    weight_integral = trapezoid(
        d_planck_d_temperature, wavelength, axis=0
    )
    inverse_mean = trapezoid(
        d_planck_d_temperature
        / np.maximum(opacity, np.finfo(np.float64).tiny),
        wavelength,
        axis=0,
    )
    return weight_integral / np.maximum(
        inverse_mean, np.finfo(np.float64).tiny
    )


def _positive_interface_values(values: FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    interface = np.empty_like(array)
    interface[0] = array[0]
    interface[1:] = np.sqrt(array[:-1] * array[1:])
    return interface


def _arithmetic_interface_values(values: FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    interface = np.empty_like(array)
    interface[0] = array[0]
    interface[1:] = 0.5 * (array[:-1] + array[1:])
    return interface


def solve_adaptive_lte_structure(
    seed: Atmosphere,
    wavelength: FloatArray,
    *,
    with_temperature: Callable[[FloatArray], Atmosphere],
    true_absorption: Callable[[Atmosphere], FloatArray],
    scattering_opacity: Callable[[Atmosphere], FloatArray],
    rosseland_opacity: Callable[[Atmosphere], FloatArray],
    thermodynamics: Callable[[Atmosphere], object],
    mixing_length_alpha: float | None,
    max_iterations: int,
    temperature_tolerance: float,
    flux_tolerance: float,
    n_angle: int,
    initial_temperature_was_supplied: bool,
    resume_supplied_structure_in_formal_flux_phase: bool | None = None,
    temperature_normalization_anchor_rosseland_depth: float | None = None,
    maximum_formal_flux_rosseland_depth: float | None = None,
    formal_flux_taper_start_rosseland_depth: float | None = None,
    project_initial_convective_gradient: bool = True,
    initial_convective_gradient_projection_mode: Literal[
        "interface-transport", "unstable-node-gradient"
    ] = "interface-transport",
    use_convective_gradient_preconditioner: bool = True,
    maximum_convective_preconditioner_iterations: int | None = None,
    preconditioner_stationary_completion_iterations: int | None = 2,
    maximum_formal_flux_continuations: int = 2,
    safeguard_surface_flux: bool = False,
    iteration_callback: IterationCallback | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Atmosphere:
    """Solve one LTE atmosphere with adaptive Newton/ML2 transport.

    The nonlinear variables are one normalization value of ``ln(T)`` and one
    ``dln(T)/dln(P)`` value per interface.  The normalization is at the
    surface by default.  A composition module may instead anchor it at a
    specified Rosseland depth when an optically thin surface must not be
    driven by the homologous/global part of an initial Newton step.  In
    convective cells a local ML2 gradient equation first preconditions the
    nearly adiabatic structure; a second phase then imposes the conservative
    formal-transfer total flux at every interface.  Composition enters only
    through the callbacks. ``initial_temperature_was_supplied`` records seed
    provenance; the separate resume option decides whether that seed is an
    exact checkpoint that may bypass conditioning. The projection mode and
    stationary-completion setting preserve validated composition-specific
    initialization policies without duplicating the nonlinear solver.
    """

    wavelength = np.asarray(wavelength, dtype=np.float64)
    if wavelength.ndim != 1 or wavelength.size < 2:
        raise ValueError("wavelength must be a one-dimensional grid")
    if np.any(~np.isfinite(wavelength)) or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength must be finite and strictly increasing")

    target_flux = STEFAN_BOLTZMANN * seed.effective_temperature**4
    log_pressure = np.log(seed.gas_pressure)
    log_pressure_step = np.diff(log_pressure)
    if np.any(~np.isfinite(log_pressure_step)) or np.any(log_pressure_step <= 0.0):
        raise ValueError("seed pressure must be finite and increase inward")
    if temperature_normalization_anchor_rosseland_depth is not None and (
        not np.isfinite(temperature_normalization_anchor_rosseland_depth)
        or temperature_normalization_anchor_rosseland_depth <= 0.0
    ):
        raise ValueError(
            "temperature normalization anchor depth must be positive or None"
        )
    if maximum_formal_flux_rosseland_depth is not None and (
        not np.isfinite(maximum_formal_flux_rosseland_depth)
        or maximum_formal_flux_rosseland_depth <= 0.0
    ):
        raise ValueError(
            "maximum formal-flux Rosseland depth must be positive or None"
        )
    if formal_flux_taper_start_rosseland_depth is not None:
        if maximum_formal_flux_rosseland_depth is None:
            raise ValueError(
                "a formal-flux taper requires a maximum formal-flux depth"
            )
        if (
            not np.isfinite(formal_flux_taper_start_rosseland_depth)
            or formal_flux_taper_start_rosseland_depth <= 0.0
            or formal_flux_taper_start_rosseland_depth
            >= maximum_formal_flux_rosseland_depth
        ):
            raise ValueError(
                "formal-flux taper start must be positive and shallower than "
                "the maximum formal-flux depth"
            )
    if maximum_formal_flux_continuations < 0:
        raise ValueError(
            "maximum formal-flux continuations must be non-negative"
        )
    if not isinstance(initial_temperature_was_supplied, bool):
        raise ValueError("initial_temperature_was_supplied must be boolean")
    if (
        resume_supplied_structure_in_formal_flux_phase is not None
        and not isinstance(
            resume_supplied_structure_in_formal_flux_phase, bool
        )
    ):
        raise ValueError(
            "resume_supplied_structure_in_formal_flux_phase must be boolean "
            "or None"
        )
    if initial_convective_gradient_projection_mode not in (
        "interface-transport",
        "unstable-node-gradient",
    ):
        raise ValueError(
            "initial convective-gradient projection mode must be "
            "'interface-transport' or 'unstable-node-gradient'"
        )
    if (
        maximum_convective_preconditioner_iterations is not None
        and maximum_convective_preconditioner_iterations < 1
    ):
        raise ValueError(
            "maximum convective-preconditioner iterations must be positive"
        )
    if (
        preconditioner_stationary_completion_iterations is not None
        and preconditioner_stationary_completion_iterations < 1
    ):
        raise ValueError(
            "preconditioner stationary-completion iterations must be "
            "positive or None"
        )
    preconditioner_iteration_limit = min(
        max_iterations,
        (
            min(20, max(8, seed.n_depth // 2))
            if maximum_convective_preconditioner_iterations is None
            else maximum_convective_preconditioner_iterations
        ),
    )
    requested_formal_flux_resume = (
        initial_temperature_was_supplied
        if resume_supplied_structure_in_formal_flux_phase is None
        else resume_supplied_structure_in_formal_flux_phase
    )
    if requested_formal_flux_resume and not initial_temperature_was_supplied:
        raise ValueError(
            "formal-flux resume requires a supplied initial temperature"
        )
    resume_in_formal_flux_phase = bool(
        use_convective_gradient_preconditioner
        and requested_formal_flux_resume
    )
    if resume_in_formal_flux_phase:
        # A complete supplied structure is already the warm start.  Repeating
        # the approximate local-gradient phase after every checkpoint or
        # interpolated-grid restart can consume the budget indefinitely;
        # resume directly with the conservative equation the atmosphere must
        # finally obey.  Fresh gray/continuum seeds still receive the local
        # convective preconditioner below.
        use_convective_gradient_preconditioner = False
    normalization_anchor_index = 0
    if temperature_normalization_anchor_rosseland_depth is not None:
        at_or_below_anchor = np.flatnonzero(
            seed.rosseland_optical_depth
            >= temperature_normalization_anchor_rosseland_depth
        )
        normalization_anchor_index = int(
            at_or_below_anchor[0]
            if at_or_below_anchor.size
            else seed.n_depth - 1
        )

    interface_gradient_operator = np.zeros(
        (seed.n_depth, seed.n_depth), dtype=np.float64
    )
    interface_depth = np.arange(1, seed.n_depth)
    interface_gradient_operator[
        interface_depth, interface_depth - 1
    ] = -1.0 / log_pressure_step
    interface_gradient_operator[
        interface_depth, interface_depth
    ] = 1.0 / log_pressure_step

    use_physical_flux_residual = not use_convective_gradient_preconditioner
    # A deep ML2-gradient residual is useful only as a nonlinear conditioner.
    # Keep it as a separate switch from ``use_physical_flux_residual`` so the
    # final phase can restore the exact discretized total-flux equation at
    # every interface.  In particular, a diffusion-based desired gradient is
    # not algebraically identical to a finite-grid formal-transfer flux at
    # large optical depth, even though the two approach the same continuum
    # limit.
    use_deep_gradient_conditioner = False
    solver_phase = (
        "convective-gradient-preconditioner"
        if use_convective_gradient_preconditioner
        else "formal-radiative-flux"
    )
    solver_iteration_offset = 0
    cached_no_jacobian_log_temperature: FloatArray | None = None
    cached_no_jacobian_solver_phase: str | None = None
    cached_no_jacobian_evaluation: (
        NonlinearEvaluation[dict[str, object]] | None
    ) = None
    cached_base_work: dict[str, object] | None = None

    def interface_atmosphere(current: Atmosphere) -> Atmosphere:
        return Atmosphere(
            effective_temperature=current.effective_temperature,
            logg=current.logg,
            rosseland_optical_depth=_positive_interface_values(
                current.rosseland_optical_depth
            ),
            column_mass=_positive_interface_values(current.column_mass),
            temperature=_positive_interface_values(current.temperature),
            gas_pressure=_positive_interface_values(current.gas_pressure),
            mass_density=_positive_interface_values(current.mass_density),
            neutral_h_density=_positive_interface_values(
                current.neutral_h_density
            ),
            proton_density=_positive_interface_values(current.proton_density),
            electron_density=_positive_interface_values(current.electron_density),
            metadata=current.metadata,
        )

    def convection_transport(
        current: Atmosphere,
        temperature_gradient: FloatArray,
        radiative_flux_interface: FloatArray | None = None,
    ) -> dict[str, FloatArray] | None:
        if mixing_length_alpha is None:
            return None
        rosseland = np.asarray(rosseland_opacity(current), dtype=np.float64)
        thermo = thermodynamics(current)
        current_interface = interface_atmosphere(current)
        rosseland_interface = _positive_interface_values(rosseland)
        heat_capacity_interface = _arithmetic_interface_values(
            np.asarray(thermo.specific_heat_constant_pressure, dtype=np.float64)
        )
        expansion_interface = _arithmetic_interface_values(
            np.asarray(thermo.density_temperature_derivative, dtype=np.float64)
        )
        adiabatic_gradient_interface = _arithmetic_interface_values(
            np.asarray(thermo.adiabatic_temperature_gradient, dtype=np.float64)
        )
        convective_flux_interface = (
            ml2_convective_flux_for_gradient_from_thermodynamics(
                current_interface,
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
            * current_interface.gravity
            * current_interface.temperature**4
            / (
                3.0
                * rosseland_interface
                * current_interface.gas_pressure
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
        desired_gradient = (
            ml2_temperature_gradient_for_total_flux_from_thermodynamics(
                current_interface,
                rosseland_interface,
                np.full(current.n_depth, target_flux),
                heat_capacity_interface,
                expansion_interface,
                adiabatic_gradient_interface,
                mixing_length_alpha=mixing_length_alpha,
                radiative_flux_coefficient=formal_radiative_coefficient,
            )
        )
        desired_convective_flux = (
            ml2_convective_flux_for_gradient_from_thermodynamics(
                current_interface,
                rosseland_interface,
                desired_gradient,
                heat_capacity_interface,
                expansion_interface,
                adiabatic_gradient_interface,
                mixing_length_alpha=mixing_length_alpha,
            )
        )
        actual_convective_flux_gradient_derivative = (
            ml2_convective_flux_gradient_derivative_from_thermodynamics(
                current_interface,
                rosseland_interface,
                temperature_gradient,
                heat_capacity_interface,
                expansion_interface,
                adiabatic_gradient_interface,
                mixing_length_alpha=mixing_length_alpha,
            )
        )
        convective_flux_gradient_derivative = (
            ml2_convective_flux_gradient_derivative_from_thermodynamics(
                current_interface,
                rosseland_interface,
                desired_gradient,
                heat_capacity_interface,
                expansion_interface,
                adiabatic_gradient_interface,
                mixing_length_alpha=mixing_length_alpha,
            )
        )
        # Convection cannot carry flux through the surface boundary, so the
        # first Jacobian row remains purely radiative.
        actual_convective_flux_gradient_derivative[0] = 0.0
        convective_flux_gradient_derivative[0] = 0.0
        return {
            "rosseland": rosseland,
            "adiabatic_gradient": adiabatic_gradient_interface,
            "convective_flux": convective_flux_interface,
            "desired_gradient": desired_gradient,
            "desired_convective_flux": desired_convective_flux,
            "formal_radiative_coefficient": formal_radiative_coefficient,
            "actual_convective_flux_gradient_derivative": (
                actual_convective_flux_gradient_derivative
            ),
            "convective_flux_gradient_derivative": (
                convective_flux_gradient_derivative
            ),
        }

    def evaluate_log_temperature(
        log_temperature: FloatArray, need_jacobian: bool
    ) -> NonlinearEvaluation[dict[str, object]]:
        nonlocal cached_no_jacobian_log_temperature
        nonlocal cached_no_jacobian_solver_phase
        nonlocal cached_no_jacobian_evaluation
        nonlocal cached_base_work
        # Warm-start selection and the nonlinear driver can request the same
        # residual several times in succession.  A structure residual is
        # deterministic at fixed ln(T) and solver phase, while its opacity and
        # transfer calculation is expensive.  Retain the most recent
        # residual-only evaluation.  When the nonlinear driver immediately
        # requests a Jacobian at that identical state, reuse the already
        # evaluated EOS, opacity, scattering source, transfer solution, and
        # convection closure; only the hotter tangent state and exact response
        # remain to be built.
        same_as_cached_residual = (
            cached_no_jacobian_evaluation is not None
            and cached_no_jacobian_solver_phase == solver_phase
            and cached_no_jacobian_log_temperature is not None
            and np.array_equal(
                np.asarray(log_temperature),
                cached_no_jacobian_log_temperature,
            )
        )
        if not need_jacobian and same_as_cached_residual:
            return cached_no_jacobian_evaluation
        reuse_base = bool(
            need_jacobian
            and same_as_cached_residual
            and cached_base_work is not None
        )
        if reuse_base:
            assert cached_base_work is not None
            current_temperature = cached_base_work["temperature"]
            current = cached_base_work["atmosphere"]
            absorption = cached_base_work["absorption"]
            scattering = cached_base_work["scattering"]
            extinction = cached_base_work["extinction"]
            optical_depth = cached_base_work["optical_depth"]
            planck = cached_base_work["planck"]
            source = cached_base_work["source"]
            field = cached_base_work["field"]
            source_relative_residual = cached_base_work[
                "source_relative_residual"
            ]
            source_worst_wavelength_index = cached_base_work[
                "source_worst_wavelength_index"
            ]
            source_worst_depth_index = cached_base_work[
                "source_worst_depth_index"
            ]
            radiative_flux_interface = cached_base_work[
                "radiative_flux_interface"
            ]
            temperature_gradient = cached_base_work["temperature_gradient"]
            convective_flux = cached_base_work["convective_flux"]
            convective_flux_interface = cached_base_work[
                "convective_flux_interface"
            ]
            transport = cached_base_work["transport"]
        else:
            current_temperature = np.exp(log_temperature)
            current = with_temperature(current_temperature)
            absorption = np.asarray(true_absorption(current), dtype=np.float64)
            scattering = np.asarray(
                scattering_opacity(current), dtype=np.float64
            )
            expected = (wavelength.size, current.n_depth)
            if absorption.shape != expected or scattering.shape != expected:
                raise ValueError(
                    "opacity callbacks must return (wavelength, depth)"
                )
            extinction = absorption + scattering
            extinction = np.maximum(extinction, np.finfo(np.float64).tiny)
            optical_depth = optical_depth_from_mass_opacity(
                current.column_mass, extinction
            )
            planck = planck_lambda_angstrom(
                wavelength[:, np.newaxis], current_temperature[np.newaxis, :]
            )
            source = planck.copy()
            for _ in range(4):
                field = feautrier_radiation_field(
                    optical_depth, source, n_angle=n_angle
                )
                source = (
                    absorption * planck + scattering * field.mean_intensity
                ) / extinction
            field = feautrier_radiation_field(
                optical_depth, source, n_angle=n_angle
            )
            source_fixed_point = (
                absorption * planck + scattering * field.mean_intensity
            ) / extinction
            source_relative_residual = np.abs(
                source_fixed_point - source
            ) / np.maximum(
                np.maximum(np.abs(source_fixed_point), np.abs(source)),
                np.finfo(np.float64).tiny,
            )
            source_worst_flat_index = int(np.argmax(source_relative_residual))
            source_worst_wavelength_index, source_worst_depth_index = (
                np.unravel_index(source_worst_flat_index, source.shape)
            )
            if field.interface_flux is None:  # pragma: no cover - API invariant
                raise RuntimeError(
                    "Feautrier solver did not return interface fluxes"
                )
            radiative_flux_interface = trapezoid(
                field.interface_flux, wavelength, axis=0
            )

            temperature_gradient = np.empty_like(log_temperature)
            temperature_gradient[0] = 0.0
            temperature_gradient[1:] = (
                np.diff(log_temperature) / log_pressure_step
            )
            convective_flux = np.zeros_like(current_temperature)
            convective_flux_interface = np.zeros_like(current_temperature)
            transport = convection_transport(
                current, temperature_gradient, radiative_flux_interface
            )
            if transport is not None:
                convective_flux_interface = transport["convective_flux"]
                convective_flux = _upper_interface_values_on_nodes(
                    convective_flux_interface
                )

        total_flux_interface = (
            radiative_flux_interface + convective_flux_interface
        )
        flux_residual = total_flux_interface / target_flux - 1.0
        residual = flux_residual.copy()
        transport_gradient_scale = None
        gradient_preconditioned = np.zeros(seed.n_depth, dtype=bool)
        physical_flux_weight = np.ones(seed.n_depth, dtype=np.float64)
        if transport is not None:
            desired_gradient = transport["desired_gradient"]
            adiabatic_gradient = transport["adiabatic_gradient"]
            transport_gradient_scale = np.maximum(
                desired_gradient, adiabatic_gradient
            )
            gradient_preconditioned = (
                transport["desired_convective_flux"] > 0.0
            )
            if not use_physical_flux_residual:
                physical_flux_weight[gradient_preconditioned] = 0.0
            else:
                if (
                    not use_deep_gradient_conditioner
                    or maximum_formal_flux_rosseland_depth is None
                ):
                    gradient_preconditioned[:] = False
                elif formal_flux_taper_start_rosseland_depth is not None:
                    candidate = gradient_preconditioned.copy()
                    physical_flux_weight[candidate] = np.clip(
                        np.log(
                            maximum_formal_flux_rosseland_depth
                            / np.maximum(
                                seed.rosseland_optical_depth[candidate],
                                np.finfo(np.float64).tiny,
                            )
                        )
                        / np.log(
                            maximum_formal_flux_rosseland_depth
                            / formal_flux_taper_start_rosseland_depth
                        ),
                        0.0,
                        1.0,
                    )
                    gradient_preconditioned &= physical_flux_weight < 1.0
                else:
                    gradient_preconditioned &= (
                        seed.rosseland_optical_depth
                        > maximum_formal_flux_rosseland_depth
                    )
                    physical_flux_weight[gradient_preconditioned] = 0.0
            gradient_preconditioned[0] = False
            gradient_residual = (
                temperature_gradient[gradient_preconditioned]
                - desired_gradient[gradient_preconditioned]
            ) / transport_gradient_scale[gradient_preconditioned]
            residual[gradient_preconditioned] = (
                physical_flux_weight[gradient_preconditioned]
                * flux_residual[gradient_preconditioned]
                + (1.0 - physical_flux_weight[gradient_preconditioned])
                * gradient_residual
            )

        jacobian = None
        if need_jacobian:
            logarithmic_step = 2.0e-4
            hotter_temperature = current_temperature * np.exp(logarithmic_step)
            hotter_planck = planck_lambda_angstrom(
                wavelength[:, np.newaxis], hotter_temperature[np.newaxis, :]
            )
            planck_derivative = (hotter_planck - planck) / logarithmic_step
            hotter = with_temperature(hotter_temperature)
            hotter_absorption = np.asarray(
                true_absorption(hotter), dtype=np.float64
            )
            hotter_scattering = np.asarray(
                scattering_opacity(hotter), dtype=np.float64
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
                    current.column_mass,
                    extinction_derivative,
                    n_angle=n_angle,
                )
            )
            jacobian = radiative_flux_jacobian / target_flux
            if (
                transport is not None
                and use_physical_flux_residual
            ):
                # ML2 supplies the dominant local response to the temperature
                # gradient, inexpensively enough for every solver phase.
                jacobian += (
                    transport[
                        "actual_convective_flux_gradient_derivative"
                    ][:, np.newaxis]
                    * interface_gradient_operator
                    / target_flux
                )
            if (
                transport is not None
                and transport_gradient_scale is not None
                and np.any(gradient_preconditioned)
            ):
                formal_coefficient = transport["formal_radiative_coefficient"]
                convective_derivative = transport[
                    "convective_flux_gradient_derivative"
                ]
                safe_gradient = np.maximum(temperature_gradient, 1.0e-8)
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
                    - radiative_flux_interface[formal_branch, np.newaxis]
                    * interface_gradient_operator[formal_branch]
                ) / safe_gradient[formal_branch, np.newaxis] ** 2
                desired_gradient_derivative = (
                    -transport["desired_gradient"][:, np.newaxis]
                    * formal_coefficient_derivative
                    / (
                        formal_coefficient + convective_derivative
                    )[:, np.newaxis]
                )
                gradient_jacobian = (
                    interface_gradient_operator[gradient_preconditioned]
                    - desired_gradient_derivative[gradient_preconditioned]
                ) / transport_gradient_scale[
                    gradient_preconditioned, np.newaxis
                ]
                jacobian[gradient_preconditioned] = (
                    physical_flux_weight[
                        gradient_preconditioned, np.newaxis
                    ]
                    * jacobian[gradient_preconditioned]
                    + (
                        1.0
                        - physical_flux_weight[
                            gradient_preconditioned, np.newaxis
                        ]
                    )
                    * gradient_jacobian
                )

        physical_flux_monitored = np.ones(seed.n_depth, dtype=bool)
        if (
            use_deep_gradient_conditioner
            and maximum_formal_flux_rosseland_depth is not None
        ):
            if formal_flux_taper_start_rosseland_depth is None:
                physical_flux_monitored = (
                    seed.rosseland_optical_depth
                    <= maximum_formal_flux_rosseland_depth
                )
            else:
                physical_flux_monitored = physical_flux_weight >= 0.5

        payload: dict[str, object] = {
            "atmosphere": current,
            "radiative_flux_interface": radiative_flux_interface,
            "convective_flux": convective_flux,
            "convective_flux_interface": convective_flux_interface,
            "total_flux_interface": total_flux_interface,
            "convection_transport": transport,
            "temperature_gradient": temperature_gradient,
            "physical_flux_monitored": physical_flux_monitored,
            "physical_flux_residual_weight": physical_flux_weight,
            "scattering_source_iterations": 4,
            "scattering_source_maximum_relative_residual": float(
                np.max(source_relative_residual)
            ),
            "scattering_source_worst_wavelength_index": int(
                source_worst_wavelength_index
            ),
            "scattering_source_worst_depth_index": int(
                source_worst_depth_index
            ),
        }
        evaluation = NonlinearEvaluation(residual, jacobian, payload)
        if not need_jacobian:
            cached_no_jacobian_log_temperature = np.asarray(
                log_temperature, dtype=np.float64
            ).copy()
            cached_no_jacobian_solver_phase = solver_phase
            cached_no_jacobian_evaluation = evaluation
            cached_base_work = {
                "temperature": current_temperature,
                "atmosphere": current,
                "absorption": absorption,
                "scattering": scattering,
                "extinction": extinction,
                "optical_depth": optical_depth,
                "planck": planck,
                "source": source,
                "field": field,
                "source_relative_residual": source_relative_residual,
                "source_worst_wavelength_index": (
                    source_worst_wavelength_index
                ),
                "source_worst_depth_index": source_worst_depth_index,
                "radiative_flux_interface": radiative_flux_interface,
                "temperature_gradient": temperature_gradient,
                "convective_flux": convective_flux,
                "convective_flux_interface": convective_flux_interface,
                "transport": transport,
            }
        return evaluation

    log_temperature_from_state = np.zeros(
        (seed.n_depth, seed.n_depth), dtype=np.float64
    )
    log_temperature_from_state[:, 0] = 1.0
    for interface in range(1, seed.n_depth):
        pressure_step = log_pressure_step[interface - 1]
        if interface <= normalization_anchor_index:
            log_temperature_from_state[:interface, interface] = -pressure_step
        else:
            log_temperature_from_state[interface:, interface] = pressure_step

    def state_from_log_temperature(log_temperature: FloatArray) -> FloatArray:
        state = np.empty_like(log_temperature)
        state[0] = log_temperature[normalization_anchor_index]
        state[1:] = np.diff(log_temperature) / log_pressure_step
        return state

    def evaluate_state(
        state: FloatArray, need_jacobian: bool
    ) -> NonlinearEvaluation[dict[str, object]]:
        evaluation = evaluate_log_temperature(
            log_temperature_from_state @ state, need_jacobian
        )
        jacobian = evaluation.jacobian
        if jacobian is not None:
            jacobian = jacobian @ log_temperature_from_state
        return NonlinearEvaluation(
            evaluation.residual, jacobian, evaluation.payload
        )

    def report_iteration(record, state, evaluation) -> None:
        if iteration_callback is None:
            return
        payload = evaluation.payload
        current = payload["atmosphere"]
        total = np.asarray(payload["total_flux_interface"], dtype=np.float64)
        radiative = np.asarray(
            payload["radiative_flux_interface"], dtype=np.float64
        )
        convective = np.asarray(
            payload["convective_flux_interface"], dtype=np.float64
        )
        flux_residual = total / target_flux - 1.0
        monitored = np.asarray(
            payload["physical_flux_monitored"], dtype=bool
        )
        maximum_depth = int(np.argmax(np.abs(flux_residual)))
        transport = payload["convection_transport"]
        desired_gradient = (
            np.asarray(transport["desired_gradient"], dtype=np.float64)
            if transport is not None
            else np.full(seed.n_depth, np.nan)
        )
        actual_gradient = np.asarray(
            payload["temperature_gradient"], dtype=np.float64
        )
        iteration_callback(
            solver_iteration_offset + record.iteration,
            current,
            {
                "solver_residual_rms": record.residual_rms,
                "solver_residual_maximum": record.residual_maximum,
                "flux_ratio": float(total[0] / target_flux),
                "maximum_log_temperature_correction": record.maximum_step,
                "maximum_correction_depth_index": maximum_depth,
                "maximum_total_flux_residual": float(
                    np.max(np.abs(flux_residual))
                ),
                "maximum_formal_region_total_flux_residual": float(
                    np.max(np.abs(flux_residual[monitored]))
                ),
                "maximum_all_depth_total_flux_residual": float(
                    np.max(np.abs(flux_residual))
                ),
                "bottom_radiative_flux_ratio": float(
                    radiative[-1] / target_flux
                ),
                "bottom_convective_flux_ratio": float(
                    convective[-1] / target_flux
                ),
                "maximum_flux_depth_temperature_gradient": float(
                    actual_gradient[maximum_depth]
                ),
                "maximum_flux_depth_desired_gradient": float(
                    desired_gradient[maximum_depth]
                ),
                "trust_radius": record.trust_radius,
                "line_search_factor": record.line_search_factor,
                "jacobian_recomputed": record.jacobian_recomputed,
                "scattering_source_iterations": int(
                    payload["scattering_source_iterations"]
                ),
                "scattering_source_maximum_relative_residual": float(
                    payload["scattering_source_maximum_relative_residual"]
                ),
                "solver_phase": solver_phase,
                "converged": bool(
                    record.residual_maximum < flux_tolerance
                    and record.maximum_step < temperature_tolerance
                    and np.max(np.abs(flux_residual)) < flux_tolerance
                ),
            },
        )

    def physical_flux_converged(state, evaluation, maximum_step) -> bool:
        total = np.asarray(
            evaluation.payload["total_flux_interface"], dtype=np.float64
        )
        return bool(
            np.max(np.abs(total / target_flux - 1.0)) < flux_tolerance
        )

    def phase_converged(state, evaluation, maximum_step) -> bool:
        if not use_physical_flux_residual or use_deep_gradient_conditioner:
            return True
        return physical_flux_converged(state, evaluation, maximum_step)

    def surface_flux_error(evaluation) -> float:
        total = np.asarray(
            evaluation.payload["total_flux_interface"], dtype=np.float64
        )
        return float(abs(total[0] / target_flux - 1.0))

    def seed_merit(evaluation) -> float:
        residual = evaluation.residual
        return float(
            np.sqrt(np.mean(residual**2))
            + 0.25 * np.max(np.abs(residual))
        )

    initial_log_temperature = np.log(seed.temperature)
    initial_bolometric_temperature_scale = 1.0
    initial_bolometric_rescaling_evaluations = 0
    initial_convective_gradient_projection_damping = 0.0
    initial_convective_gradient_projection_rejected_infeasible = 0
    initial_convective_gradient_projection_attempted = bool(
        mixing_length_alpha is not None
        and project_initial_convective_gradient
        and not resume_in_formal_flux_phase
    )
    if initial_convective_gradient_projection_attempted:
        initial = with_temperature(seed.temperature)
        if (
            initial_convective_gradient_projection_mode
            == "unstable-node-gradient"
        ):
            # The original DA cold-start construction evaluated the ML2
            # proposal on depth nodes and changed only gradients that were
            # already Schwarzschild-unstable.  Keep that validated seed
            # policy as a composition option while the nonlinear equations,
            # transfer response, continuation, and convergence checks remain
            # owned by this shared driver.
            initial_gradient = np.gradient(
                initial_log_temperature, log_pressure, edge_order=2
            )
            initial_thermodynamics = thermodynamics(initial)
            initial_rosseland = np.asarray(
                rosseland_opacity(initial), dtype=np.float64
            )
            desired_gradient = (
                ml2_temperature_gradient_for_total_flux_from_thermodynamics(
                    initial,
                    initial_rosseland,
                    np.full(seed.n_depth, target_flux),
                    np.asarray(
                        initial_thermodynamics.specific_heat_constant_pressure,
                        dtype=np.float64,
                    ),
                    np.asarray(
                        initial_thermodynamics.density_temperature_derivative,
                        dtype=np.float64,
                    ),
                    np.asarray(
                        initial_thermodynamics.adiabatic_temperature_gradient,
                        dtype=np.float64,
                    ),
                    mixing_length_alpha=mixing_length_alpha,
                )
            )
            adiabatic_gradient = np.asarray(
                initial_thermodynamics.adiabatic_temperature_gradient,
                dtype=np.float64,
            )
            unstable = initial_gradient > adiabatic_gradient
            projected_gradient = np.where(
                unstable,
                np.minimum(initial_gradient, desired_gradient),
                initial_gradient,
            )
        else:
            initial_gradient = np.empty(seed.n_depth, dtype=np.float64)
            initial_gradient[0] = 0.0
            initial_gradient[1:] = (
                np.diff(initial_log_temperature) / log_pressure_step
            )
            transport = convection_transport(initial, initial_gradient)
            assert transport is not None
            desired_gradient = transport["desired_gradient"]
            adiabatic_gradient = transport["adiabatic_gradient"]
            unstable = initial_gradient > adiabatic_gradient
            convection_required = desired_gradient > adiabatic_gradient
            projected_gradient = np.where(
                unstable,
                np.minimum(initial_gradient, desired_gradient),
                np.where(
                    convection_required, desired_gradient, initial_gradient
                ),
            )
        ordinary_evaluation = evaluate_log_temperature(
            initial_log_temperature, False
        )
        ordinary_merit = seed_merit(ordinary_evaluation)
        # The local ML2 inversion is a proposal based on the seed EOS and
        # opacity.  Its full application can be too large, particularly when
        # the seed spans many pressure scale heights.  Backtrack this
        # provisional correction exactly as an ordinary nonlinear step, and
        # regard a nonphysical trial opacity grid as a rejected proposal
        # rather than allowing an optional warm start to abort the solve.
        for projection_damping in (
            1.0,
            0.5,
            0.25,
            0.125,
            0.0625,
            0.03125,
            0.015625,
        ):
            candidate_gradient = initial_gradient + projection_damping * (
                projected_gradient - initial_gradient
            )
            candidate_log_temperature = np.empty_like(
                initial_log_temperature
            )
            candidate_log_temperature[0] = initial_log_temperature[0]
            for depth in range(1, seed.n_depth):
                if (
                    initial_convective_gradient_projection_mode
                    == "unstable-node-gradient"
                ):
                    integrated_gradient = 0.5 * (
                        candidate_gradient[depth - 1]
                        + candidate_gradient[depth]
                    )
                else:
                    integrated_gradient = candidate_gradient[depth]
                candidate_log_temperature[depth] = (
                    candidate_log_temperature[depth - 1]
                    + integrated_gradient * log_pressure_step[depth - 1]
                )
            try:
                candidate_evaluation = evaluate_log_temperature(
                    candidate_log_temperature, False
                )
            except ValueError:
                initial_convective_gradient_projection_rejected_infeasible += 1
                continue
            projection_improves_surface = (
                not safeguard_surface_flux
                or surface_flux_error(candidate_evaluation)
                <= surface_flux_error(ordinary_evaluation) + 1.0e-12
            )
            if (
                seed_merit(candidate_evaluation) < ordinary_merit
                and projection_improves_surface
            ):
                initial_log_temperature = candidate_log_temperature
                initial_convective_gradient_projection_damping = (
                    projection_damping
                )
                break

    # A checkpoint interpolated in Teff/log(g), or relaxed at a neighboring
    # composition, can have a good dimensionless gradient but the wrong
    # homologous temperature normalization.  In that case an unrestricted
    # Newton solve wastes many opacity/transfer evaluations rediscovering the
    # gray F ~ T^4 scaling through small local steps.  Treat the gray scaling
    # only as a proposed warm start: evaluate the complete non-gray transfer
    # problem at the full correction and two damped corrections, and retain a
    # candidate only when it improves both the physical surface-flux error and
    # the merit of the complete current residual.  The latter guard is
    # essential for a convective restart: a homologous temperature change can
    # repair the surface flux while destroying an already accurate deep ML2
    # closure.  Thus the gray estimate never replaces or weakens the equations
    # used for convergence.
    initial_evaluation = evaluate_log_temperature(
        initial_log_temperature, False
    )
    for _ in range(6):
        if resume_in_formal_flux_phase:
            # The caller has identified this as a complete checkpoint rather
            # than a neighboring/interpolated seed.  Preserve its thermal
            # normalization and honor the direct formal-flux resume contract;
            # a homologous rescaling would turn the checkpoint back into a
            # fresh convective-gradient warm start.
            break
        initial_total_flux = np.asarray(
            initial_evaluation.payload["total_flux_interface"], dtype=np.float64
        )
        initial_flux_ratio = float(initial_total_flux[0] / target_flux)
        if (
            not np.isfinite(initial_flux_ratio)
            or initial_flux_ratio <= 0.0
            or abs(initial_flux_ratio - 1.0) < flux_tolerance
        ):
            break
        gray_log_correction = float(np.clip(
            -0.25 * np.log(initial_flux_ratio), -0.12, 0.12
        ))
        accepted = False
        for damping in (1.0, 0.5, 0.25):
            log_correction = damping * gray_log_correction
            if abs(log_correction) <= np.finfo(np.float64).eps:
                continue
            candidate_log_temperature = (
                initial_log_temperature + log_correction
            )
            candidate_evaluation = evaluate_log_temperature(
                candidate_log_temperature, False
            )
            initial_bolometric_rescaling_evaluations += 1
            if (
                surface_flux_error(candidate_evaluation)
                < surface_flux_error(initial_evaluation)
                and seed_merit(candidate_evaluation)
                < seed_merit(initial_evaluation)
            ):
                initial_log_temperature = candidate_log_temperature
                initial_evaluation = candidate_evaluation
                initial_bolometric_temperature_scale *= float(
                    np.exp(log_correction)
                )
                accepted = True
                break
        if not accepted:
            break

    restart_reconditioned_after_bolometric_rescaling = False
    if (
        resume_in_formal_flux_phase
        and mixing_length_alpha is not None
        and initial_bolometric_temperature_scale != 1.0
    ):
        # A supplied checkpoint can resume directly in the exact flux phase
        # only while its thermal normalization is retained.  If the guarded
        # formal-transfer check above accepts a homologous rescaling, the EOS,
        # opacity, and ML2 efficiency have all changed; the old convective
        # gradient is no longer a consistent warm start.  Re-run the same
        # bounded conditioner used for a fresh atmosphere automatically.
        use_convective_gradient_preconditioner = True
        use_physical_flux_residual = False
        use_deep_gradient_conditioner = False
        solver_phase = "convective-gradient-preconditioner"
        restart_reconditioned_after_bolometric_rescaling = True

    nonlinear_options = dict(
        maximum_iterations=max_iterations,
        residual_tolerance=flux_tolerance,
        step_tolerance=temperature_tolerance,
        initial_trust_radius=0.04,
        maximum_trust_radius=0.12,
        jacobian_refresh_interval=4,
        finite_difference_fallback_step=(
            5.0e-3 if seed.n_depth <= 12 else None
        ),
        callback=report_iteration,
        convergence_test=phase_converged,
        acceptance_test=(
            (
                lambda old, trial: surface_flux_error(trial)
                <= surface_flux_error(old) + 1.0e-12
            )
            if safeguard_surface_flux
            else None
        ),
        step_measure=lambda old_state, new_state: float(
            np.max(
                np.abs(
                    log_temperature_from_state @ (new_state - old_state)
                )
            )
        ),
    )
    initial_options = dict(nonlinear_options)
    if use_convective_gradient_preconditioner:
        # This phase solves a deliberately approximate local ML2-gradient
        # problem.  It is a warm-start construction, not a second atmosphere
        # solve, so spending the full production iteration budget here when
        # the local objective stalls only delays the conservative flux solve.
        initial_options["maximum_iterations"] = (
            preconditioner_iteration_limit
        )
        initial_options["stationary_completion_iterations"] = (
            preconditioner_stationary_completion_iterations
        )
    nonlinear_solver_segments: list[dict[str, object]] = []
    result = solve_trust_region_newton(
        state_from_log_temperature(initial_log_temperature),
        evaluate_state,
        **initial_options,
    )
    nonlinear_solver_segments.append(
        {
            "phase": solver_phase,
            **nonlinear_result_metadata(result),
        }
    )
    preconditioner_iterations = (
        result.iterations if use_convective_gradient_preconditioner else 0
    )
    conditioned_transport_iterations = 0
    formal_flux_continuations = 0
    if (
        use_convective_gradient_preconditioner
        and mixing_length_alpha is not None
    ):
        # An approximate conditioning phase can never be the final authority,
        # even when its incidental physical-flux residual is already below
        # tolerance.  Always enter the exact formal-flux phase at least once;
        # its initial-state check is cheap and supplies the authoritative
        # convergence status.  Previously a preconditioner that reached the
        # physical flux tolerance at its iteration cap skipped this phase and
        # was incorrectly returned as unconverged.
        use_physical_flux_residual = True
        solver_iteration_offset = preconditioner_iterations
        if (
            maximum_formal_flux_rosseland_depth is not None
            and not physical_flux_converged(
                result.state, result.evaluation, 0.0
            )
        ):
            # First impose the exact transfer flux through the radiative
            # surface and transition layers while retaining the local ML2
            # equation in the optically thick convective interior. This is
            # another bounded warm start: its diffusion-based deep residual
            # is not used to declare physical convergence.
            use_deep_gradient_conditioner = True
            solver_phase = "conditioned-transport-warm-start"
            conditioned_options = dict(nonlinear_options)
            conditioned_options["maximum_iterations"] = min(
                max_iterations, max(12, min(30, seed.n_depth))
            )
            conditioned_options["stationary_completion_iterations"] = 2
            result = solve_trust_region_newton(
                np.asarray(result.state, dtype=np.float64).copy(),
                evaluate_state,
                **conditioned_options,
            )
            nonlinear_solver_segments.append(
                {
                    "phase": solver_phase,
                    **nonlinear_result_metadata(result),
                }
            )
            conditioned_transport_iterations = result.iterations
            solver_iteration_offset += result.iterations

        # The converged equations are always the conservative formal-transfer
        # radiative flux plus the ML2 convective flux at every interface.
        # Explicitly turn off the deep residual replacement before the final
        # solve; otherwise a stationary diffusion warm start can masquerade
        # as a failed physical completion on a finite depth grid.
        use_deep_gradient_conditioner = False
        solver_phase = "formal-radiative-flux-completion"
        completion_state = np.asarray(result.state, dtype=np.float64).copy()
        result = solve_trust_region_newton(
            completion_state, evaluate_state, **nonlinear_options
        )
        nonlinear_solver_segments.append(
            {
                "phase": solver_phase,
                **nonlinear_result_metadata(result),
            }
        )

    # A trust-region collapse means that the accumulated Broyden history,
    # not the accepted atmosphere, has become unhelpful.  Rebuilding the
    # tangent and trust state from the last accepted atmosphere is a general
    # nonlinear continuation, equivalent to restarting a saved checkpoint
    # but without user intervention.  Apply it to every exact physical-flux
    # solve, including a composition-matched checkpoint that correctly
    # bypassed the provisional convective-gradient phase above.  Previously
    # that direct-resume path accidentally bypassed continuation as well.
    # Keep the number bounded so a genuinely inconsistent physical closure
    # still returns as unconverged rather than looping indefinitely.
    if use_physical_flux_residual and not use_deep_gradient_conditioner:
        for _ in range(maximum_formal_flux_continuations):
            if result.converged:
                break
            if result.iterations == 0:
                break
            solver_iteration_offset += result.iterations
            formal_flux_continuations += 1
            result = solve_trust_region_newton(
                result.state, evaluate_state, **nonlinear_options
            )
            nonlinear_solver_segments.append(
                {
                    "phase": (
                        "formal-radiative-flux-continuation-"
                        f"{formal_flux_continuations}"
                    ),
                    **nonlinear_result_metadata(result),
                }
            )

    total_solver_iterations = solver_iteration_offset + result.iterations
    payload = result.evaluation.payload
    final_atmosphere = payload["atmosphere"]
    assert isinstance(final_atmosphere, Atmosphere)
    total_flux_interface = np.asarray(
        payload["total_flux_interface"], dtype=np.float64
    )
    radiative_flux_interface = np.asarray(
        payload["radiative_flux_interface"], dtype=np.float64
    )
    convective_flux_interface = np.asarray(
        payload["convective_flux_interface"], dtype=np.float64
    )
    convective_flux = np.asarray(payload["convective_flux"], dtype=np.float64)
    final_transport = payload["convection_transport"]
    if final_transport is not None:
        # The ML2 residual already evaluated the exact Rosseland mean at this
        # accepted atmosphere.  Reuse it, especially when the final
        # evaluation also built a Jacobian and the composition callback's
        # one-entry opacity cache now contains the hotter tangent state.
        final_rosseland_opacity = np.asarray(
            final_transport["rosseland"], dtype=np.float64
        )
    else:
        final_rosseland_opacity = np.asarray(
            rosseland_opacity(final_atmosphere), dtype=np.float64
        )
    rosseland_depth = np.empty_like(final_atmosphere.column_mass)
    rosseland_depth[0] = (
        final_rosseland_opacity[0] * final_atmosphere.column_mass[0]
    )
    rosseland_depth[1:] = rosseland_depth[0] + np.cumsum(
        0.5
        * (final_rosseland_opacity[1:] + final_rosseland_opacity[:-1])
        * np.diff(final_atmosphere.column_mass)
    )
    final_residual = total_flux_interface / target_flux - 1.0
    final_step = (
        result.history[-1].maximum_step
        if result.history
        else (0.0 if result.converged else np.inf)
    )
    common_metadata: dict[str, object] = {
        "model": (
            "non-gray-radiative-convective-equilibrium"
            if mixing_length_alpha is not None
            else "non-gray-radiative-equilibrium"
        ),
        "rosseland_opacity_cm2_g": final_rosseland_opacity,
        "structure_solver": "adaptive-trust-region-newton",
        "adaptive_structure_driver": "shared-lte",
        "structure_residual": (
            "bounded ML2-gradient warm starts followed by conservative "
            "formal interface total flux at every depth"
        ),
        "structure_jacobian": (
            "opacity-aware tangent transfer plus analytic ML2 gradient "
            "response"
        ),
        "radiative_equilibrium_iterations": total_solver_iterations,
        "convective_preconditioner_iterations": preconditioner_iterations,
        "conditioned_transport_warm_start_iterations": (
            conditioned_transport_iterations
        ),
        "convective_preconditioner_iteration_limit": (
            preconditioner_iteration_limit
            if use_convective_gradient_preconditioner else 0
        ),
        "resumed_directly_in_formal_flux_phase": (
            resume_in_formal_flux_phase
            and not restart_reconditioned_after_bolometric_rescaling
        ),
        "restart_reconditioned_after_bolometric_rescaling": (
            restart_reconditioned_after_bolometric_rescaling
        ),
        "formal_flux_completion_used": bool(
            solver_phase == "formal-radiative-flux-completion"
        ),
        "formal_flux_continuations": formal_flux_continuations,
        "nonlinear_solver_terminal_reason": (
            result.diagnostics.terminal_reason
        ),
        "nonlinear_solver_final_worst_residual_depth_index": int(
            result.diagnostics.final_worst_residual_index
        ),
        "nonlinear_solver_final_worst_residual_rosseland_depth": float(
            rosseland_depth[result.diagnostics.final_worst_residual_index]
        ),
        "nonlinear_solver_final_worst_residual_temperature_K": float(
            final_atmosphere.temperature[
                result.diagnostics.final_worst_residual_index
            ]
        ),
        "nonlinear_solver_final_worst_residual_convective_flux_fraction": float(
            convective_flux_interface[
                result.diagnostics.final_worst_residual_index
            ]
            / target_flux
        ),
        "nonlinear_solver_residual_evaluations": int(
            sum(
                int(segment["residual_evaluations"])
                for segment in nonlinear_solver_segments
            )
        ),
        "nonlinear_solver_jacobian_evaluations": int(
            sum(
                int(segment["jacobian_evaluations"])
                for segment in nonlinear_solver_segments
            )
        ),
        "nonlinear_solver_accepted_iterations": int(
            sum(
                int(segment["accepted_iterations"])
                for segment in nonlinear_solver_segments
            )
        ),
        "nonlinear_solver_rejected_trial_evaluations": int(
            sum(
                int(segment["rejected_trial_evaluations"])
                for segment in nonlinear_solver_segments
            )
        ),
        "nonlinear_solver_rejected_directions": int(
            sum(
                int(segment["rejected_directions"])
                for segment in nonlinear_solver_segments
            )
        ),
        "nonlinear_solver_segments": tuple(nonlinear_solver_segments),
        "radiative_equilibrium_converged": bool(
            result.converged
            and np.max(np.abs(final_residual)) < flux_tolerance
        ),
        "radiative_equilibrium_maximum_log_temperature_correction": float(
            final_step
        ),
        "radiative_equilibrium_flux_ratio": float(
            total_flux_interface[0] / target_flux
        ),
        "radiative_equilibrium_wavelength_points": int(wavelength.size),
        "radiative_equilibrium_maximum_total_flux_residual": float(
            np.max(np.abs(final_residual))
        ),
        "maximum_total_flux_residual": float(
            np.max(np.abs(final_residual))
        ),
        "maximum_all_depth_total_flux_residual": float(
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
        "total_flux_fraction_by_interface": total_flux_interface / target_flux,
        "convection": (
            "ML2-Bergeron-1992" if mixing_length_alpha is not None else "none"
        ),
        "mixing_length_alpha": mixing_length_alpha,
        "initial_temperature_was_supplied": bool(
            initial_temperature_was_supplied
        ),
        "initial_convective_gradient_projection": bool(
            initial_convective_gradient_projection_attempted
        ),
        "initial_convective_gradient_projection_mode": (
            initial_convective_gradient_projection_mode
        ),
        "initial_convective_gradient_projection_damping": float(
            initial_convective_gradient_projection_damping
        ),
        "initial_convective_gradient_projection_rejected_infeasible": int(
            initial_convective_gradient_projection_rejected_infeasible
        ),
        "initial_bolometric_temperature_scale": float(
            initial_bolometric_temperature_scale
        ),
        "initial_bolometric_rescaling_evaluations": int(
            initial_bolometric_rescaling_evaluations
        ),
        "convective_gradient_preconditioner_used": bool(
            use_convective_gradient_preconditioner
        ),
        "maximum_formal_flux_continuations": int(
            maximum_formal_flux_continuations
        ),
        "temperature_normalization_anchor_depth_index": int(
            normalization_anchor_index
        ),
        "temperature_normalization_anchor_rosseland_depth": float(
            seed.rosseland_optical_depth[normalization_anchor_index]
        ),
        "maximum_formal_flux_rosseland_depth": (
            None
            if maximum_formal_flux_rosseland_depth is None
            else float(maximum_formal_flux_rosseland_depth)
        ),
        "formal_flux_taper_start_rosseland_depth": (
            None
            if formal_flux_taper_start_rosseland_depth is None
            else float(formal_flux_taper_start_rosseland_depth)
        ),
        "electron_scattering_source": "coherent-isotropic Lambda iteration",
        "electron_scattering_source_iterations_per_evaluation": int(
            payload["scattering_source_iterations"]
        ),
        "electron_scattering_source_final_maximum_relative_residual": float(
            payload["scattering_source_maximum_relative_residual"]
        ),
        "electron_scattering_source_final_worst_wavelength_index": int(
            payload["scattering_source_worst_wavelength_index"]
        ),
        "electron_scattering_source_final_worst_depth_index": int(
            payload["scattering_source_worst_depth_index"]
        ),
        "electron_scattering_source_final_worst_wavelength_angstrom": float(
            wavelength[payload["scattering_source_worst_wavelength_index"]]
        ),
        "electron_scattering_source_final_worst_rosseland_depth": float(
            rosseland_depth[payload["scattering_source_worst_depth_index"]]
        ),
    }
    if metadata is not None:
        common_metadata.update(metadata)
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
        metadata=common_metadata,
        hydrogen_lte_state=final_atmosphere.hydrogen_lte_state,
        helium_lte_state=final_atmosphere.helium_lte_state,
    )
