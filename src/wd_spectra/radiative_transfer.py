"""Formal radiative transfer with interchangeable Python and C backends."""

from __future__ import annotations

from typing import Literal
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import PI

try:
    from . import _rt
except ImportError:  # A compiler is optional at installation time.
    _rt = None


FloatArray = NDArray[np.float64]
Backend = Literal["auto", "c", "python"]


@dataclass(frozen=True)
class RadiationField:
    r"""Angular moments of the monochromatic radiation field.

    ``mean_intensity`` is :math:`J`; ``flux`` is :math:`4\pi H`.  Both arrays
    have shape ``(wavelength, depth)`` and use the source function's spectral
    density units.
    """

    mean_intensity: FloatArray
    flux: FloatArray
    interface_flux: FloatArray | None = None
    diagonal_lambda: FloatArray | None = None
    subdiagonal_lambda: FloatArray | None = None
    superdiagonal_lambda: FloatArray | None = None


@dataclass(frozen=True)
class StokesIntensity:
    """Emergent specific intensities for the four Stokes parameters."""

    i: FloatArray
    q: FloatArray
    u: FloatArray
    v: FloatArray


def compiled_backend_available() -> bool:
    """Return whether the optional C transfer kernel was imported."""

    return _rt is not None


def angular_quadrature(n_angle: int = 4) -> tuple[FloatArray, FloatArray]:
    """Gauss-Legendre directions and weights on the outward hemisphere."""

    if n_angle < 1:
        raise ValueError("n_angle must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(n_angle)
    return (
        np.ascontiguousarray(0.5 * (nodes + 1.0), dtype=np.float64),
        np.ascontiguousarray(0.5 * weights, dtype=np.float64),
    )


def _validate_inputs(
    optical_depth: ArrayLike, source_function: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    tau = np.ascontiguousarray(optical_depth, dtype=np.float64)
    source = np.ascontiguousarray(source_function, dtype=np.float64)
    if source.ndim == 1:
        source = source[np.newaxis, :]
    if source.ndim != 2 or tau.ndim not in (1, 2):
        raise ValueError("source_function must be 1D or 2D and optical_depth 1D or 2D")
    if source.shape[1] < 2 or source.shape[0] < 1:
        raise ValueError("source_function must contain at least two depth points")
    if tau.ndim == 1 and tau.shape[0] != source.shape[1]:
        raise ValueError("optical_depth and source_function depth axes differ")
    if tau.ndim == 2 and tau.shape != source.shape:
        raise ValueError("2D optical_depth must have the same shape as source_function")
    if np.any(~np.isfinite(tau)) or np.any(~np.isfinite(source)):
        raise ValueError("radiative-transfer inputs must be finite")
    if np.any(source < 0.0):
        raise ValueError("source_function must be non-negative")
    if np.any(np.diff(tau, axis=-1) <= 0.0):
        raise ValueError("optical_depth must increase strictly inward")
    return tau, source


def _python_emergent_flux(
    tau: FloatArray,
    source: FloatArray,
    mu: FloatArray,
    weight: FloatArray,
) -> FloatArray:
    output = np.empty(source.shape[0], dtype=np.float64)
    for wave in range(source.shape[0]):
        t = tau if tau.ndim == 1 else tau[wave]
        flux = 0.0
        for ray_mu, ray_weight in zip(mu, weight):
            intensity = source[wave, -1]
            for depth in range(source.shape[1] - 2, -1, -1):
                delta = (t[depth + 1] - t[depth]) / ray_mu
                attenuation = np.exp(-delta)
                constant_weight = -np.expm1(-delta)
                if delta < 1.0e-3:
                    slope_integral = delta * delta * (
                        0.5
                        + delta
                        * (
                            -1.0 / 3.0
                            + delta
                            * (1.0 / 8.0 + delta * (-1.0 / 30.0 + delta / 144.0))
                        )
                    )
                else:
                    slope_integral = constant_weight - delta * attenuation
                intensity = (
                    intensity * attenuation
                    + source[wave, depth] * constant_weight
                    + (source[wave, depth + 1] - source[wave, depth])
                    * slope_integral
                    / delta
                )
            # The first atmosphere point is generally at tau[0] > 0, not at
            # the vacuum boundary.  Propagate through that unresolved surface
            # cell with the first source value held constant.  Omitting this
            # step is negligible on grids extending to tiny optical depth but
            # biases line cores on coarser imported structures.
            surface_delta = t[0] / ray_mu
            intensity = (
                intensity * np.exp(-surface_delta)
                + source[wave, 0] * -np.expm1(-surface_delta)
            )
            flux += ray_weight * ray_mu * intensity
        output[wave] = 2.0 * np.pi * flux
    return output


def emergent_specific_intensity(
    optical_depth: ArrayLike,
    source_function: ArrayLike,
    ray_mu: float,
) -> FloatArray:
    """Return the outward surface intensity for one line-of-sight cosine.

    This uses the same piecewise-linear source interpolation, thermalized
    lower boundary, and finite surface-cell correction as
    :func:`emergent_flux`.  It is the appropriate formal solution for a
    resolved stellar surface element; disk integration must subsequently
    weight the intensity by projected area.
    """

    tau, source = _validate_inputs(optical_depth, source_function)
    mu = float(ray_mu)
    if not np.isfinite(mu) or not 0.0 < mu <= 1.0:
        raise ValueError("ray_mu must be finite and lie in (0, 1]")
    if tau.ndim == 1:
        tau = np.broadcast_to(tau[np.newaxis, :], source.shape)
    intensity = np.array(source[:, -1], copy=True)
    for depth in range(source.shape[1] - 2, -1, -1):
        delta = (tau[:, depth + 1] - tau[:, depth]) / mu
        intensity = _linear_cell_solution(
            intensity,
            source[:, depth + 1],
            source[:, depth],
            delta,
        )
    surface_delta = tau[:, 0] / mu
    return (
        intensity * np.exp(-surface_delta)
        + source[:, 0] * -np.expm1(-surface_delta)
    )


def emergent_stokes_specific_intensity(
    column_mass: ArrayLike,
    eta_i: ArrayLike,
    eta_q: ArrayLike,
    eta_v: ArrayLike,
    rho_q: ArrayLike,
    rho_v: ArrayLike,
    source_function: ArrayLike,
    ray_mu: float,
    *,
    emission_stokes: ArrayLike | None = None,
    formal_solver: str = "backward-euler",
) -> StokesIntensity:
    r"""Solve the coupled LTE ``IQUV`` transfer equation for one ray.

    Coefficients are mass propagation coefficients in ``cm^2 g^-1`` and have
    shape ``(wavelength, depth)``.  The Stokes reference direction is chosen
    in the ray--field plane, so ``eta_U=rho_U=0``.  The propagation matrix is

    ``[[etaI, etaQ, 0, etaV], [etaQ, etaI, rhoV, 0],``
    `` [0, -rhoV, etaI, rhoQ], [etaV, 0, -rhoQ, etaI]]``.

    The default stable backward-Euler formal solution is used cell by cell.
    ``formal_solver='matrix-exponential'`` instead uses a midpoint-constant
    matrix exponential, matching the MATEXP class of high-field solvers. In the
    exactly unpolarized limit the established scalar piecewise-linear solver
    is called directly, giving an exact regression limit rather than merely a
    depth-grid-convergent one.  The lower boundary is thermalized and the
    unresolved surface cell is retained.  ``emission_stokes``, when supplied,
    has shape ``(wavelength, depth, 4)`` and contains the physical emission
    vector per unit mass path.  This permits an unpolarized scattering source
    to be combined with polarized LTE line emissivity without pretending that
    the scattering source is multiplied by the line dichroism.
    """

    mass = np.asarray(column_mass, dtype=np.float64)
    arrays = np.broadcast_arrays(
        np.asarray(eta_i, dtype=np.float64),
        np.asarray(eta_q, dtype=np.float64),
        np.asarray(eta_v, dtype=np.float64),
        np.asarray(rho_q, dtype=np.float64),
        np.asarray(rho_v, dtype=np.float64),
        np.asarray(source_function, dtype=np.float64),
    )
    eta_i_array, eta_q_array, eta_v_array, rho_q_array, rho_v_array, source = (
        np.ascontiguousarray(item) for item in arrays
    )
    emission = None
    if emission_stokes is not None:
        emission = np.ascontiguousarray(emission_stokes, dtype=np.float64)
    if (
        mass.ndim != 1
        or eta_i_array.ndim != 2
        or eta_i_array.shape[1] != mass.size
        or mass.size < 2
        or np.any(~np.isfinite(mass))
        or np.any(np.diff(mass) <= 0.0)
        or np.any(~np.isfinite(eta_i_array))
        or np.any(~np.isfinite(eta_q_array))
        or np.any(~np.isfinite(eta_v_array))
        or np.any(~np.isfinite(rho_q_array))
        or np.any(~np.isfinite(rho_v_array))
        or np.any(~np.isfinite(source))
        or np.any(eta_i_array <= 0.0)
        or np.any(source < 0.0)
        or (
            emission is not None
            and (
                emission.shape != eta_i_array.shape + (4,)
                or np.any(~np.isfinite(emission))
            )
        )
    ):
        raise ValueError("Stokes transfer inputs must be finite and physical")
    mu = float(ray_mu)
    if not np.isfinite(mu) or not 0.0 < mu <= 1.0:
        raise ValueError("ray_mu must be finite and lie in (0, 1]")
    if formal_solver not in {"backward-euler", "matrix-exponential"}:
        raise ValueError(
            "formal_solver must be 'backward-euler' or 'matrix-exponential'"
        )
    dichroism = np.sqrt(eta_q_array**2 + eta_v_array**2)
    if np.any(dichroism > eta_i_array * (1.0 + 2.0e-10)):
        raise ValueError("polarized absorption matrix is not positive semidefinite")

    if emission is None and (
        not np.any(eta_q_array)
        and not np.any(eta_v_array)
        and not np.any(rho_q_array)
        and not np.any(rho_v_array)
    ):
        optical_depth = np.empty_like(eta_i_array)
        optical_depth[:, 0] = eta_i_array[:, 0] * mass[0]
        optical_depth[:, 1:] = optical_depth[:, [0]] + np.cumsum(
            0.5
            * (eta_i_array[:, 1:] + eta_i_array[:, :-1])
            * np.diff(mass)[np.newaxis, :],
            axis=1,
        )
        scalar = emergent_specific_intensity(optical_depth, source, mu)
        zero = np.zeros_like(scalar)
        return StokesIntensity(scalar, zero.copy(), zero.copy(), zero.copy())

    n_wave, n_depth = eta_i_array.shape
    identity = np.broadcast_to(np.eye(4), (n_wave, 4, 4))

    def matrix_at(depth: int) -> FloatArray:
        matrix = np.zeros((n_wave, 4, 4), dtype=np.float64)
        diagonal = eta_i_array[:, depth]
        matrix[:, 0, 0] = diagonal
        matrix[:, 1, 1] = diagonal
        matrix[:, 2, 2] = diagonal
        matrix[:, 3, 3] = diagonal
        matrix[:, 0, 1] = matrix[:, 1, 0] = eta_q_array[:, depth]
        matrix[:, 0, 3] = matrix[:, 3, 0] = eta_v_array[:, depth]
        matrix[:, 1, 2] = rho_v_array[:, depth]
        matrix[:, 2, 1] = -rho_v_array[:, depth]
        matrix[:, 2, 3] = rho_q_array[:, depth]
        matrix[:, 3, 2] = -rho_q_array[:, depth]
        return matrix

    inner_matrix = matrix_at(n_depth - 1)
    if emission is None:
        stokes = np.zeros((n_wave, 4), dtype=np.float64)
        stokes[:, 0] = source[:, -1]
    else:
        # Diffusion/thermalization boundary: use the local equilibrium Stokes
        # vector K^-1 j rather than forcing Q=U=V=0 when dichroic emissivity is
        # explicitly supplied.
        stokes = np.linalg.solve(inner_matrix, emission[:, -1])
    def exponential_step(
        incident: FloatArray,
        propagation: FloatArray,
        emitted: FloatArray,
        delta: FloatArray,
    ) -> FloatArray:
        equilibrium = np.linalg.solve(propagation, emitted)
        eigenvalue, eigenvector = np.linalg.eig(
            delta * propagation
        )
        inverse = np.linalg.inv(eigenvector)
        attenuation = np.einsum(
            "wij,wj,wjk->wik",
            eigenvector,
            np.exp(-eigenvalue),
            inverse,
            optimize=True,
        )
        propagated = equilibrium + np.einsum(
            "wij,wj->wi",
            attenuation,
            incident - equilibrium,
            optimize=True,
        )
        return np.asarray(np.real(propagated), dtype=np.float64)

    for depth in range(n_depth - 2, -1, -1):
        outer_matrix = matrix_at(depth)
        propagation = 0.5 * (inner_matrix + outer_matrix)
        delta = (mass[depth + 1] - mass[depth]) / mu
        if emission is None:
            thermal = np.zeros_like(stokes)
            thermal[:, 0] = 0.5 * (
                source[:, depth + 1] + source[:, depth]
            )
            emitted_per_mass = np.einsum(
                "wij,wj->wi", propagation, thermal, optimize=True
            )
        else:
            emitted_per_mass = 0.5 * (
                emission[:, depth + 1] + emission[:, depth]
            )
        if formal_solver == "matrix-exponential":
            stokes = exponential_step(
                stokes, propagation, emitted_per_mass, delta
            )
        else:
            scaled = delta * propagation
            right_hand_side = stokes + delta * emitted_per_mass
            stokes = np.linalg.solve(identity + scaled, right_hand_side)
        inner_matrix = outer_matrix

    surface_delta = mass[0] / mu
    if emission is None:
        thermal = np.zeros_like(stokes)
        thermal[:, 0] = source[:, 0]
        surface_emission_per_mass = np.einsum(
            "wij,wj->wi", inner_matrix, thermal, optimize=True
        )
    else:
        surface_emission_per_mass = emission[:, 0]
    if formal_solver == "matrix-exponential":
        stokes = exponential_step(
            stokes,
            inner_matrix,
            surface_emission_per_mass,
            surface_delta,
        )
    else:
        surface_scaled = surface_delta * inner_matrix
        stokes = np.linalg.solve(
            identity + surface_scaled,
            stokes + surface_delta * surface_emission_per_mass,
        )
    return StokesIntensity(
        np.asarray(stokes[:, 0]),
        np.asarray(stokes[:, 1]),
        np.asarray(stokes[:, 2]),
        np.asarray(stokes[:, 3]),
    )


def _linear_cell_solution(
    incident: FloatArray, source_start: FloatArray, source_end: FloatArray, delta: FloatArray
) -> FloatArray:
    """Propagate intensity through cells with source linear in optical path."""

    attenuation = np.exp(-delta)
    constant_weight = -np.expm1(-delta)
    slope_integral = np.empty_like(delta)
    small = delta < 1.0e-3
    slope_integral[small] = delta[small] ** 2 * (
        0.5
        + delta[small]
        * (
            -1.0 / 3.0
            + delta[small]
            * (
                1.0 / 8.0
                + delta[small] * (-1.0 / 30.0 + delta[small] / 144.0)
            )
        )
    )
    slope_integral[~small] = (
        constant_weight[~small] - delta[~small] * attenuation[~small]
    )
    return (
        incident * attenuation
        + source_end * constant_weight
        + (source_start - source_end) * slope_integral / delta
    )


def _linear_cell_end_source_weight(delta: FloatArray) -> FloatArray:
    """Coefficient of the source at the destination end of a cell."""

    constant_weight = -np.expm1(-delta)
    attenuation = np.exp(-delta)
    slope_integral = np.empty_like(delta)
    small = delta < 1.0e-3
    slope_integral[small] = delta[small] ** 2 * (
        0.5
        + delta[small]
        * (
            -1.0 / 3.0
            + delta[small]
            * (
                1.0 / 8.0
                + delta[small] * (-1.0 / 30.0 + delta[small] / 144.0)
            )
        )
    )
    slope_integral[~small] = (
        constant_weight[~small] - delta[~small] * attenuation[~small]
    )
    return constant_weight - slope_integral / delta


def _linear_cell_start_source_weight(delta: FloatArray) -> FloatArray:
    """Coefficient of the source at the incident end of a cell."""

    constant_weight = -np.expm1(-delta)
    attenuation = np.exp(-delta)
    slope_integral = np.empty_like(delta)
    small = delta < 1.0e-3
    slope_integral[small] = delta[small] ** 2 * (
        0.5
        + delta[small]
        * (
            -1.0 / 3.0
            + delta[small]
            * (
                1.0 / 8.0
                + delta[small] * (-1.0 / 30.0 + delta[small] / 144.0)
            )
        )
    )
    slope_integral[~small] = (
        constant_weight[~small] - delta[~small] * attenuation[~small]
    )
    return slope_integral / delta


def radiation_field(
    optical_depth: ArrayLike,
    source_function: ArrayLike,
    *,
    n_angle: int = 4,
    calculate_diagonal_lambda: bool = False,
    calculate_tridiagonal_lambda: bool = False,
) -> RadiationField:
    r"""Calculate :math:`J_\lambda` and :math:`F_\lambda` at every depth.

    This two-stream sweep uses the same piecewise-linear formal solution and
    boundary conditions as :func:`emergent_flux`: no incident radiation at
    the surface and a thermalized outward intensity at the bottom.
    """

    tau, source = _validate_inputs(optical_depth, source_function)
    if tau.ndim == 1:
        tau = np.broadcast_to(tau[np.newaxis, :], source.shape)
    mu, weight = angular_quadrature(n_angle)
    mean_intensity = np.zeros_like(source)
    flux = np.zeros_like(source)
    calculate_diagonal_lambda = bool(
        calculate_diagonal_lambda or calculate_tridiagonal_lambda
    )
    diagonal_lambda = np.zeros_like(source) if calculate_diagonal_lambda else None
    subdiagonal_lambda = (
        np.zeros_like(source) if calculate_tridiagonal_lambda else None
    )
    superdiagonal_lambda = (
        np.zeros_like(source) if calculate_tridiagonal_lambda else None
    )
    for ray_mu, ray_weight in zip(mu, weight):
        inward = np.zeros_like(source)
        outward = np.empty_like(source)
        # Account for material between the vacuum boundary (tau=0) and the
        # first tabulated depth.  A constant extrapolation is the only stable
        # local closure available without introducing a ghost depth point.
        surface_delta = tau[:, 0] / ray_mu
        inward[:, 0] = source[:, 0] * -np.expm1(-surface_delta)
        if calculate_diagonal_lambda:
            inward_diagonal = np.empty_like(source)
            outward_diagonal = np.empty_like(source)
            inward_diagonal[:, 0] = -np.expm1(-surface_delta)
            if calculate_tridiagonal_lambda:
                inward_predecessor = np.zeros_like(source)
                outward_successor = np.zeros_like(source)
        for depth in range(source.shape[1] - 1):
            delta = (tau[:, depth + 1] - tau[:, depth]) / ray_mu
            inward[:, depth + 1] = _linear_cell_solution(
                inward[:, depth], source[:, depth], source[:, depth + 1], delta
            )
            if calculate_diagonal_lambda:
                inward_diagonal[:, depth + 1] = _linear_cell_end_source_weight(
                    delta
                )
                if calculate_tridiagonal_lambda:
                    inward_predecessor[:, depth + 1] = (
                        _linear_cell_start_source_weight(delta)
                        + np.exp(-delta) * inward_diagonal[:, depth]
                    )
        outward[:, -1] = source[:, -1]
        if calculate_diagonal_lambda:
            outward_diagonal[:, -1] = 1.0
        for depth in range(source.shape[1] - 2, -1, -1):
            delta = (tau[:, depth + 1] - tau[:, depth]) / ray_mu
            outward[:, depth] = _linear_cell_solution(
                outward[:, depth + 1], source[:, depth + 1], source[:, depth], delta
            )
            if calculate_diagonal_lambda:
                outward_diagonal[:, depth] = _linear_cell_end_source_weight(
                    delta
                )
                if calculate_tridiagonal_lambda:
                    outward_successor[:, depth] = (
                        _linear_cell_start_source_weight(delta)
                        + np.exp(-delta) * outward_diagonal[:, depth + 1]
                    )
        mean_intensity += 0.5 * ray_weight * (outward + inward)
        flux += 2.0 * PI * ray_weight * ray_mu * (outward - inward)
        if diagonal_lambda is not None:
            diagonal_lambda += 0.5 * ray_weight * (
                outward_diagonal + inward_diagonal
            )
        if subdiagonal_lambda is not None and superdiagonal_lambda is not None:
            subdiagonal_lambda += 0.5 * ray_weight * inward_predecessor
            superdiagonal_lambda += 0.5 * ray_weight * outward_successor
    return RadiationField(
        mean_intensity=mean_intensity,
        flux=flux,
        diagonal_lambda=diagonal_lambda,
        subdiagonal_lambda=subdiagonal_lambda,
        superdiagonal_lambda=superdiagonal_lambda,
    )


def feautrier_radiation_field(
    optical_depth: ArrayLike,
    source_function: ArrayLike,
    *,
    n_angle: int = 4,
) -> RadiationField:
    r"""Calculate ``J_lambda`` and ``F_lambda`` with a Feautrier solve.

    This second-order discretization is intended for atmosphere structure
    iterations, where consistency between local radiative equilibrium and
    flux constancy matters more than the lower cost of the directional sweep
    in :func:`radiation_field`. A surface point at zero optical depth is
    inserted with a constant source extrapolation, the no-incident-radiation
    Robin condition is imposed there, and the bottom symmetric intensity is
    thermalized. The tridiagonal systems are solved simultaneously over all
    wavelengths for each Gaussian angle.
    """

    tau, source = _validate_inputs(optical_depth, source_function)
    if tau.ndim == 1:
        tau = np.broadcast_to(tau[np.newaxis, :], source.shape)
    n_wavelength, n_depth = source.shape
    expanded_tau = np.empty((n_wavelength, n_depth + 1), dtype=np.float64)
    expanded_source = np.empty_like(expanded_tau)
    expanded_tau[:, 0] = 0.0
    expanded_tau[:, 1:] = tau
    expanded_source[:, 0] = source[:, 0]
    expanded_source[:, 1:] = source

    mu, weight = angular_quadrature(n_angle)
    mean_intensity = np.zeros_like(source)
    flux = np.zeros_like(source)
    interface_flux = np.zeros_like(source)
    for ray_mu, ray_weight in zip(mu, weight):
        lower = np.zeros_like(expanded_tau)
        diagonal = np.zeros_like(expanded_tau)
        upper = np.zeros_like(expanded_tau)
        right_hand_side = expanded_source.copy()

        first_step = expanded_tau[:, 1]
        diagonal[:, 0] = 1.0 + ray_mu / first_step
        upper[:, 0] = -ray_mu / first_step
        right_hand_side[:, 0] = 0.0
        previous_step = expanded_tau[:, 1:-1] - expanded_tau[:, :-2]
        next_step = expanded_tau[:, 2:] - expanded_tau[:, 1:-1]
        lower[:, 1:-1] = (
            -2.0 * ray_mu**2
            / (previous_step * (previous_step + next_step))
        )
        upper[:, 1:-1] = (
            -2.0 * ray_mu**2
            / (next_step * (previous_step + next_step))
        )
        diagonal[:, 1:-1] = 1.0 - lower[:, 1:-1] - upper[:, 1:-1]
        diagonal[:, -1] = 1.0
        right_hand_side[:, -1] = expanded_source[:, -1]

        # Vectorized Thomas elimination: the depth loop is short, while every
        # operation inside it covers the complete structure wavelength grid.
        for depth in range(1, n_depth + 1):
            multiplier = lower[:, depth] / diagonal[:, depth - 1]
            diagonal[:, depth] -= multiplier * upper[:, depth - 1]
            right_hand_side[:, depth] -= (
                multiplier * right_hand_side[:, depth - 1]
            )
        symmetric_intensity = np.empty_like(expanded_tau)
        symmetric_intensity[:, -1] = (
            right_hand_side[:, -1] / diagonal[:, -1]
        )
        for depth in range(n_depth - 1, -1, -1):
            symmetric_intensity[:, depth] = (
                right_hand_side[:, depth]
                - upper[:, depth] * symmetric_intensity[:, depth + 1]
            ) / diagonal[:, depth]

        mean_intensity += ray_weight * symmetric_intensity[:, 1:]
        interface_flux += (
            4.0
            * PI
            * ray_weight
            * ray_mu**2
            * np.diff(symmetric_intensity, axis=1)
            / np.diff(expanded_tau, axis=1)
        )
        derivative = np.empty_like(source)
        previous_step = expanded_tau[:, 1:-1] - expanded_tau[:, :-2]
        next_step = expanded_tau[:, 2:] - expanded_tau[:, 1:-1]
        derivative[:, :-1] = (
            -next_step
            / (previous_step * (previous_step + next_step))
            * symmetric_intensity[:, :-2]
            + (next_step - previous_step)
            / (previous_step * next_step)
            * symmetric_intensity[:, 1:-1]
            + previous_step
            / (next_step * (previous_step + next_step))
            * symmetric_intensity[:, 2:]
        )
        derivative[:, -1] = (
            symmetric_intensity[:, -1] - symmetric_intensity[:, -2]
        ) / (expanded_tau[:, -1] - expanded_tau[:, -2])
        flux += 4.0 * PI * ray_weight * ray_mu**2 * derivative

    return RadiationField(
        mean_intensity=mean_intensity,
        flux=flux,
        interface_flux=interface_flux,
    )


def integrated_feautrier_interface_flux_response(
    optical_depth: ArrayLike,
    wavelength_angstrom: ArrayLike,
    source_derivative: ArrayLike,
    *,
    n_angle: int = 4,
    wavelength_chunk_size: int = 64,
) -> FloatArray:
    r"""Return the integrated interface-flux response to a depth state.

    For a state vector ``x[k]`` whose local source derivative is
    ``dS(lambda,k)/dx[k]``, this routine evaluates

    ``M[d,k] = integral dF_lambda(interface d)/dS(lambda,k)
                         dS(lambda,k)/dx[k] dlambda``.

    The response uses exactly the Feautrier operator, surface boundary, deep
    boundary, angular quadrature, and interface-flux discretization used by
    :func:`feautrier_radiation_field`.  It is therefore suitable as the
    radiative block of a conservative atmosphere Newton correction.  The
    wavelength axis is processed in chunks so the full
    ``(wavelength, depth, depth)`` operator is never retained.
    """

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    derivative = np.ascontiguousarray(source_derivative, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or wavelength.size < 2
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be positive and increasing")
    if derivative.ndim != 2 or derivative.shape[0] != wavelength.size:
        raise ValueError(
            "source_derivative must have shape (wavelength, depth)"
        )
    if derivative.shape[1] < 2 or np.any(~np.isfinite(derivative)):
        raise ValueError("source_derivative must be finite with at least two depths")
    if wavelength_chunk_size < 1:
        raise ValueError("wavelength_chunk_size must be positive")

    dummy_source = np.ones_like(derivative)
    tau, _ = _validate_inputs(optical_depth, dummy_source)
    if tau.ndim == 1:
        tau = np.broadcast_to(tau[np.newaxis, :], derivative.shape)
    mu, angle_weight = angular_quadrature(n_angle)
    n_wavelength, n_depth = derivative.shape
    trapezoid_weight = np.empty(n_wavelength, dtype=np.float64)
    differences = np.diff(wavelength)
    trapezoid_weight[0] = 0.5 * differences[0]
    trapezoid_weight[-1] = 0.5 * differences[-1]
    trapezoid_weight[1:-1] = 0.5 * (
        differences[:-1] + differences[1:]
    )
    integrated = np.zeros((n_depth, n_depth), dtype=np.float64)

    for start in range(0, n_wavelength, wavelength_chunk_size):
        stop = min(start + wavelength_chunk_size, n_wavelength)
        chunk_tau = tau[start:stop]
        chunk_size = stop - start
        expanded_tau = np.empty(
            (chunk_size, n_depth + 1), dtype=np.float64
        )
        expanded_tau[:, 0] = 0.0
        expanded_tau[:, 1:] = chunk_tau
        flux_operator = np.zeros(
            (chunk_size, n_depth, n_depth), dtype=np.float64
        )

        for ray_mu, ray_weight in zip(mu, angle_weight):
            lower = np.zeros_like(expanded_tau)
            diagonal = np.zeros_like(expanded_tau)
            upper = np.zeros_like(expanded_tau)
            right_hand_side = np.zeros(
                (chunk_size, n_depth + 1, n_depth), dtype=np.float64
            )

            first_step = expanded_tau[:, 1]
            diagonal[:, 0] = 1.0 + ray_mu / first_step
            upper[:, 0] = -ray_mu / first_step
            previous_step = expanded_tau[:, 1:-1] - expanded_tau[:, :-2]
            next_step = expanded_tau[:, 2:] - expanded_tau[:, 1:-1]
            lower[:, 1:-1] = (
                -2.0
                * ray_mu**2
                / (previous_step * (previous_step + next_step))
            )
            upper[:, 1:-1] = (
                -2.0
                * ray_mu**2
                / (next_step * (previous_step + next_step))
            )
            diagonal[:, 1:-1] = (
                1.0 - lower[:, 1:-1] - upper[:, 1:-1]
            )
            diagonal[:, -1] = 1.0

            # The expanded-grid interior equation at row j uses source node
            # j-1.  The surface Robin equation has no source term, while the
            # thermalized bottom row uses the deepest source node.
            depth_index = np.arange(n_depth - 1)
            right_hand_side[
                :, depth_index + 1, depth_index
            ] = 1.0
            right_hand_side[:, -1, -1] = 1.0

            for depth in range(1, n_depth + 1):
                multiplier = lower[:, depth] / diagonal[:, depth - 1]
                diagonal[:, depth] -= multiplier * upper[:, depth - 1]
                right_hand_side[:, depth, :] -= (
                    multiplier[:, np.newaxis]
                    * right_hand_side[:, depth - 1, :]
                )
            symmetric_response = np.empty_like(right_hand_side)
            symmetric_response[:, -1, :] = (
                right_hand_side[:, -1, :]
                / diagonal[:, -1, np.newaxis]
            )
            for depth in range(n_depth - 1, -1, -1):
                symmetric_response[:, depth, :] = (
                    right_hand_side[:, depth, :]
                    - upper[:, depth, np.newaxis]
                    * symmetric_response[:, depth + 1, :]
                ) / diagonal[:, depth, np.newaxis]

            flux_operator += (
                4.0
                * PI
                * ray_weight
                * ray_mu**2
                * np.diff(symmetric_response, axis=1)
                / np.diff(expanded_tau, axis=1)[:, :, np.newaxis]
            )

        integrated += np.einsum(
            "wdk,wk,w->dk",
            flux_operator,
            derivative[start:stop],
            trapezoid_weight[start:stop],
            optimize=True,
        )
    return np.ascontiguousarray(integrated)


def integrated_feautrier_interface_state_response(
    optical_depth: ArrayLike,
    wavelength_angstrom: ArrayLike,
    source_function: ArrayLike,
    source_derivative: ArrayLike,
    column_mass: ArrayLike,
    mass_opacity_derivative: ArrayLike,
    *,
    n_angle: int = 4,
    wavelength_chunk_size: int = 32,
    backend: Backend = "auto",
) -> FloatArray:
    r"""Return the interface-flux response to source and opacity changes.

    State variable ``x[k]`` changes the local source and mass opacity at depth
    ``k``.  The latter changes every deeper optical-depth node, so a
    fixed-optical-depth Lambda response omits an important term in cool,
    neutral atmospheres.  This routine differentiates the complete discrete
    Feautrier system,

    ``A(tau) u = S``, hence ``A du = dS - dA u``,

    and also differentiates the optical-depth denominator in the conservative
    interface flux.  Feautrier coefficient derivatives and the tangent
    tridiagonal systems are evaluated exactly.  Wavelength chunks bound the temporary
    ``(wavelength, depth, state)`` storage.
    """

    if backend not in ("auto", "c", "python"):
        raise ValueError("backend must be 'auto', 'c', or 'python'")
    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    source = np.ascontiguousarray(source_function, dtype=np.float64)
    source_response = np.ascontiguousarray(
        source_derivative, dtype=np.float64
    )
    opacity_response = np.ascontiguousarray(
        mass_opacity_derivative, dtype=np.float64
    )
    mass = np.ascontiguousarray(column_mass, dtype=np.float64)
    tau, source = _validate_inputs(optical_depth, source)
    if tau.ndim == 1:
        tau = np.broadcast_to(tau[np.newaxis, :], source.shape)
    if (
        wavelength.ndim != 1
        or wavelength.size != source.shape[0]
        or wavelength.size < 2
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must match, be positive, and increase")
    if (
        source_response.shape != source.shape
        or opacity_response.shape != source.shape
        or np.any(~np.isfinite(source_response))
        or np.any(~np.isfinite(opacity_response))
    ):
        raise ValueError(
            "source and opacity derivatives must be finite and match the source"
        )
    n_wavelength, n_depth = source.shape
    if (
        mass.shape != (n_depth,)
        or np.any(~np.isfinite(mass))
        or np.any(mass <= 0.0)
        or np.any(np.diff(mass) <= 0.0)
    ):
        raise ValueError("column_mass must be positive and increase with depth")
    if wavelength_chunk_size < 1:
        raise ValueError("wavelength_chunk_size must be positive")

    mu, angle_weight = angular_quadrature(n_angle)
    trapezoid_weight = np.empty(n_wavelength, dtype=np.float64)
    wavelength_step = np.diff(wavelength)
    trapezoid_weight[0] = 0.5 * wavelength_step[0]
    trapezoid_weight[-1] = 0.5 * wavelength_step[-1]
    trapezoid_weight[1:-1] = 0.5 * (
        wavelength_step[:-1] + wavelength_step[1:]
    )
    integrated = np.zeros((n_depth, n_depth), dtype=np.float64)
    mass_step = np.diff(mass)

    compiled = (
        None
        if _rt is None
        else getattr(
            _rt, "integrated_feautrier_interface_state_response", None
        )
    )
    use_c = backend == "c" or (backend == "auto" and compiled is not None)
    if use_c:
        if compiled is None:
            raise RuntimeError(
                "the compiled Feautrier-response backend is unavailable"
            )
        return np.asarray(
            compiled(
                np.ascontiguousarray(tau, dtype=np.float64),
                source,
                source_response,
                opacity_response,
                mass,
                mu,
                angle_weight,
                np.ascontiguousarray(trapezoid_weight, dtype=np.float64),
            ),
            dtype=np.float64,
        )

    def coefficients(expanded_tau, ray_mu):
        shape = expanded_tau.shape
        lower = np.zeros(shape, dtype=np.float64)
        diagonal = np.zeros(shape, dtype=np.float64)
        upper = np.zeros(shape, dtype=np.float64)
        first_step = expanded_tau[:, 1]
        diagonal[:, 0] = 1.0 + ray_mu / first_step
        upper[:, 0] = -ray_mu / first_step
        previous = expanded_tau[:, 1:-1] - expanded_tau[:, :-2]
        following = expanded_tau[:, 2:] - expanded_tau[:, 1:-1]
        lower[:, 1:-1] = (
            -2.0 * ray_mu**2 / (previous * (previous + following))
        )
        upper[:, 1:-1] = (
            -2.0 * ray_mu**2 / (following * (previous + following))
        )
        diagonal[:, 1:-1] = 1.0 - lower[:, 1:-1] - upper[:, 1:-1]
        diagonal[:, -1] = 1.0
        return lower, diagonal, upper

    for start in range(0, n_wavelength, wavelength_chunk_size):
        stop = min(start + wavelength_chunk_size, n_wavelength)
        chunk_tau = tau[start:stop]
        chunk_source = source[start:stop]
        chunk_source_response = source_response[start:stop]
        chunk_opacity_response = opacity_response[start:stop]
        chunk_size = stop - start

        expanded_tau = np.empty((chunk_size, n_depth + 1), dtype=np.float64)
        expanded_tau[:, 0] = 0.0
        expanded_tau[:, 1:] = chunk_tau
        expanded_source = np.empty_like(expanded_tau)
        expanded_source[:, 0] = chunk_source[:, 0]
        expanded_source[:, 1:] = chunk_source

        tau_response = np.zeros(
            (chunk_size, n_depth, n_depth), dtype=np.float64
        )
        tau_response[:, 0, 0] = (
            chunk_opacity_response[:, 0] * mass[0]
        )
        for depth in range(1, n_depth):
            tau_response[:, depth, :] = tau_response[:, depth - 1, :]
            tau_response[:, depth, depth - 1] += (
                0.5
                * mass_step[depth - 1]
                * chunk_opacity_response[:, depth - 1]
            )
            tau_response[:, depth, depth] += (
                0.5
                * mass_step[depth - 1]
                * chunk_opacity_response[:, depth]
            )
        expanded_tau_response = np.zeros(
            (chunk_size, n_depth + 1, n_depth), dtype=np.float64
        )
        expanded_tau_response[:, 1:, :] = tau_response
        chunk_flux_response = np.zeros(
            (chunk_size, n_depth, n_depth), dtype=np.float64
        )

        for ray_mu, ray_weight in zip(mu, angle_weight):
            lower, diagonal, upper = coefficients(expanded_tau, ray_mu)
            lower_response = np.zeros_like(expanded_tau_response)
            diagonal_response = np.zeros_like(expanded_tau_response)
            upper_response = np.zeros_like(expanded_tau_response)
            first_step = expanded_tau[:, 1]
            first_step_response = expanded_tau_response[:, 1, :]
            diagonal_response[:, 0, :] = (
                -ray_mu
                * first_step_response
                / first_step[:, np.newaxis] ** 2
            )
            upper_response[:, 0, :] = -diagonal_response[:, 0, :]
            previous = expanded_tau[:, 1:-1] - expanded_tau[:, :-2]
            following = expanded_tau[:, 2:] - expanded_tau[:, 1:-1]
            previous_response = (
                expanded_tau_response[:, 1:-1, :]
                - expanded_tau_response[:, :-2, :]
            )
            following_response = (
                expanded_tau_response[:, 2:, :]
                - expanded_tau_response[:, 1:-1, :]
            )
            combined_response = previous_response + following_response
            lower_response[:, 1:-1, :] = lower[
                :, 1:-1, np.newaxis
            ] * (
                -previous_response / previous[:, :, np.newaxis]
                - combined_response
                / (previous + following)[:, :, np.newaxis]
            )
            upper_response[:, 1:-1, :] = upper[
                :, 1:-1, np.newaxis
            ] * (
                -following_response / following[:, :, np.newaxis]
                - combined_response
                / (previous + following)[:, :, np.newaxis]
            )
            diagonal_response[:, 1:-1, :] = (
                -lower_response[:, 1:-1, :]
                - upper_response[:, 1:-1, :]
            )

            right_hand_side = expanded_source.copy()
            right_hand_side[:, 0] = 0.0
            right_hand_side[:, -1] = expanded_source[:, -1]
            factored_diagonal = diagonal.copy()
            multipliers = np.zeros_like(diagonal)
            for depth in range(1, n_depth + 1):
                multiplier = lower[:, depth] / factored_diagonal[:, depth - 1]
                multipliers[:, depth] = multiplier
                factored_diagonal[:, depth] -= multiplier * upper[:, depth - 1]
                right_hand_side[:, depth] -= (
                    multiplier * right_hand_side[:, depth - 1]
                )
            symmetric_intensity = np.empty_like(expanded_tau)
            symmetric_intensity[:, -1] = (
                right_hand_side[:, -1] / factored_diagonal[:, -1]
            )
            for depth in range(n_depth - 1, -1, -1):
                symmetric_intensity[:, depth] = (
                    right_hand_side[:, depth]
                    - upper[:, depth] * symmetric_intensity[:, depth + 1]
                ) / factored_diagonal[:, depth]

            tangent_rhs = np.zeros(
                (chunk_size, n_depth + 1, n_depth), dtype=np.float64
            )
            local_depth = np.arange(n_depth - 1)
            tangent_rhs[:, local_depth + 1, local_depth] = (
                chunk_source_response[:, :-1]
            )
            tangent_rhs[:, -1, -1] = chunk_source_response[:, -1]
            tangent_rhs -= (
                diagonal_response
                * symmetric_intensity[:, :, np.newaxis]
            )
            tangent_rhs[:, 1:, :] -= (
                lower_response[:, 1:, :]
                * symmetric_intensity[:, :-1, np.newaxis]
            )
            tangent_rhs[:, :-1, :] -= (
                upper_response[:, :-1, :]
                * symmetric_intensity[:, 1:, np.newaxis]
            )
            for depth in range(1, n_depth + 1):
                tangent_rhs[:, depth, :] -= (
                    multipliers[:, depth, np.newaxis]
                    * tangent_rhs[:, depth - 1, :]
                )
            symmetric_response = np.empty_like(tangent_rhs)
            symmetric_response[:, -1, :] = (
                tangent_rhs[:, -1, :]
                / factored_diagonal[:, -1, np.newaxis]
            )
            for depth in range(n_depth - 1, -1, -1):
                symmetric_response[:, depth, :] = (
                    tangent_rhs[:, depth, :]
                    - upper[:, depth, np.newaxis]
                    * symmetric_response[:, depth + 1, :]
                ) / factored_diagonal[:, depth, np.newaxis]

            optical_step = np.diff(expanded_tau, axis=1)
            optical_step_response = np.diff(
                expanded_tau_response, axis=1
            )
            intensity_step = np.diff(symmetric_intensity, axis=1)
            intensity_step_response = np.diff(
                symmetric_response, axis=1
            )
            chunk_flux_response += (
                4.0
                * PI
                * ray_weight
                * ray_mu**2
                * (
                    intensity_step_response
                    / optical_step[:, :, np.newaxis]
                    - intensity_step[:, :, np.newaxis]
                    * optical_step_response
                    / optical_step[:, :, np.newaxis] ** 2
                )
            )

        integrated += np.einsum(
            "wdk,w->dk",
            chunk_flux_response,
            trapezoid_weight[start:stop],
            optimize=True,
        )
    return np.ascontiguousarray(integrated)


def integrated_emergent_flux_state_response(
    optical_depth: ArrayLike,
    wavelength_angstrom: ArrayLike,
    source_function: ArrayLike,
    source_derivative: ArrayLike,
    column_mass: ArrayLike,
    mass_opacity_derivative: ArrayLike,
    *,
    n_angle: int = 4,
    wavelength_chunk_size: int = 64,
) -> FloatArray:
    r"""Return the exact formal surface-flux response to a local state.

    State variable ``x[k]`` changes the source and mass opacity locally at
    depth ``k``.  This differentiates the same piecewise-linear outward sweep,
    thermalized lower boundary, finite surface cell, and angular quadrature as
    :func:`emergent_flux`.  It is used for the surface row of the atmosphere
    Newton system, while conservative Feautrier interface responses remain the
    appropriate discretization at interior interfaces.
    """

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    source = np.ascontiguousarray(source_function, dtype=np.float64)
    source_response = np.ascontiguousarray(
        source_derivative, dtype=np.float64
    )
    opacity_response = np.ascontiguousarray(
        mass_opacity_derivative, dtype=np.float64
    )
    mass = np.ascontiguousarray(column_mass, dtype=np.float64)
    tau, source = _validate_inputs(optical_depth, source)
    if tau.ndim == 1:
        tau = np.broadcast_to(tau[np.newaxis, :], source.shape)
    if (
        wavelength.ndim != 1
        or wavelength.size != source.shape[0]
        or wavelength.size < 2
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError(
            "wavelength_angstrom must match, be positive, and increase"
        )
    if (
        source_response.shape != source.shape
        or opacity_response.shape != source.shape
        or np.any(~np.isfinite(source_response))
        or np.any(~np.isfinite(opacity_response))
    ):
        raise ValueError(
            "source and opacity derivatives must be finite and match the source"
        )
    n_wavelength, n_depth = source.shape
    if (
        mass.shape != (n_depth,)
        or np.any(~np.isfinite(mass))
        or np.any(mass <= 0.0)
        or np.any(np.diff(mass) <= 0.0)
    ):
        raise ValueError("column_mass must be positive and increase with depth")
    if wavelength_chunk_size < 1:
        raise ValueError("wavelength_chunk_size must be positive")

    wavelength_weight = np.empty(n_wavelength, dtype=np.float64)
    wavelength_step = np.diff(wavelength)
    wavelength_weight[0] = 0.5 * wavelength_step[0]
    wavelength_weight[-1] = 0.5 * wavelength_step[-1]
    wavelength_weight[1:-1] = 0.5 * (
        wavelength_step[:-1] + wavelength_step[1:]
    )
    mass_step = np.diff(mass)
    mu, angle_weight = angular_quadrature(n_angle)
    integrated = np.zeros(n_depth, dtype=np.float64)

    def slope_weight_and_derivative(delta: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return the inner-source weight and its delta derivative."""

        attenuation = np.exp(-delta)
        constant_weight = -np.expm1(-delta)
        weight = np.empty_like(delta)
        derivative = np.empty_like(delta)
        small = delta < 1.0e-3
        small_delta = delta[small]
        weight[small] = small_delta * (
            0.5
            + small_delta
            * (
                -1.0 / 3.0
                + small_delta
                * (
                    1.0 / 8.0
                    + small_delta
                    * (-1.0 / 30.0 + small_delta / 144.0)
                )
            )
        )
        derivative[small] = (
            0.5
            - 2.0 * small_delta / 3.0
            + 3.0 * small_delta**2 / 8.0
            - 2.0 * small_delta**3 / 15.0
            + 5.0 * small_delta**4 / 144.0
        )
        ordinary = ~small
        ordinary_delta = delta[ordinary]
        ordinary_attenuation = attenuation[ordinary]
        ordinary_constant = constant_weight[ordinary]
        weight[ordinary] = (
            ordinary_constant / ordinary_delta - ordinary_attenuation
        )
        derivative[ordinary] = (
            ordinary_attenuation
            + (
                ordinary_delta * ordinary_attenuation - ordinary_constant
            )
            / ordinary_delta**2
        )
        return weight, derivative

    for start in range(0, n_wavelength, wavelength_chunk_size):
        stop = min(start + wavelength_chunk_size, n_wavelength)
        chunk_tau = tau[start:stop]
        chunk_source = source[start:stop]
        chunk_source_response = source_response[start:stop]
        chunk_opacity_response = opacity_response[start:stop]
        chunk_size = stop - start
        chunk_integrated = np.zeros((chunk_size, n_depth), dtype=np.float64)
        for ray_mu, ray_weight in zip(mu, angle_weight):
            # Forward sweep: retain the outward intensity at every node.
            # The scalar surface flux then admits an O(N_depth) reverse-mode
            # derivative, rather than propagating an N_depth-wide tangent
            # vector through every one of the N_depth cells.
            outward = np.empty_like(chunk_source)
            outward[:, -1] = chunk_source[:, -1]
            for depth in range(n_depth - 2, -1, -1):
                delta = (
                    chunk_tau[:, depth + 1] - chunk_tau[:, depth]
                ) / ray_mu
                attenuation = np.exp(-delta)
                constant_weight = -np.expm1(-delta)
                slope_weight, _ = slope_weight_and_derivative(delta)
                inner_source = chunk_source[:, depth + 1]
                outer_source = chunk_source[:, depth]
                outward[:, depth] = (
                    attenuation * outward[:, depth + 1]
                    + constant_weight * outer_source
                    + slope_weight * (inner_source - outer_source)
                )

            surface_delta = chunk_tau[:, 0] / ray_mu
            surface_attenuation = np.exp(-surface_delta)
            surface_constant = -np.expm1(-surface_delta)
            surface_source = chunk_source[:, 0]
            source_gradient = np.zeros_like(chunk_source)
            opacity_gradient = np.zeros_like(chunk_source)
            source_gradient[:, 0] = surface_constant
            opacity_gradient[:, 0] = (
                surface_attenuation
                * (surface_source - outward[:, 0])
                * mass[0]
                / ray_mu
            )
            intensity_adjoint = surface_attenuation

            # Reverse the outward sweep.  A local opacity changes only the
            # two adjacent trapezoidal optical-depth increments (plus the
            # unresolved surface cell at depth zero), so no dense optical-
            # depth response matrix is required.
            for depth in range(n_depth - 1):
                delta = (
                    chunk_tau[:, depth + 1] - chunk_tau[:, depth]
                ) / ray_mu
                attenuation = np.exp(-delta)
                constant_weight = -np.expm1(-delta)
                slope_weight, slope_derivative = (
                    slope_weight_and_derivative(delta)
                )
                inner_source = chunk_source[:, depth + 1]
                outer_source = chunk_source[:, depth]
                delta_gradient = intensity_adjoint * (
                    attenuation
                    * (outer_source - outward[:, depth + 1])
                    + (inner_source - outer_source) * slope_derivative
                )
                source_gradient[:, depth] += intensity_adjoint * (
                    constant_weight - slope_weight
                )
                source_gradient[:, depth + 1] += (
                    intensity_adjoint * slope_weight
                )
                opacity_cell_gradient = (
                    0.5 * mass_step[depth] / ray_mu * delta_gradient
                )
                opacity_gradient[:, depth] += opacity_cell_gradient
                opacity_gradient[:, depth + 1] += opacity_cell_gradient
                intensity_adjoint *= attenuation
            source_gradient[:, -1] += intensity_adjoint
            response = (
                source_gradient * chunk_source_response
                + opacity_gradient * chunk_opacity_response
            )
            chunk_integrated += (
                2.0 * PI * ray_weight * ray_mu * response
            )
        integrated += np.einsum(
            "wk,w->k",
            chunk_integrated,
            wavelength_weight[start:stop],
            optimize=True,
        )
    return np.ascontiguousarray(integrated)


def integrated_lambda_response(
    optical_depth: ArrayLike,
    wavelength_angstrom: ArrayLike,
    response_weight: ArrayLike,
    source_derivative: ArrayLike,
    *,
    n_angle: int = 4,
    wavelength_chunk_size: int = 64,
    return_surface_flux_response: bool = False,
) -> FloatArray | tuple[FloatArray, FloatArray]:
    r"""Integrate ``weight_lambda Lambda_lambda dS/dx`` over wavelength.

    The result is the dense depth-by-depth Jacobian

    ``M[d,k] = integral w(lambda,d) Lambda(lambda,d,k)
                         dS(lambda,k)/dx(k) dlambda``.

    Wavelength chunks avoid storing the full ``(wavelength, depth, depth)``
    Lambda operator.  This is intended for atmosphere Newton blocks, where
    the depth dimension is modest even when the structural frequency grid is
    large.
    """

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    weight = np.ascontiguousarray(response_weight, dtype=np.float64)
    derivative = np.ascontiguousarray(source_derivative, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or wavelength.size < 2
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be positive and increasing")
    if weight.shape != derivative.shape or weight.shape[0] != wavelength.size:
        raise ValueError(
            "response_weight and source_derivative must have shape (wavelength, depth)"
        )
    if (
        np.any(~np.isfinite(weight))
        or np.any(~np.isfinite(derivative))
    ):
        raise ValueError("Lambda-response inputs must be finite")
    if wavelength_chunk_size < 1:
        raise ValueError("wavelength_chunk_size must be positive")

    dummy_source = np.ones_like(weight)
    tau, _ = _validate_inputs(optical_depth, dummy_source)
    if tau.ndim == 1:
        tau = np.broadcast_to(tau[np.newaxis, :], weight.shape)
    mu, angle_weight = angular_quadrature(n_angle)
    n_wavelength, n_depth = weight.shape
    trapezoid_weight = np.empty(n_wavelength, dtype=np.float64)
    differences = np.diff(wavelength)
    trapezoid_weight[0] = 0.5 * differences[0]
    trapezoid_weight[-1] = 0.5 * differences[-1]
    trapezoid_weight[1:-1] = 0.5 * (
        differences[:-1] + differences[1:]
    )
    integrated = np.zeros((n_depth, n_depth), dtype=np.float64)
    integrated_surface_flux = np.zeros(n_depth, dtype=np.float64)

    for start in range(0, n_wavelength, wavelength_chunk_size):
        stop = min(start + wavelength_chunk_size, n_wavelength)
        chunk_tau = tau[start:stop]
        chunk_size = stop - start
        lambda_operator = np.zeros(
            (chunk_size, n_depth, n_depth), dtype=np.float64
        )
        surface_flux_operator = np.zeros(
            (chunk_size, n_depth), dtype=np.float64
        )
        for ray_mu, ray_weight in zip(mu, angle_weight):
            inward = np.zeros_like(lambda_operator)
            outward = np.zeros_like(lambda_operator)
            inward[:, 0, 0] = -np.expm1(-chunk_tau[:, 0] / ray_mu)
            for depth in range(n_depth - 1):
                delta = (
                    chunk_tau[:, depth + 1] - chunk_tau[:, depth]
                ) / ray_mu
                inward[:, depth + 1, :] = (
                    np.exp(-delta)[:, np.newaxis] * inward[:, depth, :]
                )
                inward[:, depth + 1, depth] += (
                    _linear_cell_start_source_weight(delta)
                )
                inward[:, depth + 1, depth + 1] += (
                    _linear_cell_end_source_weight(delta)
                )
            outward[:, -1, -1] = 1.0
            for depth in range(n_depth - 2, -1, -1):
                delta = (
                    chunk_tau[:, depth + 1] - chunk_tau[:, depth]
                ) / ray_mu
                outward[:, depth, :] = (
                    np.exp(-delta)[:, np.newaxis]
                    * outward[:, depth + 1, :]
                )
                outward[:, depth, depth + 1] += (
                    _linear_cell_start_source_weight(delta)
                )
                outward[:, depth, depth] += (
                    _linear_cell_end_source_weight(delta)
                )
            lambda_operator += 0.5 * ray_weight * (inward + outward)
            surface_attenuation = np.exp(-chunk_tau[:, 0] / ray_mu)
            surface_response = (
                surface_attenuation[:, np.newaxis] * outward[:, 0, :]
            )
            surface_response[:, 0] += -np.expm1(
                -chunk_tau[:, 0] / ray_mu
            )
            surface_flux_operator += (
                2.0 * PI * ray_weight * ray_mu * surface_response
            )
        integrated += np.einsum(
            "wd,wde,we,w->de",
            weight[start:stop],
            lambda_operator,
            derivative[start:stop],
            trapezoid_weight[start:stop],
            optimize=True,
        )
        if return_surface_flux_response:
            integrated_surface_flux += np.einsum(
                "we,we,w->e",
                surface_flux_operator,
                derivative[start:stop],
                trapezoid_weight[start:stop],
                optimize=True,
            )
    integrated = np.ascontiguousarray(integrated)
    if return_surface_flux_response:
        return integrated, np.ascontiguousarray(integrated_surface_flux)
    return integrated


def emergent_flux(
    optical_depth: ArrayLike,
    source_function: ArrayLike,
    *,
    n_angle: int = 4,
    backend: Backend = "auto",
) -> FloatArray:
    r"""Calculate outward surface flux for a pure-absorption atmosphere.

    The source function is linear in optical depth within each layer.  The
    lower boundary assumes ``I^+ = S`` at the deepest point.  Returned flux is
    in the same spectral-density units as the source function and obeys
    ``F = 2 pi integral_0^1 I(mu) mu dmu``.
    """

    if backend not in ("auto", "c", "python"):
        raise ValueError("backend must be 'auto', 'c', or 'python'")
    tau, source = _validate_inputs(optical_depth, source_function)
    mu, weight = angular_quadrature(n_angle)

    use_c = backend == "c" or (backend == "auto" and _rt is not None)
    if use_c:
        if _rt is None:
            raise RuntimeError("the compiled radiative-transfer backend is unavailable")
        return np.asarray(
            _rt.emergent_flux(tau, source, mu, weight), dtype=np.float64
        )
    return _python_emergent_flux(tau, source, mu, weight)
