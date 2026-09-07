"""Equivalent, fixed-grid representations of conservative energy balance."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._compat import trapezoid

FloatArray = NDArray[np.float64]


def cell_conservation_rows(values: FloatArray) -> FloatArray:
    """Map interface flux defects (or their Jacobian) to cell balance rows.

    Rows 0..N-2 are F[i+1]-F[i]; the final row retains the lower-boundary
    flux defect. This fixed, invertible linear transformation has precisely
    the same zero as constant flux at all interfaces. Convection must be
    included in ``values`` before taking differences. It is not a radiative
    equilibrium mask or a replacement for the physical flux convergence gate.

    The nonlinear solver's existing row equilibration then acts on local
    responses, rather than many almost-identical surface-flux derivatives.
    This helper deliberately adds no optical-depth or temperature threshold.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim not in (1, 2) or array.shape[0] < 2:
        raise ValueError("energy-balance rows must be a vector or matrix with at least two rows")
    result = array.copy()
    result[:-1] = np.diff(array, axis=0)
    return result


def discrete_cell_energy_balance(
    wavelength: FloatArray,
    optical_depth: FloatArray,
    planck: FloatArray,
    mean_intensity: FloatArray,
    absorption_fraction: FloatArray,
    convective_flux: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return signed cell-energy defects and local transport-rate scales.

    Use the same control volumes as the Feautrier discretization, including
    its inserted zero-depth surface. The radiative term is evaluated as
    4 pi integral(delta_tau_cell * epsilon * (J-B)), avoiding subtraction of
    nearly equal interface fluxes in optically thin cells. The convective
    contribution is the actual interface-flux difference, never a surrogate.

    The positive scale is thermal emission plus the magnitudes of convective
    flux through both faces. In radiative cells the resulting relative defect
    is the local heating/cooling imbalance. At the last node transfer imposes
    a boundary value instead of a cell equation, so only N-1 cells are returned.
    These diagnostics do not themselves certify domain or grid adequacy.
    """
    radiative_exchange, thermal_emission = discrete_radiative_cell_energy_balance(
        wavelength, optical_depth, planck, mean_intensity, absorption_fraction)
    defect = radiative_exchange + np.diff(convective_flux)
    scale = thermal_emission + np.abs(convective_flux[:-1]) + np.abs(convective_flux[1:])
    return defect, np.maximum(scale, np.finfo(np.float64).tiny)


def discrete_radiative_cell_energy_balance(
    wavelength, optical_depth, planck, mean_intensity, absorption_fraction,
):
    """Signed radiative cell exchange and positive thermal-emission rate."""
    previous_step = np.diff(optical_depth, axis=1, prepend=0.0)
    cell_width = 0.5 * (previous_step[:, :-1] + previous_step[:, 1:])
    weight = 4.0 * np.pi * cell_width * absorption_fraction[:, :-1]
    radiative_exchange = trapezoid(
        weight * (mean_intensity[:, :-1] - planck[:, :-1]), wavelength, axis=0
    )
    thermal_emission = trapezoid(weight * planck[:, :-1], wavelength, axis=0)
    return radiative_exchange, thermal_emission


def integrated_radiative_cell_energy_state_response(
    wavelength, optical_depth, column_mass, planck, mean_intensity, source,
    absorption, scattering, planck_response, absorption_response,
    scattering_response, *, n_angle=4, wavelength_chunk_size=16,
    response_solver=None,
):
    """Return flux, local-energy and thermal-emission tangents on fixed mass nodes.

    Each state variable changes B, kappa_abs and kappa_scat locally at one
    node. Differentiate 4*pi*integral(width*epsilon*(J-B)) directly, including
    opacity motion of the cell faces, epsilon and the coupled scattering
    response of J. This avoids differencing nearly equal flux-Jacobian rows.
    The final node imposes a transfer boundary, so the energy tangent has
    N-1 rows. Wavelength chunking bounds memory, including for scattering.

    This observational response does not choose atmosphere equations or
    alter the current field. The non-scattering limit uses the same coupled
    tangent with zero scattering fraction; the default fast production flux
    tangent need not call this experimental local-energy path.
    """
    from .radiative_transfer import integrated_coherent_scattering_feautrier_state_response

    if response_solver is None:
        response_solver = integrated_coherent_scattering_feautrier_state_response

    wavelength = np.asarray(wavelength, dtype=float)
    tau = np.asarray(optical_depth, dtype=float)
    mass = np.asarray(column_mass, dtype=float)
    arrays = [np.asarray(a, dtype=float) for a in (
        planck, mean_intensity, source, absorption, scattering, planck_response,
        absorption_response, scattering_response)]
    planck, mean, source, absorption, scattering, db, da, ds = arrays
    if (planck.ndim != 2 or planck.shape[1] < 2 or tau.shape != planck.shape
            or any(a.shape != planck.shape or np.any(~np.isfinite(a)) for a in arrays)):
        raise ValueError("cell energy inputs must be finite wavelength-by-depth arrays")
    nw, nd = planck.shape
    if (wavelength.shape != (nw,) or nw < 2 or np.any(~np.isfinite(wavelength))
            or np.any(wavelength <= 0) or np.any(np.diff(wavelength) <= 0)):
        raise ValueError("cell energy wavelengths must be positive and strictly increasing")
    if (mass.shape != (nd,) or np.any(~np.isfinite(mass))
            or np.any(mass <= 0) or np.any(np.diff(mass) <= 0)):
        raise ValueError("cell energy column mass must be positive and strictly increasing")
    extinction = absorption + scattering
    if np.any(absorption < 0) or np.any(scattering < 0) or np.any(extinction <= 0):
        raise ValueError("cell energy opacities must be nonnegative with positive extinction")
    epsilon = absorption / extinction
    dext = da + ds
    deps = (da - epsilon*dext) / extinction
    direct_source = (da*planck + absorption*db + ds*mean - dext*source) / extinction

    # Width_i = .5*(Delta_tau_i + Delta_tau_{i+1}), with an inserted
    # vacuum surface. Opacity motion is local on these control volumes;
    # construct its weights without subtracting cumulative tau responses.
    mass_width_operator = np.zeros((nd-1, nd))
    mass_step = np.diff(mass)
    for i in range(nd-1):
        if i == 0:
            mass_width_operator[i, 0] += .5*mass[0]
        else:
            mass_width_operator[i, i-1:i+1] += .25*mass_step[i-1]
        mass_width_operator[i, i:i+2] += .25*mass_step[i]
    tau_step = np.diff(tau, axis=1, prepend=0.)
    width = .5*(tau_step[:, :-1] + tau_step[:, 1:])
    wavelength_weight = np.empty(nw)
    dw = np.diff(wavelength)
    wavelength_weight[0], wavelength_weight[-1] = .5*dw[0], .5*dw[-1]
    wavelength_weight[1:-1] = .5*(dw[:-1] + dw[1:])
    energy_jacobian = np.zeros((nd-1, nd))
    emission_jacobian = np.zeros_like(energy_jacobian)
    cells = np.arange(nd-1)

    def accumulate(start, stop, mean_response):
        local = slice(start, stop)
        local_width = width[local]
        local_epsilon = epsilon[local, :-1]
        exchange = mean[local, :-1] - planck[local, :-1]
        tangent = (local_width*local_epsilon)[:, :, None] * mean_response[:, :-1, :]
        width_response = mass_width_operator[None, :, :] * dext[local, None, :]
        tangent += (local_epsilon*exchange)[:, :, None] * width_response
        tangent[:, cells, cells] += local_width * (
            deps[local, :-1]*exchange - local_epsilon*db[local, :-1])
        energy_jacobian[:] += 4*np.pi*np.einsum(
            "wdk,w->dk", tangent, wavelength_weight[local], optimize=True)
        emission_tangent = (local_epsilon*planck[local, :-1])[:, :, None]*width_response
        emission_tangent[:, cells, cells] += local_width*(
            deps[local, :-1]*planck[local, :-1] + local_epsilon*db[local, :-1])
        emission_jacobian[:] += 4*np.pi*np.einsum(
            "wdk,w->dk", emission_tangent, wavelength_weight[local], optimize=True)

    flux_jacobian, _, _ = response_solver(
        tau, wavelength, source, direct_source, db, scattering/extinction,
        mass, dext, n_angle=n_angle, wavelength_chunk_size=wavelength_chunk_size,
        return_auxiliary_response=False, mean_response_consumer=accumulate)
    return flux_jacobian, energy_jacobian, emission_jacobian


def locally_scaled_energy_rows(values, cell_scale):
    """Invertible flux-minus-local-divergence rows with fixed positive scales.

    Thin radiative cells are measured against their thermal emission, rather
    than F_star. In thick cells the emission scale is large and ordinary flux
    constancy dominates. The scale is fixed for a nonlinear segment.
    """
    array = np.asarray(values, dtype=float)
    scale = np.asarray(cell_scale, dtype=float)
    if array.ndim not in (1, 2) or array.shape[0] < 2:
        raise ValueError("energy-balance rows must be a vector or matrix with at least two rows")
    if scale.shape != (array.shape[0]-1,) or np.any(scale <= 0) or np.any(~np.isfinite(scale)):
        raise ValueError("local cell scales must be positive finite and cell-sized")
    if array.ndim == 2:
        scale = scale[:, None]
    result = array.copy()
    result[:-1] -= np.diff(array, axis=0) / scale
    return result


def transport_conditioned_rows(values: FloatArray, cell_optical_width: FloatArray) -> FloatArray:
    """Fixed invertible combination of cell balance and flux constancy.

    R[i] = r[i] - r[i+1]/(1+delta_tau[i]); R[-1] = r[-1].
    Thin cells approach local conservation, opaque cells retain flux rows.
    The unit upper-triangular map cannot change the physical root. Widths must
    be fixed for a nonlinear segment, so its Jacobian uses the identical map.
    The transition comes from cell optical thickness, not a fitted Teff split.
    """
    array = np.asarray(values, dtype=np.float64)
    width = np.asarray(cell_optical_width, dtype=np.float64)
    if array.ndim not in (1, 2) or array.shape[0] < 2:
        raise ValueError("energy-balance rows must be a vector or matrix with at least two rows")
    if width.shape != (array.shape[0] - 1,) or np.any(~np.isfinite(width)) or np.any(width < 0):
        raise ValueError("cell optical widths must be finite nonnegative and cell-sized")
    weight = 1.0 / (1.0 + width)
    if array.ndim == 2:
        weight = weight[:, None]
    result = array.copy()
    result[:-1] -= weight * array[1:]
    return result
