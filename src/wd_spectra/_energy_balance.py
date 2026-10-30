"""Discrete radiative cell energy balance and its state response."""

from __future__ import annotations

import numpy as np

from ._compat import trapezoid


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
