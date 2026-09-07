"""Rosseland quadrature on a fixed, caller-supplied spectral grid."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._compat import trapezoid
from .constants import BOLTZMANN, LIGHT_SPEED, PLANCK

FloatArray = NDArray[np.float64]


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
