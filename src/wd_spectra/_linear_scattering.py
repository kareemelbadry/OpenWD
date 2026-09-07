"""Direct source solution for the established piecewise-linear formal transfer.

Construct the exact Lambda operator by differentiating the existing sweeps.
Wavelength chunks bound storage; no iteration budget masquerades as closure.
This retains the original vacuum extrapolation and bottom source boundary.
"""

from __future__ import annotations
import numpy as np
from .radiative_transfer import (
    angular_quadrature,
    radiation_field,
    _validate_inputs,
    _linear_cell_end_source_weight,
    _linear_cell_start_source_weight,
)


def linear_lambda_operator(optical_depth, n_angle=4):
    tau = np.asarray(optical_depth, dtype=float)
    nw, nd = tau.shape
    operator = np.zeros((nw, nd, nd))
    angles, weights = angular_quadrature(n_angle)
    for mu, weight in zip(angles, weights):
        delta = np.diff(tau, axis=1) / mu
        attenuation = np.exp(-delta)
        end = _linear_cell_end_source_weight(delta)
        start = _linear_cell_start_source_weight(delta)
        row = np.zeros((nw, nd))
        row[:, 0] = -np.expm1(-tau[:, 0] / mu)
        operator[:, 0, :] += 0.5 * weight * row
        for i in range(1, nd):
            row *= attenuation[:, i - 1, None]
            row[:, i - 1] += start[:, i - 1]
            row[:, i] += end[:, i - 1]
            operator[:, i, :] += 0.5 * weight * row
        row = np.zeros((nw, nd))
        row[:, -1] = 1.0
        operator[:, -1, :] += 0.5 * weight * row
        for i in range(nd - 2, -1, -1):
            row *= attenuation[:, i, None]
            row[:, i + 1] += start[:, i]
            row[:, i] += end[:, i]
            operator[:, i, :] += 0.5 * weight * row
    return operator


def linear_scattering_source(
    optical_depth,
    planck,
    absorption,
    scattering,
    *,
    n_angle=4,
    wavelength_chunk_size=64,
):
    tau, b = _validate_inputs(optical_depth, planck)
    tau = np.broadcast_to(tau, b.shape)
    a, s = (
        np.broadcast_to(np.asarray(x, dtype=float), b.shape)
        for x in (absorption, scattering)
    )
    total = a + s
    if (
        np.any(~np.isfinite(total))
        or np.any(a < 0)
        or np.any(s < 0)
        or np.any(total <= 0)
    ):
        raise ValueError(
            "scattering opacities must be finite, nonnegative with positive extinction"
        )
    if not isinstance(wavelength_chunk_size, int) or wavelength_chunk_size < 1:
        raise ValueError("wavelength_chunk_size must be positive")
    source = b.copy()
    # Absorption-only rows have S=B exactly; preserve that fast path.
    selected = np.flatnonzero(np.any(s > 0, axis=1))
    for begin in range(0, len(selected), wavelength_chunk_size):
        index = selected[begin : begin + wavelength_chunk_size]
        lam = linear_lambda_operator(tau[index], n_angle)
        matrix = (
            np.eye(b.shape[1])[None, :, :] - (s[index] / total[index])[:, :, None] * lam
        )
        rhs = a[index] / total[index] * b[index]
        source[index] = np.linalg.solve(matrix, rhs[..., None])[..., 0]
    return source, radiation_field(tau, source, n_angle=n_angle)
