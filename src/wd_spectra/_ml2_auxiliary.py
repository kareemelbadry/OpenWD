"""Algebraic ML2 element-velocity coordinates for the convection proposals.

The signed element-velocity coordinate separates efficient-convection
stiffness from temperature differences. No material coefficient is modified.
"""

from __future__ import annotations

import numpy as np


def ml2_scaled_coefficients(loss, coefficient, target_flux):
    """Return a,b with nabla-nabla_ad = a*y + b*y*abs(y).

    For y>=0, the original ML2 relation gives F_conv/F_star=y**3 exactly.
    The signed extension for y<0 parameterizes stable, zero-convection cells.
    It does not extend convective flux into those cells.
    """
    if target_flux <= 0 or not np.isfinite(target_flux):
        raise ValueError("auxiliary ML2 requires a positive finite stellar flux")
    if np.any(loss < 0) or np.any(~np.isfinite(loss)):
        raise ValueError("auxiliary ML2 requires finite nonnegative radiative loss")
    if np.any(coefficient <= 0) or np.any(~np.isfinite(coefficient)):
        raise ValueError("auxiliary ML2 requires positive finite flux coefficients")
    velocity_scale = np.cbrt(target_flux / coefficient)
    return loss * velocity_scale, velocity_scale**2


def ml2_auxiliary_from_gradient(gradient, adiabatic, loss, coefficient, target_flux):
    a, b = ml2_scaled_coefficients(loss, coefficient, target_flux)
    excess = np.asarray(gradient) - adiabatic
    magnitude = np.abs(excess)
    denominator = a + np.sqrt(a**2 + 4 * b * magnitude)
    root = np.divide(2 * magnitude, denominator, out=np.zeros_like(magnitude), where=denominator > 0)
    return np.sign(excess) * root


def ml2_auxiliary_compatibility(gradient, y, coefficients, responses, gradient_jacobian, target_flux):
    """Gradient compatibility and its exact material/coordinate tangent.

    Coefficients are (adiabatic gradient, radiative loss, flux coefficient).
    Responses, when supplied, differentiate those same arrays with respect
    to the temperature-state variables. The normalization is left to callers.
    """
    adiabatic, loss, coefficient = coefficients
    a, b = ml2_scaled_coefficients(loss, coefficient, target_flux)
    defect = gradient - adiabatic - a * y - b * y * np.abs(y)
    y_response = -a - 2 * b * np.abs(y)
    if responses is None:
        return defect, None, y_response
    ad_response, loss_response, coefficient_response = responses
    velocity_scale = np.cbrt(target_flux / coefficient)
    log_velocity_response = -coefficient_response / (3 * coefficient[:, None])
    a_response = loss_response * velocity_scale[:, None] + a[:, None] * log_velocity_response
    b_response = 2 * b[:, None] * log_velocity_response
    temperature_response = (
        gradient_jacobian - ad_response - y[:, None] * a_response
        - (y * np.abs(y))[:, None] * b_response
    )
    return defect, temperature_response, y_response
