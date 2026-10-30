"""Finite-material ML2 coefficients for the dense-He convection proposals.

MaterialCoefficients evaluates the production EOS and Rosseland opacity at
trial temperatures; TabulatedMaterials is a local smooth table of them that
is checked against independent exact evaluations. Neither supplies the
accepted EOS, opacity, flux or convergence diagnostic.
"""
from dataclasses import replace
import logging
import numpy as np

from wd_spectra.adaptive_structure import (
    _positive_interface_values,
    _arithmetic_interface_values,
)
from wd_spectra.convection import _ml2_local_coefficients_from_thermodynamics

LOGGER = logging.getLogger(__name__)


class MaterialCoefficients:
    """Exact production material coefficients; local, fixed-pressure tangent."""

    def __init__(self, with_temperature, thermodynamics, rosseland_opacity, alpha):
        self.with_temperature = with_temperature
        self.thermodynamics = thermodynamics
        self.rosseland_opacity = rosseland_opacity
        self.alpha = alpha

    def fields(self, logt):
        atmosphere = self.with_temperature(np.exp(logt))
        thermo = self.thermodynamics(atmosphere)
        return (
            atmosphere,
            np.asarray(
                (
                    atmosphere.mass_density,
                    self.rosseland_opacity(atmosphere),
                    thermo.specific_heat_constant_pressure,
                    thermo.density_temperature_derivative,
                    thermo.adiabatic_temperature_gradient,
                )
            ),
        )

    def __call__(self, logt, derivative=False):
        atmosphere, fields = self.fields(logt)
        slopes = None
        if derivative:
            h = 2e-4
            slopes = (self.fields(logt + h)[1] - self.fields(logt - h)[1]) / (2 * h)
        return self.assemble(atmosphere, fields, slopes)

    def assemble(self, atmosphere, fields, slopes=None):
        rho, opacity, cp, expansion, adiabatic = fields
        positive, arithmetic = _positive_interface_values, _arithmetic_interface_values
        interface = replace(
            atmosphere,
            temperature=positive(atmosphere.temperature),
            gas_pressure=positive(atmosphere.gas_pressure),
            mass_density=positive(rho),
        )
        coefficients = _ml2_local_coefficients_from_thermodynamics(
            interface,
            positive(opacity),
            arithmetic(cp),
            arithmetic(expansion),
            arithmetic(adiabatic),
            self.alpha,
        )
        if slopes is None:
            return coefficients
        # All node quantities are local at fixed P, so simultaneous node
        # perturbations recover their diagonal derivatives in two EOS calls.
        n = len(atmosphere.temperature)
        average = np.eye(n) * 0.5 + np.eye(n, k=-1) * 0.5
        average[0, 0] = 1.0
        log_rho_j = average * (slopes[0] / rho)[None, :]
        log_opacity_j = average * (slopes[1] / opacity)[None, :]
        log_cp_j = average * slopes[2][None, :] / arithmetic(cp)[:, None]
        log_expansion_j = average * slopes[3][None, :] / arithmetic(expansion)[:, None]
        ad_j = average * slopes[4][None, :]
        tau = (
            positive(opacity) * self.alpha * interface.gas_pressure / interface.gravity
        )
        log_loss_j = (
            3 * average
            - log_cp_j
            - 0.5 * log_rho_j
            - 0.5 * log_expansion_j
            + (2 / (1 + 0.5 * tau ** 2) - 1)[:, None] * log_opacity_j
        )
        log_coefficient_j = log_cp_j + 0.5 * log_rho_j + average + 0.5 * log_expansion_j
        return (
            coefficients,
            (
                ad_j,
                coefficients[1][:, None] * log_loss_j,
                coefficients[2][:, None] * log_coefficient_j,
            ),
        )


class TabulatedMaterials:
    """Smooth finite-material proposal, with an independent accuracy check.

    This table lives only for one local trust region. It never supplies the
    accepted EOS, opacity, flux or convergence diagnostic. No extrapolation.
    """

    def __init__(self, exact, logt, radius):
        from numpy.polynomial import chebyshev as ch

        self.exact, self.logt, self.radius = exact, logt, radius
        self.atmosphere, base = exact.fields(logt)

        def encode(fields):
            return np.vstack((np.log(fields[:4]), fields[4]))

        self.base = encode(base)
        # Keep the existing log-linear Rosseland response: tabulated opacity
        # knots are only piecewise smooth. Only EOS quantities are represented
        # by the smooth potential and included in its accuracy requirement.
        h = 2e-4
        opacity_slope = (
            np.log(exact.fields(logt + h)[1][1]) - np.log(exact.fields(logt - h)[1][1])
        ) / (2 * h)
        for count in (9, 17):
            nodes = np.cos(np.linspace(0, np.pi, count))
            values = []
            for i, node in enumerate(nodes):
                values.append(encode(exact.fields(logt + radius * node)[1]))
                LOGGER.info("Material proposal table: %d/%d nodes", i + 1, count)
            coefficients = ch.chebfit(
                nodes, np.asarray(values).reshape(count, -1), count - 1
            )
            self.coefficients = coefficients.reshape(count, *self.base.shape)
            # Anchor the true base exactly; do not perturb the residual whose
            # tangent/proposal the outer solver requested.
            self.coefficients[0] += self.base - ch.chebval(0.0, self.coefficients)
            self.coefficients[:, 1] = 0.0
            self.coefficients[0, 1] = self.base[1]
            self.coefficients[1, 1] = radius * opacity_slope
            validation_error = 0.0
            worst = None
            for node in (-0.75, -0.25, 0.25, 0.75):
                actual = encode(exact.fields(logt + radius * node)[1])
                errors = abs(actual - ch.chebval(node, self.coefficients))
                errors[
                    1
                ] = 0.0  # Opacity remains the explicitly linearized quantity above.
                if np.max(errors) > validation_error:
                    validation_error = np.max(errors)
                    field, depth = np.unravel_index(np.argmax(errors), errors.shape)
                    worst = (
                        ("rho", "opacity", "Cp", "Q", "nabla_ad")[field],
                        depth,
                        np.exp(logt[depth] + radius * node),
                    )
            LOGGER.info(
                "Material proposal table: %d nodes, independent error %.3g at %s",
                count,
                validation_error,
                worst,
            )
            if validation_error < 1e-8:
                break
        else:
            raise RuntimeError(
                "Local material interpolation failed its accuracy check; no approximate EOS substituted"
            )
        self.derivatives = ch.chebder(self.coefficients) / radius

    def __call__(self, delta):
        from numpy.polynomial import chebyshev as ch

        if np.max(abs(delta)) > self.radius * (1 + 1e-12):
            raise ValueError("Material proposal left its interpolation interval")
        encoded = ch.chebval(delta / self.radius, self.coefficients, tensor=False)
        slopes = ch.chebval(delta / self.radius, self.derivatives, tensor=False)
        fields = np.vstack((np.exp(encoded[:4]), encoded[4]))
        slopes[:4] *= fields[:4]
        atmosphere = replace(
            self.atmosphere,
            temperature=np.exp(self.logt + delta),
            mass_density=fields[0],
        )
        return self.exact.assemble(atmosphere, fields, slopes)
