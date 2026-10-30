"""Tests of the finite-material ML2 coefficients, independent of atmospheric data files."""
from dataclasses import dataclass, replace
from types import SimpleNamespace
import numpy as np
import pytest

from wd_spectra._cool.convective_materials import (
    MaterialCoefficients,
    TabulatedMaterials,
)


@dataclass
class MaterialState:
    temperature: np.ndarray
    gas_pressure: np.ndarray
    mass_density: np.ndarray
    gravity: float = 1e8


def test_local_material_tangent_matches_independent_node_perturbations():
    logt = np.log(np.array([4000.0, 4500.0, 5500.0, 8000.0]))
    pressure = np.geomspace(1e6, 1e11, 4)

    def atmosphere(t):
        return MaterialState(
            t, pressure, 1e-7 * pressure / t * np.exp(0.02 * (np.log(t / 5000)) ** 2)
        )

    def thermo(a):
        t = a.temperature / 5000
        return SimpleNamespace(
            specific_heat_constant_pressure=1e8 * t ** 0.3,
            density_temperature_derivative=t ** -0.1,
            adiabatic_temperature_gradient=0.3 + 0.05 * np.sin(np.log(t)),
        )

    material = MaterialCoefficients(
        atmosphere, thermo, lambda a: 0.01 * (a.temperature / 5000) ** -2, 1.25
    )
    values, tangent = material(logt, True)
    h = 1e-5
    numerical = [
        np.column_stack(
            [
                (material(logt + h * d)[i] - material(logt - h * d)[i]) / (2 * h)
                for d in np.eye(4)
            ]
        )
        for i in range(3)
    ]
    for j, actual in zip(tangent, numerical):
        np.testing.assert_allclose(j, actual, rtol=3e-8, atol=1e-7)


def test_material_table_differentiates_independent_node_temperatures():
    logt = np.log(np.array([4000.0, 4500.0, 5500.0, 8000.0]))
    pressure = np.geomspace(1e6, 1e11, 4)

    def atmosphere(t):
        return MaterialState(t, pressure, 1e-7 * pressure / t)

    def thermo(a):
        t = a.temperature / 5000
        return SimpleNamespace(
            specific_heat_constant_pressure=1e8 * t ** 0.3,
            density_temperature_derivative=t ** -0.1,
            adiabatic_temperature_gradient=0.3 + 0.05 * np.sin(np.log(t)),
        )

    exact = MaterialCoefficients(
        atmosphere, thermo, lambda a: 0.01 * (a.temperature / 5000) ** -2, 1.25
    )
    table = TabulatedMaterials(exact, logt, 0.04)
    delta = np.array([-0.03, 0.02, 0.01, -0.02])
    values, tangent = table(delta)
    for predicted, actual in zip(values, exact(logt + delta)):
        np.testing.assert_allclose(predicted, actual, rtol=1e-12, atol=1e-12)
    h = 1e-6
    for i in range(3):
        measured = np.column_stack(
            [
                (table(delta + h * d)[0][i] - table(delta - h * d)[0][i]) / (2 * h)
                for d in np.eye(4)
            ]
        )
        np.testing.assert_allclose(tangent[i], measured, rtol=2e-7, atol=1e-8)
    with pytest.raises(ValueError, match="interpolation interval"):
        table(np.full(4, 0.05))
