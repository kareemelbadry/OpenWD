"""Tests of the experimental mapping, independent of atmospheric data files."""
from dataclasses import dataclass, replace
from types import SimpleNamespace
import numpy as np
import pytest

from convective_consistency_experiment import (
    MaterialCoefficients,
    TabulatedMaterials,
    repair_trial,
)
from wd_spectra._ml2_auxiliary import ml2_auxiliary_from_gradient
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.nonlinear import RecoverableEvaluationError


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


def synthetic_repair():
    n = 4
    logt = np.array([8.0, 8.402, 8.807, 9.214])
    operator = np.eye(n) - np.eye(n, k=-1)
    operator[0] = 0
    target = STEFAN_BOLTZMANN * 5000.0 ** 4
    base_ad = np.full(n, 0.4)
    loss = np.full(n, 0.001)
    coefficient = np.full(n, 1e4 * target)

    def material(log_temperature, derivative=False):
        dt = log_temperature - logt
        a = base_ad + 0.01 * dt + 3 * dt ** 2
        c = coefficient * np.exp(0.5 * dt)
        values = a, loss, c
        if not derivative:
            return values
        return values, (np.diag(0.01 + 6 * dt), np.zeros((n, n)), np.diag(0.5 * c))

    coefficients, responses = material(logt, True)
    p = dict(
        atmosphere=SimpleNamespace(
            temperature=np.exp(logt), effective_temperature=5000.0
        ),
        temperature_gradient=operator @ logt,
        interface_gradient_operator=operator,
        convection_transport=dict(
            zip(
                ("adiabatic_gradient", "ml2_radiative_loss", "ml2_flux_coefficient"),
                coefficients,
            )
        ),
        ml2_coefficient_log_temperature_responses=responses,
    )
    return SimpleNamespace(payload=p), material


def test_repair_recomputes_finite_materials_not_just_the_tangent():
    evaluation, material = synthetic_repair()
    p = evaluation.payload
    delta = np.full(4, 0.02)
    repaired = repair_trial(evaluation, delta, material)
    assert np.max(abs(repaired - delta)) > 1e-3
    assert repaired[0] == pytest.approx(delta[0])
    coefficients = list(material(np.log(p["atmosphere"].temperature)))
    responses = p["ml2_coefficient_log_temperature_responses"]
    coefficients[0] += responses[0] @ delta
    coefficients[2] *= np.exp((responses[2] / coefficients[2][:, None]) @ delta)
    target = STEFAN_BOLTZMANN * 5000.0 ** 4
    predicted = ml2_auxiliary_from_gradient(
        p["temperature_gradient"] + p["interface_gradient_operator"] @ delta,
        *coefficients,
        target
    )
    actual = ml2_auxiliary_from_gradient(
        p["temperature_gradient"] + p["interface_gradient_operator"] @ repaired,
        *material(np.log(p["atmosphere"].temperature) + repaired),
        target
    )
    np.testing.assert_allclose(
        np.maximum(actual[1:], 0) ** 3, np.maximum(predicted[1:], 0) ** 3, atol=1e-7
    )


def test_zero_step_is_exact_identity_and_failure_does_not_return_a_fallback():
    evaluation, material = synthetic_repair()
    np.testing.assert_array_equal(
        repair_trial(evaluation, np.zeros(4), material), np.zeros(4)
    )
    with pytest.raises(RecoverableEvaluationError, match="compatibility failed"):
        repair_trial(evaluation, np.full(4, 0.02), material, maximum_iterations=1)


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
