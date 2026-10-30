"""Tests of the reference nonlinear-ML2 trial model."""
from types import SimpleNamespace

import numpy as np
import pytest

from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra._cool.nonlinear_convection_reference import nonlinear_convection_model


def trial_evaluation():
    """Both stable and convecting interfaces, with nonzero material response."""
    n = 4
    target = STEFAN_BOLTZMANN * 8000.**4
    loss = np.array([.1, .3, .03, .002])
    coefficient = target * np.array([1., 3., 20., 100.])
    gradient = np.array([.6, .2, .5, .45])
    adiabatic = np.full(n, .4)
    operator = np.array([[1., 0., 0., 0.], [-1., 1., 0., 0.],
                         [0., -1., 1., 0.], [0., 0., -1., 1.]])
    response = (np.eye(n)*.03, loss[:, None]*np.eye(n)*.2,
                coefficient[:, None]*np.eye(n)*-.7)
    excess = np.maximum(gradient-adiabatic, 0.)
    contrast = excess / (np.sqrt(.25*loss**2+excess)+.5*loss)
    convective = coefficient*contrast**3
    convective[0] = 0.
    radiation = target*np.array([.95, .9, .5, .6])
    payload = dict(atmosphere=SimpleNamespace(effective_temperature=8000.),
                   radiative_flux_interface=radiation,
                   radiative_flux_log_temperature_jacobian=target*np.eye(n)*.4,
                   temperature_gradient=gradient, interface_gradient_operator=operator,
                   convection_transport=dict(adiabatic_gradient=adiabatic,
                       ml2_radiative_loss=loss, ml2_flux_coefficient=coefficient),
                   ml2_coefficient_log_temperature_responses=response,
                   log_temperature_from_state=np.linalg.inv(operator))
    return SimpleNamespace(payload=payload, residual=(radiation+convective)/target-1.)


def test_model_is_actual_ml2_at_the_expansion_point():
    evaluation = trial_evaluation()
    residual, tangent = nonlinear_convection_model(evaluation)(np.zeros(4))
    np.testing.assert_allclose(residual, evaluation.residual, rtol=1e-14, atol=1e-15)
    # The outer surface has no incoming convective flux, regardless of its
    # provisional gradient or material responses.
    np.testing.assert_array_equal(tangent[0], [.4, 0., 0., 0.])


@pytest.mark.parametrize("delta", [np.zeros(4), np.array([.01, -.02, .03, -.01])])
def test_inner_tangent_matches_finite_differences_including_materials(delta):
    model = nonlinear_convection_model(trial_evaluation())
    _, tangent = model(delta)
    h = 1e-6
    numerical = np.column_stack([
        (model(delta+h*direction)[0]-model(delta-h*direction)[0])/(2*h)
        for direction in np.eye(4)])
    np.testing.assert_allclose(tangent, numerical, rtol=3e-8, atol=2e-9)


def test_model_captures_convective_onset_missed_by_linear_tangent():
    model = nonlinear_convection_model(trial_evaluation())
    base, tangent = model(np.zeros(4))
    # Interface 1 is initially stable: its tangent contains radiation only.
    np.testing.assert_array_equal(tangent[1], [0., .4, 0., 0.])
    delta = np.array([-.2, .2, 0., 0.])
    measured, _ = model(delta)
    assert measured[1] > (base+tangent@delta)[1]+.05


@pytest.mark.parametrize("delta", [np.zeros(4), np.array([.01, -.02, .03, -.01])])
@pytest.mark.parametrize("direct", [False, True])
def test_local_energy_model_is_consistently_differentiated(delta, direct):
    evaluation = trial_evaluation()
    payload = evaluation.payload
    scale = np.array([.002, .3, 5.])
    payload["radiative_cell_energy_defect"] = np.diff(payload["radiative_flux_interface"])
    flux_difference = np.diff(payload["radiative_flux_log_temperature_jacobian"], axis=0)
    # Deliberately different tangents verify the direct field is actually
    # consumed, not silently replaced by the flux difference.
    payload["radiative_cell_energy_log_temperature_jacobian"] = 1.2*flux_difference
    model = nonlinear_convection_model(evaluation, scale, direct_energy=direct)
    residual, tangent = model(delta)
    plain = nonlinear_convection_model(evaluation)(delta)[0]
    expected = plain.copy()
    expected[:-1] -= np.diff(plain)/scale
    if direct:
        expected[:-1] -= .2*flux_difference@delta/(STEFAN_BOLTZMANN*8000.**4)/scale
    np.testing.assert_allclose(residual, expected, atol=1e-12)
    h = 1e-6
    numerical = np.column_stack([
        (model(delta+h*direction)[0]-model(delta-h*direction)[0])/(2*h)
        for direction in np.eye(4)])
    np.testing.assert_allclose(tangent, numerical, rtol=3e-8, atol=2e-8)


def test_positive_radiative_rates_work_with_fixed_equation_weights():
    evaluation=trial_evaluation();p=evaluation.payload
    target=STEFAN_BOLTZMANN*8000.**4
    p['thermal_cell_emission']=target*np.array([.1,.3,20.])
    p['thermal_cell_emission_log_temperature_jacobian']=p['thermal_cell_emission'][:,None]*np.eye(3,4)*.7
    p['radiative_cell_energy_defect']=np.diff(p['radiative_flux_interface'])
    p['radiative_cell_energy_log_temperature_jacobian']=np.diff(p['radiative_flux_log_temperature_jacobian'],axis=0)
    # Keep heating positive, independently of the convection toy model.
    p['radiative_cell_energy_defect']=target*np.array([-.02,.02,.1])
    fixed=np.array([.1,.3,20.])
    plain=nonlinear_convection_model(evaluation,fixed,direct_energy=True)
    base,j=plain(np.zeros(4))
    p['diagnostic_positive_radiative_rates']=True
    model=nonlinear_convection_model(evaluation,fixed,direct_energy=True)
    np.testing.assert_allclose(model(np.zeros(4))[0],base,rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(model(np.zeros(4))[1],j,rtol=1e-14,atol=1e-14)
    delta=np.array([.01,-.02,.03,-.01]);h=1e-6
    measured=np.column_stack([(model(delta+h*d)[0]-model(delta-h*d)[0])/(2*h) for d in np.eye(4)])
    np.testing.assert_allclose(model(delta)[1],measured,rtol=1e-7,atol=1e-8)


@pytest.mark.parametrize("delta", [np.zeros(4), np.array([.01, -.02, .03, -.01])])
def test_current_energy_model_differentiates_its_normalization(delta):
    evaluation = trial_evaluation()
    p = evaluation.payload
    target = STEFAN_BOLTZMANN*8000.**4
    p["thermal_cell_emission"] = target*np.array([.1, .3, 20.])
    p["thermal_cell_emission_log_temperature_jacobian"] = (
        p["thermal_cell_emission"][:, None]*np.eye(3, 4)*.7)
    p["radiative_cell_energy_defect"] = np.diff(p["radiative_flux_interface"])
    p["radiative_cell_energy_log_temperature_jacobian"] = np.diff(
        p["radiative_flux_log_temperature_jacobian"], axis=0)
    model = nonlinear_convection_model(evaluation, np.ones(3),
                                      direct_energy=True, current_energy=True)
    changed_initial_scale = nonlinear_convection_model(evaluation, np.full(3, 1e12),
                                      direct_energy=True, current_energy=True)
    np.testing.assert_array_equal(model(delta)[0], changed_initial_scale(delta)[0])
    h = 1e-6
    numerical = np.column_stack([
        (model(delta+h*d)[0]-model(delta-h*d)[0])/(2*h) for d in np.eye(4)])
    np.testing.assert_allclose(model(delta)[1], numerical, rtol=3e-8, atol=2e-8)
    convective = target*(evaluation.residual+1)-p["radiative_flux_interface"]
    energy = p["radiative_cell_energy_defect"]+np.diff(convective)
    scale = p["thermal_cell_emission"]+convective[:-1]+convective[1:]
    expected = evaluation.residual.copy()
    expected[:-1] -= energy/scale
    np.testing.assert_allclose(model(np.zeros(4))[0], expected, atol=1e-14)
    # The bottom is always the physical flux boundary, not a thermal cell.
    assert model(delta)[0][-1] == nonlinear_convection_model(evaluation)(delta)[0][-1]
