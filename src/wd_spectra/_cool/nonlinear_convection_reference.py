"""Reference nonlinear-ML2 trial model for testing the dense convection proposals.

It is not used by the dense-DB worker: the augmented and inverse proposal
tests compare their energy rows with this independent construction.
"""
import numpy as np
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.convection import _ml2_contrast_and_root, _ml2_flux_coefficient_response
from .radiative_rates import radiative_rates


def nonlinear_convection_model(evaluation, cell_scale=None, *, direct_energy=False,
                                current_energy=False):
    """Cheap trial model: tangent radiation/materials, nonlinear local ML2.

    Its residual is a proposal model only. Every outer trial still rebuilds
    actual thermodynamics, opacity, radiation and convection.
    """
    payload = evaluation.payload
    target = STEFAN_BOLTZMANN * payload["atmosphere"].effective_temperature**4
    radiation = payload["radiative_flux_interface"] / target
    radiation_jacobian = payload["radiative_flux_log_temperature_jacobian"] / target
    gradient = payload["temperature_gradient"]
    gradient_operator = payload["interface_gradient_operator"]
    transport = payload["convection_transport"]
    ad, loss, coefficient = (transport[k] for k in
                             ("adiabatic_gradient", "ml2_radiative_loss", "ml2_flux_coefficient"))
    ad_response, loss_response, coefficient_response = payload["ml2_coefficient_log_temperature_responses"]
    log_loss_response = loss_response / loss[:, None]
    log_coefficient_response = coefficient_response / coefficient[:, None]
    if cell_scale is not None:
        cell_radiation_jacobian = (
            payload["radiative_cell_energy_log_temperature_jacobian"]/target
            if direct_energy else np.diff(radiation_jacobian, axis=0))
        if current_energy:
            if not direct_energy:
                raise ValueError("Current energy normalization requires direct energy derivatives")
            emission = payload["thermal_cell_emission"]/target
            log_emission_response = (payload["thermal_cell_emission_log_temperature_jacobian"]
                                     / payload["thermal_cell_emission"][:, None])
    def model(delta):
        g = gradient + gradient_operator @ delta
        if 'diagnostic_material_model' in payload:
            (a,b,c), responses = payload['diagnostic_material_model'](delta)
        else:
            a = ad + ad_response @ delta
            b = loss * np.exp(log_loss_response @ delta)
            c = coefficient * np.exp(log_coefficient_response @ delta)
            responses = (ad_response, b[:, None]*log_loss_response, c[:, None]*log_coefficient_response)
        excess = np.maximum(g-a, 0.)
        contrast, root = _ml2_contrast_and_root(excess, b)
        flux = c*contrast**3
        derivative = 3*c*contrast**2/np.maximum(2*root, np.finfo(float).tiny)
        material_jacobian = _ml2_flux_coefficient_response(
            g, (a, b, c), responses)
        convection_jacobian = derivative[:, None]*gradient_operator + material_jacobian
        flux[0] = 0.
        convection_jacobian[0] = 0.
        residual = radiation + radiation_jacobian @ delta + flux/target - 1
        tangent = radiation_jacobian + convection_jacobian/target
        if cell_scale is not None:
            # Use the directly evaluated absorption*(J-B) term at the base
            # state, avoiding subtraction of nearly equal surface fluxes.
            radiative_cell = payload["radiative_cell_energy_defect"]/target
            cell_defect = (radiative_cell + cell_radiation_jacobian@delta
                           + np.diff(flux)/target)
            cell_radiation_response = cell_radiation_jacobian
            normalization = cell_scale
            if direct_energy and payload.get('diagnostic_positive_radiative_rates',False):
                exchange,cell_radiation_response,_,_=radiative_rates(payload,delta)
                cell_defect=exchange+np.diff(flux)/target
            if current_energy:
                exchange,cell_radiation_response,thermal,thermal_response=radiative_rates(payload,delta)
                cell_defect=exchange+np.diff(flux)/target
                normalization = thermal + (flux[:-1]+flux[1:])/target
                normalization_response = (thermal_response
                    + (convection_jacobian[:-1]+convection_jacobian[1:])/target)
            normalized_defect = cell_defect/normalization
            residual[:-1] -= normalized_defect
            tangent[:-1] -= (cell_radiation_response
                            + np.diff(convection_jacobian, axis=0)/target) / normalization[:, None]
            if current_energy:
                tangent[:-1] += normalized_defect[:, None]*normalization_response/normalization[:, None]
        return residual, tangent
    return model
