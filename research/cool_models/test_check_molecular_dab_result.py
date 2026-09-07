from copy import deepcopy
import pytest
from check_molecular_dab_result import convergence_failures


def passing():
    summary=dict(converged=True,maximum_all_depth_total_flux_residual=1e-5,
        maximum_relative_cell_energy_balance_residual=1e-5,bottom_absorption_escape_bound=1e-5)
    metadata=dict(config=dict(include_molecules=True),atmosphere_metadata=dict(
        radiative_equilibrium_converged=True,radiative_equilibrium_maximum_log_temperature_correction=1e-5,
        nonlinear_solver_segments=[dict(phase='formal-radiative-flux')]))
    direct=dict(converged=True,source_closure_scaled_error=1e-15,spectrum_integral_over_sigma_teff4=1.)
    history=[dict(diagnostics=dict(converged=True,solver_phase='formal-radiative-flux',
        trust_radius=.04,maximum_log_temperature_correction=1e-5,line_search_factor=0.))]
    return summary,metadata,direct,history


def test_strict_numerical_evidence_passes():assert not convergence_failures(*passing())


@pytest.mark.parametrize('failure',['flux','local','step','trust','phase','source','unconverged',
    'bottom','molecules','solver_record','transfer_record','telemetry_phase','invalid_radius','invalid_ratio'])
def test_nearly_correct_surface_flux_cannot_hide_failed_gates(failure):
    s,m,d,h=deepcopy(passing());a=m['atmosphere_metadata']
    if failure=='flux':s['maximum_all_depth_total_flux_residual']=.09
    if failure=='local':s['maximum_relative_cell_energy_balance_residual']=.25
    if failure=='step':a['radiative_equilibrium_maximum_log_temperature_correction']=None
    if failure=='trust':h[-1]['diagnostics']['trust_radius']=1e-5
    if failure=='phase':a['nonlinear_solver_segments']=[dict(phase='convective-gradient-preconditioner')]
    if failure=='source':d['source_closure_scaled_error']=1e-5
    if failure=='unconverged':s['converged']=False
    if failure=='bottom':s['bottom_absorption_escape_bound']=.006
    if failure=='molecules':m['config']['include_molecules']=False
    if failure=='solver_record':h[-1]['diagnostics']['converged']=False
    if failure=='transfer_record':d['converged']=False
    if failure=='telemetry_phase':h[-1]['diagnostics']['solver_phase']='initialization-only'
    if failure=='invalid_radius':h[-1]['diagnostics']['trust_radius']=None
    if failure=='invalid_ratio':d['spectrum_integral_over_sigma_teff4']='not recorded'
    assert convergence_failures(s,m,d,h)
