"""Independent fixed-state wavelength/angular energy check for molecular DABs.

The supplied temperature/pressure profile is not relaxed or renormalized.
Passing this audit does not replace a measured small correction or mesh test.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from wd_spectra import adaptive_structure as adaptive
from wd_spectra.constants import STEFAN_BOLTZMANN


class Finished(Exception):pass


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--audit-output',type=Path,required=True)
    args,remaining=parser.parse_known_args()
    if args.audit_output.exists():parser.error('refusing audit overwrite')
    def audit(initial,evaluate,**settings):
        ev=evaluate(initial,False);p=ev.payload
        if not p['energy_balance_is_physical_flux']:
            raise ValueError('audit requires unchanged physical energy equations')
        a=p['atmosphere'];target=STEFAN_BOLTZMANN*a.effective_temperature**4
        flux=float(np.max(abs(p['total_flux_interface']/target-1)))
        local=float(np.max(abs(p['cell_energy_balance_relative_residual'])))
        escape=float(p['lower_boundary_absorption_escape_bound'])
        report=dict(fixed_state=True,new_stationary_correction_measured=False,
            independent_depth_resolution_verified=False,full_dense_physics_validated=False,
            maximum_all_depth_total_flux_residual=flux,
            maximum_relative_cell_energy_balance_residual=local,
            spectrum_integral_over_sigma_teff4=float(p['radiative_flux_interface'][0]/target),
            bottom_absorption_escape_bound=escape,
            source_closure_relative_error=float(p['scattering_source_maximum_relative_residual']),
            flux_and_local_energy_pass=bool(flux<.003 and local<.003 and escape<.003),
            maximum_density=float(np.max(a.mass_density)),
            argv=remaining)
        tr=p['convection_transport']
        if tr is not None:
            # Quantify, rather than guess, whether T-roundoff can resolve the
            # convective flux required by the current local-energy tolerance.
            lt=np.log(a.temperature);dp=np.diff(np.log(a.gas_pressure))
            dg=np.r_[0.,(np.spacing(lt[:-1])+np.spacing(lt[1:]))/dp]
            noise=tr['actual_convective_flux_gradient_derivative']*dg
            report['temperature_roundoff_local_energy_scale']=float(np.max(
                (abs(noise[:-1])+abs(noise[1:]))/p['cell_energy_scale']))
        args.audit_output.parent.mkdir(parents=True,exist_ok=True)
        args.audit_output.write_text(json.dumps(report,indent=2)+'\n')
        np.savez_compressed(args.audit_output.with_suffix('.npz'),
            temperature=a.temperature,pressure=a.gas_pressure,density=a.mass_density,
            total_flux=p['total_flux_interface'],local_energy=p['cell_energy_balance_relative_residual'])
        if tr is not None:
            np.savez_compressed(args.audit_output.with_name(args.audit_output.stem+'-convection.npz'),
                temperature=a.temperature,pressure=a.gas_pressure,
                **{k:tr[k] for k in ('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient',
                                     'actual_convective_flux_gradient_derivative')},
                gradient=p['temperature_gradient'],thermal=p['thermal_cell_emission'],
                scale=p['cell_energy_scale'],convective_flux=p['convective_flux_interface'])
        print('FIXED-STATE AUDIT: '+json.dumps(report),flush=True)
        raise Finished
    with patch.object(adaptive,'solve_trust_region_newton',audit):
        sys.argv=[sys.argv[0]]+remaining
        from run_molecular_dab_mass_experiment import main as run
        try:run()
        except Finished:pass


if __name__=='__main__':main()
