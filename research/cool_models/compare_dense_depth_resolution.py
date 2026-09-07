"""Compare independently solved depth grids, including spectral redistribution.

Equal bolometric integrals do not establish depth-grid convergence. Report
both the integrated absolute spectral difference and a log-wavelength
weighted maximum, without fitting a normalization or discarding shape error.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from wd_spectra._compat import trapezoid
from wd_spectra.constants import STEFAN_BOLTZMANN


def validate_discretization(coarse,fine):
    """A depth certificate must not mix simultaneous transfer/grid changes."""
    a,b=coarse['atmosphere'],fine['atmosphere']
    from heminus_join_experiment import METADATA_KEY,HARD_JOIN
    if a.get(METADATA_KEY,HARD_JOIN)!=b.get(METADATA_KEY,HARD_JOIN):
        raise ValueError('depth certification requires the same He-minus opacity join')
    if bool(a.get('experimental_mass_conservative_transfer',False))!=bool(b.get('experimental_mass_conservative_transfer',False)):
        raise ValueError('depth certification requires the same transfer discretization')
    for key in ('experimental_structure_n_angle','experimental_thermal_wavelength_maximum_angstrom',
                'experimental_thermal_wavelength_count'):
        if a.get(key)!=b.get(key):
            raise ValueError('depth certification requires unchanged structure wavelength/angular quadratures')


def compare_spectra(coarse_wave,coarse_flux,fine_wave,fine_flux,target,tolerance=.003):
    lower=max(coarse_wave[0],fine_wave[0]);upper=min(coarse_wave[-1],fine_wave[-1])
    wave=np.unique(np.r_[coarse_wave[(coarse_wave>=lower)&(coarse_wave<=upper)],
        fine_wave[(fine_wave>=lower)&(fine_wave<=upper)]])
    coarse=np.interp(wave,coarse_wave,coarse_flux);fine=np.interp(wave,fine_wave,fine_flux)
    change=fine-coarse
    l1=float(trapezoid(abs(change),wave)/target)
    maximum=float(np.max(wave*abs(change))/target)
    ratios={}
    for name,lo,hi in [('optical_3500_9000',3500.,9000.),('infrared_1_5_micron',10000.,50000.)]:
        chosen=(wave>=lo)&(wave<=hi)
        if np.sum(chosen)>1:
            ratios[name]=float(trapezoid(fine[chosen],wave[chosen])/trapezoid(coarse[chosen],wave[chosen]))
    return dict(independent_depth_resolution_verified=l1<tolerance and maximum<tolerance,
        integrated_absolute_spectral_change_over_target=l1,
        maximum_lambda_weighted_spectral_change_over_target=maximum,
        coarse_flux_ratio=float(trapezoid(coarse,wave)/target),
        fine_flux_ratio=float(trapezoid(fine,wave)/target),fine_over_coarse_band_flux=ratios,
        tolerance=tolerance,comparison_minimum_wavelength=float(lower),comparison_maximum_wavelength=float(upper))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coarse-run',type=Path,required=True)
    parser.add_argument('--coarse-spectrum',type=Path,required=True)
    parser.add_argument('--fine-run',type=Path,required=True)
    parser.add_argument('--fine-spectrum',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('refusing to overwrite depth comparison')
    if args.coarse_run.name!=args.fine_run.name:raise ValueError('depth comparison requires identical Teff')
    metadata=[]
    for path in (args.coarse_run,args.fine_run):
        meta=json.loads((path/'experimental-metadata.json').read_text())
        state=meta['atmosphere']
        if not (state['radiative_equilibrium_converged'] and state['maximum_all_depth_total_flux_residual']<.003
                and state['maximum_relative_cell_energy_balance_residual']<.003
                and state['radiative_equilibrium_maximum_log_temperature_correction']<.0003):
            raise ValueError('both depth grids must pass the physical convergence gates')
        metadata.append(meta)
    if any(metadata[0][k]!=metadata[1][k] for k in ('physics','interaction_table_sha256')):
        raise ValueError('depth comparison requires identical declared physics')
    validate_discretization(*metadata)
    spectra=[np.loadtxt(path).T for path in (args.coarse_spectrum,args.fine_spectrum)]
    if not np.array_equal(spectra[0][0],spectra[1][0]):
        raise ValueError('depth certification requires spectra on identical wavelength nodes')
    if args.coarse_spectrum.name!=args.fine_spectrum.name:
        raise ValueError('depth certification requires matching synthesis method/angle artifact names')
    result=compare_spectra(*spectra[0],*spectra[1],STEFAN_BOLTZMANN*float(args.fine_run.name)**4)
    result.update(coarse_run=str(args.coarse_run.resolve()),fine_run=str(args.fine_run.resolve()),
        coarse_spectrum_sha256=hashlib.sha256(args.coarse_spectrum.read_bytes()).hexdigest(),
        fine_spectrum_sha256=hashlib.sha256(args.fine_spectrum.read_bytes()).hexdigest(),
        validated_full_physics=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
