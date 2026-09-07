"""Fixed-atmosphere synthesis with explicit matching molecular/transfer physics.

This does not relax a structure or certify atmosphere convergence. The optional
Allard comparison changes line physics explicitly on the same saved atmosphere.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import numpy as np
from wd_spectra.models import ModelData, load_atmosphere_checkpoint
from wd_spectra.models.stellar import (
    _helium_tables, _da_self_broadening_prescription,
    _allard_lyman_profiles_for_effective_temperature,
)
from wd_spectra.spectrum import synthesize_hydrogen_helium_spectrum
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra._compat import trapezoid
from run_molecular_dab_experiment import physics
from state_sum_h2_experiment import state_sum_h2_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('temperature', type=int)
    parser.add_argument('atmosphere', type=Path)
    parser.add_argument('--log-h-he', type=float, default=-2.)
    parser.add_argument('--angles', type=int, default=16)
    parser.add_argument('--allard-comparison', action='store_true')
    parser.add_argument('--physical-detuning-lyman', action='store_true')
    parser.add_argument('--consistent-stark-edge', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite synthesis experiment')
    args.output.mkdir(parents=True)
    wave = np.unique(np.r_[np.geomspace(100, 1e9, 14000), np.linspace(3500, 7500, 2001)])
    data = ModelData.default()
    he_i, he_ii = _helium_tables(data)
    reports = []
    policy_path=args.atmosphere.parent/'transport-options.json'
    policy=json.loads(policy_path.read_text()) if policy_path.exists() else None
    interpolation_matches=(policy is not None and
        bool(policy.get('physical_detuning_lyman',False))==args.physical_detuning_lyman and
        bool(policy.get('consistent_stark_edge',False))==args.consistent_stark_edge)
    with ExitStack() as stack:
        stack.enter_context(state_sum_h2_experiment())
        if args.physical_detuning_lyman:
            from physical_detuning_stark_experiment import physical_detuning_lyman
            stack.enter_context(physical_detuning_lyman())
        if args.consistent_stark_edge:
            from stark_density_edge_experiment import consistent_stark_density_edge
            stack.enter_context(consistent_stark_density_edge())
        atmosphere = load_atmosphere_checkpoint(args.atmosphere, args.temperature, 8.,
            'mixed', log_hydrogen_to_helium=args.log_h_he, include_molecules=True)
        for label in (('stark', 'allard') if args.allard_comparison else ('stark',)):
            allard = (_allard_lyman_profiles_for_effective_temperature(args.temperature,
                data, minimum_effective_temperature=9000., use_fixed_low_temperature_profile=False)
                if label == 'allard' else None)
            if label == 'allard' and allard is None:
                raise ValueError('requested Allard comparison has no supported table')
            print(f'Synthesizing {args.temperature} K: {label}, column-mass, {args.angles} angles', flush=True)
            spectrum = synthesize_hydrogen_helium_spectrum(atmosphere, wave,
                molecular_h_he=physics(), stark_table=he_i, helium_ii_stark_table=he_ii,
                include_helium_ii_lines=False, unified_allard_table=allard, allard_stark_weight=1.,
                hydrogen_self_broadening_prescription=_da_self_broadening_prescription(args.temperature, None),
                hydrogen_self_broadening_truncation_closure='stark-core',
                transfer_discretization='column-mass', n_angle=args.angles)
            np.savetxt(args.output/f'{label}-spectrum.txt', np.c_[wave, spectrum.surface_flux_lambda])
            row = dict(temperature=args.temperature, log_hydrogen_to_helium=args.log_h_he,
                atmosphere=str(args.atmosphere), fixed_structure=True,
                newly_measured_temperature_correction=False, lyman_profile=label,
                matches_research_structure_line_physics=(label == 'stark' and interpolation_matches),
                atmosphere_sha256=hashlib.sha256(args.atmosphere.read_bytes()).hexdigest(),
                physical_detuning_lyman=args.physical_detuning_lyman,
                consistent_stark_edge=args.consistent_stark_edge,
                transfer='column-mass', angles=args.angles,
                spectrum_integral_over_sigma_teff4=float(trapezoid(spectrum.surface_flux_lambda,wave)
                    /(STEFAN_BOLTZMANN*args.temperature**4)),
                source_closure=spectrum.metadata['independent_radiation_scaled_source_error'])
            reports.append(row)
            print(json.dumps(row), flush=True)
            (args.output/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__ == '__main__':
    main()
