"""Extend structure quadrature beyond 10 micron without moving existing nodes.

The original thermal interval is inadequate for opacity-weighted heating and
cooling, even when it captures nearly all bolometric flux. This scoped research
wrapper extends only that continuum quadrature, at the existing log spacing;
all line grids, opacity callbacks, physics and convergence gates are retained.
"""
import argparse
from contextlib import contextmanager
import sys
from unittest.mock import patch
import numpy as np
from wd_spectra import atmosphere


def extend_grid(grid,maximum):
    grid=np.asarray(grid)
    if maximum<=grid[-1]: raise ValueError('thermal extension must exceed the existing upper wavelength')
    step=np.log(grid[-1]/grid[-2])
    count=int(np.ceil(np.log(maximum/grid[-1])/step))
    tail=np.geomspace(grid[-1],maximum,count+1)[1:]
    return np.r_[grid,tail]


class StructureNumpy:
    def __init__(self,original,maximum):
        self.original,self.maximum=original,maximum
    def __getattr__(self,name):return getattr(self.original,name)
    def geomspace(self,start,stop,num=50,*args,**kwargs):
        grid=self.original.geomspace(start,stop,num,*args,**kwargs)
        if start==100. and stop==100000.:
            extended=extend_grid(grid,self.maximum)
            print(f'EXTENDED THERMAL QUADRATURE: {len(grid)} -> {len(extended)} continuum nodes; '
                  f'upper wavelength {self.maximum:g} A; original nodes unchanged',flush=True)
            return extended
        return grid


@contextmanager
def extended_quadrature(maximum):
    with patch.object(atmosphere,'np',StructureNumpy(atmosphere.np,maximum)):
        yield


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--thermal-maximum-wavelength',type=float,default=1e8)
    parser.add_argument('--structure-angles',type=int,default=3)
    parser.add_argument('--direct-spectrum',action='store_true')
    parser.add_argument('--mass-conservative-transfer',action='store_true')
    parser.add_argument('--smooth-heminus-join',action='store_true')
    parser.add_argument('--pseudo-time-sweeps',type=int,default=0)
    parser.add_argument('--pseudo-time-output',type=str)
    parser.add_argument('--pseudo-time-implicit-convection',action='store_true')
    parser.add_argument('--pseudo-time-until-local-balance',action='store_true')
    args,remaining=parser.parse_known_args()
    if args.pseudo_time_until_local_balance and not args.pseudo_time_sweeps:
        parser.error('local-balance handoff requires explicit pseudo-time initialization')
    if args.pseudo_time_implicit_convection and (not args.pseudo_time_sweeps or '--exact-convection-tangent' not in remaining):
        parser.error('implicit pseudo-time convection requires pseudo-time sweeps and exact material tangent')
    if args.pseudo_time_sweeps<0 or bool(args.pseudo_time_sweeps)!=bool(args.pseudo_time_output):
        parser.error('pseudo-time initialization requires positive sweeps and a separate telemetry output')
    if args.mass_conservative_transfer and ('--stable-transfer' not in remaining or not args.direct_spectrum):
        parser.error('mass-conservative transfer requires stable-transfer scope and consistent direct synthesis')
    if not np.isfinite(args.thermal_maximum_wavelength) or args.thermal_maximum_wavelength<=1e5:
        parser.error('thermal upper wavelength must be finite and exceed 1e5 A')
    if args.structure_angles<2:parser.error('at least two structure angles are required')
    from contextlib import ExitStack
    from wd_spectra import adaptive_structure as adaptive
    original=adaptive.solve_adaptive_lte_structure
    def record(seed,wave,**options):
        if wave[-1]!=args.thermal_maximum_wavelength:
            raise ValueError('structure wavelength extension was not applied')
        options['metadata']={**options.get('metadata',{}),
            'experimental_thermal_wavelength_maximum_angstrom':float(wave[-1]),
            'experimental_thermal_wavelength_count':len(wave),
            'experimental_structure_n_angle':options['n_angle']}
        from .heminus_join import METADATA_KEY,active_policy
        options['metadata'][METADATA_KEY]=active_policy()
        if args.mass_conservative_transfer:
            options['compute_local_energy_response']=True
            options['metadata']['experimental_mass_conservative_transfer']=True
        if args.pseudo_time_sweeps:
            options['metadata'].update(experimental_pseudo_time_initializer_sweeps=args.pseudo_time_sweeps,
                experimental_pseudo_time_nonlinear_convection=args.pseudo_time_implicit_convection,
                experimental_pseudo_time_handoff_on_local_balance=args.pseudo_time_until_local_balance,
                experimental_pseudo_time_is_physical_evolution=False,
                experimental_pseudo_time_is_static_convergence=False)
        return original(seed,wave,**options)
    with ExitStack() as stack:
        if args.smooth_heminus_join:
            from .heminus_join import heminus_join_scope
            stack.enter_context(heminus_join_scope())
            print('EXPLICIT He-minus opacity join experiment: smooth 0.5063--1 micron overlap; '
                  'both source prescriptions and all other opacity ranges unchanged',flush=True)
        if args.pseudo_time_sweeps:
            from .pseudo_time import pseudo_time_initializer
            stack.enter_context(pseudo_time_initializer(args.pseudo_time_sweeps,args.pseudo_time_output,
                implicit_convection=args.pseudo_time_implicit_convection,
                until_local_balance=args.pseudo_time_until_local_balance))
        if args.mass_conservative_transfer:
            from . import stable_transfer as stable
            from .mass_transfer import mass_transfer_experiment
            stack.enter_context(patch.object(stable,'stable_transfer_experiment',mass_transfer_experiment))
        stack.enter_context(extended_quadrature(args.thermal_maximum_wavelength))
        stack.enter_context(patch.object(adaptive,'solve_adaptive_lte_structure',record))
        from functools import partial
        from . import dense_db_runner as runner
        # The worker already chose its structure and formal-transfer grid.
        # Do not run the public gray-seed selector inside the dense EOS scope
        # when compute_db performs fixed-atmosphere synthesis below.
        stack.enter_context(patch.object(runner,'DBConfig',partial(runner.DBConfig,
            transfer_discretization=('column-mass' if args.mass_conservative_transfer else 'optical-depth'))))
        if args.direct_spectrum:
            from .dense_direct_spectrum import direct_result
            original_compute=runner.compute_db
            def direct_compute(*arguments,**options):
                return direct_result(original_compute(*arguments,**options),args.structure_angles,
                    mass_conservative=args.mass_conservative_transfer)
            stack.enter_context(patch.object(runner,'compute_db',direct_compute))
        original_run=runner.run
        def labeled_run(settings):
            settings.experimental_thermal_driver_options=vars(args).copy()
            return original_run(settings)
        stack.enter_context(patch.object(runner,'run',labeled_run))
        # Pass the worker's choice through the public helium entry point;
        # its default must not conflict with the scoped mass-transfer guard.
        stack.enter_context(patch.object(runner,'radiative_equilibrium_helium_atmosphere',
            partial(runner.radiative_equilibrium_helium_atmosphere,n_angle=args.structure_angles,
                transfer_discretization=('column-mass' if args.mass_conservative_transfer else 'optical-depth'))))
        from . import dense_helium_eos as experiment
        sys.argv=[sys.argv[0]]+remaining
        experiment.main()


if __name__=='__main__':main()
