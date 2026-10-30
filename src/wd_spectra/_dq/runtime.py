"""Cold DQ protocol: DQ physics on the unmodified shared structure solver.

The material supplies DQ chemistry, opacity, refractive transfer and EOS
through the solver callbacks. The structure itself is solved exactly as for
cool DBs: one ``solve_adaptive_lte_structure`` call per domain, with the DB
lower-domain recipe (deepen on any screening failure, x4 bottom pressure,
restart the full phase sequence). There is no patched Newton, proposal,
conditioning or coordinate layer, no restart from a saved atmosphere, and no
fitted parameter.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import time

import numpy as np

from wd_spectra._domain import solve_with_screened_boundary
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.models.common import ModelData, ModelResult, save_model_result
from . import base
from .c2_ca import load_ca_table
from .continuum_batches import CONTINUUM_BATCH_SIZE
from .data import data_root, validate_data
from .dq_refractive_material import RefractiveDQMaterial
from .dq_uv_sampling_experiment import refined_material
from . import dq_swan_impact_broadening as swan_broadening
from .provenance import digest, source_hashes
from .swan_strength_cache import CachedSwan
from .validation import independent_grid, qualify_spectrum

PROTOCOL = 'dq-shared-solver-v1'
STRUCTURE_STRIDE = 4
SWAN_STRUCTURE_CACHE_BYTES = 32*1024**2


class ProductionDQMaterial(RefractiveDQMaterial):
    """Released DQ opacity, chemistry and refractive transfer."""

    # Process-memory guard (safety, not physics): deeper refractive domains
    # (44+ layers) exceeded the historical 4 GiB.
    maximum_memory_bytes = 10*1024**3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Red far wings of carbon lines are bounded by the ground-state
        # Boltzmann factor (repulsive C*-He upper states; see metals docs).
        self.bounded_carbon.red_wing_boltzmann = True
        self.experiment_metadata.update(
            protocol=PROTOCOL,
            helium_reos3_interpolation='ideal-excess (PCHIP log P, C1 Hermite log T)',
            thermodynamics='analytic derivatives of the same smooth REOS interpolant',
            structure_solver='solve_adaptive_lte_structure, unpatched',
            seed=f'gray hydrostatic seed with the declared REOS, tau_min {self.seed_tau_min:g}',
            domain_extension='deepen on any screening failure, x4 bottom pressure, restart',
            carbon_red_wing='ground-state Boltzmann factor exp(-h|dnu|/kT)',
            full_physics_validated=False,
        )

    swan_helium_broadening = False

    def swan_column(self, a, j, wavelength):
        if not self.swan_helium_broadening:
            return super().swan_column(a, j, wavelength)
        if not hasattr(self.integrated_swan, 'broadened_cell_average'):
            raise ValueError('Swan He broadening requested, but the supplied Swan line list '
                             'cannot broaden; set swan_helium_broadening=False for it')
        left, right = self.cell_shifts[float(a.gas_pressure[j])]
        return self.integrated_swan.broadened_cell_average(
            wavelength, float(a.temperature[j]), left, right,
            float(a.helium_lte_state.neutral_he_density[j]))

    def solve(self, *args, **kwargs):
        # Structure iterations retain response matrices: keep a small Swan
        # strength cache there, the full one for the final synthesis.
        cache = self.integrated_swan
        budget = cache.strength_cache_bytes if isinstance(cache, CachedSwan) else None
        if budget is not None:
            cache.set_strength_cache_bytes(min(budget, SWAN_STRUCTURE_CACHE_BYTES))
        try:
            return super().solve(*args, **kwargs)
        finally:
            if budget is not None:
                cache.set_strength_cache_bytes(budget)


def make_material(config, data=None, output=None, *, structure_stride=STRUCTURE_STRIDE):
    """Construct the production DQ material; ``output`` is accepted for API compatibility."""
    validate_data()
    root = data_root()
    physical = base.DQConfig(
        effective_temperature=config.effective_temperature,
        logg=config.logg, log_carbon_to_helium=config.log_carbon_to_helium,
        quality='standard', c2_table_path=str(root/'c2-8states-r15000.npz'),
        helium_eos='reos3', swan_pressure_shift='blouin2019',
        helium_dense_continuum_path=str(root/'correction.npz'),
        helium_minus_free_free=getattr(config, 'helium_minus_free_free', 'partial-wave'))
    table = base.read_c2_cross_section_table(physical.c2_table_path)
    cls = refined_material(ProductionDQMaterial, 4000, full_continuum=True,
                           structure_stride=structure_stride)
    material = cls(physical, ModelData.default() if data is None else data, table,
                   sampling_r=10000., sampling_phase=.5)
    broadening = getattr(config, 'swan_helium_broadening', 'none')
    swan_class = swan_broadening.ImpactBroadenedSwan if broadening == 'exomol-weisskopf' else CachedSwan
    material.integrated_swan = swan_class(table, branches=root/'swan-completed.npz',
                                          cache_grids=192, strength_cache_bytes=1024**3)
    material.swan_helium_broadening = broadening == 'exomol-weisskopf'
    material.experiment_metadata.update(
        swan_helium_broadening=(swan_broadening.DESCRIPTION if material.swan_helium_broadening
                                else 'none (thermal Doppler lines only)'))
    if config.include_c2_ca:
        material.ca_table = load_ca_table(table)
    material.experiment_metadata.update(
        c2_ca_included=config.include_c2_ca,
        c2_ca_profile='historical finite-bin rigid-rotor envelope; unshifted',
        c2_ca_strength='Lino da Silva 2024 Einstein coefficients; no fitted multiplier',
        swan_coverage='Hornkohl retained; ExoMol outside per-band v/J envelope',
        swan_line_count=len(material.integrated_swan.nu),
        swan_strength_cache_bytes=material.integrated_swan.strength_cache_bytes,
        swan_structure_strength_cache_bytes=SWAN_STRUCTURE_CACHE_BYTES,
        helium_continuum_batch_size=CONTINUUM_BATCH_SIZE)
    return material


def run_cold(config, output, *, data=None, wavelength=None, n_depth=40,
             max_iterations=150, maximum_domain_expansions=4):
    """Fresh atmosphere, certified structure, then independent fine spectrum."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    frozen = source_hashes()
    params = {key: getattr(config, key) for key in
              ('effective_temperature', 'logg', 'log_carbon_to_helium')}
    report = dict(schema=2, protocol=PROTOCOL,
        status='running', requested_parameters=params, config=asdict(config),
        cold_start=True, external_atmosphere=None, external_structure_grid=None,
        prior_spectrum=None, opacity_scale=1., source_sha256=frozen,
        started_utc=datetime.now(timezone.utc).isoformat(), process_id=os.getpid(),
        independent_depth_convergence=False, full_physics_validation=False)

    def record():
        report['seconds'] = time.monotonic()-started
        (output/'run.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    def checkpoint(path, atmosphere):
        np.savez_compressed(path, **{key: getattr(atmosphere, key) for key in
            ('temperature', 'gas_pressure', 'column_mass', 'rosseland_optical_depth')})

    record()
    try:
        material = make_material(config, ModelData.default() if data is None else data)
        material.deadline = started+config.maximum_seconds
        report['constitutive_data'] = validate_data()
        report['physical_config'] = asdict(material.config)
        report['physics_metadata'] = material.experiment_metadata

        def progress(i, atmosphere, diagnostics):
            row = dict(iteration=i, depths=atmosphere.n_depth,
                phase=diagnostics.get('solver_phase'),
                flux=diagnostics.get('maximum_total_flux_residual'),
                energy=diagnostics.get('maximum_relative_cell_energy_balance_residual'),
                correction=diagnostics.get('maximum_log_temperature_correction'),
                seconds=time.monotonic()-started)
            report['latest_iteration'] = row
            checkpoint(output/'diagnostic-latest.npz', atmosphere)
            (output/'progress.json').write_text(json.dumps(row, indent=2)+'\n')
            print(json.dumps(row), flush=True)
            record()
            material.check_budget()

        def solve_segment(*args, **kwargs):
            try:
                atmosphere = material.solve(*args, **kwargs)
            finally:
                # Solver closures can form cycles around full radiation arrays.
                gc.collect()
            certificate = atmosphere.metadata.get('equilibrium_certificate', {})
            report.setdefault('completed_domain_diagnostics', []).append(dict(
                depths=atmosphere.n_depth,
                iterations=atmosphere.metadata.get('radiative_equilibrium_iterations'),
                seconds=time.monotonic()-started,
                failures=certificate.get('failures'),
                certificate=certificate,
                lower_boundary_response=atmosphere.metadata.get('lower_boundary_absorption_escape_bound')))
            checkpoint(output/f'diagnostic-depths-{atmosphere.n_depth}.npz', atmosphere)
            record()
            return atmosphere

        atmosphere = solve_with_screened_boundary(solve_segment,
            config.effective_temperature, config.logg, n_depth=n_depth, n_continuum=300,
            max_iterations=max_iterations, n_angle=3, iteration_callback=progress,
            maximum_domain_expansions=maximum_domain_expansions,
            deepen_on_any_screen_failure=True, restart_after_deepening=True,
            deepening_pressure_factor=4.0)
        report['certificate'] = atmosphere.metadata.get('equilibrium_certificate', {})
        report['structure_seconds'] = time.monotonic()-started
        if not report['certificate'].get('verified'):
            raise RuntimeError('DQ atmosphere failed its physical certificate')
        material.check_budget()
        np.savez_compressed(output/'structure-grid.npz', wavelength=material.saved_structure_grid)
        structural = material.spectrum(atmosphere, material.saved_structure_grid, 3,
                                       continuum=False)
        report['structure_spectrum_flux_ratio'] = float(structural.bolometric_flux/
            (STEFAN_BOLTZMANN*config.effective_temperature**4))
        report['status'] = 'structure_converged'
        save_model_result(ModelResult('DQ', atmosphere, structural, config,
                          {'spectral_qualification': False}), output/'structure')
        record()
        # The fine grid is also the default output, so it carries the
        # continuum unless a separate output grid was requested.
        fine = material.spectrum(atmosphere, independent_grid(), 3,
                                 continuum=wavelength is None)
        path = output/'independent-spectrum.npz'
        np.savez_compressed(path, wavelength=fine.wavelength_angstrom,
                            flux=fine.surface_flux_lambda)
        report['independent_spectrum_sha256'] = digest(path)
        report.update(qualify_spectrum(atmosphere, fine, config))
        after = source_hashes()
        report['source_consistency_verified'] = after == frozen
        if after != frozen:
            raise RuntimeError('DQ sources changed during calculation; qualification refused')
        if validate_data() != report['constitutive_data']:
            raise RuntimeError('DQ constitutive data changed during calculation')
        # A requested output grid never determines the atmosphere or
        # bypasses the independent, absolute bolometric flux gate.
        spectrum = fine if wavelength is None else material.spectrum(atmosphere, wavelength, 3)
        result = ModelResult('DQ', atmosphere, spectrum, config, dict(
            atmosphere_convergence_status='converged', spectral_qualification=True,
            atmosphere_initialization='self-contained hydrostatic gray cold start',
            dq_protocol=report['protocol'], independent_spectrum_flux_ratio=
                report['independent_spectrum_flux_ratio'],
            cold_start=True, full_physics_validation=False,
            independent_depth_grid_validation=False))
        save_model_result(result, output/'model')
        report['status'] = 'completed'
        record()
        return result
    except BaseException as error:
        report.update(status=('budget_exhausted' if isinstance(error, TimeoutError) else
                             'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'),
                      error=f'{type(error).__name__}: {error}')
        record()
        raise
