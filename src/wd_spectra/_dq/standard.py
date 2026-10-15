"""DQ physics on the unmodified shared structure solver.

The material keeps the released DQ physics (coupled He/C/C2 chemistry,
refractive transfer, cell-integrated Swan bands, C-A, dense-He continuum) but
replaces the log-linear He-REOS.3 interpolant by its ideal-excess form, whose
c_P, Q and nabla_ad are continuous. The structure is then solved exactly as
for cool DBs: one call of ``solve_adaptive_lte_structure`` per domain, no
patched Newton, proposal or conditioning layers, and the DB lower-domain
recipe (deepen on any screening failure, x4 pressure, restart).
"""
from __future__ import annotations

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
from wd_spectra.eos import HeliumThermodynamics
from wd_spectra.models.common import ModelData, ModelResult, save_model_result
from . import base
from .c2_ca import load_ca_table
from .data import data_root, validate_data
from .dq_refractive_material import RefractiveDQMaterial
from .dq_uv_sampling_experiment import refined_material
from .provenance import source_hashes
from .swan_strength_cache import CachedSwan
from .validation import independent_grid, qualify_spectrum

PROTOCOL = "dq-standard-solver-v1"


class StandardDQMaterial(RefractiveDQMaterial):
    """Released DQ opacity/transfer with smooth REOS and the shared solver."""

    declared_eos_seed = True
    use_flux_gated_handoff = False
    seed_tau_min = 1.0e-12
    carbon_impact_validity_support = False
    carbon_red_wing_boltzmann = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bounded_carbon.impact_validity_support = self.carbon_impact_validity_support
        self.bounded_carbon.red_wing_boltzmann = self.carbon_red_wing_boltzmann
        if self.helium_reos3 is None:
            raise ValueError("standard DQ requires the He-REOS.3 host EOS")
        self.helium_reos3 = self.helium_reos3.with_interpolation("ideal-excess")
        self.experiment_metadata.update(
            protocol=PROTOCOL,
            helium_reos3_interpolation="ideal-excess (PCHIP log P, C1 Hermite log T)",
            thermodynamics="analytic derivatives of the same smooth REOS interpolant",
            structure_solver="solve_adaptive_lte_structure, unpatched",
            seed="gray hydrostatic seed integrated with the declared REOS, tau_min 1e-12",
            carbon_line_support=("He impact profile limited to its Weisskopf validity detuning"
                                 if self.carbon_impact_validity_support else "100 Lorentz HWHM"),
            carbon_red_wing=("ground-state Boltzmann factor exp(-h|dnu|/kT)"
                             if self.carbon_red_wing_boltzmann else "symmetric Voigt"),
        )

    def thermodynamics(self, current):
        current = self.chemistry(current)[0]
        pressure = np.asarray(current.metadata["dq_helium_host_pressure"])
        _, heat, expansion, gradient, inside = self.helium_reos3.thermodynamics(
            pressure, current.temperature
        )
        if not np.all(inside):
            raise ValueError("DQ He-REOS.3 thermodynamics outside supplied EOS domain")
        return HeliumThermodynamics(
            specific_heat_constant_pressure=heat,
            density_temperature_derivative=expansion,
            adiabatic_temperature_gradient=gradient,
        )

    def hydrostatic_seed(self, teff, logg, n_depth):
        return base.helium_continuum_atmosphere(
            teff,
            logg,
            n_depth=n_depth,
            tau_min=self.seed_tau_min,
            correlated_microfields=True,
            metal_database=self.atomic,
            metal_abundances={"C": self.config.log_carbon_to_helium},
            include_dense_helium_metal_ionization=self.config.nonideal_carbon_ionization,
            helium_reos3_table=self.helium_reos3,
        )


def make_material(config, data=None, *, structure_stride=4, material=StandardDQMaterial):
    """Construct the standard DQ material with the released constitutive data."""
    validate_data()
    root = data_root()
    physical = base.DQConfig(
        effective_temperature=config.effective_temperature,
        logg=config.logg,
        log_carbon_to_helium=config.log_carbon_to_helium,
        quality="standard",
        c2_table_path=str(root / "c2-8states-r15000.npz"),
        helium_eos="reos3",
        swan_pressure_shift="blouin2019",
        helium_dense_continuum_path=str(root / "correction.npz"),
    )
    table = base.read_c2_cross_section_table(physical.c2_table_path)
    cls = refined_material(material, 4000, full_continuum=True, structure_stride=structure_stride)
    result = cls(physical, ModelData.default() if data is None else data, table,
                 sampling_r=10000.0, sampling_phase=0.5)
    result.integrated_swan = CachedSwan(table, branches=root / "swan-completed.npz",
                                        cache_grids=192, strength_cache_bytes=1024**3)
    if config.include_c2_ca:
        result.ca_table = load_ca_table(table)
    result.experiment_metadata.update(c2_ca_included=config.include_c2_ca)
    return result


def run_standard(config, output, *, data=None, n_depth=40, max_iterations=150,
                 maximum_domain_expansions=4, synthesize=True, material=None):
    """Cold DQ atmosphere with checkpoints, then the independent spectrum."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    frozen = source_hashes()
    report = dict(schema=1, protocol=PROTOCOL, status="running",
                  requested_parameters={k: getattr(config, k) for k in
                      ("effective_temperature", "logg", "log_carbon_to_helium")},
                  config=asdict(config), n_depth=n_depth, cold_start=True,
                  started_utc=datetime.now(timezone.utc).isoformat(), process_id=os.getpid())

    def record():
        report["seconds"] = time.monotonic() - started
        (output / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def checkpoint(path, atmosphere):
        np.savez_compressed(path, **{key: getattr(atmosphere, key) for key in
            ("temperature", "gas_pressure", "column_mass", "rosseland_optical_depth")})

    record()
    try:
        material = make_material(config, data) if material is None else material
        material.deadline = started + config.maximum_seconds
        report["physics_metadata"] = material.experiment_metadata

        def progress(i, atmosphere, diagnostics):
            row = dict(iteration=i, depths=atmosphere.n_depth,
                       phase=diagnostics.get("solver_phase"),
                       flux=diagnostics.get("maximum_total_flux_residual"),
                       energy=diagnostics.get("maximum_relative_cell_energy_balance_residual"),
                       correction=diagnostics.get("maximum_log_temperature_correction"),
                       seconds=time.monotonic() - started)
            report["latest_iteration"] = row
            checkpoint(output / "diagnostic-latest.npz", atmosphere)
            (output / "progress.json").write_text(json.dumps(row, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            record()
            material.check_budget()

        def solve_segment(*args, **kwargs):
            try:
                atmosphere = material.solve(*args, **kwargs)
            finally:
                gc.collect()
            certificate = atmosphere.metadata.get("equilibrium_certificate", {})
            report.setdefault("domains", []).append(dict(
                depths=atmosphere.n_depth,
                iterations=atmosphere.metadata.get("radiative_equilibrium_iterations"),
                seconds=time.monotonic() - started,
                failures=certificate.get("failures"),
                checks={k: v.get("value") for k, v in certificate.get("checks", {}).items()}))
            checkpoint(output / f"domain-{len(report['domains'])}-{atmosphere.n_depth}.npz", atmosphere)
            record()
            return atmosphere

        atmosphere = solve_with_screened_boundary(
            solve_segment, config.effective_temperature, config.logg,
            n_depth=n_depth, n_continuum=300, max_iterations=max_iterations, n_angle=3,
            iteration_callback=progress, maximum_domain_expansions=maximum_domain_expansions,
            deepen_on_any_screen_failure=True, restart_after_deepening=True,
            deepening_pressure_factor=4.0)
        report["certificate"] = atmosphere.metadata.get("equilibrium_certificate", {})
        report["structure_seconds"] = time.monotonic() - started
        checkpoint(output / "converged.npz", atmosphere)
        if not report["certificate"].get("verified"):
            report["status"] = "unconverged"
            record()
            return atmosphere, None
        report["status"] = "structure_converged"
        record()
        if not synthesize:
            return atmosphere, None
        fine = material.spectrum(atmosphere, independent_grid(), 3)
        np.savez_compressed(output / "independent-spectrum.npz",
                            wavelength=fine.wavelength_angstrom, flux=fine.surface_flux_lambda)
        try:
            report.update(qualify_spectrum(atmosphere, fine, config))
            report["status"] = "completed"
        except Exception as error:
            report["qualification_error"] = str(error)
            report["status"] = "spectrum_failed_qualification"
        report["independent_spectrum_flux_ratio_raw"] = float(
            fine.bolometric_flux / (STEFAN_BOLTZMANN * config.effective_temperature**4))
        report["source_consistency_verified"] = source_hashes() == frozen
        record()
        save_model_result(ModelResult("DQ", atmosphere, fine, config,
                                      {"dq_protocol": PROTOCOL}), output / "model")
        return atmosphere, fine
    except BaseException as error:
        report.update(status=("budget_exhausted" if isinstance(error, TimeoutError) else
                              "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"),
                      error=f"{type(error).__name__}: {error}")
        record()
        raise
