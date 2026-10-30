#!/usr/bin/env python3
"""Fresh transport seed and full physical solve of one dense-He DB atmosphere.

The approximate seed is NOT a solution or a fallback. All reported final flux
residuals come from the non-gray physical solver.
"""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
import logging
from pathlib import Path
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from wd_spectra._compat import trapezoid
from wd_spectra.atmosphere import (Atmosphere, helium_continuum_atmosphere,
    radiative_equilibrium_helium_atmosphere)
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.convection import ml2_temperature_gradient_for_total_flux_from_thermodynamics
from wd_spectra.eos import hummer_mihalas_helium_lte, hummer_mihalas_helium_thermodynamics
from wd_spectra.helium import helium_continuum_mass_absorption_coefficient, rosseland_mean_helium_continuum_opacity
from wd_spectra.models import DBConfig, ModelData, compute_db, save_model_result
from wd_spectra.models.common import _jsonable
from wd_spectra.models.stellar import _helium_tables
from wd_spectra.opacity import optical_depth_from_mass_opacity
from wd_spectra.spectrum import planck_lambda_angstrom

STEP_METHOD = "nonlinear-convection-current-energy"
INNER_MAX_EVALUATIONS = 100


def atmosphere_at(teff, pressure, temperature, tau, *, logg=8., column_mass=None):
    if not np.isfinite(logg):
        raise ValueError('logg must be finite')
    try:
        gravity = 10.**float(logg)
    except OverflowError as exc:
        raise ValueError('gravity must be finite and positive') from exc
    if not np.isfinite(gravity) or gravity <= 0:
        raise ValueError('gravity must be finite and positive')
    mass = pressure/gravity if column_mass is None else np.asarray(column_mass, dtype=float)
    state = hummer_mihalas_helium_lte(temperature, pressure, correlated_microfields=True)
    return Atmosphere(teff, logg, tau, mass, temperature, pressure,
                      state.mass_density, np.zeros_like(temperature), np.zeros_like(temperature),
                      state.electron_density, {}, helium_lte_state=state)


def transport_seed(teff, n_depth, bottom_tau, mesh="pressure", *, logg=8.):
    print(f"seed: constructing gray surface boundary for {teff:g} K, {n_depth} layers", flush=True)
    gray = helium_continuum_atmosphere(teff, logg, n_depth=n_depth)
    print(f"seed: gray boundary ready; integrating hydrostatic ML2 transport from "
          f"P={gray.gas_pressure[0]:.6g}, T={gray.temperature[0]:.6g}", flush=True)
    target = STEFAN_BOLTZMANN * teff**4
    evaluations = 0

    def rhs(logp, y):
        nonlocal evaluations
        temp = np.array([np.exp(y[0])])
        pressure = np.array([np.exp(logp)])
        point = atmosphere_at(teff, pressure, temp, np.ones(1), logg=logg)
        opacity = rosseland_mean_helium_continuum_opacity(point, n_frequency=160)
        thermo = hummer_mihalas_helium_thermodynamics(temp, pressure, correlated_microfields=True)
        gradient = ml2_temperature_gradient_for_total_flux_from_thermodynamics(
            point, opacity, target, thermo.specific_heat_constant_pressure,
            thermo.density_temperature_derivative, thermo.adiabatic_temperature_gradient)
        evaluations += 1
        if evaluations % 100 == 0:
            print(f"seed: evaluations={evaluations} P={pressure[0]:.4g} T={temp[0]:.4g} tau={y[1]:.4g}", flush=True)
        return [gradient[0], opacity[0] * pressure[0] / point.gravity]

    def bottom(logp, y):
        return y[1] - bottom_tau
    bottom.terminal = True
    bottom.direction = 1
    initial_logp = np.log(gray.gas_pressure[0])
    integrated = solve_ivp(rhs, (initial_logp, initial_logp + 40),
                           [np.log(gray.temperature[0]), gray.rosseland_optical_depth[0]],
                           rtol=1e-5, atol=[1e-7, 1e-9], max_step=.25, first_step=.01,
                           events=bottom, dense_output=True)
    if not integrated.success or not integrated.t_events[0].size:
        raise RuntimeError(f"Transport seed did not reach the lower optical depth: {integrated.message}")
    logp = np.linspace(initial_logp, integrated.t[-1], n_depth)
    if mesh == "optical":
        requested = np.geomspace(gray.rosseland_optical_depth[0], bottom_tau, n_depth)
        for j in range(1, n_depth-1):
            logp[j] = brentq(lambda x: integrated.sol(x)[1]-requested[j],
                             initial_logp, integrated.t[-1])
    logt, tau = integrated.sol(logp)
    return atmosphere_at(teff, np.exp(logp), np.exp(logt), tau, logg=logg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("temperature", type=int)
    parser.add_argument("--logg", type=float, default=8., help="requested log10 surface acceleration in cgs")
    parser.add_argument("--n-depth", type=int, default=80)
    parser.add_argument("--structure-wavelength-count",type=int,default=600,
                        help="Continuum transfer quadrature count; independent spectrum uses 4000 points")
    parser.add_argument("--bottom-tau", type=float, default=100.)
    parser.add_argument("--mesh", choices=("pressure", "optical"), default="pressure")
    parser.add_argument("--max-iterations", type=int, default=120)
    parser.add_argument("--physical-only", action="store_true",
                        help="Test full physical flux equations from the fresh transport seed")
    parser.add_argument("--inverse-ml2-step", action="store_true",
                        help="Signed convective-velocity proposal coordinates (required)")
    parser.add_argument("--nonlinear-materials", action="store_true")
    parser.add_argument("--stable-transfer", action="store_true",
                        help="Cancellation-resistant field AND direct tangent")
    parser.add_argument("--step-method", choices=(STEP_METHOD,), default=STEP_METHOD,
                        help="Proposal energy rows in the full physical phase")
    parser.add_argument("--no-continuations", action="store_true",
                        help="Enforce the stated per-phase budget without extra formal segments")
    parser.add_argument("--output-root", type=Path, default=Path("results/cool-db-transport-seed-20260904"))
    return run(parser.parse_args())


def run(args):
    args.logg = getattr(args, 'logg', 8.)
    if not np.isfinite(args.logg):
        raise ValueError('logg must be finite')
    if args.structure_wavelength_count < 2:
        raise ValueError('structure wavelength count must be at least two')
    if not args.inverse_ml2_step:
        raise ValueError("The dense physical phase requires --inverse-ml2-step proposals")
    if args.nonlinear_materials and not getattr(args,'experimental_exact_dense_materials',False):
        raise ValueError('Local material table requires exact dense materials with inverse proposals')
    out = args.output_root / str(args.temperature)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "solver.log").exists():
        raise RuntimeError("Refusing to overwrite an experiment")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "solver.log")])
    (out/'experiment-options.json').write_text(json.dumps(
        {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},indent=2)+'\n')
    started = time.monotonic()
    experimental_physics = getattr(args, 'experimental_physics', None)
    seed = transport_seed(args.temperature, args.n_depth, args.bottom_tau, args.mesh, logg=args.logg)
    provenance = "fresh hydrostatic ML2/diffusion ODE; not a saved atmosphere"
    import wd_spectra.adaptive_structure as adaptive
    from wd_spectra.nonlinear import NonlinearEvaluation
    material_context = {}
    original_solve = adaptive.solve_adaptive_lte_structure
    def full_flux(*positional, **kwargs):
        # This qualified worker already owns direct local-energy rows,
        # thermal initialization and measured nonlinear-ML2 proposals.
        # Do not apply the public atomic solver's completion a second time.
        kwargs['enforce_local_energy_balance'] = False
        if args.nonlinear_materials:
            from .convective_materials import MaterialCoefficients
            material_context['materials'] = MaterialCoefficients(
                kwargs['with_temperature'], kwargs['thermodynamics'],
                kwargs['rosseland_opacity'], kwargs['mixing_length_alpha'])
        if args.physical_only:
            kwargs.update(use_convective_gradient_preconditioner=False,
                          project_initial_convective_gradient=False,
                          use_initial_bolometric_rescaling=False)
        if args.no_continuations:
            kwargs['maximum_formal_flux_continuations'] = 0
        kwargs['compute_local_energy_response'] = True
        return original_solve(*positional, **kwargs)
    adaptive.solve_adaptive_lte_structure = full_flux
    original_newton = adaptive.solve_trust_region_newton
    def proposal_newton(initial, evaluate, **options):
        # Neither an analytic seed nor an interpolated/refined structure
        # supplies a measured small correction on the equations just built.
        options["allow_initial_convergence"] = False
        first = evaluate(initial, False)
        if first.payload["energy_balance_is_physical_flux"]:
            print(f"Physical phase: {args.step_method} proposals; all physical gates retained", flush=True)
            from .inverse_ml2_step import inverse_step
            options['step_builder'] = lambda state, ev, jac, radius: inverse_step(
                state, ev, radius, INNER_MAX_EVALUATIONS)
            # The cheap inner model must use current radiation and
            # material responses, not an earlier accepted atmosphere.
            options["jacobian_refresh_interval"] = 1
            target = STEFAN_BOLTZMANN * args.temperature**4
            scale = first.payload["cell_energy_scale"] / target
            physical_evaluate = evaluate
            def evaluate(state, need_jacobian):
                ev = physical_evaluate(state, need_jacobian)
                payload = {**ev.payload, "diagnostic_fixed_cell_scale": scale}
                normalization = payload["cell_energy_scale"]/target
                values = ev.residual.copy()
                cell_defect = (payload["radiative_cell_energy_defect"]
                               + np.diff(payload["convective_flux_interface"])) / target
                values[:-1] -= cell_defect / normalization
                tangent = None
                if ev.jacobian is not None:
                    tangent = ev.jacobian.copy()
                    tangent[:-1] -= (payload["cell_energy_log_temperature_jacobian"]
                                   @ payload["log_temperature_from_state"]
                                   / target / normalization[:, None])
                    tangent[:-1] += ((cell_defect/normalization)[:, None]
                        * (payload["cell_energy_scale_log_temperature_jacobian"]
                           @ payload["log_temperature_from_state"])
                        / target / normalization[:, None])
                return NonlinearEvaluation(values, tangent, payload)
            physical_convergence = options.get("convergence_test")
            local_tolerance = options.get("residual_tolerance", 2e-3)
            def convergence_test(state, ev, step):
                # Fixed inner scales must not hide a growing
                # relative defect as thermal emission changes.
                local = ev.payload["cell_energy_balance_relative_residual"]
                if np.any(~np.isfinite(local)) or np.max(abs(local)) >= local_tolerance:
                    return False
                physical = NonlinearEvaluation(
                    ev.payload["total_flux_interface"]/target-1, None, ev.payload)
                return (physical_convergence is None
                        or physical_convergence(state, physical, step))
            options["convergence_test"] = convergence_test
            if args.nonlinear_materials:
                from .convective_materials import TabulatedMaterials
                original_builder = options['step_builder']
                def material_builder(state, ev, jac, radius):
                    table = TabulatedMaterials(material_context['materials'],
                        np.log(ev.payload['atmosphere'].temperature), radius)
                    actual = NonlinearEvaluation(ev.residual, ev.jacobian,
                        {**ev.payload, 'diagnostic_material_model': table,
                         'diagnostic_positive_radiative_rates':(
                             ev.payload.get('diagnostic_positive_radiative_rates',False)
                             or getattr(args,'experimental_positive_radiative_rates',False))})
                    direction = original_builder(state, actual, jac, radius)
                    if 'diagnostic_augmented_proposed_velocity' in actual.payload:
                        ev.payload['diagnostic_augmented_proposed_velocity'] = (
                            actual.payload['diagnostic_augmented_proposed_velocity'].copy())
                    return direction
                options['step_builder'] = material_builder
            from .measured_step_guard import require_measured_proposal
            options=require_measured_proposal(options)
        return original_newton(initial, evaluate, **options)
    adaptive.solve_trust_region_newton = proposal_newton
    if experimental_physics is not None:
        provenance = experimental_physics + ': ' + provenance
    print(f"{provenance}: Pbottom={seed.gas_pressure[-1]:.6g} Tbottom={seed.temperature[-1]:.6g} "
          f"rhobottom={seed.mass_density[-1]:.6g}", flush=True)
    def save_diagnostic_structure(name, atmosphere, **metadata):
        fields = dict(temperature=atmosphere.temperature, gas_pressure=atmosphere.gas_pressure,
                      column_mass=atmosphere.column_mass,
                      rosseland_optical_depth=atmosphere.rosseland_optical_depth)
        if experimental_physics is not None:
            name = 'experimental-'+name
            fields = {'experimental_'+key: value for key, value in fields.items()}
            metadata['experimental_physics'] = experimental_physics
        np.savez_compressed(out / name, **fields, diagnostic_logg=atmosphere.logg, **metadata)
    save_diagnostic_structure('seed.npz', seed, diagnostic_seed=provenance)
    he_i, he_ii = _helium_tables(ModelData.default())
    from .stable_transfer import stable_transfer_experiment
    with (out / "iterations.jsonl").open("w") as telemetry, \
            (stable_transfer_experiment() if args.stable_transfer else nullcontext()):
        def report(iteration, atmosphere, diagnostics):
            row = dict(elapsed_seconds=time.monotonic() - started, iteration=iteration,
                       diagnostics=_jsonable(diagnostics))
            telemetry.write(json.dumps(row) + "\n")
            telemetry.flush()
            save_diagnostic_structure('latest-iteration.npz', atmosphere)
            print(json.dumps(row), flush=True)
        relaxed = radiative_equilibrium_helium_atmosphere(
            args.temperature, args.logg, stark_table=he_i, helium_ii_stark_table=he_ii,
            n_depth=args.n_depth, n_continuum_wavelength=args.structure_wavelength_count,
            max_iterations=args.max_iterations, initial_temperature=seed.temperature,
            initial_column_mass=seed.column_mass, initial_gas_pressure=seed.gas_pressure,
            initial_rosseland_optical_depth=seed.rosseland_optical_depth,
            resume_supplied_structure_in_formal_flux_phase=False,
            iteration_callback=report)
    relaxed = replace(relaxed, metadata={**relaxed.metadata,
                      "diagnostic_seed": provenance,
                      "diagnostic_logg": args.logg,
                      "diagnostic_stable_transfer": args.stable_transfer,
                      "diagnostic_step_method": args.step_method,
                      "diagnostic_inner_max_evaluations": INNER_MAX_EVALUATIONS,
                      "diagnostic_continuation_requires_evaluated_correction": True,
                      "diagnostic_requires_non_trust_limited_correction": True,
                      "diagnostic_energy_balance_form": "flux-minus-current-local-energy"})
    wave = np.geomspace(100, 1e7, 4000)
    config = DBConfig(effective_temperature=args.temperature, logg=args.logg, quality="production")
    result = compute_db(config, wave, initial_atmosphere=relaxed, relax_atmosphere=False)
    save_model_result(result, out)
    absorption = helium_continuum_mass_absorption_coefficient(
        relaxed, wave, include_electron_scattering=False, include_rayleigh_scattering=False)
    tau = optical_depth_from_mass_opacity(relaxed.column_mass, absorption)
    # Conservative vertical escape estimate: oblique paths are longer.
    leakage = trapezoid(np.pi * planck_lambda_angstrom(wave, relaxed.temperature[-1])
                        * np.exp(-tau[:, -1]), wave) / (STEFAN_BOLTZMANN * args.temperature**4)
    spectrum_ratio = float(trapezoid(result.spectrum.surface_flux_lambda, wave)
                           / (STEFAN_BOLTZMANN * args.temperature**4))
    summary = dict(elapsed_seconds=time.monotonic() - started,
                   effective_temperature=args.temperature, logg=args.logg,
                   n_depth=relaxed.n_depth,
                   physics=(experimental_physics if experimental_physics is not None else
                            "pure-He production EOS/opacity; dense-ionization limitations remain"),
                   converged=bool(relaxed.metadata["radiative_equilibrium_converged"]),
                   maximum_all_depth_total_flux_residual=relaxed.metadata["maximum_all_depth_total_flux_residual"],
                   maximum_relative_cell_energy_balance_residual=relaxed.metadata[
                       "maximum_relative_cell_energy_balance_residual"],
                   structure_grid_local_energy_verified=bool(relaxed.metadata[
                       "maximum_relative_cell_energy_balance_residual"] < 3e-3),
                   independent_local_energy_verified=None,
                   spectrum_integral_over_sigma_teff4=spectrum_ratio,
                   independent_spectrum_flux_verified=bool(abs(spectrum_ratio-1) < .003),
                   bottom_absorption_escape_bound=float(leakage), metadata=_jsonable(relaxed.metadata))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "metadata"}), flush=True)


if __name__ == "__main__":
    main()
