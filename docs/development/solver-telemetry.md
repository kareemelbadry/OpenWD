# Solver telemetry

[Documentation home](../README.md) · [Contributor guide](README.md)

The adaptive atmosphere solver writes diagnostics to the atmosphere metadata.
These diagnostics do not change the residuals, the Jacobians, the step
acceptance or the convergence tolerances. They tell how a solve ended. They
do not replace the convergence checks.

## Convergence remains authoritative

Production code and tests must require these quantities:

- `radiative_equilibrium_converged`;
- the all-depth total-flux residual;
- the final maximum logarithmic temperature correction.

A spectrum that exists is not a converged model.

## Termination

`nonlinear_solver_terminal_reason` gives the reason why the Newton solver
stopped:

| Value | Meaning |
| --- | --- |
| `initial-state-converged` | The supplied state passed all tests before the first Jacobian. |
| `residual-and-step-converged` | The residual and the last accepted step passed their tolerances. |
| `stationary-warm-start-complete` | An approximate conditioning phase ended after small accepted steps. |
| `stationary-residual-converged` | The residual passed, and the computed proposal was below the temperature-step tolerance before trust clipping. No step was applied (`line_search_factor=0`). |
| `trust-region-collapsed` | No acceptable direction remained above the minimum trust radius. |
| `iteration-limit-converged` | The last allowed step satisfied convergence. |
| `maximum-iterations-exhausted` | The iteration budget ended before convergence. |

A trust-region collapse is not evidence of a small correction. An automatic
phase change cannot call an oversized temperature correction a zero-step
success. To force a new stationarity measurement from a supplied state, use
`allow_initial_convergence=False`.

## Segments and work counts

`nonlinear_solver_segments` has one record for each conditioning, formal-flux
and continuation segment. Each record gives:

- residual and Jacobian evaluation counts;
- accepted iterations and rejected trials;
- analytic and finite-difference Jacobian rebuilds;
- the trust-radius range;
- the location of the final largest residual;
- short iteration and line-search histories.

The top level of the metadata gives the sum of the counts. The location of
the worst residual (depth index, Rosseland depth, temperature, convective-flux
fraction) refers to the last segment. A conditioning segment can use an
approximate residual; its values are not physical convergence.

`radiative_equilibrium_iterations` includes the thermal sweeps and the static
iterations. `thermal_sweeps_including_domain_adaptation` and
`radiative_equilibrium_iterations_including_domain_adaptation` include all
lower-domain segments. `final_temperature_coordinates` tells which
temperature representation the final state uses.

## Missing values

- If no step was accepted and the initial state did not converge,
  `radiative_equilibrium_maximum_log_temperature_correction` is `None` (JSON
  `null`). This means "not measured". It is not zero.
- A converged initial state has a zero correction.
- An infeasible rejected trial has no residual. Its entry in
  `rejected_trial_residual_maxima` is `null`, with depth index -1. The counter
  `infeasible_trial_rejections` counts these trials.

## Live callbacks

`maximum_flux_residual_depth_index` gives the depth of the largest flux
error. `maximum_correction_depth_index` is an older alias of the same value.
It does **not** give the depth of the largest temperature correction.

## Local energy

`maximum_relative_cell_energy_balance_residual` is the largest cell energy
imbalance. It is normalized by the thermal emission of the cell plus the
convective flux through both faces. The radiative exchange is the absorption
times `J - B` on the Feautrier control volumes. The last boundary node is not
a cell. The metadata also gives the signed array, the worst cell index and
`maximum_cell_energy_balance_defect_in_stellar_flux`.

The local energy balance is a convergence check. The configured flux
tolerance applies to the local and the global quantities. Their
normalizations are different, so use both.

Phases of the cold-start solve:

- `thermal-conditioning`: pseudo-time steps from the initial state. Live
  `converged` is always false. The `thermal_conditioning` history records the
  steps.
- `local-energy-completion`: a static solve with the local energy equations.
  It measures a full unrestricted correction.
- `rejected-step-phase-handoff`: a phase ended after a rejected direction. It
  passes the last accepted state to the next phase. It never means
  equilibrium.

`local_energy_enforcement_requested` records the requested policy.
`local_energy_completion_used` records whether the solve used it.
[Cold-start numerics](history/cold-start-numerics-2026-09-07.md) describes
the method.

## Final precision correction

`precision_polish_requested` and `precision_polish_used` describe the final
precision correction. It is on by default. It activates only for a
flux-balanced warm start whose last temperature step was too large or not
measured. It uses cancellation-resistant Feautrier arithmetic. It does not
change the discrete equations or the tolerances.

## Scattering source

The solver solves coherent scattering with a direct angular-block Feautrier
system. `scattering_source_iterations=1` means one direct solve. An
independent fixed-source Feautrier calculation checks the source. The
metadata gives the largest defect and its wavelength and depth.

A trial with a materially negative source raises
`RecoverableEvaluationError`. The line search rejects the trial and increments
`infeasible_trial_rejections`. Other exceptions are not caught.

## Lower boundary

`lower_boundary_absorption_escape_bound` integrates
`pi B_lambda(T_bottom) exp(-tau_abs,bottom) / (sigma Teff^4)` over the
structure wavelengths. It is an upper limit on the thermal radiation that
escapes directly from the lower boundary.
`lower_boundary_thermalization_verified_by_absorption` is true when this
limit is below `lower_boundary_screening_tolerance`.

The convergence certificate requires this check. If it is the only failed
check, the helium-family drivers can extend their lower mesh.
`adaptive_domain_segments` records each mesh segment.

## Preconditioner flux handoff

`solve_adaptive_lte_structure` has the option
`use_preconditioner_flux_handoff` (default `True`). The convective-gradient
preconditioner then ends early when two accepted Newton updates balance the
total flux but not the local energy. The formal-flux phase continues from that
state. The diagnostics are in `convective_preconditioner_flux_handoff`. The
[archived record](history/preconditioner-handoff-2026-10-09.md) gives the
evidence.

## Convective trial correction (off by default)

`use_convective_trial_correction=True` changes a trial state whose local ML2
flux is larger than the stellar flux. The correction uses the frozen local ML2
coefficients. The solver then computes the EOS, opacity, transfer and
convective flux again before it accepts the step. The correction is not
validated where the radiative flux points inward.
`convective_trial_correction_enabled` and
`convective_trial_correction_proposals` record its use.

## Progress output

The canary tests print the elapsed time, the phase, the all-depth residual,
the temperature correction, the trust factors, the Jacobian builds and the
rejected directions. The solver writes these messages to the
`wd_spectra.nonlinear` logger at INFO level. Normal library calls do not
print them.

Run canary selections in separate processes:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -m canary tests/test_protected_model_canaries.py
```
