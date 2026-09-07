# Solver telemetry

Adaptive atmosphere calculations store structured nonlinear diagnostics in
the returned atmosphere metadata. These fields are observational: they do not
change residuals, Jacobians, step acceptance, or convergence tolerances.

If no step was accepted and the initial state was not converged,
`radiative_equilibrium_maximum_log_temperature_correction` is `None` (JSON
`null`), meaning no correction was measured. It is neither zero nor infinity.
The unconverged flag and warning remain, and exploratory checkpoints can be
saved normally. A genuinely converged initial state retains a zero correction.
Likewise, an infeasible rejected trial has no measurable residual: its
exported `rejected_trial_residual_maxima` entry is `null`, with depth index -1
and the infeasible-rejection counter retained. Internal solver diagnostics
still use infinity for that sentinel; physical residual values are not changed.

`nonlinear_solver_terminal_reason` distinguishes successful convergence from
an iteration limit or trust-region collapse. The possible values are:

- `initial-state-converged`: the supplied state passed all convergence tests
  before a Jacobian was built;
- `residual-and-step-converged`: the residual and most recent accepted step
  passed their tolerances;
- `stationary-warm-start-complete`: an explicitly approximate conditioning
  phase completed after repeated small accepted steps;
- `stationary-residual-converged`: the residual passes and a computed proposal
  is smaller than the temperature-step tolerance before trust clipping or
  line-search damping. No step is applied (`line_search_factor=0`); the
  reported correction is the measured proposal size, not the trust radius;
- `trust-region-collapsed`: no acceptable direction remained above the
  minimum trust radius;
- `iteration-limit-converged`: the final allowed step satisfied convergence;
- `maximum-iterations-exhausted`: the iteration budget ended without
  satisfying convergence.

`nonlinear_solver_segments` contains one record for every conditioning,
formal-flux, and bounded continuation segment. Each record reports residual
and Jacobian evaluation counts, accepted iterations, rejected trial counts,
analytic or finite-difference rebuilds, trust-radius extrema, final residual
location, and compact iteration/rejected-line-search histories. Aggregate
counts are also stored at the atmosphere-metadata top level.

Automatic phase changes and continuations cannot erase a preceding oversized
temperature correction by calling the next state a zero-step success. They
require an evaluated correction in that case. An explicit public checkpoint
restart still permits the usual initial residual check; research validation
that needs a new stationarity measurement uses `allow_initial_convergence=False`.
Trust-region collapse by itself is never evidence of a small correction.

`precision_polish_requested` and `precision_polish_used` distinguish the
enabled-by-default final-correction policy from its actual activation. Only
an already flux-balanced warm start with an oversized or unmeasured last
temperature step activates it. It uses cancellation-resistant Feautrier
arithmetic for the coupled field, independent source check and tangent,
and removes the fixed Newton penalty in that final phase. It does not
change the discrete equations, clip convective flux, loosen tolerances or
restore a saved atmosphere. Explicit initial formal-flux restarts do not
automatically activate it. This can cost more iterations when it resolves
previously weakly constrained surface temperatures.

The final worst residual is localized with depth index, Rosseland depth,
temperature, and convective-flux fraction. These coordinates describe the
last solver segment; a conditioning segment can use a deliberately
approximate residual and must not be interpreted as physical convergence.

In live callbacks, `maximum_flux_residual_depth_index` locates the worst
physical flux error. The legacy `maximum_correction_depth_index` is retained
as its alias; despite that older name, it does **not** locate the largest
temperature correction.

## Scattering-source diagnostic

The shared adaptive solver now solves coherent scattering with a direct
angular-block Feautrier system. Its temperature/opacity tangent differentiates
that same coupled system. `scattering_source_iterations=1` means one direct
solve, not one Lambda iteration. The source defect is checked by an independent
scalar fixed-source Feautrier calculation, rather than by the algebraic
identity used to construct the source. The maximum defect and its wavelength
and depth are retained.

Roundoff-level negative sources are measured relative to their own wavelength
row, never to a brighter wavelength. Materially negative trial sources raise
`RecoverableEvaluationError`; backtracking rejects that trial and records
`infeasible_trial_rejections`. Other exceptions are not swallowed. These
guards do not replace the physical convergence gate.

## Lower-boundary screening

`lower_boundary_absorption_escape_bound` integrates
`pi B_lambda(T_bottom) exp(-tau_abs,bottom) / (sigma Teff^4)` over the structure
wavelength grid. It is an upper bound on directly escaping thermal boundary
radiation, since oblique paths and scattering lengthen the absorption path.
`lower_boundary_thermalization_verified_by_absorption` reports whether this
sufficient screen is below `lower_boundary_screening_tolerance` (the requested
flux tolerance). Failure of this conservative screen is not itself a measured
flux error; it calls for a deeper-domain calculation.

The shared equilibrium certificate now requires this screening evidence in
addition to the finite-grid energy, source and correction checks. The
helium-family public drivers can extend their own lower mesh when this is
the only failed gate. `adaptive_domain_segments` records node counts and
work in each segment; `radiative_equilibrium_iterations_including_domain_adaptation`
includes all segments. This is not an external-atmosphere continuation.
Depth/wavelength resolution remains separate; do not normalize a deficient
spectrum to conceal it.

## Live progress

### Local cell-energy diagnostic

`maximum_relative_cell_energy_balance_residual` reports the largest absolute
cell imbalance normalized to its thermal emission plus the magnitudes of
convective flux through both faces. The signed radiative exchange is evaluated
directly from absorption times `J-B`, using the same Feautrier control volumes;
the actual convective-flux divergence is included. The last boundary node is
not a cell-energy equation. Final metadata also records the signed array,
the worst cell index, and `maximum_cell_energy_balance_defect_in_stellar_flux`.

This local diagnostic is now an independent convergence gate. A thin cell can have
a substantial relative heating error yet a negligible defect relative to
stellar flux. Conversely, surface bolometric agreement does not certify local
thermal balance. Use the local and global quantities together. The configured
flux tolerance is applied to both; their normalizations remain different.
See `cold-start-numerics-2026-09-07.md` for the completion algorithm and results.

### Cold-start local-energy completion

`rejected-step-phase-handoff` is an unsuccessful phase termination, never an
equilibrium claim. After rejecting a whole flux-only direction, the driver can
move to direct local energy if actual interface flux already passes its gate
but cell energy does not. It carries the unchanged accepted state into the next
phase, not the rejected trial. The same flux-only segment is not retried.

`local_energy_enforcement_requested` records the requested policy;
`local_energy_completion_used` records whether its energy equations were needed.
The original formal-flux phases are retained before this completion. During
`thermal-conditioning`, live `converged` is always false. The separate
`thermal_conditioning` history records pseudo-time steps, temporal defects,
actual flux/local-energy errors and provisional temperature changes; none of
those initialization changes can certify equilibrium. `local-energy-completion`
then measures a full unrestricted static correction.

`radiative_equilibrium_iterations` includes thermal sweeps and static iterations.
Nonlinear residual/Jacobian counts describe the static solver segments; thermal
work is recorded separately, not hidden in those counters. Across lower-domain
extensions, `thermal_sweeps_including_domain_adaptation` sums the provisional
sweeps in all segments. `final_temperature_coordinates` identifies nodal log T
versus the original normalization/gradient representation.

### Experimental convective trial correction

The shared adaptive driver accepts `use_convective_trial_correction=False`.
This remains off by default and is not a separate solver. When explicitly
enabled, a proposed state with local ML2 flux greater than the stellar flux
gets a gradient correction computed with its frozen local ML2 coefficients.
The temperature normalization degree of freedom is retained; gradients and
temperatures change together. EOS, opacity, radiative transfer, and the actual
convective flux are then recalculated before ordinary merit acceptance.
No calculated flux is clipped or replaced. The generic `trial_projector`
hook also checks the actual projected step against the physical trust radius.

`convective_trial_correction_enabled` records the requested setting and
`convective_trial_correction_proposals` counts modified trial proposals,
including ones subsequently rejected. The latter is not an accepted-step
count or a convergence measure. The approximation assumes an outward net
radiative flux when choosing the cap, and is not generally validated where
the radiative flux points inward. It must not become a default merely because
it limits extreme trial fluxes.

### Watching test runs

The protected canaries print a start notice, elapsed time, phase, physical
all-depth residual, temperature correction, and line-search/trust factors.
They explicitly suspend pytest capture for these messages. Jacobian builds
and rejected directions are also shown, so a run remains observable between
accepted steps. The generic solver exposes these extra messages through the
standard `wd_spectra.nonlinear` logger at INFO level; normal library calls stay
quiet unless logging is enabled.

Run independent canary selections in separate processes with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -m canary ...`.
Do not increase a canary's iteration allowance to make a regression pass.

## Convergence remains authoritative

Telemetry explains how a solve ended; it does not relax the scientific gate.
Production code and regression tests must still require
`radiative_equilibrium_converged`, the all-depth total-flux residual, and the
final maximum logarithmic temperature correction.
