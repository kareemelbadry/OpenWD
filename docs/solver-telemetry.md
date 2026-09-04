# Solver telemetry

Adaptive atmosphere calculations store structured nonlinear diagnostics in
the returned atmosphere metadata. These fields are observational: they do not
change residuals, Jacobians, step acceptance, or convergence tolerances.

`nonlinear_solver_terminal_reason` distinguishes successful convergence from
an iteration limit or trust-region collapse. The possible values are:

- `initial-state-converged`: the supplied state passed all convergence tests
  before a Jacobian was built;
- `residual-and-step-converged`: the residual and most recent accepted step
  passed their tolerances;
- `stationary-warm-start-complete`: an explicitly approximate conditioning
  phase completed after repeated small accepted steps;
- `stationary-residual-converged`: a collapsed trust region was already below
  the step tolerance and the physical residual passed;
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

The final worst residual is localized with depth index, Rosseland depth,
temperature, and convective-flux fraction. These coordinates describe the
last solver segment; a conditioning segment can use a deliberately
approximate residual and must not be interpreted as physical convergence.

## Scattering-source diagnostic

The DA and helium-family adaptive solvers currently retain the recovered
fixed four coherent-scattering source iterations per residual evaluation.
They now measure the relative fixed-point defect using one already-required
final transfer evaluation and store its maximum, wavelength, and depth. This
measurement does not add an iteration or impose a new tolerance. It is meant
to establish whether a future bounded source-convergence criterion is needed
before that behavior is changed.

## Convergence remains authoritative

Telemetry explains how a solve ended; it does not relax the scientific gate.
Production code and regression tests must still require
`radiative_equilibrium_converged`, the all-depth total-flux residual, and the
final maximum logarithmic temperature correction.
