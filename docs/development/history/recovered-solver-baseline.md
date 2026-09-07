# Recovered solver baseline

> Research record: this page describes work at the time it was written.
> For current usage and status, see the [user guide](../../getting-started.md)
> and [tested points](../../tested-temperature-ranges.md).

OpenWD 0.1.1 checkpoints the conservative atmosphere solver recovered and
revalidated on 2026-09-03. This is a stability baseline, not a claim that the
solver is complete. In particular, cooler hydrogen-rich and helium-rich
models still require systematic investigation.

## Protected convergence cases

The recovery workspace exercised these public `quality="production"` cold
starts with no atmosphere checkpoint or fallback. The helium models used 80
depth points and the hydrogen models used 100:

| model | iterations | surface flux / target | maximum flux residual | maximum final `abs(dln T)` |
| --- | ---: | ---: | ---: | ---: |
| DB, 10,000 K, log g=8 | 18 | 1.00000004 | 3.695e-6 | 2.26e-4 |
| DB, 22,000 K, log g=8 | 45 | 0.99999927 | 2.73e-3 | 1.01e-4 |
| DA, 5,000 K, log g=8 | 29 | 0.99999984 | 5.05e-4 | 1.87e-4 |
| DA, 20,000 K, log g=8 | 26 | 1.00000078 | 3.23e-6 | 6.37e-5 |

These are deliberately exact regression cases, not evidence that every
resolution and stellar parameter currently converges.

Paper-era DAB spectra at 9,000 K and 20,000 K were reproduced bit for bit
from their composition-matched converged atmospheres. The PG 1225-079 DZ
spectrum was reproduced to floating-point roundoff.

For SDSS J0738+1835, the solver was restarted from an explicitly unconverged
metal-polluted helium checkpoint and converged without substituting the final
paper atmosphere. Across two consecutive saved solver segments it took
24 + 40 reported iterations and reached:

- surface flux / target: 1.0000012676;
- maximum total-flux residual: 1.268e-6;
- maximum final `abs(dln T)`: 4.170e-5.

The regenerated normalized spectrum retained a mean line-window RMS of
0.033688, compared with 0.033597 for the archived paper spectrum.

## Safety guardrails added in 0.1.2

- One-shot calls still return exploratory spectra, but warn unless the
  atmosphere records successful convergence.
- Saved atmosphere files retain convergence metadata and a full model-request
  fingerprint. Only an exact configuration, physics revision, and data-root
  match can resume directly in the final formal-flux phase.
- The recovered conservative helium structure-line screening, accidentally
  omitted from the 0.1.1 release tree, is restored. It removes only lines whose
  strict upper bound on line-centre optical depth is below `1e-3`; final
  spectrum synthesis remains unscreened.
- The production cold starts in the table above are executable weekly/manual
  canaries, and the ordinary test suite is forced to import this checkout.

## Solver observability and phase verification added in 0.1.3

During the 0.1.2 guardrail work, the default 40-layer
`quality="standard"` DB model at 22,000 K appeared to stop unconverged after
20 reported iterations. Structured telemetry established that this was not a
trust-region stall: 19 steps were accepted, only one direction was rejected,
and the all-depth physical flux residual had already reached `9.018e-4`.

The 20-step limit belonged to an approximate convective-gradient
preconditioner. A conditional error allowed that phase to return directly
when its incidental physical-flux residual was already below tolerance,
without entering the authoritative exact formal-flux phase. The common
adaptive path now always performs that exact
verification after an approximate convective warm start. The unchanged
40-layer atmosphere passes it at iteration zero, with no weakened tolerance
or fallback, and is protected by its own canary.

The same release adds structured terminal reasons, evaluation/rejection
counts, iteration histories, worst-depth physical context, and a
non-mutating diagnostic of the fixed four-step scattering-source residual.

## Numerical changes in this checkpoint

- DA, DB, homogeneous DAB/DBA, and DZ production atmospheres now route their
  composition-specific EOS, absorption, scattering, Rosseland-opacity, and
  thermodynamic callbacks through one adaptive LTE structure driver. The
  duplicated DA trust-region/transfer implementation has been removed. The
  legacy DA and helium Lambda solvers remain available as explicitly selected
  reference paths; they are not production defaults.
- The common driver retains the validated DA cold-start projection as an
  explicit seed policy and distinguishes a supplied warm start from an exact
  checkpoint resume. These are initialization choices only: all compositions
  use the same final conservative formal-transfer flux equation and
  convergence checks.
- The shared LTE solver uses an analytic ML2 convective-flux derivative,
  avoiding finite differences across the convection boundary.
- The completion Jacobian includes the actual local convective-flux response
  as well as the radiative response for every composition.
- Convective-gradient projection and bolometric warm starts are guarded by
  backtracking and full-residual checks.
- A deep diffusion/ML2 equation is used only as a bounded conditioner. Final
  convergence always requires conservative formal-transfer total flux at
  every depth.
- Composition-matched checkpoint resumes preserve their normalization and
  receive the same bounded formal-flux continuation used after fresh starts.

## Regression rule for future work

Changes aimed at cooler stars should first be evaluated on a separate branch.
They should not become the default unless the protected cases above still
converge without fallback and their spectra remain within explicitly recorded
tolerances. A reduced unit-test pass is not a substitute for these end-to-end
atmosphere checks.
