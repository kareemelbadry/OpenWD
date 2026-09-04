# Recovered solver baseline

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
resolution and stellar parameter currently converges. In particular, during
the 0.1.2 guardrail work the default 40-layer `quality="standard"` DB model at
22,000 K stopped after 20 reported iterations with an all-depth flux residual
of `9.02e-4` but a final maximum `abs(dln T)` of `1.42e-3`; it is correctly
marked unconverged and now produces `AtmosphereConvergenceWarning`. That stall
remains a solver issue for a later tranche and is not hidden by a checkpoint
or weaker convergence criterion.

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

## Numerical changes in this checkpoint

- The shared H/He solver uses an analytic ML2 convective-flux derivative,
  avoiding finite differences across the convection boundary.
- The DA completion Jacobian includes the actual local convective-flux
  response as well as the radiative response.
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
