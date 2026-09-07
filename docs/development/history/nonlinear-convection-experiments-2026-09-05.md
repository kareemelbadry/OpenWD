# Convergence experiments: nonlinear convection inside the trial step

> Research record: this page describes work at the time it was written.
> For current usage and status, see the [user guide](../../getting-started.md)
> and [tested points](../../tested-temperature-ranges.md).

Follow-up work and stricter local-energy checks are recorded in
`direct-local-energy-experiments-2026-09-05.md`.

This is the next tranche after `solver-review-2026-09-05.md`. Results below
distinguish convergence of the discrete structure equations from an independently
validated spectrum. No new experimental solver is enabled by default, no
temperature switch is introduced, and nothing has been pushed to GitHub.

## Main finding

The ordinary linear Newton model can be catastrophically wrong when a trial
turns on efficient convection. Changing the linear least-squares geometry alone
does not repair that nonlinearity. A cheap inner model that retains nonlinear
ML2 convection gives much better proposals, but still exposes weak constraints
on optically thin temperatures. It is a useful research direction, not yet a
replacement for the protected production solver.

## Changes retained in the production source

1. **Cancellation-safe ML2 arithmetic.** For the positive root of
   `x*x + L*x = excess`, use
   `x = excess / (hypot(L/2, sqrt(excess)) + L/2)`.
   This is algebraically the same closure as before, but avoids subtracting
   nearly equal numbers and avoids squaring a large loss coefficient. The
   flux and its gradient/material derivatives use the same root. No opacity,
   mixing-length parameter, or flux normalization changes. This addresses
   cancellation in the loss-dominated root, **not** the separate loss of
   resolution in `nabla - nabla_ad` in extremely efficient convection.
2. **Optional proposal hook in the shared nonlinear driver.** `step_builder`
   defaults to `None`, which executes the previous Newton construction.
   A supplied direction still passes through the actual temperature-step
   measure, trust limit, physical evaluation, merit/acceptance checks, and
   convergence gates. Invalid or nonfinite directions raise an error.

The root fix does not materially change the initial residual of either cool-DB
test; it is a numerical correctness fix, not the main speedup.

## Isolated research implementation

The outer research workspace contains `scripts/solver_step_experiments.py`,
`scripts/audit_cool_step_strategies.py`, and the expanded
`scripts/check_cool_db_transport_seed.py`. SciPy remains a research dependency;
it was not added to OpenWD's runtime dependencies.

The proposed nodal logarithmic temperature change is `d`. The cheap model uses:

- the current full radiation response, `F_rad(d) = F_rad(0) + J_rad*d`;
- the exact change of discrete temperature gradient, `nabla(d) = nabla(0) + G*d`;
- a linear adiabatic-gradient response and exponential first-order responses
  for the positive ML2 loss/flux coefficients;
- the full positive-branch ML2 quadratic root and cubic convective flux,
  including the stable branch with zero convection.

Its analytic derivative includes both gradient and material responses. A
[bounded trust-region reflective least-squares solve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
constructs a proposal inside the outer nodal-temperature trust radius. Every
outer trial then recomputes the **actual** EOS, opacity, scattering transfer
and ML2 flux. No modeled flux is adopted as measured flux. An inner iteration
limit is reported, not interpreted as atmosphere convergence. Inner timing,
evaluation count, residual and proposed temperature step are logged live.

The experiment refreshes the radiation/material tangent at every accepted
outer step. There is no clipping of convective flux to the stellar flux and
no independent auxiliary flux that can close the budget without a compatible
temperature gradient. Outer trust prediction still uses its linear model;
consistent prediction from the nonlinear proposal model is future work.

## Controlled direction tests

Both methods in each row evaluate the same physical residual on the same named
initial state, with the same maximum temperature trust radius of 0.04.
Errors are `max(abs((F_rad + F_conv)/F_star - 1))`, not percentages.

| State | Initial error | Ordinary Newton full trial | Nonlinear-convection full trial |
| --- | ---: | ---: | ---: |
| Fresh analytic 5000 K, 80 layers | 0.777030 | 4.87e8 | 0.762374 |
| 8000 K optical mesh, explicit 80-to-159 refinement | 0.180742 | about 4504 | 0.00408956 |

The 5000 K Newton error remained about 7.47e6 even after shrinking its direction
to 1/16. Temperature-coordinate regularization and box-bounded linear solves,
with and without row equilibration, also failed these direction tests. None
is promoted. The audits retain both predicted and independently measured
residuals in `results/cool-step-audit-20260905` and
`results/cool-nonlinear-step-audit-20260905` in the research workspace.

## Full bounded solves

All cases are log(g)=8, pure He. A reused `seed.npz` is explicitly verified
as an unrelaxed hydrostatic ML2/diffusion analytic seed, not a converged
atmosphere. Refinements are explicitly identified separately from fresh starts.
The nonlinear-convection fresh starts use physical equations directly, without
the approximate convective warm-start phase or initial bolometric rescaling.

| Experiment | Attempts / seconds | Final actual flux error | Discrete flux-and-temperature gates | Public spectrum / F_star |
| --- | ---: | ---: | --- | ---: |
| Fresh 8000 K, 80 layers, nonlinear convection | 20 / 379 | 9.26e-9 | Pass; last dlnT 2.62e-5 | 0.980900 |
| Fresh 5000 K, 80 layers, nonlinear convection | 12 / 64 | 0.587542 | Fail | 0.921707 |
| Same fresh 5000 K seed, longer independent run | 40 / 133 | 0.262469 | Fail | 0.904795 |
| 8000 K, 80-to-159 optical refinement, nonlinear convection | 12 / 683 | 2.56e-7 | Fail; last dlnT 0.00281 | 0.995328 |
| Fresh 5000 K, local-energy variant | 20 / 87 | 0.728944 | Fail | 0.692154 |

The longer 5000 K run accepted all 40 directions (some after backtracking),
where the previous direct auxiliary solve accepted none from this seed.
That is progress, not convergence. Its ideal/HM seed has bottom density about
267 g/cm3, where the dense-He physical limitations discussed in the preceding
review remain essential. No trace hydrogen or electron floor was added.

### What the timing comparison does and does not show

The earlier 8000 K optical-grid control took 1275 seconds and 38 iterations
(20 approximate warm-start plus 18 physical iterations). Its seed temperature,
pressure, mass and optical-depth arrays are **identical** to the new run's.
The new direct-physical experiment took 379 seconds and 20 iterations. Its
solver accounting is 60 residual evaluations and 21 rejected trial evaluations,
versus 120 and 76 for the historical control. Jacobian builds increased from
10 to 20. Thus fewer bad trials are a concrete improvement.

The wall-time ratio is not a clean 3.4x kernel benchmark: the earlier run
generated its analytic seed, the new controlled experiment explicitly reused
that unrelaxed seed, and concurrent machine load differed. The workflow also
omits the old approximate warm-start phase. An identical-preparation timing
comparison and broader tests are required before promising a general speedup.

### Why the 8000 K result is not ready for promotion

The public spectral integral remains 1.91% low (the historical control was
1.93% low). More seriously, the new local heating diagnostic reaches 1.80
relative to local thermal emission in a very thin cell, despite excellent
all-depth flux constancy. The maximum **absolute** cell energy defect is only
7.54e-9 F_star, so this is not a 180% bolometric error. It means the flux-only
system tolerates poorly determined thin-layer temperatures. The older model's
maximum relative local defect was about 0.16. The new model is therefore not
a demonstration of more accurate temperatures or a validated atmosphere.

The 159-layer refinement likewise cannot be called converged: a tiny flux
defect does not override its excessive temperature correction. Its independent
spectrum still misses the 0.003 flux requirement. Tolerances were not relaxed.

### Additional alternatives rejected in this tranche

- Combining nonlinear convection with the existing row equilibration and
  original-coordinate 1e-8 Tikhonov penalty produced nonproductive directions
  on the 159-layer refinement. It was stopped with no accepted step.
- Combining the nonlinear proposal with fixed, locally normalized energy
  rows did not fix the 5000 K solve. On the 8000 K refinement it made three
  accepted steps, then stalled. At the retained state actual flux error was
  0.002648, local relative heating error 0.163, and dlnT 0.00152. It was stopped
  after about five minutes, with its logs and latest iterate preserved; no
  completed atmosphere/spectrum was substituted for the interrupted run.

The local-energy variant uses direct absorption*(J-B) for its base residual,
but still differences interface-flux Jacobian rows for the tangent. It also
freezes local scales for a segment. These are conditioning experiments, not
evidence that a directly differentiated local energy equation cannot work.

## Protection checks

- Full regular package suite: **356 passed, 5 optional-data skips**; the seven
  expensive canaries were excluded here and run separately.
- **All seven unique protected cold-start canaries passed**, with unchanged
  iteration allowances: DB 10000/22000 K production and 22000 K standard;
  DA 3000/4000/5000/20000 K production. The overlapping test-name selections
  happened to execute the standard DB test twice; this is not eight canaries.
- Nine separate research-model tests check the ML2 proposal, its material
  derivative, convective onset, nodal trust bounds and local-energy transform.
- DAB 9000 K: the saved paper atmosphere passes current equations immediately;
  flux error 8.87e-5. Its spectrum is identical to the preceding review's result.
- DAB 20000 K: re-relaxes from the saved paper atmosphere in three steps;
  flux error 6.31e-7. Relative to the preceding review's re-relaxed spectrum,
  maximum change is 1.58e-6 of the spectral peak (0.102% at the most sensitive
  relative-flux sample). This is not a new 11% regression: the larger deep-core
  difference from the original paper figure was already present and documented.
- PG 1225 and J0738 fixed-paper-atmosphere spectra agree to maximum relative
  differences 1.83e-11 and 4.24e-14. Independently evaluated all-depth flux errors
  are 0.001679 and 0.002233, passing the existing flux gate. These are saved-state
  checks, not cold-start claims.
- `git diff --check` and focused final arithmetic/nonlinear tests pass.

## Recommended next work

1. Directly differentiate the local thermal energy exchange, including
   material and scattering responses, rather than subtracting nearly equal
   interface-flux tangents. Check directional accuracy on the troublesome
   thin layers before another long atmosphere run. Keep nonlinear convection
   in the proposal and retain independent flux/spectral checks.
2. If global steps remain poor, test heat-capacity-weighted pseudo-time
   continuation of the physical energy equation. This is an untested proposal,
   not an implemented improvement. Its local thermal timescale is preferable
   to an effective-temperature switch or fitted convection threshold.
3. Establish depth/domain convergence of the independent spectrum and address
   nonideal helium chemistry, ionization and dense-fluid opacity separately.
   Faster convergence of an under-resolved or physically incomplete model
   must not be presented as scientific validation.

Research outputs are in `results/cool-nonlinear-proposal-*`,
`results/cool-nonlinear-local-energy-*`, and
`results/cool-nonlinear-regularized-optical159-20260905` in the outer workspace.
All experimental jobs were finished or explicitly stopped at this handoff.
