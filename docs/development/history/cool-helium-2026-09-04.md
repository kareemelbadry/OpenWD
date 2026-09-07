# Cool helium investigation (2026-09-04)

> Research record: this page describes work at the time it was written.
> For current usage and status, see the [user guide](../../getting-started.md)
> and [tested points](../../tested-temperature-ranges.md).

## Scope and retained changes

The 5000/8000 K, log(g)=8 pure-He production cold starts were stopped at the
user's request. Their last accepted states remain in the research workspace
under `results/cool-db-current-20260904`. No failed atmosphere was replaced
with a saved successful model, and no convergence tolerance was relaxed.

Changes retained in the production tree:

1. Continue the published declining high-wavenumber branch of the He–He–He
   CIA fit instead of setting it abruptly to zero above 6000 cm^-1
   (lambda=1.6667 micrometers). The join is continuous, including its first
   derivative. Coefficients and the existing temperature domain are unchanged.
   This is an extrapolation of the analytical fit, **not** new optical CIA
   data. Tests cover the join, the old cutoff, the declining tail, density
   scaling, and numerical overflow. The model-request physics revision changes
   so older checkpoints cannot silently claim an identical calculation.
2. Restore the separable H2–H2 CIA tail correction behind the earlier ultracool
   DA figure. The distributed Borysow table ends at 16480 cm^-1 (6068 Angstrom)
   with a declining terminal log-slope in every temperature column. Continue
   that slope without a fitted scale. Other input tables whose endpoints are
   not declining retain their explicitly tabulated domain. This change does
   not add any hydrogen opacity to pure-He DB models. The old solver was not
   restored or imported.
3. Treat non-finite or numerically collapsed cumulative optical-depth
   increments in a nonlinear trial as a recoverable trial-domain error. The
   solver may shorten that step; it never repairs depth by adding fictitious
   opacity or catches unrelated transfer programming errors. An invalid initial
   state still fails explicitly.
4. Record a conservative thermal lower-boundary escape diagnostic separately
   from the finite-grid flux residual. See `solver-telemetry.md` for its
   interpretation. It does not renormalize any spectrum or alter convergence
   tolerances. The historical 10000 K DB also fails this *sufficient* absorption
   screen (bound approximately 0.021); that is a reason for a domain-refinement
   test, not a measured 2.1% error or a claim of a newly introduced regression.

The CIA prescription is based on [Kowalski (2014)](https://arxiv.org/abs/1406.4591).
An arbitrary cutoff at the edge of the plotted calculation is not part of its
two-branch analytic form. Its extension remains uncertain at high frequency
and at densities where higher-order many-body contributions matter.

## What the original cool runs actually showed

| Diagnostic | 5000 K | 8000 K |
| --- | ---: | ---: |
| Last accepted iteration | 238 | 22 |
| Maximum physical all-depth flux error | 1.02e18 | 0.00574 |
| Independently synthesized bolometric flux / sigma Teff^4 | 0.781 | 0.751 |
| Maximum density (g/cm^3) | 400 | 0.172 |

The broad-spectrum integrals were checked on denser wavelength grids extending
from 100 to 1e7 Angstrom; missing wavelength coverage is not the explanation.
Both spectra are explicitly exploratory, not converged models.

At 5000 K the gray hydrostatic seed fixes nearly coincident pressure nodes:
the smallest delta ln(P) is about 5.9e-8. Its temperature gradient can then be
hundreds of thousands, compared with an adiabatic gradient near 0.4. Full ML2
flux consequently becomes enormous. CIA smoothing alone does not cure this.

At 8000 K the fixed gray-seed mass domain becomes optically shallow as the
temperature relaxes. Its lower optical depth is only about 0.3–0.6 through
much of the optical/near-UV. The bottom temperature is only 7666 K. The
Feautrier boundary u=B assumes a thermalized lower boundary; the independent
formal synthesis supplies outgoing intensity B instead. These agree deep in
the diffusion regime, not in a transparent slab. A near-unity Feautrier
surface flux therefore cannot certify this inadequate physical domain.

An isothermal absorption-only slab isolates the problem: at bottom tau=0.3,
the Feautrier surface flux is 1.384 pi B, while outgoing formal transfer gives
pi B. At tau=10 the discrepancy is about 0.2% for the tested mesh. This is a
boundary/domain issue in addition to discretization, not a missing gray
normalization factor.

The public spectrum's four scattering sweeps were also compared against a
dense Lambda solve using the **same** formal-transfer operator. In these two
captured cool states the error is tiny at the sampled wavelengths (at most
0.0035% at 5000 K and 0.000038% at 8000 K). That does not establish four sweeps
as generally sufficient, but it rules them out as the principal failure here.

## Controlled algorithm experiments

`scripts/check_cool_db_transport_seed.py` in the research workspace constructs
a new approximate seed by integrating hydrostatic balance and local
ML2-plus-diffusion transport in log pressure. It then calls the same production
non-gray solver, with its full physical flux verification. The initialization
has no effective-temperature switch. The script supports pressure/optical
meshes and an explicitly selected bulk-EOS comparison. These experiments are
not production defaults and have not replaced the validated initializer.

The 80-layer regular-pressure 8000 K experiment passes the discrete physical
flux equations (maximum residual 1.43e-6) and its lower boundary is now opaque.
However, its independently synthesized bolometric flux is 0.9543 of the
target. It is **not yet a validated converged spectrum**. An 80-layer seed
sampled in optical depth also converged (38 iterations, maximum physical
residual 0.001934, maximum dlnT=0.000210), with independent spectral flux
0.98067. A 10000 K control on that optical mesh converged in 18 iterations
(1.19e-7 residual), with spectral flux 0.99280. These are genuine new-seed
calculations; no paper checkpoint supplied their atmospheres.

Directly initializing a 160-layer optical mesh was difficult and was stopped.
Explicit 80-to-159-layer refinement of the newly solved 8000/10000 K models is
checked separately. The pressure-mesh 8000 K refinement converged in 29
iterations (1319 seconds), with maximum physical flux error 1.3325e-5 and
maximum dlnT=7.005e-5. Its spectral integral improved from 0.95429 to 0.98769,
but still fails the independent 0.003 spectral-flux requirement. The optical
10000 K refinement converged in five iterations (203 seconds), with maximum
physical flux error 0.0002485 and spectral integral 0.998513; both pass.
The 8000 K optical-mesh refinement also converged (11 iterations, 1137 seconds;
physical flux error 0.00264055, dlnT=2.44e-6), with spectral integral 0.995328.
It improves substantially on its 80-layer value but still misses the 0.003
independent spectral-flux requirement.
A further 159-to-317-layer optical refinement started at physical residual
0.09032 and rejected its first two directions (14 trial states). It was
stopped after about nine minutes without an accepted step; no 317-layer
convergence or spectral integral is claimed. More layers alone are therefore
not yet a robust automatic prescription.
Such refinement is not a new cold-start claim. These resolution changes are
evidence against treating a small discrete flux residual as sufficient
spectral validation.
Bypassing the convection conditioner was inferior in a controlled comparison
using the same freshly generated analytic seed.

At 5000 K the regular mesh removes the catastrophic compression but the ideal
bulk EOS still reaches hundreds of g/cm^3. The existing optional He-REOS.3
bulk density alone lowers that substantially without providing a satisfactory
solution. It still uses the HM chemical equilibrium; this is not a complete
dense-fluid ionization model. No temperature threshold, opacity multiplier,
electron floor, trace hydrogen, or fitted convective efficiency was introduced
to force a solution.

### Optional convective trial correction

The shared solver now has an opt-in local trial-gradient correction. For a
trial interface with calculated ML2 flux above the stellar flux, it inverts
the same ML2 relation using frozen trial material coefficients to propose a
smaller gradient. It retains the temperature normalization, reconstructs the
temperature profile consistently, and evaluates **all** physical fluxes again.
Trust-region, merit, and convergence checks still apply to that actual state.
The returned convective flux is never overwritten. No Teff threshold or tuned
efficiency parameter selects the correction.

This is a bounded experiment motivated by TLUSTY's `ITMCOR` treatment of
overcarrying convective trials, not a reproduction of its complete correction
algorithm or `CONREF`. The [TLUSTY operational manual, section 12.6](https://arxiv.org/html/1706.01937v1#S12.SS6)
itself describes `ITMCOR` as an older optional method, disabled by default.
It remains **off by default**. An inward net radiative flux can in principle
coexist with convective flux greater than the stellar flux; the proposed cap
must not be confused with a universally valid final constraint.

A directional audit of a stalled 8000 K, 159-layer state found that the
regularized tangent was accurate for very small steps, while a full proposed
step created an all-depth residual near 4937 from an initial 0.202. Simply
removing the regularization did not fix the finite-step behavior. This points
to convective-branch crossing and conditioning, rather than justification
for adjusting convergence tolerances or silently changing final equations.

The corrected 8000 K pressure-mesh refinement reached physical flux error
0.18535 after 560 seconds, versus 0.16941 after 584 seconds without correction.
The corrected experiment was stopped after about eleven minutes with its
last state saved; the uncorrected control subsequently converged. This does
not justify promoting the optional correction to a production default.

## Physics that remains important

Already active are He bound/free continua, He-minus free-free absorption,
He+He+ dimer opacity, He–He–He CIA, Thomson scattering, density-corrected
Rayleigh scattering, and ML2 convection. Pure He should not acquire H2 CIA
unless hydrogen is actually part of the requested composition.

For reliable dense, cool He atmospheres the remaining major omissions are a
consistent fluid EOS/ionization calculation (including molecular ions in
charge balance), dense-fluid corrections to He-minus free-free absorption,
and refractive transfer. The available He-REOS.3 option is not a substitute
for the chemical part. Modern cool-WD work treats these effects explicitly:
[Blouin et al. (2018), sections II–III](https://arxiv.org/html/1807.06616v1),
[Kowalski et al. (2007)](https://doi.org/10.1103/PhysRevB.76.075112).
The existing helium line tables also have temperature-domain limitations;
they cannot be claimed validated cool-He line profiles by extrapolation.
The automatic He-minus free-free prescription also switches source formulae
at their tabulated temperature/wavelength boundaries. Around 10080 K, sampled
infrared discontinuities are a few percent. They remain to be reconciled with
the source data; they were not tuned away during this investigation and have
not been demonstrated to be the main cause of the 5000 K failure.

## Regression evidence

- Ordinary suite after both CIA changes, boundary/trial diagnostics, and the
  optional trial projector: 323 passed, 5 optional-data skips, 7 separately run
  canaries deselected (79 seconds). Convergence tolerances were not changed.
- Protected DB 10000 K cold start: passed, maximum flux error 2.97e-8.
- Protected DB 22000 K cold starts: both 80 and 40 layers passed, approximately
  5.2e-7 maximum flux error.
- All four protected DA cold starts passed before and again after restoring
  the H2 CIA tail. With the tail, 3000 K took 64 iterations and reached
  0.0009844 maximum physical flux error (unchanged required tolerance 0.002);
  4000 K took 36 iterations and reached 1.135e-5. These are finite-grid
  atmosphere convergence tests, not claims of observational validation.
- Fixed paper DZ spectra: PG 1225 maximum relative change 1.84e-11; J0738
  4.25e-14. Current-equation all-depth flux errors on those saved grids are
  0.001679 and 0.002233, both below the 0.003 requirement.
- DAB 9000 K: unchanged paper UV spectrum to roundoff; the saved state passes
  current equations with maximum flux error 8.87e-5.
- DAB 20000 K: three physical iterations, maximum flux error 6.14e-7; the
  re-relaxed comparison reproduces the preceding solver-consistency tranche
  (UV integrated change from paper +0.00612%).

These distinguish fixed-paper reproduction, re-relaxation of a paper state,
and genuine cold starts. No GitHub push has been made for this work.

## Reproducibility and limits

Research commands below run from the containing `spectral_model` workspace
and import this authoritative checkout. Choose unused output directories;
the script refuses to overwrite an existing experiment's log.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u scripts/check_cool_db_transport_seed.py 8000 --mesh optical --output-root results/my-db-optical-check
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u scripts/check_cool_db_transport_seed.py 8000 --refine-atmosphere results/my-db-optical-check/8000/atmosphere.npz --output-root results/my-db-refinement-check
```

The second command is explicitly a refinement, never a fresh-start test.
`--max-iterations` is the per-segment nonlinear iteration budget; the shared
driver can additionally perform its existing bounded continuation segments.
The first experimental refinement files inadvertently inherited the generic
`diagnostic_seed` text from the fresh-seed branch. Their `refined_from` summary
field and invocation identify the actual source; none was a cold start. The
script now stores distinct, checked fresh-seed/refinement provenance.

Live `iterations.jsonl` records accepted states and `solver.log` reports
Jacobian builds and rejected directions. `summary.json` separately records
the physical atmosphere residual, independent spectral integral, and boundary
screen. Stopped runs keep their last accepted state without being relabeled
converged. The 5000 K problem is still unsolved, and neither experimental
initialization nor the trial correction is a production default.
