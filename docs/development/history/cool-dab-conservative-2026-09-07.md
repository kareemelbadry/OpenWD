# Molecular DAB conservative-transport investigation

> Research record: this page describes work at the time it was written.
> For current usage and status, see the [user guide](../../getting-started.md)
> and [tested points](../../tested-temperature-ranges.md).

Partial implementation report. The requested 5000–10000 K reliability claim
has **not** been established. All experiment workers were stopped or finished
at handoff. No default stellar physics was changed and no push
was made. Research artifacts live in
`results/molecular-dab-conservative-20260907` in the outer workspace.

## Scope and physical choices

The initial experiments use log g = 8, log10(N_H/N_He) = -2, the existing
molecular H/He chemistry (H, H+, H2, H2+, H-, H3+, He I/II/III and electrons),
H2-He/H2-H2 CIA and neutral Lyalpha wings. The Roueff/RACPPK H2 state sum
uses the same partition weights for equilibrium and energy. This remains
an explicit research scope, not a changed DA or atomic-DAB default.

The pure-He REOS/HNC trace-ion closure is **not** substituted into this
mixture. Missing nonideal molecular dissociation, dense-mixture bulk
thermodynamics and pressure-distorted CIA remain physical limitations,
particularly near 5000 K. The first goal is a numerically verified solution
of the declared molecular equations; that is not full physical validation.

[Blouin et al. 2018](https://arxiv.org/html/1807.06616v1) treats mixture bulk
EOS, chemical potentials and density-dependent opacity separately.
[Hubeny 2017](https://academic.oup.com/mnras/article/469/1/841/3092374)
motivates material/gradient-consistent convective initialization and a final
coupled correction. No fitted opacity multiplier or temperature cutoff has
been introduced here.

## Changes under test

- The molecular research runner can now construct a fresh hydrostatic
  ML2/diffusion seed using its mixed chemistry, opacity and thermodynamics.
  The old pure-He default path is unchanged. Approximate seeds never inherit
  convergence. Mixed seeds cannot substitute a pure-He bulk EOS.
- Explicit per-call column-mass transfer, extended thermal wavelength
  integration and independent source closure are reused from the DB work.
- A 5000 K diagnostic exposed a pseudo-time boundary problem: when bottom
  convection is initially absent, the algebraic flux boundary demands a
  finite temperature increment even as the time step tends to zero. The
  trial switches on enormous ML2 flux. An optional bottom thermal reservoir
  makes that increment vanish with the time step; its stationary equation
  is still F_bottom = Fstar. Time steps stop at floating-point temperature
  resolution, not a 24-halving retry limit. None of this declares convergence.
- A nonlinear auxiliary-ML2 pseudo-time variant was tested separately. Its
  material tangent remains approximate; actual EOS/opacity/ML2 flux are
  always reevaluated before accepting a step.
- The SVD proposal previously put a box on rotated coordinates, excluding
  physically admissible global temperature corrections by approximately
  sqrt(N). Its bounds now apply to nodal log temperatures for all SVD
  proposals, not just the tabulated-material variant. A synthetic regression
  explicitly tests a uniform thermal mode against stiff difference modes.
- `synthesize_helium_spectrum` now accepts an explicit `column-mass`
  discretization and solves its scattering source directly, checking it with
  a separate prescribed-source solve. Its default optical-depth/four-sweep
  path is unchanged. Mixed synthesis can forward the option. The new option
  has isolation, invalid-selection and failed-source-closure tests.
- The research synthesis now explicitly matches the structure's column-mass
  transfer, angular order, Stark Lyman selection, and existing He II line
  selection. Neither the spectrum nor the structure is renormalized.
- A low-density Stark-table coordinate defect was diagnosed. The bundled
  tables already include Doppler convolution, but their old extrapolation
  clips the tabulated profile while continuing to shrink its wavelength
  scale with electron density. At 4000 K the sampled Lyalpha FWHM decreases
  from 0.0511 A at the lowest table density to 0.00000511 A at ne=1e4,
  although the thermal Doppler FWHM is about 0.0549 A. The explicit
  `--consistent-stark-edge` experiment holds the whole lowest-density
  profile fixed; it does not floor the EOS electron density.
- The explicit `--physical-detuning-lyman` experiment interpolates the
  Doppler-convolved Lyman tables at fixed wavelength offset instead of
  field-scaled offset. Original tabulated density/temperature nodes and
  power-law wings are preserved; profiles are not renormalized. A stalled
  5000 K layer's off-center opacity derivative changed from about -94 to
  -0.92 and became stable under probe refinement. This is an alternative
  interpolation prescription, not a new Stark calculation or production
  default. The [table distributor's documentation](https://warwick.ac.uk/fac/sci/physics/research/astro/people/tremblay/modelgrids/aareadme.txt)
  confirms that these files include Doppler convolution.
- A measured-secant correction to the nonlinear inner proposal remains an
  explicit `--measured-proposals` experiment, OFF by default. Applying raw
  secants can double-count the ML2 curvature already present in that model;
  combining it with the interpolation experiment did not improve the tests.
- Strict direct-energy and full-coupled research runs now require a measured
  small proposal before outer trust clipping/backtracking, including after
  an accepted step. Merely accepting a small trust-limited step cannot certify
  stationarity. A synthetic regression reproduces the generic driver's
  premature accepted-small-step exit and verifies that the research guard
  rejects it. The generic production stopping policy is unchanged. Successful
  cases are checked for a measured small proposal in a non-collapsed region,
  not just a small accepted update.
- `--require-convergence` makes the research runner exit nonzero when its
  recorded physical-flux/local-energy/correction/source-closure/bottom-screen
  evidence fails. Exploratory output is still retained. The standalone
  `scripts/check_molecular_dab_result.py` checks the same evidence without
  rerunning an atmosphere; it explicitly does not certify mesh convergence
  or full dense-mixture physics. Fifteen focused checker tests pass. The bounded
  5000 K failure is rejected despite its nearly correct surface integral.

An implicit molecular-thermodynamics prototype was **withdrawn from the
active runner**, and its supporting chemical refactor was restored exactly
to the pre-prototype source. It did not demonstrate a useful accuracy gain.
Diagnostics exposed the pre-existing H3+ partition clamp at 10000 K: the
finite-difference internal energy creates a sharp heat-capacity feature there.
Direct differentiation cannot repair a nonsmooth physical prescription.
The prototype and failing diagnostics are preserved under the result directory,
not enabled or treated as passed tests. The H2 bound-state sum does not
remove this separate H3+ limitation.

A subsequent cancellation-resistant H3+ energy experiment evaluates the
**identical** polynomial/clamps/logarithmic stencil through polynomial divided
differences. Twelve tests, including 60-digit reference calculations at the
clamps, pass. In the stalled 5000 K structure its maximum change in adiabatic
gradient is only 2.9e-13, and the thermal-tangent noise is not materially
improved. It is not enabled in atmosphere runs and is not a demonstrated
convergence fix.

## Completed measurements so far

All errors below are fractions, not percentages. Strict tolerances are
unchanged. Runs marked as restarts are not cold-start demonstrations.

| Case | Initialization / static attempts | All-depth flux error | Local energy error | Measured max dlnT |
| --- | --- | ---: | ---: | ---: |
| Molecular 8000 K, 162-layer restart | zero pseudo sweeps / 4 | 7.29e-10 | 1.53e-6 | 1.86e-4 |
| Molecular 10000 K, fresh 80-layer seed | 6 pseudo sweeps / 18 | 3.30e-10 | 9.61e-7 | 3.09e-5 |
| Molecular 7750 K, 162-layer continuation, physical-detuning Lyman | 8000 K scaled prediction / 20 | 5.67e-10 | 1.81e-6 | 1.59e-4 |
| Molecular 10000 K, 80-layer re-relaxation, physical-detuning Lyman | original-profile molecular root / 3 | 6.07e-10 | 1.23e-7 | 3.02e-6 |
| Molecular 8000 K, fresh 80-layer seed, physical-detuning Lyman | 40 pseudo sweeps / 18 | 1.91e-10 | 1.75e-6 | 1.79e-4 |
| Molecular 7500 K, 166-layer extended continuation, physical-detuning Lyman | 27 attempts on shorter domain, then extension / 1 | 7.62e-8 | 3.37e-8 | 1.19e-4 |
| Molecular 8000 K, 159-layer refinement, physical-detuning Lyman | fresh 80-layer root / 3 | 1.20e-5 | 3.33e-6 | 1.42e-5 |
| Molecular 8000 K, 317-layer refinement, physical-detuning Lyman | 159-layer root / 2 | 0.002405 | 7.04e-5 | 1.70e-4 |
| Molecular 9000 K, 80-layer continuation, physical-detuning Lyman | 10000 K scaled prediction / 25 + 8 | 3.01e-9 | 2.23e-7 | 1.58e-5 |

The first two successes use the original field-scaled profiles; the remaining
rows use the explicit interpolation/edge experiments. The new-profile 10000 K
re-relaxation is not a fresh cold start. None changes the production defaults.

Independent fixed-state audits use 6000 continuum nodes over the original
interval, extended at the same spacing to 1e9 A (13999 continuum nodes),
plus line nodes, and 16 angles. Both pass:

- 8000 K: maximum flux error 0.000152715, local energy 0.0000660836,
  surface integral/Fstar 1.00005433.
- 10000 K: maximum flux error 0.000463395, local energy 0.0000924588,
  surface integral/Fstar 1.00021918.

Further audits retained those 8000/10000 K atmospheres and subdivided each
original Lyman core interval tenfold. Both passed again: maximum local
errors were 1.10e-5 and 1.33e-4, respectively. The full fine-grid 7750 K audit
(6000 original-interval continuum nodes, extended to 1e9 A; tenfold Lyman
core refinement; 16 angles) gives flux error 1.09e-4, local error 6.78e-5,
surface integral/Fstar 1.00003108, and bottom absorption escape 0.00118.
An independent 16-angle synthesis of that 7750 K structure gives
0.99968010 Fstar, with source-closure error 3.96e-16.
The same fine continuum/line/angular audit on the new-profile 10000 K
re-relaxation passes: all-depth flux error 0.000463395, local energy error
0.0000866338, surface integral/Fstar 1.00021918, and bottom escape 3.65e-43.
The new-profile **cold-start** 8000 K audit passes at flux error 0.000153496,
local energy error 0.0000638521, surface integral/Fstar 1.00005941 and bottom
escape 2.99e-32. The deeper 7500 K audit passes at flux error 0.0000765399,
local error 0.0000756070, surface integral/Fstar 1.00001568 and bottom escape
0.000163048. These all use the same refined quadrature and profile policy.

The fresh new-profile 8000 K run took 2790.45 seconds, including its fresh
ODE seed, 40 linearly implicit initializer sweeps, and 18 static attempts.
This is a successful cold start, not a demonstrated speedup. Concurrent
worker timings are not controlled performance benchmarks.

Refining that 8000 K atmosphere from 80 to 159 layers converged in three
attempts (287.44 seconds). On the identical 4000-point synthesis grid, the
absolute spectral difference integrated over wavelength is 0.006885 times
the finer model's bolometric flux. Maximum pointwise differences are 2.19%
over 0.3–1 micron and 2.32% over 1–3 microns; the UV/line regions can differ
more. Thus 80 layers are **not** a sub-percent spectrum certificate even
though their energy equations converge. The further 317-layer test converged
in two attempts (484.21 seconds). The integrated absolute spectral difference
from 159 layers is 0.0017233 times bolometric flux; pointwise maxima shrink to
0.559% over 0.3–1 micron and 0.572% over 1–3 microns. The approximately
fourfold reduction supports second-order depth convergence at this model,
not a certificate across the temperature/abundance range. Its stricter
independent quadrature audit passes at flux error 0.00240442, local error
0.0000753304 and surface integral/Fstar 1.00007199. Note that its last deep
flux residual is close to the existing 0.003 tolerance, not 1e-9. No spectrum
was renormalized in these comparisons.

The new measured-proposal guard was also exercised on a rebuilt 7750 K
checkpoint: it passed with a freshly computed max dlnT=1.586e-4 in a
0.04 trust region, all-depth flux error 5.85e-6, and local error 1.82e-6.
The rebuilt-state flux audit is recorded separately from the original
iteration's 5.67e-10 value; the two must not be conflated.

These are not depth-resolution certificates. Matched independent Stark /
column-mass / 16-angle synthesis gives 0.99969749 Fstar at 8000 K and
1.00018172 Fstar at 10000 K. An explicit Allard-on-the-same-10000-K-structure
ablation gives 0.99856359 Fstar. The older ordinary research synthesis's
0.995705 Fstar mixed transfer and line prescriptions; it was not normalized
away or attributed to scattering alone. The 10000 K converged structure is
not a certification of a structure relaxed with Allard profiles.

The 5000 K pseudo-time variants, 6000 K cold start, and 7000 K continuation
did not converge. SVD nodal bounds, fixed/current local-energy weights,
finite-material convection repairs, a fully coupled auxiliary-ML2 system,
small material probes, and fresh discrete diffusion/ML2 seeds were tested
separately. Some reduced global errors while local errors remained large;
none is a successful 5000 K model. The completed physical-detuning-only
5000 K discrete-seed restart still has 9.92% all-depth flux error and 16.6%
local energy error after 50 attempts. Its nearly correct surface integral
does not make it converged. Interrupted runs and diagnostic snapshots are
preserved, not overwritten or relabeled.

The 7500 K continuation reached a stationary solution after 27 attempts:
all-depth flux error 3.85e-10, local energy error 2.56e-6, and measured
max dlnT 1.25e-4. However, its bottom escape rose to 0.00616 after relaxation,
so it **fails the domain screen**. A second extension to 166 layers, using
the identical molecular transport materials, passed after one correction;
its final bottom escape is 0.000290727 (fine-grid audit 0.000163048).
The seed's boundary screen is not accepted as evidence about the final
atmosphere. Continuation from this verified parent to 7250 K completed 35
attempts without convergence: flux/local errors 0.000215220/0.00139425,
last accepted max dlnT 0.00162823, and bottom escape 0.0197778. The small
flux errors do not satisfy the separate stationarity and domain checks.
A further two-node domain extension completed 25 additional attempts:
flux/local errors 0.000159953/0.00105181, last accepted max dlnT 0.00122568,
and bottom escape 0.000283898. The domain problem is repaired, but
stationarity still fails. A bounded compatible-coordinate restart on this
explicitly unconverged atmosphere was stopped after three accepted updates
(866 seconds): flux/local errors 0.0000967365/0.000877186 and last accepted
max dlnT 0.000625003. The raw root proposal remained outside the trust region;
this is not a correction certificate. This run used the earlier composite
merit, before the separately documented objective-consistency change below.
The additional 5000 K convection-initialization attempt completed without
convergence (14.4% all-depth flux error, 45.1% local energy error; trust
region collapsed). The bounded, fully coupled temperature/ML2 experiment
also failed: 9.41% all-depth flux error and 25.4% local energy error after
20 attempts, despite surface integral/Fstar 1.00106. It retains actual-gradient
flux/local-energy gates; auxiliary flux is not a substitute for convergence.

A directional audit of the stalled new-profile 5000 K Jacobian finds a
condition number about 6.6e9. Uniform-temperature and worst-local-row probes
agree with the tangent at approximately 1.6e-5 and 7.2e-6 relative error
(probe 1e-5). Its weakest singular direction is far less reliable: relative
directional discrepancy is about 9.7 at probe 1e-5 and 466 at probe 1e-7.
This implicates cancellation/conditioning in weak global modes, but does not
prove which material derivative is responsible or that missing dense physics
causes the numerical stall. Decreasing the finite-difference step alone is
not a demonstrated repair.

The new-profile molecular 9000 K continuation from 10000 K used 25 attempts
and remained unconverged: flux error 0.00377636, local error 0.00560767.
It was still improving after a long plateau, so an explicitly labeled
additional 15-attempt continuation was allowed, with unchanged physics and
tolerances. It converged after eight additional attempts (322.44 seconds):
flux error 3.01e-9, local error 2.23e-7, measured max dlnT 1.58e-5. The fine
continuum/line/angular audit also passes: flux error 0.000312799, local error
0.0000897303, surface integral/Fstar 1.00013812, and bottom escape 7.68e-6.
This is distinct from the older atomic paper-DAB check below.

## Further conditioning experiments (not production selections)

A direct physical-box Gauss-Newton/BVLS inner optimizer was compared with
the SVD/tanh optimizer at 5000 K using the **identical** saved starting
atmosphere and ten-attempt budgets. Neither converged. SVD ended at flux/local
errors 0.0952077/0.160990 in 303.85 seconds; BVLS ended at
0.0989803/0.168690 in 267.00 seconds. The small elapsed-time difference is
not a convergence speedup. Both fail the strict output checker. The explicit
research `--inner-scaling bvls` selection is not promoted to a default.
Tiny proposals from an unsolved inner model cannot certify stationarity.

An additional **exact-material compatible-coordinate** prototype is under
test. Unlike the existing cheap inverse-ML2 proposal, it reconstructs
temperature/material/gradient compatibility at every actual atmosphere trial
and applies its implicit tangent to the radiative response. The latest
Newton residual uses the independently ML2-constrained coordinate flux to
avoid subtracting nearly adiabatic gradients; all conservation rows remain.
Every trial must reproduce that flux from its actual temperature gradient
to within 1e-7 Fstar. Final flux/local-energy gates and all output fluxes
use the actual gradient, not the auxiliary coordinate. This is not an
overwrite with Fstar minus radiative flux. Synthetic reconstruction and
physical-energy chain-rule checks pass. An initial internal 1e-11
log-temperature reconstruction tolerance was below the nested material
derivative noise floor; its failed checks and interrupted 7750 K run are
preserved. The revised prototype uses the existing finite-material repair's
1e-9 log-temperature accuracy plus a 1e-7 actual flux-compatibility check
and an additional polishing step. These are internal coordinate tolerances;
the atmosphere's convergence gates are unchanged. This prototype is not yet
a demonstrated 5000 K atmosphere-convergence fix.

An explicit tighter molecular conservation tolerance was also added. The
default remains 2e-11 and is tested to produce bitwise-identical populations
to an explicit default argument; only tightening is allowed. Six added tests
cover conservation and invalid selections (16 chemistry tests pass in total).
No reaction, population prescription, opacity or thermal equation changes.
This investigates chemistry error propagated through successive numerical
enthalpy and material derivatives. At 5000 K, tightening to 2e-13 reduces
the compatible-coordinate weakest-direction discrepancy at probe 1e-5 from
91.3 to 3.18, but that direction is still not reliably differentiated at
small probes. This is not sufficient evidence of a convergence repair.
The same tightened-chemistry, polished-coordinate formulation preserves the
7750 K root with a newly measured max dlnT=0.000159205, actual flux error
5.85e-6 and local error 1.82e-6; the strict recorded-evidence checker passes.
Larger weak-direction probes at 5000 K, producing actual log-temperature
changes 2.1e-5 and 2.1e-6, have relative discrepancies 0.0815 and 0.0957.
These finite-probe results are better than the smallest noisy probes, but
are not proof of an accurate Newton direction on every mode.

The unrestricted coordinate Newton direction initially left the chemical
domain before its physical step size was checked. The prototype now bounds
the predicted physical temperature size before nonlinear reconstruction.
When the unconstrained direction exceeds the trust region, it solves a
linear least-squares subproblem with independent nodal temperature bounds,
then checks the actual nonlinear temperature change. Synthetic tests confirm
that a weak mode does not throttle resolved modes, and that an admissible
Newton root step is unchanged. A 7750 K real-model recheck passes unchanged
under this bounded method. At 5000 K both the globally scaled and nodally
bounded variants rejected complete trial sequences without improvement;
they were stopped, with input/provenance files retained. A one-attempt
diagnostic records trial merits and actual ML2-compatibility errors to
distinguish physical reconstruction rejection from an uphill direction.

That one-attempt diagnostic completed: all seven rejections were **merit**
rejections, with zero physical-domain or acceptance-guard rejections. At
backtracking factor 0.25, mean squared residual fell from 0.00142958589 to
0.00142767720, while maximum residual rose from 0.11822510 to 0.11912415.
The generic objective, RMS plus 0.25 times maximum, therefore rejected a
least-squares descent direction. This is a mismatch between the bounded
Gauss-Newton subproblem and its globalization objective, not evidence that
the convection-reconstruction check should be relaxed.

The compatible-coordinate prototype now uses its matching smooth
least-squares objective for step acceptance. A separate explicit
`--least-squares-merit` selection permits a controlled comparison with the
existing SVD/nonlinear-ML2 proposal. Both retain the same maximum physical
flux/local-energy and measured-correction convergence gates; the generic
production objective is unchanged. A synthetic test reproduces the
objective disagreement and verifies restoration of the production function
after leaving the research scope. The formerly rejected 5000 K step is now
accepted, but this has not yet established atmospheric convergence.
The completed SVD-proposal comparison with the matching merit used 15
attempts (304.57 seconds) and failed at flux/local errors
0.0994810/0.168048. The coordinate variant was stopped after three accepted
updates (474 seconds), with flux/local errors 0.101363/0.168555. Thus neither
demonstrates a useful 5000 K convergence improvement. Their small reductions
in squared residual are not substitutes for the physical gates. Interrupted
states remain explicitly unconverged diagnostic checkpoints.

The final matching-merit, bounded-coordinate, tightened-chemistry method
also preserves the 10000 K root: freshly measured max dlnT 3.016e-6, actual
flux/local errors 6.07e-10/1.23e-7, direct structure-grid integral
1.0000000000000004 Fstar, and independent synthesis integral 1.0001342621
Fstar. The strict checker passes. This recheck is not a new cold start.

New runs also preserve a compressed archive of the exact Python source
bytes alongside their SHA-256 manifest. These are provenance artifacts, not
another active production source tree; binary dependencies and input data
are not included. Earlier runs contain source hashes but no such archive.

## Existing-model protection

The latest complete regression run passed **583 tests including all seven
protected cold-start canaries**: DA 3000/4000/5000/20000 K and DB
10000/22000 K (standard and production resolution). It took 1245.54 seconds
with atmosphere workers running concurrently, and includes the tighter
chemistry option with its unchanged default. The release pytest configuration
was loaded explicitly and all canaries were included. Subsequent research-only
coordinate trust bounds, objective consistency, archive and guard/checker tests
pass in a separate 34-test selection. The final handoff selection, including
the molecular chemistry, mass synthesis and generic nonlinear tests, passes
all 80 tests (`handoff-focused.xml`). These timings are not controlled speed
benchmarks. See `precision-regressions.xml`; the earlier 532-test run is
retained as `regular-final.xml` in the result directory.

Paper checks are deliberately separated by what they establish:

- DAB 20000 K: fixed paper spectrum agrees to 8.66e-12 maximum relative
  difference, and a newly measured relaxation passed in three iterations.
- DAB 9000 K: fixed paper spectrum remains unchanged (about 3e-9 maximum
  relative difference, UV integral unchanged). However, the stronger test
  requiring a newly measured small correction stalled despite 4.59e-5 flux
  error. It **did not pass** that stronger convergence test. The old check
  accepted the initial flux without measuring a new correction; this does
  not demonstrate a regression caused by this turn's opt-in changes.
- DZ PG1225 and J0738: fixed spectra agree at 1.83e-11 and 4.24e-14 maximum
  relative difference. Their current all-depth flux audits pass at 0.00168
  and 0.00223. These are not new relaxation/correction certificates.

## Required before a reliability claim

Obtain cold-start and continuation agreement across the requested interval,
independent wavelength/angular checks, depth/domain checks, additional H/He
ratios, and regression protection for established DA/DB/DAB/DZ models.

The remaining physics work is a genuine mixed-fluid closure, not replacing
the mixture's thermodynamics with pure helium or suppressing molecules to
make Newton converge. For comparison, Section III of
[Blouin et al. 2018](https://arxiv.org/html/1807.06616v1#S3) combines separate
hydrogen/helium bulk EOSs by an additive-volume rule and treats nonideal
ionization separately. Bulk density/energy, charge equilibrium and the
opacity populations must therefore be validated together. This does not
prove that missing dense physics caused the present numerical stall;
the demonstrated weak thermal modes still need an accurate coupled tangent.
Production API integration should use explicit physics/solver objects rather
than the process-global research scopes. Dense-physics caveats must remain
visible even for numerically converged outputs.

## Reproducing the explicit research configuration

Run from the outer workspace with the release source on `PYTHONPATH`, one
process per experiment (the research scopes must not be used concurrently
inside a process). For example, the successful fresh 8000 K calculation was:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KMP_USE_SHM=0 \
PYTHONPATH=release/OpenWD/src:scripts python \
  scripts/run_molecular_dab_mass_experiment.py \
  --physical-detuning-lyman --consistent-stark-edge --require-convergence \
  --pseudo-sweeps 40 --relax-bottom \
  8000 --log-h-he -2 --physical-only --stable-transfer \
  --step-method nonlinear-convection-current-energy --inner-scaling svd \
  --inner-max-evaluations 1000 --max-iterations 35 --no-continuations \
  --state-sum-h2 --output-root results/my-new-dab-experiment
```

This is a research command with installed SciPy and local opacity data, not
a new public production API. It refuses to overwrite a previous experiment.
The original cold run predates the CLI checker; its stored evidence was
subsequently checked with the same strict checker. For existing results:

```sh
PYTHONPATH=release/OpenWD/src:scripts python scripts/check_molecular_dab_result.py \
  results/molecular-dab-conservative-20260907/detuning-cold/8000
```

`solver.log`, `iterations.jsonl`, checkpoints, a source-SHA manifest, declared
physics, and separate structure-transfer/synthesis summaries are retained.
The independent quadrature audits also store their full argument lists.
Successful declared-equation convergence does not resolve the missing
dense-mixture physics or validate a different abundance or gravity.
