# Direct local-energy response and cool-model continuation

Numerical milestone after `nonlinear-convection-experiments-2026-09-05.md`.
Experimental energy/proposal methods remain research-only. Convergence
certification and the guarded final precision correction below are in the
shared solver's default path; there is no new
temperature switch, adopted surrogate convective flux, trace-H addition to
the DBs, spectrum renormalization, or GitHub push.

## Verified response correction

The older local-energy experiment evaluated absorption times `(J-B)` directly,
but subtracted interface-flux Jacobian rows to differentiate it. This loses
the local response in nearly transparent cells. The new optional
`compute_local_energy_response` path differentiates

`E_rad,i = 4*pi*integral(width_i*epsilon_i*(J_i-B_i))`

directly. It includes the coupled coherent-scattering response of J, Planck
response, absorption/scattering-fraction response and opacity motion of cell
widths. Wavelength chunks are consumed without retaining a full spectral
response tensor. The same pass differentiates thermal emission. Actual ML2
material and gradient derivatives supply the convective divergence response.
Only this opt-in path uses centered material differences; the existing default
flux tangent retains its prior arithmetic.

On the previously saved **fresh-start** 8000 K, 80-layer flux-only result, a
centered dlnT perturbation of 1e-4 gives these maximum absolute derivative
errors, normalized by local emission/convective transport:

| Perturbed node | Direct derivative error | Flux-difference derivative error |
| --- | ---: | ---: |
| Surface, node 0 | 9.59e-5 | 388.339 |
| Node 23, T=4611 K, tau_R=5.5e-7 | 1.92e-4 | 58.3143 |

These are derivative errors, not atmosphere flux errors. At node 23 the
measured derivative at the worst cell is 47.639885; direct differentiation
gives 47.640077, while the old difference gives 105.954187. Perturbation-size
checks and independent complete-callback tests cover pure absorption and
strong scattering. Outputs: `results/direct-local-energy-audit-20260905` in
the research workspace.

## Current local normalization and inner conditioning

The new research residual is `F_i/F_star - 1 - E_i/S_i(T)` for thermal cells,
with the last row retaining the bottom flux condition. S is current thermal
emission plus the actual nonnegative convective flux through both faces.
Its derivative includes `d(E/S)=dE/S-E*dS/S**2`. In exact arithmetic it has the
same steady root as conservative flux balance; independent local and global
checks remain required. It uses no convection mask or effective-temperature
threshold. The cheap proposal keeps tangent radiation and nonlinear ML2;
the outer trial recomputes actual EOS, opacity, scattering transfer and ML2.

Fixed initial energy scales become inappropriate after substantial thermal
evolution. However, current scaling alone was insufficient: the inner
least-squares optimizer repeatedly spent its 100-evaluation budget without
finding a useful direction. An audit on the identical stalled 8000 K state
found:

- Unit variable scaling, 100 evaluations: proposed maximum dlnT=2.08e-6;
  the full trial did not reduce the outer merit.
- Unit scaling needed 9766 evaluations (13.75 s) to find the useful direction.
- Jacobian-column scaling reached essentially the same model minimum in
  243 evaluations (0.31 s). Its full trial reduced merit from 0.06759 to
  0.05629. This does not mean every physical diagnostic decreases at once.

This is [SciPy's documented inverse-column-norm variable scaling](https://docs.scipy.org/doc/scipy-1.11.1/reference/generated/scipy.optimize.least_squares.html),
not a change in physical equations or bounds. A 1000-evaluation research budget
is now explicitly selectable and logged. It is not an atmosphere iteration
budget or a guarantee that the inner optimizer converged. Earlier fixed-scale
audits showed little benefit from scaling; the improvement was measured on
the current-normalization stalled state, not inferred universally.

## Completed and interrupted experiments

All DB cases are pure He, log g=8. Fresh runs explicitly reuse the identical
unrelaxed analytic hydrostatic ML2/diffusion seed, not a relaxed atmosphere.
Same-grid continuations and refinements are separately identified.

| Experiment | Attempts / seconds | Actual all-depth flux error | Relative local energy error | Public spectrum/F_star |
| --- | ---: | ---: | ---: | ---: |
| 5000 K, fixed direct-energy scale, 80 layers | 20 / 126 | 0.340450 | 0.653499 | 0.863014 |
| 5000 K, current energy scale, unit variables | 20 / 121 | 0.368764 | 0.332470 | 0.850933 |
| 5000 K, current scale, Jacobian-scaled variables | 30 / 209 | 0.348459 | 0.331053 | 0.853461 |
| 8000 K, fixed direct-energy scale | stopped after 3 accepted | 0.112264 | 0.266659 | not synthesized |
| 8000 K, current scale, unit variables | stopped after 9 accepted | 0.108181 | 0.131929 | not synthesized |
| 8000 K, current/Jacobian scaling, fresh 80 layers | 30 / 885 | 7.00e-5 | 0.001527 | 0.980718 |
| 8000 K, explicit 80-to-159 refinement | 12 / 1052 | 8.45e-6 | 0.024608 | 0.995329 |
| Same 159-layer run, explicit same-grid continuation | 9 / 536 | 2.30e-6 | 7.73e-5 | 0.995327 |
| Explicit 159-to-317 refinement | 12 / 1670 | 1.26e-5 | 0.001145 | 0.999159 |

The 159-layer sequence passes the actual flux, local energy and temperature
correction tests (last dlnT=2.64e-4, below 3e-4). It is **not yet an independently
validated spectrum**: its spectral integral misses the 0.003 requirement.
A 317-layer refinement subsequently **passed** all atmosphere gates and the
independent spectrum integral; its final measured dlnT is 4.64e-6 and the
absorption escape screen is 2.66e-54. This is a refinement sequence, not a
fresh-start speed claim. The 80-layer fresh run's last dlnT=0.00805
still fails temperature stationarity despite passing both energy thresholds;
its guarded 20-attempt continuation also fails (last dlnT=0.001465). The slow local corrections
mean this tranche is not an overall speedup relative to flux-only convergence.
It enforces an additional meaningful constraint that the faster result failed.

Interrupted runs retain `latest-iteration.npz`, telemetry and `stopped.json`;
they are not completed atmospheres or substituted spectra.

## Restart/phase-transition certification fix

The 80-layer diagnostic exposed an existing shortcut: a restart can accept a
residual-only evaluation with step zero, even if the preceding run stopped
with an oversized temperature correction. Its zero-iteration continuation is
retained with `validation-caveat.json` and is **not** counted as verification.

The same issue occurs between internal solver phases: the new 8000 K DAB
baseline ended its 20-step conditioning phase with dlnT=0.005265, then the
formal segment reported success at its unchanged initial state and recorded
zero dlnT. This is not a newly converged temperature correction.

The generic driver now accepts `allow_initial_convergence=False`, which
requires computing a correction. Defaults preserve the normal public explicit
restart behavior. Shared atmosphere phase transitions and automatic formal
continuations disable the shortcut if the preceding segment has no measured
step or an oversized step. Small measured steps may still use the inexpensive
initial flux check. The strict research continuation uses the new option too.
Tests cover a weak residual requiring a large temperature correction, a truly
stationary root, and propagation across the atmosphere phase boundary.

This correction changes convergence certification, not opacity, transport or
the target steady solution. The four protected DA and two production-resolution
DB cold starts pass with their original iteration allowances. Without the
precision correction described below, the standard 40-layer 22000 K DB
converges in 47 iterations versus its old 30-iteration test cap. That
intermediate performance failure was not hidden or reclassified as physical
failure. Its former 20-step result had dlnT=0.000989, above 0.0003,
before the zero-step shortcut. A matched same-run spectrum comparison finds
maximum UV (1000–3000 A) change 3.06e-6, optical (3500–7500 A) change 0.001351,
and integrated change -2.17e-7. The raw maximum relative change, 0.481, occurs
at 100 A where the old flux is only 1.9e-31 of its peak; it is not a 48% change
in an observable band. Output: `results/db22000-phase-certification-spectrum-20260905`.

A second stationarity loophole was removed: trust collapse previously
recorded the small trust radius as if it were a measured correction. A small
computed proposal can now certify a residual before clipping/backtracking;
collapsing the trust region alone cannot. Bounded proposal hooks cannot
certify their size from a sub-tolerance trust radius. Tests cover both an
exact stationary root and an inaccurate tangent whose large proposed steps
never reduce an already small residual. The latter must remain a failed solve.

## Thermal-inertia proposal: tested but not promoted

An isolated audit added frozen-Cp thermal inertia to the proposal model,
`q_i=Cp_i*T_i*Delta_m_i*(exp(dlnT_i)-1)/dt`, and transformed `E-q` through
the same energy rows. The bottom flux remains an algebraic boundary. This is
inspired by [pseudo-transient continuation](https://petsc.gitlab.io/petsc/main/manualpages/TS/TSPSEUDO/),
not a new equilibrium constraint; the actual outer residual never includes q.
Its analytic derivative passes finite differences.

On the fresh 5000 K seed, thermal times span 0.0046 to 1.57e10 s. The explicit
0.04 logarithmic-temperature timescale is 1835 s. Tests at that timestep,
100 times it, and 10000 times it did not beat the unregularized nonlinear
proposal. No full atmosphere run or default selection was made on that basis.
This rejects these trial constructions, not all possible pseudo-time methods.

## Cooler DABs and missing chemistry

New production-physics cold starts at log(H/He)=-2, log g=8 use 80 layers and
a 40-iteration per-phase diagnostic limit, with no automatic formal
continuations, supplied atmosphere or spectrum rescaling.

- **8000 K atomic DAB, old phase shortcut:** 20 conditioning steps / 329 s, flux error 8.53e-6,
  local error 0.001137, spectrum/F_star=0.995356. Its reported zero final
  correction is the phase shortcut described above. The corrected cold start
  genuinely converges in 25 steps / 372 s, flux 1.08e-7, local 0.001113,
  dlnT=0.000191, spectrum/F_star=0.995353.
- **5000 K atomic DAB:** 29 total steps / 261 s, actual flux error 8.86e-4 and
  dlnT=1.73e-4: it genuinely passes the existing production flux/step gates.
  But local energy error is 0.05147 and spectrum/F_star=0.990283, so it fails
  the independent validation checks.

Following these explicitly identified cold starts, the new local-energy
method, mesh refinement and absorption-screened domain extension give:

| Atomic DAB experiment, log(H/He)=-2 | Added attempts / seconds | Flux error | Local energy error | Spectrum/F_star |
| --- | ---: | ---: | ---: | ---: |
| 5000 K, same-grid local-energy correction | 5 / 52 | 2.62e-8 | 1.75e-8 | 0.990274 |
| 5000 K, 80-to-159 refinement | 5 / 125 | 7.32e-8 | 8.06e-8 | 0.996823 |
| 5000 K, deeper 161-layer domain | 2 / 73 | 1.26e-7 | 1.01e-7 | 0.998644 |
| 5000 K, further 162-layer domain, strict correction | 1 / 50* | 5.13e-8 | 2.89e-8 | 0.998650 |
| 8000 K, 80-to-159 refinement | 7 / 313 | 2.65e-9 | 7.77e-9 | 0.998544 |
| 8000 K, deeper 162-layer domain | 1 / 91 | 5.76e-7 | 1.30e-8 | 0.999190 |

`*` Plus 24 seconds for the explicit additional-domain seed/residual check;
that initial zero-correction check was not counted as temperature convergence.
Both deeper DABs pass the separate flux, local heating, measured correction,
and spectrum-integral requirements. The 5000 K 161-to-162 domain change is
6.07e-6 in units of stellar flux. Absorption escape screens are 2.95e-5 at
5000 K and 2.92e-4 at 8000 K. These are **atomic-model numerical successes**,
not validation of omitted molecular physics.

The extension keeps every original mass node and integrates only appended
hydrostatic ML2/diffusion seed layers until their continuum absorption screen
passes. The full atmosphere then relaxes with actual physics. No stellar
temperature threshold or opacity floor chooses the domain. An initial attempt
propagated the gray seed's accidentally compressed bottom spacing (dlnP about
0.00056), making the seed unnecessarily costly. The research helper now uses
the existing mesh's median dlnP (about 0.10), with independent adaptive ODE
error control. Both stopped attempts retain `stopped.json`; they are not
solutions. The appended seed never certifies its own final convergence.

The production DAB model explicitly excludes H2 chemistry and H2-He CIA.
The trace mass-action estimate `2*n(H I)**2/(K_H2*n_H,nuclei)` at the atomic
photosphere is 0.00460 at 8000 K, but **8.69** at 5000 K (maximum 214).
Values above one are not physical molecular fractions: they demonstrate that
the atomic state is not a self-consistent approximation for molecular
chemistry. A 5000 K convergence flag alone is insufficient.

For the DB seeds, the existing Stancil molecular-ion opacity implies
`n(He2+)/n(He+) = n(He I)/K`, while the EOS omits that ion from charge balance.
At the analytic photosphere this ratio is 3.90 at 8000 K and 2.99e4 at 5000 K.
The latter seed has density 203 g/cm3 at the photosphere and 267 g/cm3 at its
bottom. These are inconsistency diagnostics, not corrected ionization
fractions or credible dense-He structures. No electron floor was inserted.

A consistent molecular extension must conserve nuclei, charge and pressure,
include reaction/excitation energy in Cp and nabla_ad, and feed the same
populations to structure and synthesis opacities. Recoverable older molecular
source can be reviewed for individual components, but the quarantined solver
must not be reinstated. Dense-He chemistry remains distinct from a bulk-EOS
table; [Blouin et al. (2018), section III](https://arxiv.org/html/1807.06616v1)
also distinguishes these ingredients and uses nonideal helium ionization.

## Protection checks

Before the phase-certification correction: 364 regular tests passed, five
optional-data skips; all seven distinct DA/DB cold-start canaries passed.
Both DAB paper checks (9000 and 20000 K) reproduce the preceding tranche's
re-relaxed spectra exactly. The 20000 K case still takes three corrections;
its previously documented difference from the original paper spectrum was
not newly introduced. PG 1225 and J0738 fixed-paper spectra reproduce to
1.83e-11 and 4.24e-14 maximum relative error, with all-depth flux errors
0.001679 and 0.002233 respectively.

After both certification corrections and the precision-transfer tests,
all seven distinct DA/DB cold-start canaries pass with precision correction
enabled and their original iteration allowances unchanged (the standard DB
was also exercised twice by overlapping selections). The regular suite has
402 passing tests and five optional-data skips. Twenty-four research tests
cover the nonlinear model, thermal inertia, domain extension and the stable
transfer experiment below.

The final 9000 K DAB check is unchanged from the preceding tranche. For
20000 K the revised stopping rule takes two applied corrections plus one
measured small proposal, instead of three applied corrections. Compared
with the preceding re-relaxed spectrum, maximum optical relative change is
2.37e-5, maximum UV change 9.29e-4, and integrated change -8.65e-8. The much
larger line-core difference from the original paper figure was already
present before this stopping-rule change. These are saved-paper-atmosphere
checks, not cold-start claims. Outputs:
`results/precision-polish-protected-dab-20260905`.

## Cancellation-resistant Feautrier experiment

`src/wd_spectra/_stable_feautrier.py` (also used by the explicitly scoped
`scripts/stable_feautrier_experiment.py`) preserves the existing boundary and
interior equations but performs elimination in intensity differences. For
`Q*u_i + A*(u_i-u_(i-1)) + C*(u_i-u_(i+1)) = b`, it carries
`G_i=Q_i+A_i*solve(P_(i-1),G_(i-1))`, `P_i=C_i+G_i`, avoiding subtraction of
large transport coefficients. Back substitution and opacity-motion tangents
likewise retain increments instead of subtracting almost equal intensities.
This addresses the small-optical-step precision issue associated with improved
Feautrier elimination (see [Rybicki and Hummer 1991](https://www.nist.gov/publications/accelerated-lambda-iteration-method-multilevel-radiative-transfer-i-non-overlapping)).
The block extension and tangent here were derived and tested independently;
no foreign implementation was copied.

Six tests compare well-conditioned original equations, scattering and opacity
finite differences, and a 70-digit discrete scalar reference with surface
tau=1e-12. On the actual 8000 K coarse state, independent scattering closure
improves from about 1e-6 to 1e-15. Direct and flux-difference energy derivatives
now agree with finite differences to about 1e-5 in tested thin cells. It does
**not** cure the stalled 80-layer atmosphere: its 12-attempt continuation still
misses the temperature gate, and its spectrum remains 0.980718 of target.
A fresh 5000 K test still fails after 20 attempts / 93 s, flux 0.401 and local
0.382. The missing dense-He physics is not repaired by numerical precision.

Additional proposal-only row equilibration on the same stalled 8000 K state
made the actual merit worse, and is not selected. Applying local-energy
equations globally to the warm 22000 K standard model was also slower; two
such experiments were interrupted after exceeding the protection budget.
Removing the fixed Newton Tikhonov term improved formal residuals but alone
took 35 iterations, still above the old test budget. None of these isolated
alternatives was promoted.

## Guarded precision correction selected after protection tests

The combination of stable transfer arithmetic and an unpenalized final Newton
correction works better than either alone. The shared solver enables it only
at the formal-completion transition when actual all-depth flux already passes
but the preceding temperature step is oversized or unmeasured. It uses no
Teff, abundance or layer-count threshold. The warm-start trajectory is
unchanged. Coupled field, independent scalar source check and tangent switch
together, with the phase change invalidating the field cache. The trust
region, full physical trial evaluation and all convergence criteria remain.

The generic nonlinear penalty still defaults to 1e-8; only this final phase
sets it to zero. `use_precision_polish` defaults to true in the shared
atmosphere driver, with requested/used metadata. Explicit checkpoint resumes
are not silently routed through this phase.

| Model | Steps without / with correction | Final all-depth flux error with correction | Final measured dlnT |
| --- | ---: | ---: | ---: |
| DB 22000 K, standard 40 layers | 47 / 28 | 7.56e-9 | 8.89e-5 |
| DB 22000 K, production 80 layers | 29 / 31 | 1.52e-10 | 7.32e-5 |
| Atomic DAB 8000 K, production 80 layers | 25 / 33 | 1.03e-9 | 9.61e-5 |

This is **not a universal speedup**. The standard DB comparison took 182 s
without versus 127 s with correction, but timings have different parallel
contention. The production DB and cool DAB spend extra steps resolving weak
temperature directions. For the 8000 K DAB, local relative energy imbalance
improves from 0.001113 to 4.06e-6, while its coarse spectrum integral remains
0.995353: the independent spectrum test still correctly fails until mesh
and domain refinement. The optical spectrum changes by at most 8.47e-6;
the UV (1200–3000 A) maximum relative change is 0.00547.

The standard DB spectrum changes from its old 20-step warm state by at most
1.96e-5 in the UV (1000–3000 A), 8.98e-4 in the optical (3500–7500 A), and
-1.84e-7 in its integral. Some very thin surface temperatures change by as
much as 0.294 in ln(T). Even after polishing, the first two cells do not pass
the additional local-emission-normalized criterion (maximum 0.256); this
finite-grid global-flux convergence is not a claim of fully verified local
thermal balance. The cooler research successes above require that extra
gate explicitly. Outputs:
`results/db22000-precision-polish-spectrum-20260905`,
`results/cool-dab-precision-polish-20260905`.

The production 80-layer DB comparison likewise gives maximum UV/optical
relative changes 2.31e-5 / 5.01e-4 and integrated change -2.93e-7 relative to
its own 20-step warm state. Output:
`results/db22000-production-precision-polish-spectrum-20260905`.
The unmodified public standard-DB canary was rerun after enabling the default:
28 iterations / 118 s, passed without a wrapper or altered test allowance.

## Saved numerical successes and remaining physics boundary

`scripts/summarize_cool_energy_validation.py` audits all three completed
refinement sequences above. It checks measured nonzero small corrections,
actual all-depth flux, local energy, boundary screening, and recomputes the
independent spectrum integral without normalization. Its summary is
`results/cool-energy-validation-20260905.json`. This is a reproducible audit
of saved results, not a new cold solve.

The updated fixed-state 8000 K DB opacity audit finds line-forming densities
around 0.08–0.12 g/cm3 and maximum density 0.202 g/cm3. This is very different
from the failed 5000 K analytic seed, but does not validate the approximate
nonideal chemistry. See that model's `physics-audit.json`; its four-sweep
column is an explicitly reconstructed diagnostic comparator, not a statement
that production synthesis still uses four sweeps.

The next physical extension should conserve H/He nuclei and charge while
including H2 and He2+ consistently in equilibrium, reaction energy and
opacity. H2-He CIA must use those solved populations. Dense-He ionization
needs a validated nonideal prescription, separately from a bulk-density EOS
table. No 5000 K pure-He success is claimed, and no quarantined solver or
replacement atmosphere has been adopted.

An isolated He2+ charge/enthalpy/opacity prototype now tests the molecular-ion
part of this proposal. Its initial fresh 5000 K atmosphere still fails; it
is not enabled in production or counted among the numerical successes above.
See [the separate experiment record](helium-dimer-experiment-2026-09-05.md).
