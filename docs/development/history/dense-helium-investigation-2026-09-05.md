# Dense-helium EOS and ionization investigation

> Research record: this page describes work at the time it was written.
> For current usage and status, see the [user guide](../../getting-started.md)
> and [tested points](../../tested-temperature-ranges.md).

This is a research experiment, **not a production default or a validated
dense-He atmosphere model**. No production Python or C source was changed in
this investigation. The production worktree already contained the earlier
solver/physics changes; those have been preserved. No commit or push was made.

## Current verified result (September 6)

**Later cross-model review:** the changes have now been tested explicitly,
not only through unchanged production defaults. The opacity-only 421-test
suite and seven cold-start DA/DB canaries pass, but the full dense-He path
does **not** pass a general cross-model qualification. The new 10000 K
experiment fails independent local-energy checks, 22000 K exceeds its
material domain, and separate transfer-component tests expose iteration
regressions and changed DZ flux balance. See
[the cross-model report](dense-helium-cross-model-checks-2026-09-06.md).
Production defaults remain unchanged.

### Latest status: literature comparison and regression review

The opacity-join repair is now implemented and independently checked at
**both 5000 K and 8000 K on 317 layers**. It remains an opt-in research
experiment, not a production-ready replacement. The independent checks
establish numerical convergence for the declared pure-He physics, not
completeness of dense-fluid physics or applicability to other compositions.

The new 8000 K run explicitly re-relaxes the pre-repair, same-EOS 317-layer
saved atmosphere; **it is not a cold start**:
`results/cool-db-dense-smooth-heminus-warm-angle8-317-20260906/8000`.
Two correction attempts take 342.491 s including consistent synthesis.
Maximum all-depth total-flux error is 5.52984e-6, local-energy error
1.96491e-5, and measured maximum dlnT 1.13485e-5. The independent audit at
8000 wavelengths, extending to 10 cm with eight/sixteen angles, gives
spectrum integrals 0.999970648/0.999978336, local-energy errors
0.000226721/0.00100627, and radiation-scaled source-equation error
3.71e-16. All pass the existing gates. Audit:
`results/dense-helium-spectrum-audit-20260906/8000-smooth-heminus-317`.
No separate depth-refinement certificate is claimed for the repaired
opacity: the earlier depth studies used the old join.

The new paired plot, raw arrays and SHA-256 input provenance are in
`results/dense-helium-literature-status-20260906`. Both plotted spectra
come from their independent sixteen-angle audits. The comparison is to
the existing distributed Montreal/Tremblay 1D LTE pure-He IR grid,
ML2/alpha=1.25, at exact Teff and log g=8. The legacy spectral file is
**not identified here as a modern Blouin dense-He grid**. Reference air
wavelengths are converted to vacuum by the existing reader; Hnu is
converted to surface Flambda. No fitted normalization, peak scaling or
output smoothing is applied.

| Model | Optical band difference (0.35--0.90 micron) | IR band difference (1--5 micron) | Spectrum integral / expected |
| --- | ---: | ---: | ---: |
| 5000 K, 317 layers, repaired join | +7.7219% | -18.7247% | 0.999990643 |
| 8000 K, 317 layers, repaired join | -0.5959% | -0.5793% | 0.999978336 |

The remaining 5000 K spectral discrepancy is real; convergence and removal
of the small opacity discontinuity do not resolve it. Collective He-minus
free-free and refraction are still absent, and the nonideal material model
retains the approximations and validity limits documented below. There is
not yet a consistent dense H/He mixture implementation for DABs.

Regression checks repeated for this status review: **416 production tests
passed, five optional-data tests skipped, seven slow canaries deselected**
(57.95 s); **186 portable research tests passed** (7.26 s). The seven
production cold-start canaries had already passed earlier in this
investigation, covering DA 3000/4000/5000/20000 K and DB 10000/22000 K
(including both 22000 K depth presets). These test the unchanged production
path, **not use of the new dense-He path in those other models**. Its safety
as a general replacement has not been demonstrated; warm DBs, DAs, DABs
and DZs must not be described as regression-validated under that path.
Production defaults and Python/C implementation were not changed by this
status review. No commit or push was made. No runs remain active.

**Earlier numerical milestone, 12:49 PDT:** the declared pure-He dense
material experiment converges at both 5000 K and 8000 K. The 8000 K fresh
start independently reproduces the resumed result, and the separate
5000 K 317-to-633 and 8000 K 159-to-317 depth studies pass their pre-existing
0.003 Fstar spectral-change criteria. The requested 0.5063 micron opacity
join repair also reconverges and passes independent thermal/flux checks at
5000 K (317 layers) and 8000 K (80 layers), as detailed below. The repaired
8000 K demonstration is still on the coarse grid: the pre-repair depth
certificate is not automatically a certificate for the changed opacity.
No runs remain active at this milestone, and no production default was changed.
This is a numerical result, not completion of dense-fluid opacity physics or
a consistent nonideal H/He mixture model.

**Update, 12:44 PDT: fresh-start 8000 K reproduction succeeds.**
`results/cool-db-dense-mass-pseudotime-cold-angle8-80-20260906/8000`
starts from a fresh hydrostatic/discrete-transport seed, takes 28 nonlinear
pseudo-time initialization sweeps, then five static correction attempts.
The handoff is automatic when the existing local-energy tolerance is met;
no saved atmosphere or alternative EOS is used. Final all-depth flux error
is 1.40899e-5, local error 2.15002e-5, measured maximum dlnT 1.13313e-5.
Elapsed time including initialization and consistent synthesis is 1288.555 s
under concurrent load, not a controlled performance benchmark. Independent
eight/sixteen-angle audit integrals are 0.999969368/0.999976955 and local
errors 0.000226892/0.000939292. Relative to the earlier resumed 80-layer
result, spectral L1 difference is 1.15970e-5 Fstar and optical/IR band
changes are -0.0009373%/+0.0012932%. Audit:
`results/dense-helium-spectrum-audit-20260906/8000-mass-cold-80`.

The 159-layer 8000 K refinement also converged in four static attempts:
flux error 5.28748e-6, local error 7.71530e-6, measured dlnT 4.06957e-6;
400.622 s including synthesis. Independent eight/sixteen-angle integrals
are 0.999974057/0.999981720 and local errors 0.000227438/0.00101040.
Run `results/cool-db-dense-massconservative-depthrefinement-angle8-159-20260906/8000`,
audit `results/dense-helium-spectrum-audit-20260906/8000-mass-volume-159`.
The 80-to-159 spectral L1 difference is 0.0118991 Fstar (IR +2.3232%), so
80 layers are not depth-resolved. Furthermore the seed-screened structure
line quadrature changed from 4324 to 4273 wavelengths; the strict comparison
tool correctly refused to label this an isolated depth certificate.
The 317-layer refinement subsequently converged in four static attempts,
1106.207 s including synthesis: flux error 7.92417e-5, local error
0.000104133, maximum measured dlnT 5.58030e-5. Its 8000-wavelength
eight/sixteen-angle audit integrals are 0.999929979/0.999937664 and local
errors 0.000274753/0.000967916. Both fine grids use the same 4273-node
structure wavelength count. Their matched spectral L1 change is
0.00285032 Fstar and maximum lambda-weighted change 0.00191504 Fstar,
passing the pre-existing 0.003 criteria (not a stronger band-relative
precision claim: the IR-band change is 0.5562%). Certificate:
`results/dense-helium-spectrum-audit-20260906/8000-mass-depth159-to317.json`.
The 317-layer spectrum differs from the exact Montreal reference by -0.552%
in integrated optical flux and -0.604% in the 1--5 micron band:
`results/dense-helium-spectrum-audit-20260906/verified-comparison8000-mass317`.

The new mass-transfer 5000 K 317-to-633 study now passes independently:
`results/dense-helium-spectrum-audit-20260906/5000-mass-depth317-to633.json`.
Spectral L1 change is 0.000539162 Fstar and maximum lambda-weighted change
0.000451691 Fstar. The 633-layer physical flux/local errors are
0.000831300/0.000174708. Its bounded inner iteration hit its budget, so the
small proposal was checked independently with an **unregularized actual
static Newton solve**: maximum dlnT is 8.85798e-5 (below the unchanged
0.0003 gate), and a full, admissible, independently evaluated trial reduces
the local error to 6.74480e-5. This independent correction audit is
`results/dense-helium-tangent-audit-20260906/5000-mass633-static-newton.json`.
The broad-wavelength audit gives eight/sixteen-angle integrals
1.000039360/1.000048948 and local errors 0.000174707/0.00128003.
The long runtime, 2582.058 s including synthesis, exposes inefficiency in
the near-converged augmented least-squares proposal; it is not a speedup.

**Update, 12:18 PDT: 8000 K has also converged on its 80-layer grid, and
passed independent wavelength/angular/source checks.** The successful
sequence is mass-conservative transfer, nonlinear-ML2 pseudo-time local
relaxation, then full static correction, with the same dense EOS/ionization
throughout. This first demonstration resumes an earlier *unconverged*
iteration, not a cold start. Fresh initialization and depth refinement are
now being tested. No production default has changed.

Run: `results/cool-db-dense-mass-static-after-pseudotime-angle8-80-20260906/8000`.
Four static correction attempts take 158.775 s (168.020 s including consistent
synthesis, under concurrent load), **excluding** the preceding pseudo-time
initialization. All-depth flux error is 9.12235e-7, local energy error
1.83454e-6, measured maximum dlnT 1.30824e-6. Direct spectrum integral is
0.999972465. The lower-boundary absorption escape bound is 1.60e-81.

Independent audit:
`results/dense-helium-spectrum-audit-20260906/8000-mass-volume-80`.
At 8000 independently sampled wavelengths out to 10 cm, eight/sixteen angles
give flux integrals 0.999977266/0.999984857 and local errors
0.000218276/0.00106019. Radiation-scaled independent source-equation error
is 3.05e-16. The unweighted relative source error in the negligible Wien tail
is larger (up to 5.69e-9) and is retained in the diagnostic. Fixed-depth
numerical qualification is true; depth qualification and full-physics
validation are still false. The ordinary four-sweep spectrum differs by
about 0.77% in bolometric flux and is not the matched transfer result.

The **5000 K pure-He experiment now has independently verified numerical
convergence for its declared experimental physics**. This is not validation
of all dense-helium physics, nor a solution for mixed H/He atmospheres.

Run: `results/cool-db-dense-fullthermal-angle8-hierarchy317-20260906/5000`
(paths here are relative to the outer research workspace). It starts fresh,
solves 80 -> 159 -> 317 layers with unchanged physics, and uses eight angles
and a thermal wavelength interval of 100 A to 1 cm. The final all-depth flux
error is 5.86609e-4, local energy error 5.15266e-5, and measured maximum dlnT
1.47608e-5. Elapsed time including ordinary synthesis was 677.997 s under
concurrent load; this is not a controlled speed benchmark.

Independent audit:
`results/dense-helium-spectrum-audit-20260906/5000-fullthermal-angle8-317`.
It uses 8000 separately sampled wavelengths extending to **10 cm**, directly
coupled scattering, and finer angular quadratures:

| Angles | Spectrum integral / sigma Teff^4 | Maximum local energy error |
| --- | --- | --- |
| 8 | 0.999987578 | 5.11754e-5 |
| 12 | 0.999995854 | 9.73721e-4 |
| 16 | 0.999997170 | 1.27098e-3 |

All pass the unchanged 0.003 physical-energy/flux gates. The audit's
`qualification.json` records numerical qualification true and
`validated_full_physics` false. The recommended spectrum is the audit's
`coupled-feautrier-16.txt`, not the ordinary four-sweep synthesis output.
Its columns are vacuum wavelength in A and surface F_lambda in
erg s^-1 cm^-2 A^-1. No flux renormalization is applied.
The qualification scope is explicitly **fixed-depth Feautrier equations**:
wavelength/angular/source tests do not, by themselves, establish depth-grid
convergence. Qualification files now carry this scope and a separate false
`independent_depth_resolution_verified` field until a depth study is supplied.

**The separate 317-to-633 depth study now passes for 5000 K.** Run:
`results/cool-db-dense-fullthermal-depthrefinement-angle8-633-20260906/5000`.
This explicitly resumes/refines the same-physics 317-layer solution; it is
not a separate cold start. The 633-layer atmosphere converged with flux error
6.81596e-6, local energy error 8.00568e-7, and measured dlnT 1.58683e-7.
Elapsed time including consistent synthesis was 766.748 s under concurrent
load. The independent audit at 8000 wavelengths out to 10 cm gives:

| Angles | Spectrum integral / sigma Teff^4 | Maximum local energy error |
| --- | --- | --- |
| 8 | 1.000009969 | 2.30773e-6 |
| 16 | 1.000019560 | 1.27875e-3 |

On identical wavelength and angular grids, integrated absolute spectral
change from 317 to 633 layers is **2.85097e-4 Fstar**, and maximum
wavelength-weighted change is **2.28288e-4 Fstar**. Both are below the
unchanged 0.003 tolerance. Optical and IR band flux changes are +0.01513%
and -0.03185%. The depth certificate is separate from the fixed-depth audit:
`results/dense-helium-spectrum-audit-20260906/5000-depth317-to633-matched.json`.
The 633-layer audit and spectra are in
`results/dense-helium-spectrum-audit-20260906/5000-depthrefined-633`.
No claim of complete dense-helium physics follows from these numerical tests.

A preliminary comparison with different wavelength grids had a 0.00699
pointwise discrepancy despite tiny integrated change; matching the grids
removes it. Interpolation across the sharp opacity-prescription join must
not be confused with a depth error. That preliminary failed comparison is
retained, not silently overwritten.

The verified literature plot is
`results/dense-helium-spectrum-audit-20260906/verified-comparison5000/dense-helium-5000-literature.png`.
Against the exact 5000 K/log g=8/pure-He Montreal reference, integrated
optical flux (3500--9000 A) is 7.87% high and infrared flux (1--5 micron)
18.80% low. Numerical convergence has not removed the physics discrepancy.

An additional fresh run replaces the preliminary gradient relaxation with
material-exact local thermal initialization and a finite-material convective
trial correction. The second fresh hierarchy completed in 1024.952 s: final flux and
local errors are 5.86611e-4 and 5.15266e-5. Temperatures agree with the first
run within 2.33e-10 relative and spectrum integrals within 4e-14. This
establishes reproducibility with a different initializer, not a speedup.
The initializer is explicitly not a convergence certificate: radiation and
all actual ML2 fluxes are recalculated before acceptance. Backtracking tests
the actual energy residual, and no physical flux is capped or overwritten.

Before the successful sequence above, 8000 K was **not solved**. The unprojected thermal initializer oscillated,
then the physical solve stalled near 1.16% local energy error despite good
total flux. That variant was stopped with artifacts retained. A safeguarded
fresh run and an explicitly labeled same-physics temperature predictor from
the verified 5000 K atmosphere were tested without convergence. The latter maps T/Teff at
fixed optical depth onto a fresh target pressure grid, refuses extrapolation,
and must solve the target's complete equations; it is **not a cold start**.

Latest complete portable research batch: **186 passed in 7.13 s**, including
the new conservative mass-volume transfer, tangent, scope-restoration and
consistent synthesis tests. The portable batch is
`pytest scripts --ignore=scripts/test_d6_mg_na_profile_policy_crossstar.py`;
the unrelated D6 research test imports a module absent from production and
cannot be collected against this package. This is not a production test
failure and was not repaired by restoring an alternate source tree.
Production sources remain unchanged by this investigation, and the earlier
416-test regular suite and seven protected cold-start canaries remain the
production regression results (details below). The unchanged production
helium tests were rerun: 37 passed, four skipped (optional profile tables
not downloaded), in 0.95 s.

### Repair of the 0.5063 micron He-minus opacity join

The visible spectral jump is not CIA or a physical threshold. It is the
automatic switch from the Carbon/John-1968 fit to the John-1994 He-minus
free-free table. At 0.5063 micron the two coefficients differ by 4.19% at
8000 K, 7.62% at 6000 K and 11.36% at 5000 K. John (1994), Table 2 and
section 4 were checked against the original scanned pages:
`tmp/pdfs/helium/john1994.pdf`, published pp. 875 and 878.
[Primary source](https://doi.org/10.1093/mnras/269.4.871).

`scripts/heminus_join_experiment.py` supplies an **explicit research-only**
`--smooth-heminus-join` scope. The automatic coefficient is a convex cubic
smoothstep blend in log wavelength over 0.5063--1 micron. The endpoints are
the published table edge and the paper's stated infrared-accuracy boundary,
not parameters fitted to a target spectrum. The blend is our numerical
prescription, **not a formula attributed to John or a demonstrated increase
in opacity accuracy**. Neither source is rescaled; the John table is not
extrapolated outside its temperature/wavelength domain. The coefficient
stays positive and monotonically increasing with wavelength over the full
tabulated temperature range, and matches value/first derivative at both
join boundaries. Explicit `john1968`/`john1994`, UV, IR >=1 micron, and
temperatures outside the original 1400--10080 K range remain bitwise
unchanged. The separate hard temperature-boundary joins are not repaired
by this wavelength-only change.

Both structure and spectrum use the declared join policy. Metadata and
independent audits record/reapply it; synthesis rejects a mismatch, depth
certification rejects different opacity policies, and continuation cannot
silently revert a smooth-opacity run to the historical switch. The
production default has **not** been changed.

At 8000 K, explicit same-grid re-relaxation after the opacity change took
two correction attempts (61.157 s including synthesis). Flux error is
5.03373e-6, local error 5.78915e-5, measured dlnT 4.58316e-5. Independent
eight/sixteen-angle spectra integrate to 0.999971477/0.999979073 and have
local errors 0.000217661/0.00107994. The source-equation check is 3.30e-16
on the radiation scale. At 5000 K, 317 layers reconverged in two attempts,
153.780 s: flux error 7.49046e-5, local 4.99487e-5, dlnT 1.86472e-5.
Its independent eight/sixteen-angle integrals are 0.999981082/0.999990643
and local errors 4.92926e-5/0.00130385, with a radiation-scaled source
equation error 3.89e-16. Both repaired models pass the unchanged gates.

Resolved 8000 K comparison:
`results/dense-helium-spectrum-audit-20260906/8000-heminus-boundary-comparison/heminus-boundary-comparison.png`.
Across just 0.002 Angstrom around the old boundary the historical spectrum
jumps by 0.65517%; the repaired spectrum changes by -6.33874e-7 relative,
consistent with its ordinary continuum slope. This is a fresh transfer
calculation on each re-converged atmosphere, not smoothing an output plot.
These 80-layer spectra are not advertised as depth-resolved final models.
The corresponding 5000 K edge changes from a 1.93204% jump to a smooth
9.51643e-8 relative change across the same 0.002 Angstrom interval; plot:
`results/dense-helium-spectrum-audit-20260906/5000-heminus-boundary-comparison/heminus-boundary-comparison.png`.
The repaired versus historical optical/IR band flux changes are
-0.04612%/+0.01796% at 8000 K and -0.16506%/+0.18152% at 5000 K.
The change fixes an artificial discontinuity, not the remaining broader
5000 K spectral discrepancy or the missing collective opacity/refraction.

### Column-mass-conservative transfer experiment

A fixed-radiation diagnostic exposed a material-dependent discretization
artifact. The original Feautrier cell width is the mean of adjacent optical
depth increments. Multiplying this by the *nodal* absorption fraction mixes
neighbouring extinction into local heating/cooling, differently at every
wavelength. Its discrete flux conservation is correct, but it is not the
physical fixed-column-mass cell quadrature when opacity varies sharply.
At the original stalled 8000 K state, node 42 has only a distant 4266--4301 K
root with those weights, whereas the physical nodal absorption weighting
gives a nearby 5590--5636 K root. This is a fixed-J diagnostic, not proof of
multiple global atmosphere solutions or their stability.

The research `mass_conservative_feautrier.py` changes **transfer and energy
together**: face optical resistances retain trapezoidal extinction integrals,
while each material cell uses its actual nodal extinction times its physical
column-mass width. Consequently its flux divergence equals
`4 pi Delta_m integral(kappa_abs (J-B))`. The exact material/radiation tangent
and its local cooling derivative use the same volumes; cooling has no
spurious neighbouring-opacity derivative. Surface and bottom conditions and
all physical gates are unchanged. Synthesis and independent source/energy
audits explicitly select the same operator. The wrapper refuses a mismatched
mass/extinction cache or a synthesis/structure discretization mismatch.
No flux is overwritten and no EOS or opacity coefficient is changed.

Nine new numerical tests cover constant-opacity equivalence to the earlier
operator, variable-opacity flux conservation, and complete finite-difference
responses down to absorption fraction 1e-6. A perturbation-size study exposes
subtractive noise in thin-cell near-conservative-scattering finite differences;
the tested 1e-3 perturbation resolves these derivatives without loosening
atmosphere tolerances. Additional tests check independent fixed-source closure,
scope restoration, and consistent synthesis.

The **5000 K, 317-layer** same-state comparison converged in two attempts:
all-depth flux error 2.33916e-5, local energy error 9.08079e-6, final measured
correction 3.16533e-6. Direct spectrum integral is 0.999996008. It starts from
the previously verified 317-layer state, not from scratch. Largest first
correction was 0.00114651 in log T. Independent 8000-wavelength/10-cm checks
pass: 8/16 angles give flux integrals 1.000010494/1.000020072, local energy
errors 8.67836e-6/0.00128336, and radiation-scaled independent source error
3.42e-16. Audit: `results/dense-helium-spectrum-audit-20260906/5000-mass-volume-317`.
Compared with the old optical-depth-volume **633-layer** result on matched
wavelengths/16 angles, integrated absolute spectral change is 8.09764e-4
Fstar and maximum wavelength-weighted change 6.64302e-4 Fstar. This is a
cross-discretization comparison, not a same-operator depth certificate.
Optical/IR band changes are +0.0150%/-0.0596%. Run:
`results/cool-db-dense-massconservative-refinement-angle8-317-20260906/5000`.

The **8000 K** continuation of the old uncompleted iteration did *not* converge.
It initially reduced its new local error from about 0.33 to 0.024, then stalled
near 0.0244 after ten attempts and was stopped. The worst node moved to 46;
its physical fixed-J thermal root is now around 3890 K, far below its 5229 K
trial temperature. Changing the cell volumes is therefore a supported
discretization improvement, not by itself a demonstrated 8000 K solution.
The temperature-initializer barrier-crossing variants with the old operator
also failed and were stopped with artifacts retained. A same-physics warm
7000-to-8000 K continuation with the new operator is being tested separately.

### Exact bulk-material tangent and pseudo-time initialization

The actual-material directional audit exposed a second numerical issue:
the centered finite difference used in the atmosphere's convection tangent
can cross a REOS PCHIP knot. Rho and internal energy are C1, but derivatives
of Cp and Q are only piecewise smooth. In one tested uniform perturbation,
the old adiabatic-gradient tangent has relative error 0.3163 even as the
independent perturbation is reduced to 2e-6. The ML2 flux-coefficient tangent
error is 0.03983. These are not merely unused surface-interface discrepancies.

`exact_dense_tangent_experiment.py` explicitly replaces only the bulk-material
derivatives with analytic derivatives of the **same interpolants**. Both
physical flux and local-energy/scale Jacobians are updated before any row
transformation. The accepted EOS, radiation and convective flux do not change.
The Rosseland response remains centered. Independent uniform perturbations
at 2e-5 now give errors 1.16e-9 (adiabatic gradient), 5.99e-10 (ML2 flux
coefficient), and 2.55e-8 (radiative-loss coefficient). At 2e-6 these remain
small. Larger perturbations crossing a knot correctly do not agree with a
single local derivative. Audit:
`results/dense-helium-tangent-audit-20260906/8000-mass-exact-convection.json`.

A separate research initializer uses one linearly implicit pseudo-time step
per actual radiation/material evaluation. Positive net cell heating raises
log T and cooling lowers it; the lower stellar-flux boundary is algebraic.
Positive local transport rates precondition the time derivative. They are
not physical heat capacities, and no physical evolution times are claimed.
Temporal-defect and temperature-step controls replace static-merit monotonicity
**only during initialization**, allowing a thermal trajectory past a local
static-error minimum. The ordinary static solve must still measure its own
correction and pass all physical gates afterward. Every accepted initializer
state and residual is logged and explicitly marked not converged.
This follows the one-Newton-step formulation described in
[PETSc TSPSEUDO](https://web.cels.anl.gov/projects/petsc/vault/petsc-3.22.5/docs/manualpages/TS/TSPSEUDO.html),
based on [Kelley and Keyes 1998](https://epubs.siam.org/doi/10.1137/S0036142996304796).
Its application here and the local time preconditioner are an experiment,
not a claim that those references validate this particular atmosphere model.
Runs with the old finite-step and corrected analytic material tangents are
compared. The linearly implicit initializer hits the same convection-onset
bottleneck with either material tangent: after initial progress, steps shrink
to microscopic values near local error 0.09097. Both were stopped, retaining
their separate JSONL telemetry and initializer snapshots.

The next explicit variant solves **nonlinear ML2 compatibility and convective
flux inside each pseudo-time proposal**, while retaining a positive-rate
radiative tangent model. Its fixed-scale cell equations include the transient
log-temperature term and the algebraic lower-boundary flux. Actual materials,
radiation and gradient-derived convection are recalculated before acceptance;
the same temporal-defect and temperature controls apply. This is not the
strictly one-Newton-step PETSc formulation, and is labeled a nonlinear
convective proposal. Its complete inner tangent passes finite-difference
tests at multiple pseudo-time steps. It crosses the earlier onset bottleneck:
after five accepted steps, local error is 0.04068, versus the linear variant's
0.09097 stall. All-depth flux error remains 0.02265. It is **not yet converged**.
Run: `results/cool-db-dense-mass-pseudotime-implicit-partialresume-angle8-80-20260906`.

The implicit variant subsequently traversed the local cooling transition;
its temporary physical fluxes and local error rise during parts of this
initialization, so it must not be described as monotonic static convergence.
At sweep 20, local error is 9.15774e-7 but the all-depth flux level is still
about 2.3% low. An exact saved initializer state was explicitly handed to
the static solver, which converged as documented above. Source bytes are
read once, hashed, and retained as the new run's seed; no alternate EOS,
opacity, atmosphere or normalized flux is substituted. The successful
result is not the initializer's small local residual alone.

An explicit `--pseudo-time-until-local-balance` now automates this handoff
using the existing local physical tolerance, not a Teff threshold. It skips
unnecessary relaxation of a locally balanced state, but always requires
the static solver to measure its own correction and satisfy every gate.
A fresh discrete-transport/8000 K run is testing the complete sequence
without an external state. A 159-layer refinement of the verified 80-layer
result and a 633-layer 5000 K refinement are running independently.

The separate 7000-to-8000 warm comparison did not converge. It stalled near
24.3% local error and 1.96% all-depth flux error, then a material derivative
probe exceeded the declared upper domain (17002.6 K versus 17000 K).
That run stopped with a domain error and no substitution; it is not a
completed or certified 8000 K atmosphere.

New run options also record the thermal driver's complete flags and the
actual interaction-table SHA before initialization, so failed/warm-start
iterations can be resumed with explicit material provenance. The new
version-2 certificate does not require a fictitious discrete cold seed;
old version-1 hierarchy artifacts retain their original certificate checks.

Further operator checks independently solve a variable-opacity mass-volume
matrix at 70-digit precision and reproduce the computed field/flux. An
isothermal semi-infinite test converges quadratically toward its known pi B
surface flux: errors at 40/80/160/320 nodes are
0.0077750/0.0019197/0.00047546/0.00011821. Direct research synthesis now uses
an independent prescribed-source solve for its source-equation guard, not
the algebraic identity involving the coupled J that constructed S.

### Temperature continuation and synthesis consistency

An explicitly labeled fixed-pressure continuation from the verified 5000 K
model to **6000 K** converged in 124.936 s, eight attempts, on 80 layers.
All-depth flux error: 1.80657e-6; local energy error: 1.04417e-6; measured
dlnT: 4.04513e-7. Output:
`results/cool-db-dense-fullthermal-pressurecontinuation-fixed-angle8-80-20260906/6000`.
This is a warm start, not a fresh 6000 K demonstration.

The independent 8000-wavelength/10-cm audit passes: eight/sixteen angles
give flux integrals 0.999968452/0.999978273 and maximum local energy errors
2.66778e-6/0.001174208. The ordinary four-sweep piecewise-linear synthesis
instead gives 0.951755843. Thus its 4.8% discrepancy cannot be attributed
to an unconverged atmosphere. The recommended spectra remain the coupled
audit spectra. `dense_direct_spectrum.py` adds an explicit research-only
`--direct-spectrum` option using the same coupled transfer law as structure;
it checks source closure and never changes the atmosphere convergence flag.
Production synthesis remains unchanged.

An additional direct Lambda-matrix solve isolates the source-iteration issue
from depth discretization. At 6000 K, 4000 wavelengths to 10 cm, the old
piecewise-linear operator gives 0.951762547 after four sweeps and 0.951825211
after exact coupled scattering. The Feautrier operator gives 0.999975580.
**Most of the 4.8% difference is therefore the transfer discretization, not
the four-sweep limit.** Merely increasing scattering iterations will not fix
it. Using consistent structure/synthesis transfer removes the inconsistency,
but depth refinement is additionally required to assess spectral accuracy.
The 80-layer continuation spectra are not claimed depth-resolved.

The depth-controlled 7000 K continuation passed all physical gates in
301.796 s (nine attempts): flux error 1.27795e-6, local error 1.52589e-6,
measured dlnT 7.87924e-7. Its consistent synthesis integral is 0.999963527.
The independent 8000-wavelength/10-cm, 8/16-angle audit gives flux integrals
0.999971122/0.999979080 and local errors 4.23441e-6/0.000983923. Again this
establishes a fixed-depth solution, not depth resolution.

The depth-controlled **6000 K 317-layer** refinement completed in 349.619 s
(three attempts): all-depth flux error 2.13599e-6, local error 8.23321e-7,
measured dlnT 1.88887e-7. Its direct spectrum integral is 0.999957788 and
the estimated bottom-boundary escape bound is 3.85e-15. The independent
8000-wavelength/10-cm audit gives flux integrals 0.999968862/0.999977548
and local errors 2.04467e-6/0.001185357 for 8/16 angles.

The 80-to-317 depth comparison is **not within the 0.003 spectral-accuracy
tolerance**: integrated absolute spectral change is 0.0186964 Fstar,
maximum wavelength-weighted change is 0.0137936 Fstar, and IR-band flux
changes by -2.443%. This does not mean the refined atmosphere failed its
equations; it means the coarse spectrum was not depth-resolved. A further
refinement comparison is required before certifying depth resolution.
Artifacts: `results/dense-helium-spectrum-audit-20260906/6000-depth80-to317.json`.

Direct temperature mapping onto a fresh hotter-star pressure grid was a
poor predictor: enforcing the old convective fractions there drove the
initializer outside its material domain. Those attempts were rejected.
Keeping the source pressure grid enabled the 6000 K success, but retained
too much deep atmosphere (bottom Rosseland depth 2.28e5). A subsequent
explicitly requested source-depth-100 truncation/resampling is being tested;
independent lower-boundary opacity checks remain necessary. No EOS is
extrapolated and no failed target is substituted by its cooler source.

The first `pressurecontinuation` directory contains a failed wrapper test:
its seed hook was installed too late, so its printed retained-grid label
was incorrect. `VALIDATION_STATUS.md` marks this explicitly. The separate
`pressurecontinuation-fixed` directory uses the corrected hook, with a
regression test verifying installation before seed construction and scope
restoration. No atmosphere success was claimed for the failed wrapper.

Latest complete batch after the predictor/scoping work: **123 passed in
9.74 s**, followed by six passing direct-synthesis/predictor tests. All
these changes remain outside production Python/C sources.
The subsequent complete batch had **126 passed in 9.77 s**; the new depth
comparison tests also check that equal bolometric integrals cannot conceal
a redistributed spectrum.

### 8000 K: local convection scaling and actual-material trial consistency

An 8000 K trial formed tiny convective pockets in optically thin cells.
One interface carried 2.12665e-5 Fstar, while its adjacent cell emitted
only 3.57988e-13 Fstar thermally. A compatibility tolerance scaled solely
to stellar flux can therefore permit an enormous local energy defect.
This is a local numerical scaling issue, not evidence for a Teff cutoff.

The research-only `--thermal-scaled-compatibility` uses a smooth scale
based on actual adjacent-cell emission and auxiliary convective flux,
capped smoothly by the stellar-flux scale. Its complete temperature and
velocity derivatives are tested against finite differences. The stable
branch retains its signed stability-gap normalization. A regression test
explicitly reproduces the thin-cell counterexample. This changes the
proposal norm, not physical ML2 flux, energy equations or final gates.
The 8000 K comparison initially avoids the false local-convection spikes
but has not converged; it is not adopted as a successful solver fix.

The cheaper augmented proposal still linearizes Rosseland opacity, even
though its bulk EOS response is exact. `--exact-ml2-trial-projection` now
reconstructs the proposed convective flux with **actual EOS and Rosseland
opacity at every outer backtracking factor**. It corrects trial temperatures
only. The ordinary evaluator then recalculates the full radiation and ML2
fluxes; trust limits and physical acceptance gates remain unchanged.
This is distinct from scaling a previously projected temperature step.
An initial integration run caught a lost auxiliary diagnostic across a
payload copy; the copy now explicitly propagates that proposal. The failed
`actualcompat-continuation` directory is not a solved model. The separate
`actualcompat-fixed-continuation` run tests the correction.

A separate fully reevaluated temperature-plus-ML2 Newton experiment applies
the same local compatibility norm, recalculating all material/transfer fields
at every trial instead of using a cheap inner radiation/opacity model.
It retains separate actual-gradient flux and local-energy convergence gates.
Neither 8000 K experiment is a convergence claim while it is running.

The actual-material projection comparison also failed to reach useful local
accuracy and was stopped with artifacts retained. The fully coupled system
was tested with both dynamic norms and norms held fixed within each Newton
step. Holding weights fixed avoids differentiating a saturated normalization,
but did not by itself solve the atmosphere.

An explicit continuation of the best **unconverged** 8000 K iteration keeps
its Teff, pressure grid, temperature values and material-table SHA unchanged.
`dense_iteration_resume.py` refuses Teff/resolution changes and checks the
earlier initializer's material certificate. Its source is not certified
converged: local energy error is 0.0116404 despite total flux error 6.36e-5.

A directional audit of that state found a row/column-scaled Jacobian
condition number of 4.43e7 and an unconstrained Newton temperature correction
of **5333.57 in log-temperature units**. Clipping the entire direction to
0.04 leaves a maximum predicted residual change only 8.77e-8; nonlinear
compatibility error overwhelms that useful change. This explains the tiny
accepted changes without attributing them to a missing radiation derivative.
The separate actual-radiation finite-difference checks agree closely on
smooth/global perturbations; thin perturbations also expose expected
roundoff sensitivity, so those global relative norms alone are not a
blanket derivative-accuracy certificate.

The next research proposal eliminates the linear ML2 velocity constraints
and minimizes linear energy error inside a physical log-T box (BVLS), rather
than clipping an enormous unconstrained direction. Its block elimination
supports the dense secant updates supplied after rejected steps; the first
test incorrectly assumed the analytic diagonal block persisted and exited
with an assertion. A regression test now covers that rank-one update.
An optional nonlinear material repair then makes the trial's temperatures
and auxiliary velocities compatible with the actual EOS/opacity. Neither
procedure changes the evaluated physical flux or final acceptance gates.
8000 K remains an unresolved test, not a successful result.

## Critical finding: thermal wavelength truncation

**The historical 5000 K successes below are convergence on the original 10-micron
structure interval, NOT yet full-range thermal convergence.** Their spectra
have good bolometric integrals, but that is insufficient. An independent
4000-wavelength coupled Feautrier check on the 317-layer state gives local
energy error 0.094414 when integrated to 1 mm, versus 1.27e-5 when restricted
to the structure's 10-micron upper limit. At the surface, 29.48% of the
opacity-weighted thermal emission lies beyond 10 microns. The bolometric
flux changes by only about 0.03%, explaining why a spectrum-integral test
did not expose this defect.

`extended_thermal_wavelength_experiment.py` now preserves every original
quadrature node and extends the thermal interval at the same log spacing.
Fresh 5000/8000 K same-physics hierarchy tests to 1 cm were launched. This
is a quadrature correction, not a fitted opacity or relaxed tolerance.
The old literature-comparison plot is superseded as a convergence claim.
It remains a record of the truncated-interval spectrum.

The first full-range 5000 K hierarchy (80 -> 159 -> 317, three angles) has
now converged from scratch in 447.16 s including ordinary synthesis:
all-depth flux error 2.9912e-4, local error 2.5672e-5, measured dlnT
7.9658e-6. Its ordinary spectrum integral is 0.999206. An independent
8000-wavelength check to **10 cm**, with the same three angles, gives
flux integral 0.99997886 and maximum local error 2.5310e-5. Thus the
long-wavelength defect is actually corrected, not concealed by the flux
integral. Optical and IR band fluxes change by only -0.03385% and -0.02035%
relative to the original truncated spectrum.

However, increasing the independent angular quadrature to four/eight angles
gives local errors 0.002908/0.005882. The eight-angle check fails the 0.003
gate, despite good spectrum integrals. The fresh eight-angle hierarchy above
resolves this; the three-angle result is **not fully numerically
qualified**. `audit_dense_spectrum.py` writes a separate `qualification.json`
combining broader-interval thermal, bolometric and source-equation checks.
The research runner's misleading `independent_local_energy_verified` field
has been corrected for future runs: its old calculation only checked the
structure grid. It now records `structure_grid_local_energy_verified` and
leaves the independent field null until an actual independent audit.

`test_dense_thermal_verification.py` explicitly catches a case with negligible
missing bolometric flux but a large omitted thermal-cooling tail. The latest
complete portable research batch at that stage had 115 passing tests (9.77 s),
followed by a passing five-test thermal-initializer/cache batch. Production source and
the protected seven-model regression results remain unchanged.

The bounded 8000 K incomplete-coarse experiment stopped at its 159-layer
initializer's own transport check (error 0.0788); no final atmosphere was
substituted. The separate 10-micron exact-thermal-initializer run was stopped
once the wavelength defect was identified. Their diagnostic files remain.

## Physical construction

The implementation is under the outer workspace's `scripts/` directory:

- `dense_helium_fluid_experiment.py`: neutral/trace-ion Ornstein–Zernike
  equations, explicit HNC closure, chemical-potential generating functional.
- `dense_helium_chemical_equilibrium.py`: He, He+, He2+, electrons, exact
  nuclear and charge conservation and both nonideal mass-action equations.
- `chang_helium_potential.py`, `helium_dimer_state_sum.py`: documented molecular
  interaction and a bound rovibrational state sum, not a fitted electron floor.
- `dense_helium_molecular_experiment.py`: scoped EOS/opacity/runner adapter.
- `smooth_reos3_experiment.py`: C1 shape-preserving interpolation of REOS.3
  density and internal energy, with Cp and Q differentiated from those same
  interpolants. This is NOT a globally Maxwell-consistent free-energy fit.

The structure of the calculation follows the neutral-dominated bulk EOS plus
trace nonideal chemistry approach described in
[Blouin et al. 2018](https://arxiv.org/html/1807.06616v1). Density and bulk
caloric quantities come from He-REOS.3. Trace chemistry uses excess potentials
for He, He+, He2+ and e. The ionization shift is mu(He+) + mu(e) - mu(He);
the dissociation shift is mu(He2+) - mu(He) - mu(He+). Opacity uses the actual
molecular density for bound-free absorption and the actual He*He+ product
for collision free-free absorption. The chemical shifts do not move
photoionization edges.

The calculation enforces a <=0.1% ionization domain for the neutral-bath
approximation. It has no automatic alternate EOS, electron floor, opacity
scale fitted to a star, or effective-temperature switch. Out-of-domain trial
steps can be rejected by the nonlinear driver; no alternative material state
is substituted. An invalid initial state remains an error.

### Sources and limitations

This is **not an exact reproduction of Kowalski et al. 2007**:

- Neutral He uses the exp-six potential of Young, McMahan & Ross (1981),
  PRB 24, 5119, rather than the Ross & Young (1986) effective potential.
  [Primary full text](https://harvest.aps.org/v2/journals/articles/10.1103/PhysRevB.24.5119/fulltext).
- The two He–He+ channels use Bruno et al. (2010), Physics of Plasmas 17,
  112315, equations 3 and 6. Those transport fits lack the far polarization
  tail. Equal-weight bath channels share an indirect correlation; their
  Boltzmann factors, not their potentials, enter the trace-fluid equations.
- He2+–He uses the ground He3+ surface from
  [Chang's 2002 dissertation](https://hdl.handle.net/2346/9503), chapter 2,
  Tables 2.2–2.3, at a fixed isolated-dimer bond and spherically averaged
  potential. It is not the Scifoni surface used by Kowalski.
- The signs printed for two Chang switching functions contradict the
  stated asymptotic limits. The implementation explicitly follows the
  stated limits and independently reproduces the published minima:
  r(He2+) = 2.046179 bohr, r(linear He3+) = 2.340 bohr, and trimer binding
  0.175116 eV versus the reported 0.1751 eV. This interpretation is documented
  rather than silently presented as recovered author code. The fit's very
  short-range/high-energy region is not automatically validated for fluids.
- Electron insertion uses equations 164–166 and Table 2 in
  [Kowalski's 2006 dissertation](https://whitedwarf.org/theses/kowalski.pdf)
  **as printed**. The rho=0.05 join is continuous but has a derivative jump,
  despite prose describing a smooth join. The final 2007 article/data would
  help resolve that discrepancy. It has not been patched with guessed
  coefficients.
- HNC is an explicit alternative fluid closure. PY trials yielding negative
  pair distributions were rejected. HNC numerical convergence is not proof
  of quantitative agreement with Kowalski's chemical potentials.
- Collective dense-He corrections to He-minus free-free and refractive
  transfer are still absent. Stancil molecular cross sections retain their
  existing hold below 4200 K. None of these omissions is disguised by a
  convergence flag.

The first atomic-only experiment was deliberately incomplete and did not
include molecular bound-free opacity. It must not be mistaken for the
subsequent molecular calculation.

## Material validation

- Charge and nuclei conservation and both equilibrium equations checked
  across extreme algebraic regimes and the physical trace regime.
- Chemical-potential gauge invariance checked.
- Molecular opacity reproduces the old ideal-equilibrium coefficient when
  supplied exactly those ideal populations, and responds to the actual
  molecular donor density independently of the atomic product.
- The He2+ state sum uses odd rotational N for spin-zero 4He nuclei and
  includes 394 bound levels on the refined radial grid. Grid and box changes
  agree in K(T) to better than 1e-4. With the different Chang potential,
  K differs from the first five Stancil tabulated values by about 3.6–4.7%;
  the difference is retained, not fitted away.
- A global angular quadrature was inadequate at the potential's sharp
  short-range switch. Integration is now subdivided at the known sorting
  and switching locations; the potential itself is unchanged.
- The molecular-ion r^-4 and neutral r^-6 tails require explicit
  chemical-potential contributions beyond the finite HNC box. Those tails
  and the missing half-weight boundary are integrated analytically. At
  T=5000 K, rho=1.6, doubling the box with the same spacing changes molecular
  mu by about 3e-8 eV; halving spacing changes it by about 5.3e-5 eV.
- Actual REOS Cp/Q derivatives agree with independent finite differences
  at off-knot points to about 6e-10 in normalized units.
- Scoped monkeypatches restore the production EOS and opacity even after
  exceptions. Experimental checkpoints have incompatible field names and
  cannot silently load as production models.

Earlier complete portable research batch: **96 passed** (9.93 s), followed
by seven passing hierarchy/prolongation tests after adding the explicitly
incomplete-coarse-stage experiment.
The regular
production suite run earlier in this investigation: **416 passed, 5 skipped**
(85.09 s; slow/protected atmosphere canaries excluded from that command).
Protected production cold-start canaries were then rerun: **7 passed in
1634.47 s**, concurrently with dense-He experiments. This covers DB 10000
and 22000 K at production resolution, DB 22000 K at standard resolution,
and DA 3000, 4000, 5000 and 20000 K. This is not a controlled speed benchmark.

## Solver findings so far

1. Atomic dense-He trials did not converge. At 5000 K, a 159-layer run
   collapsed with all-depth flux error 0.492. C1 REOS interpolation alone
   reduced that only to 0.474. Matching the outer merit to the inner RMS
   objective avoided immediate collapse but still left error 0.446 after
   20 iterations.
2. A local Chebyshev material surrogate failed its independent accuracy
   check at REOS knots. `dense_helium_materials.py` now evaluates the exact
   fixed-pressure REOS polynomials and their analytic derivatives in the
   proposal. No tolerance was relaxed and accepted materials are unchanged.
3. With exact material evaluation, the first molecular 5000 K nodal run
   progressed to flux error 0.223 and spectral integral 0.8875, then stalled;
   its local energy error was still 0.567. **Not converged.**
4. Removing an unnecessary rotated-coordinate box admits valid smooth
   global temperature corrections (unit tested), but did not solve this
   atmosphere. Its 5000 K test stalled near flux error 0.501.
5. A finite-difference audit of the actual equations found global radiation
   response errors of only a few parts in 1e6. The initial minimum gradient
   excess needed to carry the stellar flux was about 1.7e-5. Finite
   perturbations can strongly change convection, so infinitesimal tangent
   accuracy alone is insufficient.
6. `inverse_ml2_proposal.py` eliminates ML2 compatibility in terms of surface
   temperature and signed element velocity. Exact material responses and
   the derivative of the eliminated temperature have been unit tested.
   Enforcing temperature constraints *inside* this proposal avoids destroying
   its predicted convective flux by scaling temperatures afterward. Both
   flux-minus-cell and surface-plus-cell energy formulations are under test;
   neither has yet delivered a certified atmosphere.
7. A fractional-velocity trust box cannot cross convective onset from a
   sufficiently stable cell: it approaches zero geometrically. An explicit
   proposal experiment admits the natural +/- stellar-flux velocity range
   while retaining the true temperature constraints. No evaluated physical
   flux is overwritten or capped.
8. Sampling the fresh continuous transport ODE on 317 layers produced a
   discrete initial residual of 187, versus about 0.68 on 159 layers.
   `discrete_dense_transport_seed.py` solves each discrete diffusion+ML2
   interface with the same material averaging and opacity wavelength grid.
   The initializer's transport defects are below 1e-6; the 317-layer initial
   formal residual falls to 0.535. This is a better *initializer*, not an
   atmosphere convergence result.
9. The augmented temperature/velocity proposal was checked by full finite
   differences on an actual dense atmosphere, including nonzero corrections.
   Its global relative tangent error was below 9e-12 (worst normalized row
   about 1.3e-8), but the initial matrix condition number was 5.5e9. Exact
   algebraic compatibility alone is not sufficient globalization: compatible
   Newton stalled after 39 attempts at flux error 0.42265, local error 0.55582
   and spectral integral 0.66169 in 347.2 s. An augmented proposal progressed
   further, below 0.08 flux error, but retained a >0.3 local defect. Neither
   is a converged result.
10. A later-state finite-difference audit found the worst local defect at the
    surface, not in the deepest efficient-convection layers. The trial model
    linearized radiative exchange but exponentiated the cooling rate used to
    normalize it. This can manufacture a root when the linear opacity factor
    crosses zero. `radiative_rate_proposal.py` instead represents positive
    heating and cooling separately, preserving their base values and exact
    first derivatives. Its stiff-common-opacity test explicitly demonstrates
    and eliminates that spurious root. This remains a proposal only; the
    accepted state still uses actual transfer, opacity and gradient-derived
    ML2 flux.
11. Combining positive-rate proposals with the existing *preliminary*
    convective-gradient relaxation produces a physical 5000 K solution on
    80 layers from scratch. The full physical phase takes 13 attempts after
    20 preliminary attempts: maximum flux defect 9.35e-9, local energy defect
    6.87e-6, measured final dlnT correction 2.44e-6, elapsed 138.1 s including
    ordinary spectrum synthesis. No asymptotic flux replacement is active.
    The earlier physical-only variants had not reached this root. This is
    evidence for the *combined* algorithm, not an isolated EOS speedup claim.
12. The ordinary spectrum on that state integrates to 0.984352 Fstar. Raising
    structural wavelength sampling from 600 to 2400 changes this to 0.984331,
    so that is not the explanation. `audit_dense_spectrum.py` reproduces the
    ordinary spectrum identically, then independently solves scattering:
    a fully coupled piecewise-linear Lambda solve gives 0.984374; coupled
    Feautrier with 3/4/8 angles gives 1.000320/1.000651/1.000803. Thus four
    scattering sweeps do not ensure source convergence, but **depth-grid
    transfer-discretization error**, not those sweeps, dominates the integral
    discrepancy here. No spectrum was renormalized. Feautrier independent
    scalar source checks are accurate to 4e-16 of the global radiation scale;
    unweighted relative errors in underflowing Wien tails are also retained.
13. `dense_mesh_hierarchy.py` adds an explicit fresh coarse-to-fine experiment.
    All levels use the same EOS, opacities and physical equations. PCHIP in
    log(T)/log(P) initializes the finer grid; every level must converge
    independently or the run errors without substituting its coarse result.
    A fresh 80 -> 159 -> 317 run has passed 80 layers, then 159 layers in four
    additional attempts (flux 3.91e-6, local 1.14e-6, dlnT 4.45e-7). The 317
    level passes after two more attempts (flux 1.39e-6, local 2.06e-6, measured
    dlnT 8.38e-7); total fresh runtime including all levels and ordinary
    spectrum is 416.1 s. Its unscaled ordinary spectrum integrates to
    0.999518 Fstar. A direct coupled piecewise-linear source solve independently
    gives 0.999539 (source equation error 4.45e-15); coupled Feautrier at 3/4/8
    angles gives 1.000278/1.000609/1.000760. Thus depth refinement resolves
    the original 1.6% transfer mismatch without flux normalization.
    The 8000 K dense-fluid tests remain unresolved at this writing. There is no
    temperature-specific branch or saved-model fallback.
14. The stalled 8000 K positive-rate test has flux defect 0.005538 and local
    defect 0.024156. Its radiation tangent agrees with finite differences
    at a few parts in 1e6. The worst cells lie at convective onset; there the
    infinitesimal convective tangent is extremely nonlinear even for 1e-6
    temperature perturbations. Current velocity-dependent compatibility
    scaling is now being tested in the *cheap augmented proposal*, not just
    in the unsuccessful full coupled Newton variant. The scaling becomes
    the stellar-flux superadiabatic excess on the convective branch, so a
    stable seed's large gradient gap cannot make a new convective trial's
    compatibility error appear small. All derivatives are included and
    checked independently (13 coupled-proposal/scaling tests pass).
    This change preserves the 5000 K solution (317 layers: flux 1.88e-4,
    local 1.60e-5, spectrum 0.999524, 530.1 s), but does not by itself resolve
    8000 K. Counterfactual tests on the stalled state show both augmented
    scalings making essentially no correction to the worst energy defect.
    Exact inverse Newton also fails to take a bounded step. The balanced
    Jacobian condition is 1.69e7 (raw 3.38e12); its unconstrained Newton step
    demands |dlnT|=37010 in transparent surface layers. The actual radiation
    tangent, not just a surrogate, was independently checked beforehand.
15. Finer mesh verification reveals an initialization cost: temperature-only
    interpolation can generate flux defects around 1200 in efficient
    convection. `convective_mesh_prolongation.py` interpolates the *evaluated*
    coarse ML2 flux and solves the new grid's exact finite-material gradient
    relation to initialize its temperatures. It preserves interpolated
    radiative slopes where the coarse convective flux is zero. It neither
    defines a final flux nor substitutes Fstar-Frad. The initializer must
    pass its own 1e-6 transport check; formal energy balance is solved anew.
16. A 41-layer 8000 K solve also stalls (local defect about 0.0135). An explicit
    `--allow-incomplete-coarse` experiment therefore bounds *intermediate*
    physical solves to 20 attempts, labels incomplete stages, and uses them
    only to initialize finer grids. This is a fresh same-physics hierarchy,
    not a recovered atmosphere. The requested final grid must pass the
    convergence flag AND independently checked flux, local-energy and step
    gates, or the run errors. The default still requires every level to pass.

The 80-layer solution has photospheric density about 1.04 g/cm3 rather than
the ideal-EOS hundreds of g/cm3. That is a substantial improvement in the
regime of the EOS, **not independent proof of accurate nonideal ionization**.
The specified-potential/closure differences and missing collective opacity
and refraction above still apply.

The unscaled 317-layer spectrum comparison is in
`results/dense-helium-spectrum-audit-20260906/comparison/`.
Relative to the exact-temperature, log(g)=8 pure-He Montréal reference, its
3500–9000 A band has 7.83% more flux and its 1–5 micron band has 18.87% less.
The broad SED is reasonably close, but these are significant physical
differences. Numerical convergence must not be described as complete
literature agreement. The reference integral is 0.998718 Fstar.

New experimental runs record local source SHA-256 hashes and all dense-solver
options before initialization, including uncommitted source files. They also
save the actual discrete initializer separately from the continuous ODE seed.
Six early atomic-only checkpoints were relabeled with incompatible experimental
array names; their original bytes remain recoverable beside them under the
`.experimental-unsafe-original.quarantined` extension. No user data was deleted.

## Artifacts and runtime

Artifacts live below the outer workspace's `results/`, not in the GitHub
package. Main table currently:

`results/dense-helium-hnc-20260905/molecular-hnc-electron-domain-table.npz`

It covers local T=1160.4518–17000 K and rho=1e-12–2 g/cm3 with explicit bounds,
81 temperature and 161 density nodes. The lower bound is the electron model's
published kT=0.1 eV limit, not a stellar temperature switch. The molecular
equilibrium is computed from bound levels, not extrapolated from Stancil's
equilibrium table. The low-temperature bound-state grid test includes 1001 K.
Older tables have distinct version
tags/paths and are not interchangeable. Every atmosphere experiment records
the table SHA-256. `tangent-5000.json`, `augmented-tangent.json` and
`tangent-late-5000.json` hold derivative audits. Independent off-node HNC
calculations at 1300, 1800, 3500 and 7500 K are in
`off-node-table-audit.json`. The largest tested log-equilibrium-constant error
is 0.00314 at 1300 K, rho=1.6; at the tested 3500/7500 K states it is below
8.7e-5. This interpolation error is not the source of the large atmosphere
residuals, although denser table sampling remains appropriate for precision
tests at the coldest, densest corner.

Use the project runtime, one BLAS thread per parallel test process:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KMP_USE_SHM=0 \
  PYTHONPATH=release/OpenWD/src:scripts \
  /Users/kareem/opt/miniconda3/bin/python scripts/dense_helium_molecular_experiment.py \
  --interaction-table results/dense-helium-hnc-20260905/molecular-hnc-electron-domain-table.npz \
  --least-squares-merit --exact-dense-materials --augmented-convection-proposal \
  --positive-radiative-rates --discrete-transport-seed \
  5000 --stable-transfer \
  --step-method nonlinear-convection-current-energy \
  --inner-max-evaluations 1000 --max-iterations 100 --no-continuations \
  --mesh optical --n-depth 80 --nonlinear-materials --inverse-ml2-step \
  --output-root results/cool-db-molecular-dense-hnc-preconditionedpositive80-20260906
```

Runs refuse to overwrite existing logs. The above command has already been
completed: inspect its outputs rather than rerunning into the same path.
For fresh mesh continuation add `--mesh-hierarchy`, request `--n-depth 317`,
and choose a new output directory. This is initialization by a same-physics
coarse solve, not a cold fine-grid Newton start; metadata records each level.
The present investigation is unfinished. No spectrum from it is certified
as a converged full-physics cool DB/DAB spectrum.
