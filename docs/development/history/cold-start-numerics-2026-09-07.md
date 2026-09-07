# Cold-start numerical repair — 7 September 2026

> Research record: this page describes work at the time it was written.
> For current usage and status, see the [user guide](../../getting-started.md)
> and [tested points](../../tested-temperature-ranges.md).

## What was wrong

The strict certificate exposed a real numerical weakness, not just a reporting
problem. In very optically thin cells, interface fluxes and their temperature
derivatives are nearly identical. Constant interface flux could therefore be
satisfied while a cell's own absorption/emission remained substantially out of
balance. Restarting a flux solve at that point could also return without ever
measuring another temperature correction. The 10000 K DB had an additional
unscreened lower boundary.

Examples before repair (same material physics and production presets):

| Case | All-depth flux error | Cell-local energy error | Additional failure |
| --- | ---: | ---: | --- |
| DA 3000 | 8.40e-4 | 0.738 | Unmeasured final correction |
| DA 4000 | 6.60e-6 | 1.846 | Unmeasured final correction |
| DA 5000 | 1.71e-6 | 0.313 | Unmeasured final correction |
| DA 20000 | 2.87e-6 | 0.803 | Unmeasured final correction |
| DB 10000 | 5.62e-8 | 0.158 | Absorption escape bound 0.0281 |
| DB 22000, standard 40 nodes | 7.55e-9 | 0.256 | — |

These local errors are relative to each cell's energy exchange, **not** fractions
of the stellar luminosity. The production 80-layer 22000 K DB already satisfied
the stricter criteria and needed no repair.

## Numerical implementation

1. Preserve the established cold initializer and complete its original formal
   flux solve. This is important: immediately replacing the initializer's
   objective with local heating destabilized some previously working models.
2. When local energy or a measured final correction is still missing, evaluate
   cell heating directly from the same Feautrier control volumes. Differentiate
   the full material, opacity, scattering and actual ML2 response. Do not obtain
   the tangent by subtracting nearly equal interface-flux Jacobian rows.
   A rejected flux-only direction hands off immediately if actual flux already
   passes but local energy does not. The rejected state is never accepted and
   the old phase returns `converged=False`. Already-successful original paths
   are unaffected; no additional tolerance or stellar-parameter threshold is used.
3. Use bounded linearly implicit thermal conditioning to damp troublesome
   temperature modes. The time step follows its nonlinear temporal defect,
   with a common 0.04 maximum change in log T. The phase uses unchanged physical
   equations and cannot certify equilibrium. It particularly prevents the
   alternating outer-temperature corrections that spuriously switched on
   convection in the 4000 K DA.
4. Finish with a static, unpenalized Newton solve in nodal log T. Row/column
   equilibration preserves weakly constrained physical modes without discarding
   small singular values. Positive energy weights are frozen within each
   linearization; the full physical response and final current-state energy
   checks are not frozen.
5. If a stationary helium-family solution fails **only** lower-boundary
   screening, append a bounded deeper pressure interval to its own mesh and
   re-solve with the same physics. Original mesh nodes are retained. No saved
   or neighboring stellar atmosphere is read. The bounded adaptation can still
   fail and then returns an uncertified exploratory result.

The static equations are, for cells i < N-1,

    R_i = (F_i/F_star - 1) - E_i/S_i,
    R_(N-1) = F_(N-1)/F_star - 1.

E is direct radiative cell exchange plus the difference of actual ML2 interface
flux; S is positive thermal emission plus the magnitudes of convective flux at
the cell faces. In exact arithmetic E_i = F_(i+1)-F_i, so this is an invertible
upper-triangular combination of the constant-flux equations. Direct evaluation
retains information lost by subtracting neighboring fluxes in finite precision.
No radiative/convective mask selects different final energy equations.

The integral/differential energy-balance motivation is consistent with
[Hubeny's CoolTLUSTY discussion, equations 31 and 33](https://academic.oup.com/mnras/article/469/1/841/3092374).
The particular frozen positive weights, thermal globalization and staged
integration here are OpenWD implementation choices, not a claim of reproducing
TLUSTY's complete algorithm.

The public hydrogen and helium-family factories enable local completion by
default. Already-qualified dense/molecular research workers explicitly retain
their own conservative energy equations and thermal/nonlinear-ML2 conditioning;
the new atomic completion is not applied on top of them. There is no new Teff
switch, removal of physics, manufactured convective flux or spectral rescaling.

## Completed cold-start checks

`research/validate_public_cold_start.py` calls the real public API without
supplying any atmosphere. It opens the old spectral controls only after the
new calculation finishes. Stored object abundances are request parameters,
never a temperature/pressure seed.

| Public case | All-depth flux error | Cell-local energy error | Raw max dlnT | Total steps including thermal conditioning |
| --- | ---: | ---: | ---: | ---: |
| DA 3000 | 3.82e-4 | 3.24e-4 | 8.34e-5 | 69 |
| DA 4000 | 4.28e-6 | 4.18e-6 | 3.64e-7 | 59 |
| DA 5000 | 3.53e-6 | 1.17e-5 | 2.51e-6 | 45 |
| DA 20000 | 1.43e-7 | 4.29e-4 | 4.67e-5 | 38 |
| DB 10000, 80 then 84 nodes | 5.64e-4 | 4.31e-5 | 1.33e-5 | 32 |
| DB 22000, production | 1.24e-10 | 1.05e-3 | 7.31e-5 | 31 |
| DB 22000, standard | 4.63e-10 | 1.69e-3 | 2.82e-4 | 41 |
| Atomic DAB 20000 | 8.53e-8 | 1.59e-3 | 1.82e-4 | 38 |
| PG 1225, production 80 nodes | 6.73e-10 | 1.86e-4 | 3.38e-5 | 24 |
| J0738, production 80 nodes | 9.76e-7 | 2.54e-6 | 5.85e-6 | 35 |

All five strict gates pass for these cases. DA tolerances are 0.002 in flux
and local energy, 0.0002 in correction; DB tolerances are 0.003 and 0.0003.
Source and lower-boundary checks are required separately. Completion work is
counted honestly; this is not a claim of the previous iteration budgets or a
speedup. Recorded parallel wall times of roughly 4–7 minutes per DA/warm DB
depend on simultaneous jobs and are not controlled benchmarks.

Fresh molecular DABs with log10(N_H/N_He)=-2 at 7500, 9000 and 10000 K now
pass with the final molecular and Lyman/Stark policy, using the same fresh
80-node recipe and 40 provisional thermal sweeps. Their subsequent static
solves and independent refined wavelength/16-angle audits pass. The historical
8000 K fresh demonstration is retained; the old 7750 K continuation is excluded
from the public cold-start table. See [tested temperatures](../../tested-temperature-ranges.md)
for the numerical values and physical limitations.

## Spectral protection and a disclosed change

The original synthetic regression fixtures are not regenerated. Fixed-state
synthesis retains its tight 0.1% significant-bin and 1e-5 band-change controls,
plus reproducibility against independently source-checked outputs. These cover
DA/DB and the paper DAB/DZ cases.

For newly relaxed cold atmospheres, compare peak-scaled and integrated absolute
spectral changes against the existing 0.3% stellar-flux accuracy, with a common
0.3% band limit for bands carrying at least 0.1% of stellar flux. These are
explicitly new cold-solve gates, not a claim that every spectral bin retains
the fixed-atmosphere tolerance. Far-UV relative differences are also reported.
These difference integrals cover the control interval (900–300000 Angstrom),
not an independent bolometric-closure test. In particular, hot models radiate
outside that interval. Atmosphere-grid flux conservation, source closure and
independent full-spectrum/depth resolution remain distinct checks.

The production 22000 K DB spectrum is array-identical to its protected checked
reference. Maximum significant-bin DA changes are about 0.0013% (3000),
0.000264% (4000), 0.000362% (5000), and 0.0589% (20000).

The 10000 K DB is not identical. Its required deeper boundary changes the
1150–3000 Angstrom band by +0.271%, optical by +0.0176%, and IR by +0.0092%.
The integrated absolute change is 0.119% of stellar flux; portions near the
faint far-UV tail change by about 10.3%. The new absorption escape bound is
8.27e-26 instead of 0.0281. Independent depth/quadrature convergence is still
distinct from this structure-grid certificate.

The new 80-node 7500 K DAB differs from the old 166-node calculation by 0.985%
in integrated absolute flux. That is a resolution comparison, not identical
reproduction or a universal depth-grid certificate. No reliable 5000 K DAB
or arbitrary abundance/gravity-grid claim is made.

## Reproducibility and remaining limits

The final core suite passes 545 tests on Python 3.9 / NumPy 1.26.4. On Python
3.11 / NumPy 2.3.5, all 779 core-plus-research tests pass. Each run skips five
optional external profile/reference-data tests. The 234 research components
also pass separately on Python 3.9. Local NumPy-2 evidence is not a claim of a
local Python-3.12 run; GitHub remains responsible for the configured 3.12 job.

All seven established DA/warm-DB cold-start canaries pass. Fresh automatic
dense DB 5000 and 8000 runs also pass their strict and independent audits;
both spectra are array-identical to the preceding successful dense calculations.
The fresh atomic 20000 K DAB also passes its paper-spectrum comparison gate;
its largest significant-bin change is 0.1284%, and its integrated absolute
change is 0.00138% of stellar flux. Its 38 reported iterations include 13
thermal sweeps and the final measured static correction.

### PG 1225: cold-start regression versus paper resolution

The fresh production calculation passes all five convergence gates but fails
the strict comparison against the saved paper spectrum: integrated absolute
change 0.531%, UV band -0.938%, optical +0.112%, IR +0.393%. The paper atmosphere
has 40 nodes; production generates 80. The mismatch was not waived or fixed by
changing this object's numerical settings.

An independent cold run of the original GitHub commit
`263b92e77bf207eb21ee8bd7b25f2b9312fb6828`, imported from its immutable temporary
archive, gives **array-identical T, P and column mass** to the repaired code at
80 nodes. Spectra at this same cold structure differ by at most 0.03644% in
significant bins and by 6.70e-8 of stellar flux in integrated absolute difference,
consistent with the separately tested scattering-source correction. Thus the
larger discrepancy from the 40-node paper control predates this repair; it is
not a cold-start solver regression. The new local-energy completion was not
needed for this case. No fresh paper-resolution or universal depth-convergence
claim follows from the production comparison.

### Known spectrum-consistency limits; unfinished changes excluded

This checkpoint retains the established piecewise-linear spectrum transfer,
with the separately verified exact scattering-source correction. It does not
promote the subsequent experiment that made high-level synthesis use the
atmosphere's Feautrier operator by default. No spectrum is renormalized.

A fixed-state audit of the newly cold-solved atmospheres on 50–1e9 Angstrom
found integrated spectrum/stellar-flux ratios of **0.98613 for DA 3000** and
**0.99314 for DB 10000** with the retained public spectrum method. These are
known numerical transfer-consistency errors, not failures hidden by the
atmosphere certificate: that certificate explicitly covers the declared
structure grid. The later matched-transfer experiment reduced those deficits,
but introduced spectral changes that have not completed regression and
depth-resolution review. It is excluded from this release, not automatically
used as a fallback. An atmosphere certificate alone must not be cited as a
bolometric or depth-grid certificate for the separately synthesized spectrum.

J0738 passes the repaired production atmosphere's five static gates, but fails
the comparison against the 40-node paper spectrum: integrated absolute change
0.868% of stellar flux, UV band -1.448%, and maximum significant-bin change
8.857%. Unlike PG 1225, its same-resolution original-GitHub cold comparison
was not completed. No claim that this difference predates the repair is made.
An exploratory matched-transfer audit also retained an approximately 2% flux
deficit for J0738; its structure/synthesis opacity and quadrature consistency
remain unresolved. The paper fixed-atmosphere spectral control still passes,
but J0738 is not a newly qualified cold-start paper-spectrum reproduction.

The unfinished matched-transfer edits and their extra tests were preserved
in a local recovery archive outside the public repository. The active source
and cold-start canaries use the tested pre-experiment spectrum path; their
spectral controls were not relaxed to accommodate that experiment.

Public examples, supported-point documentation and automatic CI use cold
starts. `run_model` rejects supplied checkpoints. Historical archives and
low-level restart functions remain diagnostic tools, never prerequisites for
generation. Failed outputs remain available with warnings; strict workflows
raise after retaining those artifacts.

Local run records are under `results/cold-numerics-20260907/`, with live
iteration JSON, full metadata and spectra. They are validation artifacts,
not inputs to the public solver. GitHub CI cannot certify these edits until
they are committed and pushed. The cool experimental workers still isolate
their callback overrides in child processes; converting those internals into
fully explicit per-model library objects remains an architectural follow-up.
