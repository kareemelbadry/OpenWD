# Cross-model checks with the experimental changes enabled

> Research record: this page describes work at the time it was written.
> For current usage and status, see the [user guide](../../getting-started.md)
> and [tested points](../../tested-temperature-ranges.md).

This is a regression investigation, not a production/default change. The
production worktree's earlier edits were preserved. No solver tolerance,
iteration-limit assertion, reference spectrum, or production Python/C source
was changed, and no commit or push was made.

**Outcome:** the existing opacity-only regression suite passes. The full
dense-He pipeline is not regression-qualified as a general replacement.
The completed transfer-component checks expose iteration regressions and
changed DZ flux balance; the full 10000 K experiment fails independent
local-energy qualification and the full 22000 K request is out of domain.
Two supplemental stricter DAB checks were stopped as incomplete; all other
launched runs finished. No runs remain active.

Artifacts are under `results/experimental-crossmodel-20260906` in the outer
research workspace. `review-summary.json` distinguishes physical checks from
process exit codes. In particular, the old DZ diagnostic prints a failed
flux test but returns zero; zero must not be reported as physical success.

## Three distinct things were tested

1. **Opacity join alone:** apply `heminus_join_scope` while running the actual
   existing tests. Each model retains its normal EOS, atmospheric solver and
   synthesis. Execution counters verify whether the affected opacity domain
   is actually encountered. Pure DA models do not use He-minus absorption.
2. **Conservative mass-transfer component plus the join:** explicitly install
   `mass_transfer_experiment` in existing DA/DB/DAB/DZ runs. This tests the
   proposed transfer component as a direct change to existing model equations.
   It is **not** the entire cool-He pipeline: it retains each original EOS,
   Newton proposal and public synthesis. In particular, public DAB/DZ
   synthesis has not been replaced by matched conservative mass-transfer
   synthesis. A component run must not be advertised as full integration.
3. **Full dense-He experiment:** the new molecular HNC/REOS material model,
   exact material tangent, nonlinear-ML2 pseudo-time initializer, static
   correction, conservative transfer, extended thermal interval and smooth
   opacity join, using the same cold-start settings as the earlier 8000 K
   demonstration. No saved atmosphere or substitute EOS is used.

## Opacity-only results

All **421 regular production tests pass** with the repair enabled, and all
**seven protected cold-start atmosphere tests pass**:

| Model | Layers | Final iteration count | Result |
| --- | ---: | ---: | --- |
| DA 3000 K | 100 | 54 | pass |
| DA 4000 K | 100 | 35 | pass |
| DA 5000 K | 100 | 35 | pass |
| DA 20000 K | 100 | 22 | pass |
| DB 10000 K | 80 | 18 | pass |
| DB 22000 K | 80 | 31 | pass |
| DB 22000 K | 40 | 28 | pass |

The five optional-data tests that were skipped when launched from the release
directory are available in the outer research environment: the total is
416 + 5, not a claim of five new tests.

The existing warm DAB checks at 9000 and 20000 K run successfully. The
20000 K atmosphere re-relaxes in three iterations; its layers do not enter
the changed He-minus temperature domain. At 9000 K, fixed-paper-atmosphere
UV flux is unchanged to roundoff, optical integrated flux changes by about
-0.098%, and the maximum pointwise change is 0.698%. Its saved atmosphere
passes the default initial-state flux gate at 0.000471604; that gate does
not supply a newly measured temperature correction. A stricter paired
baseline/join check was therefore added below.

The PG 1225 and J0738 **fixed-atmosphere** paper spectra reproduce to maximum
relative differences 1.84e-11 and 4.25e-14. Independently recomputed
all-depth flux errors on those saved grids are 0.00160038 and 0.00223328,
respectively, both below the existing 0.003 gate. This is not a claim of
new cold-start DZ convergence or a measured temperature correction.

## Transfer-component results

The 80-layer DB 10000 K canary passes, with **27 iterations** (opacity-only:
18), final flux error 2.03e-9 and maximum measured dlnT 7.43e-5. This
retains the existing helium EOS; it is not the full dense-He run below.

The 40-layer DB 22000 K converges physically, but takes **31 iterations**,
failing its existing **30-iteration** limit. Final flux error is 3.52e-11
and maximum measured dlnT 5.04e-5. The opacity-only test passes, and no layer
in this model enters the changed opacity domain, isolating this difference
to the transfer component.

DA 20000 K also converges physically but takes **57 iterations**, failing
its **45-iteration** regression limit (opacity-only: 22). Hydrogen does
not use the repaired helium opacity; the new mass transfer was actually
called 132 times. This is a genuine iteration-efficiency regression, not
an opacity-domain effect.

The 9000/20000 K DAB paper states re-relax in nine/three iterations with
the new transfer component. Relative to the opacity-only checks, synthesized
UV-band flux changes by +0.666%/+0.752%, and optical by +0.384%/+0.439%.
These are diagnostics, **not matched-transfer spectral validation**:
the public formal synthesis still uses its original transfer law.

The saved DZ atmospheres no longer pass the current-equation flux test:

| Saved paper state | Opacity join only | New mass transfer + join | Limit |
| --- | ---: | ---: | ---: |
| PG 1225 | 0.1600% | 0.4251% | 0.3% |
| J0738 | 0.2233% | 1.1647% | 0.3% |

Changing the discretized equations changes the fixed point. This demonstrates
that the saved structures cannot simply be reused as converged; it does
**not** demonstrate that the new equations cannot converge after relaxation.
Their unchanged fixed-atmosphere spectra alone do not validate the new
atmospheric balance or the missing matched synthesis integration.

## Full dense-He results

### 10000 K: internal convergence, independent qualification fails

The fresh 80-layer run takes seven pseudo-time initialization sweeps and
five static correction attempts. Final structure-grid all-depth flux error
is 1.26322e-6, local error 9.83404e-7 and maximum measured dlnT 1.19534e-6.
The lower-boundary absorption escape bound is 1.08e-25. This is a genuine
fresh initialization with the declared dense physics, not a recovered
production atmosphere.

It nevertheless **fails the independent local-energy check**:

| Independent wavelengths | Angles | Spectrum integral / expected | Maximum local-energy error |
| ---: | ---: | ---: | ---: |
| 8000 | 8 | 0.999992359 | 0.00765050 |
| 8000 | 16 | 1.000000283 | 0.00771219 |
| 32000 | 8 | 0.999991626 | 0.01140863 |
| 32000 | 16 | 0.999999550 | 0.01133119 |

The unchanged local gate is 0.003. The worst layer is the surface, and its
signed defect changes with wavelength sampling. This is evidence of
unresolved wavelength-integration accuracy; it is not yet a precise
grid-independent estimate of the atmosphere's local defect. Both independent
qualifications remain false. Excellent bolometric flux and structure-grid
convergence are insufficient, and this model is not certified as successful.

### 22000 K and other compositions: unsupported by the full material model

The full 22000 K attempt fails during fresh initialization:
`DenseHeliumDomainError`, surface T=18499.7 K outside the HNC table's
17000 K upper bound. It does not substitute the production EOS.

An explicit DAB request with `--log-h-he -2` is rejected by the full dense
driver: it requires pure helium. No consistent dense mixed H/He or
metal-polluted closure has been implemented there. The component tests
above must not be mistaken for full new-EOS DA/DAB/DZ tests.

## Stricter saved-state DAB check

The additional 9000 K pair explicitly sets `allow_initial_convergence=False`
in the test harness for both the original opacity and the repaired opacity.
This retains the numerical tolerances but requires a newly computed small
temperature correction instead of accepting the saved state immediately.
Both were slow and remained incomplete after more than 22 CPU-minutes
each. The final reported baseline/join corrections were 0.000441/0.00125,
still above the 0.0003 gate despite flux errors near 4.6--4.7e-5. Both
had entered the existing solver's automatic continuation loop. These two
supplemental processes alone were explicitly stopped with SIGINT; their
scope-restoration checks passed and their interrupted status is retained in
`additional-run-outcomes.json`. They are **neither passes nor proof of
eventual failure**. The baseline was also affected, so the difficulty cannot
be assigned to the opacity repair alone. No matched final-spectrum comparison
is claimed for this pair.

This exposes a remaining validation weakness in the ordinary saved-state
DAB check: its immediate acceptance demonstrates a sufficiently small flux
residual, but not a newly measured small correction. The existing test's
pass is reported accurately above and is not inflated into that stronger
claim. Seventeen targeted scope, opacity-join, audit and synthesis tests
were also rerun and passed.

## Reproduction

The new drivers are `scripts/check_experimental_crossmodel.py`,
`scripts/run_experimental_crossmodel_matrix.py`, and
`scripts/summarize_experimental_crossmodel.py`. The matrix launches separate
single-threaded processes with live per-case log files; it never uses
concurrent global monkeypatch scopes in one process. `-m canary` is supplied
explicitly because the release pytest configuration deselects these tests
by default. An initial deselected probe was retained and is not counted as
a model test.

No controlled speed benchmark is claimed. Runs were concurrent, and wall-clock
time changed substantially during this session; the worker reports also
record separate monotonic elapsed times.
