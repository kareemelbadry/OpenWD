# Reliability and automatic physics selection

This work tightens certification, isolates automatic physics selection, and
repairs the previously exposed local-energy and stationarity failures with
cold-start numerical completion. It preserves material physics and the original
initial solve, adding completion only when independent evidence requires it.
See [the numerical repair and results](cold-start-numerics-2026-09-07.md).

## Solver completion versus equilibrium

`radiative_equilibrium_solver_converged` records the previous solver/flux
completion criterion. `radiative_equilibrium_converged` now requires a
versioned `equilibrium_certificate` with finite passing evidence for:

1. actual all-depth radiative + ML2 flux conservation;
2. cell-local energy balance;
3. a measured, unrestricted temperature correction;
4. scattering-source closure; and
5. screening of the bottom boundary by absorption.

The configured flux tolerance applies to flux, local energy and the escape
bound; the configured temperature tolerance applies to the raw correction.
The atmosphere source residual tolerance is 1e-6. Missing evidence fails
closed. Initial-state completion without a proposal records `None`, not a
fabricated zero. A trust-clipped accepted step cannot certify stationarity.
Copied seed metadata cannot overwrite newly measured diagnostics.

Local energy is normalized by the cell's radiative/convective exchange scale;
it is **not** a fraction of the whole star's bolometric flux. A local residual
of 0.24 does not mean that 24% of the stellar luminosity is missing. The
absorption escape bound is sufficient, not necessary: a failed bound means
boundary independence is unverified, not proof that the spectrum is wrong.
Independent wavelength/angle/depth convergence and full-physics validation
remain separate, explicitly recorded questions.

Previously successful warm/DA solver controls included cases without all this
evidence. They now undergo local-energy completion and, where required,
same-calculation boundary expansion. Old checkpoints without
the new certificate are `unknown` unless a known mismatch/failure makes them
`unconverged`. Exploratory synthesis remains available.

## Synthesis source and checkpoint identity

The default formal spectrum retains its established piecewise-linear
discretization and surface/bottom boundary treatment. Instead of four fixed
scattering sweeps (or a Ca-II-specific iteration budget), it solves the exact
linear Lambda source system, in wavelength chunks. A separate prescribed-
source transfer calculation must close the source law to 1e-10 on the
lambda-weighted radiation scale. There is no flux normalization. Explicit
optical-depth and column-mass Feautrier spectra use their own direct systems
and independently checked closure. Atmosphere Feautrier equations are unchanged.
The old spectrum keyword `optical-depth` remains an alias for the established
piecewise-linear formal discretization; the optional new formal variant is
named `feautrier-optical-depth` to avoid changing an existing explicit request.

Ten fixed-state controls cover DA 3000/4000/5000/20000, DB 10000/22000,
paper DAB 9000/20000, PG 1225 and J0738. Relative to the original GitHub
commit, the largest significant change is 0.0367% (PG 1225); DB 10000 changes
by at most 0.0311%. The paper 9000-K DAB UV band changes by 2.91e-13 fraction,
and 20000-K DAB agrees to roundoff. These are synthetic regression checks,
not new comparisons to external observations or atmosphere-convergence claims.

Fixed synthesis checks both the fingerprint and actual Teff/logg. A mismatched
checkpoint can still produce an exploratory spectrum, but cannot certify the
new request. Restoring a checkpoint re-closes its EOS: changed or unrecorded
Teff/logg/chemistry, or changed density/electron populations invalidate its
certificate. Caller-owned atmosphere metadata is not mutated. The request
physics revision has been advanced for these contracts.

## One automatic entry point

```python
from wd_spectra import DBConfig, DABConfig, select_physics, run_model

config = DBConfig(effective_temperature=5000, logg=8, quality="production")
print(select_physics(config))  # optional diagnostic; not needed before run_model
run = run_model(config, "results/db-example", require_convergence=True)
spectrum = run.spectrum

# A fresh molecular DAB; no previous model is an input.
run = run_model(DABConfig(effective_temperature=7500, quality="production"),
    "results/dab-example", research_data="/path/to/pinned-molecular-tables",
    require_convergence=True)
```

The selection is frozen before solving. It is based on provisional local
material diagnostics, not a list of successful stars or a global Teff cutoff:

| Composition | Screening and selection |
| --- | --- |
| DA | Preserve the established hydrogen EOS/opacity policy. |
| Pure-He DB | At the seed photosphere (Rosseland tau=2/3), require an in-domain REOS density correction of at least 5% and electron/nucleus fraction below 1e-3 to select the dense-neutral workflow. Otherwise select the established DB prescription. |
| DAB/DBA | Over 0.01 <= seed Rosseland tau <= 3, select molecular transport if the H2-bound fraction of hydrogen nuclei reaches 1e-4 or electron-density change reaches 1e-3; an explicit molecular request also selects it. Otherwise use atomic DAB. |
| DZ | Preserve the trace-metal mixture; never insert a pure-He dense EOS. |

`PhysicsSelectionPolicy` records the three relevance thresholds. They are
screening accuracy scales, **not calibrated universal validity boundaries**.
The seed is approximate; screening does not establish a self-consistent
solution. Dense runs retain local table/trace-ion domain guards. A rejected
domain or failed solve raises/records the failure; it never retries with a
different EOS. The selected recipe remains experimental where the underlying
physics is experimental. The `tested_point` flag identifies documented cold-start
Teff/logg/composition points only, not validation of a new result or arbitrary
custom numerical settings. Missing dense-mixture physics is not supplied by
this dispatcher. The DA seed's existing cool initialization policy is unchanged.

Cool recipes currently require a source checkout, research dependencies,
production quality, log g=8 and integer-K Teff. Unsupported solver, broadening
or mixing-length overrides are rejected, not ignored or replaced. The pure-He
wrapper and public mixed-model runs start fresh. `run_model` rejects checkpoint
inputs. The external
molecular tables are checked against fixed checksums. Missing data cause an
actionable error, not an atomic-model substitution.

`run_model` returns artifact paths and a `Spectrum`, not an atmosphere silently
reconstructed with the wrong EOS. `model-run.json` records the request,
selection metrics, commands, data locations, cold-start provenance,
result status and qualification. Full worker
artifacts retain their physical assumptions, source snapshots and audits.
Experimental callbacks execute in a child process, so they cannot alter the
parent interpreter or another model's callbacks. Public preset fingerprints
do not certify these experimental scopes; their independent workflow checkers
are the authority for the returned `convergence_verified` field.

A neutral-helium screen outside the REOS table, or a nonfinite screening
diagnostic, raises rather than silently choosing ideal/atomic physics.

Unqualified completed models remain available with a warning by default.
`require_convergence=True` raises after retaining the exploratory artifacts.
Hard execution/data/domain failures always raise. Existing output directories
are never overwritten. No failed model triggers another physics choice.

## Regression gates

Normal Python 3.9/3.12 CI includes ten broad-spectrum controls and tests for
strong-scattering closure, corrupt diagnostics, checkpoint mismatch, and
failure-without-retry. On source/test changes, separate parallel jobs run all
seven established cold-start controls, an atomic DAB 20000 cold/spectral check,
fresh automatic DB 5000/8000 with
independent wavelength/angular audits, and fresh DAB 7500/9000/10000 calculations
with a newly measured correction. Missing molecular data fail the job instead
of silently skipping it. Reports are retained as CI artifacts.

The broad-spectrum fixtures contain only our synthetic outputs. Atmosphere
tests retain their physical flux/step tolerances and compare spectra, P and
column mass. Their work budgets account for the extra completion. Outer T may
change to satisfy local energy; fixed-atmosphere spectral tests keep their
separate tighter bounds and immutable references. All cold canaries require
strict qualification, not just a plausible
plot or solver terminal flag. GitHub execution is pending until these local
changes are committed and pushed.

## Local verification

Current cold-start results and the numerical implementation are recorded in
[the numerical report](cold-start-numerics-2026-09-07.md). The supported-point
table is [cold-start based](tested-temperature-ranges.md); the previous
checkpoint-dependent 7500/9000/10000 DAB demonstrations have been replaced by
fresh calculations with final molecular/line physics.

The preceding certificate/source-isolation tranche passed 524 Python-3.9 core
tests, seven historical solver controls and 234 research components, with
supplemental NumPy-2 testing. Those counts describe that earlier stage, not
the final numerical revision. Its finding that six historical controls lacked
strict evidence is the problem addressed by the current completion phase.

Real material screens preserve established DB 10000/22000 and atomic DAB
20000 physics, select molecular DAB at 7500/9000/10000, and select dense DB
at 5000/8000. Routing tests include Teff-independent decisions and rejection
of missing/out-of-domain diagnostics without another physics choice.

This is a numerical reliability repair, not a demonstrated speedup. Parallel
execution timings are not controlled performance benchmarks. No universal
depth-grid or full-physics certification is claimed.
