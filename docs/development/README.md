# Development and regression policy

[Documentation home](../README.md)

Use the [user guide](../getting-started.md) for model generation. This section
is for changing the implementation and checking it safely; detailed previous
investigations live in the [research history](history/README.md).

## A short development loop, then full qualification

From the repository root:

```bash
python -m pip install -e '.[test]'
python tools/validate.py fast
python tools/validate.py spectra --jobs 2
python tools/validate.py cold --case da-4000 --case db-22000 --jobs 2
```

Use focused `python -m pytest tests/test_...py` checks after individual edits.
The ordinary suite includes `tests/test_da_public_spectrum.py`: frozen DA
continuum and Balmer-profile comparisons through the **public default**, with
atmosphere iterations forbidden. Do not add a method override to make this
guard pass. These quick checks complement, not replace, fresh-atmosphere tests.
The approved cubic DA controls also check the sampled stellar-flux integral.
Intentional reviewed physics changes use separate, frozen approved outputs
with a provenance manifest; the historical inputs remain untouched. Never
regenerate references inside a test or relax tolerances to hide a change.
The `fast` tier runs numerical/API/documentation and cool-component tests,
excluding both expensive markers. `spectra` synthesizes the immutable fixed
atmospheres; it **does not certify equilibrium**. Select a subset with repeated
`--case` options, for example `spectra --case da-4000 --case daz-g149-28`.
Use targeted `cold` checks once an implementation looks promising, not after
every small edit. Existing `python -m pytest` still includes spectral controls
and excludes cold canaries; its coverage has not silently been reduced.

For a finished solver/physics candidate, run:

```bash
python tools/validate.py full --jobs 2 --output results/validation-candidate
```

This runs fast checks, fixed-atmosphere spectra, then every protected cold
case, stopping before an expensive stage if an earlier stage failed. It
includes the established controls, automatic dense/molecular workflows, and
DAZ cold starts. It does not change tolerances, reduce science resolutions,
renormalize spectra, or supply a previous atmosphere. See the
[cool-model guide](../../research/cool_models/README.md) for required external
data. Missing data are failures, not successful physical tests. Additional
affected comparisons in the paper-validation workspace remain necessary.

`--plan` prints the exact commands without running anything. `--jobs` bounds
simultaneous processes (default two); numerical-library threads are fixed to
one per process. Use fewer jobs on memory-limited machines. Progress streams
live with case labels and elapsed-time heartbeats. Each case gets its own log,
and `summary.json` records outcomes, elapsed time, commands, source/data hashes,
Python, packages and compiled binaries. Protected DA/DB tests also save the
last accepted iteration as a **diagnostic-only** artifact and the completed
model before physical assertions. Interrupted or timed-out work never passes.

To avoid repeating completed expensive tests after prose-only edits, supply
`--reuse results/previous-run/summary.json`. Reuse is explicit, applies only
to completed passed spectral/cold tests with retained logs and an identical
numerical-input/runtime hash, and is identified in the new report. Fast tests
always rerun. Changed equations, data, compiled binaries, checker inputs or
dependencies invalidate reuse conservatively; custom `OPENWD_DATA` roots
disable it. Inputs changed during a run also invalidate its qualification.
This reuses **test evidence**, never an atmosphere or solver starting state.

Only a successful `full` report is a full-suite qualification; a green subset
or fixed-atmosphere comparison is not. See [telemetry](solver-telemetry.md)
and [performance](performance.md) for interpreting diagnostics and threading.

## GitHub checks

Fast/component and fixed-spectrum checks run on pull requests and pushes to
`main`; there is no duplicate feature-branch push run. New PR commits cancel
obsolete checks for that PR without cancelling unrelated main-branch work.

The expensive qualification workflow is explicit: add a **full-validation**
label to a non-draft PR when the candidate is ready. Remove the label or return
the PR to draft while iterating. A new commit on a labelled PR is a new candidate
and must be requalified. Without the label, numerical changes leave the cheap
`qualification` check failing with an explanation; skipped atmosphere jobs
are never called successful qualification. Prose and ordinary unit-test edits
do not require cold starts. Unknown paths and shared numerical/data/checker
changes conservatively require the full suite.

Maintainers can also run **protected model canaries → Run workflow** for the
selected branch; the weekly scheduled full run remains enabled. There is no
automatic expensive merge-push rerun. The cold matrix runs one star per job,
after cheap preflight checks, and retains logs/results even on failure.

For GitHub-enforced merging, select the stable **qualification** check in
branch protection along with ordinary checks, rather than individual matrix
job names. Workflow files cannot configure repository branch protection;
without that setting the check reports failures but does not prevent a
maintainer overriding them. Direct local development on `main` remains
supported: run full qualification before pushing numerical changes.

## Protect existing models

The standalone OpenWD repository is the source of truth for the released
Python package. Research results, private observations, and large validation
products may live in an adjacent workspace, but must import this checkout
rather than maintain a copied `wd_spectra` source tree. Pytest explicitly puts
this repository's `src` directory first so an unrelated editable installation
cannot make the suite certify the wrong code.

Local solver or physics work stays unpushed until its final-candidate checks pass. The
maintained GitHub branch is `main`; additional published development branches
are not required. A change is eligible to be pushed only after:

1. the fast and fixed-spectrum suites pass;
2. full protected no-fallback cold-start qualification passes;
3. any affected paper-spectrum regressions have been rerun in the validation
   workspace; and
4. the model-request physics revision is updated if equations or physical data
   changed.

These are final-candidate requirements, not an instruction to rerun everything
after every edit. Documentation/interface-only changes need the relevant
tests; the full atmosphere suite is reserved for changes that affect numerical
inputs or qualification. A new model needs its own cold-start and spectral
controls as well as protection of the existing models.

Public one-shot functions deliberately return exploratory spectra from
unconverged or provenance-unknown atmospheres, but issue
`AtmosphereConvergenceWarning` and report `atmosphere_convergence_status` in
the result metadata. Tests and production workflows should turn that warning
into an error and assert the atmosphere's all-depth convergence metrics.
Cold-start canaries must pass all five certificate gates and spectral comparisons
against immutable controls. They must not be replaced with checkpoint starts.
Outer temperatures previously unconstrained by interface flux may change only
with explained local-energy corrections and independently checked spectra.
Runtime budgets account for the added physical completion, not relaxed physical
tolerances. Fixed-atmosphere synthesis has separate, tighter regression bounds.
Iteration ceilings are coarse work guards, not equilibrium criteria. The two
100-layer ultracool DA canaries share a 120-iteration ceiling; a verified
4000 K calculation took 61 iterations on one CI run and 59 on another, making
the former 60-iteration test ceiling too tight. All physical and spectral
checks remain mandatory and unchanged.

Public generation and examples are cold-start workflows. `run_model` rejects
checkpoint inputs. Low-level checkpoint/fixed-synthesis utilities are retained
for explicit diagnostics, not as prerequisites or recovery paths for users.
Refining or extending the mesh inside a fresh calculation is allowed, provided
the same physics is retained and the final state is independently certified.

Adaptive solvers must preserve the structured fields documented in
`solver-telemetry.md`. New numerical changes should be justified with those
diagnostics, not by increasing iteration limits or accepting a surface-flux
ratio in place of all-depth convergence.

The explicit cool-model workflow source and component tests live in
`research/cool_models`, importing this same `src/wd_spectra` tree. Local
compatibility symlinks may preserve historical outer-workspace script names;
do not keep separate editable implementations. Document tested compositions,
temperatures, initialization and physical limitations before claiming a new
range. Research comparison flags are not regression-qualified defaults.
