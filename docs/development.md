# Development and regression policy

The standalone OpenWD repository is the source of truth for the released
Python package. Research results, private observations, and large validation
products may live in an adjacent workspace, but must import this checkout
rather than maintain a copied `wd_spectra` source tree. Pytest explicitly puts
this repository's `src` directory first so an unrelated editable installation
cannot make the suite certify the wrong code.

Local solver or physics work stays unpushed until its checks pass. The
maintained GitHub branch is `main`; additional published development branches
are not required. A change is eligible to be pushed only after:

1. the ordinary test suite passes;
2. the protected no-fallback atmosphere canaries pass;
3. any affected paper-spectrum regressions have been rerun in the validation
   workspace; and
4. the model-request physics revision is updated if equations or physical data
   changed.

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
