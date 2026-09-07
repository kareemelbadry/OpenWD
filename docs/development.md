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

Exact checkpoint continuation requires the `model_request_fingerprint` stored
in atmosphere metadata. A missing or mismatched fingerprint is not an error:
the atmosphere remains a useful warm start, but receives normal conditioning
instead of entering the final formal-flux phase directly.

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
