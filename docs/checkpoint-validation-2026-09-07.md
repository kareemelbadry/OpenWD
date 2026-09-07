# Cool-model checkpoint validation — 2026-09-07

This checkpoint collects the successful work since `dcd3037`. It does not
claim that the dense-He or molecular research equations are complete physics,
nor that their availability makes them safe replacements for every warm model.
The [tested-temperature table](tested-temperature-ranges.md) is the concise
statement of the qualified points and limitations.

## Included changes

- Cool-DA initialization recovery and declining CIA tails, followed by actual
  all-depth flux verification; no manufactured final convective flux.
- Direct coherent-scattering structure solve and consistent tangent, complete
  ML2 material response, fixed Rosseland quadrature, cancellation-resistant
  optional transfer and physical-domain-aware derivative probes.
- Explicit conservative column-mass structure/synthesis, local-energy checks,
  nonlinear convection proposals, dense-He REOS/HNC/He2+ chemistry and smooth
  He-minus opacity join in the separately selected cool-DB workflow.
- Explicit molecular H/He closure, state-consistent H2 partition/energy,
  H2-He/H2-H2 CIA, physical-detuning Lyman interpolation and consistent
  low-density Doppler-profile edge, pseudo-time initialization, nodal SVD
  trust bounds and strict measured-correction checks in the cool-DAB recipe.
- Source/provenance records, live progress, independent audits and versioned
  reproduction drivers. The original outer-workspace driver paths are now
  compatibility links, not a divergent source tree.

The research toolkit retains explicitly selected diagnostic alternatives to
make the investigation inspectable. Failed 5000 K DAB proposals are not
enabled by the qualified recipe or promoted to public defaults. No atmosphere
convergence tolerances, protected iteration ceilings or reference spectra
were relaxed to obtain this checkpoint.

## Tests repeated for the checkpoint

- **694 passed, 5 optional-data skips**, on Python 3.9.16 / NumPy 1.26.4 /
  SciPy 1.11.1, including the relocated research component tests.
- **694 passed, 5 optional-data skips**, on Python 3.11.9 / NumPy 2.3.5 /
  SciPy 1.15.3, using an isolated temporary environment and its freshly built
  C extension. These checks are not a claim of local Python 3.12 testing;
  GitHub's matrix covers 3.9 and 3.12.
- All **seven slow protected cold-start canaries** passed separately on the
  Python 3.9 runtime: DA 3000/4000/5000/20000 K, DB 10000/22000 K production,
  and DB 22000 K standard. The regular-suite counts above exclude them.
- The opt-in component suite also passed without external molecular data
  (225 passes, 8 explicit skips at the packaging check, before the additional
  independent-scattering test was included). CI does not download the large
  optional molecular tables or imply full cool-atmosphere validation.
- A clean snapshot of the Git index builds a wheel successfully, and its
  research entry points import without the outer workspace. The source-root
  guard verifies that they use the same checkout's `wd_spectra` implementation.

Cross-version checking caught batched vector-RHS incompatibilities in mixed
chemistry and the independent linear-transfer audit, plus SciPy's renamed
GMRES tolerance argument. Explicit RHS axes and capability-based argument
selection preserve the equations and tolerances. Mixed populations on an
18-state NumPy-1 comparison are bit-for-bit unchanged. The scattering-tangent
test now uses fourth-order finite differences at two wider steps to reduce
thin-cell subtraction noise; its acceptance tolerances are unchanged.

## End-to-end and paper checks

- The packaged **5000 K DB cold start** reproduces the earlier result: 56
  pseudo-time sweeps followed by 17 static attempts; flux error 0.000126479,
  local energy 1.23591e-5 and measured dlnT 2.57831e-6. Its separate
  8000-wavelength, 8/16-angle, longer-domain audit passes. The largest
  independently sampled local-energy error is 0.00125589.
- The packaged **8000 K DB cold start** also reproduces the earlier result:
  30 pseudo-time sweeps followed by eight static attempts; flux 6.76210e-6,
  local energy 1.04129e-5 and measured dlnT 5.46098e-6. The separate
  8000-wavelength, 8/16-angle audit passes with maximum local error
  0.000954710. No saved atmosphere was supplied in either DB replay.
- The packaged **7500 K DAB supplied-state re-solve** passes the strict
  checker with a newly measured correction 4.02467e-7, flux error 7.61766e-8
  and local error 3.36894e-8. Its independent finer wavelength/16-angle audit
  passes with flux 7.65399e-5, local 7.56070e-5 and bottom escape 0.000163048.
  Neither check is presented as a new cold start.
  The same explicit re-solve also passes from a clean Git-index snapshot
  using Python 3.11 / NumPy 2 and a freshly compiled C extension: correction
  4.02467e-7, flux 7.61766e-8, local 3.36894e-8. Its matched spectrum integral
  is 0.999691443626 times the stellar flux, agreeing with the NumPy-1 replay.
- **Paper DAB 20000 K:** fixed-atmosphere spectrum agrees to maximum relative
  difference 8.66e-12. Re-relaxation takes three attempts, with flux 4.69802e-6
  and measured dlnT 0.000151396. Relative to the original paper, its re-relaxed
  integrated UV/optical flux changes by +0.00611%/-0.02320%. Pointwise relative
  differences in weak/deep features are larger (maximum 11.1% over the
  comparison mask); broad-band agreement is not a pointwise-identity claim.
- **Paper atomic DAB 9000 K:** fixed spectrum agrees to 3.06e-9 maximum
  relative difference and its UV integral to roundoff. The saved atmosphere
  passes the initial flux-only gate at 8.86835e-5, with zero new iterations.
  This is explicitly **not** a newly measured stationary-correction pass;
  the stronger check remains unresolved as documented in the prior report.
- **DZ PG 1225 / J0738:** fixed spectra agree to maximum relative differences
  1.84e-11 / 4.25e-14. Current-equation all-depth flux errors are 0.00167938 /
  0.00223328, below 0.003. These are fixed-paper-state checks, not new
  atmosphere relaxations. Legacy provenance warnings were not suppressed.

Full raw logs and XML reports are retained locally under
`release/OpenWD/results/github-checkpoint-20260907` in the original research
workspace; they are not runtime inputs. The small continuation checkpoints
and generated HNC table shipped here have separately documented checksums.
No controlled speed benchmark is claimed from these concurrent runs.

## CI portability follow-up

The initial push passed Python 3.12 and both research-component jobs. The
Python 3.9 / NumPy 2.0.2 / Linux job exposed one additional cancellation-limited
finite-difference comparison in the stable-transfer tangent test (difference
2.38e-7 against an approximately 2.0e-7 allowance). Its reference now uses
fourth-order differences at two wider step sizes, retaining all physical
cases and the original tolerances. This follow-up changes only tests and
this note, not atmosphere/source code or solver settings.
