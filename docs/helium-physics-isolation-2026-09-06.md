# Helium physics applicability and solver isolation — 2026-09-06

## What changed

1. The shared adaptive solver now has an explicit per-call
   `transfer_discretization="column-mass"` option. The field, independent
   fixed-source check, material response, and local energy balance all use
   physical mass control volumes together. The canonical implementation is
   `src/wd_spectra/_mass_feautrier.py`; the old research module only re-exports it.
   Default calls still use the established optical-depth equations. There is
   no Teff threshold and no change of discretization during a Newton solve.
2. The research mass-transfer adapter no longer replaces field functions or
   stores a globally shared last-evaluated mass/opacity/tau array. Numerical
   unit tests exercise interleaved and concurrent calls with different options.
   The remaining research physics/proposal scopes still require separate
   processes; this change does **not** make the entire research driver a
   thread-safe production API.
3. Exceeding the dense chemical model's trace-ion/excited-state approximation
   during a candidate step now raises the same recoverable domain exception
   as leaving its material table. The line search can shorten such a step
   within the **same** EOS. An invalid initial/accepted state still fails.
4. A complete numerical residual derivative can use a backward probe when
   the forward probe crosses a physical domain boundary. The existing valid
   forward-probe path is unchanged. If neither side is admissible, the solver
   returns its last evaluated state explicitly **unconverged**, with reason
   `finite-difference-domain-exhausted`. Other exceptions still propagate.
5. A fresh 5000 K test exposed the same issue in centered **material**
   derivatives: the actual surface was at 1160.5206 K (valid), but the minus
   0.0002 log-temperature probe was at 1160.29 K (outside HNC support).
   Shared `temperature_response_probes` now selects an admissible one-sided
   stencil there. The dense ML2 opacity response uses that same helper;
   its REOS thermodynamic response remains analytic. Interior stencils are
   unchanged. An artificial table extension was not introduced.
6. The pseudo-time initializer hands the same valid state to the full static
   solver when its derivative becomes domain limited. It need not continue a
   temporary thermal trajectory toward an unavailable state: the constrained
   static solution can lie back inside the physical domain. This handoff is
   never a convergence declaration, and a newly measured stationary correction
   is still required. The aborted 5000 K state recovered in 17 static iterations.

No opacity/EOS value, convergence threshold, reference spectrum, or default
stellar configuration was changed in this tranche. No GitHub push was made.

## Physics: density, ionization, and electron donors are separate questions

Measurements are on the saved log g = 8 profiles, at Rosseland tau = 2/3.
Different wavelengths form at different depths; these are diagnostics, not
universal switching thresholds. Machine-readable values and source paths:
`results/helium-physics-isolation-20260906/regimes-v2.json` in the research workspace.

| DB Teff | Photospheric density (g cm^-3) | Interpretation |
| --- | ---: | --- |
| 5000 K, dense model | 1.037 | Strongly nonideal bulk; P/(n_He kT) ≈ 3.99. |
| 8000 K, dense model | 0.0881 | Moderate nonideality; pressure ratio ≈ 1.114. |
| 10000 K, dense model | 0.00899 | Bulk pressure correction ≈ 1%; chemistry is a separate issue. |
| 22000 K, established model | 4.82e-6 | Dilute, substantially ionized: ne/n_He ≈ 0.375. |

The dense treatment is a neutral-background, trace-ion approximation, not a
general replacement for the full He I/II/III chemical model. The 22000 K
profile exceeds both its ionization validity and its temperature table.
It must retain the established warm-star physics; extrapolating the dense
table or silently substituting another EOS inside its Newton residual is not
an acceptable way to make the experiment run.

Conversely, a small molecular *mass fraction* does not establish a negligible
molecular-ion contribution to the scarce electrons. Applied diagnostically
to the established 10000 K photospheric state, the dense chemical model gives
He2+ about 9.35% of the electrons, but only about 5.2e-7 of all helium nuclei
bound into those ions. H2 is not present in a pure-He atmosphere. No existing
warm-star molecular-ion opacity was removed. The very dense 5000 K model
also illustrates why ideal molecular equilibrium must not simply be assumed:
its nonideal chemical potentials strongly suppress He2+ relative to the
ideal trace-ion estimate.

This separation follows the physical considerations in
[Blouin et al. (2018)](https://www.astro.umontreal.ca/~bergeron/CoolingModels/Blouin2018.pdf):
dense-fluid thermodynamics, helium ionization, molecular chemistry, and metal
electron donation need separate consistent treatments. Pure-He infrared
three-body absorption is also density dependent
([Kowalski 2014](https://arxiv.org/abs/1406.4591)). Neither source justifies a
universal switch based on Teff alone.

## Verification

- Final combined suite: **491 passed**, including all seven slow protected
  atmosphere canaries and the affected research tests (`final-unit.xml`,
  1255 s). Running both test roots together omitted the release-only marker
  registration, producing one harmless unknown-`canary`-marker warning;
  the canaries were included and passed, not skipped.
- Earlier regular suite: **449 passed**, 7 slow canaries explicitly excluded
  (`unit.xml`); earlier affected research tests: **22 passed**
  (`research-unit.xml`). These preceded the final material-probe additions.
- Both 317-layer cool continuations: newly evaluated stationary corrections,
  unchanged physical gates, **bit-for-bit identical emergent spectra** to
  the previous smoothed-opacity results. These are warm continuations, not
  cold starts. At 5000 K the structure differs only at floating-point
  roundoff; at 8000 K its temperature array is bit-for-bit identical.
- First protected-canary pass with the smooth opacity join: **7/7 passed**.
  Observed iteration counts were DA 3000/4000/5000/20000 K: 54/35/35/22;
  DB 10000/22000 K production: 18/31; DB 22000 K standard: 28.
  The final full-suite pass also passes all seven, with the final material-probe
  fix and the unchanged production opacity default.
- Paper DAB 9000 K: current all-depth flux error 0.000471604; saved state
  accepted without a newly measured correction. Its UV spectrum is unchanged;
  the previously introduced opacity join changes integrated optical flux by
  -0.0981%. DAB 20000 K re-relaxes in three iterations, flux error 4.69802e-6,
  step 0.000151396. These are saved-state checks, not DAB cold starts.
  Both spectra and both temperature arrays are bit-for-bit identical to the
  corresponding pre-tranche runs with the same opacity join enabled.
- DZ PG1225 and J0738: fixed paper spectra agree to 1.84e-11 and 4.25e-14;
  current all-depth flux errors 0.001600379 and 0.002233277 pass the unchanged
  0.003 gate. These are not new atmosphere relaxations or measured steps.
- Fresh 8000 K: 30 pseudo-time initialization sweeps, then 8 static iterations;
  flux error 6.76210e-6, local energy 1.04129e-5, temperature step 5.46098e-6.
  Independent 8000-wavelength, 8/16-angle audits pass (largest local-energy
  error 0.000954710). This is a fixed-depth convergence check, not a new depth
  resolution certificate. This worker preceded the late material-probe fix
  and stayed inside the material domain; a static handoff probe and the
  317-layer continuation separately verify the final code at 8000 K.
- Fresh 5000 K with the final automatic handoff: **56 initialization sweeps,
  then 17 static iterations**, no supplied atmosphere. Flux error 0.000126479,
  local energy 1.23591e-5, temperature step 2.57831e-6; spectrum integral
  0.999982431 times sigma Teff^4. Total reported time 565 s with concurrent
  workers (not a controlled speed benchmark). This replaces the numerical
  derivative abort observed at sweep 57 of the original fresh test.
- The final automatic 5000 K result passes its own independent
  8000-wavelength, 8/16-angle audit: largest local-energy error 0.00125589,
  integrated flux ratios 0.999981721/0.999991102. The manually recovered
  5000 K separately passes the same checks (largest local-energy error
  0.00125580). Neither fixed-depth audit is a new depth-resolution certificate
  or a validation of all dense-helium physics.

The machine-readable consolidated evidence is
`results/helium-physics-isolation-20260906/verification.json` in the research
workspace. The original failed fresh-5000 diagnostic remains preserved there
alongside the successful rerun; failed results were not relabeled or replaced.

## Deliberate limits

The dense REOS/HNC/He2+ implementation is still experimental. Its approximate
interaction potentials, low-temperature cross-section hold, non-free-energy
bulk interpolation, missing collective He-minus free-free corrections, and
missing refraction were not repaired by this numerical tranche. A consistent
dense H/He/metal mixture is still absent. The earlier dense 10000 K calculation
also failed an independent, wavelength-sensitive local-energy check; the
established warm 10000 K canary is not a validation of that different model.
Numerical convergence is not evidence that all dense-helium physics is complete.
