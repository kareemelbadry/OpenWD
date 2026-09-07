# Tested temperatures and limitations — September 2026 checkpoint

These are **tested points**, not a guarantee of convergence throughout an
interval. All temperatures are effective temperatures; all rows below use
log g = 8. Numerical convergence does not establish completeness or accuracy
of the physical model. In particular, the cool DB and DAB workflows remain
explicit, experimental choices, not replacements for the warm-star defaults.

| Composition / workflow | Lowest successful point | Other tested points | What was established |
| --- | ---: | --- | --- |
| Pure-H DA, `compute_da`, production quality | **3000 K** | 4000, 5000, 20000 K | Protected cold starts, with actual all-depth radiative + ML2 flux and temperature-correction gates. |
| Pure-He DB, experimental dense-He workflow | **5000 K** | 8000 K | Fresh hydrostatic initialization, pseudo-time conditioning, then static convergence; independent wavelength, angle and source-closure checks. |
| Pure-He DB, established `compute_db` | **10000 K** | 22000 K (production and standard) | Protected cold-start convergence and iteration budgets. The dense-neutral EOS is not used. |
| Molecular homogeneous DAB/DBA, experimental workflow, log10(N_H/N_He) = -2 | **7500 K** | 7750, 8000, 9000, 10000 K | Static convergence plus independent fixed-state flux/local-energy audits. 7500/7750/9000 K are continuations. 8000 K has a fresh-start demonstration. |
| Atomic homogeneous DAB, paper checks, log10(N_H/N_He) = -2 | **9000 K spectrum only** | 20000 K spectrum and re-relaxation | The 9000 K UV spectrum is preserved, but its stricter newly measured correction check is not qualified. Do not count it as a new cold-start or stationary-convergence success. |

The molecular 10000 K calculation originally converged from a fresh seed with
the older Lyman interpolation, then re-relaxed with the final physical-detuning
interpolation. It is not a demonstrated cold start with that final policy.
The 7500 K continuation required an explicitly deeper domain to screen the
lower boundary. No saved atmosphere is silently selected by a production call.

## How to obtain the cool results

DA uses the normal public API:

```python
from wd_spectra import DAConfig, compute_da
result = compute_da(DAConfig(effective_temperature=3000, logg=8, quality="production"))
print(result.metadata["atmosphere_convergence_status"])
```

For dense-He DBs and molecular DABs use the
[versioned cool-model workflows](../research/cool_models/README.md). That
directory contains the exact proposal/physics components, strict checks,
data instructions, and explicitly named molecular-DAB continuation examples.
`DABConfig(include_molecules=True)` enables molecular chemistry in the public
API, but **by itself does not select the qualified conservative-transport
workflow**. Similarly, a default `compute_db` call at 5000 K is not the tested
dense-He calculation. Research workflows require a source checkout and SciPy;
they are not installed as part of the ordinary Python interface.

## Numerical evidence for the cool helium points

Errors are fractions of the relevant flux/energy scale, not percentages.
These measurements are from the completed investigations accompanying this
checkpoint, not timings or claims of a new full grid calculation.

| Case | Structure all-depth flux error | Local energy error | Measured maximum dlnT | Independent audit |
| --- | ---: | ---: | ---: | --- |
| DB 5000 K, fresh 80 layers | 1.26e-4 | 1.24e-5 | 2.58e-6 | Pass; largest independently sampled local error 0.00126. |
| DB 8000 K, fresh 80 layers | 6.76e-6 | 1.04e-5 | 5.46e-6 | Pass; largest independently sampled local error 0.000955. |
| DAB 7500 K, 166-layer deeper continuation | 7.62e-8 | 3.37e-8 | 1.19e-4 | Pass; flux 7.65e-5, local 7.56e-5. |
| DAB 7750 K, 162-layer continuation | 5.67e-10 | 1.81e-6 | 1.59e-4 | Pass; flux 1.09e-4, local 6.78e-5. |
| DAB 8000 K, fresh 80 layers | 1.91e-10 | 1.75e-6 | 1.79e-4 | Pass; flux 1.53e-4, local 6.39e-5. |
| DAB 9000 K, 80-layer continuation | 3.01e-9 | 2.23e-7 | 1.58e-5 | Pass; flux 3.13e-4, local 8.97e-5. |
| DAB 10000 K, 80-layer re-relaxation | 6.07e-10 | 1.23e-7 | 3.02e-6 | Pass; flux 4.63e-4, local 8.66e-5. |

The DAB 8000 K depth study used 80, 159 and 317 layers. Its integrated
absolute spectral differences decreased from 0.688% to 0.172% of the stellar
flux, approximately second-order behavior. That is evidence for this case,
not a depth-resolution certificate for every table entry. Independent
wavelength/angular audits hold the atmosphere fixed and do not replace a
small measured temperature correction.

## What is not established

- No DA result below 3000 K or pure-He DB below 5000 K is qualified here.
- Molecular DAB at **7250 K is not qualified**: the deeper-domain test had
  small flux/local residuals but a 0.00123 temperature correction, above the
  0.0003 gate. The 5000 K experiments remain far from convergence (roughly
  10% flux and 17% local-energy errors). No 5000–10000 K reliability claim.
- No abundance or gravity grid has been demonstrated. “DAB down to 7500 K”
  means the homogeneous 1% hydrogen-by-number mixture specified above.
- The dense-He closure combines neutral-background REOS thermodynamics with
  approximate HNC chemical potentials and trace He/He+/He2+/electron chemistry.
  It rejects states outside its table or trace-ion validity. Collective
  He-minus free-free corrections and refraction remain absent; other dense
  approximations are documented. Dense 10000 K failed an independent local
  energy check and dense 22000 K is outside the model domain. Use the
  established warm-star model there; no automatic EOS substitution occurs.
- Molecular H/He includes H2, H2+, H-, H3+, H/He ionization, H2-He/H2-H2 CIA
  and neutral Ly-alpha wings. A consistent dense-mixture free energy,
  nonideal dissociation, HeH+/He2+ mixed charge treatment and pressure-distorted
  CIA remain missing. The pure-He dense EOS is never inserted into this mixture.
- The DA seed retains the previously validated below-5000 K initialization
  policy. This is not a newly solved local-criterion policy; final physical
  convergence still uses real ML2 flux, not an imposed flux remainder.

## Protection of existing models

The protected suite checks fresh DA 3000/4000/5000/20000 K and DB 10000/22000 K
models, including the 40-layer 22000 K case. Assertions and iteration budgets
are not relaxed to accommodate the new checkpoint. General changes include
consistent scattering tangents, fixed Rosseland quadrature and material-domain
derivative handling. Conservative mass transfer and dense/molecular research
physics remain explicit selections because making them global defaults had
shown regressions in other models.

The retained paper tests distinguish spectra from atmosphere convergence:
the 20000 K DAB re-relaxes; the 9000 K atomic-DAB UV spectrum is preserved but
the stronger stationary check is unresolved. PG 1225-079 and SDSS J0738+1835
fixed-atmosphere DZ spectra reproduce to roundoff and pass current-grid flux
checks, **not** new cold-start or measured-correction tests.

Detailed historical records:
[checkpoint verification](checkpoint-validation-2026-09-07.md),
[solver consistency](solver-consistency-2026-09-04.md),
[dense-He applicability and verification](helium-physics-isolation-2026-09-06.md),
[cross-model limitations](dense-helium-cross-model-checks-2026-09-06.md), and
[molecular-DAB investigation](cool-dab-conservative-2026-09-07.md).
Those reports retain the chronology of unsuccessful experiments; their
original outer-workspace paths are historical, not the current run commands.
