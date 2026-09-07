# Cool DB / molecular DAB: material consistency investigation

> Research record: this page describes work at the time it was written.
> For current usage and status, see the [user guide](../../getting-started.md)
> and [tested points](../../tested-temperature-ranges.md).

This follows `cool-dab-molecular-2026-09-05.md`. The target is **pure DB and
helium-dominated molecular DAB**, not pure DA. All new atmosphere experiments
below use log g=8; DAB uses log10(N_H/N_He)=-2. No GitHub push was made.

## Literature-led choices

[Rohrmann 2001](https://academic.oup.com/mnras/article/323/3/699/1155144)
describes the instability of efficient convection in cool, high-gravity
atmospheres: tiny gradient changes can alternate between excessive and zero
convective flux. A consistent convective starting structure and accurate
thermodynamic derivatives precede a global correction. This suggests improving
material consistency, not inserting a 5000 K switch or suppressing convection.

[Hubeny 2017, sections 3.4.2 and 5.4](https://academic.oup.com/mnras/article/469/1/841/3092374)
describes CoolTLUSTY's coupled global correction and explicit convection
repairs, including re-evaluation of chemistry and opacity. The public
[TLUSTY 208 source](https://www.as.arizona.edu/~hubeny/tlusty208-package/tl208-s54.tar.gz)
was inspected (`CONCOR`/`CONREF`). Its flux-target inverse-convection repair
motivated an experiment, but its numerical clipping and iteration-window
rules were not copied into production.

The experimental correction enforces finite-temperature ML2 compatibility
using recomputed Cp, density, expansion coefficient and adiabatic gradient.
It then returns a **proposal**, which must pass the existing full-physics
line search and final convergence checks. No actual convective flux is
replaced by Fstar-Frad. A separate literal flux-target proposal was also
tested and rejected when it worsened physical residuals.

## Production changes retained

1. **Cancellation-resistant pure-He thermodynamics** in `eos.py`.
   For ideal-pressure HM, the neutral translational enthalpy 2.5 kT/mHe is
   differentiated analytically. Only the remaining electron, ionization and
   excitation enthalpies are differenced. Thermal expansion uses
   `1 + d ln(1+mean_ion_charge)/d lnT`, evaluated with `log1p`. This retains
   trace-ion responses without subtracting large density logarithms. No
   temperature or ionization threshold, Cp clamp, electron floor or physical
   opacity change is introduced. The REOS derivative algorithm is unchanged.
2. **Experimental-checkpoint safety** in `models/common.py`.
   On loading a checkpoint tagged `experimental_h2_partition`, its structure
   remains available for explicit re-solving, but its convergence flag and
   production request fingerprint are invalidated. Rebuilding populations
   with the default EOS cannot inherit an external experiment's convergence.

Across a 360-state pressure/temperature test, the He derivatives agree with
the original full-enthalpy stencil to well within 2e-9 relative. In nearly
neutral tests at 5000/6000 K, variation in nabla_ad induced by dlnT~1e-10
perturbations fell from about 1e-12 to machine precision (~6e-17). Trace
ionization remains present. This is a conditioning improvement, **not** a
demonstrated cure for 5000 K DB convergence or a general runtime speedup.

## H2 inconsistency found, not silently installed as a new default

Production interpolates log Q versus log T piecewise linearly, but derives
H2 rovibrational energy from separately interpolated nodal gradients. Thus
the energy is not the derivative of the Q used in molecular equilibrium.
Above the table's upper edge, Q is constant while the separately held energy
slope remains nonzero. These are inconsistent thermodynamic definitions.
Artificial slope changes near table knots also affect Cp and nabla_ad.

Two isolated research treatments were tested:

- A common quintic spline for log Q and its energy derivative, interpolating
  the exact [Barklem--Collet nodes](https://arxiv.org/html/1602.03304v1).
  It is smooth over the atmosphere range but generates negative energy/heat
  capacity near 10–13 K and uses an explicit hot-end extrapolation. It is
  **not suitable as a general production default**, even if an atmosphere
  using it passes numerical convergence.
- A direct sum over the 302 ground-electronic H2 levels of
  [Roueff et al. 2019](https://arxiv.org/html/1909.11585v2), using identical
  weights for Q, mean energy and heat capacity. No interpolation is needed;
  heat capacity is proportional to the nonnegative energy variance. This
  changes the molecular model as well as the numerical consistency: Q is
  4.47% lower at 5000 K and 7.10% lower at 8000 K than the old table. It does
  not add nonideal dissolution or dense-fluid thermodynamics.

Data provenance, normalization, hashes and CC BY-SA attribution are in the
research workspace's `scripts/H2_STATE_SUM_DATA.md`. Data remain optional
external cache files; missing or changed state data fail explicitly. Neither
research treatment changes production DA or DAB defaults.

## Completed numerical experiments

Errors are dimensionless fractions, **not percentages**. All-depth flux uses
the actual radiative plus ML2 flux. Local error uses current thermal/convective
cell scales. The spectrum is independently integrated without renormalization.
All continuations explicitly reuse an **unconverged** saved state, not a
hidden successful atmosphere, and are not called cold starts.

| Test | Attempts | All-depth flux error | Local energy error | Spectrum/Fstar | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| 5000 molecular DAB, same-state control | 15 | .0355861 | .0598461 | .970813 | Failed |
| 5000 molecular DAB, positive-face finite-material repair | 15 | .0263689 | .0388753 | .971218 | Failed |
| 5000 molecular DAB, signed inverse-ML2 + repair | 10 | .0348315 | .0583418 | .971229 | Failed |
| 5000 pure DB, signed inverse-ML2 + stable He + repair | 10 | .348241 | .419942 | .856199 | Failed |
| 5000 molecular DAB, shared H2 spline | 20 | .0348938 | .0507771 | .968735 | Failed |
| 5000 molecular DAB, shared H2 spline + nonlinear EOS table, Jacobian scaling | 10 | .0345145 | .0627729 | .970673 | Failed |
| 5000 molecular DAB, shared H2 spline + nonlinear EOS table, bounded SVD coordinates | 12 | .0289359 | .0361416 | .969165 | Failed; trust collapse |
| 5000 pure DB, fresh analytic seed + nonlinear EOS table | 20 | .581572 | .361381 | .810415 | Failed |
| 8000 molecular DAB, shared H2 spline | 11 | 4.69e-10 | 2.17e-6 | .999089 | Passed on this grid |
| 8000 molecular DAB, matched old-partition control | 16 | 7.54e-10 | 3.27e-6 | .999089 | Passed on this grid |
| 5000 molecular DAB, direct H2 state sum | 20 | .0386334 | .0538256 | .966557 | Failed |
| 8000 molecular DAB, direct H2 state sum | 10 | 4.01e-10 | 6.63e-7 | .999089 | Passed on this grid |

The positive-face 5000 DAB repair took 736 s versus 209 s for its paired
control. The improvement is modest and **slower**, not a production speedup.
The later signed-face version did not retain that improvement. The first
repair's code version differed: it repaired only convecting faces and used a
stricter proposal compatibility tolerance. Current scripts repair signed ML2
coordinates on all faces, including stable faces; neither version was promoted.

The literal TLUSTY-style 5000 DAB flux-target proposal worsened the actual
merit at every tested multiplier (1, 1/2, 1/4, 1/8). Pure DB proposals failed
the explicit local branch bound. No such candidate was adopted. Locally
tabulating the opacity with a global Chebyshev polynomial failed independent
accuracy tests at a CIA table knot; the EOS-table experiment therefore leaves
opacity on its existing tangent. This limitation is explicit, not a fallback.

The spline 8000 K result took 475 s and also passed a measured small final
proposal (max dlnT=1.80026e-4) and the lower-boundary absorption escape check
(1.74e-5). It is a **162-layer restart**, not a cold-start or mesh-convergence
demonstration. The paired old-partition control escaped an initial eight-step
stall when allowed further iterations and converged at iteration 16, with a
measured final max dlnT=2.46520e-4 and spectrum/Fstar=.9990894305. The spline
cannot therefore be called the reason 8000 K is solvable. Its 11 versus 16
iterations is one controlled numerical comparison, not broad evidence of
robustness. Control elapsed time was 962 s, but CPU contention changed between
runs: do not interpret 475 versus 962 s as a reliable 2x speedup.

The direct-state-sum 8000 K run converged in ten attempts (621 s), with final
max dlnT=5.65445e-5, independent spectrum/Fstar=.9990891244 and bottom escape
bound=1.75041e-5. This is a more defensible molecular representation than the
overshooting spline, but is still an experimental 162-layer restart. At
5000 K it failed after 20 attempts (464 s), so it is not a general cure.

## Why pure 5000 K DB remains a separate problem

Both compositions suffer from efficient-convection conditioning, but the H2
bug cannot explain a hydrogen-free DB. Its fresh ideal-pressure HM seed reaches
rho~207 g/cm3 near Rosseland tau=0.94 and ~267 g/cm3 at the bottom, far outside
a credible ideal neutral-He treatment. The sampled photospheric seed point
has T=4870 K, P=2.09e13 dyn/cm2. In the mixed 5000 K saved trial, the closest
tau~1 point has rho=.122 g/cm3 (tau=1.44), with .208 g/cm3 at the bottom.
Improving the numerical derivative does not make that dense structure physical.

[Blouin et al. 2018, sections II–III](https://arxiv.org/html/1807.06616v1)
distinguishes dense-fluid EOS, nonideal chemical potentials/ionization,
He2+ chemistry, density-dependent opacities and refractive transfer. Existing
He3 CIA, He2+ absorption, He-minus free-free and density-dependent Rayleigh
terms do not by themselves supply consistent dense-fluid charge chemistry.
The mixed 5000 K case also reaches densities where nonideal dissociation and
pressure-distorted CIA can matter. Neither is advertised as containing all
relevant dense physics. An arbitrary electron floor or opacity multiplier is
not an acceptable solution.

## Reproduction and regression status

All runner options are saved to `experiment-options.json`, with live
`solver.log`, `iterations.jsonl`, intermediate states and final summaries.
Typical commands, from the research workspace (not the production repo):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 KMP_USE_SHM=0 PYTHONPATH=release/OpenWD/src \
  /Users/kareem/opt/miniconda3/bin/python -u scripts/run_molecular_dab_experiment.py \
  8000 --log-h-he -2 \
  --resume-atmosphere results/cool-dab-full-molecules-20260905/8000/atmosphere.npz \
  --physical-only --stable-transfer \
  --step-method nonlinear-convection-current-energy --inner-scaling svd \
  --inner-max-evaluations 1000 --max-iterations 20 --no-continuations \
  --state-sum-h2 --output-root results/NEW-UNUSED-OUTPUT-DIRECTORY
```

Omit `--state-sum-h2` for the matched old-partition control. Use `--smooth-h2`
instead for the explicitly non-production interpolation experiment. The 5000 K
restart is `results/cool-dab-molecular-cold-correction-20260905/5000/atmosphere.npz`.
Use an unused output directory; saved experiments must not be overwritten.

The initial stable-He derivative prototype passed all seven protected DA/DB
canaries (1252 s). The final production He implementation was independently
retested on all three DB canaries: 10000 K/80 layers and 22000 K/80 and 40
layers; all passed (862 s). DA production calculations do not use the modified
He routine. The final regular suite, including the checkpoint guard, passed
416 tests with five optional-data skips and seven separately tested canaries
deselected (102 s). Atomic paper DAB and DZ defaults are untouched by these
changes. No molecular partition or research proposal became a default.

An additional 43 focused production/research tests passed after formatting.
They cover finite-material tangents, nonlinearly repaired gradients, explicit
repair failure, independently checked local EOS tables, bounded SVD mapping
and its chain rule, state-sum thermodynamic identities and nonnegative heat
capacity, the independent ExoMol PF, scoped restoration of EOS functions,
invalid-temperature errors and checkpoint safety. `git diff --check` passed.

## Next priorities, without temperature-specific tuning

1. Treat the 8000 K converged molecular control as a regression target, with
   the same all-depth/local-energy/correction/spectrum gates. Establish a cold
   start and mesh check separately; a restart does not prove those properties.
2. For 5000 K pure He, obtain a validated dense-He **thermodynamic and chemical
   closure together**. A bulk density table alone, HM populations evaluated
   independently of that table, or He2+ opacity without charge equilibrium
   cannot establish consistency. Check its density/ionization validity before
   spending another large budget optimizing an implausible atmosphere.
3. For molecular H/He, choose a common free-energy/partition representation
   for mass action and derivatives, with an explicit dense-regime validity
   range. Validate any changed molecular physics on cool DA as well as DAB
   before changing defaults. The failed 5000 K direct-state-sum test shows that
   fixing the H2 identity alone is insufficient.
4. Continue finite-material/transfer globalization only on physically supported
   closures. The present repair/table variants are measured research failures,
   not a recommendation to add another automatic solver fallback.
