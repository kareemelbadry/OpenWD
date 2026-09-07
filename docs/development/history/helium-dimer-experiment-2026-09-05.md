# Isolated He2+ charge/energy/opacity experiment

> Research record: this page describes work at the time it was written.
> For current usage and status, see the [user guide](../../getting-started.md)
> and [tested points](../../tested-temperature-ranges.md).

This follows the completed numerical work in
`direct-local-energy-experiments-2026-09-05.md`. No production EOS, opacity
default, public checkpoint format or solver dispatch is changed here.

## Why test the molecular ion?

The existing Stancil opacity computes molecular bound-free absorption using
`n(He2+) = n(He I)*n(He II)/K(T)`, but the atomic EOS does not include that
positive ion in charge balance. Consequently, opacity and chemistry imply
different ion populations. A consistent molecular charge model is a distinct
issue from bulk density: the [Kowalski et al. (2007) chemical model](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.76.075112)
includes He, He+, He2+ and electrons and their interactions. The later
[Blouin et al. (2018) framework](https://arxiv.org/html/1807.06616v1)
also separates the bulk EOS from chemical equilibrium.

The experiment **does not implement those nonideal interaction terms**. It
isolates the missing molecular charge using existing HM atomic partitions
and the equilibrium table already distributed with OpenWD.

## Closure and thermal consistency

With `n0,n1,n2,nd,ne` denoting He I, He II, He III, He2+ and electrons:

- Particle pressure: `P/kT = n0+n1+n2+nd+ne`.
- Helium nuclei: `NHe = n0+n1+n2+2*nd`.
- Charge: `ne = n1+2*n2+nd`.
- Dissociation equilibrium: `n0*n1/nd = K(T)`.
- Atomic Saha relations retain the HM partition functions.

Atomic fractions are evaluated by log-sum-exp; a positive quadratic root
gives the number of atomic particles at each trial electron density. A
bracketed charge solve closes the state. This avoids overflow of individual
Saha ratios in the highly ionized limit; an initial overflow-prone version
was interrupted and its partial products marked as unusable.

The molecular internal energy is derived from the same equilibrium law:
`u_d,int = chi_HeI + u_HeI,exc + u_HeII,exc + kT*(1.5-dlnK/dlnT)`.
Thus dissociation energy is not inserted independently of the population
law. Heat capacity and expansion differentiate the complete enthalpy and
density. This improves internal reaction consistency, but does not make the
underlying density-dependent HM atomic approximation a complete nonideal
free-energy model.

The interpolation is a natural cubic in reciprocal temperature for
`ln(K/T^1.5)`. It reproduces the existing 4200--50400 K nodes and has C2 tangent
continuations at both ends. This guarantees positive K without a temperature
clamp. It is an explicit extrapolation outside the tabulated range, not new
molecular data. The opacity uses the identical K; its tabulated cross sections
retain their existing low-temperature hold and wavelength support.

## Fixed-profile findings

These compare identical pressure and temperature arrays, not new atmospheres:

| Reference state | Electron-density ratio | Rosseland-opacity ratio | Fraction of charge in He2+ |
| --- | ---: | ---: | ---: |
| Failed 5000 K model, old photospheric layer (T=5199 K, rho=43.2 g/cm3) | 75.81 | 19.83 | 0.999826 |
| Numerically validated 8000 K model, old photospheric layer (T=8173 K, rho=0.109 g/cm3) | 1.871 | 1.129 | 0.714398 |

Density barely changes at fixed P,T because these ions remain trace particles;
their effect on electron-mediated opacity is much larger. These ratios
motivate a new hydrostatic calculation but cannot predict its final density
or prove convergence. Output:
`results/helium-dimer-fixed-state-v2-20260905.json`.

## Isolation and verification

The implementation is in `scripts/helium_dimer_eos_experiment.py`; the
fixed-state audit is `scripts/audit_helium_dimer_closure.py`. Unit checks
cover pressure, nuclei, charge, mass action, the atomic limit, smooth thermal
derivatives, extreme-ionization arithmetic and restored defaults after the
experimental context exits.

The experimental run writes `experimental-dimer-structure.npz` with deliberately
different array names, plus `experimental-spectrum.txt`, metadata and live
telemetry. It cannot be loaded as a converged production checkpoint. A test
checks this rejection. No production result is overwritten or adopted.

Fresh 5000/8000 K atmosphere tests use `--helium-dimer --mesh optical
--physical-only --no-continuations --stable-transfer --step-method
nonlinear-convection-current-energy --inner-scaling jac
--inner-max-evaluations 1000 --max-iterations 20` in the existing research
runner. Both chemistry and opacity are scoped together. No trace hydrogen,
REOS substitution, saved atmospheric solution or flux renormalization is used.
The corrected run outputs are under
`results/cool-db-dimer-equilibrium-fresh-v2-20260905`.

## Completed fresh-model results

Both corrected runs completed their 20-attempt budget. Neither converged:

| Teff | Elapsed including fresh seed and synthesis | Maximum all-depth flux error | Maximum relative local-energy error | Spectrum / sigma Teff^4 |
| --- | ---: | ---: | ---: | ---: |
| 5000 K | 207 s | 0.516600 | 0.374220 | 0.764151 |
| 8000 K | 558 s | 0.007701 | 0.029153 | 0.976133 |

Their final maximum applied log-temperature changes are 0.032036 and
0.009888, respectively, also well above the 0.0003 gate. The 8000 K run
improved initially but made little progress over its last several iterations;
this bounded experiment does not establish whether a longer or refined run
would converge. It is not an improvement over the validated numerical
8000 K production-physics refinement sequence.

The 5000 K experimental photospheric density is still 27.9 g/cm3 and its
maximum density is 47.9 g/cm3. Forty-three of 80 layers are below 4200 K,
where this experiment extrapolates molecular equilibrium and holds the
tabulated cross sections. Such a model cannot validate the dense-He physics.
The corresponding 8000 K densities are 0.102 and 0.182 g/cm3, with no layer
below 4200 K. These are diagnostics of failed states, not recovered physical
atmospheres.

**Decision:** keep this prototype research-only. The missing molecular charge
is a real inconsistency to address, but its isolated correction is not a
validated cool-DB solution. Do not add an electron floor or adjust molecular
equilibrium to force convergence. A subsequent physical implementation needs
validated nonideal chemical potentials and opacity applicability, along with
the already demonstrated mesh/local-energy checks. For cool mixed atmospheres,
H2 equilibrium, reaction thermodynamics and H2-He CIA must be coupled before
the numerically converged 5000 K atomic DAB can be called physically complete.

All seven molecular-experiment unit tests pass; the combined research suite
(step constructions, stable transfer, domain extension, molecular closure)
passes 31 tests. This is in addition to the production suite and protected
model checks recorded in the numerical report. No experimental chemistry
default or experimental atmosphere has been adopted in production.
