# Atmosphere solver review and controlled experiments — 2026-09-05

> Research record: this page describes work at the time it was written.
> For current usage and status, see the [user guide](../../getting-started.md)
> and [tested points](../../tested-temperature-ranges.md).

For the subsequent bounded-step and nonlinear-convection experiments, see
[the follow-up tranche](nonlinear-convection-experiments-2026-09-05.md).
It adds a cancellation-safe ML2 root and reports a faster fresh-seed 8000 K
discrete solve, with important local-temperature and spectral-flux limitations.

## Conclusions and protection boundary

The cool-He failure is not one missing damping constant. The evidence separates
an inadequate gray initial mass domain, poorly resolved efficient convection,
weak constraints on optically thin temperatures, discretization errors in the
emergent spectrum, and incomplete dense-He physics. Changing convergence
tolerances or renormalizing spectra would obscure these distinct problems.

The shared production driver remains the default for DA, DB, DAB and DZ.
This tranche adds observational local-energy diagnostics and isolated,
explicitly requested research formulations. None of the new formulations is
selected automatically, and none imports the quarantined solver. Existing
validated initialization policies, including the cool-DA policy, are retained
until replacements pass cross-composition cold-start tests. No GitHub push
was made during this investigation.

## Comparison with published implementations

| Reference | Relevant design | Implication for OpenWD |
| --- | --- | --- |
| [Rohrmann (2001), sections 3.3–3.4](https://academic.oup.com/mnras/article/323/3/699/1155144) | Hydrostatic pressure integration, Rybicki elimination, and a convection-consistent initial structure before global correction; explicitly discusses convective flux switching instability. | Construct a suitable hydrostatic/convective seed and solve the final non-gray equations rigorously. Do not interpret an adiabatic warm-start constraint as measured flux conservation. |
| [Koester (2010), atmosphere iteration and formal solution](https://articles.adsabs.harvard.edu/pdf/2010MmSAI..81..921K) | Pressure coordinate, Feautrier transfer and Rybicki/variable-Eddington treatment; a separate formal spectral solution. | Fixed pressure is legitimate. The physical depth domain and mesh still need verification. Independent synthesis should agree after boundary and grid convergence; it need not numerically equal the structure discretization on a coarse grid. |
| [Hubeny (2017), sections 2.1, 3.3–3.4 and 7.3](https://academic.oup.com/mnras/article/469/1/841/3092374) | Combines integral heating balance and differential flux constancy, with convective divergence included; globally coupled correction and material/transfer consistency checks. | Flux constancy alone poorly determines thin-layer temperatures. Add a local heating diagnostic, and test consistently differentiated local/flux equations. Recompute chemistry and opacity after actual temperature changes. |
| [Published TLUSTY 208 source](https://www.as.arizona.edu/~hubeny/tlusty208-package/tl208-s54.tar.gz), `RYBENE`, `CONCOR`, `CONREF`, `TEMCOR` | Energy rows include radiative and convective terms and derivatives; the temperature gradient participates consistently in the correction. Optional convective trial repairs are distinct from final equilibrium. | An auxiliary convective coordinate is a better structural experiment than clipping convective flux. Copying individual empirical switching constants without the surrounding formulation is unjustified. |
| [Blouin et al. (2018), sections II–III](https://arxiv.org/html/1807.06616v1) | Dense-He bulk EOS and chemical ionization are separate ingredients; dense-fluid opacity and refractive effects also matter. | Turning on the existing density table alone does not complete the cool-He physics. A numerical solution of the ideal/HM model would not validate it at hundreds of g/cm³. |

The TLUSTY source was inspected from the previously downloaded official archive
in the research workspace, `tmp/helium/tlusty208/tl208-s54/tlusty/tlusty208.f`.
In that copy `RYBENE` begins around line 47239; `REINT`/`REDIF` setup is around
4400; `CONCOR` and `CONREF` around 27924 and 27977. This is source inspection,
not a claim that the OpenWD experiment reproduces TLUSTY's complete algorithm.
The [TLUSTY operational guide, section 12.6](https://arxiv.org/html/1706.01937v1#S12.SS6)
identifies the older `ITMCOR` correction as optional and disabled by default.
The earlier OpenWD trial cap remains off: its controlled test was slower.

## What the current equations do

The shared solver holds hydrostatic pressure/mass nodes fixed while correcting
temperature. Its state contains a temperature normalization and neighboring
logarithmic temperature gradients. Each residual rebuilds the requested EOS,
continuum/line opacities, coherent scattering and ML2 convection. The direct
angular-block Feautrier response differentiates the same scattering problem
used by the accepted residual. ML2 material derivatives include changes in
the local thermodynamics and Rosseland opacity.

The deliberately approximate convective-gradient phase only constructs a
warm start. Physical completion requires actual radiative plus ML2 flux at
all interfaces and a small temperature correction. The independent spectrum
integral, lower-boundary thermalization screen, and mesh convergence remain
separate checks. Continuum-only Rosseland means are preserved for DA/DB/DAB;
DZ retains its established full-extinction mean. The previously unsuccessful
full-line Rosseland change was not reinstated.

### A newly measured weakness: thin-layer heating

`_energy_balance.py` reconstructs each Feautrier control volume's radiative
exchange as the wavelength integral of `4*pi*delta_tau*epsilon*(J-B)`, then
adds the actual difference of convective flux through its faces. Scattering
cancels from this energy exchange. The normalization is thermal emission plus
the magnitudes of convective flux through both faces. The final boundary node
is excluded because it obeys a boundary condition, not a cell equation.

Read-only re-evaluation with `scripts/audit_saved_energy_balance.py` found:

| Saved 8000 K model | Maximum actual flux error | Maximum relative cell-energy error |
| --- | ---: | ---: |
| Optical mesh, 80 layers | 0.00193434 | 0.16314 |
| Optical mesh, refined to 159 | 0.00264055 | 0.20155 |
| Pressure mesh, refined to 159 | 0.0000133248 | 0.13752 |

In the last row the worst cell is the very thin surface cell: its absolute
energy defect is only `1.15e-10 F_star`. Thus the 14% local imbalance is not
a 14% bolometric error. It means the flux tolerance hardly constrains that
cell's temperature. This diagnostic currently makes no automatic changes to
convergence flags or working spectra. Local normalization near machine-noise
limits must be assessed before introducing a general mandatory heating gate.

## Implemented experimental formulations

### Equivalent energy rows

Three fixed, invertible transformations retain the same constant-flux root:
cell flux differences plus the bottom flux; a cell-optical-thickness weighted
combination; and flux minus locally normalized cell divergence. The first
two did not improve the tested cool-DB refinement and are not promoted.
The third measures thin radiative cells against their actual thermal emission
and tends toward ordinary flux constancy in large-emission cells. Scales are
fixed within a nonlinear segment, avoiding a moving hard equation mask.

### Auxiliary ML2 coordinate

Write the original ML2 relation as `delta = L*x + x²`, `F_conv = K*x³`,
where `delta = nabla - nabla_ad`, and `x >= 0` on the convective branch.
With `x_star = (F_star/K)^(1/3)`, the independent signed coordinate `y` obeys
`delta = a*y + b*y*abs(y)`, with `a=L*x_star` and `b=x_star²`.
The physical positive-branch flux is exactly `F_star*max(y,0)³`.
The signed negative branch describes stable cells with **zero**, not negative,
convective flux. No radiative-loss or flux coefficient is changed.

The enlarged system solves radiative-plus-auxiliary flux conservation together
with gradient compatibility. Its analytic tangent includes all existing ML2
material responses. Both residual families must pass, and an additional gate
recalculates ML2 from the **actual temperature gradient**. It is impossible
to declare convergence merely because the independent auxiliary flux closes
the budget. Tests explicitly inject such a false closure and require rejection.

Stable cells can have large signed coordinates despite carrying zero flux.
The experiment therefore uses a fixed local coordinate scale covering the
stellar-flux velocity or the initial absolute velocity, whichever is larger.
This is nondimensionalization, not a fitted convection threshold. Compatibility
uses the corresponding gradient scale. The generic trust radius still limits
the actual change in log temperature. The initial unscaled experiment and its
replacement have distinct output directories and are not conflated.

The implementation lives in the private `_ml2_auxiliary.py` module and is only
called by the explicitly opted-in research harness. Enlarged-system telemetry
is stored under `diagnostic_auxiliary_ml2_segments`; final physical residuals
are reported separately. No archived atmosphere is silently substituted.
Explicit refinement of a named model is always labelled as refinement, not
a fresh cold start.

### Bounded comparison results

All entries below are 8000 K, log(g)=8, pure He, and use identical named
80-layer parent atmospheres for their respective 159-layer refinements.
Existing physical and temperature tolerances were retained.

| Representation | Pressure-grid result | Optical-grid result |
| --- | --- | --- |
| Earlier production-equation control | Converged in 1319 s; flux error 1.33e-5 | Converged in 1137 s; flux error 0.00264 |
| Pure cell-difference rows | Stopped, physical error about 0.557 | Stopped, physical error about 0.179 |
| Optical-thickness combination | 12-step limit, physical error 0.483 | 12-step limit, physical error 0.168 |
| Auxiliary ML2 with fixed coordinate scaling | Converged in 223 s / 10 iterations; physical error 0.000180 | 12-step limit after 754 s; physical error 0.002265 but temperature step too large |

The faster pressure-grid solution's public spectrum integrates to 0.987679
of the target, essentially the control's 0.987687. Thus this is a real
improvement in the tested nonlinear solve, not a repair of its spectral-grid
error. The optical auxiliary result integrates to 0.995329 but is explicitly
unconverged. Fixed local thermal-energy scaling with the auxiliary system
also struggled and was stopped with physical error about 0.283. None of the
unsuccessful alternatives becomes a default.

The scaled auxiliary 159-to-317-layer optical refinement made three accepted
steps in about nine minutes, reducing actual flux error from 0.09032 only to
0.08789. It was stopped with diminishing steps. This is not a converged fine
grid or evidence that the spectral-integral test passes.

Fresh 5000 K, 80-layer optical-grid initialization still failed. The bounded
warm-start-plus-auxiliary calculation ended at physical error 1.81e7. A direct
physical solve using the *same unrelaxed analytic seed* rejected all steps and
ended at physical error 0.7770. Its public spectral integral was 0.64589.
The seed's bottom density was 267.5 g/cm³ with the existing ideal/HM EOS.
This isolates an algorithm failure in addition to the unresolved physical
validity of that dense model. No warm result replaced either failed solve.

A fresh 8000 K optical-grid calculation also failed its bounded test (20
warm-start plus 20 physical iterations), ending at actual flux error 0.33768
after approximately twelve minutes. Thus the pressure-grid refinement speedup
does not establish an improved general cold-start method. This run is saved
under `results/auxiliary-ml2-fresh-optical-20260905/8000` in the research workspace.

### Exploratory-output bug fixed

The zero-accepted-step 5000 K run exposed an unrelated serialization failure:
the absent final correction was stored as infinity, invalid in strict JSON.
The shared driver now uses `None`/`null` for an unmeasured correction, retains
the unconverged status and warning, and saves the exploratory result. A unit
test rejects every trial and verifies checkpoint round-trip; the actual
5000 K failure was rerun and saved successfully. Strict JSON validation was
not weakened, and no unknown correction was replaced by zero.
The same export audit found infinity sentinels for infeasible rejected
trials. Their exported residual entries now become null, preserving the
internal diagnostics, -1 depth index and rejection count. A converged
synthetic solve with an infeasible trial verifies strict-JSON export too.

## Regression verification for the default path

The seven protected cold-start canaries passed, including the 10000/22000 K
production DBs and the 3000/4000 K production DAs. These checks exercise the
unchanged default equations, not a claim that auxiliary ML2 is validated for
all compositions. Derivative tests additionally exercise the enlarged system
through the real shared transfer/material callbacks, both without scattering
and with 98% scattering.
The full regular suite passed 345 tests, with five optional-data skips and
the seven separately executed canaries deselected. `git diff --check` passed.

Saved-atmosphere checks are reported separately from cold starts:

- PG 1225 and J0738 paper spectra agree to maximum relative differences
  1.84e-11 and 4.25e-14. Their current all-depth flux errors on the saved grids
  are 0.0016794 and 0.0022333; both pass the existing 0.003 flux requirement.
- The 9000 K DAB paper atmosphere already passes current equations with flux
  error 8.87e-5; its UV spectrum is unchanged to roundoff.
- The 20000 K DAB saved atmosphere relaxes in three steps to flux error
  6.14e-7. Its integrated UV and optical differences from the old figure are
  +0.0061% and -0.0232%, respectively. Individual deep line-core relative
  differences are larger (up to 11.2%); they must not be called exact spectral
  identity. The fixed-atmosphere synthesis itself matches the paper to 8.66e-12.

Relative to the previous cleanup's *re-relaxed* 20000 K DAB spectrum, the
largest change is 1.64e-6 of the spectral peak (0.1072% in the most sensitive
relative-flux sample). The 9000 K peak-normalized change is below 4.7e-11.

## Next acceptance requirements

1. Demonstrate improvement on identical 8000 K pressure and optical grids,
   including the auxiliary compatibility, actual-gradient flux, temperature
   correction, local heating and independent spectral integral.
2. Test fresh hydrostatic/convective initialization and explicit mesh/domain
   refinement. A refined saved-model test does not establish cold-start ability.
3. Repeat working DA/DB/DAB/DZ controls before any default change. Do not enlarge
   the protected iteration allowances to disguise a slowdown or failure.
4. Address dense-He chemical ionization, molecular-ion charge balance and
   thermodynamic consistency independently of numerical conditioning; then
   validate dense-fluid opacity and refractive transfer against literature.
   Do not force a pure-He solution with added hydrogen or an electron floor.

For the prior CIA cutoff repair, shallow-boundary evidence, and unresolved
5000 K EOS issues, see `cool-helium-2026-09-04.md`. CIA extrapolation uncertainty
remains explicitly documented; smoothing an opacity boundary is not itself
a convergence solution.
