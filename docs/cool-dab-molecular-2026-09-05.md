# Molecular physics in the cool mixed H/He model

## Finding and status

The 5000 K DAB/DBA comparison used an **atomic** H/He atmosphere. Its excellent
numerical flux balance was not a test of physical completeness. The mixed
branch omitted H2 chemical equilibrium, H2-He/H2-H2 collision-induced
absorption (CIA), and the neutral-collision Lyalpha red wing already available
to the DA branch. Adding a CIA opacity to the old atomic population is not a
consistent remedy: molecular formation changes charge, density, reaction
enthalpy, and convection as well as opacity.

These ingredients are now implemented as an **explicit, experimental molecular
option**. Established production defaults remain atomic until the new
atmospheres pass validation. No temperature cutoff, fitted opacity multiplier,
electron floor, spectrum renormalization, failed-state replacement, recovered
solver, or GitHub push was introduced.

The new physics substantially improves the 5000 K spectral shape. It does
**not yet yield a validated converged molecular atmosphere**. Keep that
distinction in any use of the new spectra.

## Implemented physics

- `_mixed_molecules.py`: H I, H+, H2, H2+, H-, H3+, He I/II/III and electrons.
  Three log-density unknowns solve nuclei ratio, particle pressure and charge
  conservation together with mass action. Log-sum arithmetic avoids capped
  molecular populations. The atomic solution supplies an initial guess only;
  failure raises instead of returning atomic chemistry. An analytic H/H+/H2
  balance improves the starting guess without changing the final equations.
- Molecular/atomic binding, ionization, excitation and rotational/vibrational
  energies enter the same mixture enthalpy used for Cp, Q and the adiabatic
  gradient. The existing HM atomic occupation prescription is retained.
- `_mixed_cia.py`: strict Abel/HITRAN H2-He reader, interpolation, and
  `kappa = coefficient_cm5 * n(H2) * n(He I) / rho`. Native data are in
  cm5 molecule^-2, not cm^-1 amagat^-2. There is no extra factor of two or
  stimulated-emission multiplier. H2-H2 uses the existing Borysow data.
- H-H and H-H2 Lyalpha wings use the existing DA implementation, with its
  complementary Allard coverage to avoid double counting. Existing H2/H2+
  photoabsorption is enabled on the solved molecular populations.
- Structure absorption, Rosseland means, thermodynamics and synthesis receive
  the same physics object. Changing checkpoint chemistry invalidates its
  convergence/fingerprint. A molecular spectrum without matching molecular
  physics now raises. External H2-He file identity is included in molecular
  request fingerprints; file size/mtime invalidate the parsed-data cache.

The H2-He table covers 200–9900 K and 20–20000 cm^-1. Temperature interpolation
holds at its endpoints; this is a limitation, not validated hot-gas CIA.
Only declining terminal log slopes are continued above the wavenumber edge;
unsupported rising tails are zero. Below the lowest frequency the coefficient
is zero. These policies are explicit and covered by tests. The H2-H2 6068 Å
cutoff already had its declining-tail repair; that was not the principal
omission in the atomic DAB comparison.

Sources: [HITRAN CIA data and conventions](https://hitran.org/cia/), and
[Blouin et al. 2018, sections II–III](https://arxiv.org/html/1807.06616v1).
The latter distinguishes bulk nonideal EOS, chemistry, and opacity effects.
The implementation here is **not** a dense-fluid free-energy EOS. Nonideal H2
dissociation, pressure-distorted H2-He CIA, HeH+, and He2+ charge chemistry are
not newly implemented. Dense CIA distortion can matter above approximately
0.1 g cm^-3; the 5000 K test reaches that regime. Do not fit away residual
differences to the legacy Montreal grid with an arbitrary density correction.

## Measurements

All DAB tests use log g=8 and log10(N_H/N_He)=-2.

At the old atomic photosphere, re-solving chemistry at unchanged pressure and
temperature puts about 76% of the hydrogen nuclei into H2 at 5000 K, versus
about 0.6% at 8000 K. Fixed-P/T ablations show that CIA suppresses 5000 K
1–5 micron flux by about 22% relative to molecular chemistry alone. Neutral
Lyalpha absorption supplies the missing UV suppression. These ablations do
not conserve the old atmosphere's stellar flux and are not final atmospheres.

The closest completed molecular trial is the 80-layer cold-start calculation
followed by 15 explicit full-physics correction attempts. Its separate
all-depth flux error is 0.0353457, local relative energy error 0.0589638,
last maximum dlnT 0.0008255, and independent spectrum integral/Fstar 0.971225.
It **fails** convergence. The bottom absorption escape bound is 1.03e-15, so
that trial's error is not explained by a transparent lower boundary.

| Model/reference integrated band flux | Old atomic model | Molecular trial, not converged |
| --- | ---: | ---: |
| UV, 1500–3000 Å | 4.8932 | 0.8797 |
| Optical, 3500–9000 Å | 0.8119 | 0.9348 |
| IR, 1–5 microns | 1.5830 | 1.1734 |

No flux scale is fitted. The reference uses the same exact-temperature
Montreal grid as the previous comparison, converted from Hnu to Flambda and
from air to vacuum wavelength. The historical **reader only** is imported
explicitly; no historical solver is used. Its sampled spectrum integrates to
0.982723 Fstar, which is recorded rather than normalized away.

## Numerical experiments and limits

Re-relaxing the old atomic 162-layer 5000 K solution stalls at flux error
0.6786 and local error 0.4143 after molecular physics changes the adiabat.
The analogous 8000 K restart stalls near 0.0182 and 0.0270. A molecular cold
start is not secretly replaced by either checkpoint: it was run separately,
and remains explicitly identified as the source of the better 5000 K trial.

Finite-difference tests on the stalled molecular state agree with the new
direct radiation/material tangent: the thin-cell derivative error is about
2e-7 absolute for a measured derivative of 1.53; a stiff convective node
agrees to about 4.4e-7 relative at dlnT=1e-8. The cheap inner ML2 model also
matches its own derivatives. This does not establish accuracy of a finite
temperature correction: the inner model still linearizes material quantities,
particularly nabla_ad, and even a small curvature error matters near an
efficient-convection boundary.

Research-only tests tried SVD-rotated temperature proposals, disabled cost-
change stopping, inverse ML2 compatibility coordinates, and alternative
locally scaled energy rows. None is enabled in the production solver.
Rotation helped the cold-start structure initially but did not remove its
remaining stall. Alternative row representations could lower their numerical
merit while worsening actual flux; the independent physical gate correctly
rejects these as converged models. Do not adopt those formulations based on
their optimizer status. A finer-grid diagnostic is recorded separately.

The 80-to-159 layer refinement also stalls: actual all-depth flux error
0.0371019 and local relative energy error 0.0627007, despite a correction
collapsing below 1e-11. It does not establish mesh convergence. Its source is
the explicitly unconverged molecular 80-layer trial, not the old atomic model.
This supports treating the remaining problem as a nonlinear convective
correction issue rather than assuming another doubling of the grid solves it.

## Reproduction and regression protection

Public opt-in (requires the external CIA file; missing data raise):

```python
from wd_spectra.models import DABConfig, compute_dab

result = compute_dab(DABConfig(
    effective_temperature=5000,
    logg=8,
    log_hydrogen_to_helium=-2,
    include_molecules=True,
    h2_he_cia_path="/absolute/path/to/H2-He_2011.cia",
    quality="production",
))
```

This is an experimental request, **not a promise of convergence**. The normal
convergence warning remains active. Explicit tests must additionally require
all-depth flux balance, local heating balance, a measured small temperature
correction, adequate lower-boundary absorption, and independent spectral/grid
convergence.

Validation completed after implementation:

- 412 regular tests passed; five optional-data tests skipped; seven canaries
  excluded from that command and run separately.
- All seven DA/DB canaries passed: DA 3000, 4000, 5000, 20000 K; DB 10000 K
  (80 layers), DB 22000 K (80 and 40 layers).
- 9000/20000 K DAB paper regressions retain the preceding production behavior.
  The fixed paper spectra agree to maximum relative differences 3.06e-9 and
  8.66e-12 respectively. The 20000 K re-relaxation takes two applied corrections
  plus one small unapplied proposal; the 9000 K saved state passes the existing
  initial-state check. Those are warm/paper regressions, not cold-start claims.
- Ten new molecular tests cover 72 thermodynamic states, conservation, mass
  action, reaction enthalpy, scalar/batch agreement, CIA units/interpolation,
  chemistry mismatch, missing data and external-data fingerprint identity.
- 28 focused molecular/research-proposal tests passed after formatting.

Research workspace paths (not copies of production source):

- `scripts/run_molecular_dab_experiment.py`: explicit molecular transport runner.
- `scripts/diagnose_cool_dab_molecular_opacity.py`: fixed-state ablations.
- `scripts/plot_cool_dab_molecular_diagnosis.py`: the labelled comparison plot.
- `results/cool-dab-molecular-fixed-state-20260905`: ablation spectra/metrics.
- `results/cool-dab-molecular-cold-20260905`: separate molecular cold start.
- `results/cool-dab-molecular-cold-correction-20260905`: plotted failed trial.
- `results/cool-dab-molecular-comparison-20260905`: PNG/SVG and numerical ratios.
- `results/cool-dab-molecular-refined-20260905`: separate 80-to-159 mesh test.
- `results/dab-molecular-default-guards-20260905`: protected DAB spectra.

The next numerical priority is a material/gradient-consistent finite convective
correction, rather than larger iteration budgets or a Teff-specific damping
constant. Dense-physics refinements require a converged, mesh-checked baseline
before spectral agreement can tell us which additional terms are needed.
