# Solver consistency tranche: 2026-09-04

> Research record: this page describes work at the time it was written.
> For current usage and status, see the [user guide](../../getting-started.md)
> and [tested points](../../tested-temperature-ranges.md).

Local work on the authoritative OpenWD checkout, starting from `dcd3037` and
the previous turn's uncommitted scattering/cool-DA work. No GitHub push or new
branch is part of this tranche.

## Findings and implementation

1. Replaced fixed four-sweep coherent scattering and the truncated scattering
   tangent with direct coupled angular-block Feautrier solves. The tangent
   includes optical-depth motion and implicit scattering feedback. Independent
   fixed-source transfer and finite-difference tests check the implementation.
   Batched vector solves use a NumPy-1/2-compatible RHS shape.
2. The ML2 Jacobian had only its temperature-gradient derivative. It now also
   differentiates density, opacity, heat capacity, thermal expansion, and the
   adiabatic gradient. Material coefficients are differenced locally using
   the existing hotter state; analytic differentiation of the ML2 cubic avoids
   differencing across a tiny superadiabatic excess. Two interleaved endpoint
   assemblies recover both nodes of each interface without a separate
   radiative-transfer solve per temperature variable.
3. The old continuum Rosseland quadrature moved with temperature. Near the
   stalled 22,000 K DB, a ~0.04 K perturbation moved a sample through an opacity
   edge, changing kappa_R by about 8% and total flux by about 2%. Newton cannot
   reliably linearize such a numerical jump. Production structure callbacks
   now integrate their existing continuum opacity on a fixed spectral grid;
   the Planck weights and actual opacities still change with the material state.
   DZ retains its existing explicit metal-containing extinction prescription.
4. Removed the unvalidated CONREF-like trial-repair prototype and its iteration,
   flux-fraction, and residual-window tuning. Fixing inconsistent equations and
   derivatives came first. A future explicit convective correction must be a
   separately tested global safeguard, not a special case for the DB canary.

No new effective-temperature, gravity, or object-specific switches were added.
The earlier <5000 K DA *initialization* policy is retained, not newly certified
as the best general policy. Its replacement by a local stiffness criterion is
still separate work. Final convergence continues to require real ML2 plus
radiative flux at every depth and the unchanged temperature-step tolerances.
The model-request physics revision has been changed so old checkpoints cannot
silently masquerade as exact current-physics restarts.

## Controlled checks

The original stalled DB's full-flux Jacobian had a column error of about 1081
for a finite-difference derivative of about 1035, including the wrong sign at
one interface. After the grid discontinuity was removed, the remaining missing
material term was about 10.1 for a derivative of about 63.5. Including that term
reduced the corresponding column's maximum error to about 0.0031 in the
fixed-extinction-grid diagnostic. Synthetic independent full-residual tests
now cover both absorption-only and scattering-dominated cases, including
temperature-dependent material coefficients and a sharp spectral opacity edge.

An initial experiment used total extinction for every composition's Rosseland
mean. That also added line opacity to DA/DB/DAB convection, changing a physical
prescription, and was **not retained**. The final implementation separates
fixed quadrature from the choice of opacity processes. All canaries were rerun
after that correction.

Seven protected no-checkpoint/no-fallback cold starts pass unchanged assertions:

| Model | Layers | Last reported maximum physical flux error |
| --- | ---: | ---: |
| DB 10,000 K | 80 | 3.24e-8 |
| DB 22,000 K | 80 | 5.22e-7 |
| DB 22,000 K standard | 40 | 5.24e-7 |
| DA 5,000 K | 100 | 2.08e-6 |
| DA 20,000 K | 100 | 2.87e-6 |
| DA 3,000 K | 100 | 1.83e-3 |
| DA 4,000 K | 100 | 4.29e-6 |

The final ordinary suite passes: 307 passed, 5 optional-data skips, with the
seven canaries excluded from that count and exercised separately above.

These are protected points at log g=8, not a general grid-completeness claim.
The 3000 K case passes its existing 0.2% flux tolerance but has less margin
than the warmer cases. No controlled speed claim is made from these parallel
runs. Source-solve smoke tests also pass on NumPy 2.1.2/Python 3.13; this is
not a substitute for the full supported-Python CI matrix.

## Paper regressions in the adjacent validation workspace

Reproduction scripts use the production checkout, not recovered/quarantined
source. Fixed-atmosphere synthesis and current-equation checks are distinguished:

- DAB 9000 K: the saved paper structure passes the current equations unchanged,
  maximum all-depth flux error 8.87e-5. Its UV spectrum agrees to roundoff.
- DAB 20,000 K: re-relaxes in three iterations, maximum flux error 6.23e-7 and
  final |dlnT| 1.50e-4. The integrated 1200--3000 A flux changes by +0.0061%;
  integrated 3500--8000 A flux changes by -0.0232%. RMS differences normalized
  to each band's peak are 0.0173% and 0.0179%. It is **not identical**: the dark
  Ly-alpha core at 1215.76 A changes by -5.44% relative to its own low flux.
- PG 1225-079: fixed-atmosphere spectrum agrees to 4.7e-15 maximum relative;
  current-equation all-depth flux check on its saved grid is 1.68e-3 (<3e-3).
- SDSS J0738+1835: fixed-atmosphere spectrum agrees to 4.4e-16 maximum relative;
  current-equation all-depth flux check is 2.23e-3 (<3e-3).

The DZ checks do not claim new cold-start convergence or re-relaxation. Some
archive files lack the current provenance metadata and correctly still issue
the exploratory-spectrum warning; that warning was not suppressed or replaced
by a fabricated current fingerprint.

The 20,000 K DAB change was isolated to quadrature, not scattering. On its paper
state, old and exact scattering with the old quadrature both give maximum flux
error 2.21e-4; fixed continuum quadrature gives 1.28e-2 before correction. At the
affected depth, the old 120-point kappa_R is 87.6529, a 15,360-point independent
quadrature gives 84.3720, and the new fixed-grid value is 84.3626. Neighboring
depths show the same agreement with refinement. Do not tune a model-specific
opacity correction to recover the old undersampled value.

Research scripts: `scripts/check_dab_solver_regression.py` and
`scripts/check_dz_fixed_atmosphere_regression.py --check-current-flux` in the
adjacent validation workspace. Re-relaxed DAB outputs are in
`results/solver-consistency-20260904-dab-continuum/{9000,20000}` there.

## Remaining work

- Replace global cool-star seed policy only after transition/gravity/composition
  tests justify a local criterion. Do not confuse seed selection with a change
  to the final equations.
- Consider an explicit convection trial correction after these consistency
  fixes, with full residual re-evaluation and documented acceptance rules.
- Extend grid-refinement, surface-layer conditioning, and cool-DB validation.
  Passing the protected set does not establish convergence throughout parameter
  space, nor validate the physical CIA/dense-EOS approximations there.
