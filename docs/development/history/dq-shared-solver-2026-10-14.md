# DQ audit and shared-solver protocol (2026-10-14)

## Why

DQ was the slowest and least reliable OpenWD module. J1311 had never
converged from a cold start (stalls after 0.5–3 h), J1803 ran out of memory,
J0804 depended on its seed, and production used ~60 `_dq` modules in which a
stack of patched Newton, coordinate, conditioning and proposal layers sat on
top of the physics. The fixed-parameter SDSS fits of the cooler stars were
poor (J1803 blue flux 0.52× observed, J0804 0.70×, J1311 optical 1.3–1.5×).

## Root causes found

1. **He-REOS.3 interpolation.** Isotherms at 3000/6000/10000/20000 K; the
   specific energy was interpolated linearly in log T. In the dilute limit
   this gives c_P/c_P,ideal = 0.72–1.44 and ∇ad = 0.34–0.48 (exact: 0.400),
   jumping at every isotherm (0.46 → 0.35 across 6000 K at 10⁸ dyn cm⁻²; Q
   0.41 → 0.51 at 10¹¹). These jumps produced the wrong-sign Jacobian entries
   recorded in the J1311 stall audit; the numerical tower (EOS-corner
   proposals, knot searches, secant windows) was compensating for them.
   *Fix:* interpolate only the non-ideal excess (ln ρ/ρ_ideal, u − 3kT/2m)
   with PCHIP in log P and a monotone C¹ Hermite in log T; analytic c_P, Q,
   ∇ad. Every tabulated point is reproduced; the ideal gas is exact.

2. **Carbon line wings.** At n_He ≈ 10²²–10²³ cm⁻³ the Unsöld He impact
   widths of C I lines exceed their Weisskopf detuning (ratio 0.01–0.5), and
   symmetric Lorentz wings of the far-UV lines — dominated by the 2p nd
   Rydberg series (n* = 7–10) near 1110–1130 Å — set the optical continuum
   of the cool and carbon-poor stars: C-line/He⁻ ff opacity at 4000 Å and
   τR ≈ 0.1–1 was 100–450 (J1803), 3–18 (J0804), 0.05–0.3 (J1311).
   Neutral-He occupation probabilities could not remove them (the existing
   option only counts bulk *metal* perturbers). *Fix:* the red wing of every
   DQ carbon line is bounded by the ground-state Boltzmann factor
   exp(−h|Δν|/kT): a red detuning requires a pair configuration whose
   ground-state energy exceeds the upper-state energy by h|Δν|, and C*–He
   upper states are repulsive (Allard & Spiegelman 2024,
   doi:10.1093/mnras/stae477). Optical contamination falls below 10⁻³ of He⁻
   ff while UV line cores keep ~90% of their strength. No multiplier.

3. **Numerics.** With 1 and 2 fixed, the unmodified shared solver with the
   cool-DB domain recipe (τR,min = 10⁻¹², deepen ×4 on any screening failure,
   restart) converges the DQ cases. One opt-in progress policy was added: the
   formal-flux phase hands over to the energy phases once flux is balanced
   but local energy is not (two genuine accepted updates), instead of chasing
   stationarity in optically thin layers.

## Code

`_dq` went from 60 to 27 modules (−4358/+219 lines in the cleanup commit).
`runtime.ProductionDQMaterial` = refractive transfer + cell-integrated Swan +
C–A + bounded carbon/He lines + base chemistry; `runtime.run_cold` keeps the
public worker contract (five gates + independent 218,520-point synthesis,
0.2% bolometric gate, source and data integrity).

## Validation

See the table at the end (cold starts, fixed published parameters, 40 initial
layers, one job at a time on an M-series laptop).

## Open issues

* Dense He⁻ ff correction: with the same Iglesias et al. (2002) kernel and a
  Ross–Young PY S(k), δ_ff(u) rises only ×1.3 from u = 1 to 20 at 3240 K,
  1.23 g cm⁻³, versus ×2.9 in their Fig. 2; their I₀(k) peaks (Fig. 3) lie at
  2× our momentum transfer. Matters for ρ ≳ 0.3 g cm⁻³ (J0804, J1803).
* Swan band shapes: model flux just redward of each band head is too high
  (violet-shaded tails too weak); 3900–4300 Å too bright in J1311.
* Thermal conditioning can oscillate for ~15 sweeps before Newton energy
  completion (J1311: 25 of 45 minutes); a stagnation handoff would help.
* Fixed-spectrum regressions (`tests/test_dq_spectral_regression.py`) and the
  `dq-j1235` cold canary need user-approved recapture: the physics changed.
