# Approved regression spectra

These are synthetic OpenWD outputs, not observations or published model grids.
No stellar parameters, convergence tolerances or spectral tolerances were fitted.

`current/` holds the approved fixed-atmosphere spectrum for every case in
`tests/test_spectral_regressions.py`, `tests/test_daz_regressions.py`,
`tests/test_da_public_spectrum.py` (`public-*`) and `tests/test_dah_paper.py`
(`dah-default-*`, `dah-paper-*`). Each test resynthesizes the spectrum on an
unchanged frozen structure from `../spectral_regressions`, `../daz_regressions`,
`../da_cubic_regressions` or `../dah_paper`, with atmosphere iteration
forbidden, and compares it with the file here at the test's tolerance. Only
current default physics is checked; superseded approvals and the scripts and
manifests that recorded each physics change are in git history (removed
2026-10-29). The last approvals were the 2026-10-27 Montreal neutral-hydrogen
defaults (`da-3000`, `da-4000`, `da-5000`), combined with full hydrogen
dissolution on 2026-10-28 for every other hydrogen-bearing case; `db-22000`
dates from the 2026-10-01 flux-conserving numerics. On 2026-11-10 `da-3000`,
`da-4000` and `da-5000` were recaptured for the cool-hydrogen physics revision
(H3+ nuclear-spin normalization and 0 K dissociation energy, Kowalski & Saumon
H--H2 Lyalpha wing, H2- free-free, H2--H CIA) with
`research/capture_cool_h_chemistry_controls.py`; that revision moved no other
control. Separately, the 2026-11-10 audit fixes re-approved `da-5000`,
`da-20000`, `dab-9000`, `dab-20000`,
`dz-pg1225` and `dz-j0738` (resolved far wing of the hydrogen Stark x
neutral-impact Lorentz convolution, at most 0.26%) and `dah-default-j1018+0111`,
`-j1254+5612`, `-j1351+5419` and `-j2247+1456` (Rohrmann RWA n=7, l=6
photoionization coefficient, at most 8e-5);
`research/capture_audit_fix_controls.py` wrote them.

When the audit fixes were rebased onto the cool-hydrogen revision on
2026-11-11, `da-5000` was synthesized with both changes on its unchanged
frozen structure and wavelength grid. Its flux differs from the cool-hydrogen
control by at most 0.013303%; the stored pressure and column-mass arrays are
unchanged. The combined `da-3000` and `da-4000` spectra still pass their
existing strict comparisons, so those control files are retained byte for
byte. [Integration provenance](audit_cool_h_integration_2026_11_11.json)
records both predecessor hashes, the capture inputs and the band changes.
All comparison tolerances are unchanged.

On 2026-11-27 the helium synthesis gained the infrared He I lines
(`HELIUM_I_IMPACT_LINES`: 10830, 20581 and the n = 3 series, 8361 Å–7.4 µm).
`db-22000`, `dab-9000`, `dab-20000`, `dz-pg1225` and `dz-j0738` were
re-synthesized on their unchanged frozen structures and grids with
`research/capture_he_i_infrared_controls.py`, which first reproduces each
previous control at the test tolerance with the new lines switched off.
Below 7300 Å no pixel moves by more than 1.3e-5; the 7000–300000 Å band
changes by at most 0.69% (`db-22000`). The
[manifest](he_i_infrared_2026_11_27.json) records the hashes and band changes.
The DB, DAB and DZ structure solves include the same lines. The cold-start
canaries `db-22000` and `db-22000-standard` move by -0.65% in that band
(-0.69% from the line opacity, the rest backwarming), over their 3e-3 bound.
`he_i_infrared_2026_11_27/cold/` holds `db-10000`, `db-22000` and
`db-22000-standard` from verified local cold starts of the canary models on
the canary grid (`research/capture_he_i_infrared_cold_controls.py`). The
22,000 K gas pressure and column mass equal the previous references; their band
fluxes move by at most 4.6e-4 outside the infrared, and the He I 5876 core by
2.3%. `db-10000` also includes the column-mass Rosseland remesh of the same
date: its nodes moved onto the intended Rosseland mesh, and its UV, optical
and infrared band fluxes fall by 0.62%, 0.22% and 0.18% (the previous mesh
over-predicted the bolometric flux by 0.6%). Its nodes come from a macOS cold
start; if a Linux canary moves them by more than the 2e-5 node tolerance, take
the gas pressure and column mass from that certified candidate, as on
2026-11-19.

`helium_line_cores_2026_11_27/cold/db-10000.npz` follows the resolved He II and
He I resonance Doppler cores in the LTE helium structure grid. The structure is
unchanged below τ_R ≈ 10⁻¹¹ (temperature within 0.05%, τ_R(m) within 1% for
m ≥ 10⁻⁶ g cm⁻²). The outermost layers (τ_R 10⁻¹⁴–10⁻¹¹) cool by up to 4.6%, which
moves the top τ_R that the Rosseland remesh spreads its target depths from.
So the remeshed nodes shift by up to 69% in column mass while the spectrum
moves by at most 6.1e-4 (900 Å) and 7e-5 in the optical. Gas pressure, column
mass and flux come from the certified Linux CI candidate
(`research/capture_helium_line_core_cold_controls.py`;
[manifest](helium_line_cores_2026_11_27/cold/manifest.json)).

The `dab-20000` cold check (`research/validate_public_cold_start.py`) is
also affected by the infrared He I lines. On the branch that added them, its
7000–300000 Å band fell by 0.52%, over the 0.3% bound. Re-synthesizing that
same structure without the lines gives −0.513%, so the drop is the line
opacity itself.

On the same date, the mixed H/He structure started sampling each Paschen
and Brackett line out to ±300 Å, as the hydrogen solver does; before, it used
five points within 5 Å. For this DBA, that moves the structure temperature
by at most 9.5e-4 and the band fluxes by at most 1.6e-6.

`paschen_brackett_wings_2026_11_27/cold/dab-20000.npz` is a verified local
cold start of the CI command, captured by
`research/capture_paschen_brackett_cold_controls.py`. Its gas pressure and
column mass equal the previous reference to 1.2e-6. The validator now reads
the newest reviewed folder first, as the protected canaries do.
[Provenance](paschen_brackett_wings_2026_11_27/cold/manifest.json).

On 2026-11-11 the hydrogen Stark-table fix (the log-bilinear interpolation
in electron density and temperature is rescaled to the bilinear weighting of
the corner-profile areas; midway between density nodes it had kept only
86-88% of the Lyman-alpha and Balmer area at log ne <= 13) re-approved
`da-3000`, `da-4000`, `da-5000`, `da-20000`, `db-22000`, `dab-9000`,
`dab-20000`, `dz-pg1225`, `dz-j0738` and `public-da-12000-public`. Largest
changes in significant flux: 5.3% in the trace-hydrogen Lyalpha wing of
`dz-pg1225`, 1.1-2.8% at Lyalpha in the warm DA/DAB/DZ cases, 0.52% at the
`da-5000` Halpha core; integrated flux at most 4e-4.
`research/capture_stark_area_controls.py` wrote them (and measures without
`--write`). The DAZ and DAH controls stay within their tolerances and are
retained byte for byte.
In the first full validation of that fix only the `da-5000` cold canary moved
beyond tolerance: its structure matched the previous reviewed reference to
4e-9 and its Halpha core deepened by 0.37% (3.27e-3 on the energy scale against
the 3e-3 bound). `stark_area_2026_11_11/cold/da-5000.npz` is the CI cold start,
verified and re-synthesized by `research/capture_stark_area_cold_controls.py`;
its manifest records the source certificate and the change.

On 2026-11-18 two cool-metal defaults changed. DZConfig now includes HeH+
(H+ + He <-> HeH+, Pachucki & Komasa 2012 D0 and the ExoMol partition
function) in the trace-hydrogen ionization balance. DZConfig and DAZConfig
also emit Ca II H&K complete-redistribution light as
kappa_L S_L B_nu(nu)/B_nu(nu0), which keeps Kirchhoff's law across the broadened
wings instead of applying the line-centre source function at every frequency.
That re-approved `dz-pg1225` (1618 of 2527 significant pixels moved by more
than 0.1%; at most -0.60%, at 1221 A; integrated about -0.1%) and `dz-j0738`
(at most 0.04%). `research/capture_heh_kirchhoff_default_controls.py` wrote
them after first showing that both options off reproduce the previous
controls. The DAZ controls move by at most 0.07% and stay within tolerance,
so they are retained byte for byte. The `dz-pg1225` cold canary passes with
the new defaults against its existing reference.
[Provenance](heh_kirchhoff_2026_11_18.json) records the hashes and band changes.

On 2026-11-19 two shared defaults changed. The homogeneous atomic and
molecular H/He EOS include HeH+ (as the DZ trace-hydrogen closure already
did), and He- free-free below John's (1994) 0.5063-micron edge is John's edge
value times the Carbon et al. (1969)/John (1968) wavelength shape
(`john1968-shape`). The previous `automatic` choice switched to the Carbon fit
itself at the edge, which is 3-24% higher there, and left a flux step at
5063 A in cool helium-rich models (0.71% in `dab-9000`). DZ and DQ select
`partial-wave` explicitly and are unchanged. That re-approved `dab-9000` (at
most +0.51%, at 5062 A; integrated +0.4% below 5063 A and -0.1 to -0.2%
above) and `dab-20000` (at most 1e-5, from HeH+).
`research/capture_heh_mixed_heminus_controls.py` wrote them after showing
that the previous defaults reproduce the previous controls. Every other
fixed-spectrum control stays within tolerance and is retained byte for byte.
On the frozen `dab-9000` structure the new spectrum is within 0.2-1.6% of the
Montreal 9000 K, log g 8, log N(H)/N(He) = -2 model from 2000 A to 1 micron,
with no step at 5063 A. [Provenance](heh_mixed_heminus_2026_11_19.json).

For the 10,000 K pure-helium DB cold start, the new He⁻ opacity moves the
hydrostatic pressure and column mass by up to 0.845%. Recomputing the
hydrostatic seed with each prescription reproduces its respective CI grid.
The new comparison file takes those two arrays from the certified cold start
and retains the previous wavelength and flux arrays. The maximum spectral
change on the stellar energy scale is 0.111%, within the existing 0.3% limit.
[Cold-start provenance](heh_mixed_heminus_2026_11_19/cold/manifest.json).

`helium_line_cores_2026_11_27/cold/`, `paschen_brackett_wings_2026_11_27/cold/` (the `dab-20000` cold check only), `he_i_infrared_2026_11_27/cold/`, `heh_mixed_heminus_2026_11_19/cold/`, `stark_area_2026_11_11/cold/`, `cool_h_chemistry_2026_11_10/cold/`,
`montreal_h_radius_2026_10_27/cold/` and `cold/` (newest first) supply comparison outputs for
the protected cold-start canaries. Those tests always calculate a fresh
atmosphere before reading them and still require every independent equilibrium
check; the Montreal copies replace the DA references moved by the 2026-10-27
hydrogen defaults. Their manifests record the source certificates.
`research/capture_release_controls.py` and
`research/capture_montreal_h_radius_cold_controls.py` wrote them; neither is
run by tests or CI.

An intentional physics change that moves a spectrum needs an explicit
physics/numerics review and user approval, not just a failing test. Replace
the affected files in `current/` with spectra synthesized on the same frozen
structures and wavelength grids, and record the reason and size of the change
in the pull request.
