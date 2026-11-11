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

`cold/`, `montreal_h_radius_2026_10_27/cold/` and
`cool_h_chemistry_2026_11_10/cold/` (newest first) supply comparison outputs for
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
