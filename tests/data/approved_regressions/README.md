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
dates from the 2026-10-01 flux-conserving numerics.

`cold/` and `montreal_h_radius_2026_10_27/cold/` supply comparison outputs for
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
