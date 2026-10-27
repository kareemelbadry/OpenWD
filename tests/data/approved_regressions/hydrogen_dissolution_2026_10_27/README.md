# Full hydrogen dissolved fraction, 2026-10-27

These fixed-atmosphere outputs follow the reviewed removal of the SYNSPEC/TLUSTY
pseudo-continuum cutoff (physics revision tag
`hydrogen-full-dissolution-2026-10-27` for DA, DAB, DAO, DAH, DZ and DAZ).

Previously the Daeppen--Anderson--Mihalas pseudo-continuum kept the full
dissolved fraction `1 - w(n*)/w(i)` only blueward of the transition to `i + 3`
and only its neutral-perturber part from there to the first series member, in
both `opacity.hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient`
and the magnetic `magnetic_continuum._dissolved_rwa_extinction`. For the Balmer
series the cut fell exactly on H-gamma. The bundled Tremblay--Bergeron Balmer,
Paschen and Brackett profiles include the occupation-probability reduction of
every line (only the first three Lyman profiles are ideal), so their dissolved
strength belongs in the pseudo-continuum; the full fraction is now kept up to
the first member. The Lyman continuation is still cut at 925 A.

Evidence: on converged OpenWD DA structures at 12000 K (log g 7.5, 8.0, 8.5) and
20000 K, the old cut made the flux between H-gamma and H-beta 1.0--2.4% higher
than both the Koester (SVO) and Tremblay DA grids, while the full fraction
agrees with both within about 0.5%; 8000 K is unaffected. On the public
12000 K control the 4420--4720 A mean ratio to Koester goes from 1.0200 to
1.0005. In the zero-field paper prescription of the high-field DAHs the cut was
not hidden by a line core and left a single-pixel jump of 1.5--7.4% at
4341.7 A; that jump is removed.

Every original structure, configuration, wavelength grid and comparison
tolerance is retained. Existing approved references and the frozen DAH paper
spectra are untouched. Under `research/hydrogen_dissolution_control_legacy.py`
every case first recovers its previous approved spectrum (and the DAH paper
spectra) at its own test tolerance before the new output is recorded.

| Fixture | Previous control | Max change (significant flux) | Flux-integral change |
| --- | --- | --- | --- |
| `da-20000` | `fixed/da-20000` | 1.305% at 4452 A | -0.021% |
| `dab-9000` | `stark_density_floor_2026_10_05` | 0.059% at 4408 A | -0.004% |
| `dab-20000` | `fixed/dab-20000` | 0.097% at 4622 A | -0.002% |
| `dz-pg1225` | `metal_populations_2026_10_24` | 0.004% at 4366 A | -0.0002% |
| `dz-j0738` | `metal_populations_2026_10_24` | <0.001% | <0.0001% |
| `public-da-12000-public` | `da_cubic_regressions` | 2.425% at 4483 A | -0.132% |
| `public-da-20000` | `da_cubic_regressions` | 1.298% at 4455 A | -0.001% |
| `daz-g149_28` | `independent_kurucz_2026_10_14` | 0.286% at 4434 A | -0.042% |
| `daz-galex1931` | `independent_kurucz_2026_10_14` | 1.081% at 4447 A | -0.135% |
| `dah-default-*` (8) | `dah_rwa_default_2026_10_25` | 0.38--2.92% | -0.13 to -1.05% |
| `dah-paper-*` (8) | `../dah_paper` paper spectra | 1.29--6.68% | -0.15 to -0.93% |

The manifest records per-case input/output hashes, request fingerprints,
band-integrated changes and the legacy recovery. Capture is offline via
`research/capture_hydrogen_dissolution_controls.py`; tests never run it.
No atmosphere is reconverged or certified by this fixed-state capture; the
protected cold canaries remain required. The magnetic (RWA) continuation has
no external grid comparison; it is changed for consistency with the zero-field
continuum it reduces to.
