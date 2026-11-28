# HD 4539 metal-free hybrid sdB controls

These are new controls for the public sdB adapter, using the current main
physics after PRs #52 and #54. They do not replace a historical reference.

The separate research recipe constructs a fresh standard-resolution LTE DAB
structure at 23,200 K, log g 5.20, log10 N(He)/N(H) = -2.27, then solves
16-level H I / 14-term He I populations with MALI and dominant-state helium
conservation on that fixed structure. The inputs and actual convergence are
recorded in `manifest.json`, together with the driver and control checksums.

- `hd4539-hhe.npz` contains only wavelength and absolute surface flux. The
  public cold test loads it after computing a fresh model and compares the
  complete 3000–9200 Angstrom grid at 0.05 Angstrom spacing to 0.1%, with no
  normalization or fitted scale.
- `hd4539-fixed-state.npz` contains the research atmosphere and populations
  for the separate fixed-spectrum test, whose relative flux tolerance is
  1e-7 and which forbids structure and population iterations. This state is
  never read by the cold test.

On 2026-11-12 both controls were rebuilt with the recipe below for the
hydrogen Stark-table fix (the log-bilinear interpolation in electron density
and temperature is rescaled to the bilinear weighting of the corner-profile
areas). The NLTE populations converged in 295 iterations (change 9.7e-5).
Against the previous control the H-alpha core is 0.67% higher, H-beta 0.48%
and the continuum 0.064%: the LTE structure and the re-solved populations
both respond to the restored line area. A fresh public cold start on CI
agrees with the new control to 1e-6 at every wavelength it reported. An
interim control that only re-synthesized the old fixed state (keeping the old
populations) missed that cold start by 1.15% at H-alpha and was withdrawn.
`manifest.json` records the previous checksums under `supersedes`.
Tolerances are unchanged.

On 2026-11-19 HeH+ joined the mixed H/He EOS. It moves the LTE populations of
the fixed state by 3.1e-9 (the test compares them at 1e-10), so
`research/capture_heh_mixed_heminus_controls.py` recomputed them with the new
EOS and kept the stored NLTE departure coefficients. The emergent flux changes
by 4.6e-10 and `hd4539-hhe.npz` is retained byte for byte. `manifest.json`
records the change under `amendments`. The cold-start control is unaffected
at its 0.1% tolerance.

On 2026-11-27 both controls were rebuilt with the recipe below. The mixed
H/He (DAB) structure now samples each Paschen and Brackett line out to
±300 Å, as the hydrogen solver does. It used to sample only five points
within 5 Å of each centre, on a continuum grid 1.4% apart, and that left the
upper layers too cool. The new structure is 1.5% hotter at
τ_R = 10⁻⁶–10⁻³ and 0.17% hotter at 10⁻². The structure also includes the
infrared He I lines of the DB module, which change its temperature by less
than 2e-4. The populations converged in 264 iterations (change 7.4e-5).
Against the previous control, the H-alpha core is 1.07% lower, H-beta 0.79%
lower, and H-delta and H-epsilon 0.72–0.74% higher. The continuum moves by
at most 6e-4 (at 3200 Å).

A fresh public cold start agrees with the new control to 3e-6. The branch
before the sampling fix (6e781c7) missed the previous control by 0.85% at
H-alpha. Its infrared He I wing grids also happened to sample nearby
hydrogen lines, and it moved the same amount with the He I line opacity set
to zero. `manifest.json` records the previous checksums and their history
under `supersedes`. `capture_sdb_cold_control.py` now also requires the
recorded structure abundances to be empty, because `sdb_lte_hhe.py` has
stored them since the metal-blanketed recipe.

For an explicitly reviewed new control, use new output directories:

```bash
python research/sdb_lte_hhe.py --teff 23200 --logg 5.20 --log-he-h -2.27 \
    --output results/sdb-control-lte
python research/sdb_hybrid_nlte.py results/sdb-control-lte \
    --hydrogen-levels 16 --accelerated-lambda --population-iterations 600 \
    --output results/sdb-control-hybrid
python research/capture_sdb_cold_control.py results/sdb-control-lte \
    results/sdb-control-hybrid results/new-sdb-control
```

The capture script requires converged source calculations and refuses an
existing output directory. Tests never regenerate the controls. These checks
cover the metal-free HD 4539 recipe; they do not add observational validation,
qualify the whole sdB grid or establish coupled NLTE radiative equilibrium.
