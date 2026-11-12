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

On 2026-11-11 the hydrogen Stark-table fix (the log-bilinear interpolation
in electron density and temperature is rescaled to the bilinear weighting of
the corner-profile areas) re-approved `hd4539-hhe.npz`. Its flux was
re-synthesized with `sdb._emergent_flux` on the unchanged
`hd4539-fixed-state.npz` and wavelength grid; the research drivers were not
re-run. The largest change is -1.16% at the H-alpha core (H-beta -0.66%,
H-gamma -0.34%); integrated flux changes by -1.6e-5. `manifest.json` records
the previous checksum under `control_reapprovals`. Tolerances are unchanged.

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
