# Hot-subdwarf (sdB) H/He models

[Contributor guide](README.md) · [sdB model guide](../models/sdB.md)

The [sdB model guide](../models/sdB.md) tells users how to compute a model
and shows the validation figure. This note describes the solver options and
the research drivers for developers.

## Method

The sdB models use the hybrid LTE/NLTE method for B stars and sdBs
(Przybilla et al. 2006; Irrgang et al. 2018):

1. **LTE H/He structure.** A public DAB cold start (`compute_dab`, LTE, no
   metals) at the requested Teff, log g and He/H.
2. **NLTE H/He populations.** The restricted hot H/He atom (`HotNLTEModel`)
   solves the statistical equilibrium on that fixed temperature and
   gas-pressure structure, with the formal-solution radiation field.
3. **Spectrum.** The emergent spectrum uses the NLTE populations.

`SdBConfig` and `compute_sdb` (in `src/wd_spectra/models/sdb.py`) implement
this method. `run_model` accepts `SdBConfig`. The research drivers
`research/sdb_lte_hhe.py` and `research/sdb_hybrid_nlte.py` contain the same
recipe. `sdb_hybrid_nlte.py` also writes the same synthesis with LTE
populations (`spectrum-lte.npz`) to isolate the NLTE effect.

The self-consistent DO/DAO solver (`solve_hot_nlte_atmosphere`) is not used.
At sdB densities, its full-NLTE stage is nearly singular in the outer layers
and does not converge.

## Solver options for sdBs

`compute_sdb` sets these options. Their defaults elsewhere do not change.

- `HotNLTEModel.solve_populations(..., accelerated_lambda=True)`: multilevel
  accelerated lambda iteration (`_hot_mali.py`) in the Rybicki & Hummer
  (1991) form. It changes the path, not the solution. Without it, the
  fixed-point iteration through the optically thick H/He lines seems to
  converge but stays far from the solution.
- `HotNLTEModel.helium_conservation_row="dominant"` (default `"last"`):
  particle conservation replaces the equation of the most populous helium
  state, not the He III equation. In cool sdB surface layers
  (He III/He ~ 10⁻¹⁸), the He III equation makes the system singular to
  roundoff.
- `SdBConfig.helium_i_atom`: `"14"` (default) is the TLUSTY 14-term He I atom.
  `"24"` loads the TLUSTY 24-term atom with `read_tlusty_helium_i_atom`. For
  sdB optical He I lines, the 24-term atom changes the core depths by at most
  1.5%.

The coupled helium solver enforces particle conservation by a common
rescaling. It does not compute He III as the remainder of the total.

## Trace metals (research driver)

`research/sdb_trace_metals.py` adds metals to a finished hybrid model with
the fixed-host solver `solve_hot_trace_metals` (see
[hot DA/DAO trace metals](hot-daz-trace-metals.md)). The H/He structure and
populations stay fixed.

- NLTE elements: C, N, O, Si and S. The Stout atoms reach the standard sdB
  optical lines. C II and O III use bundled CHIANTI collision data.
- LTE line opacity in the fixed background: Fe, Mg and Al (`--lte-abundance`).
  The optical Fe III 4s–4p lines need a 410-level atom, which is too slow
  for NLTE.
- The driver uses `mali_overlap_velocity=15` (km/s) and
  `mali_overlap_mode="subordinate"`. Without these settings, overlapping lines
  of different elements (S III 702.78/702.82 Å on O III 702.84 Å) make the
  iteration oscillate.

At fixed published parameters, without abundance fits, the metal lines of
HD 4539 (C II) and Feige 38 (N II, O II, Si III 4813–4829 Å, Fe III 4164 Å)
match the observed depths within 0.01–0.02. Si III 4552–4575 Å, S III and
Si IV are too weak in Feige 38.

## Research scripts

| Script | Purpose |
| --- | --- |
| `research/sdb_grid_check.sh`, `sdb_grid_report.py`, `sdb_grid_plot_spectra.py` | 12-point H/He grid at 20–40 kK, log g 5.0–6.2, log He/H -4 to -1 |
| `research/sdb_metal_grid.sh` | The same grid with typical metal abundances |
| `research/sdb_fixed_point_check.py` | One unpreconditioned update on a converged model |
| `research/sdb_blanketing_check.py` | Effect of metal opacity on the structure (LTE DA against DAZ) |
| `research/sdb_coadd_xshooter.py`, `sdb_compare_hd4539.py` | HD 4539 X-shooter comparison |

## Limitations

- The structure contains no metals. At Feige 38, metal opacity heats the
  structure by 1.6–2.4% at fixed column mass. The metal-line depths change by
  at most 0.01, except Si IV.
- In both He I atoms, n = 5 is a superlevel coupled to n = 4 by
  Van Regemorter-type rates.
- The hydrostatic equation does not include radiation pressure. This is
  necessary for hot sdOs (above about 40 kK), which are out of scope.
