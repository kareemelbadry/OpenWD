# Hot DA/DAO trace metals

[Contributor guide](README.md) · [DO/DAO model guide](../models/DO-DAO.md)

This solver computes NLTE populations and line formation of trace metals on a
fixed hot H/He atmosphere. The [DO/DAO model guide](../models/DO-DAO.md)
shows the G191-B2B comparison and the commands for users. This note
describes the code for developers.

## Method

1. A public DAO cold start gives the H/He host: temperature, density and H/He
   NLTE populations.
2. The host stays fixed. The solver iterates the statistical equilibrium of
   the explicit metal atoms with the formal-solution radiation field.
3. The emergent spectrum includes the metal opacity and emissivity.

Abundances are `log10 N(element)/N(H)`. The metals do not change the host
temperature structure. This is the main physical limitation (see the model
guide).

## Code

| Item | Purpose |
| --- | --- |
| `wd_spectra.hot_trace_metals.solve_hot_trace_metals` | Metal statistical equilibrium and transfer on a fixed host |
| `wd_spectra.hot_trace_metals.synthesize_dao_trace_metals` | Uses a `compute_dao` result as the host |
| `wd_spectra.light_metal_nlte` | Rates, line-profile means and Kramers continua |
| `research/hot_daz_g191b2b.py` | G191-B2B: DAO host cold start, then nine metals from LTE |
| `research/explore_hot_composition.py` | Atom-size and composition experiments |
| `research/hot_metal_structure.py` | Exploratory relaxation of the host temperature with metal opacity |
| `tests/test_trace_metal_speedups.py` | Equivalence tests for the speed features |
| `tests/test_hot_daz_g191b2b_canary.py` | 40-depth G191-B2B canary (about 2.7 h on one thread) |

`research/hot_daz_g191b2b.py --quality standard` (40 depths, 3 angles) is the
canary configuration. `--quality production` (80 depths, 4 angles) gives the
published model.

## Convergence criteria

The solver tests the undamped update. Select the test with
`convergence_criterion`:

| Value | Test | Default tolerance |
| --- | --- | --- |
| `"population"` (default) | Relative population change of levels above `POPULATION_DEFECT_FLOOR` (1e-6) of the element, at Rosseland depth at least `POPULATION_DEFECT_MINIMUM_TAU` (1e-6) | 1e-3 |
| `"opacity"` | Change of metal absorption and emissivity relative to the total extinction and emission, at every wavelength and depth | 1e-3 |
| `"flux"` | Change of the emergent flux at every wavelength in `flux_wavelength_range` (default 900–1e5 Å) | `FLUX_TOLERANCE` (3e-3) |

The G191-B2B driver uses `"flux"`. The population and opacity tests are
dominated by layers and levels that do not form observed lines. Thus, they
can stall after the spectrum has converged.

## Iteration options

| Option | Default | Effect |
| --- | --- | --- |
| `accelerated_lambda` | `False` | Diagonal approximate lambda operator, weighted by each line's share of the line-centre extinction. Changes the path, not the solution. |
| `mali_overlap_velocity` | `None` | Withholds the operator from lines within this velocity (km/s) of a line of another element. |
| `mali_overlap_mode` | `"exclude"` | `"subordinate"` withholds the operator only from the weaker line of an overlapping pair. |
| `acceleration_scope` | `"tested"` | Levels that can reject an Anderson step: only the levels that the population test counts. |
| `total_recombination` | `None` | Total recombination coefficients. The excess over the explicit atom is added with its LTE inverse. |
| `profile_block_tolerance` | `1e-4` | Block quadrature of line-profile means where the profile is linear to this tolerance. Emergent spectra use the exact quadrature. |
| `opacity_check_gate` | `3.0` | Computes the undamped residual only when the iterate change is below this multiple of the tolerance. |

From LTE, the solver accepts the first statistical-equilibrium solution in
full. If the Anderson step is rejected, falling populations move halfway in
log and rising populations move halfway linearly.

## Speed features

All these features keep the converged solution within the stated tolerances.
`tests/test_trace_metal_speedups.py` tests each one.

- A fused C kernel (`metal_line_profile_means`) averages the photon occupation,
  its inverse and the lambda operator in one pass over each profile.
- Block quadrature (`expand_metal_line_blocks`) replaces fine-grid sums where
  a profile is linear.
- Cumulative Kramers continua (`kramers_cumulative`) sum the hydrogenic
  continua with prefix sums over levels sorted by edge.
- Electron-impact rates depend only on the fixed host. The solver caches them
  for each element (`rate_cache`).

## Resources

Single thread, production resolution (80 depths): the H/He host cold start
takes about 5.3 h. The nine metals take about 4.8 h at a flux tolerance of
1e-3 and about 3.1 h at the default 3e-3. Each step has a peak memory of
about 24 GB. Do not run the two steps at the same time on a 48 GB machine.

## History

The [archived development record](history/hot-daz-trace-metals-2026-10-05.md)
contains the earlier C/Si tests, the iron-group audit, the TMAP comparison,
the convergence studies and the timing measurements.
