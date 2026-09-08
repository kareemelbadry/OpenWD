# Tested cold starts and limitations — September 2026

[Documentation home](README.md) · [Limitations overview](limitations.md)

These are **tested points**, not a guarantee throughout an interval.
Unless noted, log g = 8 and quality is production. Numerical convergence
certifies the declared equations on the structure grid, not complete physics
or depth-grid independence.
The separately synthesized spectrum also has known transfer-consistency
limitations: the retained public method integrates to 0.98613 of stellar flux
for DA 3000 and 0.99314 for DB 10000 in a broad-wavelength audit. The unfinished
matched-transfer experiment is not enabled. See the
[checkpoint's known spectrum limits](development/history/cold-start-numerics-2026-09-07.md#known-spectrum-consistency-limits-unfinished-changes-excluded).

Public generation starts from scratch. No saved atmosphere or neighboring
stellar model is required. Provisional thermal conditioning and lower-boundary
mesh adaptation occur inside the same calculation. Historical continuation
experiments remain in the research record but are excluded from this table.

| Composition / public workflow | Cold-start tested temperatures | Evidence |
| --- | --- | --- |
| DA, `compute_da` or `run_model` | 3000, 4000, 5000, 20000 K | All-depth flux, cell-local energy, measured unrestricted correction, scattering closure and bottom screening pass. |
| Dense pure-He DB, `run_model` | 5000, 8000 K | Fresh 80-layer initialization, static convergence and independent wavelength/angular/source audits. Experimental dense physics. |
| Established pure-He DB, `compute_db` or `run_model` | 10000, 22000 K | Strict structure-grid checks pass; 22000 K also tested at standard 40-layer resolution. The 10000 K calculation extends its own lower boundary from 80 to 84 nodes. |
| Molecular DAB/DBA, `run_model`, log10(N_H/N_He) = -2 | 7500, 8000, 9000, 10000 K | Fresh initialization with the final molecular/line physics. Strict static checks and independent fixed-state wavelength/angular audits. |
| Atomic DAB, `compute_dab` or automatically selected at 20000 K | 20000 K | Strict cold-start convergence and the paper-spectrum comparison gate pass. |
| DZ, PG 1225 composition, `compute_dz` | 10800 K | Strict production cold-start convergence; identical atmosphere to a same-resolution original-GitHub cold run. This is not exact reproduction of the 40-node paper spectrum. |
| DAZ, `compute_daz` or `run_model`, standard 40-layer resolution | G149-28: 8600 K, log g = 8.10; G29-38: 11820 K, log g = 8.40; GALEX J1931+0117: 20890 K, log g = 7.90 | Fresh public cold starts with each object's metal composition pass all five structure-grid certificate gates. These are individual points, not a temperature/abundance grid. |

The revised solver repairs local energy errors that previously survived a
small interface-flux residual. It does not inherit historical success flags,
renormalize spectra, or impose convective flux as a remainder.
See [numerical repair](development/history/cold-start-numerics-2026-09-07.md) and
[certificate definitions](development/history/reliability-2026-09-07.md).

## Reproduction from scratch

For ordinary DA and warm DB/DAB presets:

```python
from wd_spectra import DAConfig, compute_da
result = compute_da(DAConfig(effective_temperature=3000, logg=8,
                            quality="production"))
assert result.atmosphere.metadata["equilibrium_certificate"]["verified"]
```

For automatic material selection, including dense helium and molecular mixtures:

```python
from wd_spectra import DBConfig, DABConfig, run_model

db = run_model(DBConfig(effective_temperature=5000, quality="production"),
               "results/db-5000-fresh", require_convergence=True)
dab = run_model(DABConfig(effective_temperature=7500, quality="production"),
                "results/dab-7500-fresh", require_convergence=True)
```

Use a new output directory for each calculation. Cool workers require a source
checkout with the normal installation and, for DAB, the
[declared molecular data](../research/cool_models/README.md).
Missing data or invalid material domains fail explicitly; there is no alternate
physics retry. `run_model` rejects checkpoint inputs.

`compute_*` functions remain explicit physics presets. In particular,
`compute_db` at 5000 K does not select dense helium, and merely enabling
`DABConfig(include_molecules=True)` does not select the qualified conservative
molecular transport workflow. Use `run_model` for that automatic selection.

## Numerical evidence

Errors below are fractions, not percentages. Local energy uses each cell's
exchange scale, not total stellar luminosity. Fresh molecular models use
80 layers and the same 40-sweep provisional thermal budget; a subsequent
full static solve must independently pass every physical gate.

| Fresh case | All-depth flux error | Cell-local energy error | Independent audit |
| --- | ---: | ---: | --- |
| DA 3000 | 3.82e-4 | 3.24e-4 | Structure grid only |
| DA 4000 | 4.28e-6 | 4.18e-6 | Structure grid only |
| DA 5000 | 3.53e-6 | 1.17e-5 | Structure grid only |
| DA 20000 | 1.43e-7 | 4.29e-4 | Structure grid only |
| DB 10000 | 5.64e-4 | 4.31e-5 | Structure grid only; bottom escape bound 8.27e-26 |
| DB 22000, standard | 4.63e-10 | 1.69e-3 | Structure grid only |
| Atomic DAB 20000 | 8.53e-8 | 1.59e-3 | Structure grid only |
| Dense DB 5000 | 1.26e-4 | 1.24e-5 | Independent local error 0.00126 |
| Dense DB 8000 | 6.76e-6 | 1.04e-5 | Independent local error 0.000955 |
| Molecular DAB 7500 | 3.74e-10 | 2.90e-6 | Independent flux 8.01e-5, local 7.69e-5 |
| Molecular DAB 8000 | 1.91e-10 | 1.75e-6 | Independent flux 1.53e-4, local 6.39e-5 |
| Molecular DAB 9000 | 1.16e-10 | 2.53e-7 | Independent flux 3.16e-4, local 5.98e-5 |
| Molecular DAB 10000 | 2.23e-10 | 4.85e-8 | Independent flux 4.63e-4, local 8.66e-5 |
| DAZ G149-28, standard | 4.10e-6 | 2.54e-6 | Structure grid only; 32 iterations |
| DAZ G29-38, standard | 6.12e-5 | 4.05e-6 | Structure grid only; 58 iterations |
| DAZ GALEX J1931+0117, standard | 7.92e-6 | 8.88e-4 | Structure grid only; 97 iterations |

Independent quadrature audits hold the newly calculated atmosphere fixed.
They are checks of that result, not re-relaxations or inputs needed for a cold
start. They do not replace stationarity or depth-resolution tests.

The three DAZ cases were rerun from scratch on September 8 with the established
atmosphere and formal-integral synthesis methods. Temperatures, gas pressures,
column masses and saved spectra exactly reproduce their pre-experiment cold
results. The configurations and metal abundances are in
[`research/validate_daz_cold_start.py`](../research/validate_daz_cold_start.py).
The separate fixed-atmosphere DAZ paper comparisons also pass; these do not
imply that a new cold atmosphere is identical to a paper checkpoint or that
its final-spectrum integral is certified. See the [DAZ guide](models/DAZ.md).

The 8000 K DAB depth study compared 80, 159 and 317 layers: integrated absolute
spectral differences decreased from 0.688% to 0.172% of stellar flux, approximately
second order. The new 80-layer 7500 K cold spectrum differs by 0.985% in integrated
absolute flux from the historical 166-layer calculation. These are different
depth grids; neither comparison establishes universal depth independence.

## Limits and preservation of established results

- No DA below 3000 K or pure-He DB below 5000 K is qualified here.
- Molecular DAB 7250 K remains unqualified under the measured correction gate;
  5000 K experiments remain far from equilibrium. There is no claim of a
  reliable 5000–10000 K DAB grid.
- The old 7750 K continuation is not a cold-start qualification.
- No abundance or gravity grid has been demonstrated. The cool DAB points
  mean the homogeneous 1% hydrogen-by-number mixture above.
- Dense pure-He thermodynamics combine REOS with approximate HNC chemical
  potentials and trace-ion chemistry. Local table and trace-ion limits are
  enforced. Refraction and collective He-minus corrections remain absent.
  The dense-neutral closure is not inserted into warm ionized helium or
  hydrogen/helium mixtures.
- Molecular mixtures include H2, H2+, H-, H3+, H/He ionization, H2-He/H2-H2 CIA
  and neutral Ly-alpha wings. A consistent dense-mixture free energy,
  nonideal dissociation, mixed HeH+/He2+ chemistry and pressure-distorted CIA
  remain missing.
- The DA seed retains its previously tested below-5000 K initialization policy.
  The new local-energy completion and thermal step selection introduce no
  additional Teff/composition switch.

Immutable fixed-atmosphere controls protect UV, optical lines and IR for
DA/DB, the paper DAB 9000/20000 models, PG 1225-079 and SDSS J0738+1835.
Those checks alone are not cold-start or equilibrium evidence. The legacy
atomic 9000 K paper spectrum remains a spectral control; automatic generation
at that point selects molecular physics.
J0738's new production atmosphere passes its static checks, but its cold
spectrum fails the paper comparison; it is not in the qualified reproduction
table above. That discrepancy is not waived or explained away as a proven
pre-existing difference.

The repaired 10000 K DB changes optical and IR band flux by only 0.0176% and
0.0092%, but its 1150–3000 Angstrom flux rises by 0.271% and portions of the
faint far-UV tail change by about 10%. This is disclosed, not described as an
identical spectrum. The original boundary was not absorption-screened.

Historical investigations, including unsuccessful runs and continuation
experiments, remain in the [research archive](development/history/README.md).
They document chronology, not current public reproduction instructions.
