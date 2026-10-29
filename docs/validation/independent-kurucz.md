# Independently corroborated Kurucz lines

[Contributor guide](../development/README.md) · [DZ model guide](../models/DZ.md)

The default Stout reader adds Kurucz transitions that the Stout atoms do not
have. One atomic-data rule selects them. No white-dwarf observation enters the
selection. No oscillator strength, width, abundance or atmospheric parameter
is fitted. The data are bundled and checksum-pinned.

## Selection rule

The rule screens the neutral and singly ionized ions of C, N, O, Na, Mg, Al,
Si, P, S, Ca, Ti, Cr, Mn, Fe and Ni (30 ions). A transition is added only if
all these conditions are true:

1. The Kurucz level energies are observed.
2. Each level matches one level of the Stout atom, by energy within 0.1 cm⁻¹
   and by statistical weight.
3. Both levels are below the ground ionization limit.
4. An independent source gives the oscillator strength.
5. The Kurucz oscillator strength agrees with the independent value within a
   factor of ten.

The preferred independent sources, in order:

1. NIST evaluations of grade C or better, or measured branching fractions;
2. the 2024 Si II GRASP calculation, with a gauge discrepancy of at most 0.25;
3. other independent calculations between known levels.

Conflicting duplicate strengths are rejected. The factor-ten and gauge tests
are eligibility tests, not uncertainty estimates. Most corroboration is
independent theory, not laboratory data.

The rule adds 19,966 absorption transitions with f ≥ 10⁻⁴:

| Ion | Added absorption transitions |
|---|---:|
| C I | 1 |
| Si I | 433 |
| Si II | 17 |
| S I | 7 |
| Ti I | 4 |
| Cr II | 14,891 |
| Fe II | 4,613 |

The added lines keep the Kurucz oscillator strengths and damping constants.
Weaker qualified downward transitions contribute to the natural widths but
add no absorption lines.

The usual LTE ranking and line budgets apply. The formal synthesis also drops
a supplement line if its peak opacity is below 10⁻³ of the background at all
depths. This screen never removes an original Stout line.

The [atomic bundle](../../src/wd_spectra/data/atomic/independent-kurucz/README.md)
records the sources, checksums and evidence for each transition. The
[reproduction scripts](../../scripts/independent_kurucz/README.md) regenerate
and verify the selection from pinned public inputs. The
[NIST reference pages](independent-kurucz/nist-reference-pages.json) give the
bibliography URLs and checksums of the NIST references.

## Observational check

The check compares models with and without the added lines at fixed
published parameters, compositions, normalizations, velocities, line-spread
functions and masks. It covers 14 objects: GD 40, G241−6, GD 61, GD 378,
WD 1425+540, PG 1225−079, GD 362, WD 1232+563, WD 1551+175, WD 2207+127,
G29−38, GD 133, G149−28 and GALEX J1931+0117. Eight pairs re-solve the
atmosphere; all pass the convergence checks.

No whole-object score degrades by more than 5%. Examples of χ² changes:
−3.1% for GD 40, −1.9% for G241−6, +0.3% for GD 362, −1.8% for WD 1232+563.
These numbers compare two models. They do not measure the absolute fit.

![Selected Fe II improvements and Cr II lines in HIRES spectra](independent-kurucz/focus.png)

Black is the observation. Blue is the model without the added lines. Red is
the model with the added lines. The UV panels use vacuum wavelengths. The
HIRES panels use air wavelengths.

For J1637 (D6), the added lines change narrow line pixels by up to 37.6% and
the integrated 3400–7700 Å flux by −0.09%. At R = 2500, the squared residual
against the GTC spectrum decreases by 0.3–0.8%. The
[D6 check](independent-kurucz/d6-fixed-observation.json) gives the scores for
each panel.

![J1637 GTC spectrum and OpenWD at R = 2500](independent-kurucz/j1637-comparison.png)

Gray is the GTC spectrum. Blue is the model without the added lines. Red is
the model with the added lines, on the same atmosphere.

The runtime change is between −7% and +6% for three DZ/DAZ cold starts. The
largest peak memory is about 4 GiB (J1637).

The [observation scores](independent-kurucz/observation-scores.json), the
[window scores](independent-kurucz/observation-window-scores.json.xz)
(xz-compressed JSON) and the [calculation audit](independent-kurucz/assessment.json)
contain the measured scores, hashes and convergence checks.

## Limitations

- The rule excludes 111 of 142 Fe II lines at 1189–1198 Å that an earlier
  supplement contained. For example, Fe II 1190.882 Å involves a 6p level
  outside the independent Fe II calculation. Local residuals near 1167, 1191
  and 1403 Å in GD 40, and near 1167 Å in G241−6, are larger.
- Si II 1305.59 Å is excluded, because its upper level is slightly above the
  ground ionization limit. The broad Si II 1260/1265 Å wings remain too weak.
- The added infrared lines (C I 7.47 μm, S I 0.95–2.29 μm, many Si I lines)
  have no observational test here.
- An ion with no added lines is not therefore complete.

## Usage

`compute_dz` and `compute_daz` include the added lines automatically. To
read the Stout atoms without them:

```python
from wd_spectra.metals import read_stout_atomic_database
from wd_spectra.models import ModelData

stout = read_stout_atomic_database(
    ModelData.default().stout,
    elements=("Mg", "Si", "Ca", "Cr", "Fe"),
    maximum_charge=3,
    include_default_supplements=False,
)
```

The explicit augmentation functions do not change an atom that already has
the qualified lines. The configuration field `si_ii_kurucz_transitions` does
not remove qualified Si II lines from the default reader.
