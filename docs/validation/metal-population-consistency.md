# Metal population consistency

[Contributor guide](../development/README.md) · [DZ model guide](../models/DZ.md)

The level populations that the metal opacity uses are consistent with the
partition functions that give each ion abundance. This page describes the
method and the tests.

## Ionization in a helium host

The neutral-helium excluded volume applies only to levels with a mean radius
of at least `NEUTRAL_HARD_SPHERE_MINIMUM_LEVEL_RADIUS_CM` (20.25 Bohr radii).
Lower levels keep their full neutral population. Charged-perturber occupation
probabilities apply to all levels. The dense-helium ionization shift of
Blouin et al. and the Saha partition ratio are applied as usual.

## Populations for opacity

The stored partition functions are absolute sums. The population of a bound
level uses the same statistical weight, Boltzmann factor and occupation
probability as its term in the partition function. Thus, the sum of the
level populations is the ion abundance. The partition cutoff is 0.1 eV.
Autoionizing resonances use a separate LTE approximation and are not part of
this sum.

These consumers share the calculation: ordinary lines, Verner ground terms,
the special Mg II, Ca I, Ca II and Na I profiles, and the lower populations
of the O I dissolved series. The trace and bulk state builders use different
charged-perturber conventions. Each consumer uses the convention that formed
its partition function.

## TOPbase photoionization levels

TOPbase level indices are not Stout indices. A uniquely identified native
fine-structure level or complete LS term uses the native level populations.
Each channel keeps its own threshold energy.

Some TOPbase entries are aggregate model levels without constituent
identities. `TOPbasePhotoionizationDatabase.unmapped_population_policy`
selects their treatment:

- `"effective-level"` (default): the level keeps its source excitation energy
  and aggregate statistical weight. The hydrogenic survival probability uses
  the true lower binding energy. This is an approximation. The aggregate
  populations do not necessarily sum to the native partition function.
- `"strict"`: unresolved levels cause an error. Use this setting for audits.

No channel is omitted without a record. The model and spectrum metadata
record the policy and the section counts. The
[TLUSTY guide, section 8.1](https://tlusty.oca.eu/tlusty/Tlusty2002/pdf/tlguide200.pdf)
discusses model levels and the accuracy of hydrogenic occupation for complex
ions.

RAP photoionization tables do not define a single edge. Thus, a request for a
dissolved pseudocontinuum on such a table causes an error.

## Tests

- Ground-only Saha tests compute the reference independently.
- Opacity tests hold the ion density fixed. This prevents cancellation
  between ionization and opacity errors.
- Other tests cover the dilute limit, the disabled correction, higher ion
  stages, charge balance, included-level sums, ground terms, source-local
  level IDs, excited-parent channels and the pseudocontinuum edge.

These tests check the implementation. They do not validate all cool-DZ
atmospheres or the inferred abundances.

## Effect on the spectra

The population method changes the fixed-atmosphere spectra by small amounts:

| Star | Maximum relative flux change | Location | Integrated flux change |
| --- | ---: | ---: | ---: |
| PG 1225−079 | 0.0049% | 2659.76 Å | +0.0003% |
| SDSS J0738+1835 | 0.36% | 1222 Å | +0.017% |

Nearly all of each change comes from the TOPbase lower populations. The
[historical controls](https://github.com/kareemelbadry/OpenWD/blob/116ab827d134749ebbdcf4f4e66a4dcfd78cfe65/tests/data/approved_regressions/metal_populations_2026_10_24/README.md)
and their [manifest](https://github.com/kareemelbadry/OpenWD/blob/116ab827d134749ebbdcf4f4e66a4dcfd78cfe65/tests/data/approved_regressions/metal_populations_2026_10_24/manifest.json)
record the structures, the hashes and the attribution.

## Population export

The population state keeps a reference to its `atomic_database`. JSON and
NPZ exports omit dataclass fields marked `serialize=False`. Thus, the export
does not include the line-selection caches of the atom. These caches refer
to whole ions, so a recursive export grows quadratically with the line list.
Tests check that the export size does not depend on the cache size.
