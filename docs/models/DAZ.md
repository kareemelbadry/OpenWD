# DAZ: polluted hydrogen atmospheres

[Model guide](README.md) · [Getting started](../getting-started.md)

Use `DAZConfig` for a hydrogen-dominated atmosphere containing metals. Unlike
DZ/DBZ, its host EOS, hydrogen opacity, and convection are those of DA. Metals
participate in the common electron-density/charge-neutrality solution and in
the opacity used to solve the atmosphere, not only in the final spectrum.

```python
from wd_spectra import DAZConfig, run_model

run = run_model(
    DAZConfig(effective_temperature=11_820, logg=8.40, quality="standard"),
    "results/daz-g29-38",
)
print(run.convergence_verified)
```

The default abundances describe G29-38. Supply `abundances={"Ca": -8.0, ...}`
to replace the entire mixture; values are **log10 N(element)/N(H)**, not ratios
to helium. All required standard atomic and profile data are bundled. A new
run starts from scratch, and `require_convergence=True` requires all the
current atmosphere-certificate gates. Unqualified exploratory spectra warn.

The equivalent command is:

```bash
python examples/one_shot_daz.py --teff 11820 --logg 8.40 \
  --quality standard --output results/daz-example
```

`compute_daz` is also available for in-memory calculations. The automatic
interface never routes DAZ through a helium or pure-DA substitute on failure.

## Physics and paper comparisons

The preset uses the shared adaptive DA solver, ML2/alpha=0.7, the DA hydrogen
line/molecular policy, Stout metal lines through charge 3, and Verner
photoionization. It retains the established Feautrier atmosphere solver and
formal-integral final synthesis. As in DZ, the atmosphere uses a
composition-dependent budget for significant metal lines; final synthesis
includes a more detailed line list. No forced atmosphere/spectrum grid matching
or experimental wavelength-refinement loop is enabled.
Dense-helium ionization corrections and helium-perturber profiles are
not applied to hydrogen hosts.

Stout strengths and Unsold neutral-H metal-line widths preserve the paper's
metal-line prescription. The later `strong_line_atomic_data="nist-asd"` and
`metal_neutral_h_broadening="barklem"` options are explicit alternatives;
neither is selected by target name. Hydrogen Balmer self-broadening is a
separate prescription and follows the DA policy.

The fixed-atmosphere regression controls include G149-28 (8600 K) and GALEX
J1931+0117 (20890 K) from the paper's metal-polluted-star figure. These tests
use the paper's Stark-only Lyman setting (`lyman_profile_source="stark"`);
new calculations default to the DA Allard policy. Fixed-state comparisons
protect the plotted synthetic spectra, not cold-start convergence. See
[tested points](../tested-temperature-ranges.md) for the latter.

Fresh standard-resolution checks of G149-28, G29-38, and GALEX J1931+0117 use
`research/validate_daz_cold_start.py` and require all five physical atmosphere
gates. These are individual composition/gravity points, not a qualified
temperature or abundance grid. Atmosphere convergence does not certify depth
independence, exact reproduction of a paper atmosphere, or the integral of a
separately sampled final spectrum.

Metals remain trace contributors in the thermodynamic derivatives used by
ML2. Ca II can use a reduced scattering source in the final spectrum while
the atmospheric populations and extinction remain LTE; this is not a full
metal-NLTE atmosphere. A low-temperature pure-DA validation does not establish
the same validity range for arbitrary metal abundances.
