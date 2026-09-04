<p align="center">
  <img src="docs/assets/openwd_logo.png" alt="OpenWD logo" width="520">
</p>

# OpenWD

OpenWD is an early, open-source white-dwarf atmosphere and spectrum code. This
first public release calculates plane-parallel DA, DB, homogeneous DAB/DBA,
and helium-dominated polluted DZ/DBZ atmospheres. It solves the atmosphere
structure rather than interpolating a precomputed spectral grid.

> **Pre-alpha:** the code is suitable for inspection, validation, and
> development. One-shot calls warn when a returned spectrum does not have a
> verified converged atmosphere; check the convergence metadata before using a
> result scientifically.

## Installation

OpenWD requires Python 3.9 or newer. From a clone:

```bash
python -m pip install -e .
```

The build attempts to compile an optional C radiative-transfer kernel and
falls back to the tested Python implementation when a compiler is unavailable.
All data required by the four released presets are included in the repository.

## Python interface

```python
from wd_spectra import DAConfig, compute_da

result = compute_da(DAConfig(
    effective_temperature=12_000,
    logg=8.0,
    quality="standard",
))

wavelength = result.spectrum.wavelength_angstrom
surface_flux = result.spectrum.surface_flux_lambda
print(result.atmosphere.metadata)
print(result.metadata["atmosphere_convergence_status"])
```

The equivalent entry points are `compute_db`, `compute_dab`, and `compute_dz`.
Each returns the atmosphere, emergent surface spectrum, input configuration,
and detailed provenance metadata. Exploratory spectra from incomplete or
legacy atmospheres remain available, but emit
`AtmosphereConvergenceWarning`. Saved models retain a request fingerprint;
only an exact configuration, physics-revision, and data-root match is allowed
to bypass fresh-start conditioning on a later relaxation.

## One-shot calculations

Each script starts from an approximate atmosphere, iterates the coupled
structure, calculates the final spectrum, and writes `atmosphere.npz`,
`spectrum.txt`, `metadata.json`, and `spectrum.png`.

```bash
python examples/one_shot_da.py  --teff 12000 --logg 8.0 --quality standard
python examples/one_shot_db.py  --teff 20000 --logg 8.0 --quality standard
python examples/one_shot_dab.py --teff 20000 --logg 8.0 --log-h-he -2
python examples/one_shot_dz.py  --teff 15300 --logg 8.0 \
  --abundance O=-5.61 --abundance Mg=-6.24 --abundance Ca=-6.88
```

`quick` is intended only for smoke tests. `standard` is the normal starting
point; `production` increases depth, wavelength, angular, and iteration
budgets. Runtime ranges from minutes for warm DA/DB models to hours for
line-rich DZ atmospheres.

## Performance

The optional C extension accelerates the formal-transfer, metal-line,
helium-profile, and neutral-broadened hydrogen-profile kernels.  Compiled
hydrogen profiles use up to eight worker threads by default because every
atmospheric depth is independent.  Set `OPENWD_NUM_THREADS` to a positive
integer to control that work, including `OPENWD_NUM_THREADS=1` for serial
execution or when parallelizing a model grid at the process level.

A bundled, data-independent benchmark exercises the dominant Balmer-opacity
path without changing any numerical settings:

```bash
python benchmarks/benchmark_hot_paths.py
python benchmarks/benchmark_hot_paths.py --full
OPENWD_NUM_THREADS=1 python benchmarks/benchmark_hot_paths.py --full
```

The benchmark reports a checksum with its timing.  It is diagnostic rather
than a wall-clock test because absolute runtimes depend on the machine.

## Included physics

- Composition-specific hydrostatic, radiative/convective-equilibrium
  structure solvers using a common trust-region Newton engine, conservative
  Feautrier transfer, backtracking, and Broyden updates.
- LTE H/He equations of state with Hummer--Mihalas occupation probabilities
  and correlated Q-MHD microfields.
- ML2 convection (`alpha=0.7` for DA and `alpha=1.25` for helium-dominated
  models).
- Lyman through Brackett hydrogen lines, dissolved-level pseudo-continuum,
  Tremblay--Bergeron Stark profiles, Allard Lyman profiles, and
  temperature-dependent Ali--Griem/Barklem Balmer self broadening.
- Beauchamp He I and Schoening/SYNSPEC He II Stark profiles, with neutral-He
  broadening and helium continuum opacity.
- For DZ: metal electron donation and structural blanketing, the paper-figure
  Stout line data by default (with evaluated NIST strong-line replacements as
  an explicit option), Verner photoionization, dense-He ionization shifts,
  reduced Ca II resonance source functions, and available unified Mg I--He
  and Ca I--He profiles.

The current modules are LTE. DAB assumes a homogeneous atomic H/He mixture;
DZ assumes a helium-dominated host and treats abundances as fixed inputs, not
fit parameters. See the concise module notes in [`docs/models`](docs/models).

## Data and attribution

Runtime data are bundled so a fresh clone is self-contained. They retain their
source licenses and provenance where supplied. The temperature-dependent
Allard Lyman tables are included with the author's permission. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the data-directory
README/license files. The BSD-3-Clause license applies to OpenWD's source code,
not automatically to independently authored scientific tables.

## Tests

```bash
python -m pip install -e ".[test]"
pytest -q
```

The ordinary suite contains nearly 300 solver, EOS, opacity, line-profile,
transfer, model-component, acceleration-equivalence, and safety tests. Slow
no-fallback atmosphere canaries run separately in GitHub Actions every week
and on manual request. Run them locally with:

```bash
pytest tests/test_protected_model_canaries.py -o addopts=-ra
```

The repository does not include observational or published-grid validation
spectra. See the
[recovered solver baseline](docs/recovered-solver-baseline.md) for the
convergence cases used to protect the current numerical core, and the
[development policy](docs/development.md) for the merge gates around solver
and physics changes. The [solver telemetry](docs/solver-telemetry.md) reference
describes the recorded terminal reasons and nonlinear histories.

## Status

This is the deliberately minimal first public version. Hot NLTE, magnetic,
PG 1159, D6, and other experimental modules remain under development and are
not included here.

## How it works

![OpenWD workflow](docs/assets/openwd_workflow.png)
