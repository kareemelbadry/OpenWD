# Getting started

[Documentation home](README.md) · [Installation](installation.md) ·
[Model guides](models/README.md) · [Limitations](limitations.md)

## Check the installation

Install OpenWD as described in [installation](installation.md). Then run these
commands from the repository root:

```bash
python -c "import wd_spectra; print(wd_spectra.__version__, wd_spectra.__file__)"
python -c "from wd_spectra import compiled_backend_available; print(compiled_backend_available())"
```

The second command shows `True` when the optional C extension is active.
Without the extension, OpenWD uses equivalent Python code, which is slower.

## Run your first model

This example calculates a DA model at 12,000 K and log g = 8. It uses only
bundled data. Allow several minutes.

```python
import matplotlib.pyplot as plt
from wd_spectra import DAConfig, run_model

config = DAConfig(effective_temperature=12_000, logg=8.0, quality="standard")
run = run_model(config, "results/da-12000")
print("Selected workflow:", run.selection.workflow)
print("Converged:", run.convergence_verified)

spectrum = run.spectrum
plt.plot(spectrum.wavelength_angstrom, spectrum.surface_flux_lambda)
plt.xlabel("Vacuum wavelength (Angstrom)")
plt.ylabel("Surface flux (erg s^-1 cm^-2 Angstrom^-1)")
plt.show()
```

`run_model` always does a cold start: it builds a new atmosphere from the
input parameters. It does not use a saved atmosphere. It prints its
progress while it iterates.

The output directory must not exist. `run_model` does not overwrite a
directory. Use a new directory name for each run.

For an interactive version, open the
[example notebook](../examples/generate_spectrum.ipynb).

## Outputs

### Output files

`run.output_directory` contains these files:

| File | Contents | Written for |
| --- | --- | --- |
| `spectrum.txt` | Columns: vacuum wavelength (Å), surface flux, continuum flux | All classes except the dense-helium DB workflow |
| `model-run.json` | Request, selected physics, status and convergence result | All classes |
| `metadata.json` | Configuration, convergence certificate and spectrum metadata | All classes except the dense-helium DB workflow |
| `atmosphere.npz` | Optical depth, column mass, temperature, pressure, density and electron density at each depth | All classes except the dense-helium DB workflow |
| `populations.npz` | NLTE level populations | DO, DAO, PG 1159 and sdB |
| `stokes.txt` | Stokes I, Q, U and V | DAH with `return_stokes=True` |
| `worker/` | Solver products and independent checks | DQ and the dense-helium DB workflow |

`run.spectrum_path` gives the location of the spectrum file for every class.
`run.spectrum` reads it and returns these arrays:

| Attribute | Unit |
| --- | --- |
| `wavelength_angstrom` | Å, vacuum |
| `surface_flux_lambda` | erg s⁻¹ cm⁻² Å⁻¹ |
| `continuum_flux_lambda` | erg s⁻¹ cm⁻² Å⁻¹ (`None` if the workflow gives no continuum) |

The flux is the emergent flux at the stellar surface. OpenWD does not scale it
to a distance, apply extinction or convolve it with an instrument profile.
Each workflow uses its own wavelength grid.

The continuum flux is the emergent flux of the same atmosphere with all line
opacity removed. The C₂ bands and the far wing of Lyman α count as lines.
Bound-free and free-free absorption, collision-induced absorption, scattering
and the dissolved-level pseudo-continuum stay in the continuum. Divide the
flux by the continuum to get a normalized spectrum.

### Convergence

`run.convergence_verified` is `True` when the atmosphere passed all the
convergence checks of its class. `model-run.json` records the same result.

By default, `run_model` keeps a spectrum that did not converge. It shows an
`AtmosphereConvergenceWarning`. To make convergence mandatory, set
`require_convergence=True`:

```python
run = run_model(config, "results/da-12000-strict", require_convergence=True)
```

Then a run that does not converge raises `RuntimeError`. The output files stay
in the directory for diagnosis. Missing data, invalid parameters and solver
failures always raise an error. OpenWD never retries with different physics.

Two classes have special rules:

- DQ always requires full convergence and an independent flux check. A DQ run
  that fails these checks raises an error, also with
  `require_convergence=False`.
- PG 1159 reports the status `spectrum-qualified` or `converged`. See
  [PG 1159](#pg-1159-hot-heliumcarbonoxygen-atmospheres).

Convergence is a numerical result. It does not show that the physics is
correct for your star. See [limitations](limitations.md).

## Change the composition

Each model class has its own configuration class. Every example below is a
complete script. Read the [model guide](models/README.md) for the class
before you change other options.

Abundance conventions differ between classes:

| Configuration | Composition input |
| --- | --- |
| `DAZConfig` | `abundances`: log10 N(Z)/N(H) |
| `DABConfig` | `log_hydrogen_to_helium`: log10 N(H)/N(He) |
| `DZConfig` | `abundances`: log10 N(Z)/N(He); `log_hydrogen_abundance`: log10 N(H)/N(He) |
| `DQConfig` | `log_carbon_to_helium`: log10 N(C)/N(He) |
| `DAOConfig` | `log_hydrogen_to_helium`: log10 N(H)/N(He) |
| `PG1159Config` | `mass_fractions`: mass fractions, normalized to a sum of 1 |
| `D6Config` | `abundances`: log10 N(Z)/N(C) |
| `SdBConfig` | `log_helium_to_hydrogen`: log10 N(He)/N(H) |

An `abundances` or `mass_fractions` dictionary replaces the complete default
dictionary. Elements that you do not list are not included.

### DA and DAZ hydrogen atmospheres

```python
from wd_spectra import DAZConfig, run_model

# G29-38; the DAZConfig defaults use this star's metal abundances.
config = DAZConfig(effective_temperature=11_820, logg=8.40,
                   abundances={"C": -6.9, "O": -5.0, "Mg": -5.77, "Si": -5.6,
                               "Ca": -6.58, "Ti": -7.9, "Cr": -7.51, "Fe": -5.9})
run = run_model(config, "results/daz-11820")
```

### DB, DAB/DBA and DZ helium atmospheres

```python
from wd_spectra import DABConfig, DBConfig, DZConfig, run_model

run_db = run_model(DBConfig(effective_temperature=22_000, logg=8.0), "results/db-22000")

# GD 40; the DZConfig defaults use this star's composition.
dz = DZConfig(effective_temperature=15_300, logg=8.0, log_hydrogen_abundance=-6.16,
              abundances={"C": -7.2, "O": -5.61, "Mg": -6.24, "Si": -6.76,
                          "Ca": -6.88, "Ti": -8.61, "Cr": -8.31, "Mn": -8.62,
                          "Fe": -6.48})
run_dz = run_model(dz, "results/dz-15300")

# Needs the external H2-He_2011.cia file (see below).
dab = DABConfig(effective_temperature=20_000, logg=8.0, log_hydrogen_to_helium=-2.0)
run_dab = run_model(dab, "results/dab-20000")
```

### Cool helium and mixed atmospheres

`run_model` checks the photospheric conditions before it solves the
atmosphere:

- For a cool pure-helium DB, it selects the dense-helium workflow when the
  photosphere is dense (below about 8500 K at log g = 8). This workflow needs
  `quality="production"` and an integer Teff. Always use `run_model` for cool
  DBs. The lower-level `compute_db` does not select the dense-helium workflow.
- Every metal-free DAB/DBA uses molecular H/He chemistry (H2, H2+, H-, H3+)
  and H2-He and H2-H2 collision-induced absorption, at all Teff and log g. This needs the external `H2-He_2011.cia` file. See
  [external data](installation.md#external-data).
- A DAB/DBA with trace metals uses atomic H/He chemistry. `run_model` refuses
  it where molecules are important.

```python
from wd_spectra import DABConfig, DBConfig, run_model

cool_db = run_model(
    DBConfig(effective_temperature=8000, logg=8.0, quality="production"),
    "results/db-8000", require_convergence=True,
)
cool_dab = run_model(
    DABConfig(effective_temperature=7500, logg=8.0, log_hydrogen_to_helium=-2.0,
              quality="production"),
    "results/dab-7500", research_data="~/openwd-data", require_convergence=True,
)
```

### DQ helium/carbon atmospheres

DQ models are hydrogen-free, nonmagnetic helium atmospheres with trace carbon.
All DQ data are bundled.

```python
from wd_spectra import DQConfig, run_model

dq = run_model(
    DQConfig(effective_temperature=9347, logg=8.041,
             log_carbon_to_helium=-4.107, quality="standard",
             maximum_seconds=28800),
    "results/dq-9347",  # new directory; allow hours for this calculation
    require_convergence=True,
)
```

- DQ accepts only `quality="standard"` and -12 ≤ `log_carbon_to_helium` ≤ -3.
- `maximum_seconds` is the time limit for the structure and the final spectrum.
- A DQ run can take hours and several GB of memory.
- Hot carbon-dominated DQs, hydrogen-bearing DQs and DQp stars are outside the
  scope of this module.

See the [DQ guide](models/DQ.md).

### DO/DAO hot helium and hydrogen–helium atmospheres

Use `DOConfig` for pure helium and `DAOConfig` for mixed H/He. These classes
solve NLTE populations. A high Teff in `DAConfig` or `DBConfig` does not
select NLTE.

```python
from wd_spectra import DAOConfig, DOConfig, run_model

run_do = run_model(DOConfig(effective_temperature=50_000, logg=8.0), "results/do-50000")
run_dao = run_model(
    DAOConfig(effective_temperature=60_000, logg=8.0, log_hydrogen_to_helium=2.0),
    "results/dao-60000",
)
```

Allow one to several hours for each model. See the [DO/DAO guide](models/DO-DAO.md).

### PG 1159 hot helium–carbon–oxygen atmospheres

`PG1159Config` solves He, C and O in NLTE for the atmospheric structure. The
other elements in `mass_fractions` (N, Ne, F, Si, P, S, Ar, Fe) enter the NLTE
line formation on the final structure. The defaults are the published
parameters of PG 1424+535.

```python
import json
from wd_spectra import PG1159Config, run_model

run = run_model(PG1159Config(effective_temperature=110_000, logg=7.0),
                "results/pg1159-110000")
status = json.loads((run.output_directory / "metadata.json").read_text())
print(status["model_metadata"]["atmosphere_convergence_status"])
```

- `spectrum-qualified` means that the flux, local energy, population, source
  and boundary checks pass.
- `converged` also requires a full-rank temperature certificate.
- `run.convergence_verified` and `require_convergence=True` accept only
  `converged`. Thus, read the status from `metadata.json` as shown.
- For the hottest stars, add `include_radiative_acceleration=True`.

See the [PG 1159 guide](models/PG1159.md).

### D6 carbon–oxygen atmospheres

`D6Config` describes a hydrogen- and helium-free atmosphere. The defaults are
the Hollands et al. (2025) solution for SDSS J1637+3631.

```python
from wd_spectra import D6Config, run_model

config = D6Config(effective_temperature=15_680, logg=6.3,
                  abundances={"C": 0.0, "O": 0.27, "Ne": -1.86, "Mg": -2.33,
                              "Al": -4.0, "Si": -1.76, "S": -2.84, "Ca": -2.85,
                              "Fe": -3.0})
run = run_model(config, "results/d6-15680")
```

See the [D6 guide](models/D6.md).

### DAH magnetic hydrogen atmospheres

`DAHConfig` adds a uniform or offset-dipole magnetic field to a DA
atmosphere. This example uses the parameters of J2149−0728 from Hardy et al.
(2023).

```python
from wd_spectra import DAHConfig, run_model

config = DAHConfig(effective_temperature=22_642, logg=8.37,
                   magnetic_field_megagauss=45.09, field_geometry="dipole",
                   dipole_inclination_deg=66.0,
                   dipole_offset_radius=(0.0, 0.0, 0.17))
run = run_model(config, "results/dah-22642")
```

See the [DAH guide](models/DAH.md).

### sdB hot subdwarfs

`SdBConfig` calculates an LTE H/He structure, then NLTE H/He populations and
line formation on that structure. This example uses the parameters of HD 4539.

```python
from wd_spectra import SdBConfig, run_model

config = SdBConfig(effective_temperature=23_200, logg=5.20, log_helium_to_hydrogen=-2.27)
run = run_model(config, "results/sdb-23200")
```

`run_model` does not add metals to sdB models. The [sdB guide](models/sdB.md)
gives the command-line driver for trace metals.

## Quality levels

The `quality` option sets the numerical resolution. It does not change the
physics.

| Quality | Depth points | Use |
| --- | --- | --- |
| `quick` | 20 | Test the interface only. The result does not converge. |
| `standard` | 40 | Default. Use it for most models. |
| `production` | 80 | Higher resolution. Required for the dense-helium DB workflow. |

Some classes use other depth counts or have their own limits. DQ accepts only
`standard`. sdB accepts `standard` and `production`. The model guides give the
details.

## Threads

The compiled hydrogen line profiles use up to 8 threads by default. To set the
number of threads, set `OPENWD_NUM_THREADS` before you start Python:

```bash
export OPENWD_NUM_THREADS=4
```

To run several models at the same time, run each in a separate process. Set
`OPENWD_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` for
each process. DQ and the dense-helium DB workflow always run in a separate
single-threaded process.

## Command-line scripts

The `examples/one_shot_*.py` scripts run one model from the command line, for
example:

```bash
python examples/one_shot_da.py --teff 12000 --logg 8 --quality standard
```

Run a script with `--help` to see its options.

## Next steps

- [Model guides](models/README.md): physics, options and paper comparisons
  for each class.
- [Tested ranges](tested-temperature-ranges.md): the parameters that were
  run from a cold start.
- [Limitations](limitations.md): known accuracy limits.
- [DZ fitting example](examples/fitting-dz.md): fit abundances to an observed
  spectrum.
