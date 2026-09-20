# Getting started

[Documentation home](README.md) · [Model guide](models/README.md) ·
[Limitations](limitations.md)

## Install

Use Python 3.9 or newer, preferably in a dedicated environment:

```bash
git clone https://github.com/kareemelbadry/OpenWD.git
cd OpenWD
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e .
```

Run subsequent commands from the OpenWD directory. The installer attempts to
build the optional C acceleration extension. The tested Python implementation
is also available if no compiler is present; the selected physical model is
unchanged. NumPy, Matplotlib, SciPy, mpmath, and Numba are installed automatically.
Established-preset and DQ constitutive data are bundled.

For the notebook, install Jupyter in the same environment:

```bash
python -m pip install jupyterlab
python -m jupyter lab examples/generate_spectrum.ipynb
```

Select this environment's kernel. Restart the kernel after updating the package
so an old import cannot be mistaken for the current code.

## Run and plot a spectrum

```python
import matplotlib.pyplot as plt
from wd_spectra import DAConfig, run_model

config = DAConfig(effective_temperature=12_000, logg=8.0, quality="standard")
run = run_model(config, "results/da-12000")
print(run.selection.workflow)
print("Numerical convergence verified:", run.convergence_verified)

spectrum = run.spectrum
plt.plot(spectrum.wavelength_angstrom, spectrum.surface_flux_lambda)
plt.xlabel("Vacuum wavelength (Angstrom)")
plt.ylabel("Surface flux (erg s^-1 cm^-2 Angstrom^-1)")
plt.show()
```

Temperature is in kelvin; `logg` is log10 of gravity in cm s⁻². The flux is
at the stellar surface, not at Earth: it has not been distance-scaled, reddened,
or convolved with an instrument. The output uses the selected workflow's native
wavelength grid.

Each calculation starts from scratch. Choose a new output directory each time;
existing directories are never overwritten. Iteration progress is printed as
the solver works. Runtime depends strongly on model and machine: allow minutes
for warm models and potentially much longer for cool or metal-rich atmospheres.
Refractive DQ calculations can take hours and several GiB of memory.

### Quality and convergence

Use `standard` to start; `production` increases numerical budgets and is
required by the experimental cool DB/DAB workflows. `quick` is only an
interface smoke test, not a converged science model. DQ currently accepts only
`quality="standard"`; its full cold-start protocol has no quick/production variant.

By default, a completed but unqualified spectrum is saved with a warning for
exploratory work. To make numerical qualification mandatory:

```python
run = run_model(config, "results/da-12000-strict", require_convergence=True)
```

An unqualified result then raises an error after retaining diagnostic files.
Missing data, unsupported parameters, and execution failures always raise;
they never trigger a retry with different physics. Passing the convergence
checks does not certify all physical approximations or grid accuracy; see
[limitations](limitations.md).

DQ is stricter: it always requires both the atmosphere certificate and the
independent final-spectrum flux check, even with `require_convergence=False`.
An unqualified DQ request raises; diagnostic states are retained, but no
exploratory result is returned as successful.

## Change the composition

Replace the configuration above, keeping the same `run_model` call:

```python
from wd_spectra import DBConfig, DABConfig, DAZConfig, DZConfig, DQConfig

helium = DBConfig(effective_temperature=22_000, logg=8.0, quality="standard")
mixed = DABConfig(effective_temperature=20_000, logg=8.0,
                  log_hydrogen_to_helium=-2.0, quality="standard")
polluted = DZConfig(effective_temperature=15_300, logg=8.0, quality="standard")
polluted_hydrogen = DAZConfig(effective_temperature=11_820, logg=8.40,
                             quality="standard")
carbon_helium = DQConfig(effective_temperature=9347, logg=8.041,
                        log_carbon_to_helium=-4.107, quality="standard")
```

`log_hydrogen_to_helium=-2` means N(H)/N(He) = 0.01, not a hydrogen mass
fraction. DZ defaults to a bundled GD 40 composition; supplying an
`abundances` dictionary replaces that complete dictionary. See the
[composition guides](models/README.md) for details. Example parameters are
not guarantees of convergence or paper-spectrum reproduction. For DAZ, metal
abundances are relative to hydrogen, and the defaults describe G29-38.

### DQ helium/carbon atmospheres

For a hydrogen-free, nonmagnetic helium atmosphere with trace carbon:

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

`log_carbon_to_helium` is log10 N(C nuclei)/N(He nuclei), not a mass fraction
or C₂ molecule abundance. These parameters remain fixed; nothing is fitted.
The solver constructs its own gray seed and wavelength grid: no saved
atmosphere, previous spectrum, observation, or external DQ download is needed.

DQ uses refractive transfer for both structure and final synthesis. Success
requires all five atmosphere checks plus a finite, positive spectrum on an
independent 154000-point grid with
`abs(F_bol/(sigma Teff^4) - 1) <= 0.002`. Surface flux is not rescaled.
Progress and diagnostic checkpoints are retained under `worker/`; the public
API does not accept restart inputs. The 28800-second budget includes structure
and final synthesis, not a promise of completion within that time.

This is a preliminary classical-DQ implementation, not a guarantee of
observational agreement or convergence for all parameters. Hot carbon-dominated,
hydrogen-bearing and pressure-distorted DQp atmospheres are outside its scope.
See [DQ physics and limitations](models/DQ.md) and the
[qualified release point](tested-temperature-ranges.md#dq-release-qualification).

### Cool helium and mixed atmospheres

The automatic interface screens local material conditions before solving and
selects the dense-helium or molecular workflow when indicated. These workflows
currently require this source checkout, `quality="production"`, log g = 8,
and integer-K temperatures. Their Python dependencies are included in the
normal installation; there is no separate dependency extra to enable.

Molecular DAB additionally needs the checksum-pinned public tables in the
[data setup instructions](../research/cool_models/README.md#additional-molecular-dab-data).
Supply their directory as `research_data=...` or set `OPENWD_RESEARCH_DATA`.
OpenWD does not download those data during a model calculation.

```python
cool_db = run_model(
    DBConfig(effective_temperature=8000, logg=8.0, quality="production"),
    "results/db-8000", require_convergence=True,
)
cool_dab = run_model(
    DABConfig(effective_temperature=7500, logg=8.0, quality="production",
              log_hydrogen_to_helium=-2.0),
    "results/dab-7500", research_data="/path/to/pinned-molecular-tables",
    require_convergence=True,
)
```

Read the [tested points](tested-temperature-ranges.md) before extrapolating
these examples to other temperatures, gravities, or mixtures.

## Output files

`run.spectrum_path` locates the saved wavelength/flux table, and
`run.output_directory` contains the complete run record.

- `model-run.json`: input request, selected physics, cold-start provenance,
  completion status, and numerical qualification.
- Established presets also write `atmosphere.npz`, `spectrum.txt`, and
  `metadata.json`; the latter contains the atmosphere certificate.
- Experimental cool calculations retain their detailed solver products and
  independent audits under `worker/`. Their dedicated checkers determine
  `run.convergence_verified`; do not reconstruct them with an unrelated EOS.
- DQ writes the common atmosphere/spectrum/metadata files and retains its
  certificate, input provenance, progress and independent spectrum under
  `worker/`. `worker/run.json` records the independent flux ratio and checksum;
  the fine-grid data are in `worker/independent-spectrum.npz`.
- The notebook optionally saves PNG and PDF plots alongside the numerical
  results. Display normalization does not change the saved physical flux.

Established presets use formal-integral spectrum synthesis; DQ instead uses
its refractive transfer on an independent wavelength grid. Neither rescales
the flux. A spectrum's `bolometric_flux`
property integrates the supplied wavelengths only. To check total flux against
sigma Teff^4, the grid must cover the thermal spectrum and resolve its lines.
See [spectrum accuracy](limitations.md#spectrum-accuracy-and-reference-comparisons).

## Explicit presets and command-line examples

The existing `compute_da`, `compute_daz`, `compute_db`, `compute_dab`, and `compute_dz`
interfaces return both an atmosphere and a spectrum in memory. They remain
useful when you deliberately want a particular preset, but they do **not**
provide the automatic dense/molecular workflow selection. Prefer `run_model`
when changing parameters across regimes.

`compute_dq` also returns an atmosphere and spectrum in memory, but always
uses the same isolated refractive cold-start worker and mandatory qualification
as `run_model(DQConfig(...), ...)`. See the [direct DQ example](models/DQ.md#run-a-model).

The `examples/one_shot_*.py` scripts use these explicit presets, for example:

```bash
python examples/one_shot_da.py --teff 12000 --logg 8 --quality standard
```

Run a script with `--help` for its options. No saved atmosphere is needed.
For a deeper explanation of selection and certification, see the
[technical reference](development/history/reliability-2026-09-07.md).

## DO/DAO hot H/He atmospheres

Use the explicit `DOConfig` (pure helium) or `DAOConfig` (mixed H/He).
A high temperature in `DAConfig` or `DBConfig` does not select NLTE.
These experimental models require external CCC/TLUSTY files; see the
[DO/DAO setup](models/DO-DAO.md#external-data) before running.

```python
from wd_spectra import DOConfig, ModelData, run_model

run = run_model(
    DOConfig(effective_temperature=50_000, logg=8, quality="standard"),
    "results/do-50000",  # new directory; always starts from scratch
    data=ModelData.default("/path/to/data"),
    require_convergence=True,
)
```

The CLI equivalent is:

```bash
python examples/one_shot_do.py --teff 50000 --data-root /path/to/data --output results/do-50000
```

Add `--log-hydrogen-to-helium 2` for DAO. Expect roughly one to several hours
for the tested standard/production configurations, depending on hardware.
`quick` is an intentionally short smoke test and does not establish equilibrium.
The [tested points](models/DO-DAO.md#cold-start-qualification-and-runtime)
and remaining observed-profile discrepancies delimit current qualification.
