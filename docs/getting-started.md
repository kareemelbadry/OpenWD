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
unchanged. Established-preset data are bundled.

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

### Quality and convergence

Use `standard` to start; `production` increases numerical budgets and is
required by the experimental cool DB/DAB workflows. `quick` is only an
interface smoke test, not a converged science model.

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

## Change the composition

Replace the configuration above, keeping the same `run_model` call:

```python
from wd_spectra import DBConfig, DABConfig, DZConfig

helium = DBConfig(effective_temperature=22_000, logg=8.0, quality="standard")
mixed = DABConfig(effective_temperature=20_000, logg=8.0,
                  log_hydrogen_to_helium=-2.0, quality="standard")
polluted = DZConfig(effective_temperature=15_300, logg=8.0, quality="standard")
```

`log_hydrogen_to_helium=-2` means N(H)/N(He) = 0.01, not a hydrogen mass
fraction. DZ defaults to a bundled GD 40 composition; supplying an
`abundances` dictionary replaces that complete dictionary. See the
[composition guides](models/README.md) for details. Example parameters are
not guarantees of convergence or paper-spectrum reproduction.

### Cool helium and mixed atmospheres

The automatic interface screens local material conditions before solving and
selects the dense-helium or molecular workflow when indicated. These workflows
currently require this source checkout, `quality="production"`, log g = 8,
integer-K temperatures, and the research dependencies:

```bash
python -m pip install -e '.[research]'
```

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
- The notebook optionally saves PNG and PDF plots alongside the numerical
  results. Display normalization does not change the saved physical flux.

## Explicit presets and command-line examples

The existing `compute_da`, `compute_db`, `compute_dab`, and `compute_dz`
interfaces return both an atmosphere and a spectrum in memory. They remain
useful when you deliberately want a particular preset, but they do **not**
provide the automatic dense/molecular workflow selection. Prefer `run_model`
when changing parameters across regimes.

The `examples/one_shot_*.py` scripts use these explicit presets, for example:

```bash
python examples/one_shot_da.py --teff 12000 --logg 8 --quality standard
```

Run a script with `--help` for its options. No saved atmosphere is needed.
For a deeper explanation of selection and certification, see the
[technical reference](development/history/reliability-2026-09-07.md).
