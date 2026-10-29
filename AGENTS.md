# Using OpenWD with an AI assistant

OpenWD calculates a white-dwarf atmosphere and its emergent spectrum from
Teff, log g and composition. Use the configuration classes and `run_model`.

## 1. Check the installation

If OpenWD is not installed, follow [installation](docs/installation.md). Run
all commands from the repository root. Then check the environment:

```bash
python -c "import sys, wd_spectra; print(sys.executable); print(wd_spectra.__file__)"
python -c "from wd_spectra import compiled_backend_available; print(compiled_backend_available())"
```

- For an editable install, the package path is `src/wd_spectra` in this
  checkout.
- `False` from the second command means that the C extension is not active.
  The calculation still works, but it is slower.

## 2. Run one model

Save this script as `first_spectrum.py` and run `python first_spectrum.py`.
The model uses only bundled data and takes several minutes.

```python
import matplotlib.pyplot as plt
from wd_spectra import DAConfig, run_model

run = run_model(
    DAConfig(effective_temperature=12_000, logg=8.0, quality="standard"),
    "results/my-first-da",  # must be a new directory
    require_convergence=True,
)
print("Converged:", run.convergence_verified)
print("Spectrum:", run.spectrum_path)

spectrum = run.spectrum
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
ax.plot(spectrum.wavelength_angstrom, spectrum.surface_flux_lambda)
ax.set_xlabel("Vacuum wavelength (Å)")
ax.set_ylabel(r"Surface $F_\lambda$ (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
fig.savefig(run.output_directory / "spectrum.png", dpi=200)
```

The flux is at the stellar surface. It has no distance scaling, extinction or
instrumental broadening.

## 3. Check convergence

- `run.convergence_verified` is the convergence result. `model-run.json` in the
  output directory records the same value.
- With `require_convergence=True`, a run that does not converge raises
  `RuntimeError`. The output files stay for diagnosis.
- A spectrum file alone does not show success.
- PG 1159 reports `spectrum-qualified` or `converged` in `metadata.json`.
  `require_convergence=True` accepts only `converged`. See
  [getting started](docs/getting-started.md#pg-1159-hot-heliumcarbonoxygen-atmospheres).
- `quality="quick"` only tests the interface. It does not converge.

## 4. Choose the next task

| Task | Start here |
| --- | --- |
| Change the class or the composition | [Getting started](docs/getting-started.md) |
| Read the physics and options of a class | [Model guides](docs/models/README.md) |
| Check if parameters were tested | [Tested ranges](docs/tested-temperature-ranges.md), [limitations](docs/limitations.md) |
| Fit an observed DZ spectrum | [DZ fitting example](docs/examples/fitting-dz.md) |
| Change the code | [Development guide](docs/development/README.md) |

## Rules

- Read the model guide before you change the class. Abundance conventions,
  external data and runtime differ between classes.
- Run one model before a grid or a fit.
- Do not change the solver, the physics or the convergence checks to make a
  run pass. Do not rescale the flux.
- Keep the inputs, the results and the diagnostics of failed runs.
- Report the input parameters, the commit (`git rev-parse HEAD`), any local
  changes, the output directory and the convergence result.
- After you edit documentation, run `python -m pytest tests/test_documentation.py`.
