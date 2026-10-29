# Installation

[Documentation home](README.md) · [Getting started](getting-started.md)

## Requirements

- Python 3.9 or newer. The continuous-integration tests use Python 3.9 and 3.12
  on Linux.
- A C compiler (optional). The installer uses it to build the C extension.

The installer gets these Python packages automatically:

| Package | Minimum version |
| --- | --- |
| NumPy | 1.23 (NumPy 2 is supported) |
| SciPy | 1.10 |
| Matplotlib | 3.6 |
| mpmath | 1.2 |
| Numba | 0.58 |

Two optional groups add more packages:

| Group | Installs | Use |
| --- | --- | --- |
| `test` | pytest | Run the test suite. |
| `validation` | astropy | Read observed spectra in the comparison scripts under `research/`. |

## Install

1. Get the source code:

   ```bash
   git clone https://github.com/kareemelbadry/OpenWD.git
   cd OpenWD
   ```

2. Make a new environment and activate it:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. Install OpenWD in editable mode:

   ```bash
   python -m pip install -e .
   ```

   To add the optional groups, use `python -m pip install -e '.[test,validation]'`.

4. Check the installation:

   ```bash
   python -c "import wd_spectra; print(wd_spectra.__version__, wd_spectra.__file__)"
   python -c "from wd_spectra import compiled_backend_available; print(compiled_backend_available())"
   ```

   The first command shows the version and the package location. For an
   editable install, the location is `src/wd_spectra` in your checkout. The
   second command shows `True` when the C extension is active.

For the example notebooks, install Jupyter in the same environment:

```bash
python -m pip install jupyterlab
python -m jupyter lab examples/generate_spectrum.ipynb
```

Select the kernel of this environment. Restart the kernel after you update
OpenWD.

## C extension

The C extension `wd_spectra._rt` (sources in `csrc/`) makes the formal
transfer, the metal lines, and the hydrogen and helium line profiles faster.
It does not change the physics or the numerical settings.

- `pip install` builds it automatically when a C compiler is available.
  On macOS, install the compiler with `xcode-select --install`. On Linux,
  install `gcc` and the Python development headers.
- If the build fails, the installation continues without the extension.
  OpenWD then uses the equivalent Python code, which is slower.
- To see the compiler output, install with `python -m pip install -v -e .`.
- After you change a file in `csrc/`, run `python -m pip install -e .` again.
- To check the extension, use `compiled_backend_available()` (step 4 above).

## Data files

### Bundled data

The package includes the data that all model classes need, in
`src/wd_spectra/data` (about 230 MB). This includes:

- hydrogen and helium line-profile tables (Tremblay, Allard, Beauchamp,
  TLUSTY/SYNSPEC),
- metal line lists and photoionization data (Kurucz, Stout, NIST ASD,
  Verner, NORAD),
- NLTE model atoms (TLUSTY, TMAD, CHIANTI, SIROCCO) and the CCC electron
  collision data,
- the He-REOS.3 dense-helium equation of state and the H2-H2 collision-induced
  absorption (CIA) table,
- the C₂ opacity tables for DQ models,
- unified He-broadened profiles for Mg, Ca, Na and Si lines,
- the H2db tables for magnetic (DAH) models.

OpenWD does not download data during a calculation. If a bundled file is
missing, the calculation stops with `FileNotFoundError`. Reinstall OpenWD to
restore the file.

### External data

One file is not bundled. Only metal-free DAB/DBA models need it.

| File | `H2-He_2011.cia`: the HITRAN H2-He CIA table from the Abel et al. calculations (147 MB) |
| --- | --- |
| Needed by | Every metal-free `DABConfig` request through `run_model`, at all Teff and log g. Also `compute_dab` with `include_molecules=True`. |
| Not needed by | All other classes, and DAB/DBA models with trace metals (`abundances` set). |
| Source | [HITRAN CIA page](https://hitran.org/cia/), file `H2-He_2011.cia` in the main list. Direct link: <https://hitran.org/data/CIA/main/H2-He_2011.cia> |
| SHA-256 | `4f0eb9cd69a1c383f53a1431495bae0c01a30a41cc1b8433c2d726763ff45431` |
| If missing | `run_model` stops with `FileNotFoundError` before it makes the output directory. A file with a different checksum stops the run with `ValueError`. |

Install the file:

1. Download `H2-He_2011.cia` into a directory of your choice, for example
   `~/openwd-data`.
2. Check the checksum. Compare the result with the SHA-256 value above.

   ```bash
   shasum -a 256 ~/openwd-data/H2-He_2011.cia   # Linux: sha256sum
   ```

3. Tell OpenWD where the file is. Use one of these methods:

   | Method | Example | Used by |
   | --- | --- | --- |
   | Environment variable | `export OPENWD_RESEARCH_DATA=~/openwd-data` | `run_model` |
   | `run_model` argument | `run_model(config, "results/dab", research_data="~/openwd-data")` | `run_model` |
   | Configuration field | `DABConfig(..., h2_he_cia_path="~/openwd-data/H2-He_2011.cia")` | `run_model` and `compute_dab` |
   | Package data directory | Copy the file to `src/wd_spectra/data/runtime/cache/molecular-opacity/` | `run_model` and `compute_dab` |

   The configuration field has priority. Then `run_model` uses the
   `research_data` argument, then `OPENWD_RESEARCH_DATA`, then the package
   data directory.

OpenWD does not redistribute this file. When you use it, follow the HITRAN
citation policy and cite the Abel et al. H2-He calculations. See the
[third-party notices](../THIRD_PARTY_NOTICES.md).

## Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `OPENWD_RESEARCH_DATA` | not set | Directory that contains `H2-He_2011.cia` for molecular DAB/DBA models. |
| `OPENWD_NUM_THREADS` | `min(8, number of CPUs)` | Number of threads for the compiled hydrogen line profiles. Must be a positive integer. |
| `OPENWD_DATA` | `src/wd_spectra/data/runtime` | Replaces the complete runtime data directory. Use it only for a full copy of that directory. |
| `OPENWD_DQ_DATA` | `src/wd_spectra/data/dq` | Replaces the complete DQ data directory. OpenWD checks every file against its checksum. |

## Run the tests

```bash
python -m pip install -e '.[test]'
python -m pytest
```

The default test run omits the slow end-to-end model tests. The
[development guide](development/README.md) describes the full test policy.
