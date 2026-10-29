# Contributor guide

[Documentation home](../README.md)

This guide is for people who change OpenWD. To compute spectra, read
[getting started](../getting-started.md).

## Install for development

1. Clone the repository and go to its root directory.
2. Install the package and the test dependencies:

   ```bash
   python -m pip install -e '.[test]'
   ```

3. Make sure that Python imports this checkout and the C extension:

   ```bash
   python -c "import wd_spectra; print(wd_spectra.__file__)"
   python -c "from wd_spectra import compiled_backend_available; print(compiled_backend_available())"
   ```

The first command must print a path in `src/wd_spectra` of this checkout.
The second command must print `True`.

## C extension

The C extension `wd_spectra._rt` contains the radiative-transfer,
line-profile and Stark-manifold kernels. The sources are in `csrc/`.
`setup.py` declares the extension.

- The editable install compiles the extension into `src/wd_spectra/`.
- The extension is optional. If compilation fails, the install continues
  and OpenWD uses the slower NumPy code. Check the result with
  `compiled_backend_available()`.
- After you change a file in `csrc/`, compile again with
  `python -m pip install -e '.[test]'` or `python setup.py build_ext --inplace`.
- Compile once for each Python version that you use. The file name of the
  compiled module includes the Python version.
- CI requires the extension. A CI job fails if `_rt` does not import.

[Performance](performance.md) describes the threads and the benchmarks.

## Run the tests

Run all tests from the repository root:

```bash
pytest
```

- `pyproject.toml` sets `addopts = "-ra -m 'not canary'"`. Thus, `pytest`
  runs all tests in `tests/` except the `canary` tests. Canary tests compute
  full atmospheres from a cold start and take hours.
- The `spectral` marker identifies the fixed-atmosphere spectrum
  regressions. These tests are slow. To skip them, use
  `pytest -m 'not canary and not spectral'`. CI uses this selection for the
  unit tests.
- To run one file, give its path, for example
  `pytest tests/test_documentation.py`.
- The cool-model component tests are in `research/cool_models`. Run them with
  `PYTHONPATH=src:research/cool_models pytest research/cool_models -o addopts=-ra`.

### Import path

pytest reads its import path from `pyproject.toml` (`pythonpath = ["src"]`).
It puts `src/` of this checkout first on `sys.path`. This setting has
priority over the `PYTHONPATH` environment variable.

- To test a different source tree, use `-o pythonpath=<tree>/src`.
- If pytest does not read this `pyproject.toml` (for example, with `-c`),
  add `-o pythonpath=src`.

## Validation tiers

`tools/validate.py` runs the slow tests in tiers. Each test runs in its own
process with one numerical thread. Each test has its own log. The runner
writes `summary.json` with the results, the commands, and hashes of the
source, data and compiled modules.

| Tier | Contents | When to run |
| --- | --- | --- |
| `fast` | `pytest tests -m 'not canary and not spectral'` and the cool-model component tests | After each change |
| `spectra` | Fixed-atmosphere spectrum regressions for DA, DB, DAB, DAH, DZ, DAZ, DQ, D6 and sdB | After a change to opacity, line profiles or transfer |
| `regression` | J1637 (D6) two-step warm trajectory for five frozen reference versions | After a change that affects D6 |
| `cold` | Cold-start canaries: each protected model from parameters, with all convergence checks | For a final candidate |
| `full` | `fast`, then `spectra`, then `cold`; stops at the first failed tier | Before a release |

The `spectra` tier does not test convergence. Only a passed `full` run sets
`full_qualification` to `true` in `summary.json`.

Options:

| Option | Effect |
| --- | --- |
| `--case NAME` | Selects one case. Repeat it for more cases. Only for `spectra`, `regression` and `cold`. |
| `--jobs N` | Maximum number of processes. The default is the smaller of 2 and the CPU count. |
| `--timeout S` | Time limit for each task in seconds (default 7200). A timeout is a failure. |
| `--output DIR` | Output directory. The default is `results/validation/<UTC time>-<id>`. |
| `--plan` | Prints the commands and does not run them. |
| `--reuse FILE` | Reuses passed `spectra` and `cold` results from an earlier `summary.json`, only if all numerical inputs are identical. `fast` always runs again. |

Examples:

```bash
python tools/validate.py fast
python tools/validate.py spectra --case da-4000 --case dz-pg1225
python tools/validate.py regression --case d6-j1637 --jobs 1
python tools/validate.py cold --case da-4000 --jobs 1
python tools/validate.py full --jobs 2 --output results/validation-candidate
```

The `SPECTRA`, `COLD` and `REGRESSIONS` lists in `tools/validate.py` give the
case names. `regression --case d6-j1637` selects all five J1637 versions.

The `molecular-dab-*` cold cases need two external tables: HITRAN
`H2-He_2011.cia` and ExoMol `1H2__RACPPK.states.bz2`. Put them in the
directory given by `OPENWD_RESEARCH_DATA` (default `.cache/molecular-opacity`).
Check them with `python research/cool_models/check_data.py`. Missing data
cause a failure. The [cool-model guide](../../research/cool_models/README.md)
gives the details.

## GitHub checks

### Tests workflow

`.github/workflows/tests.yml` runs on each pull request, on each push to
`main`, and on manual start. All jobs use one numerical thread.

| Job | Contents |
| --- | --- |
| `unit-and-component` | Python 3.9 and 3.12, four shards each (`tools/ci_shard.py`). Each shard runs `pytest -m 'not canary and not spectral'` on its test files. The job first checks the dependencies, the import path and the C extension. |
| `fixed-spectra` | Python 3.12, four lanes of `tools/validate.py spectra`. On a pull request, it runs only if numerical inputs changed. |
| `cool-workflow-components` | `pytest research/cool_models` on Python 3.9 and 3.12. |

### Protected model canaries

`.github/workflows/canaries.yml` runs on pull request events, every Monday
at 09:17 UTC, and on manual start (**Run workflow**).

1. `plan` runs `tools/validation_policy.py`. It sets two flags:
   - *required*: the pull request changes numerical inputs.
   - *requested*: the pull request is not a draft and has the
     `full-validation` label. Scheduled and manual runs are always requested.
2. `preflight` runs the complete `fast` and `spectra` tiers on separate
   workers. It runs only for scheduled and manual runs. On a pull request,
   the tests workflow runs these tiers.
3. `cold-model` runs each cold case in its own job, with
   `tools/validate.py cold --timeout 19800`. On a pull request, the J1637
   warm regression replaces the J1637 cold start.
4. `d6-regression` runs on pull requests only. It runs the five J1637
   reference versions in separate jobs.
5. `qualification` is the summary check. It fails if a required run was not
   requested, or if a requested job failed. Select this check in branch
   protection.

`tools/validation_policy.py` does not require cold starts for changes only to
these paths:

- `docs/`, `README.md`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, `.gitignore`;
- `tools/validation_policy.py`;
- test files in `tests/*.py`, except the canary and spectral test files that
  the policy lists;
- `research/test_*` files and `examples/generate_spectrum.ipynb`;
- the descriptive fields `sigma` and `v_over_i_level` in
  `src/wd_spectra/data/validation/dah/re-j0317-853-vi.json`.

A change to any other path requires cold starts. The same rule decides
whether `fixed-spectra` runs on a pull request.

### Procedure for a numerical change

1. Work on a draft pull request, or keep the `full-validation` label off.
   The `qualification` check fails with an explanation. This is correct.
2. When the candidate is ready, mark the pull request ready for review.
3. Add the `full-validation` label.
4. Each new commit runs the selected jobs again.
5. Rerun the observational comparisons that the change affects.

A pass on a pull request is not a full cold-start qualification, because
J1637 uses the warm regression. The weekly and manual runs include the J1637
cold start.

## Regression controls

Frozen reference outputs are in `tests/data/`. The directory
`tests/data/approved_regressions/` holds the controls for intentional physics
changes. Its [README](../../tests/data/approved_regressions/README.md) and the
README in each subdirectory record each change.

Rules:

- Do not compute a reference inside a test.
- Do not relax a tolerance or increase an iteration limit to make a test pass.
- Do not change an existing control. Earlier controls stay in the tests.

To approve an intentional physics change:

1. Get a physics review and the approval of a maintainer.
2. Write a capture script `research/capture_<topic>_controls.py`. The script
   must refuse an existing output directory. Tests and CI never run it.
3. Save the new outputs in a new directory
   `tests/data/approved_regressions/<topic>_<date>/`.
4. Add a `README.md` and a `manifest.json` to that directory. The manifest
   records the source, data, input and output hashes.
5. If equations or physical data change, update `_MODEL_PHYSICS_REVISION` in
   `src/wd_spectra/models/common.py`.

## Rules for numerical changes

- Public functions return spectra from unconverged atmospheres with an
  `AtmosphereConvergenceWarning`. The result metadata gives
  `atmosphere_convergence_status`. Tests must make the warning an error, or
  assert the convergence metrics.
- A cold-start canary must pass all five convergence checks: all-depth flux,
  local energy, unrestricted temperature correction, scattering-source closure
  and lower-boundary screening.
- A cold-start canary never starts from a saved atmosphere. `run_model`
  accepts no saved atmosphere.
- Iteration limits are work limits, not convergence criteria.
- Show the effect of a solver change with the [solver telemetry](solver-telemetry.md).
- A new model needs its own cold-start canary and fixed-spectrum control.

## Development notes

- [Performance](performance.md): threads and benchmarks.
- [Solver telemetry](solver-telemetry.md): convergence diagnostics in the
  atmosphere metadata.
- [Hot DA/DAO trace metals](hot-daz-trace-metals.md): the fixed-host NLTE
  metal solver.
- [Niobium, zinc, copper and nickel lines](niobium.md): HS 0209+0832.
- [Hot subdwarf (sdB) models](sdb-hybrid.md): solver options and the trace-metal
  driver.
- [History](history/README.md): dated investigation records.

Validation records:

- [Independently corroborated Kurucz lines](../validation/independent-kurucz.md)
- [Observational comparisons: DZ, DBZ and DAZ](../validation/dz-observational-comparisons.md)
- [Metal population consistency](../validation/metal-population-consistency.md)
