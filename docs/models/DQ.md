# DQ: refractive helium/carbon atmospheres

[Model guide](README.md) · [Getting started](../getting-started.md)

`DQConfig` predicts hydrogen-free, nonmagnetic, helium-dominated atmospheres
with trace carbon and C₂. The public calculation starts from its own
hydrostatic gray seed. It does not read a previous atmosphere, model spectrum,
observed spectrum, object-name lookup, or neighboring grid model.

## Run a model

Install OpenWD normally; the necessary constitutive tables are bundled. Model
generation performs no downloads. The DQ runtime requires NumPy, SciPy and
Numba, installed as package dependencies.

```python
from wd_spectra import DQConfig, run_model

run = run_model(
    DQConfig(
        effective_temperature=9347,
        logg=8.041,
        log_carbon_to_helium=-4.107,  # log10 N(C nuclei)/N(He nuclei)
        maximum_seconds=28800,
    ),
    "results/my-dq",               # must be a new directory
    require_convergence=True,
)
```

These are inputs, not fitting starting guesses: OpenWD does not adjust them.
Arbitrary finite positive temperatures and finite gravities are accepted,
with `-12 <= log_carbon_to_helium <= -3`; this is not a guarantee of
convergence or data-table coverage everywhere in that range. Only
`quality="standard"` is exposed for this protocol. Hot carbon-dominated DQs,
hydrogen-bearing DQs and DQp molecular distortion are outside its scope.

For an in-memory result and explicit diagnostics directory:

```python
from wd_spectra import compute_dq

result = compute_dq(
    DQConfig(9347, 8.041, -4.107),
    output_directory="results/my-dq-direct",
)
```

The returned spectrum is unscaled surface `F_lambda` in
`erg s^-1 cm^-2 Angstrom^-1`, on a vacuum-wavelength grid. An optional
`wavelength=` array affects only output synthesis; it cannot reduce the
structure sampling or bypass the independent flux test.

Calculations can take hours and several GiB. Each request runs in an isolated
worker using one numerical thread, so its process-local DQ solver adapters do
not modify another spectral class in the caller. Run separate objects as
separate requests, allowing for their memory requirements. Progress is printed
and written to `worker/progress.json` and `worker/automatic-phases.json` under
`run_model` (directly in the chosen directory under `compute_dq`).

## What is required for success

The same shared nonlinear solver, ML2 helpers and screened-domain controller
are used with DQ material/coordinate/transfer adapters. The current protocol
uses current-state energy normalization, EOS-corner-aware thermal proposals,
automatic between-phase coordinate refreshes, and one-cell lower-boundary
extensions. Pseudo-time conditioning is only an initializer; its looser
transient contraction target is not a final atmosphere tolerance.

A successful public request must pass all five atmosphere checks: all-depth
flux, local energy balance, temperature stationarity, source closure, and
lower-boundary screening. It must then pass an independent **154000-point**
synthesis: 4000 logarithmic points from 1000 to 100000 Å plus 150000 optical
midpoints at 0.02 Å spacing between 3800 and 6800 Å. All fluxes must be finite
and positive, and `abs(F_bol/(sigma Teff^4) - 1) <= 0.002`. No flux
normalization or parameter adjustment is applied. Source integrity and data
checksums are recorded.

Failed checks or exhausted budgets raise an error. Accepted diagnostic states
and any intermediate products remain available; an atmosphere certificate
alone does not cause an unqualified spectrum to be returned as successful.
The public API deliberately has no saved-state restart mode.

## Physics and limitations

The implementation includes He-REOS.3 host thermodynamics, coupled He/C/C₂
chemistry, dense-helium continuum corrections, and conservative refractive
transfer with bent and trapped rays, scattering, and a screened lower
boundary. Structure and synthesis use the same carbon-line declaration and
molecular opacity. UV sampling is determined without reference to a past
spectrum.

Resolved thermal Swan lines use the Hornkohl/Parigger list, with one absolute
Brooke band-rate calibration and a density shift. The shift is integrated
over each depth cell; this is spatial quadrature, not extra collision
broadening. Other C₂ systems and the partition function use ExoMol 8states.
The continuum table and all molecular data are checksum-pinned; their
provenance is in the bundled data and [third-party notices](../../THIRD_PARTY_NOTICES.md).

This follows important ingredients of the Blouin approach but is not an exact
reproduction. Dense-mixture EOS, molecular collision profiles and the grey
refractive ML2 bridge remain approximations. Numerical convergence does not
guarantee observational agreement, abundance accuracy, or independent depth
convergence. These limitations are recorded in every result rather than
silently marked as validated.

See the [release validation record](../development/history/dq-release-2026-09-17.md)
for tests and the distinction between earlier research cold starts and the
packaged release qualification.

## Regression tests

The normal fast test suite covers chemistry, refractive conservation and
responses, controller transitions, and public cold-start/qualification
contracts. The separate fixed-atmosphere spectrum test is included in
`python tools/validate.py spectra --case dq-j1235`; this is not cold-start
evidence. To run the expensive public cold canary explicitly:

```sh
python tools/validate.py cold --case dq-j1235 --jobs 1 --timeout 28800 --output results/dq-cold-check
```

This test fixes the J1235 parameters, supplies no atmosphere or spectrum,
and retains its worker outputs in the validation artifacts. It is also part
of the scheduled/manual protected-canary workflow, not the fast push tests.
