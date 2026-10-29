# DQ: helium atmospheres with carbon

[Model guide](README.md) · [Getting started](../getting-started.md)

`DQConfig` computes hydrogen-free, nonmagnetic, helium-dominated LTE
atmospheres with trace carbon and C₂. The transfer is refractive in both the
structure and the final spectrum. Each request is a cold start from a gray
seed. OpenWD does not fit observations.

All DQ data are bundled; see [Getting started](../getting-started.md) for the
installation.

## Run a model

```python
from wd_spectra import DQConfig, run_model

run = run_model(
    DQConfig(
        effective_temperature=9347,
        logg=8.041,
        log_carbon_to_helium=-4.107,  # log10 N(C)/N(He)
    ),
    "results/dq-9347",                # must be a new directory
    require_convergence=True,
)
spectrum = run.spectrum  # vacuum wavelength (Å), surface F_lambda
```

A DQ run can take several hours and several GiB of memory. It runs in an
isolated worker with one numerical thread. Progress is printed and written to
`worker/progress.json` in the output directory. Run one object at a time.

`compute_dq(config, output_directory=...)` returns the result in memory. Its
optional `wavelength` array changes only the output grid.

## Options

| Option | Default | Effect |
| --- | --- | --- |
| `effective_temperature` | 8000 K | Teff. |
| `logg` | 8.0 | log10 of the surface gravity (cgs). |
| `log_carbon_to_helium` | -5.0 | log10 N(C)/N(He). Allowed range: -12 to -3. |
| `quality` | `"standard"` | The only accepted value. |
| `maximum_seconds` | 28800 | Wall-clock budget. The run raises an error when it is used up. |
| `include_c2_ca` | `True` | C₂ C–A band opacity. `False` is a diagnostic omission. |

## Physics

- **Equation of state.** The helium fluid uses the He-REOS.3 table
  (Becker et al. 2014). He, C and C₂ are in coupled chemical equilibrium.
- **Continuum.** The He⁻ free–free opacity includes the dense-helium
  correction of Iglesias et al. (2002), as applied by Blouin et al. (2018).
- **C₂ Swan bands.** The Hornkohl/Parigger line list (Parigger et al. 2015) is
  scaled to the band strength of Brooke et al. (2013). ExoMol 8states
  (Yurchenko et al. 2018; McKemmish et al. 2020) fills the levels outside its
  coverage. The density shift of the bands follows Blouin et al. (2019).
- **Other C₂ systems.** ExoMol 8states gives the other systems and the
  partition function. The Deslandres–d'Azambuja (C–A) system uses band
  envelopes from the Einstein coefficients of Lino da Silva (2024).
- **Atomic carbon.** C I–IV lines come from the Atomic Line List v3
  (van Hoof et al. 2018).
- **Transfer.** Refractive radiative transfer includes bent and trapped
  rays, scattering and a screened lower boundary. The method follows
  Blouin et al. but is not an exact copy.
- **Sampling.** The structure uses every fourth wavelength node of the
  combined line and continuum grid. The final spectrum uses the full grid.
- **Acceptance.** A run must pass the five atmosphere checks. It must then
  pass an independent bolometric check on 218,520 wavelength points:
  |F_bol/(σ Teff⁴) − 1| ≤ 0.002. No flux is rescaled. A failed check raises
  an error.

## Validation

[![SDSS spectra of four DQ white dwarfs compared with OpenWD predictions](../assets/dq-paper.png)](../assets/dq-paper.pdf)

SDSS spectra (gray) and OpenWD predictions (orange) of SDSS J0940+0210,
J1225+4706, J0804+1714 and J1803+2320 (4400–7256 K). Parameters come from
Koester & Kepler (2019) for J0940 and Blouin & Dufour (2019) for the others.
The flux scale uses (R/d)², with R² = GM/g and the published masses and
parallaxes. Nothing is fitted. J1225 is a public cold start. The other three
models were relaxed from saved research atmospheres. The models reproduce the
main Swan bands. Differences in the continuum and band shapes grow toward
the coolest star. J0804 and J1803 are peculiar DQs.

## Limitations

- Only trace-carbon, hydrogen-free, nonmagnetic DQs are in scope. Hot
  carbon-dominated DQs, hydrogen-bearing DQs and DQp stars are not.
- The dense-mixture EOS, the molecular collision profiles and the gray
  refractive ML2 coupling are approximations.
- In the coolest, densest stars, the band and continuum shapes differ from
  the observations.
- The C–A system is a band-envelope approximation, not a resolved line list.
- The bolometric check fixes the integrated flux, not its wavelength
  distribution. The stride-four structure sampling is not a 1% spectral bound.

See [limitations](../limitations.md) for the shared caveats.

## Tested range

See [tested points](../tested-temperature-ranges.md).

## Regression tests

The fast test suite covers the chemistry, transfer and controller. The fixed
spectrum check, `python tools/validate.py spectra --case dq-j1311`, uses the
current default physics. The cold canary solves J1235 from scratch:

```sh
python tools/validate.py cold --case dq-j1235 --jobs 1 --timeout 28800 --output results/dq-cold-check
```

The [transfer-kernel record](../development/history/dq-ca-default-2026-09-17.md)
describes the bitwise tests of the DQ transfer kernels.
