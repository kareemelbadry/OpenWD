# OpenWD documentation

OpenWD calculates white-dwarf model atmospheres and spectra from Teff, log g
and composition. To use OpenWD with an AI assistant, start with
[AGENTS.md](../AGENTS.md).

## Start here

- [Installation](installation.md): requirements, the C extension and the one
  external data file.
- [Getting started](getting-started.md): first run, outputs, convergence and
  one example for each class.
- [Example notebook](../examples/generate_spectrum.ipynb): calculate and plot
  a DA, DB, DAB, DZ or DQ spectrum.

## Model guides

The [model guide index](models/README.md) compares the classes.

- [DA](models/DA.md): hydrogen atmospheres.
- [DB](models/DB.md): helium atmospheres, including the dense-helium workflow.
- [DAB/DBA](models/DAB.md): mixed hydrogen–helium atmospheres.
- [DZ/DBZ/DAZ](models/DZ.md): helium or hydrogen atmospheres with metals.
- [DQ](models/DQ.md): helium atmospheres with trace carbon and C₂.
- [DO/DAO](models/DO-DAO.md): hot NLTE helium and hydrogen–helium atmospheres.
- [PG 1159](models/PG1159.md): hot NLTE helium–carbon–oxygen atmospheres.
- [D6](models/D6.md): carbon–oxygen atmospheres without hydrogen or helium.
- [DAH](models/DAH.md): magnetic hydrogen atmospheres.
- [sdB](models/sdB.md): hot subdwarf atmospheres.

## Reference

- [Limitations](limitations.md): known accuracy limits.
- [Tested ranges](tested-temperature-ranges.md): parameters run from a cold
  start.
- [Output files](getting-started.md#outputs): files, units and convergence
  flags.
- [DZ fitting example](examples/fitting-dz.md): fit abundances to an observed
  spectrum, with the [fitting notebook](../examples/fit_dz_spectrum.ipynb).

## Validation

- [Observational comparisons: DZ, DBZ and DAZ](validation/dz-observational-comparisons.md)
- [Independently corroborated Kurucz lines](validation/independent-kurucz.md)
- [Metal population consistency](validation/metal-population-consistency.md)

## For developers

- [Development guide](development/README.md): tests, regression policy,
  performance and development records.

Data attribution and licenses: [third-party notices](../THIRD_PARTY_NOTICES.md).
