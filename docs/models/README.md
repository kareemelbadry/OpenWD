# Model guide

[Documentation home](../README.md) · [Getting started](../getting-started.md)

Use the configuration for your composition with `run_model`. It selects the
implemented material treatment before solving a new atmosphere.

| Configuration | Composition | Details |
| --- | --- | --- |
| `DAConfig` | Pure hydrogen | [DA physics](DA.md) |
| `DAZConfig` | Hydrogen with metals; abundances relative to H | [DAZ physics](DAZ.md) |
| `DBConfig` | Pure helium | [DB physics](DB.md) |
| `DABConfig` | Homogeneous H/He; `log_hydrogen_to_helium` sets log10 N(H)/N(He) | [DAB/DBA physics](DAB.md) |
| `DZConfig` | Helium with metals and optional trace hydrogen | [DZ/DBZ physics](DZ.md) |

All five describe plane-parallel LTE atmospheres. They predict spectra for
specified parameters; they do not fit observations. Established presets and
experimental cool workflows have different applicability limits; see
[limitations](../limitations.md) and [tested points](../tested-temperature-ranges.md).

The individual guides also describe `compute_*` and command-line presets.
Those explicit presets do not automatically switch to the cool dense/molecular
workflows; use the [automatic interface](../getting-started.md) for that.
