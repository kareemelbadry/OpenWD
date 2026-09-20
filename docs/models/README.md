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
| `DQConfig` | Helium with trace carbon and C₂ Swan bands; preliminary | [DQ physics and setup](DQ.md) |
| `DOConfig` | Pure helium; experimental restricted NLTE | [DO/DAO physics and data](DO-DAO.md) |
| `DAOConfig` | Homogeneous H/He; experimental restricted NLTE | [DO/DAO physics and data](DO-DAO.md) |

`DOConfig` and `DAOConfig` add experimental hot helium and mixed H/He
[restricted NLTE models](DO-DAO.md). Their LTE charge closure and incomplete
observational qualification are documented explicitly.

All configurations describe plane-parallel atmospheres; DO/DAO add restricted
NLTE populations while retaining an LTE charge/pressure closure. They predict spectra for
specified parameters; they do not fit observations. Established presets and
experimental cool workflows have different applicability limits; see
[limitations](../limitations.md) and [tested points](../tested-temperature-ranges.md).

The individual guides also describe `compute_*` and command-line presets.
The established explicit presets do not automatically switch to the cool
dense/molecular workflows; use the [automatic interface](../getting-started.md)
for that. `compute_dq` is different: it uses the same isolated refractive cold
worker and mandatory atmosphere-plus-spectrum qualification as `run_model`.
