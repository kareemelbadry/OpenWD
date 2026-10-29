# Model guide

[Documentation home](../README.md) · [Getting started](../getting-started.md)

Choose the configuration class for your star and pass it to `run_model`. Each
guide shows how to run the model, its options, its physics, the paper
comparison and its tested range.

| Class | Configuration | Composition | Populations | Guide |
| --- | --- | --- | --- | --- |
| DA | `DAConfig` | Pure hydrogen | LTE | [DA](DA.md) |
| DB | `DBConfig` | Pure helium | LTE | [DB](DB.md) |
| DAB, DBA | `DABConfig` | Homogeneous H/He | LTE | [DAB/DBA](DAB.md) |
| DZ, DBZ | `DZConfig` | Helium with metals | LTE | [DZ/DBZ/DAZ](DZ.md) |
| DAZ | `DAZConfig` | Hydrogen with metals | LTE | [DZ/DBZ/DAZ](DZ.md) |
| DQ | `DQConfig` | Helium with trace carbon and C₂ | LTE | [DQ](DQ.md) |
| DO | `DOConfig` | Hot pure helium | NLTE H/He | [DO/DAO](DO-DAO.md) |
| DAO, hot DA | `DAOConfig` | Hot homogeneous H/He | NLTE H/He | [DO/DAO](DO-DAO.md) |
| PG 1159 | `PG1159Config` | Hot He/C/O with trace elements | NLTE He/C/O | [PG 1159](PG1159.md) |
| D6 | `D6Config` | Carbon/oxygen-dominated, no H or He | LTE | [D6](D6.md) |
| DAH | `DAHConfig` | Magnetic pure hydrogen | LTE | [DAH](DAH.md) |
| sdB | `SdBConfig` | Low-gravity H/He; metals with a driver script | LTE structure, NLTE H/He | [sdB](sdB.md) |

## How `run_model` selects the physics

The configuration class sets the physics. `run_model` makes only two further
choices, before it starts the solution. For `DBConfig`, it uses the
dense-helium workflow when the photosphere is dense, neutral helium. For a
metal-free `DABConfig`, it always uses the molecular H/He closure. The choice
is saved in `model-run.json` and never changes after a failure.

A high temperature in `DAConfig`, `DBConfig` or `DABConfig` does not select
NLTE. Use `DOConfig`, `DAOConfig` or `PG1159Config`. The `compute_*` functions
run one fixed preset: `compute_db` does not select the dense-helium workflow,
and `compute_dab` stays atomic unless `include_molecules=True`. See
[limitations](../limitations.md) and [tested points](../tested-temperature-ranges.md).
