# Limitations

[Documentation home](README.md) · [Getting started](getting-started.md) · [Tested points](tested-temperature-ranges.md)

This page lists what the convergence checks establish and the main physical
approximations. Each model guide gives the details for its class.

## What the convergence checks establish

After the solver stops, OpenWD evaluates the final atmosphere again. It issues
an equilibrium certificate only if five checks pass:

1. The total flux equals σTeff⁴ at every depth (all-depth flux).
2. Each depth cell is in energy balance (local energy).
3. The full, unrestricted Newton temperature correction is small (temperature
   stationarity).
4. The scattering source function is consistent (source closure).
5. The lower boundary does not leak radiation (boundary screening).

Typical tolerances are 2–3e-3 for flux and energy and 2–3e-4 for the
temperature correction. The values are saved in `metadata.json`.

A certificate shows that the declared equations are solved on the structure
grid. It does not show independence of the depth or wavelength grid. It does
not show that the physics is complete or that the spectrum matches a star.

Some classes have different rules:

- **DQ** also requires an independent 218,520-point spectrum with
  |F_bol/(σTeff⁴) − 1| ≤ 0.002. DQ raises an error if any check fails.
- **PG 1159** uses looser checks (1% flux, 0.3% local heating) and no
  temperature-stationarity check. A model that passes is `spectrum-qualified`,
  not `converged`.
- **DO/DAO and sdB** also require converged NLTE populations.
- **DAH** certifies the nonmagnetic structure, not a magnetic radiative
  equilibrium.

A failed calculation never selects a different physics prescription.

### When to use `require_convergence=True`

`run_model` always saves its outputs. If the certificate fails, it gives a
warning and sets `run.convergence_verified` to `False`.

- Set `require_convergence=True` in scripts, grids and fits, so that an
  unconverged spectrum causes an error.
- Leave it `False` to inspect a failed run.
- Do not set it for PG 1159. It accepts only `converged`, so a
  `spectrum-qualified` PG 1159 run always raises an error. Read
  `atmosphere_convergence_status` in `metadata.json` instead.

## Physical approximations

These approximations apply to all classes unless noted:

- **1D, static, plane-parallel atmospheres.** Spherical extension, winds and
  3D convection are not modeled.
- **Homogeneous composition.** Abundances are the same at all depths.
  Gravitational settling, radiative levitation and stratified layers are not
  modeled.
- **LTE.** DA, DB, DAB/DBA, DZ/DBZ/DAZ, DQ, D6 and DAH use LTE populations.
  DO/DAO and PG 1159 use NLTE. sdB uses an LTE structure with NLTE H/He
  populations. NLTE line cores of cool DAs are not modeled.
- **Convection.** Convection uses local ML2 mixing-length theory, with
  α = 0.7 for hydrogen-rich and α = 1.25 for helium-rich and mixed atmospheres.
  Overshoot is not included. For cool DAs, the optional 3D line-profile mode
  improves Balmer cores at Teff ≲ 10,000 K, but its spectra do not conserve
  flux.
- **Trace metals.** In DZ, DBZ and DAZ, metals add electrons and opacity but
  are not in the pressure, density or thermodynamic derivatives. In hot DA/DAO
  and sdB models, NLTE metals are solved on a fixed H/He structure without
  metal blanketing.
- **Atomic data.** Model atoms are truncated. Many collision rates use the van
  Regemorter approximation, and many excited levels use hydrogenic
  photoionization. Missing data can bias single lines.
- **Line broadening in dense helium.** Only a few strong metal lines have
  unified profiles; other lines use the impact approximation.

## Spectrum accuracy and reference comparisons

- Every comparison uses published stellar parameters that other codes
  derived. Agreement shows consistency with earlier analyses, not independent
  accuracy.
- Each class is tested on a small number of stars or reference grids. Outside
  the [tested points](tested-temperature-ranges.md), the accuracy is not known.
- The final spectrum is calculated on a finer wavelength grid than the
  structure. OpenWD never scales the spectrum to correct its total flux.
- Regression tests protect earlier spectra. They are not observational
  validation.

## Class-specific limits

- **[DA](models/DA.md).** Below about 4500 K, the Lyα red wing is the main
  uncertainty. Above about 40,000 K, use `DAOConfig` (NLTE).
- **[DB](models/DB.md).** The dense-helium workflow for cool DBs requires
  `quality="production"` and an integer Teff.
- **[DAB/DBA](models/DAB.md).** The mixture is homogeneous. There are no
  unified H–He line profiles. The molecular closure has no consistent
  dense-mixture free energy. Metals and molecules cannot be combined.
- **[DZ, DBZ and DAZ](models/DZ.md).** In the coolest, most metal-rich stars,
  the wings of the strongest lines are not matched in detail.
- **[DQ](models/DQ.md).** Hydrogen-free, nonmagnetic helium with trace carbon
  only. Hot carbon-rich DQs and DQp stars are excluded. Continuum and Swan-band
  differences increase toward the coolest stars.
- **[DO/DAO](models/DO-DAO.md).** The electron density is LTE. The model atoms
  are restricted. Metals do not enter the structure.
- **[PG 1159](models/PG1159.md).** The outermost temperatures are uncertain.
  Trace elements do not change the structure.
- **[D6](models/D6.md).** No molecules or negative ions. Tested on few stars.
- **[DAH](models/DAH.md).** Nonmagnetic structure, approximate Stark–Zeeman
  profiles, Stokes I by default, hydrogen only.
- **[sdB](models/sdB.md).** The LTE structure is likely inaccurate above about
  40,000 K.
