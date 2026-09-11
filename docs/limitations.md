# Caveats and limitations

[Documentation home](README.md) · [Getting started](getting-started.md)

OpenWD is pre-alpha research software. A useful-looking spectrum, numerical
convergence, agreement with a reference spectrum, and adequate physical
approximations are separate questions.

## Tested temperatures and compositions

Tested cold-start points reach **3000 K for DA, 5000 K for pure-He DB, and
7500 K for a DAB mixture with N(H)/N(He) = 0.01**, at log g = 8. These are
individual tested points, not validity ranges. The cool DB/DAB workflows are
experimental and require the setup described in the [user guide](getting-started.md#cool-helium-and-mixed-atmospheres).

The [tested-point table](tested-temperature-ranges.md) is the detailed record
of temperatures, compositions, settings, and convergence evidence. In particular:

- No abundance or gravity grid has been demonstrated.
- DAB at 7250 K remains unqualified, and 5000 K DAB experiments remain far
  from equilibrium. There is no validated 5000–10000 K mixed-atmosphere grid.
- Historical continuation calculations are not evidence for a fresh run.
  Public generation requires no prior model.

## What convergence means

`run.convergence_verified` reports qualification for the selected workflow's
declared equations and numerical checks. Established presets require actual
all-depth flux conservation, local energy balance, a measured unrestricted
temperature correction, scattering-source closure, and lower-boundary screening.
Experimental dense/molecular workflows have dedicated checkers and retained
audit records. A solver's terminal success flag alone is insufficient.

Unqualified completed spectra remain available with a warning for exploration.
Use `require_convergence=True` to require numerical qualification. A failed
calculation never selects an alternative physics prescription automatically.

## Spectrum accuracy and reference comparisons

Atmosphere qualification does not by itself establish wavelength, angle, or
depth-grid independence, nor the accuracy of a separately synthesized spectrum.
The former piecewise-linear spectrum method has known transfer-consistency limits:
a broad-wavelength audit integrated to 0.98613 of the expected stellar flux for
DA 3000 K and 0.99314 for DB 10000 K. These historical numbers are not measurements
of the new cubic DA default. Spectra are not renormalized to hide flux errors.
See the [numerical report](development/history/cold-start-numerics-2026-09-07.md#known-spectrum-consistency-limits-unfinished-changes-excluded).

Different atmosphere and synthesis methods are intentional, not by themselves
a convergence failure. The established Feautrier atmosphere and formal-integral
synthesis remain the defaults; the experimental matched-transfer and forced
fine-wavelength atmosphere calculations are not enabled. Convergence checks
and spectral-accuracy checks remain separate. Default DA regression tests cover
absolute flux and Balmer cores/wings without selecting an alternative method.

DA calculations now default to `synthesis_transfer="formal-pchip"`
for higher-order source interpolation on the same atmosphere. Explicit
`synthesis_transfer="formal-linear"` retains the former interpolation. The checked
12000-K standard model's broad sampled flux ratio improves from about 0.980
to 1.0002 without flux rescaling. This is not a universal flux-conservation
guarantee. Separate checked cold-state controls now protect the new DA default
at 12000 and 20000 K; the historical spectral controls remain unchanged.
See the [DA guide](models/DA.md#spectrum-synthesis).

Regression controls protect previously calculated spectra; they are not
independent observational validation. Published-grid and observational spectra
are not distributed in this repository. The fresh production SDSS J0738+1835
atmosphere passes static checks but does not pass the paper-spectrum comparison.
PG 1225's production cold-start check is also distinct from exact reproduction
of the lower-resolution paper model. These distinctions are recorded in the
[tested-point details](tested-temperature-ranges.md#limits-and-preservation-of-established-results).

The undoubled Q-MHD critical-field correction intentionally changes some
warm-model higher-series features beyond the old spectral regression bounds.
Reviewed corrected outputs are now protected by separate regression controls;
the historical files and numerical tolerances have not been overwritten or
relaxed. This is not a claim of improved agreement at every wavelength.
See the [microphysics audit](development/history/microphysics-audit-2026-09-10.md)
for the distinction between corrected equations, cold-start convergence, and
preservation of historical spectra.

## Physical approximations

All current modules are plane-parallel LTE models. DAB/DBA assumes a homogeneous
mixture, not a stratified hydrogen layer; DAZ assumes a hydrogen-dominated host,
and DZ/DBZ a helium-dominated host, with fixed input abundances. Hot NLTE,
magnetic, PG 1159, and D6 models are not
part of the public modules.

The dense pure-He treatment combines tabulated bulk thermodynamics with
approximate chemical potentials and trace-ion chemistry. Refraction and
collective He-minus corrections remain absent. Molecular mixtures still lack
a consistent dense-mixture free energy, nonideal dissociation, some molecular
ions, and pressure-distorted CIA. The [full limitations list](tested-temperature-ranges.md#limits-and-preservation-of-established-results)
and [model guides](models/README.md) describe the scope in more detail.

Automatic selection identifies the relevance of implemented physics; it does
not establish convergence or supply missing physics. Unsupported overrides,
missing data, or invalid material domains are reported explicitly.
