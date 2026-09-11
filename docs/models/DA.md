# DA module

[Model guide](README.md) · [Getting started](../getting-started.md)

`compute_da` solves a pure-hydrogen plane-parallel LTE atmosphere in
hydrostatic and radiative/convective equilibrium. It includes Hummer--Mihalas
occupation probabilities, correlated Q-MHD microfields, dissolved-series
opacity, H-/H2/H2+/H3+ chemistry at low temperature, and ML2/alpha=0.7
convection.

Hydrogen bound-bound opacity spans Lyman through Brackett. Charged-particle
profiles use the Tremblay--Bergeron tables. Neutral-H Balmer broadening uses
Ali--Griem below 10,000 K and Barklem at higher temperatures. The validated
Lyman policy uses the fixed TLUSTY/Allard Ly-alpha table at 9000--13,000 K,
temperature-dependent supplied Allard profiles at higher temperature, and the
independent cool neutral-H/H2 wing below 9000 K.

```bash
python examples/one_shot_da.py --teff 12000 --logg 8.0 \
  --quality standard --output results/da-12000-8.0
```

Warm standard models normally take minutes; cool convective production models
can take tens of minutes. `quick` verifies the interface but is not a science
atmosphere. Wavelengths are vacuum Angstrom and output fluxes are surface
`F_lambda`.

Protected cold starts reach 3000 K at log g = 8, with additional checks at
4000, 5000, and 20000 K. See [tested points](../tested-temperature-ranges.md)
and [limitations](../limitations.md) for settings and the scope of that evidence.

## Spectrum synthesis

`compute_da` uses monotone cubic (PCHIP) source interpolation by default to
reduce coarse-depth-grid flux bias:

```python
import numpy as np
from wd_spectra import DAConfig, compute_da

result = compute_da(
    DAConfig(effective_temperature=12000, logg=8, quality="standard"),
    np.geomspace(100, 1_000_000, 6000),
)
```

This is a cold start. Cubic interpolation changes only the final spectrum calculation:
the atmosphere solver, depth grid, opacities and convection are unchanged.
Scattering is solved consistently with the cubic interpolation, with a
separate source-closure check. There is no fitted flux scaling or imposed
bolometric normalization, and failure does not select another method.
The selected method is recorded in `result.spectrum.metadata`.

On the checked 12000-K, log-g=8 standard model, this reduces the broad sampled
flux deficit from about 2% to about 0.02%, adding roughly 1-2 seconds of
transfer work. It can also change normalized Balmer profiles by about 1-2%
on that 40-layer grid. Wavelength, depth and full-physics validation remain
separate requirements. To explicitly reproduce the former interpolation
method, pass `synthesis_transfer="formal-linear"`; this does not undo physics
corrections in the EOS. Both methods are independently source-checked.
Low-level synthesis routines retain their explicit legacy defaults, including
the directional-intensity interface. Other public models' synthesis defaults
are unchanged.
