# Synthetic regression controls

These are **OpenWD outputs**, not observations or published-grid spectra.
Captured from immutable Git commit `263b92e77bf207eb21ee8bd7b25f2b9312fb6828`
on 2026-09-07 with Python 3.9.16, NumPy 1.26.4, SciPy 1.11.1 and the compiled
backend, using one thread. DA/DB structures were cold starts with the stored
configuration. The two DAB and two DZ structures are recovered paper-model
checkpoints, used for fixed-state synthesis, not new cold-start successes.

Each NPZ contains the original atmosphere arrays/metadata, serialized request,
original synthetic flux, and (except the two-wavelength standard DB control)
the independently source-checked flux at the **same** atmosphere and opacity.
The spectrum uses 1100 logarithmic nodes from 900 to 300000 Angstrom, augmented
by 2-Angstrom optical nodes from 3700 to 7000 and 1-Angstrom UV nodes from 1150
to 1400. No bolometric rescaling or alignment was applied.

The checked source solves the exact Lambda system for the established
piecewise-linear formal transfer. Its independent closure is below 1e-10.
Tests compare new output against that checked reference and separately against
the original GitHub spectrum. The common bounds are 0.1% at wavelengths where
lambda F_lambda exceeds 1% of its peak, and 1e-5 fractional change in each
integrated UV/optical/IR band. Profile reproducibility is checked separately.

These fixtures do **not** establish equilibrium or validate the physical
model. In particular, historical `radiative_equilibrium_converged=True` fields
are retained as historical evidence, not adopted as strict certificates.
Do not regenerate these references merely because a test failed. Any update
requires an explained physical/numerical change and review of both controls.
