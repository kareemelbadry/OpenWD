# J1637 J-resolved Si II reference

This version records the 2026-11-11 correction of the TLUSTY Si II
photoionization zero point. Its seed is copied exactly from the
checksum-verified v5 capsule: Teff, log g, composition, column mass,
optical-depth grid, base temperature, 48 layers, both 25,000-line budgets and
the prescribed 1.5% perturbation are unchanged.

Physics change: TLUSTY's `si2_36+4lev` atom lists the 3p 2Po ground term
once per J, so its first level is the lowest J level, the Stout zero point.
The Stout ground-term centroid (0.0237 eV) was nevertheless added to every
model-atom excitation, which put all 35 Si II edges 0.024 eV too low (for
example 3537 A was placed at 3561 A and 1307 A at 1311 A). LS-coupled atoms
with a single ground-term level are unchanged. The same release renormalizes
the hydrogen Stark-table interpolation, which has no hydrogen to act on here.

The fixed spectrum changes by at most 0.323% (3560.8 A, just redward of the
corrected Si II 3537 A edge) and by -6.1e-5 in integrated flux over the
stored grid. The warm trajectory's temperatures move by up to 2.5e-4. The two
measured warm updates do not establish cold convergence. Source-convergence,
transfer, profile and spectral tolerances are unchanged, including the
fixed-spectrum relative tolerance of 2e-6.
