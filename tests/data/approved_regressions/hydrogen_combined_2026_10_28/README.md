# Combined hydrogen dissolution and Montreal defaults

These fixed-state controls cover the intentional combination of PR #52's
full hydrogen dissolved fraction with PR #54's Montreal neutral-H radius
and Ali–Griem defaults. The model-request physics revision retains both
revision tokens. The optional mean-3D line-profile mode is disabled for
these controls.

Neither predecessor directory is changed:

- `hydrogen_dissolution_2026_10_27` records standalone PR #52.
- `montreal_h_radius_2026_10_27` records standalone PR #54.

Offline capture uses `research/capture_hydrogen_combined_controls.py`.
For each case it reloads the same frozen structure inside each research
scope and independently reproduces PR #54, standalone PR #52 and the
pre-both reference at the consuming test's existing tolerance. Only after
all three recoveries pass does it save the combined spectrum. No flux is
fitted or rescaled. Every original structure, grid, reference and tolerance
is retained. Capture inputs are hashed and must remain unchanged, including
when resuming completed cases.

The manifest records both predecessors, their input hashes, their numerical
recovery, differences from each standalone physics choice, and the
nonadditive interaction. `source_sha256` identifies the actual captured
implementation. A new interpreter is used for every case to isolate the
research scopes.

The regular spectral tests check the combined outputs and replay the
immediate main/PR #54 predecessor before retaining the earlier historical
checks. The standalone PR #52 outputs remain immutable and are also
numerically recovered by this offline capture. Hash-integrity checks cover
all three predecessor recoveries and all 25 combined cases.

These are fixed-atmosphere synthesis controls, not cold-start convergence
certificates or independent observational validation. No atmosphere is
iterated by capture. Protected cold starts, all five J1637 warm regressions,
and the unchanged physical tolerances remain required.
