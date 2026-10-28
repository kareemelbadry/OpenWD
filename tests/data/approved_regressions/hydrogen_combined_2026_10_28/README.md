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

## Measured changes

Maximum changes below use the region where wavelength times the predecessor flux exceeds 1% of its peak. The interaction is the nonadditive part of the two changes, relative to the pre-both spectrum. These are spectral diagnostics on fixed structures.

| Control | Change from main / PR #54 | Change from standalone PR #52 | Nonadditive interaction |
| --- | ---: | ---: | ---: |
| `da-20000` | 1.3053% | 0.0016% | 0.0000% |
| `dab-9000` | 0.0594% | 0.0000% | 0.0000% |
| `dab-20000` | 0.0967% | 0.0075% | 0.0000% |
| `dz-pg1225` | 0.0040% | 13.0997% | 0.0016% |
| `dz-j0738` | 0.0004% | 14.0012% | 0.0000% |
| `public-da-12000-public` | 2.4389% | 0.2893% | 0.0166% |
| `public-da-20000` | 1.3039% | 0.0016% | 0.0000% |
| `daz-g149_28` | 0.3054% | 5.4140% | 0.0222% |
| `daz-galex1931` | 1.0810% | 0.0007% | 0.0000% |
| `dah-default-j1007+1237` | 1.1486% | 0.0019% | 0.0001% |
| `dah-paper-j1007+1237` | 1.5199% | 0.0024% | 0.0001% |
| `dah-default-j1034+0327` | 2.9274% | 0.0318% | 0.0038% |
| `dah-paper-j1034+0327` | 4.2846% | 0.0431% | 0.0074% |
| `dah-default-j1154+0117` | 0.6450% | 0.0000% | 0.0000% |
| `dah-paper-j1154+0117` | 1.7048% | 0.0001% | 0.0000% |
| `dah-default-j2149-0728` | 0.3749% | 0.0002% | 0.0000% |
| `dah-paper-j2149-0728` | 1.2848% | 0.0006% | 0.0000% |
| `dah-default-j1254+5612` | 0.8655% | 0.1732% | 0.0046% |
| `dah-paper-j1254+5612` | 2.9031% | 0.3308% | 0.0306% |
| `dah-default-j1018+0111` | 0.6722% | 0.4303% | 0.0114% |
| `dah-paper-j1018+0111` | 2.2246% | 1.8366% | 0.0785% |
| `dah-default-j1351+5419` | 2.5785% | 0.0550% | 0.0102% |
| `dah-paper-j1351+5419` | 6.7111% | 0.2077% | 0.0361% |
| `dah-default-j2247+1456` | 1.6073% | 0.0016% | 0.0002% |
| `dah-paper-j2247+1456` | 2.5753% | 0.0039% | 0.0002% |
