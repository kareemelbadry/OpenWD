# J1637 warm trajectory and fixed spectrum

`warm_v5_excited_parent_o_i/j1637.npz` is the full-resolution regression
reference for the standard J1637 request: 48 layers, 25,000-line budgets for
structure and synthesis, and the normal optical spectrum grid.
`manifest.json` records its hashes, equations, configuration, archived source
certificate, and measured build timings. `warm_v5_excited_parent_o_i-seed.npz`
contains the corresponding input; both hashes are checked, and its arrays must
exactly match those in the reference. The version's own README describes the
physics change it records.

The seed comes from a previously certified standard J1637 **cold** atmosphere.
That certificate applies to its recorded code revision; it does not certify
convergence under the current code. The seed temperature is perturbed
smoothly by up to 1.5% in the line-forming layers, and the current
conservative equations freeze two measured solver steps, including complete
temperature, pressure, density and electron-density profiles and
flux/energy/correction/source residuals. The frozen optical spectrum uses the
unperturbed atmosphere.

The trajectory is intentionally stopped through the iteration callback.
Its success is regression consistency, not a convergence certificate or a
new cold-start qualification. EOS populations, opacity, transfer and
corrections are freshly computed by each test. Tests never regenerate the
reference.

Run the bounded trajectory and independent fixed spectrum with:

```bash
python tools/validate.py regression --case d6-j1637 --jobs 1
python tools/validate.py spectra --case d6-j1637 --jobs 1
```

The full public cold-start canary remains available through
`python tools/validate.py cold --case d6-j1637 --jobs 1`; weekly and manually
dispatched CI still run it. A successful warm regression does not satisfy
`full_qualification` in the local validation report.

An intentional reviewed numerical change requires a new reference and
explicit review of its differences. Build it offline from the checked-in
capsule, which reuses the checksum-verified seed and its archived cold
certificate:

```bash
PYTHONPATH=src python tools/build_d6_regression.py \
  tests/data/d6_regressions/warm_v5_excited_parent_o_i NEW_DIRECTORY \
  --from-frozen-reference --spectrum-first
```

or from a newly certified standard J1637 cold atmosphere:

```bash
PYTHONPATH=src python tools/build_d6_regression.py SOURCE_ATMOSPHERE.npz NEW_DIRECTORY
```

The builder refuses to overwrite an existing directory and records the
differences from the source capsule in the new manifest. Point
`tests/test_d6_regression.py` at the new directory and remove the old one;
earlier references remain in git history.
