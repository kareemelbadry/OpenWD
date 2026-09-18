# DQ fixed-state spectral control

`j1235-fixed.npz` contains the 41-layer atmosphere and 79 unscaled flux samples
from the independently qualified J1235 research cold run of 2026-09-17.
`manifest.json` records parameters and source/output checksums.

This fixture is used only by `test_dq_spectral_regression.py` to detect changes
in refractive synthesis and material physics. It is not observed data, is not
proof of physical accuracy, and never initializes a cold-start test.
`test_j1235_public_true_cold_and_independent_spectrum` starts from parameters
alone and is separately marked as a slow canary.

`j1311-completed-fixed.npz` preserves all 30,000 optical samples from the
pre-promotion 2024 C–A + completed-Swan warm model. Its independent broad-band
check passed the unchanged luminosity tolerance; production packaging reproduces
its spectrum bit-for-bit. `completed-manifest.json` records the source hashes.
This separate fixture protects the new default without replacing the historical
control. Rebuild with `tools/build_dq_completed_regression.py --source
results/dq-continuum-followup/j1311-combined-warm --output NEW_DIRECTORY`.
