# DQ fixed-state spectral control

`j1235-fixed.npz` contains the 41-layer atmosphere and 79 unscaled flux samples
from the independently qualified J1235 research cold run of 2026-09-17.
`manifest.json` records parameters and source/output checksums.

This fixture is used only by `test_dq_spectral_regression.py` to detect changes
in refractive synthesis and material physics. It is not observed data, is not
proof of physical accuracy, and never initializes a cold-start test.
`test_j1235_public_true_cold_and_independent_spectrum` starts from parameters
alone and is separately marked as a slow canary.
