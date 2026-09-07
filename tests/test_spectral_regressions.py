"""Broad, fixed-atmosphere controls are not equilibrium certificates.

These protect our synthetic UV, optical lines and IR against regressions.
The atmosphere/cold-start tests separately protect equilibrium diagnostics.
"""

import json
from pathlib import Path
import numpy as np
import pytest
from wd_spectra._compat import trapezoid
from wd_spectra.models import (
    DAConfig,
    DBConfig,
    DABConfig,
    DZConfig,
    compute_da,
    compute_db,
    compute_dab,
    compute_dz,
    load_atmosphere_checkpoint,
    AtmosphereConvergenceWarning,
)

CONTROLS = Path(__file__).parent / "data/spectral_regressions"
CASES = [
    "da-3000",
    "da-4000",
    "da-5000",
    "da-20000",
    "db-10000",
    "db-22000",
    "dab-9000",
    "dab-20000",
    "dz-pg1225",
    "dz-j0738",
]


@pytest.mark.parametrize("case", CASES)
def test_checked_scattering_preserves_broad_spectral_controls(case):
    path = CONTROLS / (case + ".npz")
    with np.load(path) as saved:
        kind = str(saved["spectral_type"])
        config = {"DA": DAConfig, "DB": DBConfig, "DAB": DABConfig, "DZ": DZConfig}[
            kind
        ](**json.loads(str(saved["config_json"])))
        wave = saved["wavelength"]
        old = saved["original_surface_flux"]
        expected = saved["checked_surface_flux"]
    molecular = kind == "DA" and config.effective_temperature <= 12000
    atmosphere = load_atmosphere_checkpoint(
        path,
        config.effective_temperature,
        config.logg,
        {"DA": "hydrogen", "DB": "helium", "DAB": "mixed", "DZ": "helium"}[kind],
        include_molecules=molecular,
        include_negative_hydrogen=molecular,
        log_hydrogen_to_helium=getattr(config, "log_hydrogen_to_helium", None),
        trihydrogen_ion_partition_model="neale-tennyson-1995" if kind == "DA" else None,
    )
    # These historical checkpoints are deliberately uncertified. Synthesis
    # must remain available with a warning, without pretending to re-solve.
    with pytest.warns(AtmosphereConvergenceWarning):
        result = {
            "DA": compute_da,
            "DB": compute_db,
            "DAB": compute_dab,
            "DZ": compute_dz,
        }[kind](config, wave, initial_atmosphere=atmosphere, relax_atmosphere=False)
    new = result.spectrum.surface_flux_lambda
    np.testing.assert_allclose(new, expected, rtol=2e-6, atol=1e-12 * np.max(expected))
    assert result.spectrum.metadata["source_converged"]
    assert result.spectrum.metadata["independent_radiation_scaled_source_error"] < 1e-10
    assert result.metadata["atmosphere_convergence_status"] != "converged"
    # A common 0.1% bound in significant-flux regions is stricter than
    # observational agreement; no star-specific tolerance is tuned here.
    important = wave * old > 0.01 * np.max(wave * old)
    np.testing.assert_allclose(new[important], old[important], rtol=1e-3, atol=0.0)
    for lo, hi in ((1150, 3000), (3500, 7000), (7000, 300000)):
        take = (wave >= lo) & (wave <= hi)
        change = trapezoid(new[take] - old[take], wave[take]) / trapezoid(
            old[take], wave[take]
        )
        assert abs(change) < 1e-5
