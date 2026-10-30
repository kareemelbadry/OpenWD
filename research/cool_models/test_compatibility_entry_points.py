"""Historical import names still refer to the package implementation."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("name,canonical", [
    ("dense_thermal_verification", "wd_spectra._cool.dense_thermal_verification"),
    ("extended_thermal_wavelength_experiment", "wd_spectra._cool.dense_db_driver"),
    ("mass_conservative_feautrier", "wd_spectra._mass_feautrier"),
])
def test_historical_import_alias_from_another_directory(name, canonical, tmp_path):
    wrappers = Path(__file__).resolve().parent
    source = wrappers.parents[1] / "src"
    code = (
        "import importlib; "
        f"legacy = importlib.import_module({name!r}); "
        f"canonical = importlib.import_module({canonical!r}); "
        "assert legacy is canonical"
    )
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True,
                   env=dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (source, wrappers)))))
