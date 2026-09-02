from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from wd_spectra import gray_helium_atmosphere
from wd_spectra.cool_metal_nlte import (
    ca_ii_resonance_scattering_probabilities,
)
from wd_spectra.metals import (
    ATOMIC_MASS_U,
    AtomicDatabase,
    AtomicIon,
    AtomicLevel,
    AtomicTransition,
)


def _calcium_database() -> AtomicDatabase:
    levels = (
        AtomicLevel(1, 0.0, 2.0, "4s 2S1/2"),
        AtomicLevel(2, 13_650.0, 4.0, "3d 2D3/2"),
        AtomicLevel(3, 13_711.0, 6.0, "3d 2D5/2"),
        AtomicLevel(4, 25_192.0, 2.0, "4p 2P1/2"),
        AtomicLevel(5, 25_414.0, 4.0, "4p 2P3/2"),
    )
    transitions = (
        AtomicTransition(1, 4, 100.0, "E1", 3969.6, 0.3),
        AtomicTransition(2, 4, 20.0, "E1", 8664.0, 0.02),
        AtomicTransition(1, 5, 200.0, "E1", 3934.8, 0.6),
        AtomicTransition(2, 5, 10.0, "E1", 8500.0, 0.01),
        AtomicTransition(3, 5, 30.0, "E1", 8544.0, 0.03),
    )
    ion = AtomicIon(
        "Ca", 1, ATOMIC_MASS_U["Ca"], 11.87, levels, transitions
    )
    return AtomicDatabase(MappingProxyType({("Ca", 1): ion}))


def _write_scups(path: Path) -> None:
    path.write_text(
        """1 4 2.296e-1 0.0 1.0 2 2 1.0
0.0 1.0
2.0 2.0
1 5 2.316e-1 0.0 1.0 2 2 1.0
0.0 1.0
4.0 4.0
-1
""",
        encoding="ascii",
    )


def test_ca_ii_scattering_includes_radiative_branching_and_electron_destruction(
    tmp_path: Path,
):
    scups = tmp_path / "ca_2.scups"
    _write_scups(scups)
    atmosphere = gray_helium_atmosphere(10_000.0, 8.0, n_depth=8)
    collisionless = replace(
        atmosphere, electron_density=np.zeros(atmosphere.n_depth)
    )
    database = _calcium_database()

    low_density = ca_ii_resonance_scattering_probabilities(
        collisionless, database, scups
    )
    assert np.all(low_density[(1, 4)].probability == pytest.approx(100.0 / 120.0))
    assert np.all(low_density[(1, 5)].probability == pytest.approx(200.0 / 240.0))

    high_density = ca_ii_resonance_scattering_probabilities(
        replace(
            atmosphere,
            electron_density=np.full(atmosphere.n_depth, 1.0e16),
        ),
        database,
        scups,
    )
    assert np.all(
        high_density[(1, 4)].probability < low_density[(1, 4)].probability
    )
    assert np.all(
        high_density[(1, 5)].probability < low_density[(1, 5)].probability
    )
