"""PG1159 formal-model Rydberg upgrades and the local-energy gate fallback."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from wd_spectra._pg1159_rydberg import (
    _hydrogenic_oscillator_strength,
    _principal_quantum_number,
    _stout_principal_n,
    _van_regemorter_gbar,
)
from wd_spectra.models.pg1159 import (
    PG1159Config,
    _gate_stall_monitor,
    _LocalEnergyGateStall,
    validate_config,
)
from wd_spectra.pg1159 import PG1159NLTEModel


def test_defaults_enable_the_validated_policy():
    config = PG1159Config()
    assert config.rydberg_upgrades is True
    assert config.refinement_local_maximum_rosseland_depth == pytest.approx(1e-3)
    assert config.local_energy_minimum_rosseland_depth == pytest.approx(1e-5)
    assert config.maximum_local_energy_minimum_rosseland_depth == pytest.approx(1e-3)
    # The structure model keeps its historical gate unless the public model
    # deepens it.
    field = PG1159NLTEModel.__dataclass_fields__[
        "atmosphere_local_energy_minimum_rosseland_depth"
    ]
    assert field.default == pytest.approx(1e-5)


@pytest.mark.parametrize(
    "change",
    [
        {"rydberg_upgrades": 1},
        {"refinement_local_maximum_rosseland_depth": 0.0},
        {"local_energy_minimum_rosseland_depth": float("nan")},
        {"maximum_local_energy_minimum_rosseland_depth": 2.0},
        {"local_energy_minimum_rosseland_depth": 1e-2},
    ],
)
def test_invalid_policy_options_are_rejected(change):
    with pytest.raises(ValueError):
        validate_config(replace(PG1159Config(), **change))


def test_hydrogenic_oscillator_strengths_match_tabulated_values():
    # Hydrogen Lyman alpha, Balmer alpha, Paschen alpha (Wiese & Fuhr 2009).
    assert _hydrogenic_oscillator_strength(1, 2) == pytest.approx(0.4164, rel=0.01)
    assert _hydrogenic_oscillator_strength(2, 3) == pytest.approx(0.6407, rel=0.01)
    assert _hydrogenic_oscillator_strength(3, 4) == pytest.approx(0.8421, rel=0.02)


def test_van_regemorter_gbar_has_the_ion_floor():
    assert _van_regemorter_gbar(1.0e6) == pytest.approx(0.2)
    assert _van_regemorter_gbar(10.0) > 0.2


@pytest.mark.parametrize(
    "label, n",
    [("1s2.7d.(2D<5/2>)", 7), ("O608S", 8), ("O611", 11), ("C414", 14), ("junk", None)],
)
def test_principal_quantum_number_parses_stout_tmad_and_shell_labels(label, n):
    assert _principal_quantum_number(label) == n


def test_stout_principal_n():
    assert _stout_principal_n("n=12") == 12
    assert _stout_principal_n(" 1s2.5g") == 5
    assert _stout_principal_n("1s2.2p") == 2
    assert _stout_principal_n("2s2") is None


def _stall_diagnostics(index, residual, step, gate=1e-5, flux=0.1):
    return {
        "local_energy_minimum_rosseland_depth": gate,
        "maximum_local_energy_balance_depth_index": index,
        "maximum_relative_cell_energy_balance_residual": residual,
        "nonlinear_step": step,
        "maximum_all_depth_total_flux_residual": flux,
    }


ATMOSPHERE = SimpleNamespace(rosseland_optical_depth=np.geomspace(1e-7, 100.0, 50))
BOUNDARY = int(np.flatnonzero(ATMOSPHERE.rosseland_optical_depth >= 1e-5)[0])


def test_gate_monitor_raises_on_a_boundary_stall_and_forwards_iterations():
    seen = []
    monitor = _gate_stall_monitor(lambda i, a, p, d: seen.append(i))
    monitor(1, ATMOSPHERE, None, _stall_diagnostics(BOUNDARY, 0.37, 0.025))
    monitor(2, ATMOSPHERE, None, _stall_diagnostics(BOUNDARY, 0.37, 0.0125))
    with pytest.raises(_LocalEnergyGateStall):
        monitor(3, ATMOSPHERE, None, _stall_diagnostics(BOUNDARY, 0.37, 0.0008))
    assert seen == [1, 2, 3]


@pytest.mark.parametrize(
    "index, residual",
    [(BOUNDARY + 1, 0.37), (BOUNDARY, 0.01)],
)
def test_gate_monitor_ignores_interior_cells_and_small_residuals(index, residual):
    monitor = _gate_stall_monitor(None)
    for i in range(5):
        monitor(i, ATMOSPHERE, None, _stall_diagnostics(index, residual, 1e-4))


def test_gate_monitor_needs_two_boundary_iterations_and_tolerates_other_payloads():
    monitor = _gate_stall_monitor(None)
    monitor(1, ATMOSPHERE, None, _stall_diagnostics(BOUNDARY, 0.37, 1e-4))
    # Refinement iterations carry different diagnostics; they reset the count.
    monitor(2, ATMOSPHERE, None, {"temperature_change": 0.01})
    monitor(3, ATMOSPHERE, None, _stall_diagnostics(BOUNDARY, 0.37, 1e-4))


def test_gate_monitor_raises_on_a_boundary_crawl_after_the_flux_converged():
    monitor = _gate_stall_monitor(None)
    # Large flux error: the boundary cell may lead while the flux settles.
    for i in range(10):
        monitor(i, ATMOSPHERE, None, _stall_diagnostics(BOUNDARY, 0.05, 0.05, flux=0.05))
    for i in range(7):
        monitor(10 + i, ATMOSPHERE, None, _stall_diagnostics(BOUNDARY, 0.05, 0.05, flux=0.005))
    with pytest.raises(_LocalEnergyGateStall, match="converged-flux"):
        monitor(17, ATMOSPHERE, None, _stall_diagnostics(BOUNDARY, 0.05, 0.05, flux=0.005))


def test_gate_monitor_crawl_count_resets_when_another_cell_leads():
    monitor = _gate_stall_monitor(None)
    for i in range(20):
        index = BOUNDARY if i % 5 else BOUNDARY + 1
        monitor(i, ATMOSPHERE, None, _stall_diagnostics(index, 0.05, 0.05, flux=0.005))
