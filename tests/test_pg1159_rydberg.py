"""PG1159 formal-model Rydberg upgrades and the local-energy gate fallback."""

from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

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


def test_upgraded_ion_database_does_not_inherit_the_warmed_line_cache():
    from wd_spectra._pg1159_rydberg import _with_ion
    from wd_spectra.metals import AtomicDatabase, AtomicIon, AtomicLevel, AtomicTransition

    levels = tuple(AtomicLevel(i, 100.0 * (i - 1), 2.0, f"level-{i}") for i in (1, 2, 3))
    old = AtomicIon("C", 3, 12.0, 64.5, levels, (AtomicTransition(1, 2, 1e8, "E1", 1550.0, 0.2),))
    new = replace(old, transitions=old.transitions + (AtomicTransition(1, 3, 1e8, "E1", 1200.0, 0.1),))
    database = AtomicDatabase(MappingProxyType({("C", 3): old}))
    database._line_selection_cache[("warm",)] = ((old, old.transitions[0], levels[0]),)
    database._unsold_hydrogen_coefficient_cache[("C", 3, 1, 2)] = None
    upgraded = _with_ion(database, ("C", 3), new)
    assert upgraded.ions[("C", 3)] is new
    assert isinstance(upgraded.ions, MappingProxyType)
    assert upgraded.source == database.source
    assert upgraded._line_selection_cache == {}
    assert upgraded._unsold_hydrogen_coefficient_cache == {}
    assert database._line_selection_cache  # the original keeps its own cache


def test_appended_threshold_entries_follow_the_target_gbar_convention():
    from wd_spectra.light_metal_nlte import tmap_seaton_gbar
    from wd_spectra.pg1159_oxygen import _threshold_cross_sections_for

    tlusty = SimpleNamespace(
        threshold_cross_section_cm2=np.array([1.0, 2.0, 3.0]) * 0.3e-18,
        threshold_cross_section_includes_gbar=True,
    )
    raw = SimpleNamespace(
        threshold_cross_section_cm2=np.array([1.0, 2.0, 3.0]) * 1e-18,
        threshold_cross_section_includes_gbar=False,
    )
    indices = np.array([0, 2])
    assert tmap_seaton_gbar(5) == pytest.approx(0.3)
    np.testing.assert_allclose(_threshold_cross_sections_for(raw, tlusty, indices), [1e-18, 3e-18], rtol=1e-12, atol=0)
    np.testing.assert_allclose(_threshold_cross_sections_for(tlusty, raw, indices), [0.3e-18, 0.9e-18], rtol=1e-12, atol=0)
    np.testing.assert_allclose(_threshold_cross_sections_for(raw, raw, indices), [1e-18, 3e-18], rtol=1e-12, atol=0)


def test_extended_oxygen_vi_high_l_terms_get_one_seaton_gbar():
    from wd_spectra._pg1159_builder import build_model
    from wd_spectra.models.common import ModelData
    from wd_spectra.models.pg1159 import required_atomic_files

    data = ModelData.default()
    if not all(path.is_file() for path in required_atomic_files(data, "extended54-complete")):
        pytest.skip("PG1159 atomic data not installed")
    model, _ = build_model(
        data, {"He": 0.33, "C": 0.48, "O": 0.17}, structure_only=True,
        population_iterations=1, oxygen_atom_preset="extended54-complete",
    )
    table = model.oxygen_photoionization_threshold_data[5]
    assert table.threshold_cross_section_includes_gbar is False
    high_l = [index for index, label in enumerate(table.level_label)
              if label.strip() in ("O VI 2Fo 1", "O VI 2Fo 2", "O VI 2Ge 1")]
    assert len(high_l) == 3
    for index in high_l:
        # Formula-2 amplitude (Mb) is the raw threshold cross section.
        amplitude = table.photoionization_parameters[index][1]
        assert table.threshold_cross_section_cm2[index] == pytest.approx(amplitude * 1e-18, rel=1e-3, abs=0.0)
