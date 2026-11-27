"""DA upper-atmosphere thermal-front relocation.

The structure solver is stubbed: these tests check which failed solves are
re-solved, from what seed, and what the result records.
"""

from dataclasses import replace

import numpy as np
import pytest
from wd_spectra import gray_hydrogen_atmosphere
from wd_spectra.models import stellar
from wd_spectra.models.stellar import (
    DAConfig,
    _da_thermal_front_seed,
    _da_upper_thermal_front,
    compute_da,
)

_COOL = dict(
    effective_temperature=4_500.0, logg=7.5, quality="production",
    lyman_profile_source="stark", include_molecules=False,
)
_N = 12


def _front_temperature():
    # Hot H- cap over a cold molecular layer: the drop is between layers 4|5.
    t = np.array([3840, 3838, 3836, 3830, 3800, 3050, 2880, 2870, 2900, 3300,
                  4000, 5000], dtype=float)
    return t


def _atmosphere(temperature, failures, *, convective=None, worst=None):
    atmosphere = gray_hydrogen_atmosphere(4_500.0, 7.5, n_depth=_N)
    if convective is None:
        convective = np.zeros(_N)
        convective[9:] = 0.5
    residual = np.full(_N - 1, 1e-6)
    residual[5 if worst is None else worst] = 2.5e-2
    return replace(atmosphere, temperature=np.asarray(temperature, float), metadata={
        "equilibrium_certificate": {"failures": list(failures)},
        "radiative_equilibrium_iterations": 10,
        "convective_flux_fraction_by_interface": convective,
        "cell_energy_balance_relative_residual": residual,
        "lower_boundary_absorption_escape_bound": 1e-6,
    })


def test_front_is_the_largest_radiative_inversion():
    atmosphere = _atmosphere(_front_temperature(), ["local_energy"])
    assert _da_upper_thermal_front(atmosphere) == (4, 5)


def test_monotonic_structure_has_no_front():
    atmosphere = _atmosphere(np.linspace(3000.0, 6000.0, _N), ["local_energy"])
    assert _da_upper_thermal_front(atmosphere) is None


def test_inversions_in_convective_layers_are_not_fronts():
    temperature = np.linspace(3000.0, 6000.0, _N)
    temperature[7] = 6500.0  # drop 7|8 inside the convection zone
    convective = np.zeros(_N)
    convective[6:] = 0.5
    atmosphere = _atmosphere(temperature, ["local_energy"], convective=convective, worst=7)
    assert _da_upper_thermal_front(atmosphere) is None


def test_front_far_from_the_worst_layer_is_ignored():
    atmosphere = _atmosphere(_front_temperature(), ["local_energy"], worst=0)
    assert _da_upper_thermal_front(atmosphere) is None
    atmosphere = _atmosphere(_front_temperature(), ["local_energy"], worst=6)
    assert _da_upper_thermal_front(atmosphere) == (4, 5)


def test_front_seed_moves_one_layer_either_way():
    t = _front_temperature()
    deeper = _da_thermal_front_seed(t, (4, 5), "deeper")
    shallower = _da_thermal_front_seed(t, (4, 5), "shallower")
    assert deeper[5] == t[4] and np.array_equal(np.delete(deeper, 5), np.delete(t, 5))
    assert shallower[4] == t[5] and np.array_equal(np.delete(shallower, 4), np.delete(t, 4))
    with pytest.raises(ValueError):
        _da_thermal_front_seed(t, (4, 5), "sideways")


class _Done(Exception):
    def __init__(self, atmosphere):
        self.atmosphere = atmosphere


def _run(monkeypatch, config, results):
    """Each solver call returns the next (temperature, failures) entry."""
    calls = []

    def solve(teff, logg, *, n_depth, tau_max, **kwargs):
        temperature, failures = results[min(len(calls), len(results) - 1)]
        calls.append(dict(kwargs, n_depth=n_depth, tau_max=tau_max))
        return _atmosphere(temperature, failures)

    def finish(atmosphere, kind):
        raise _Done(atmosphere)

    monkeypatch.setattr(stellar, "radiative_equilibrium_hydrogen_atmosphere", solve)
    monkeypatch.setattr(stellar, "atmosphere_with_model_request_fingerprint", lambda a, f: a)
    monkeypatch.setattr(stellar, "fixed_synthesis_atmosphere", lambda a, f: a)
    monkeypatch.setattr(stellar, "warn_if_atmosphere_not_converged", finish)
    with pytest.raises(_Done) as done:
        compute_da(config)
    return calls, done.value.atmosphere


def test_front_failure_is_resolved_warm_one_layer_deeper(monkeypatch):
    t = _front_temperature()
    calls, atmosphere = _run(monkeypatch, DAConfig(**_COOL), [
        (t, ["solver_completion", "local_energy", "temperature_stationarity"]),
        (t, []),
    ])
    assert len(calls) == 2
    assert calls[0]["initial_temperature"] is None
    seed = calls[1]["initial_temperature"]
    assert seed[5] == t[4] and np.array_equal(np.delete(seed, 5), np.delete(t, 5))
    assert calls[1]["initial_column_mass"] is not None
    # Same mesh as the unfinished structure (the stub always returns _N layers).
    assert calls[1]["n_depth"] == _N
    assert calls[1]["tau_max"] == calls[0]["tau_max"]
    record, = atmosphere.metadata["thermal_front_relocations"]
    assert record["direction"] == "deeper"
    assert (record["upper_layer"], record["lower_layer"]) == (4, 5)
    assert record["failures"] == []
    assert atmosphere.metadata["equilibrium_certificate"]["failures"] == []
    assert atmosphere.metadata[
        "radiative_equilibrium_iterations_including_front_relocations"
    ] == 20


def test_unmoved_front_is_tried_shallower_then_stops(monkeypatch):
    t = _front_temperature()
    failing = ["solver_completion", "temperature_stationarity"]
    calls, atmosphere = _run(monkeypatch, DAConfig(**_COOL), [(t, failing)])
    # Same front after each attempt: deeper, then shallower, then nothing left.
    assert len(calls) == 3
    assert calls[2]["initial_temperature"][4] == t[5]
    records = atmosphere.metadata["thermal_front_relocations"]
    assert [r["direction"] for r in records] == ["deeper", "shallower"]
    # No attempt certified: the original unfinished solve is returned.
    assert atmosphere.metadata["equilibrium_certificate"]["failures"] == failing


def test_relocation_budget_and_disable(monkeypatch):
    t = _front_temperature()
    failing = ["local_energy"]
    calls, _ = _run(monkeypatch, DAConfig(**_COOL, thermal_front_relocations=1), [(t, failing)])
    assert len(calls) == 2
    calls, atmosphere = _run(monkeypatch, DAConfig(**_COOL, thermal_front_relocations=0), [(t, failing)])
    assert len(calls) == 1
    assert "thermal_front_relocations" not in atmosphere.metadata


@pytest.mark.parametrize("failures", [
    [],
    ["all_depth_flux"],
    ["local_energy", "source_closure"],
])
def test_other_outcomes_are_not_relocated(monkeypatch, failures):
    calls, atmosphere = _run(monkeypatch, DAConfig(**_COOL), [(_front_temperature(), failures)])
    assert len(calls) == 1
    assert "thermal_front_relocations" not in atmosphere.metadata


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_relocation_count_is_validated(bad):
    with pytest.raises(ValueError, match="thermal_front_relocations"):
        compute_da(DAConfig(**_COOL, thermal_front_relocations=bad))
