"""DA upper-atmosphere thermal-branch restarts.

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


def _atmosphere(temperature, failures, *, convective=None, worst=None, branch=None, local=None):
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
        **({} if branch is None else {"frozen_field_thermal_branch_seed": branch}),
        **({} if local is None else {"local_newton_thermal_seed": local}),
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
    """Each solver call returns the next (temperature, failures[, branch]) entry."""
    calls = []

    def solve(teff, logg, *, n_depth, tau_max, **kwargs):
        temperature, failures, *seeds = results[min(len(calls), len(results) - 1)]
        calls.append(dict(kwargs, n_depth=n_depth, tau_max=tau_max))
        seeds = (seeds + [None, None])[:2]
        return _atmosphere(temperature, failures, branch=seeds[0], local=seeds[1])

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
    record, = atmosphere.metadata["thermal_branch_restarts"]
    assert record["kind"] == "deeper"
    assert record["layers"] == [5]
    assert record["temperatures_seeded"] == [t[4]]
    assert record["failures"] == []
    assert atmosphere.metadata["equilibrium_certificate"]["failures"] == []
    assert atmosphere.metadata[
        "radiative_equilibrium_iterations_including_thermal_branch_restarts"
    ] == 20
    assert calls[0]["thermal_branch_seed"] is True


def test_unmoved_front_is_tried_shallower_then_stops(monkeypatch):
    t = _front_temperature()
    failing = ["solver_completion", "temperature_stationarity"]
    calls, atmosphere = _run(monkeypatch, DAConfig(**_COOL), [(t, failing)])
    # Same front after each attempt: deeper, then shallower, then nothing left.
    assert len(calls) == 3
    assert calls[2]["initial_temperature"][4] == t[5]
    records = atmosphere.metadata["thermal_branch_restarts"]
    assert [r["kind"] for r in records] == ["deeper", "shallower"]
    # No attempt certified: the original unfinished solve is returned.
    assert atmosphere.metadata["equilibrium_certificate"]["failures"] == failing


def test_relocation_budget_and_disable(monkeypatch):
    t = _front_temperature()
    failing = ["local_energy"]
    calls, _ = _run(monkeypatch, DAConfig(**_COOL, thermal_branch_restarts=1), [(t, failing)])
    assert len(calls) == 2
    calls, atmosphere = _run(monkeypatch, DAConfig(**_COOL, thermal_branch_restarts=0), [(t, failing)])
    assert len(calls) == 1
    assert "thermal_branch_restarts" not in atmosphere.metadata
    assert calls[0]["thermal_branch_seed"] is False


@pytest.mark.parametrize("failures", [
    [],
    ["all_depth_flux"],
    ["local_energy", "source_closure"],
])
def test_other_outcomes_are_not_relocated(monkeypatch, failures):
    calls, atmosphere = _run(monkeypatch, DAConfig(**_COOL), [(_front_temperature(), failures)])
    assert len(calls) == 1
    assert "thermal_branch_restarts" not in atmosphere.metadata


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_relocation_count_is_validated(bad):
    with pytest.raises(ValueError, match="thermal_branch_restarts"):
        compute_da(DAConfig(**_COOL, thermal_branch_restarts=bad))


def test_frozen_field_branch_seed_is_tried_before_moving_the_front(monkeypatch):
    t = _front_temperature()
    branch_t = t.copy()
    branch_t[:5] = 4000.0
    branch = {"temperature": branch_t, "layers": [0, 1, 2, 3, 4]}
    failing = ["solver_completion", "local_energy"]
    calls, atmosphere = _run(monkeypatch, DAConfig(**_COOL), [
        (t, failing, branch), (t, failing, branch), (t, []),
    ])
    assert np.array_equal(calls[1]["initial_temperature"], branch_t)
    # The same branch seed is not repeated; the front move comes next.
    assert calls[2]["initial_temperature"][5] == t[4]
    records = atmosphere.metadata["thermal_branch_restarts"]
    assert [r["kind"] for r in records] == ["branch", "deeper"]
    assert records[0]["layers"] == [0, 1, 2, 3, 4]
    assert records[0]["temperatures_seeded"] == [4000.0] * 5
    assert records[-1]["failures"] == []


def test_stable_branch_root_follows_the_residual_sign():
    from wd_spectra._thermal_branches import stable_branch_temperature

    grid = np.linspace(2000.0, 5000.0, 31)
    # Heating below 2800 K, cooling to 3200 K, heating again to 3800 K:
    # stable roots at 2800 and 3800 K, unstable at 3200 K.
    balance = -(grid - 2800.0) * (grid - 3200.0) * (grid - 3800.0)
    assert stable_branch_temperature(grid, balance, 3000.0, True) == pytest.approx(3800.0)
    assert stable_branch_temperature(grid, balance, 3500.0, False) == pytest.approx(2800.0)
    assert stable_branch_temperature(grid, balance, 3900.0, True) is None
    assert stable_branch_temperature(grid, balance, 2500.0, False) is None


def test_frozen_field_seed_moves_only_stuck_layers():
    from wd_spectra._thermal_branches import frozen_field_branch_seed, radiative_layers

    wavelength = np.array([1.0, 2.0])
    temperature = np.array([3000.0, 3000.0, 3000.0, 6000.0])
    # Two-level toy: J is fixed, kappa switches on above 3400 K, and the
    # Planck function is linear in T, so the balance is J - T in each band.
    mean = np.full((2, 4), 3600.0)

    def absorption_at(values):
        return np.where(values > 3400.0, 1.0, 0.01)[np.newaxis, :].repeat(2, axis=0)

    def planck(lam, values):
        return np.repeat(np.asarray(values, float)[np.newaxis, :], lam.size, axis=0)

    stuck = radiative_layers(np.array([0.0, 0.0, 0.0, 0.5]), 4)
    assert stuck.tolist() == [True, True, False, False]
    stuck[1] = False  # layer 1 already balanced
    seed, changed = frozen_field_branch_seed(
        temperature, wavelength, mean, np.array([0.3, 0.0, 0.3]), stuck,
        np.linspace(2000.0, 5000.0, 61), absorption_at, planck)
    assert changed == [0]
    assert seed[0] == pytest.approx(3600.0, abs=50.0)
    assert np.array_equal(seed[1:], temperature[1:])


def test_local_newton_seed_is_the_last_candidate(monkeypatch):
    t = _front_temperature()
    local_t = t.copy()
    local_t[3:7] *= 1.0001
    local = {"temperature": local_t, "layers": [3, 4, 5, 6]}
    failing = ["solver_completion", "temperature_stationarity"]
    calls, atmosphere = _run(monkeypatch, DAConfig(**_COOL), [
        (t, failing, None, local), (t, failing, None, local),
        (t, failing, None, local), (t, []),
    ])
    records = atmosphere.metadata["thermal_branch_restarts"]
    assert [r["kind"] for r in records] == ["deeper", "shallower", "local-newton"]
    assert np.array_equal(calls[3]["initial_temperature"], local_t)
    assert records[-1]["failures"] == []


def test_local_newton_seed_solves_a_coupled_block():
    from wd_spectra._thermal_branches import local_newton_seed

    root = np.array([3000.0, 3100.0, 3300.0, 3500.0, 3600.0, 5000.0])

    def residual_at(values):
        x = np.log(values / root)
        # Nonlinear, neighbour-coupled cells; the last cell is convective.
        r = -4.0 * x[:-1] + 0.5 * np.roll(x, -1)[:-1] + 30.0 * x[:-1] ** 2
        return r

    temperature = root * np.exp([0.0, 0.0, 4e-4, -3e-4, 0.0, 0.0])
    radiative = np.array([True, True, True, True, True, False])
    seed, layers = local_newton_seed(
        temperature, residual_at(temperature), radiative, residual_at, half_width=1)
    assert layers == [1, 2, 3]
    assert np.max(np.abs(residual_at(seed))) < 1e-2 * np.max(np.abs(residual_at(temperature)))
    assert seed[0] == temperature[0] and seed[4] == temperature[4]


def test_local_newton_seed_requires_a_real_reduction():
    from wd_spectra._thermal_branches import local_newton_seed

    temperature = np.array([3000.0, 3100.0, 3200.0])

    def residual_at(values):
        # A residual minimum with no root: Newton cannot halve it.
        return 1e-3 + (np.log(values[:-1] / 3000.0)) ** 2

    assert local_newton_seed(temperature, residual_at(temperature),
                             np.array([True, True, True]), residual_at) is None
