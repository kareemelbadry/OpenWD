"""Column-mass nodes moved onto the intended Rosseland mesh (wd_spectra._domain)."""

from dataclasses import dataclass, field

import numpy as np
import pytest

from wd_spectra._domain import (
    ROSSELAND_MESH_TOLERANCE,
    remesh_to_rosseland_depths,
    rosseland_mesh_deviation,
    solve_with_screened_boundary,
)
from wd_spectra.atmosphere import photosphere_concentrated_optical_depths


@dataclass(frozen=True)
class _Structure:
    column_mass: np.ndarray
    rosseland_optical_depth: np.ndarray
    temperature: np.ndarray
    gas_pressure: np.ndarray
    metadata: dict = field(default_factory=dict)
    gravity: float = 1.0e8

    @property
    def n_depth(self):
        return self.column_mass.size


def _relaxed(column_mass):
    # A steep tau(m), as for He- free-free in cool helium.
    tau = 1.0e-12 * (column_mass / column_mass[0]) ** 2.5
    return _Structure(
        column_mass=column_mass,
        rosseland_optical_depth=tau,
        temperature=6000.0 * (0.75 * (tau + 2.0 / 3.0)) ** 0.25,
        gas_pressure=1.0e8 * column_mass,
        metadata={"radiative_equilibrium_iterations": 3, "equilibrium_certificate": {"failures": []}},
    )


def test_seed_mesh_is_far_from_and_remesh_lands_on_the_rosseland_target():
    # Nodes crowded near the bottom, as left by the gray seed.
    column_mass = np.geomspace(1.0e-6, 1.0e2, 60) ** np.linspace(1.0, 0.55, 60)
    column_mass = np.sort(column_mass)
    relaxed = _relaxed(column_mass)
    assert rosseland_mesh_deviation(relaxed, 1.0) > 2.0
    options = remesh_to_rosseland_depths(relaxed, 1.0)
    new_m = options["initial_column_mass"]
    assert new_m[0] == column_mass[0] and new_m[-1] == column_mass[-1]
    assert new_m.size == column_mass.size and np.all(np.diff(new_m) > 0)
    target = photosphere_concentrated_optical_depths(
        relaxed.rosseland_optical_depth[0], relaxed.rosseland_optical_depth[-1], 60, 1.0
    )
    np.testing.assert_allclose(options["initial_rosseland_optical_depth"], target, rtol=1e-12)
    # tau(m) is a power law here, so the new nodes reproduce the target exactly.
    np.testing.assert_allclose(_relaxed(new_m).rosseland_optical_depth, target, rtol=1e-9)
    assert rosseland_mesh_deviation(_relaxed(new_m), 1.0) < 1e-6
    assert options["resume_supplied_structure_in_formal_flux_phase"] is True
    # Temperature and pressure are interpolated in log m.
    np.testing.assert_allclose(options["initial_gas_pressure"], 1.0e8 * new_m, rtol=1e-6)


def test_screened_solve_remeshes_until_the_nodes_follow_the_target():
    calls = []

    def solver(**options):
        calls.append(options)
        mass = options.get("initial_column_mass")
        if mass is None:
            mass = np.sort(np.geomspace(1.0e-6, 1.0e2, 50) ** np.linspace(1.0, 0.55, 50))
        return _relaxed(np.asarray(mass))

    result = solve_with_screened_boundary(solver, remesh_depth_concentration=1.0)
    deviations = result.metadata["rosseland_mesh_deviations"]
    assert len(calls) == 2 and deviations[0] > ROSSELAND_MESH_TOLERANCE
    assert deviations[-1] <= ROSSELAND_MESH_TOLERANCE
    assert calls[1]["resume_supplied_structure_in_formal_flux_phase"] is True
    assert result.metadata["adaptive_domain_segments"][-1]["rosseland_remesh"] == 1
    # Without the option nothing is re-solved.
    calls.clear()
    plain = solve_with_screened_boundary(solver)
    assert len(calls) == 1 and plain.metadata["rosseland_mesh_deviations"] == []


def test_remesh_rejects_unordered_nodes():
    relaxed = _relaxed(np.geomspace(1.0, 10.0, 5))
    broken = _Structure(
        column_mass=relaxed.column_mass[::-1],
        rosseland_optical_depth=relaxed.rosseland_optical_depth,
        temperature=relaxed.temperature,
        gas_pressure=relaxed.gas_pressure,
    )
    with pytest.raises(ValueError):
        remesh_to_rosseland_depths(broken, 1.0)
