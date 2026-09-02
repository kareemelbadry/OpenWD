from pathlib import Path

import numpy as np

from wd_spectra.dense_eos import read_helium_reos3_table
from wd_spectra.eos import (
    hummer_mihalas_helium_lte,
    hummer_mihalas_helium_lte_at_nuclei_density,
    hummer_mihalas_helium_lte_with_reos3,
)


def test_reos_reader_inverts_exact_table_nodes(tmp_path: Path) -> None:
    path = tmp_path / "he-reos.dat"
    np.savetxt(
        path,
        np.asarray([
            [1.0e-4, 1_000.0, 1.0e-3, -2.0],
            [1.0e-3, 1_000.0, 1.0e-2, -1.0],
            [2.0e-4, 2_000.0, 2.0e-3, 1.0],
            [2.0e-3, 2_000.0, 2.0e-2, 2.0],
        ])
    )
    table = read_helium_reos3_table(path)
    density, energy, inside = table.evaluate(
        np.asarray([1.0e7, 2.0e8]),
        np.asarray([1_000.0, 2_000.0]),
    )
    np.testing.assert_allclose(density, [1.0e-4, 2.0e-3])
    np.testing.assert_allclose(energy, [-2.0e10, 2.0e10])
    np.testing.assert_array_equal(inside, True)


def test_reos_reports_out_of_domain_without_extrapolating(tmp_path: Path) -> None:
    path = tmp_path / "he-reos.dat"
    np.savetxt(
        path,
        np.asarray([
            [1.0e-4, 1_000.0, 1.0e-3, 0.0],
            [1.0e-3, 1_000.0, 1.0e-2, 0.0],
            [2.0e-4, 2_000.0, 2.0e-3, 0.0],
            [2.0e-3, 2_000.0, 2.0e-2, 0.0],
        ])
    )
    table = read_helium_reos3_table(path)
    _, _, inside = table.evaluate(
        np.asarray([1.0e4, 1.0e7, 1.0e7]),
        np.asarray([1_500.0, 500.0, 3_000.0]),
    )
    np.testing.assert_array_equal(inside, [False, False, False])


def test_reos_mixed_domain_retains_dilute_eos_outside_table(tmp_path: Path) -> None:
    path = tmp_path / "he-reos.dat"
    np.savetxt(
        path,
        np.asarray([
            [1.0e-4, 5_000.0, 1.0e-3, 0.0],
            [1.0e-3, 5_000.0, 1.0e-2, 0.0],
            [2.0e-4, 10_000.0, 2.0e-3, 0.0],
            [2.0e-3, 10_000.0, 2.0e-2, 0.0],
        ])
    )
    table = read_helium_reos3_table(path)
    temperature = np.asarray([7_000.0, 20_000.0])
    pressure = np.asarray([2.0e7, 1.0e7])
    combined = hummer_mihalas_helium_lte_with_reos3(
        temperature, pressure, table
    )
    dilute = hummer_mihalas_helium_lte(
        temperature[1:], pressure[1:]
    )
    np.testing.assert_allclose(
        combined.mass_density[1:], dilute.mass_density, rtol=2.0e-11
    )


def test_fixed_density_helium_solver_recovers_pressure_solution() -> None:
    temperature = np.asarray([6_000.0, 12_000.0, 25_000.0])
    pressure = np.asarray([1.0e8, 3.0e9, 2.0e8])
    pressure_state = hummer_mihalas_helium_lte(
        temperature, pressure, correlated_microfields=True
    )
    density_state = hummer_mihalas_helium_lte_at_nuclei_density(
        temperature,
        pressure_state.helium_nuclei_density,
        correlated_microfields=True,
    )
    np.testing.assert_allclose(
        density_state.electron_density,
        pressure_state.electron_density,
        rtol=1.0e-11,
    )
    np.testing.assert_allclose(
        density_state.neutral_he_density,
        pressure_state.neutral_he_density,
        rtol=1.0e-11,
    )
