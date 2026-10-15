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


def _smooth_reos3():
    from wd_spectra.models.common import ModelData
    return read_helium_reos3_table(ModelData.default().helium_reos3).with_interpolation(
        "ideal-excess"
    )


def test_ideal_excess_reos_reproduces_every_tabulated_point():
    table = _smooth_reos3()
    linear = table.with_interpolation("log-linear")
    for index in (6, 7, 8, 9, 10):
        pressure = table.pressure_by_temperature[index][5:95:7]
        temperature = np.full(pressure.shape, table.temperature_grid[index])
        smooth_rho, smooth_u, inside = table.evaluate(pressure, temperature)
        rho, u, _ = linear.evaluate(pressure, temperature)
        assert np.all(inside)
        np.testing.assert_allclose(smooth_rho, rho, rtol=1e-13)
        np.testing.assert_allclose(smooth_u, u, rtol=1e-12)


def test_ideal_excess_reos_has_ideal_limit_and_no_knot_jumps():
    from wd_spectra.constants import BOLTZMANN, HELIUM_MASS

    table = _smooth_reos3()
    ideal_cp = 2.5 * BOLTZMANN / HELIUM_MASS
    temperature = np.array([4000.0, 5999.0, 6001.0, 8000.0, 9900.0])
    _, cp, expansion, gradient, inside = table.thermodynamics(
        np.full(temperature.shape, 1.0e8), temperature
    )
    assert np.all(inside)
    # Dilute neutral He: exact monatomic ideal gas between the isotherms
    # (the log-linear interpolant is wrong by up to ~40% here).
    np.testing.assert_allclose(cp / ideal_cp, 1.0, atol=3e-3)
    np.testing.assert_allclose(gradient, 0.4, atol=2e-3)
    for pressure in (1.0e8, 1.0e10, 1.0e11):
        for knot in (6000.0, 10000.0):
            values = table.thermodynamics(
                np.full(2, pressure), np.array([knot * (1 - 1e-7), knot * (1 + 1e-7)])
            )
            for field in values[1:4]:
                np.testing.assert_allclose(field[0], field[1], rtol=1e-5)


def test_ideal_excess_reos_thermodynamics_are_derivatives_of_its_density_and_energy():
    table = _smooth_reos3()
    pressure = np.array([1.0e6, 1.0e9, 3.0e10, 1.0e11, 5.0e11])
    temperature = np.array([4500.0, 6100.0, 7300.0, 11000.0, 16000.0])
    density, cp, expansion, gradient, inside = table.thermodynamics(pressure, temperature)
    assert np.all(inside) and np.all(cp > 0) and np.all(expansion > 0)
    h = 1e-5
    hot_rho, hot_u, _ = table.evaluate(pressure, temperature * np.exp(h))
    cold_rho, cold_u, _ = table.evaluate(pressure, temperature * np.exp(-h))
    numeric_cp = ((hot_u + pressure / hot_rho) - (cold_u + pressure / cold_rho)) / (
        temperature * (np.exp(h) - np.exp(-h))
    )
    np.testing.assert_allclose(cp, numeric_cp, rtol=1e-6)
    np.testing.assert_allclose(
        expansion, -(np.log(hot_rho) - np.log(cold_rho)) / (2 * h), rtol=1e-6
    )
    np.testing.assert_allclose(
        gradient, pressure * expansion / (density * temperature * cp), rtol=1e-12
    )
