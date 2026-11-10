from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from wd_spectra.constants import BOLTZMANN
from wd_spectra.molecules import (
    molecular_hydrogen_dissociation_constant,
    molecular_hydrogen_internal_partition_function,
    molecular_hydrogen_rovibrational_energy,
    negative_hydrogen_ionization_constant,
    read_borysow_h2_h2_cia_table,
    trihydrogen_ion_dissociation_constant,
    trihydrogen_ion_internal_partition_function,
)


def test_barklem_collet_h2_partition_and_equilibrium_nodes():
    temperature = np.array([4_000.0, 5_000.0, 6_000.0, 7_000.0])
    np.testing.assert_allclose(
        molecular_hydrogen_internal_partition_function(temperature),
        [34.6011, 50.7619, 71.1512, 95.8762],
        rtol=2.0e-7,
    )

    dissociation = molecular_hydrogen_dissociation_constant(temperature)
    log_pressure_equilibrium = np.log10(
        dissociation * BOLTZMANN * temperature / 10.0
    )
    np.testing.assert_allclose(
        log_pressure_equilibrium,
        [5.39364, 6.59790, 7.40150, 7.97669],
        atol=1.2e-3,
    )
    assert np.all(molecular_hydrogen_rovibrational_energy(temperature) > 0.0)


def test_borysow_cia_reader_and_logarithmic_interpolation(tmp_path):
    path = tmp_path / "h2-h2.dat"
    path.write_text(
        "@SPECIES\n"
        "H2 H2\n"
        "@TEMPERATURES\n"
        "100 200\n"
        "@DATA\n"
        "100 1e-6 1e-4\n"
        "200 1e-4 1e-2\n",
        encoding="utf-8",
    )
    table = read_borysow_h2_h2_cia_table(path)

    coefficient = table.coefficient_for_wavelength_temperature(
        np.array([1.0e6, 1.0e3]), np.array([100.0, 150.0, 200.0])
    )

    np.testing.assert_allclose(coefficient[0], [1.0e-6, 1.0e-5, 1.0e-4])
    np.testing.assert_array_equal(coefficient[1], 0.0)


def test_neale_tennyson_h3plus_partition_and_equilibrium_are_physical():
    temperature = np.array([1_000.0, 3_000.0, 5_000.0, 8_000.0])
    partition = trihydrogen_ion_internal_partition_function(temperature)
    # Equation (6) of Neale & Tennyson (1995); the published polynomial is a
    # fit to their complete bound-state sum (their Table 3).
    np.testing.assert_allclose(
        partition,
        [264.4989837, 3824.167439, 24388.96167, 153482.1731],
        rtol=2.0e-9,
    )
    dissociation = trihydrogen_ion_dissociation_constant(temperature)
    assert np.all(np.diff(dissociation) > 0.0)
    assert dissociation[2] < 1.0e20

    hminus = negative_hydrogen_ionization_constant(temperature)
    assert np.all(np.diff(hminus) > 0.0)


def test_h3plus_equilibrium_divides_out_proton_spin_like_h2():
    from wd_spectra.molecules import (
        H3_PLUS_DISSOCIATION_ENERGY,
        H3_PLUS_NUCLEAR_SPIN_STATES,
        molecular_hydrogen_internal_partition_function,
    )
    from wd_spectra.constants import BOLTZMANN, HYDROGEN_MASS, PI, PLANCK

    # Barklem--Collet H2 divides the four proton-spin states out ...
    np.testing.assert_allclose(
        molecular_hydrogen_internal_partition_function(np.array([5.0])), 0.25
    )
    # ... whereas Neale & Tennyson count all eight H3+ spin states: a
    # rigid-rotor + harmonic estimate with full spin weights (8/6 of the
    # classical rotor, energy zero at J=K=1) reproduces their 1000 K value.
    temperature = 1000.0
    theta_b, theta_c = 43.6 * 1.4388, 20.6 * 1.4388
    rotor = np.sqrt(PI * temperature**3 / (theta_b**2 * theta_c))
    vibration = (1.0 / -np.expm1(-3178.0 * 1.4388 / temperature)) * (
        1.0 / -np.expm1(-2521.0 * 1.4388 / temperature)
    ) ** 2
    full_spin = 8.0 / 6.0 * rotor * vibration * np.exp(64.12 * 1.4388 / temperature)
    nt = trihydrogen_ion_internal_partition_function(np.array([temperature]))[0]
    assert abs(nt / full_spin - 1.0) < 0.1
    assert H3_PLUS_NUCLEAR_SPIN_STATES == 8.0

    # The equilibrium constant uses Q_NT / 8 and the 0 K dissociation energy
    # from the lowest allowed level (Mizus et al. 2019 components).
    np.testing.assert_allclose(
        H3_PLUS_DISSOCIATION_ENERGY / 1.602176634e-12, 4.3329, atol=1.0e-4
    )
    t = np.array([4000.0, 5000.0])
    translation = (
        2.0 * PI * (2.0 / 3.0) * HYDROGEN_MASS * BOLTZMANN * t / PLANCK**2
    ) ** 1.5
    expected = (
        translation
        * molecular_hydrogen_internal_partition_function(t)
        / (trihydrogen_ion_internal_partition_function(t) / 8.0)
        * np.exp(-H3_PLUS_DISSOCIATION_ENERGY / (BOLTZMANN * t))
    )
    np.testing.assert_allclose(
        trihydrogen_ion_dissociation_constant(t), expected, rtol=1.0e-12
    )


def test_bell_h2_minus_free_free_reproduces_table_and_scales_smoothly():
    from wd_spectra.molecules import (
        h2_minus_free_free_coefficient_per_electron_pressure,
    )

    # Bell (1980) Table 1: theta = 1.0 (5040 K), 9113 A: 4.22e-27 cm^4/dyn.
    value = h2_minus_free_free_coefficient_per_electron_pressure(
        np.array([9113.0, 3000.0, 2.0e5]), np.array([5040.0, 1000.0, 20000.0])
    )
    np.testing.assert_allclose(value[0, 0], 4.22e-27, rtol=1.0e-10)
    assert np.all(value > 0.0) and np.all(np.isfinite(value))
    # Larger theta (cooler) gives more absorption; longer wavelength too.
    assert value[0, 1] > value[0, 0] > value[0, 2]
    assert value[2, 0] > value[0, 0] > value[1, 0]


def test_hitran_h2_h_cia_is_checksummed_and_held_above_2500_k():
    from wd_spectra.molecules import h2_h_cia_table

    table = h2_h_cia_table()
    np.testing.assert_array_equal(
        table.temperature_K, [1000.0, 1500.0, 2000.0, 2500.0]
    )
    coefficient = table.coefficient_for_wavelength_temperature(
        np.array([1.0e8 / 4160.0, 1.0e8 / 9990.0, 1.0e8 / 10010.0, 1.0e8 / 50.0]),
        np.array([1000.0, 2500.0, 5000.0]),
    )
    # HITRAN H2-H_2011.cia, 1000 K block: 1.628e-45 at 4160 cm^-1.
    np.testing.assert_allclose(coefficient[0, 0], 1.628e-45, rtol=1.0e-10)
    np.testing.assert_array_equal(coefficient[:, 2], coefficient[:, 1])
    # Smooth continuation past 10000 cm^-1; zero below 100 cm^-1.
    assert 0.9 < coefficient[2, 1] / coefficient[1, 1] < 1.1
    assert coefficient[2, 1] < coefficient[1, 1]
    np.testing.assert_array_equal(coefficient[3], 0.0)


def test_borysow_cia_declining_tail_is_continuous_and_uses_terminal_slope(tmp_path):
    path = tmp_path / "declining-cia.dat"
    path.write_text("@TEMPERATURES\n100 200\n@DATA\n"
                    "100 1e-4 1e-2\n200 1e-6 1e-4\n", encoding="utf-8")
    table = read_borysow_h2_h2_cia_table(path)
    assert table.extrapolate_high_wavenumber_tail
    frequencies = np.array([199.99, 200., 200.01, 300., 1e8])
    values = table.coefficient_for_wavelength_temperature(
        1e8 / frequencies, np.array([100., 150., 200.]))
    np.testing.assert_allclose(values[1], [1e-6, 1e-5, 1e-4], rtol=2e-14)
    np.testing.assert_allclose(values[3], [1e-8, 1e-7, 1e-6], rtol=2e-14)
    slopes = np.diff(np.log(values[:3]), axis=0) / .01
    np.testing.assert_allclose(slopes[0], slopes[1], rtol=1e-10)
    assert np.all(np.diff(values, axis=0) <= 0.)
    assert np.all(np.isfinite(values))
    # Enabling the tail cannot change any in-table coefficient.
    bounded = replace(table, extrapolate_high_wavenumber_tail=False)
    np.testing.assert_array_equal(
        values[:2], bounded.coefficient_for_wavelength_temperature(
            1e8 / frequencies[:2], np.array([100., 150., 200.])))
    with pytest.raises(ValueError, match="declining terminal"):
        replace(table, absorption_coefficient=table.absorption_coefficient[::-1])
