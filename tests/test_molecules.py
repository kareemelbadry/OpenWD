from __future__ import annotations

import numpy as np

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
    assert dissociation[2] < 1.0e19

    hminus = negative_hydrogen_ionization_constant(temperature)
    assert np.all(np.diff(hminus) > 0.0)
