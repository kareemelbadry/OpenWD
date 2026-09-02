import numpy as np

from wd_spectra.conduction import (
    conductive_opacity_from_thermal_conductivity,
    hydrogen_electron_thermal_conductivity,
)
from wd_spectra.constants import STEFAN_BOLTZMANN


def test_potekhin_hydrogen_table_is_exact_at_a_grid_node():
    conductivity = hydrogen_electron_thermal_conductivity(1.0e5, 1.0e-2)
    np.testing.assert_allclose(np.log10(conductivity), 7.508, atol=1.0e-13)


def test_potekhin_cubic_interpolation_matches_reference_fortran_program():
    conductivity = hydrogen_electron_thermal_conductivity(
        10.0**5.123, 10.0**-1.234
    )
    # Potekhin's condint.f prints 7.975 and 5.108 for this query.
    np.testing.assert_allclose(
        np.log10(conductivity), 7.9754078175465954, rtol=0.0, atol=2.0e-13
    )
    opacity = conductive_opacity_from_thermal_conductivity(
        10.0**5.123, 10.0**-1.234, conductivity
    )
    np.testing.assert_allclose(np.log10(opacity), 5.108, atol=3.0e-4)


def test_conductivity_is_not_extrapolated_outside_published_grid():
    conductivity = hydrogen_electron_thermal_conductivity(
        np.asarray([900.0, 1.0e5, 1.0e10]),
        np.asarray([1.0e-2, 1.0e-7, 1.0e-2]),
    )
    np.testing.assert_array_equal(conductivity, 0.0)


def test_conductive_opacity_has_diffusion_equivalence():
    temperature = np.asarray([1.0e5, 2.0e5])
    density = np.asarray([1.0e-2, 2.0e-2])
    conductivity = np.asarray([1.0e8, 3.0e8])
    opacity = conductive_opacity_from_thermal_conductivity(
        temperature, density, conductivity
    )
    recovered = 16.0 * STEFAN_BOLTZMANN * temperature**3 / (
        3.0 * opacity * density
    )
    np.testing.assert_allclose(recovered, conductivity, rtol=2.0e-15)
