import numpy as np
import pytest

from wd_spectra import (
    TREMBLAY_2013_LOG_ROSSELAND_DEPTH,
    TREMBLAY_2013_MEAN_3D_MINUS_1D_TEMPERATURE_K,
    atmosphere_with_tremblay_2013_mean_3d_temperature_difference,
    hydrostatic_atmosphere_with_tremblay_2013_mean_3d_temperature_difference,
    hydrogen_continuum_atmosphere,
    tremblay_2013_mean_3d_temperature_difference,
)


def test_tremblay_2013_temperature_difference_reproduces_digitized_7012_k_curve():
    difference = tremblay_2013_mean_3d_temperature_difference(
        7_012.0,
        10.0**TREMBLAY_2013_LOG_ROSSELAND_DEPTH,
    )
    np.testing.assert_array_equal(
        difference,
        TREMBLAY_2013_MEAN_3D_MINUS_1D_TEMPERATURE_K[1],
    )


def test_tremblay_2013_structure_control_preserves_pressure_and_recomputes_eos():
    atmosphere = hydrogen_continuum_atmosphere(
        7_012.0,
        8.0,
        n_depth=30,
        correlated_microfields=True,
        include_molecules=True,
    )
    corrected = atmosphere_with_tremblay_2013_mean_3d_temperature_difference(
        atmosphere
    )

    np.testing.assert_array_equal(corrected.gas_pressure, atmosphere.gas_pressure)
    np.testing.assert_array_equal(corrected.column_mass, atmosphere.column_mass)
    assert corrected.temperature[0] < atmosphere.temperature[0]
    assert np.max(np.abs(corrected.mass_density - atmosphere.mass_density)) > 0.0
    assert corrected.hydrogen_lte_state is not atmosphere.hydrogen_lte_state
    assert corrected.hydrogen_lte_state.microfield_model == "qmhd"
    assert corrected.hydrogen_lte_state.molecular_hydrogen_density is not None
    assert not corrected.metadata[
        "mean_3d_temperature_differential_is_equilibrium_model"
    ]


def test_tremblay_2013_structure_control_can_use_montreal_neutral_radius():
    atmosphere = hydrogen_continuum_atmosphere(
        7_012.0,
        8.0,
        n_depth=18,
        correlated_microfields=True,
        include_molecules=True,
    )
    corrected = atmosphere_with_tremblay_2013_mean_3d_temperature_difference(
        atmosphere,
        neutral_radius_scale=0.5,
    )

    assert corrected.hydrogen_lte_state is not None
    assert corrected.hydrogen_lte_state.neutral_radius_scale == 0.5
    assert (
        corrected.metadata[
            "mean_3d_temperature_differential_neutral_radius_scale"
        ]
        == 0.5
    )


def test_tremblay_2013_temperature_difference_rejects_extrapolation():
    with pytest.raises(ValueError, match="5997--8032"):
        tremblay_2013_mean_3d_temperature_difference(9_000.0, [1.0e-3, 1.0])


def test_tremblay_2013_hydrostatic_control_reintegrates_pressure():
    atmosphere = hydrogen_continuum_atmosphere(
        7_012.0,
        8.0,
        n_depth=10,
        correlated_microfields=True,
        include_molecules=True,
    )
    corrected = (
        hydrostatic_atmosphere_with_tremblay_2013_mean_3d_temperature_difference(
            atmosphere
        )
    )

    assert np.all(np.diff(corrected.gas_pressure) > 0.0)
    np.testing.assert_allclose(
        corrected.column_mass,
        corrected.gas_pressure / corrected.gravity,
        rtol=2.0e-15,
    )
    assert np.max(np.abs(corrected.gas_pressure / atmosphere.gas_pressure - 1.0)) > 0.01
    assert corrected.metadata[
        "mean_3d_temperature_differential_preserves_pressure"
    ] is False
