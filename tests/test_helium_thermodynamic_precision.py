"""Thermal derivatives must resolve the small ML2 superadiabatic excess."""
import numpy as np
from wd_spectra import eos


def test_separated_thermal_derivatives_reproduce_the_same_enthalpy_stencil():
    t, p = np.meshgrid(np.geomspace(2500.0, 100000.0, 30), np.geomspace(1e2, 1e12, 12))
    h = 2e-4
    options = dict(
        maximum_level=eos.HM_MAX_BOUND_LEVEL,
        neutral_radius_scale=eos.HM_HELIUM_NEUTRAL_RADIUS_SCALE,
        correlated_microfields=True,
    )
    cold_h, cold = eos._hummer_mihalas_helium_specific_enthalpy(
        t * np.exp(-h), p, **options
    )
    hot_h, hot = eos._hummer_mihalas_helium_specific_enthalpy(
        t * np.exp(h), p, **options
    )
    cp = (hot_h - cold_h) / (t * np.exp(h) - t * np.exp(-h))
    expansion = -(np.log(hot.mass_density) - np.log(cold.mass_density)) / (2 * h)
    actual = eos.hummer_mihalas_helium_thermodynamics(t, p, **options)
    np.testing.assert_allclose(actual.specific_heat_constant_pressure, cp, rtol=2e-9)
    np.testing.assert_allclose(
        actual.density_temperature_derivative, expansion, rtol=2e-9
    )


def test_neutral_limit_is_smooth_under_tiny_temperature_updates():
    t, p = np.array([5000.0, 6000.0]), np.array([1e8, 1e12])
    values = np.array(
        [
            eos.hummer_mihalas_helium_thermodynamics(
                t * np.exp(step), p, correlated_microfields=True
            ).adiabatic_temperature_gradient
            for step in np.arange(-5, 6) * 1e-10
        ]
    )
    # The old full-enthalpy/log-density subtraction jitters by ~1e-12 here,
    # enough to disturb extremely efficient ML2 convection.
    assert np.max(np.ptp(values, axis=0)) < 1e-15
    assert np.all(values < 0.4)  # Trace-ion response was retained, not clamped.


def test_separated_enthalpy_is_an_algebraic_decomposition_not_a_new_eos():
    t, p = np.array([3000.0, 8000.0, 22000.0, 80000.0]), np.geomspace(1e4, 1e10, 4)
    options = dict(
        maximum_level=eos.HM_MAX_BOUND_LEVEL,
        neutral_radius_scale=eos.HM_HELIUM_NEUTRAL_RADIUS_SCALE,
        correlated_microfields=True,
    )
    full, _ = eos._hummer_mihalas_helium_specific_enthalpy(t, p, **options)
    rest, _ = eos._hummer_mihalas_helium_specific_enthalpy(
        t, p, separate_neutral_translation=True, **options
    )
    np.testing.assert_allclose(
        rest + 2.5 * eos.BOLTZMANN * t / eos.HELIUM_MASS, full, rtol=2e-14
    )
