from dataclasses import replace

import numpy as np

from wd_spectra.adaptive_structure import solve_adaptive_lte_structure
from wd_spectra.atmosphere import Atmosphere


def test_jacobian_reuses_identical_residual_base_state():
    n_depth = 3
    seed = Atmosphere(
        effective_temperature=10_000.0,
        logg=8.0,
        rosseland_optical_depth=np.geomspace(1.0e-4, 1.0e2, n_depth),
        column_mass=np.geomspace(1.0e-4, 1.0e2, n_depth),
        temperature=np.full(n_depth, 5_000.0),
        gas_pressure=np.geomspace(1.0e2, 1.0e8, n_depth),
        mass_density=np.ones(n_depth),
        neutral_h_density=np.ones(n_depth),
        proton_density=np.ones(n_depth),
        electron_density=np.ones(n_depth),
        metadata={},
    )
    wavelength = np.asarray([1_000.0, 2_000.0, 4_000.0, 8_000.0])
    opacity_states: list[np.ndarray] = []

    def with_temperature(temperature):
        return replace(seed, temperature=np.asarray(temperature))

    def true_absorption(atmosphere):
        opacity_states.append(atmosphere.temperature.copy())
        return np.ones((wavelength.size, n_depth))

    solve_adaptive_lte_structure(
        seed,
        wavelength,
        with_temperature=with_temperature,
        true_absorption=true_absorption,
        scattering_opacity=lambda atmosphere: np.zeros(
            (wavelength.size, n_depth)
        ),
        rosseland_opacity=lambda atmosphere: np.ones(n_depth),
        thermodynamics=lambda atmosphere: None,
        mixing_length_alpha=None,
        max_iterations=1,
        temperature_tolerance=1.0e-4,
        flux_tolerance=1.0e-3,
        n_angle=2,
        initial_temperature_was_supplied=False,
    )

    # The nonlinear engine asks first for a residual and then for a Jacobian
    # at the same state.  Only the hotter tangent opacity should add another
    # callback invocation; the base state must not be rebuilt.
    assert len(opacity_states) > 1
    assert all(
        not np.array_equal(previous, current)
        for previous, current in zip(opacity_states, opacity_states[1:])
    )
