from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from wd_spectra.adaptive_structure import (
    _adiabatic_asymptotic_convection_mask,
    _clip_roundoff_negative_scattering_source,
    _thermal_boundary_absorption_escape_bound,
    _cap_overcarrying_ml2_gradient,
    solve_adaptive_lte_structure,
)
from wd_spectra.atmosphere import Atmosphere


def test_convective_trial_correction_changes_gradients_not_flux_values():
    gradient = np.array([0., .3, .5, .9])
    adiabatic = np.full(4, .4)
    loss = np.full(4, .1)
    coefficient = np.full(4, 1e6)
    def flux(g):
        excess = np.maximum(g-adiabatic, 0.)
        return coefficient * (np.sqrt(.25*loss**2+excess)-.5*loss)**3
    actual = flux(gradient)
    transport = dict(convective_flux=actual.copy(), adiabatic_gradient=adiabatic,
                     ml2_flux_coefficient=coefficient, ml2_radiative_loss=loss)
    corrected = _cap_overcarrying_ml2_gradient(gradient, transport, 1.)
    np.testing.assert_array_equal(corrected[:2], gradient[:2])
    np.testing.assert_allclose(flux(corrected)[2:], 1., rtol=2e-12)
    np.testing.assert_array_equal(transport['convective_flux'], actual)
    np.testing.assert_array_equal(gradient, [0., .3, .5, .9])


def test_no_accepted_step_can_save_exploratory_checkpoint(monkeypatch, tmp_path):
    import wd_spectra.adaptive_structure as adaptive
    from wd_spectra.models import DBConfig, ModelResult, save_model_result, load_atmosphere_checkpoint
    from wd_spectra.spectrum import Spectrum
    n = 3
    mass = np.geomspace(1e-4, 100., n)
    seed = Atmosphere(
        effective_temperature=10000., logg=8., rosseland_optical_depth=mass,
        column_mass=mass, gas_pressure=1e8*mass, temperature=np.full(n, 5000.),
        mass_density=np.ones(n), neutral_h_density=np.zeros(n),
        proton_density=np.zeros(n), electron_density=np.ones(n), metadata={})
    wavelength = np.array([1000., 2000., 4000., 8000.])
    original_solver = adaptive.solve_trust_region_newton
    def reject_every_trial(initial, evaluate, **options):
        options.update(acceptance_test=lambda old, new: False,
                       finite_difference_fallback_step=None, maximum_iterations=1)
        return original_solver(initial, evaluate, **options)
    monkeypatch.setattr(adaptive, "solve_trust_region_newton", reject_every_trial)
    atmosphere = solve_adaptive_lte_structure(
        seed, wavelength, with_temperature=lambda t: replace(seed, temperature=t),
        true_absorption=lambda a: np.ones((4, n)),
        scattering_opacity=lambda a: np.zeros((4, n)),
        rosseland_opacity=lambda a: np.ones(n), thermodynamics=lambda a: None,
        mixing_length_alpha=None, max_iterations=1, temperature_tolerance=1e-4,
        flux_tolerance=1e-3, n_angle=2, initial_temperature_was_supplied=False,
        use_initial_bolometric_rescaling=False, maximum_formal_flux_continuations=0)
    assert not atmosphere.metadata["radiative_equilibrium_converged"]
    assert atmosphere.metadata["radiative_equilibrium_maximum_log_temperature_correction"] is None
    spectrum = Spectrum(wavelength, np.ones(4), {})
    result = ModelResult("DB", atmosphere, spectrum, DBConfig(), {})
    saved = save_model_result(result, tmp_path / "exploratory")
    restored = load_atmosphere_checkpoint(saved / "atmosphere.npz", 10000., 8., "helium")
    assert not restored.metadata["radiative_equilibrium_converged"]
    assert restored.metadata["radiative_equilibrium_maximum_log_temperature_correction"] is None


@pytest.mark.parametrize("previous_step", [.01, 1e-5])
@pytest.mark.parametrize("polish_requested,flux_passes", [(False, True), (True, True), (True, False)])
def test_formal_phase_does_not_erase_an_oversized_conditioning_step(
    monkeypatch, previous_step, polish_requested, flux_passes
):
    import wd_spectra.adaptive_structure as adaptive
    from wd_spectra.constants import STEFAN_BOLTZMANN
    n = 3
    mass = np.geomspace(.01, 100., n)
    seed = Atmosphere(
        effective_temperature=10000., logg=8., rosseland_optical_depth=mass,
        column_mass=mass, gas_pressure=1e8*mass, temperature=np.full(n, 5000.),
        mass_density=np.ones(n), neutral_h_density=np.zeros(n),
        proton_density=np.zeros(n), electron_density=np.ones(n), metadata={})
    original = adaptive.solve_trust_region_newton
    stable = adaptive.cancellation_safe_field
    stable_calls = []
    def checked_field(*args, **kwargs):
        stable_calls.append(True)
        return stable(*args, **kwargs)
    monkeypatch.setattr(adaptive, "cancellation_safe_field", checked_field)
    calls = []
    class Checked(Exception):
        pass
    def segment(initial, evaluate, **options):
        calls.append(options)
        if len(calls) == 2:
            assert options["allow_initial_convergence"] == (previous_step < 3e-4)
            expected_polish = polish_requested and flux_passes and previous_step >= 3e-4
            assert (options.get("linear_regularization") == 0.) == expected_polish
            # A phase switch must invalidate the cached warm field.
            evaluate(initial, False)
            assert bool(stable_calls) == expected_polish
            raise Checked()
        result = original(initial, evaluate, **{**options, "maximum_iterations": 1})
        assert result.history
        # Synthetic previous phase diagnostics isolate the transition policy.
        payload = dict(result.evaluation.payload,
            total_flux_interface=np.full(n, STEFAN_BOLTZMANN*10000.**4*(1 if flux_passes else 2)))
        return replace(result,
            evaluation=replace(result.evaluation, payload=payload),
            history=(replace(result.history[-1], maximum_step=previous_step),))
    monkeypatch.setattr(adaptive, "solve_trust_region_newton", segment)
    with pytest.raises(Checked):
        solve_adaptive_lte_structure(
            seed, np.array([1000., 2000., 4000., 8000.]),
            with_temperature=lambda t: replace(seed, temperature=t),
            true_absorption=lambda a: np.ones((4, n)),
            scattering_opacity=lambda a: np.zeros((4, n)),
            rosseland_opacity=lambda a: np.ones(n),
            thermodynamics=lambda a: SimpleNamespace(specific_heat_constant_pressure=np.full(n, 1e8),
                density_temperature_derivative=np.ones(n), adiabatic_temperature_gradient=np.full(n, .4)),
            mixing_length_alpha=1.25, max_iterations=1, temperature_tolerance=3e-4,
            flux_tolerance=3e-3, n_angle=2, initial_temperature_was_supplied=False,
            project_initial_convective_gradient=False, use_initial_bolometric_rescaling=False,
            use_precision_polish=polish_requested,
            maximum_formal_flux_rosseland_depth=None, maximum_formal_flux_continuations=0)
    assert len(calls) == 2


def test_thermal_boundary_escape_distinguishes_transparent_and_deep_domains():
    wavelength = np.array([1000., 4000., 8000.])
    planck = np.array([1., 2., 3.])
    mass = np.array([1., 2., 3.])
    # Use a quadrature-matched target so the expected attenuation is exact.
    from wd_spectra._compat import trapezoid
    target = np.pi * trapezoid(planck, wavelength)
    for bottom_depth in (0., .3, 1., 10., 100.):
        opacity = np.full((3, 3), bottom_depth / mass[-1])
        bound = _thermal_boundary_absorption_escape_bound(
            wavelength, mass, opacity, planck, target)
        np.testing.assert_allclose(bound, np.exp(-bottom_depth), rtol=2e-14)
    # The diagnostic uses B at the bottom, not an assumed Teff normalization.
    assert _thermal_boundary_absorption_escape_bound(
        wavelength, mass, np.zeros((3, 3)), 2*planck, target) == pytest.approx(2.)


@pytest.mark.parametrize("scattering_fraction", [0.0, 0.98])
@pytest.mark.parametrize("formulation", ["flux", "auxiliary", "auxiliary-local", "direct-local", "current-local"])
def test_full_flux_jacobian_includes_local_material_response(
    monkeypatch, scattering_fraction, formulation
):
    """Check the residual actually solved, not an isolated ML2 derivative."""
    import wd_spectra.adaptive_structure as adaptive

    n = 6
    pressure = np.geomspace(1e5, 1e7, n)
    temperature = 5000.0 * (pressure / pressure[0])**0.3
    seed = Atmosphere(
        effective_temperature=8000.0, logg=8.0,
        rosseland_optical_depth=np.geomspace(1e-3, 100.0, n),
        column_mass=pressure / 1e8, temperature=temperature,
        gas_pressure=pressure, mass_density=pressure / (1e8 * temperature),
        neutral_h_density=np.ones(n), proton_density=np.ones(n),
        electron_density=np.ones(n), metadata={},
    )
    wavelength = np.geomspace(500.0, 1e5, 60)

    def with_temperature(values):
        return replace(seed, temperature=values,
                       mass_density=pressure / (1e8 * values))

    def extinction(atmosphere):
        return (5.0 * (atmosphere.temperature[None, :] / 5000.0)**1.5
                * np.where(wavelength[:, None] < 3000.0, 2.0, 1.0))

    def thermodynamics(atmosphere):
        ratio = atmosphere.temperature / 5000.0
        return SimpleNamespace(
            specific_heat_constant_pressure=2e8 * ratio**0.7,
            density_temperature_derivative=1.0 + 0.1*ratio,
            adiabatic_temperature_gradient=0.18 + 0.01*np.log(ratio),
        )

    def fixed_grid_mean(atmosphere):
        return adaptive.rosseland_mean_from_opacity_grid(
            wavelength, extinction(atmosphere), atmosphere.temperature
        )

    class Checked(Exception):
        pass

    def check(initial_state, evaluate, **kwargs):
        base = evaluate(initial_state, True)
        assert np.max(base.payload["convective_flux_interface"]) > 1e9
        for column in range(initial_state.size):
            delta = np.zeros_like(initial_state)
            delta[column] = 1e-6
            finite_difference = (
                evaluate(initial_state + delta, False).residual
                - evaluate(initial_state - delta, False).residual
            ) / (2e-6)
            np.testing.assert_allclose(
                base.jacobian[:, column], finite_difference,
                rtol=2e-3, atol=2e-5,
            )
        raise Checked()

    def checked_solve(initial, evaluate, **options):
        if formulation == "flux":
            return check(initial, evaluate, **options)
        if formulation in ("direct-local", "current-local"):
            from wd_spectra.constants import STEFAN_BOLTZMANN
            from wd_spectra.nonlinear import NonlinearEvaluation
            target = STEFAN_BOLTZMANN*seed.effective_temperature**4
            scale = evaluate(initial, False).payload["cell_energy_scale"]/target
            def local_evaluate(state, need_jacobian):
                ev = evaluate(state, need_jacobian)
                normalization = ev.payload["cell_energy_scale"]/target if formulation == "current-local" else scale
                values = ev.residual.copy()
                ratio = ev.payload["cell_energy_balance_defect_in_stellar_flux"]/normalization
                values[:-1] -= ratio
                jacobian = None
                if need_jacobian:
                    jacobian = ev.jacobian.copy()
                    jacobian[:-1] -= (ev.payload["cell_energy_log_temperature_jacobian"]
                                     @ ev.payload["log_temperature_from_state"]
                                     / target / normalization[:, None])
                    if formulation == "current-local":
                        jacobian[:-1] += (ratio[:, None]
                            * (ev.payload["cell_energy_scale_log_temperature_jacobian"]
                               @ ev.payload["log_temperature_from_state"])
                            / target / normalization[:, None])
                return NonlinearEvaluation(values, jacobian, ev.payload)
            return check(initial, local_evaluate, **options)
        from wd_spectra._ml2_auxiliary import solve_auxiliary_ml2_experiment
        from wd_spectra.constants import STEFAN_BOLTZMANN
        return solve_auxiliary_ml2_experiment(
            initial, evaluate, check, STEFAN_BOLTZMANN*seed.effective_temperature**4,
            local_energy_scaling=formulation == "auxiliary-local", **options)
    monkeypatch.setattr(adaptive, "solve_trust_region_newton", checked_solve)
    with pytest.raises(Checked):
        solve_adaptive_lte_structure(
            seed, wavelength, with_temperature=with_temperature,
            true_absorption=lambda a: (1-scattering_fraction)*extinction(a),
            scattering_opacity=lambda a: scattering_fraction*extinction(a),
            rosseland_opacity=fixed_grid_mean,
            thermodynamics=thermodynamics, mixing_length_alpha=1.25,
            max_iterations=1, temperature_tolerance=2e-4, flux_tolerance=3e-3,
            n_angle=3, initial_temperature_was_supplied=True,
            resume_supplied_structure_in_formal_flux_phase=True,
            project_initial_convective_gradient=False,
            use_initial_bolometric_rescaling=False,
            compute_local_energy_response=formulation in ("direct-local", "current-local"),
        )


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


def test_collapsed_trial_optical_depth_is_recoverable(monkeypatch):
    import wd_spectra.adaptive_structure as adaptive
    from wd_spectra.nonlinear import RecoverableEvaluationError

    n = 3
    seed = Atmosphere(
        effective_temperature=8000., logg=8.,
        rosseland_optical_depth=np.array([.1, 1., 10.]),
        column_mass=np.array([1., 2., 3.]),
        temperature=np.full(n, 8000.), gas_pressure=np.array([1e8, 2e8, 3e8]),
        mass_density=np.ones(n), neutral_h_density=np.ones(n),
        proton_density=np.ones(n), electron_density=np.ones(n), metadata={},
    )
    # Emulate cumulative floating-point collapse, not malformed opacity shapes.
    monkeypatch.setattr(adaptive, "optical_depth_from_mass_opacity",
                        lambda mass, opacity: np.tile([1., 2., 2.], (2, 1)))
    with pytest.raises(RecoverableEvaluationError, match="unresolved optical-depth"):
        solve_adaptive_lte_structure(
            seed, np.array([4000., 5000.]),
            with_temperature=lambda t: replace(seed, temperature=t),
            true_absorption=lambda a: np.ones((2, n)),
            scattering_opacity=lambda a: np.zeros((2, n)),
            rosseland_opacity=lambda a: np.ones(n), thermodynamics=lambda a: None,
            mixing_length_alpha=None, max_iterations=1,
            temperature_tolerance=3e-4, flux_tolerance=3e-3, n_angle=3,
            initial_temperature_was_supplied=False,
            use_initial_bolometric_rescaling=False,
        )


def test_adiabatic_asymptotic_mask_requires_subresolution_increment():
    increment, mask = _adiabatic_asymptotic_convection_mask(
        np.asarray([0.0, 0.5, 0.5, 1.0]),
        np.asarray([0.0, 0.40001, 0.40010, 0.40001]),
        np.asarray([0.0, 0.4, 0.4, 0.4]),
        np.asarray([1.0, 1.0, 1.0]),
        target_flux=1.0,
        flux_tolerance=2.0e-3,
        temperature_tolerance=2.0e-4,
    )

    np.testing.assert_allclose(increment, [0.0, 1.0e-5, 1.0e-4, 1.0e-5])
    np.testing.assert_array_equal(mask, [False, True, False, True])


def test_scattering_source_clips_only_wavelength_local_numerical_negatives():
    source, clipped, maximum_relative = (
        _clip_roundoff_negative_scattering_source(
            np.asarray([[1.0, -3.0e-9], [1.0e-100, 2.0e-100]])
        )
    )

    np.testing.assert_array_equal(
        source, [[1.0, 0.0], [1.0e-100, 2.0e-100]]
    )
    assert clipped == 1
    assert maximum_relative == 3.0e-9


def test_scattering_source_rejects_material_negative_values():
    with np.testing.assert_raises_regex(ValueError, "materially negative"):
        _clip_roundoff_negative_scattering_source(
            np.asarray([[1.0, -2.0e-8]])
        )
