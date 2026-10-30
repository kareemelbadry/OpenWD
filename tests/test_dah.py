"""Fast checks of the magnetic hydrogen (DAH) physics and public surface."""

from dataclasses import replace

import numpy as np
import pytest
from types import SimpleNamespace
import weakref

from wd_spectra import DAHConfig, compute_dah, select_physics
from wd_spectra.atmosphere import gray_hydrogen_atmosphere
from wd_spectra.constants import BOLTZMANN, HYDROGEN_IONIZATION_ENERGY, LIGHT_SPEED, PLANCK
from wd_spectra.eos import hummer_mihalas_hydrogen_lte
from wd_spectra.magnetic import (
    PolarizedOpacity,
    dipole_surface_cells,
    frequency_hilbert_dispersion,
    linear_zeeman_triplet,
    magnetic_continuum,
    weak_field_balmer_manifolds,
)
from wd_spectra.magnetic_atomic import (
    read_h2db_energy_database,
    read_h2db_transition_database,
)
from wd_spectra.magnetic_continuum import (
    _explicit_rwa_atomic_kernels,
    explicit_rwa_hydrogen_bound_free_mass_absorption_coefficient,
)
from wd_spectra.magnetic_eos import magnetic_hummer_mihalas_hydrogen_lte
from wd_spectra.models.common import ModelData
from wd_spectra.opacity import (
    hydrogen_continuum_mass_absorption_coefficient,
    hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient,
)
from wd_spectra.validation.magnetic_da import (
    DAH_VALIDATION_TARGETS,
    load_dah_observation,
    score_dah_spectrum,
)

H2DB = ModelData.default().h2db_balmer_subset


@pytest.fixture(scope="module")
def transitions():
    return read_h2db_transition_database(H2DB)


@pytest.fixture(scope="module")
def energies():
    return read_h2db_energy_database(H2DB)


@pytest.fixture(scope="module")
def atmosphere():
    return gray_hydrogen_atmosphere(15_000.0, 8.0, n_depth=24, correlated_microfields=True)


def test_h2db_subset_restores_the_mirrored_pi_components(transitions):
    # 15 Halpha branches: the archive stores 14; the loosely bound 2p(m=+1)
    # pi branch is its exact mirror.
    assert len(transitions.transitions_by_upper_level[3]) == 15
    temperature = np.array([12_000.0])
    for level, rest, tolerance in ((3, 6564.636, 2e-5), (4, 4862.694, 5e-5), (6, 4102.898, 2e-4)):
        components = transitions.balmer_components(level, rest, 0.05, temperature)
        strength = components.normalized_strength[:, 0]
        assert np.sum(strength[components.delta_m == 0]) == pytest.approx(1 / 3, abs=2e-3)
        assert components.line_strength_scale[0] == pytest.approx(1.0, abs=2e-3)
        # Delta m = +1 is the blue component, displaced by the normal triplet
        # (checked at 0.5 MG: at 0.05 MG the shift is at H2db's 1e-5 Ry
        # interpolation accuracy; the residual is the quadratic shift).
        components = transitions.balmer_components(level, rest, 0.5, temperature)
        strength = components.normalized_strength[:, 0]
        blue = components.delta_m > 0
        centroid = np.sum(components.wavelength_angstrom[blue] * strength[blue]) / np.sum(strength[blue])
        assert centroid == pytest.approx(
            linear_zeeman_triplet(rest, 0.5).sigma_blue_angstrom, rel=tolerance
        )


def test_h2db_components_exist_through_the_high_field_track_endpoints(transitions):
    temperature = np.array([20_000.0])
    for level in range(3, 13):
        components = transitions.balmer_components(level, 4000.0, 2_000.0, temperature)
        assert np.all(np.isfinite(components.wavelength_angstrom))


def test_magnetic_eos_has_the_exact_zero_field_limit(energies):
    temperature = np.array([6_000.0, 15_000.0, 25_000.0])
    pressure = np.full(3, 1.0e5)
    reference = hummer_mihalas_hydrogen_lte(temperature, pressure, include_molecules=False)
    weak = magnetic_hummer_mihalas_hydrogen_lte(temperature, pressure, 1.0e-3, energies)
    np.testing.assert_allclose(weak.ionization_fraction, reference.ionization_fraction, rtol=3e-4)
    np.testing.assert_allclose(
        weak.level_population_density[:, :4], reference.level_population_density[:, :4], rtol=3e-4
    )
    assert energies.ground_state_binding_energy_erg(1.0e-4) == pytest.approx(
        HYDROGEN_IONIZATION_ENERGY, rel=1e-6
    )


def test_line_dispersion_is_positive_on_the_low_frequency_side():
    wavelength = np.linspace(4000.0, 6000.0, 4001)
    line = 1.0 / (1.0 + ((wavelength - 5000.0) / 2.0) ** 2)
    psi = frequency_hilbert_dispersion(wavelength, line[:, np.newaxis])[:, 0]
    assert psi[np.searchsorted(wavelength, 5003.0)] > 0.0 > psi[np.searchsorted(wavelength, 4997.0)]


def test_stokes_coefficients_reduce_to_the_isotropic_mean():
    shape = (5, 3)
    rng = np.random.default_rng(1)
    manifolds = PolarizedOpacity(*(rng.random(shape) for _ in range(3)))
    nodes, weights = np.polynomial.legendre.leggauss(8)
    average = sum(
        0.5 * w * manifolds.stokes(0.5 * (x + 1.0))[0] for x, w in zip(nodes, weights)
    )
    np.testing.assert_allclose(average, manifolds.isotropic(), rtol=1e-12)
    eta_i, eta_q, eta_v = manifolds.stokes(0.3)
    assert np.all(np.sqrt(eta_q**2 + eta_v**2) <= eta_i + 1e-15)


def test_rwa_continuum_tends_to_the_zero_field_continuum(atmosphere, energies):
    wavelength = np.geomspace(1_000.0, 12_000.0, 800)
    reference = hydrogen_continuum_mass_absorption_coefficient(
        atmosphere, wavelength, include_electron_scattering=False,
        include_rayleigh_scattering=False, include_molecular_absorption=False,
    ) + hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient(atmosphere, wavelength)
    unpolarized, rwa = magnetic_continuum(
        atmosphere, wavelength, 0.01, energies, include_rwa_photoionization=True
    )
    total = unpolarized + rwa.isotropic()
    relative = np.abs(total / reference - 1.0)
    assert np.mean(relative) < 1e-3


def test_weak_field_triplet_conserves_the_line_opacity(atmosphere):
    wavelength = np.arange(4700.0, 5030.0, 0.05)
    manifolds = weak_field_balmer_manifolds(atmosphere, wavelength, 0.5)
    zero = weak_field_balmer_manifolds(atmosphere, wavelength, 0.0)
    np.testing.assert_allclose(
        np.sum(manifolds.isotropic()[:, 12]), np.sum(zero.isotropic()[:, 12]), rtol=5e-3
    )


def test_dipole_cells_are_normalized_and_polar_field_is_the_pole():
    cells = dipole_surface_cells(100.0, inclination_deg=0.0, n_field_bins=None)
    assert np.sum(cells.projected_weight) == pytest.approx(1.0)
    assert np.max(cells.field_strength_megagauss) == pytest.approx(100.0, rel=0.02)
    assert np.min(cells.field_strength_megagauss) >= 50.0 - 1e-9
    compressed = dipole_surface_cells(100.0, inclination_deg=60.0, offset_vector_radius=(0.0, 0.1, 0.2))
    assert compressed.projected_weight.size == 21
    assert np.sum(compressed.projected_weight) == pytest.approx(1.0)


def test_dah_configuration_and_selection():
    assert select_physics(DAHConfig()).workflow == "dah"
    with pytest.raises(ValueError, match="dipole_offset_radius"):
        compute_dah(DAHConfig(dipole_offset_radius=(0.0, 0.0, 0.1)))
    with pytest.raises(ValueError, match="field_angle_deg"):
        compute_dah(DAHConfig(field_geometry="dipole", field_angle_deg=30.0))
    with pytest.raises(ValueError, match="3400"):
        compute_dah(DAHConfig(), np.arange(3000.0, 5000.0))


def test_bundled_validation_targets_load_and_score_a_flat_model():
    for target in DAH_VALIDATION_TARGETS:
        observation = load_dah_observation(target.key)
        assert observation.wavelength_angstrom.size > 500
        assert np.all(observation.flux_nu > 0.0)
    wavelength = np.arange(3600.0, 7000.0, 1.0)
    score = score_dah_spectrum("j2149-0728", wavelength, wavelength**-2.0)
    assert np.isfinite(score["broad_rms"]) and "hardy" in score
    assert score["noise_source"] == "survey inverse variance"
    # A featureless model cannot fit the magnetic components; the published
    # model does, so its noise-weighted score is much closer to unity.
    assert score["chi2_cubic"] <= score["chi2_scale"]
    assert 1.0 < score["hardy"]["chi2_cubic"] < 3.0 < score["chi2_cubic"]
    assert load_dah_observation("j1033+2309").flux_nu_sigma is None
    estimated = score_dah_spectrum("j1033+2309", wavelength, wavelength**-2.0)
    assert estimated["noise_source"].startswith("estimated")


def test_rwa_cache_releases_previous_fields_and_grids(energies):
    cache = _explicit_rwa_atomic_kernels
    cache.cache_clear()
    root = str(energies.root.resolve())
    previous = []
    # Successive surface cells and a second model must not accumulate kernels.
    for field in (10., 30., 100., 120.):
        for n_wave in (16, 24):
            current = []
            grid = tuple(np.linspace(4000., 8000., n_wave))
            for q in (-1, 0, 1):
                kernels = cache(root, grid, field, q, 2)
                assert cache(root, grid, field, q, 2) is kernels
                current.extend(weakref.ref(shell[2]) for shell in kernels)
            del kernels
            assert all(reference() is None for reference in previous)
            assert cache.cache_info().currsize == 3
            previous = current
    cache.cache_clear()
    assert all(reference() is None for reference in previous)


def test_stationary_rwa_uses_boltzmann_substate_populations(atmosphere, energies):
    from wd_spectra.opacity import _atmosphere_level_distribution

    wave = np.linspace(3400., 8000., 40)
    field = 100.0
    kernels = _explicit_rwa_atomic_kernels(
        str(energies.root.resolve()), tuple(wave), field, 1, 3, False
    )
    levels = _atmosphere_level_distribution(atmosphere, maximum_level=40)
    expected = np.zeros((wave.size, atmosphere.n_depth))
    for n, (energy, mass, kernel, _, _) in enumerate(kernels):
        np.testing.assert_array_equal(mass, np.ones_like(mass))
        weight = np.exp(
            -(energy - energy.min())[:, None] * HYDROGEN_IONIZATION_ENERGY
            / (BOLTZMANN * atmosphere.temperature[None, :])
        )
        population = levels.population_density[:, n] * weight / weight.sum(axis=0)
        expected += kernel.T @ population
    expected *= -np.expm1(
        -PLANCK * LIGHT_SPEED / (wave[:, None] * 1e-8 * BOLTZMANN * atmosphere.temperature)
    ) / atmosphere.mass_density
    actual = explicit_rwa_hydrogen_bound_free_mass_absorption_coefficient(
        atmosphere, wave, field, 1, energies, maximum_level=3,
        include_dissolved_levels=False, include_centered_motion=False,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-13)
    moving = explicit_rwa_hydrogen_bound_free_mass_absorption_coefficient(
        atmosphere, wave, field, 1, energies, maximum_level=3,
        include_dissolved_levels=False, include_centered_motion=True,
    )
    absorbing = actual > 0
    assert np.max(np.abs(moving[absorbing] / actual[absorbing] - 1)) > 1e-6


def test_crossing_dipole_uses_the_h2db_regime_in_every_cell(monkeypatch, atmosphere, transitions):
    import wd_spectra.models.dah as dah
    from wd_spectra.spectrum import Spectrum

    config = DAHConfig(field_geometry="dipole", magnetic_field_megagauss=1.1,
                       include_centered_motion=False)
    cells = dah.dah_surface_cells(config)
    field = dah.structure_field_megagauss(cells)
    assert field < 1 < cells.field_strength_megagauss.max()
    wave = np.linspace(3900., 7000., 20)

    def da(*args, **kwargs):
        return SimpleNamespace(atmosphere=atmosphere, metadata={"convection": "test"})

    def synthesis(atm, wavelength, cells, physics, **kwargs):
        assert physics.regime(field) == physics.regime(cells.field_strength_megagauss.max()) == "h2db"
        return Spectrum(wavelength, np.ones_like(wavelength), {
            "magnetic_line_regime": "h2db", "polarized_transfer": "test",
        })

    monkeypatch.setattr(dah, "compute_da", da)
    monkeypatch.setattr(dah, "synthesize_magnetic_hydrogen_spectrum", synthesis)
    monkeypatch.setattr(dah, "warn_if_atmosphere_not_converged", lambda *args: "unknown")
    result = compute_dah(config, wave)
    assert result.metadata["line_physics"] == "h2db"


@pytest.mark.parametrize("polar,offset", [(0.02, 0.79), (0.05, 0.70)])
def test_polar_caps_keep_physical_field_bounds_and_strong_area(polar, offset):
    options = dict(inclination_deg=0., offset_vector_radius=(0., 0., offset))
    # One bin must not erase either the continuous bound or a strong cap.
    cells = dipole_surface_cells(polar, n_field_bins=1, **options)
    raw = dipole_surface_cells(polar, n_field_bins=None, **options)
    assert cells.field_bounds_exact
    assert cells.maximum_field_megagauss == pytest.approx(polar / (1-offset)**3, rel=2e-14)
    assert cells.maximum_field_megagauss > 1.
    assert np.any(cells.field_strength_megagauss > 1.)
    assert np.any(cells.field_strength_megagauss < 1.)
    for c in [cells, raw]:
        assert c.projected_weight.sum() == pytest.approx(1.)
    np.testing.assert_allclose(
        cells.projected_weight[cells.field_strength_megagauss > 1.].sum(),
        raw.projected_weight[raw.field_strength_megagauss > 1.].sum(), rtol=1e-14,
    )
    np.testing.assert_allclose(cells.projected_weight @ cells.field_strength_megagauss,
                               raw.projected_weight @ raw.field_strength_megagauss, rtol=1e-14)


@pytest.mark.parametrize("inclination", [0., 40., 90., 130., 180.])
@pytest.mark.parametrize("offset", [(0., 0., .7), (0., 0., -.7), (.12, -.2, .4)])
def test_continuous_field_bounds_enclose_independent_dense_surface(inclination, offset):
    options = dict(inclination_deg=inclination, offset_vector_radius=offset)
    cells = dipole_surface_cells(30., n_field_bins=3, **options)
    fine = dipole_surface_cells(30., n_mu=100, n_azimuth=200, n_field_bins=None, **options)
    lo, hi = cells.field_bounds_megagauss
    assert lo <= fine.field_strength_megagauss.min() * (1+1e-13)
    assert hi >= fine.field_strength_megagauss.max() * (1-1e-13)
    assert cells.field_bounds_exact == (offset[:2] == (0., 0.))


@pytest.mark.parametrize("field", [.001, .01, .03])
def test_h2db_weak_continuation_has_analytic_triplets(transitions, field):
    from wd_spectra.opacity import BALMER_LINES

    for line in BALMER_LINES[:10]:
        result = transitions.balmer_components(line.upper_level, line.wavelength_vacuum_angstrom,
                                                field, np.array([8000., 25000.]))
        assert result.delta_m.tolist() == [-1, 0, 1]
        np.testing.assert_array_equal(result.line_strength_scale, [1., 1.])
        np.testing.assert_allclose(result.normalized_strength, 1/3, rtol=1e-14)
        triplet = linear_zeeman_triplet(line.wavelength_vacuum_angstrom, field)
        # H2db uses the reduced-mass line-center convention; the displacement
        # agrees with the analytic physical-constant triplet within 0.001 A.
        np.testing.assert_allclose(result.wavelength_angstrom,
            [triplet.sigma_red_angstrom, triplet.pi_angstrom, triplet.sigma_blue_angstrom],
            atol=.001, rtol=0.)


@pytest.mark.parametrize("polar,offset", [(0.02, .79), (.05, .70)])
def test_public_mixed_field_synthesis_stays_inside_atomic_domain(monkeypatch, atmosphere, polar, offset):
    import wd_spectra.models.dah as dah

    monkeypatch.setattr(dah, "warn_if_atmosphere_not_converged", lambda *args: "unverified")
    config = DAHConfig(effective_temperature=atmosphere.effective_temperature, logg=atmosphere.logg,
        magnetic_field_megagauss=polar, field_geometry="dipole", dipole_inclination_deg=0.,
        dipole_offset_radius=(0.,0.,offset))
    result = compute_dah(config, np.array([4000.,5000.,6000.]), initial_atmosphere=atmosphere,
                         relax_atmosphere=False)
    assert result.metadata["line_physics"] == "h2db"
    assert result.metadata["maximum_visible_field_megagauss"] == pytest.approx(polar/(1-offset)**3)
    assert np.all(np.isfinite(result.spectrum.surface_flux_lambda))
    assert np.all(result.spectrum.surface_flux_lambda > 0.)


def test_centered_motion_switch_reaches_absorption_and_dispersion(monkeypatch, atmosphere):
    import wd_spectra.magnetic as magnetic
    import wd_spectra.models.dah as dah

    calls = []

    def continuum(atm, wave, field, polarization, energies, **kwargs):
        calls.append((wave.size, polarization, kwargs.get("include_centered_motion")))
        return np.zeros((wave.size, atm.n_depth))

    monkeypatch.setattr(magnetic, "explicit_rwa_hydrogen_bound_free_mass_absorption_coefficient", continuum)
    monkeypatch.setattr(dah, "warn_if_atmosphere_not_converged", lambda *args: "unknown")
    wave = np.linspace(4000., 7000., 20)
    compute_dah(
        DAHConfig(magnetic_field_megagauss=100., field_angle_deg=40.,
                  include_centered_motion=False, include_magnetic_eos=False,
                  include_rwa_photoionization=True, polarized_transfer="full-stokes-iquv"),
        wave, initial_atmosphere=atmosphere, relax_atmosphere=False,
    )
    assert len(calls) == 6
    assert {n for n, _, _ in calls} == {wave.size, wave.size + 1000}
    assert all(flag is False for _, _, flag in calls)


def test_dah_physics_revision_invalidates_old_fixed_structure_claim(monkeypatch, atmosphere):
    from wd_spectra.models import common

    config = DAHConfig(effective_temperature=atmosphere.effective_temperature, logg=atmosphere.logg)
    data = ModelData.default()
    with monkeypatch.context() as historical:
        historical.setattr(common, "_MODEL_FAMILY_PHYSICS_REVISIONS", {})
        old = common.model_request_fingerprint("DAH", config, data)
    current = common.model_request_fingerprint("DAH", config, data)
    checkpoint = common.atmosphere_with_model_request_fingerprint(atmosphere, old)
    fixed = common.fixed_synthesis_atmosphere(checkpoint, current)
    assert old["sha256"] != current["sha256"]
    assert not fixed.metadata["fixed_synthesis_request_verified"]


@pytest.mark.parametrize("return_stokes", [False, True])
def test_scalar_synthesis_shares_the_structure_claim_of_iquv(monkeypatch, atmosphere, return_stokes):
    import warnings
    import wd_spectra.models.dah as dah

    captured = {}

    def da(*args, **kwargs):
        captured["atmosphere"] = dah.atmosphere_with_model_request_fingerprint(atmosphere, {})
        return SimpleNamespace(atmosphere=captured["atmosphere"], metadata={"convection": "test"})

    monkeypatch.setattr(dah, "compute_da", da)
    monkeypatch.setattr(dah, "warn_if_atmosphere_not_converged", lambda *args: "converged")
    wave = np.linspace(6500.0, 6600.0, 5)
    config = DAHConfig(
        effective_temperature=atmosphere.effective_temperature, logg=atmosphere.logg,
        polarized_transfer="full-stokes-iquv",
        return_stokes=return_stokes, sky_position_angle_deg=37. if return_stokes else 0.,
    )
    first = compute_dah(config, wave)
    second = compute_dah(
        replace(config, polarized_transfer="scalar-stokes-i", return_stokes=False,
                sky_position_angle_deg=0.), wave,
        initial_atmosphere=first.atmosphere, relax_atmosphere=False,
    )
    assert second.atmosphere.metadata["fixed_synthesis_request_verified"]
    if return_stokes:
        rotated = compute_dah(
            replace(config, sky_position_angle_deg=83.), wave,
            initial_atmosphere=first.atmosphere, relax_atmosphere=False,
        )
        assert rotated.atmosphere.metadata["fixed_synthesis_request_verified"]
        assert rotated.metadata["model_request_fingerprint"] == first.metadata["model_request_fingerprint"]
    changed_physics = compute_dah(
        replace(config, include_rwa_photoionization=not config.include_rwa_photoionization), wave,
        initial_atmosphere=first.atmosphere, relax_atmosphere=False,
    )
    assert not changed_physics.atmosphere.metadata["fixed_synthesis_request_verified"]
    other = compute_dah(
        replace(config, logg=config.logg + 0.1), wave,
        initial_atmosphere=first.atmosphere, relax_atmosphere=False,
    )
    assert not other.atmosphere.metadata["fixed_synthesis_request_verified"]


@pytest.mark.parametrize("wave", [[], [4000.], [np.nan, 5000.], [4000., np.inf],
                                  [5000., 4000.], [4000., 4000.]])
def test_invalid_dah_grid_is_rejected_before_structure_work(monkeypatch, wave):
    import wd_spectra.models.dah as dah

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid grids must fail before solving an atmosphere")

    monkeypatch.setattr(dah, "compute_da", forbidden)
    with pytest.raises(ValueError, match="finite, increasing 1D wavelength grid"):
        compute_dah(DAHConfig(), wave)
