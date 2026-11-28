"""Motional Stark kernel of the H-alpha and H-beta Zeeman components (balmer_line_broadening="motional")."""

import numpy as np
import pytest

from wd_spectra._compat import trapezoid
from wd_spectra.atmosphere import gray_hydrogen_atmosphere
from wd_spectra.constants import BOLTZMANN, HYDROGEN_MASS, LIGHT_SPEED
from wd_spectra.magnetic import (
    BalmerLineTemplate, _kurucz_griem_templates, _motional_stark_template, _translated_profile,
    h2db_balmer_manifolds, motional_stark_profile, motional_stark_shift_hz,
)
from wd_spectra.magnetic_atomic import (
    MotionalStarkTable, read_h2db_transition_database, read_motional_stark_table,
)
from wd_spectra.models.common import ModelData
from wd_spectra.opacity import BALMER_LINES


def _data():
    data = ModelData.default()
    return read_h2db_transition_database(data.h2db_balmer_subset), read_motional_stark_table(data.motional_stark_table)


def test_profile_area_mean_and_doppler_limit():
    sigma = 1.0
    x = np.linspace(-40., 800., 840001)
    for shift in (3., -3., 30.):
        p = motional_stark_profile(x if shift > 0 else -x, sigma, shift)
        assert trapezoid(p, x) == pytest.approx(1., rel=1e-6)
        assert trapezoid(x * p, x) == pytest.approx(abs(shift), rel=1e-6)
    # s -> 0 (B -> 0 or motional_scale = 0): the Doppler profile.
    gauss = np.exp(-.5 * x**2) / np.sqrt(2 * np.pi)
    np.testing.assert_allclose(motional_stark_profile(x, sigma, 0.), gauss, rtol=1e-9, atol=1e-300)
    np.testing.assert_allclose(motional_stark_profile(x, sigma, 1e-6), gauss, rtol=0, atol=1e-6)


def test_table_wavelengths_match_h2db_components():
    db, table = _data()
    rest = BALMER_LINES[0].wavelength_vacuum_angstrom
    with np.load(ModelData.default().motional_stark_table) as values:
        grid, labels, wavelength = values["field_megagauss"], values["labels"], values["wavelength_infinite_mass_angstrom"]
    for field in (190., 280., 450.):
        h2db = {}
        for upper, line in ((3, rest), (4, BALMER_LINES[1].wavelength_vacuum_angstrom)):
            components = db.balmer_components(upper, line, field, np.array([18000.]), minimum_relative_strength=0.)
            h2db.update({tuple(label): w for label, w in zip(components.labels, components.wavelength_angstrom)})
        j = int(np.argmin(np.abs(grid - field)))
        for label, w in zip(labels, wavelength[:, j]):
            if 0. < w < 2e4:  # optical/IR components (others vanish or turn negative)
                assert h2db[tuple(label)] == pytest.approx(w, rel=1e-3)


def test_stationary_pi_component_shift():
    # Halpha pi 2s0-3p0 at its stationary field: mean shift +7.0 A to the red at
    # 17.7 kK (s = Delta alpha kT beta^2 / M, Pavlov & Meszaros 1993, Eq. 3.28).
    _, table = _data()
    shift = motional_stark_shift_hz(table.delta_alpha_at((0, 1, 2, 0, -1, 2), 231.9), 231.9, 17700.)
    assert 5829.7**2 * shift / LIGHT_SPEED * 1e-8 == pytest.approx(7.0, abs=.05)


def test_motional_lookup_keeps_finite_knots_beside_flagged_nodes():
    label = (0, 1, 2, 0, -1, 2)
    table = MotionalStarkTable(
        np.array([100., 105., 110., 115.]), {label: np.array([4., np.nan, 8., 10.])},
    )
    assert table.delta_alpha_at(label, 100.) == 4.
    assert np.isnan(table.delta_alpha_at(label, 102.5))
    assert np.isnan(table.delta_alpha_at(label, 105.))
    assert table.delta_alpha_at(label, 112.5) == 9.
    assert table.delta_alpha_at(label, 115.) == 10.


def test_motional_table_keeps_weak_h2db_cell_on_analytic_triplet():
    db, table = _data()
    atmosphere = gray_hydrogen_atmosphere(18_000., 8., n_depth=4)
    wave = np.linspace(6560., 6568., 65)
    templates = _kurucz_griem_templates(atmosphere, 3, True)
    # A positive field below the table floor can occur in a strong-field map.
    baseline = h2db_balmer_manifolds(
        atmosphere, wave, .01, db, templates=templates, maximum_upper_level=3, dispersion=True,
    )
    motional = h2db_balmer_manifolds(
        atmosphere, wave, .01, db, templates=templates, maximum_upper_level=3, dispersion=True,
        motional_stark_table=table,
    )
    for plain, actual in zip(baseline, motional):
        for q in ("minus", "pi", "plus"):
            np.testing.assert_array_equal(getattr(actual, q), getattr(plain, q))


@pytest.mark.parametrize("scale", [1., 2.])
def test_broad_red_motional_component_survives_in_physical_frequency(scale):
    db, table = _data()
    atmosphere = gray_hydrogen_atmosphere(18_000., 8., n_depth=12)
    field = 450.
    components = db.balmer_components(3, BALMER_LINES[0].wavelength_vacuum_angstrom,
                                     field, atmosphere.temperature)
    label = (0, 1, 2, -1, 1, 2)
    i = next(i for i, row in enumerate(components.labels) if tuple(row) == label)
    center = LIGHT_SPEED / (components.wavelength_angstrom[i] * 1e-8)
    shift = scale * motional_stark_shift_hz(table.delta_alpha_at(label, field),
                                           field, atmosphere.temperature)
    template = _kurucz_griem_templates(atmosphere, 3, False)[3]
    kernel = _motional_stark_template(template, center, shift, atmosphere.temperature, True)
    assert np.all(np.isfinite(kernel.wavelength_angstrom))
    assert kernel.wavelength_angstrom[0] > 0.
    assert np.all(np.diff(kernel.wavelength_angstrom) > 0.)
    assert np.all(np.isfinite(kernel.dispersion))
    sigma = center * np.sqrt(BOLTZMANN * atmosphere.temperature / HYDROGEN_MASS) / LIGHT_SPEED
    frequency = np.unique(np.concatenate((
        center * np.geomspace(1.e-6, 4., 16_001),
        center + np.linspace(-12. * sigma.max(), 12. * sigma.max(), 3001),
    )))
    sample = _translated_profile(kernel, LIGHT_SPEED / frequency[::-1] / 1.e-8, 0.)[::-1]
    assert np.all(sample[np.searchsorted(frequency, center)] > 0.)
    np.testing.assert_allclose(trapezoid(sample, frequency, axis=0), template.strength, rtol=1.e-3)


@pytest.mark.parametrize("scale", [1., 0.])
def test_motional_kernel_conserves_line_strength(scale):
    db, table = _data()
    atm = gray_hydrogen_atmosphere(18000., 8., n_depth=8)
    templates = _kurucz_griem_templates(atm, 3, False)
    nu = np.linspace(LIGHT_SPEED / 1.2e-4, LIGHT_SPEED / 2.5e-5, 400001)  # 2500-12000 A, finer than Doppler
    wave = LIGHT_SPEED / nu[::-1] / 1e-8
    # Only Halpha components the table lists at 280 MG take the motional kernel;
    # with a template of zero-field Stark area, the summed manifold area is the line strength.
    opacity = h2db_balmer_manifolds(atm, wave, 280., db, templates=templates, maximum_upper_level=3,
                                    motional_stark_table=table, motional_scale=scale).isotropic()
    reference = h2db_balmer_manifolds(atm, wave, 280., db, templates=templates, maximum_upper_level=3).isotropic()
    area = trapezoid(opacity[::-1], nu, axis=0)
    thin = atm.rosseland_optical_depth < .1  # zero-field Stark wings fit inside the grid
    # rtol 2e-3: the exact profiles hold the strength of their windows, which end 0.12 lambda0 from the line
    # (at 280 MG the H-alpha pi rows keep 0.995-1.000 of it; -0.06 % of all H-alpha strength).
    np.testing.assert_allclose(area[thin], trapezoid(reference[::-1], nu, axis=0)[thin], rtol=2e-3)
    assert np.all(area[~thin] > 0)


def test_motional_dispersion_is_the_hilbert_partner_of_its_absorption():
    # On a disk the V/I of narrow lines comes from their dispersion wing, so the
    # kernel's partner must hold far from the line, not only in its core.
    from wd_spectra.magnetic import frequency_hilbert_dispersion

    _, table = _data()
    atm = gray_hydrogen_atmosphere(18000., 8., n_depth=4)
    template = _kurucz_griem_templates(atm, 3, False)[3]
    rest = LIGHT_SPEED / (BALMER_LINES[0].wavelength_vacuum_angstrom * 1e-8)
    # sigma- 2s0-3d-1 and pi 2s0-3p0 at 280 MG; the sign flip checks a blue (Delta alpha < 0) wing.
    for label, sign in (((0, 1, 2, -1, 1, 2), 1.), ((0, 1, 2, 0, -1, 2), -1.)):
        shift = sign * motional_stark_shift_hz(table.delta_alpha_at(label, 280.), 280., atm.temperature)
        kernel = _motional_stark_template(template, rest, shift, atm.temperature, True)
        wave = np.unique(np.concatenate((kernel.wavelength_angstrom, np.geomspace(3000., 12000., 2000))))
        numeric = frequency_hilbert_dispersion(wave, _translated_profile(kernel, wave, 0.))
        phase = _translated_profile(kernel, wave, 0., kernel.dispersion)
        np.testing.assert_allclose(phase, numeric, rtol=0, atol=2e-3 * np.abs(numeric).max())
        far = np.abs(wave - BALMER_LINES[0].wavelength_vacuum_angstrom) > 500.  # the Faraday wing
        np.testing.assert_allclose(phase[far], numeric[far], rtol=2e-3)


def _profile_moments(table, label, field, temperature, edge):
    """Blue and red shares and red mean within +-edge A of the exact profile at one temperature."""
    knots, cumulative = (v[0] for v in table.profile_at(label, field, np.array([temperature])))
    x = np.linspace(0., edge, 100001)
    c = np.interp(x, knots, cumulative)
    red = c[-1] - c[0]
    return dict(blue=c[0] - np.interp(-edge, knots, cumulative), red=red,
                red_mean=edge - trapezoid(c - c[0], x) / red)  # int x dC by parts


def test_exact_hbeta_sigma_profile_at_366_megagauss():
    # sigma- 2s0-4f-1 at 366 MG, 17.7 kK, inside +-296.6 A: the solver's own lines at the 366 MG node, weighted by
    # e^-x dx and by omega/omega0 (oscillator strengths), give 0.2451 blue and 0.7329 red with mean 47.94 A. The
    # table's quantiles, interpolated between 360 and 370 MG and 15 and 20 kK, put the red mean ~0.3 A lower.
    _, table = _data()
    label = (0, 1, 2, -1, 1, 3)
    assert np.isnan(table.delta_alpha_at(label, 366.))
    m = _profile_moments(table, label, 366., 17700., 296.6)
    assert m['blue'] == pytest.approx(0.2451, abs=0.005)
    assert m['red'] == pytest.approx(0.7329, abs=0.005)
    assert m['red_mean'] == pytest.approx(47.94, abs=1.5)


def test_exact_windows_keep_the_sigma_strength():
    # The windows of sigma- 2s0-4f-1 and 2s0-4p-1 tile the spectrum, so the strength they hold together is close to
    # the F = 0 strength of the two lines; their windows reach 3648-5845 A, where ~1.4 % more comes in from outside
    # the pair (the next F = 0 line is 3p-1 at 7280 A). Same lower level, so H2db's strengths weigh the shares.
    db, table = _data()
    components = db.balmer_components(4, BALMER_LINES[1].wavelength_vacuum_angstrom, 366., np.array([17700.]))
    s = {tuple(label): w[0] for label, w in zip(components.labels, components.normalized_strength)}
    labels = [(0, 1, 2, -1, 1, 3), (0, 1, 2, -1, 1, 4)]
    share = [table.profile_at(label, 366., np.array([17700.]))[1][0, -1] for label in labels]
    assert sum(s[label] * w for label, w in zip(labels, share)) == pytest.approx(sum(s[label] for label in labels), rel=2e-2)


def test_profile_on_a_node_next_to_a_field_without_one():
    # On a field node only that node counts: a node whose upper neighbour has no exact profile still gives its own.
    # A single node without a profile is bridged by its neighbours (2s0-4f-1 at 285 MG: the lower state mixes with
    # the lowest m = -2 state in a few F cells); next to two or more such nodes there is no profile.
    _, table = _data()
    label, grid = (0, 1, 2, -1, 1, 3), table.profile_field_megagauss
    ok = np.isfinite(table.profiles[label][1][:, 0, 0])
    assert not ok[grid == 285.] and ok[grid == 280.] and ok[grid == 290.]
    bridged = [table.profile_at(label, b, np.array([17000.]))[1][0, -1] for b in (280., 285., 290.)]
    assert bridged[1] == pytest.approx(0.5 * (bridged[0] + bridged[2]))
    k = np.flatnonzero(ok[:-2] & ~ok[1:-1] & ~ok[2:])[0]
    assert table.profile_at(label, grid[k], np.array([17000.])) is not None
    assert table.profile_at(label, grid[k] + 1.0, np.array([17000.])) is None


def test_exact_profile_keeps_narrow_structure_far_from_the_line():
    # H-alpha sigma+ (-1,1,1,0,1,3) at 300 MG, 17.7 kK: ~1.4 % of its strength sits in a narrow pile-up at +329 to
    # +337 A from the line. The kernel's strength between fixed offsets there equals the table's to 1 %, because the
    # kernel's mesh contains the profile's knots.
    _, table = _data()
    label, field, temperature = (-1, 1, 1, 0, 1, 3), 300., np.array([17700.])
    values = np.load(ModelData.default().motional_stark_table)
    row = [tuple(int(x) for x in v) for v in values["labels"]].index(label)
    lam0 = float(np.interp(field, values["field_megagauss"], values["wavelength_infinite_mass_angstrom"][row]))
    knots, cumulative = table.profile_at(label, field, temperature)
    nu0 = LIGHT_SPEED / (lam0 * 1e-8)
    template = BalmerLineTemplate(np.array([lam0]), np.zeros((1, 1)), None, np.ones(1))
    kernel = _motional_stark_template(template, nu0, None, temperature, True, (knots, cumulative))
    detuning = nu0 - LIGHT_SPEED / (kernel.wavelength_angstrom * 1e-8)  # positive towards the red
    order = np.argsort(detuning)
    d, opacity = detuning[order], kernel.mass_absorption_coefficient[order, 0]
    strength = np.concatenate(([0.], np.cumsum(0.5 * (opacity[1:] + opacity[:-1]) * np.diff(d))))

    def held(offset):
        return np.interp(nu0 - LIGHT_SPEED / ((lam0 + offset) * 1e-8), d, strength)

    for a, b in ((325., 333.), (333., 338.)):
        expected = np.interp(b, knots[0], cumulative[0]) - np.interp(a, knots[0], cumulative[0])
        assert expected > 1e-3
        assert held(b) - held(a) == pytest.approx(expected, rel=1e-2)


def test_exact_profiles_with_dispersion_at_98_megagauss():
    # Knots of different depths can nearly coincide on the kernel mesh; with dispersion on, the Hilbert transform
    # rejects coincident nodes with different values. 98 MG on a 64-depth atmosphere triggered it.
    db, table = _data()
    atmosphere = gray_hydrogen_atmosphere(17_000., 8.5, n_depth=64)
    wave = np.arange(3700., 8000., 1.0)
    templates = _kurucz_griem_templates(atmosphere, 4, True)
    manifolds = h2db_balmer_manifolds(atmosphere, wave, 98., db, templates=templates, maximum_upper_level=4,
                                      dispersion=True, motional_stark_table=table)
    assert all(np.all(np.isfinite(getattr(m, q))) for m in manifolds for q in ("minus", "pi", "plus"))
