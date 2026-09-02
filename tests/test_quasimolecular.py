from pathlib import Path

import numpy as np
import pytest

from wd_spectra import gray_hydrogen_atmosphere
from wd_spectra.opacity import (
    LYMAN_LINES,
    allard_unified_lyman_alpha_mass_absorption_coefficient,
    lyman_mass_absorption_coefficient,
    _allard_profile_temperature_by_depth,
    _allard_table_coverage_switch,
)
from wd_spectra.quasimolecular import (
    ALLARD_TEMPERATURE_GRID_FILES,
    AllardUnifiedLymanTable,
    AllardNeutralLymanAlphaTable,
    read_allard_hydrogen_line_profile,
    read_allard_neutral_lyman_alpha_profile,
    read_allard_temperature_dependent_tables,
    read_allard_tlusty205_tables,
)


def _write_synthetic_allard_table(
    path: Path,
    scale: float,
    *,
    neutral_excluded_volume: float = 2.0e-2,
    proton_excluded_volume: float = 3.0e-2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        (
            "3 17.0 18.0 "
            f"{neutral_excluded_volume:.8E} {proton_excluded_volume:.8E}"
        ),
        f"1200.0 {1*scale} {2*scale} {3*scale} {4*scale} {5*scale}",
        f"1250.0 {1*scale} {2*scale} {3*scale} {4*scale} {5*scale}",
        f"1300.0 {1*scale} {2*scale} {3*scale} {4*scale} {5*scale}",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_allard_synspec_reader_reproduces_five_term_density_expansion(tmp_path):
    path = tmp_path / "alpha_10000.dat"
    _write_synthetic_allard_table(path, 1.0)
    profile = read_allard_neutral_lyman_alpha_profile(path, 10_000.0)

    cross_section = profile.profile_cross_section(
        np.array([1210.0, 1275.0]),
        np.array([2.0e17]),
        np.array([1.0e17]),
    )
    neutral_scaled = 2.0
    proton_scaled = 0.1
    expansion = (
        neutral_scaled
        + 2.0 * neutral_scaled**2
        + 3.0 * proton_scaled
        + 4.0 * proton_scaled**2
        + 5.0 * neutral_scaled * proton_scaled
    )
    volume_sum = 0.02 * neutral_scaled + 0.03 * proton_scaled
    finite_volume = 1.0 / (1.0 + volume_sum + 0.5 * volume_sum**2)
    line_normalization = 8.8528e-29 * 1215.6**2 * 0.41618
    expected = expansion * finite_volume * line_normalization
    np.testing.assert_allclose(cross_section[:, 0], expected, rtol=1.0e-14)


def test_allard_lyman_beta_uses_tlusty205_line_normalization(tmp_path):
    path = tmp_path / "lbquasi.dat"
    _write_synthetic_allard_table(path, 1.0)
    profile = read_allard_hydrogen_line_profile(
        path, lower_level=1, upper_level=3
    )
    cross_section = profile.profile_cross_section(
        np.array([1210.0]), np.array([1.0e17]), np.array([1.0e18])
    )
    expansion = 1.0 + 2.0 + 3.0 + 4.0 + 5.0
    finite_volume = 1.0 / (1.0 + 0.05 + 0.5 * 0.05**2)
    normalization = 8.8528e-29 * 1025.73 * 1025.7 * 0.0791
    np.testing.assert_allclose(
        cross_section[0, 0], expansion * finite_volume * normalization
    )


def test_allard_tlusty205_directory_reader_loads_three_transitions(tmp_path):
    for filename in ("laquasi.dat", "lbquasi.dat", "lgquasi.dat"):
        _write_synthetic_allard_table(tmp_path / filename, 1.0)
    table = read_allard_tlusty205_tables(tmp_path, verify_checksums=False)
    assert set(table.lines) == {(1, 2), (1, 3), (1, 4)}
    assert all(line.profiles[0].temperature_K is None for line in table.lines.values())


def test_allard_temperature_delivery_reader_loads_manifest_and_gamvar(tmp_path):
    for files_by_temperature in ALLARD_TEMPERATURE_GRID_FILES.values():
        for temperature, relative_path in files_by_temperature.items():
            _write_synthetic_allard_table(
                tmp_path / relative_path, temperature / 10_000.0
            )
    legacy_gamma = tmp_path / "Lyman_gamma/fort_GAMMA_abs25000.23"
    _write_synthetic_allard_table(legacy_gamma, 99.0)

    table = read_allard_temperature_dependent_tables(
        tmp_path, verify_checksums=False
    )

    assert set(table.lines) == {(1, 2), (1, 3), (1, 4)}
    np.testing.assert_array_equal(
        table[(1, 2)].temperatures_K,
        [9_000.0, 10_000.0, 11_000.0, 12_000.0, 13_000.0, 25_000.0],
    )
    np.testing.assert_array_equal(
        table[(1, 3)].temperatures_K,
        [12_000.0, 25_000.0, 40_000.0, 50_000.0, 60_000.0],
    )
    np.testing.assert_array_equal(table[(1, 4)].temperatures_K, [25_000.0])
    assert table[(1, 4)].profiles[0].source_path.name == "fort.23_GAMVAR_abs25"


def test_allard_temperature_grid_interpolates_in_log_temperature(tmp_path):
    low_path = tmp_path / "alpha_10000.dat"
    high_path = tmp_path / "alpha_20000.dat"
    _write_synthetic_allard_table(low_path, 1.0)
    _write_synthetic_allard_table(high_path, 3.0)
    low = read_allard_neutral_lyman_alpha_profile(low_path, 10_000.0)
    high = read_allard_neutral_lyman_alpha_profile(high_path, 20_000.0)
    table = AllardNeutralLymanAlphaTable((low, high))
    wavelength = np.array([1220.0])
    neutral = np.array([1.0e17])
    proton = np.array([1.0e17])

    interpolated = table.profile_cross_section(
        wavelength, np.array([np.sqrt(2.0) * 10_000.0]), neutral, proton
    )
    expected = 0.5 * (
        low.profile_cross_section(wavelength, neutral, proton)
        + high.profile_cross_section(wavelength, neutral, proton)
    )
    np.testing.assert_allclose(interpolated, expected, rtol=1.0e-14)


def test_allard_temperature_grid_interpolates_finite_volume_weight(tmp_path):
    low_path = tmp_path / "alpha_10000.dat"
    high_path = tmp_path / "alpha_20000.dat"
    _write_synthetic_allard_table(
        low_path,
        1.0,
        neutral_excluded_volume=0.02,
        proton_excluded_volume=0.03,
    )
    _write_synthetic_allard_table(
        high_path,
        1.0,
        neutral_excluded_volume=0.06,
        proton_excluded_volume=0.09,
    )
    low = read_allard_neutral_lyman_alpha_profile(low_path, 10_000.0)
    high = read_allard_neutral_lyman_alpha_profile(high_path, 20_000.0)
    table = AllardNeutralLymanAlphaTable((low, high))
    neutral = np.array([2.0e17])
    proton = np.array([1.0e18])

    interpolated = table.unperturbed_absorber_probability(
        np.array([np.sqrt(2.0) * 10_000.0]), neutral, proton
    )
    expected = 0.5 * (
        low.unperturbed_absorber_probability(neutral, proton)
        + high.unperturbed_absorber_probability(neutral, proton)
    )
    np.testing.assert_allclose(interpolated, expected, rtol=1.0e-14)


def test_allard_temperature_grid_vectorizes_one_exact_profile_over_depth(tmp_path):
    low_path = tmp_path / "alpha_10000.dat"
    high_path = tmp_path / "alpha_20000.dat"
    _write_synthetic_allard_table(low_path, 1.0)
    _write_synthetic_allard_table(high_path, 3.0)
    low = read_allard_neutral_lyman_alpha_profile(low_path, 10_000.0)
    table = AllardNeutralLymanAlphaTable(
        (
            low,
            read_allard_neutral_lyman_alpha_profile(high_path, 20_000.0),
        )
    )
    wavelength = np.array([1210.0, 1275.0, 1400.0])
    temperature = np.full(4, 10_000.0)
    neutral = np.geomspace(1.0e15, 1.0e18, 4)
    proton = np.geomspace(1.0e14, 1.0e17, 4)

    expected_profile = low.profile_cross_section(
        wavelength,
        neutral,
        proton,
        extend_lyman_alpha_red_wing=True,
    )
    expected_probability = low.unperturbed_absorber_probability(neutral, proton)
    np.testing.assert_allclose(
        table.profile_cross_section(
            wavelength,
            temperature,
            neutral,
            proton,
            extend_lyman_alpha_red_wing=True,
        ),
        expected_profile,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        table.unperturbed_absorber_probability(temperature, neutral, proton),
        expected_probability,
        rtol=0.0,
        atol=0.0,
    )


def test_allard_default_temperature_selection_uses_one_nearest_profile(tmp_path):
    low_path = tmp_path / "alpha_9000.dat"
    high_path = tmp_path / "alpha_20000.dat"
    _write_synthetic_allard_table(low_path, 1.0)
    _write_synthetic_allard_table(high_path, 3.0)
    table = AllardNeutralLymanAlphaTable(
        (
            read_allard_neutral_lyman_alpha_profile(low_path, 9_000.0),
            read_allard_neutral_lyman_alpha_profile(high_path, 20_000.0),
        )
    )
    atmosphere = gray_hydrogen_atmosphere(10_000.0, 8.0, n_depth=12)
    selected = _allard_profile_temperature_by_depth(table, atmosphere)
    np.testing.assert_array_equal(selected, 9_000.0)


def test_allard_profile_converts_to_positive_atmosphere_mass_opacity(tmp_path):
    path = tmp_path / "alpha_10000.dat"
    _write_synthetic_allard_table(path, 1.0)
    profile = read_allard_neutral_lyman_alpha_profile(path, 10_000.0)
    table = AllardNeutralLymanAlphaTable((profile,))
    atmosphere = gray_hydrogen_atmosphere(10_000.0, 8.0, n_depth=12)
    wavelength = np.array([1150.0, 1210.0, 1275.0, 1350.0])
    opacity = allard_unified_lyman_alpha_mass_absorption_coefficient(
        atmosphere, wavelength, table
    )
    assert opacity.shape == (wavelength.size, atmosphere.n_depth)
    np.testing.assert_array_equal(opacity[0], 0.0)
    assert np.all(opacity[1:] > 0.0)


def test_allard_lyman_alpha_red_tail_matches_synspec_power_law(tmp_path):
    path = tmp_path / "alpha.dat"
    _write_synthetic_allard_table(path, 1.0)
    profile = read_allard_neutral_lyman_alpha_profile(path, 10_000.0)
    wavelength = np.array([1400.0, 1500.0])
    cross_section = profile.profile_cross_section(
        wavelength,
        np.array([1.0e17]),
        np.array([1.0e17]),
        extend_lyman_alpha_red_wing=True,
    )[:, 0]
    expected_ratio = (
        (wavelength[1] - 1215.67) / (wavelength[0] - 1215.67)
    ) ** -2.5
    np.testing.assert_allclose(
        cross_section[1] / cross_section[0], expected_ratio, rtol=2.0e-14
    )


def test_allard_table_edge_taper_smoothly_restores_stark_wing(tmp_path):
    path = tmp_path / "laquasi.dat"
    _write_synthetic_allard_table(path, 1.0)
    line_table = AllardNeutralLymanAlphaTable(
        (
            read_allard_hydrogen_line_profile(
                path, lower_level=1, upper_level=2
            ),
        )
    )
    wavelength = np.array(
        [1199.0, 1200.0, 1200.75, 1201.6, 1291.5, 1295.75, 1300.0, 1301.0]
    )
    switch = _allard_table_coverage_switch(
        line_table,
        wavelength,
        LYMAN_LINES[0].wavelength_vacuum_angstrom,
    )

    np.testing.assert_array_equal(switch[[0, 1, -2, -1]], 0.0)
    np.testing.assert_allclose(switch[[3, 4]], 1.0)
    assert 0.0 < switch[2] < 1.0
    assert 0.0 < switch[5] < 1.0


def test_allard_combination_is_half_stark_plus_unified_profile(tmp_path):
    path = tmp_path / "laquasi.dat"
    _write_synthetic_allard_table(path, 1.0)
    line_table = AllardNeutralLymanAlphaTable(
        (read_allard_hydrogen_line_profile(path, lower_level=1, upper_level=2),)
    )
    allard = AllardUnifiedLymanTable(
        lines={(1, 2): line_table},
        source_directory=tmp_path,
        source_sha256={(1, 2): "synthetic"},
    )
    atmosphere = gray_hydrogen_atmosphere(10_000.0, 8.0, n_depth=12)
    wavelength = np.array([1150.0, 1215.67, 1250.0, 1400.0])
    alpha_line = (LYMAN_LINES[0],)
    stark = lyman_mass_absorption_coefficient(
        atmosphere, wavelength, lines=alpha_line
    )
    allard_only = allard_unified_lyman_alpha_mass_absorption_coefficient(
        atmosphere, wavelength, line_table
    )
    combined = lyman_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        unified_allard_table=allard,
        lines=alpha_line,
    )
    np.testing.assert_allclose(combined, 0.5 * stark + allard_only, rtol=2.0e-14)

    full_stark = lyman_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        unified_allard_table=allard,
        allard_stark_weight=1.0,
        lines=alpha_line,
    )
    np.testing.assert_allclose(full_stark, stark + allard_only, rtol=2.0e-14)


@pytest.mark.parametrize("weight", [-0.1, 1.1, np.nan])
def test_allard_stark_weight_must_lie_in_unit_interval(weight):
    atmosphere = gray_hydrogen_atmosphere(10_000.0, 8.0, n_depth=4)
    with pytest.raises(ValueError, match="allard_stark_weight"):
        lyman_mass_absorption_coefficient(
            atmosphere,
            np.array([1215.67]),
            allard_stark_weight=weight,
            lines=(LYMAN_LINES[0],),
        )
