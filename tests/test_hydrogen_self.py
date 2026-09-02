import struct

import numpy as np

from wd_spectra import (
    BarklemSelfBroadeningTable,
    gray_hydrogen_atmosphere,
    read_barklem_self_broadening_table,
)
from wd_spectra.opacity import BALMER_LINES, balmer_mass_absorption_coefficient


def _write_record(stream, payload: bytes) -> None:
    marker = struct.pack("<i", len(payload))
    stream.write(marker)
    stream.write(payload)
    stream.write(marker)


def test_reads_little_endian_barklem_fortran_table(tmp_path):
    path = tmp_path / "bpo_self.grid.DEC"
    densities = np.asarray([1.0e14, 1.0e15], dtype="<f8")
    temperatures = np.asarray([5_000.0, 10_000.0], dtype="<f8")
    offsets = np.asarray([-1.0, 0.0, 1.0])
    with path.open("wb") as stream:
        _write_record(stream, struct.pack("<i", 1))
        _write_record(stream, np.asarray([2, 3], dtype="<i4").tobytes())
        _write_record(stream, struct.pack("<i", densities.size))
        _write_record(stream, densities.tobytes())
        _write_record(stream, struct.pack("<i", temperatures.size))
        _write_record(stream, temperatures.tobytes())
        _write_record(stream, struct.pack("<i", offsets.size))
        for density_index in range(densities.size):
            for temperature_index in range(temperatures.size):
                scale = 1.0 + density_index + 0.5 * temperature_index
                profile = np.asarray([0.2, scale, 0.3])
                for offset, value in zip(offsets, profile):
                    _write_record(stream, struct.pack("<dd", offset, value))

    table = read_barklem_self_broadening_table(path)
    assert table.transitions == ((2, 3),)
    assert table.profile_per_angstrom.shape == (1, 2, 2, 3)
    interpolated = table.profile_at_state((2, 3), 7_000.0, 3.0e14)
    np.testing.assert_allclose(
        np.trapz(interpolated, table.wavelength_offset_angstrom),
        1.0,
        rtol=2.0e-15,
    )


def test_full_barklem_kernel_changes_balmer_opacity_without_losing_finiteness():
    atmosphere = gray_hydrogen_atmosphere(7_000.0, 8.0, n_depth=8)
    # A broad, normalized synthetic kernel exercises the same convolution
    # path as the official table without making the test suite ship that
    # external BSD-licensed binary asset.
    offset = np.asarray([-20.0, -2.0, 0.0, 2.0, 20.0])
    base_profile = np.asarray([0.001, 0.08, 0.35, 0.07, 0.001])
    base_profile /= np.trapz(base_profile, offset)
    density = np.asarray([1.0e10, 1.0e30])
    temperature = np.asarray([1_000.0, 100_000.0])
    profiles = np.broadcast_to(
        base_profile,
        (1, density.size, temperature.size, offset.size),
    ).copy()
    table = BarklemSelfBroadeningTable(
        lower_level=np.asarray([2]),
        upper_level=np.asarray([3]),
        neutral_h_density=density,
        temperature=temperature,
        wavelength_offset_angstrom=offset,
        profile_per_angstrom=profiles,
        source="synthetic unit-test table",
    )
    wavelength = np.linspace(6540.0, 6590.0, 81)
    grid = balmer_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        lines=(BALMER_LINES[0],),
        self_broadening_prescription="barklem-grid",
        barklem_self_table=table,
    )
    impact = balmer_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        lines=(BALMER_LINES[0],),
        self_broadening_prescription="barklem",
    )
    assert np.all(np.isfinite(grid))
    assert np.all(grid >= 0.0)
    assert not np.allclose(grid, impact)
