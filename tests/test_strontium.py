"""Bundled Sr atomic data and the NIST strong-line file registry."""

import csv
import hashlib

import numpy as np
import pytest

from wd_spectra.atmosphere import hydrogen_helium_continuum_atmosphere
from wd_spectra.metals import (
    IONIZATION_ENERGY_EV,
    NIST_ASD_STRONG_ION_FILES,
    STRONTIUM_ATOMIC_DATA_FILES,
    _strontium_data_directory,
    metal_line_mass_absorption_coefficient,
    metal_lte_state,
    read_barklem_neutral_hydrogen_broadening,
    read_nist_asd_strong_atomic_database,
    read_stout_atomic_database,
    read_strontium_atomic_ion,
)
from wd_spectra.models.common import ModelData


DATA = ModelData.default()
# Ions screened by the independent-Kurucz rule; the NIST strong-line files
# cover each of them in the UV (115-300 nm) and optical (300-900 nm) windows.
SCREENED_ELEMENTS = ("C", "N", "O", "Na", "Mg", "Al", "Si", "P", "S", "Ca",
                     "Ti", "Cr", "Mn", "Fe", "Ni")


def _rows(name):
    directory = _strontium_data_directory(DATA.stout)
    with (directory / name).open(newline="", encoding="utf-8") as stream:
        return [
            {key: (value or "").strip().strip('"').lstrip("=").strip('"')
             for key, value in row.items() if key}
            for row in csv.DictReader(stream, delimiter="\t" if name.endswith(".tsv") else ",")
        ]


def test_strontium_is_a_consecutive_stout_compatible_ladder():
    database = read_stout_atomic_database(DATA.stout, elements=("Sr",), maximum_charge=4)
    stages = database.ion_stages("Sr")
    assert [ion.charge for ion in stages] == list(range(5))
    assert [len(ion.transitions) for ion in stages] == [86, 31, 0, 0, 0]
    assert all(ion.levels[0].energy_wavenumber == 0.0 for ion in stages)
    assert stages[0].levels[0].label == "5s2.(1S<0>)"
    assert stages[1].levels[0].label == "4p6.5s.(2S<1/2>)"
    temperature = np.array([4000.0, 6000.0, 9000.0])
    for ion in stages[:2]:
        partition = ion.partition_function(temperature)
        assert np.all(np.diff(partition) > 0.0)
        assert partition[0] >= ion.levels[0].statistical_weight
    with pytest.raises(ValueError, match="Sr I-V"):
        read_strontium_atomic_ion(DATA.stout, 5)


def test_strontium_files_and_ionization_energies_match_the_pinned_snapshot():
    directory = _strontium_data_directory(DATA.stout)
    for name, (_, checksum) in STRONTIUM_ATOMIC_DATA_FILES.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == checksum
    values = [float(row["Ionization Energy (eV)"]) for row in _rows("nist-asd-sr-ionization.csv")]
    assert IONIZATION_ENERGY_EV["Sr"] == tuple(values[:5])


def test_strontium_lines_reproduce_nist_wavelengths_and_strengths():
    ion = read_strontium_atomic_ion(DATA.stout, 1)
    by_wavelength = {round(line.wavelength_vacuum_angstrom, 1): line for line in ion.transitions}
    rows = [row for row in _rows("nist-asd-sr2-lines.tsv") if row["Aki(s^-1)"] and not row["Type"]]
    assert len(rows) == len(ion.transitions) == 31
    for row in rows:
        printed = row["ritz_wl_vac(nm)"]
        ritz = 10.0 * float(printed)
        # NIST prints some Ritz wavelengths with only three decimals in nm.
        rounding = 5.0 * 10.0 ** -len(printed.split(".")[1])
        line = min(ion.transitions, key=lambda item: abs(item.wavelength_vacuum_angstrom - ritz))
        assert line.wavelength_vacuum_angstrom == pytest.approx(ritz, abs=rounding + 1.0e-3)
        assert line.einstein_a == float(row["Aki(s^-1)"])
    # Resonance doublet, NIST grade AA (f = 0.703 and 0.341).
    assert by_wavelength[4078.9].absorption_oscillator_strength == pytest.approx(0.703, rel=0.01)
    assert by_wavelength[4216.7].absorption_oscillator_strength == pytest.approx(0.341, rel=0.01)


def test_barklem_widths_attach_to_the_strontium_resonance_lines():
    database = read_barklem_neutral_hydrogen_broadening(
        DATA.barklem_neutral_h_broadening,
        read_stout_atomic_database(DATA.stout, elements=("Sr",), maximum_charge=2),
    )
    widths = {
        round(line.wavelength_vacuum_angstrom, 1): line.neutral_h_vdw_rate_coefficient_cm3_s
        for line in database.ions[("Sr", 1)].transitions
    }
    assert widths[4078.9] is not None and widths[4216.7] is not None


def test_strontium_lines_enter_the_metal_line_opacity():
    atmosphere = hydrogen_helium_continuum_atmosphere(6000.0, 8.0, 5.0, n_depth=20)
    database = read_stout_atomic_database(DATA.stout, elements=("Sr",), maximum_charge=2)
    state = metal_lte_state(atmosphere, database, {"Sr": -6.0}, log_hydrogen_abundance=5.0)
    wave = np.array([4078.86, 4100.0, 4216.71])
    opacity = metal_line_mass_absorption_coefficient(atmosphere, wave, database, state)
    photosphere = np.argmin(abs(np.log(atmosphere.rosseland_optical_depth / 0.1)))
    assert opacity[0, photosphere] > 1e3 * opacity[1, photosphere]
    assert opacity[2, photosphere] > 1e3 * opacity[1, photosphere]


def test_nist_strong_line_files_cover_every_screened_ion_in_both_windows():
    registered = {
        (element, charge, filename.endswith("-uv.tsv"))
        for filename, (_, _, element, charge) in NIST_ASD_STRONG_ION_FILES.items()
    }
    expected = {
        (element, charge, uv)
        for element in SCREENED_ELEMENTS for charge in (0, 1) for uv in (False, True)
    }
    assert expected <= registered
    for filename, (_, checksum, _, _) in NIST_ASD_STRONG_ION_FILES.items():
        path = DATA.nist_asd_strong / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum, filename


def test_optical_nist_files_correct_matched_stout_strengths():
    base = read_stout_atomic_database(
        DATA.stout, elements=("Ni",), maximum_charge=2, include_default_supplements=False
    )
    paths = [
        DATA.nist_asd_strong / filename
        for filename, (_, _, element, _) in NIST_ASD_STRONG_ION_FILES.items()
        if element == "Ni"
    ]
    corrected = read_nist_asd_strong_atomic_database(paths, base)

    def strength(database, vacuum):
        return max(
            line.absorption_oscillator_strength
            for line in database.ions[("Ni", 0)].transitions
            if abs(line.wavelength_vacuum_angstrom - vacuum) < 0.03
        )

    # Ni I 3472.54 A (air; 3D2 - 3d9 4p J=3): Stout f = 2.8e-3, NIST (C+) 0.030.
    assert strength(base, 3473.537) == pytest.approx(2.81e-3, rel=0.01)
    assert strength(corrected, 3473.537) == pytest.approx(0.0304, rel=0.02)
