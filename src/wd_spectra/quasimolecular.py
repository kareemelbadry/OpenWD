"""Readers for the bundled unified Allard hydrogen Lyman profiles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]

_INTEGRATED_CROSS_SECTION_WAVELENGTH = 8.8528e-29

ALLARD_TLUSTY205_FILES = {
    (1, 2): "laquasi.dat",
    (1, 3): "lbquasi.dat",
    (1, 4): "lgquasi.dat",
}
ALLARD_TLUSTY205_SHA256 = {
    (1, 2): "8aee190c56d18368395462eb50d406b514373d24bef6d2e2dee37b7c1c5efa81",
    (1, 3): "9407aabd4893e11ca25fdac26dafa9f5d92f7a5c2a86f0f053332068aa21f622",
    (1, 4): "9f2d397dc65c46d5cf67c0284280797ca70003030474bf33b3606ba4e1c62334",
}
# Temperature-dependent profiles supplied by N. F. Allard.  These are kept as
# external scientific data rather than bundled package assets.  The paths and
# temperatures reproduce the author's public table index; the variable-dipole
# GAMVAR calculation is the current Ly-gamma entry on that index.
ALLARD_TEMPERATURE_GRID_FILES = {
    (1, 2): {
        9_000.0: "Lyman_alpha/fort_ALPHA_abs9000.3",
        10_000.0: "Lyman_alpha/fort_ALPHA_abs10000.3",
        11_000.0: "Lyman_alpha/fort_ALPHA_abs11000.3",
        12_000.0: "Lyman_alpha/fort_ALPHA_abs12000.3",
        13_000.0: "Lyman_alpha/fort_ALPHA_abs13000.3",
        25_000.0: "Lyman_alpha/fort_ALPHA_abs25000.3",
    },
    (1, 3): {
        12_000.0: "Lyman_beta/fort_BETA_abs12000.18",
        25_000.0: "Lyman_beta/fort_BETA_abs25000.18",
        40_000.0: "Lyman_beta/fort.18_BETA_abs40",
        50_000.0: "Lyman_beta/fort.18_BETA_abs50",
        60_000.0: "Lyman_beta/fort.18_BETA_abs60",
    },
    (1, 4): {
        25_000.0: "Lyman_gamma/fort.23_GAMVAR_abs25",
    },
}
ALLARD_TEMPERATURE_GRID_SHA256 = {
    (1, 2): {
        9_000.0: "dfa29920671f3a334c3e1da9dc91ead02bda574065ce811f893181177f0fda1f",
        10_000.0: "a23c70475a183c84bc69efbe0c5e3a0918c48fbff0eadd450c5c2991d64a1ace",
        11_000.0: "205df4cfc33042153671c77c1277e2ec530bbe8087e499971e1abc6bc2c33853",
        12_000.0: "d816a518ae918dc1bb1decb4b5d8efab379aba3a6bc1e8dd2f5ac65291967373",
        13_000.0: "292c39aa57b6afb5379c21f816c6d47149304973de69ecd746c189cf6c328286",
        25_000.0: "f31df9f9f181c66d1b45d3571bc4a7d977b95d77fc2e4fa48682dd638b3b1365",
    },
    (1, 3): {
        12_000.0: "ff781d8d5b0d7e49a1461144f4c7d0a1ec8276abf0bcef7c59a43d3e6138035f",
        25_000.0: "9407aabd4893e11ca25fdac26dafa9f5d92f7a5c2a86f0f053332068aa21f622",
        40_000.0: "c984a55762a570e5987b65a1efce6b40638548ca02d685640270877016e15b82",
        50_000.0: "46e2cc03bea6281dc1be73d2cb263d65bc3b2fabd86ab51207283551b93fa7be",
        60_000.0: "693adcd18b9c8b38c5f5063079784936b51339b203b3beede173186c6fd546d8",
    },
    (1, 4): {
        25_000.0: "7554c9c48ed99297854ea6567fae7353066be2ddf5129372a7fe0f82677e02c9",
    },
}
# These values deliberately reproduce the XNORMA/B/G constants in TLUSTY205
# and SYNSPEC51 rather than silently substituting newer atomic data.
_ALLARD_TLUSTY205_LINE_NORMALIZATION = {
    (1, 2): _INTEGRATED_CROSS_SECTION_WAVELENGTH * 1215.6**2 * 0.41618,
    (1, 3): _INTEGRATED_CROSS_SECTION_WAVELENGTH * 1025.73 * 1025.7 * 0.0791,
    (1, 4): _INTEGRATED_CROSS_SECTION_WAVELENGTH * 972.53**2 * 0.0290,
}

@dataclass(frozen=True)
class AllardNeutralLymanAlphaProfile:
    """One SYNSPEC-format unified H--H/H--H+ density-expansion table.

    Each wavelength row contains the first- and second-order neutral-H and
    proton coefficients plus their mixed term.  The header normalization and
    excluded-volume factors follow ``GETLAL``/``ALLARD`` in SYNSPEC 49.

    The historical class name is retained for API compatibility, but
    ``lower_level`` and ``upper_level`` allow the same exact implementation to
    represent the standard Lyalpha, Lybeta, and Lygamma files.
    """

    temperature_K: float | None
    wavelength_angstrom: FloatArray
    expansion_coefficients: FloatArray
    neutral_reference_density: float
    proton_reference_density: float
    neutral_excluded_volume: float
    proton_excluded_volume: float
    source_path: Path
    lower_level: int = 1
    upper_level: int = 2

    def __post_init__(self) -> None:
        wavelength = np.asarray(self.wavelength_angstrom, dtype=np.float64)
        coefficients = np.asarray(self.expansion_coefficients, dtype=np.float64)
        if wavelength.ndim != 1 or wavelength.size < 2:
            raise ValueError("wavelength_angstrom must contain at least two values")
        if np.any(~np.isfinite(wavelength)) or np.any(np.diff(wavelength) <= 0.0):
            raise ValueError("wavelength_angstrom must be finite and increasing")
        if coefficients.shape != (wavelength.size, 5):
            raise ValueError("expansion_coefficients must have shape (wavelength, 5)")
        if np.any(~np.isfinite(coefficients)) or np.any(coefficients < 0.0):
            raise ValueError("expansion_coefficients must be finite and non-negative")
        positive_header = np.array(
            [self.neutral_reference_density, self.proton_reference_density]
        )
        volume_header = np.array(
            [self.neutral_excluded_volume, self.proton_excluded_volume]
        )
        if np.any(~np.isfinite(positive_header)) or np.any(positive_header <= 0.0):
            raise ValueError("profile reference densities must be positive")
        if self.temperature_K is not None and (
            not np.isfinite(self.temperature_K) or self.temperature_K <= 0.0
        ):
            raise ValueError("profile temperature must be positive when supplied")
        if np.any(~np.isfinite(volume_header)) or np.any(volume_header < 0.0):
            raise ValueError("excluded-volume factors must be finite and non-negative")
        if (self.lower_level, self.upper_level) not in (
            _ALLARD_TLUSTY205_LINE_NORMALIZATION
        ):
            raise ValueError("supported Allard transitions are Lyalpha, Lybeta, and Lygamma")

    @property
    def transition(self) -> tuple[int, int]:
        return self.lower_level, self.upper_level

    def unperturbed_absorber_probability(
        self,
        neutral_h_density: ArrayLike,
        proton_density: ArrayLike,
    ) -> FloatArray:
        """Return the table's zero-perturber probability at every depth.

        The Allard density expansion separates absorbers with one or more
        close H/H+ perturbers from the unperturbed atomic component.  Its
        finite-volume factor is the probability assigned to the latter,

        ``P0 = 1 / (1 + V + V**2 / 2)``.

        A conservative hybrid therefore weights the ordinary Stark profile
        by ``P0`` before adding the tabulated perturbed-absorber profile.  If
        the full Stark profile is retained, the line oscillator strength is
        counted again by the Allard contribution.
        """

        neutral = np.atleast_1d(np.asarray(neutral_h_density, dtype=np.float64))
        proton = np.atleast_1d(np.asarray(proton_density, dtype=np.float64))
        if neutral.ndim != 1 or proton.shape != neutral.shape:
            raise ValueError("neutral_h_density and proton_density must be matching 1D arrays")
        if np.any(~np.isfinite(neutral)) or np.any(neutral < 0.0):
            raise ValueError("neutral_h_density must be finite and non-negative")
        if np.any(~np.isfinite(proton)) or np.any(proton < 0.0):
            raise ValueError("proton_density must be finite and non-negative")

        volume_sum = (
            neutral / self.neutral_reference_density * self.neutral_excluded_volume
            + proton / self.proton_reference_density * self.proton_excluded_volume
        )
        return 1.0 / (1.0 + volume_sum + 0.5 * volume_sum**2)

    def profile_cross_section(
        self,
        wavelength_angstrom: ArrayLike,
        neutral_h_density: ArrayLike,
        proton_density: ArrayLike,
        *,
        extend_lyman_alpha_red_wing: bool = False,
    ) -> FloatArray:
        """Evaluate the unified profile cross section in cm2.

        The returned array has shape ``(wavelength, depth)``.  This is the
        exact five-term density expansion and finite-volume normalization used
        by SYNSPEC, including neutral-neutral, proton-proton, and mixed terms.
        """

        wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
        neutral = np.atleast_1d(np.asarray(neutral_h_density, dtype=np.float64))
        proton = np.atleast_1d(np.asarray(proton_density, dtype=np.float64))
        if wavelength.ndim != 1 or np.any(~np.isfinite(wavelength)):
            raise ValueError("wavelength_angstrom must be a finite 1D array")
        if neutral.ndim != 1 or proton.shape != neutral.shape:
            raise ValueError("neutral_h_density and proton_density must be matching 1D arrays")
        if np.any(~np.isfinite(neutral)) or np.any(neutral < 0.0):
            raise ValueError("neutral_h_density must be finite and non-negative")
        if np.any(~np.isfinite(proton)) or np.any(proton < 0.0):
            raise ValueError("proton_density must be finite and non-negative")

        coefficients = np.zeros((wavelength.size, 5), dtype=np.float64)
        inside = (
            (wavelength >= self.wavelength_angstrom[0])
            & (wavelength <= self.wavelength_angstrom[-1])
        )
        for column in range(5):
            coefficients[inside, column] = np.interp(
                wavelength[inside],
                self.wavelength_angstrom,
                self.expansion_coefficients[:, column],
            )
        # GETLAL/ALLARD in SYNSPEC does not truncate Ly-alpha at the last
        # tabulated wavelength.  It continues every density-expansion term
        # with the static-wing |delta lambda|^-5/2 asymptote, normalized to
        # the final row.  Ly-beta and Ly-gamma are explicitly zero outside
        # their tabulated intervals.  Keep this opt-in at the low-level API
        # so callers can still inspect the literal finite file if desired.
        if extend_lyman_alpha_red_wing and self.transition == (1, 2):
            red = wavelength > self.wavelength_angstrom[-1]
            if np.any(red):
                line_center = 1215.67
                edge_detuning = self.wavelength_angstrom[-1] - line_center
                detuning = wavelength[red] - line_center
                coefficients[red, :] = self.expansion_coefficients[-1, :] * (
                    detuning[:, np.newaxis] / edge_detuning
                ) ** -2.5
        neutral_scaled = neutral / self.neutral_reference_density
        proton_scaled = proton / self.proton_reference_density
        normalization = self.unperturbed_absorber_probability(neutral, proton)
        density_terms = np.stack(
            (
                neutral_scaled,
                neutral_scaled**2,
                proton_scaled,
                proton_scaled**2,
                neutral_scaled * proton_scaled,
            ),
            axis=0,
        )
        line_normalization = _ALLARD_TLUSTY205_LINE_NORMALIZATION[self.transition]
        return (
            coefficients @ density_terms
            * normalization[np.newaxis, :]
            * line_normalization
        )


@dataclass(frozen=True)
class AllardNeutralLymanAlphaTable:
    """Temperature grid for one SYNSPEC-format unified Lyman line."""

    profiles: tuple[AllardNeutralLymanAlphaProfile, ...]

    def __post_init__(self) -> None:
        if not self.profiles:
            raise ValueError("profiles must not be empty")
        transitions = {profile.transition for profile in self.profiles}
        if len(transitions) != 1:
            raise ValueError("all profiles in a temperature grid must be for one line")
        if len(self.profiles) > 1:
            temperature = self.temperatures_K
            if np.any(np.diff(temperature) <= 0.0):
                raise ValueError("profile temperatures must be strictly increasing")

    @property
    def transition(self) -> tuple[int, int]:
        return self.profiles[0].transition

    @property
    def temperatures_K(self) -> FloatArray:
        if any(profile.temperature_K is None for profile in self.profiles):
            raise ValueError("fixed-temperature Allard profiles have no temperature grid")
        return np.array(
            [profile.temperature_K for profile in self.profiles], dtype=np.float64
        )

    def unperturbed_absorber_probability(
        self,
        temperature_K: ArrayLike,
        neutral_h_density: ArrayLike,
        proton_density: ArrayLike,
    ) -> FloatArray:
        """Interpolate the table's zero-perturber probability by depth.

        The excluded-volume factors in the native header vary between the
        supplied temperature tables.  Interpolating only the tabulated cross
        section while using one profile's finite-volume factor for the Stark
        share would therefore be inconsistent.  This uses the same log(T)
        bracket and linear interpolation as :meth:`profile_cross_section`.
        """

        temperature = np.atleast_1d(np.asarray(temperature_K, dtype=np.float64))
        neutral = np.atleast_1d(np.asarray(neutral_h_density, dtype=np.float64))
        proton = np.atleast_1d(np.asarray(proton_density, dtype=np.float64))
        if neutral.shape != temperature.shape or proton.shape != temperature.shape:
            raise ValueError("temperature and density arrays must have matching shapes")
        if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
            raise ValueError("temperature_K must be finite and positive")

        # The profile-consistent atmosphere path deliberately selects one
        # native Allard file for the whole atmosphere.  Evaluate that file
        # over all depths at once instead of repeating the same scalar call
        # for every layer.  Besides being faster, this branch is algebraically
        # identical to the loop below: the selected temperature comes
        # directly from ``temperatures_K``, so no interpolation is skipped.
        if len(self.profiles) == 1:
            return self.profiles[0].unperturbed_absorber_probability(
                neutral, proton
            )
        grid = self.temperatures_K
        if np.all(temperature == temperature[0]):
            exact = np.flatnonzero(grid == temperature[0])
            if exact.size:
                return self.profiles[int(exact[0])].unperturbed_absorber_probability(
                    neutral, proton
                )

        result = np.empty_like(temperature)
        for depth, local_temperature in enumerate(temperature):
            lower, fraction = _bracket(np.log(grid), np.log(local_temperature))
            low = self.profiles[lower].unperturbed_absorber_probability(
                neutral[depth : depth + 1], proton[depth : depth + 1]
            )[0]
            high = self.profiles[lower + 1].unperturbed_absorber_probability(
                neutral[depth : depth + 1], proton[depth : depth + 1]
            )[0]
            result[depth] = (1.0 - fraction) * low + fraction * high
        return result

    def profile_cross_section(
        self,
        wavelength_angstrom: ArrayLike,
        temperature_K: ArrayLike,
        neutral_h_density: ArrayLike,
        proton_density: ArrayLike,
        *,
        extend_lyman_alpha_red_wing: bool = False,
    ) -> FloatArray:
        """Interpolate profile cross sections independently at every depth."""

        temperature = np.atleast_1d(np.asarray(temperature_K, dtype=np.float64))
        neutral = np.atleast_1d(np.asarray(neutral_h_density, dtype=np.float64))
        proton = np.atleast_1d(np.asarray(proton_density, dtype=np.float64))
        if neutral.shape != temperature.shape or proton.shape != temperature.shape:
            raise ValueError("temperature and density arrays must have matching shapes")
        if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
            raise ValueError("temperature_K must be finite and positive")

        wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)

        # The normal atmosphere path chooses one native profile globally.
        # Its density expansion already broadcasts over depth, so retain that
        # vectorization here rather than recomputing wavelength interpolation
        # and the same five coefficients once per layer.
        if len(self.profiles) == 1:
            return self.profiles[0].profile_cross_section(
                wavelength,
                neutral,
                proton,
                extend_lyman_alpha_red_wing=extend_lyman_alpha_red_wing,
            )
        grid = self.temperatures_K
        if np.all(temperature == temperature[0]):
            exact = np.flatnonzero(grid == temperature[0])
            if exact.size:
                return self.profiles[int(exact[0])].profile_cross_section(
                    wavelength,
                    neutral,
                    proton,
                    extend_lyman_alpha_red_wing=extend_lyman_alpha_red_wing,
                )

        result = np.empty((wavelength.size, temperature.size), dtype=np.float64)
        for depth, local_temperature in enumerate(temperature):
            lower, fraction = _bracket(np.log(grid), np.log(local_temperature))
            low = self.profiles[lower].profile_cross_section(
                wavelength,
                neutral[depth : depth + 1],
                proton[depth : depth + 1],
                extend_lyman_alpha_red_wing=extend_lyman_alpha_red_wing,
            )[:, 0]
            high = self.profiles[lower + 1].profile_cross_section(
                wavelength,
                neutral[depth : depth + 1],
                proton[depth : depth + 1],
                extend_lyman_alpha_red_wing=extend_lyman_alpha_red_wing,
            )[:, 0]
            result[:, depth] = (1.0 - fraction) * low + fraction * high
        return result


@dataclass(frozen=True)
class AllardUnifiedLymanTable:
    """Standard unified Lyalpha--Lygamma tables used by TLUSTY205."""

    lines: dict[tuple[int, int], AllardNeutralLymanAlphaTable]
    source_directory: Path
    source_sha256: dict[tuple[int, int], str]

    def __post_init__(self) -> None:
        if not self.lines:
            raise ValueError("lines must not be empty")
        for transition, table in self.lines.items():
            if transition != table.transition:
                raise ValueError("Allard line-table key does not match its transition")

    def __getitem__(
        self, transition: tuple[int, int]
    ) -> AllardNeutralLymanAlphaTable:
        return self.lines[transition]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_allard_hydrogen_line_profile(
    path: str | Path,
    *,
    lower_level: int,
    upper_level: int,
    temperature_K: float | None = None,
) -> AllardNeutralLymanAlphaProfile:
    """Read one fixed- or specified-temperature Allard/Koester table."""

    transition = (int(lower_level), int(upper_level))
    if transition not in _ALLARD_TLUSTY205_LINE_NORMALIZATION:
        raise ValueError("supported Allard transitions are Lyalpha, Lybeta, and Lygamma")
    return _read_allard_hydrogen_line_profile(
        path,
        lower_level=transition[0],
        upper_level=transition[1],
        temperature_K=temperature_K,
    )


def read_allard_neutral_lyman_alpha_profile(
    path: str | Path,
    temperature_K: float,
) -> AllardNeutralLymanAlphaProfile:
    """Read one Allard/Koester unified Lyalpha table in SYNSPEC format."""

    return read_allard_hydrogen_line_profile(
        path,
        lower_level=1,
        upper_level=2,
        temperature_K=temperature_K,
    )


def _read_allard_hydrogen_line_profile(
    path: str | Path,
    *,
    lower_level: int,
    upper_level: int,
    temperature_K: float | None,
    expected_row_count_if_header_mismatches: int | None = None,
) -> AllardNeutralLymanAlphaProfile:
    """Implementation shared by the public line-specific readers."""

    path = Path(path)
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as stream:
        header_tokens = stream.readline().replace("D", "E").replace("d", "e").split()
        if len(header_tokens) != 5:
            raise ValueError("Allard table header must contain five values")
        count = int(header_tokens[0])
        neutral_reference_density = 10.0 ** float(header_tokens[1])
        proton_reference_density = 10.0 ** float(header_tokens[2])
        neutral_excluded_volume = float(header_tokens[3])
        proton_excluded_volume = float(header_tokens[4])
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "!")):
                continue
            values = [
                float(token.replace("D", "E").replace("d", "e"))
                for token in stripped.split()
            ]
            if len(values) != 6:
                raise ValueError("each Allard table row must contain six values")
            rows.append(values)
    if len(rows) != count and (
        expected_row_count_if_header_mismatches is None
        or len(rows) != expected_row_count_if_header_mismatches
    ):
        raise ValueError(f"Allard table declares {count} rows but contains {len(rows)}")
    values = np.asarray(rows, dtype=np.float64)
    order = np.argsort(values[:, 0])
    values = values[order]
    return AllardNeutralLymanAlphaProfile(
        temperature_K=(None if temperature_K is None else float(temperature_K)),
        wavelength_angstrom=values[:, 0],
        expansion_coefficients=values[:, 1:],
        neutral_reference_density=neutral_reference_density,
        proton_reference_density=proton_reference_density,
        neutral_excluded_volume=neutral_excluded_volume,
        proton_excluded_volume=proton_excluded_volume,
        source_path=path,
        lower_level=lower_level,
        upper_level=upper_level,
    )


def read_allard_neutral_lyman_alpha_table(
    paths_by_temperature: dict[float, str | Path],
) -> AllardNeutralLymanAlphaTable:
    """Read and order a temperature grid of external unified Lyalpha tables."""

    profiles = tuple(
        read_allard_neutral_lyman_alpha_profile(path, temperature)
        for temperature, path in sorted(paths_by_temperature.items())
    )
    return AllardNeutralLymanAlphaTable(profiles=profiles)


def read_allard_tlusty205_tables(
    directory: str | Path,
    *,
    verify_checksums: bool = True,
) -> AllardUnifiedLymanTable:
    """Read the fixed-temperature Lyalpha--Lygamma tables from TLUSTY205.

    The checksums identify the public TLUSTY205 files exactly.  Disable
    verification only to test a compatible locally supplied table set.
    """

    directory = Path(directory)
    lines: dict[tuple[int, int], AllardNeutralLymanAlphaTable] = {}
    source_sha256: dict[tuple[int, int], str] = {}
    for transition, filename in ALLARD_TLUSTY205_FILES.items():
        path = directory / filename
        actual_sha256 = _sha256(path)
        expected_sha256 = ALLARD_TLUSTY205_SHA256[transition]
        if verify_checksums and actual_sha256 != expected_sha256:
            raise ValueError(
                f"checksum mismatch for {path}: expected {expected_sha256}, "
                f"found {actual_sha256}"
            )
        profile = read_allard_hydrogen_line_profile(
            path,
            lower_level=transition[0],
            upper_level=transition[1],
        )
        lines[transition] = AllardNeutralLymanAlphaTable((profile,))
        source_sha256[transition] = actual_sha256
    return AllardUnifiedLymanTable(
        lines=lines,
        source_directory=directory,
        source_sha256=source_sha256,
    )


def read_allard_temperature_dependent_tables(
    directory: str | Path,
    *,
    verify_checksums: bool = True,
) -> AllardUnifiedLymanTable:
    """Read Nicole Allard's temperature-dependent Lyman profile delivery.

    ``directory`` must contain the ``Lyman_alpha``, ``Lyman_beta``, and
    ``Lyman_gamma`` subdirectories described by
    :data:`ALLARD_TEMPERATURE_GRID_FILES`.  The current Ly-gamma manifest uses
    the supplied variable-dipole ``GAMVAR`` profile rather than the older
    constant-dipole TLUSTY205 file that may accompany the delivery.
    """

    directory = Path(directory)
    lines: dict[tuple[int, int], AllardNeutralLymanAlphaTable] = {}
    source_sha256: dict[tuple[int, int], str] = {}
    for transition, files_by_temperature in ALLARD_TEMPERATURE_GRID_FILES.items():
        profiles: list[AllardNeutralLymanAlphaProfile] = []
        hashes: list[str] = []
        for temperature, relative_path in sorted(files_by_temperature.items()):
            path = directory / relative_path
            actual_sha256 = _sha256(path)
            expected_sha256 = ALLARD_TEMPERATURE_GRID_SHA256[transition][temperature]
            if verify_checksums and actual_sha256 != expected_sha256:
                raise ValueError(
                    f"checksum mismatch for {path}: expected {expected_sha256}, "
                    f"found {actual_sha256}"
                )
            profiles.append(
                _read_allard_hydrogen_line_profile(
                    path,
                    lower_level=transition[0],
                    upper_level=transition[1],
                    temperature_K=temperature,
                    # The delivered 11,000 K Ly-alpha file declares 699 rows
                    # but contains the same complete 697-point grid as its
                    # neighbors.  Accept only that exact known row count;
                    # every other header/data mismatch remains an error.
                    expected_row_count_if_header_mismatches=(
                        697
                        if transition == (1, 2) and temperature == 11_000.0
                        else None
                    ),
                )
            )
            hashes.append(actual_sha256)
        lines[transition] = AllardNeutralLymanAlphaTable(tuple(profiles))
        source_sha256[transition] = hashlib.sha256(
            "\n".join(hashes).encode("ascii")
        ).hexdigest()
    return AllardUnifiedLymanTable(
        lines=lines,
        source_directory=directory,
        source_sha256=source_sha256,
    )


# Clearer generic names for new code; retain the historical public names above
# so existing users of the Lyalpha temperature-grid reader are unaffected.
AllardHydrogenLineProfile = AllardNeutralLymanAlphaProfile
AllardHydrogenLineTable = AllardNeutralLymanAlphaTable


def _bracket(grid: FloatArray, value: float) -> tuple[int, float]:
    clipped = float(np.clip(value, grid[0], grid[-1]))
    upper = int(np.searchsorted(grid, clipped, side="right"))
    lower = min(max(upper - 1, 0), grid.size - 2)
    fraction = (clipped - grid[lower]) / (grid[lower + 1] - grid[lower])
    return lower, float(np.clip(fraction, 0.0, 1.0))
