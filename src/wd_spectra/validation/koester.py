"""Read Koester DA spectra distributed by the SVO Theory Server."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class KoesterSpectrum:
    """A Koester DA surface-flux spectrum from SVO.

    SVO distributes ``4 pi H_lambda`` in erg s^-1 cm^-2 Angstrom^-1 and
    converts the original wavelength coordinate to air.  This reader reverses
    the SVO conversion at every wavelength so that the returned coordinate is
    directly comparable with the vacuum wavelengths used by
    ``wd_spectra.Spectrum``.  Applying that inversion in the far UV is
    essential: the SVO files otherwise place Ly-beta and the higher Lyman
    cores about 0.7--0.8 Angstrom blueward of their laboratory wavelengths.
    """

    effective_temperature: float
    logg: float
    wavelength_angstrom: FloatArray
    surface_flux_lambda: FloatArray
    source_path: Path


def _svo_air_conversion_factor(vacuum_angstrom: FloatArray) -> FloatArray:
    """Return the Morton (1991) factor used for the SVO air coordinate.

    The polynomial is evaluated here as the inverse of the transformation
    already applied to the downloaded validation spectra, including in the
    far UV.  It must not be interpreted as a physical refractive index below
    2000 Angstrom, where ordinary air is opaque.
    """

    inverse_wavelength_squared = vacuum_angstrom**-2
    return (
        1.0
        + 2.735182e-4
        + 131.4182 * inverse_wavelength_squared
        + 2.76249e8 * inverse_wavelength_squared**2
    )


def air_to_vacuum(wavelength_air_angstrom: ArrayLike) -> FloatArray:
    """Invert the SVO/Morton air-coordinate conversion.

    Five fixed-point iterations invert ``lambda_air = lambda_vacuum / n``.
    The inversion is deliberately applied to the entire SVO spectrum because
    the service applied the same algebraic conversion to its UV coordinate.
    """

    wavelength_air = np.asarray(wavelength_air_angstrom, dtype=np.float64)
    if np.any(~np.isfinite(wavelength_air)) or np.any(wavelength_air <= 0.0):
        raise ValueError("air wavelengths must contain finite positive values")

    wavelength_vacuum = wavelength_air.copy()
    for _ in range(5):
        wavelength_vacuum = (
            wavelength_air
            * _svo_air_conversion_factor(wavelength_vacuum)
        )
    return wavelength_vacuum


def physical_air_to_vacuum(wavelength_air_angstrom: ArrayLike) -> FloatArray:
    """Convert an ordinary optical air coordinate to vacuum wavelength.

    Unlike :func:`air_to_vacuum`, this routine does not reproduce SVO's
    algebraic extrapolation into the far ultraviolet.  Wavelengths below
    2000 Angstrom are retained as vacuum values, while the Morton (1991)
    refractive-index expression is inverted in the optical/near-UV.  This is
    the convention used by the legacy Montreal DB spectral grids.
    """

    wavelength_air = np.asarray(wavelength_air_angstrom, dtype=np.float64)
    if np.any(~np.isfinite(wavelength_air)) or np.any(wavelength_air <= 0.0):
        raise ValueError("air wavelengths must contain finite positive values")
    wavelength_vacuum = wavelength_air.copy()
    optical = wavelength_air >= 2000.0
    for _ in range(5):
        wavelength_vacuum[optical] = (
            wavelength_air[optical]
            * _svo_air_conversion_factor(wavelength_vacuum[optical])
        )
    return wavelength_vacuum


def read_svo_koester_ascii(path: str | Path) -> KoesterSpectrum:
    """Read an ASCII Koester spectrum downloaded from the SVO service."""

    path = Path(path)
    header_lines: list[str] = []
    with path.open("r", encoding="ascii") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            header_lines.append(line)
    header = "".join(header_lines)
    teff_match = re.search(r"^# teff\s*=\s*([0-9.]+)\s*K", header, re.MULTILINE)
    logg_match = re.search(r"^# logg\s*=\s*([0-9.]+)", header, re.MULTILINE)
    if teff_match is None or logg_match is None:
        raise ValueError(f"missing Koester teff/logg metadata in {path}")

    data = np.loadtxt(path, comments="#", dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 2 or data.shape[0] < 2:
        raise ValueError(f"expected two spectral columns in {path}")
    if np.any(np.diff(data[:, 0]) <= 0.0) or np.any(data[:, 1] < 0.0):
        raise ValueError(f"invalid wavelength or flux values in {path}")

    return KoesterSpectrum(
        effective_temperature=float(teff_match.group(1)),
        logg=float(logg_match.group(1)),
        wavelength_angstrom=air_to_vacuum(data[:, 0]),
        surface_flux_lambda=data[:, 1],
        source_path=path,
    )
