"""Neutral-helium impact broadening for optical He I lines.

The tabulation is from Deridder & van Rensbergen (1976, A&AS, 23,
147), Tables 2 and 4.  Their damping constant is the Lorentz HWHM in
angular-frequency units,

    gamma = N_He * alpha * T**beta,

where the published alpha values are in units of 1e-8.  Only the part of
the tables needed by the 36 optical He I transitions is transcribed here.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import LIGHT_SPEED, PI


FloatArray = NDArray[np.float64]
Orbital = Literal["s", "p", "d"]


# Rows are the effective quantum number of the s state; columns are that of
# the p state.  The 1.25 row/column in the source is outside the range of the
# optical He I transitions and is deliberately omitted.
_SP_EFFECTIVE_N = np.asarray(
    [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
)
_SP_ALPHA_1E8 = np.asarray(
    [
        [.0165, .0232, .0353, .0538, .0813, .1199, .1725, .2420, .3320, .4464, .5895, .7658],
        [.0372, .0249, .0491, .0671, .0941, .1323, .1844, .2536, .3434, .4575, .6003, .7765],
        [.0731, .0732, .0716, .0923, .1191, .1571, .2089, .2778, .3672, .4810, .6235, .7994],
        [.1279, .1279, .1285, .1262, .1567, .1944, .2460, .3146, .4035, .5169, .6590, .8345],
        [.2074, .2074, .2074, .2084, .2056, .2486, .2999, .3683, .4569, .5699, .7115, .8866],
        [.3182, .3182, .3182, .3182, .3198, .3166, .3752, .4432, .5318, .6445, .7858, .9603],
        [.4679, .4679, .4679, .4679, .4680, .4703, .4669, .5448, .6330, .7457, .8868, 1.0610],
        [.6655, .6655, .6655, .6655, .6655, .6656, .6686, .6655, .7666, .8791, 1.0202, 1.1944],
        [.9208, .9208, .9208, .9208, .9208, .9208, .9211, .9246, .9224, 1.0513, 1.1923, 1.3666],
        [1.2451, 1.2451, 1.2451, 1.2451, 1.2451, 1.2451, 1.2451, 1.2455, 1.2495, 1.2490, 1.4106, 1.5849],
        [1.6508, 1.6508, 1.6508, 1.6508, 1.6508, 1.6508, 1.6508, 1.6509, 1.6513, 1.6555, 1.6578, 1.8574],
        [2.1516, 2.1516, 2.1516, 2.1516, 2.1516, 2.1516, 2.1516, 2.1516, 2.1516, 2.1521, 2.1563, 2.1625],
    ],
    dtype=np.float64,
)
_SP_BETA = np.asarray(
    [
        [.3162, .3329, .3610, .3804, .3946, .4057, .4147, .4222, .4286, .4341, .4389, .4431],
        [.3347, .3340, .3550, .3741, .3893, .4016, .4117, .4200, .4269, .4328, .4379, .4424],
        [.3652, .3648, .3545, .3767, .3885, .3993, .4091, .4176, .4249, .4312, .4366, .4413],
        [.3805, .3805, .3796, .3756, .3897, .3986, .4073, .4154, .4228, .4293, .4349, .4399],
        [.3919, .3919, .3919, .3908, .3895, .3999, .4071, .4142, .4211, .4275, .4332, .4384],
        [.4016, .4016, .4016, .4015, .4003, .4004, .4086, .4145, .4204, .4262, .4318, .4369],
        [.4099, .4099, .4099, .4099, .4099, .4085, .4095, .4161, .4210, .4260, .4310, .4358],
        [.4172, .4172, .4172, .4172, .4172, .4172, .4159, .4172, .4227, .4288, .4310, .4353],
        [.4237, .4237, .4237, .4237, .4237, .4237, .4237, .4224, .4240, .4285, .4320, .4356],
        [.4295, .4295, .4295, .4295, .4295, .4295, .4295, .4294, .4282, .4298, .4337, .4367],
        [.4346, .4346, .4346, .4346, .4346, .4346, .4346, .4346, .4345, .4334, .4350, .4383],
        [.4391, .4391, .4391, .4391, .4391, .4391, .4391, .4391, .4391, .4391, .4381, .4396],
    ],
    dtype=np.float64,
)


# The optical 2p-nD transitions lie very close to n*_p=2.  Rows bracketing
# that value are retained from Table 4; columns are n*_d.
_PD_P_EFFECTIVE_N = np.asarray([1.5, 2.0, 2.5])
_PD_D_EFFECTIVE_N = np.asarray(
    [2.25, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
)
_PD_ALPHA_1E8 = np.asarray(
    [
        [.0313, .0450, .0684, .0962, .1303, .1728, .2257, .2913, .3719, .4703, .5893, .7321, .9020],
        [.0249, .0703, .0693, .0986, .1334, .1761, .2292, .2947, .3754, .4737, .5927, .7355, .9053],
        [.0562, .0523, .0511, .1012, .1381, .1820, .2356, .3015, .3823, .4807, .5998, .7425, .9124],
    ],
    dtype=np.float64,
)
_PD_BETA = np.asarray(
    [
        [.2926, .3049, .3310, .3505, .3670, .3810, .3931, .4035, .4124, .4202, .4269, .4327, .4378],
        [.3198, .2393, .3335, .3511, .3669, .3807, .3927, .4031, .4121, .4199, .4266, .4325, .4376],
        [.3409, .3423, .3773, .3571, .3695, .3817, .3930, .4030, .4119, .4196, .4263, .4322, .4374],
    ],
    dtype=np.float64,
)


def _bilinear(
    rows: FloatArray,
    columns: FloatArray,
    values: FloatArray,
    row_value: float,
    column_value: float,
) -> float:
    """Bilinearly interpolate a small rectangular source table."""

    row = float(np.clip(row_value, rows[0], rows[-1]))
    column = float(np.clip(column_value, columns[0], columns[-1]))
    along_columns = np.asarray(
        [np.interp(column, columns, source_row) for source_row in values]
    )
    return float(np.interp(row, rows, along_columns))


def deridder_van_rensbergen_helium_hwhm_angstrom(
    temperature: ArrayLike,
    neutral_helium_density: ArrayLike,
    wavelength_angstrom: float,
    lower_effective_quantum_number: float,
    upper_effective_quantum_number: float,
    lower_orbital: Orbital,
    upper_orbital: Orbital,
) -> FloatArray:
    """Return the He-perturber Lorentz HWHM from the 1976 tables.

    Temperatures outside the published 4000--100000 K interval are clipped
    to its nearest boundary.  The finite radiator-mass correction in their
    equation (4) is exactly ``2**beta`` for a He radiator perturbed by He.
    Effective quantum numbers outside the transcribed optical-line region
    are likewise held at its boundary.
    """

    thermal = np.asarray(temperature, dtype=np.float64)
    perturber = np.asarray(neutral_helium_density, dtype=np.float64)
    if (
        np.any(~np.isfinite(thermal))
        or np.any(thermal <= 0.0)
        or np.any(~np.isfinite(perturber))
        or np.any(perturber < 0.0)
        or not np.isfinite(wavelength_angstrom)
        or wavelength_angstrom <= 0.0
        or not np.isfinite(lower_effective_quantum_number)
        or lower_effective_quantum_number <= 0.0
        or not np.isfinite(upper_effective_quantum_number)
        or upper_effective_quantum_number <= 0.0
    ):
        raise ValueError("invalid state or transition for neutral-He broadening")
    try:
        thermal, perturber = np.broadcast_arrays(thermal, perturber)
    except ValueError as error:
        raise ValueError("temperature and density must be broadcast-compatible") from error

    orbitals = {lower_orbital, upper_orbital}
    if orbitals == {"s", "p"}:
        if lower_orbital == "s":
            s_effective = lower_effective_quantum_number
            p_effective = upper_effective_quantum_number
        else:
            s_effective = upper_effective_quantum_number
            p_effective = lower_effective_quantum_number
        alpha = _bilinear(
            _SP_EFFECTIVE_N,
            _SP_EFFECTIVE_N,
            _SP_ALPHA_1E8,
            s_effective,
            p_effective,
        )
        beta = _bilinear(
            _SP_EFFECTIVE_N,
            _SP_EFFECTIVE_N,
            _SP_BETA,
            s_effective,
            p_effective,
        )
    elif lower_orbital == "p" and upper_orbital == "d":
        alpha = _bilinear(
            _PD_P_EFFECTIVE_N,
            _PD_D_EFFECTIVE_N,
            _PD_ALPHA_1E8,
            lower_effective_quantum_number,
            upper_effective_quantum_number,
        )
        beta = _bilinear(
            _PD_P_EFFECTIVE_N,
            _PD_D_EFFECTIVE_N,
            _PD_BETA,
            lower_effective_quantum_number,
            upper_effective_quantum_number,
        )
    else:
        raise ValueError("only optical s-p and p-d transitions are tabulated")

    bounded_temperature = np.clip(thermal, 4_000.0, 100_000.0)
    angular_hwhm = (
        perturber
        * alpha
        * 1.0e-8
        * bounded_temperature**beta
        * 2.0**beta
    )
    wavelength_cm = wavelength_angstrom * 1.0e-8
    return wavelength_cm**2 * angular_hwhm / (2.0 * PI * LIGHT_SPEED) * 1.0e8


__all__ = ["deridder_van_rensbergen_helium_hwhm_angstrom"]
