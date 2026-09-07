"""Molecular and collision-induced continuum opacity in pure helium.

The numerical tables are a Python transcription of the Stancil (1994)
tables distributed in Korg.jl's ``ContinuumAbsorption/Stancil1994.jl``.
Korg.jl is BSD-3-Clause licensed; its retained notice is included in
``THIRD_PARTY_NOTICES.md``.  Wavelength interpolation is linear, matching
that implementation.  The analytic dense-He infrared opacity is from
Kowalski (2014, A&A 566, L8).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]

_AMAGAT_NUMBER_DENSITY = 2.68678e19
_HELIUM_THREE_BODY_BETA = 1.56e-19
_HELIUM_THREE_BODY_BREAK_WAVENUMBER = 4000.0

_TEMPERATURE = np.asarray(
    [4200.0, 6300.0, 8400.0, 12600.0, 16800.0, 25200.0, 33600.0, 50400.0]
)
_EQUILIBRIUM_CONSTANT = 1.0e21 * np.asarray(
    [0.3606, 2.6330, 7.1409, 21.097, 39.842, 87.960, 147.41, 293.57]
)
_WAVELENGTH_FREE_FREE = 10.0 * np.asarray(
    [70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
     210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 350, 400, 450, 500,
     600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 11000, 15000, 20000]
)
_WAVELENGTH_BOUND_FREE = 10.0 * np.asarray(
    [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180,
     190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 350, 400,
     450, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4500, 5000, 10000,
     15000, 20000]
)

# The free-free coefficient has units cm^5 and multiplies n(He I)n(He II).
_FREE_FREE = 1.0e-39 * np.asarray(
    [
        [7.08e-3,6.05e-3,5.45e-3,4.78e-3,4.41e-3,3.99e-3,3.75e-3,3.50e-3],
        [.0103,8.76e-3,7.84e-3,6.79e-3,6.20e-3,5.52e-3,5.15e-3,4.74e-3],
        [.0134,.0113,.0101,8.66e-3,7.87e-3,6.97e-3,6.48e-3,5.91e-3],
        [.0165,.0139,.0124,.0106,9.65e-3,8.52e-3,7.89e-3,7.19e-3],
        [.0196,.0165,.0147,.0126,.0114,.0101,9.30e-3,8.47e-3],
        [.0226,.0190,.0169,.0145,.0131,.0116,.0107,9.72e-3],
        [.0259,.0217,.0193,.0166,.0150,.0132,.0122,.0111],
        [.0288,.0242,.0215,.0185,.0167,.0147,.0136,.0124],
        [.0317,.0267,.0237,.0204,.0185,.0163,.0151,.0137],
        [.0347,.0292,.0260,.0223,.0203,.0179,.0165,.0151],
        [.0377,.0317,.0283,.0243,.0220,.0195,.0180,.0164],
        [.0405,.0341,.0304,.0262,.0237,.0210,.0195,.0178],
        [.0433,.0365,.0326,.0280,.0254,.0225,.0209,.0191],
        [.0460,.0388,.0346,.0298,.0271,.0240,.0223,.0204],
        [.0487,.0411,.0367,.0316,.0288,.0255,.0237,.0217],
        [.0514,.0434,.0387,.0335,.0304,.0270,.0251,.0230],
        [.0540,.0457,.0408,.0352,.0321,.0285,.0266,.0244],
        [.0567,.0479,.0429,.0371,.0338,.0301,.0280,.0257],
        [.0593,.0502,.0449,.0389,.0354,.0316,.0294,.0271],
        [.0619,.0524,.0469,.0407,.0371,.0331,.0308,.0284],
        [.0645,.0546,.0489,.0424,.0387,.0346,.0323,.0297],
        [.0669,.0567,.0508,.0441,.0403,.0360,.0336,.0310],
        [.0693,.0587,.0527,.0457,.0418,.0374,.0350,.0323],
        [.0716,.0607,.0545,.0474,.0433,.0388,.0363,.0335],
        [.0825,.0702,.0632,.0551,.0506,.0455,.0427,.0396],
        [.0934,.0798,.0719,.0630,.0580,.0524,.0494,.0460],
        [.1036,.0887,.0802,.0706,.0651,.0591,.0558,.0522],
        [.1122,.0964,.0873,.0771,.0714,.0650,.0615,.0577],
        [.1280,.1106,.1006,.0894,.0831,.0762,.0724,.0683],
        [.1446,.1255,.1146,.1024,.0956,.0882,.0841,.0797],
        [.1573,.1371,.1257,.1129,.1058,.0980,.0938,.0892],
        [.1673,.1465,.1347,.1215,.1143,.1063,.1020,.0973],
        [.1787,.1570,.1448,.1312,.1237,.1155,.1111,.1064],
        [.2615,.2361,.2221,.2067,.1984,.1894,.1847,.1796],
        [.3078,.2825,.2687,.2537,.2456,.2371,.2325,.2278],
        [.3556,.3301,.3162,.3014,.2934,.2850,.2806,.2760],
        [.4175,.3909,.3766,.3612,.3531,.3445,.3400,.3353],
        [.5835,.5607,.5487,.5360,.5294,.5225,.5190,.5153],
        [.6620,.6406,.6293,.6176,.6114,.6051,.6018,.5985],
        [.7563,.7359,.7252,.7140,.7083,.7023,.6992,.6961],
    ], dtype=np.float64,
)

# Bound-free cross section per He2+ molecular ion in cm^2.
_BOUND_FREE = 1.0e-18 * np.asarray(
    [
        [5.99e-6,1.15e-5,1.50e-5,1.86e-5,2.62e-5,2.18e-5,2.25e-5,2.31e-5],
        [.0431,8.49e-3,.0112,.0140,.0153,.0165,.0171,.0176],
        [.0523,.0816,.0973,.1108,.1166,.1213,.1232,.1249],
        [.2503,.2883,.2956,.2927,.2876,.2809,.2770,.2729],
        [.6587,.6169,.5716,.5169,.4869,.4584,.4446,.4313],
        [1.2197,1.0054,.8757,.7493,.6879,.6333,.6081,.5842],
        [1.7949,1.3801,1.1630,.9667,.8758,.7973,.7617,.7285],
        [2.2739,1.6955,1.4083,1.1557,1.0409,.9426,.8984,.8573],
        [2.5948,1.9282,1.5999,1.3121,1.1816,1.0701,1.0200,.9734],
        [2.7699,2.0876,1.7438,1.4385,1.2986,1.1784,1.1241,1.0735],
        [2.8293,2.1868,1.8478,1.5389,1.3950,1.2699,1.2129,1.1596],
        [2.7978,2.2354,1.9185,1.6198,1.4775,1.3523,1.2948,1.2407],
        [2.7012,2.2420,1.9582,1.6787,1.5418,1.4194,1.3625,1.3087],
        [2.5652,2.2185,1.9747,1.7208,1.5924,1.4754,1.4203,1.3678],
        [2.4108,2.1766,1.9764,1.7532,1.6360,1.5269,1.4750,1.4250],
        [2.2515,2.1226,1.9663,1.7752,1.6702,1.5701,1.5217,1.4748],
        [2.0956,2.0615,1.9468,1.7873,1.6944,1.6034,1.5586,1.5146],
        [1.9478,1.9976,1.9221,1.7934,1.7127,1.6310,1.5899,1.5492],
        [1.8099,1.9334,1.8946,1.7958,1.7275,1.6553,1.6182,1.5809],
        [1.6822,1.8694,1.8646,1.7945,1.7382,1.6756,1.6425,1.6087],
        [1.5643,1.8060,1.8322,1.7894,1.7449,1.6919,1.6628,1.6326],
        [1.4557,1.7438,1.7985,1.7817,1.7488,1.7054,1.6805,1.6541],
        [1.3561,1.6834,1.7641,1.7720,1.7506,1.7168,1.6962,1.6736],
        [1.2648,1.6249,1.7291,1.7601,1.7497,1.7253,1.7088,1.6901],
        [1.1815,1.5685,1.6935,1.7458,1.7457,1.7302,1.7177,1.7026],
        [1.1056,1.5144,1.6579,1.7297,1.7391,1.7320,1.7234,1.7119],
        [.8192,1.2864,1.4950,1.6420,1.6905,1.7188,1.7266,1.7304],
        [.6390,1.1210,1.3675,1.5665,1.6448,1.7017,1.7233,1.7406],
        [.5156,.9900,1.2562,1.4893,1.5890,1.6678,1.7003,1.7283],
        [.4267,.8818,1.1552,1.4080,1.5221,1.6161,1.6566,1.6927],
        [.3145,.7275,1.0010,1.2736,1.4051,1.5195,1.5711,1.6187],
        [.2506,.6309,.9016,1.1870,1.3314,1.4619,1.5226,1.5799],
        [.2079,.5578,.8202,1.1078,1.2583,1.3977,1.4638,1.5271],
        [.1772,.4991,.7498,1.0327,1.1842,1.3269,1.3955,1.4618],
        [.1551,.4541,.6941,.9710,1.1219,1.2660,1.3360,1.4041],
        [.0716,.2458,.4036,.6028,.7194,.8366,.8958,.9549],
        [.0454,.1634,.2743,.4207,.5041,.5919,.6368,.6820],
        [.0344,.1268,.2151,.3334,.4015,.4738,.5110,.5486],
        [.0290,.1088,.1861,.2908,.3516,.4165,.4501,.4841],
        [.0134,.0517,.0900,.1429,.1742,.2080,.2257,.2437],
        [.0104,.0408,.0711,.1134,.1384,.1656,.1798,.1943],
        [8.88e-3,.0349,.0611,.0976,.1194,.1430,.1554,.1681],
    ], dtype=np.float64,
)


def _linear_interpolate_table(
    wavelength_angstrom: FloatArray,
    temperature: FloatArray,
    wavelength_grid: FloatArray,
    table: FloatArray,
) -> FloatArray:
    """Bilinearly interpolate a wavelength-by-temperature Stancil table."""

    wavelength, temperature = np.broadcast_arrays(
        wavelength_angstrom, temperature
    )
    flat_wavelength = wavelength.ravel()
    flat_temperature = temperature.ravel()
    w_hi = np.searchsorted(wavelength_grid, flat_wavelength, side="right")
    w_hi = np.clip(w_hi, 1, wavelength_grid.size - 1)
    w_lo = w_hi - 1
    t_hi = np.searchsorted(_TEMPERATURE, flat_temperature, side="right")
    t_hi = np.clip(t_hi, 1, _TEMPERATURE.size - 1)
    t_lo = t_hi - 1
    wf = (
        (flat_wavelength - wavelength_grid[w_lo])
        / (wavelength_grid[w_hi] - wavelength_grid[w_lo])
    )
    tf = (
        (flat_temperature - _TEMPERATURE[t_lo])
        / (_TEMPERATURE[t_hi] - _TEMPERATURE[t_lo])
    )
    values = (
        (1.0 - wf) * (1.0 - tf) * table[w_lo, t_lo]
        + wf * (1.0 - tf) * table[w_hi, t_lo]
        + (1.0 - wf) * tf * table[w_lo, t_hi]
        + wf * tf * table[w_hi, t_hi]
    )
    return values.reshape(wavelength.shape)


def helium_dimer_ion_continuum_coefficient(
    wavelength_angstrom: ArrayLike,
    temperature: ArrayLike,
) -> FloatArray:
    """Return ``(sigma_bf/K + sigma_ff)`` for He + He+ absorption.

    The returned coefficient has units cm^5.  Multiply it by the neutral-He
    and He+ number densities and by the stimulated-emission factor to obtain
    the linear absorption coefficient in cm^-1.  The original tables cover
    4200--50400 K, 500--200000 A (bound-free), and 700--200000 A
    (free-free).  Contributions outside those wavelength ranges are zero;
    temperature is linearly extrapolated above the table as in Korg/Stancil.
    Below 4200 K it is held at the boundary: the published equilibrium-
    constant extrapolation crosses zero near 3860 K and would otherwise create
    a non-physical opacity singularity in cool surface layers.
    """

    wavelength, temperature = np.broadcast_arrays(
        np.asarray(wavelength_angstrom, dtype=np.float64),
        np.asarray(temperature, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(~np.isfinite(temperature))
        or np.any(temperature <= 0.0)
    ):
        raise ValueError("wavelength and temperature must be finite and positive")
    interpolation_temperature = np.maximum(temperature, _TEMPERATURE[0])
    equilibrium = np.interp(
        interpolation_temperature,
        _TEMPERATURE,
        _EQUILIBRIUM_CONSTANT,
    )
    # np.interp clamps, while the source prescription linearly extrapolates.
    above = interpolation_temperature > _TEMPERATURE[-1]
    equilibrium = np.where(
        above,
        _EQUILIBRIUM_CONSTANT[-1]
        + (interpolation_temperature - _TEMPERATURE[-1])
        * (_EQUILIBRIUM_CONSTANT[-1] - _EQUILIBRIUM_CONSTANT[-2])
        / (_TEMPERATURE[-1] - _TEMPERATURE[-2]),
        equilibrium,
    )
    equilibrium = np.maximum(equilibrium, np.finfo(np.float64).tiny)
    bound_free = _linear_interpolate_table(
        wavelength,
        interpolation_temperature,
        _WAVELENGTH_BOUND_FREE,
        _BOUND_FREE,
    )
    free_free = _linear_interpolate_table(
        wavelength,
        interpolation_temperature,
        _WAVELENGTH_FREE_FREE,
        _FREE_FREE,
    )
    bound_free = np.where(
        (wavelength >= _WAVELENGTH_BOUND_FREE[0])
        & (wavelength <= _WAVELENGTH_BOUND_FREE[-1]),
        bound_free,
        0.0,
    )
    free_free = np.where(
        (wavelength >= _WAVELENGTH_FREE_FREE[0])
        & (wavelength <= _WAVELENGTH_FREE_FREE[-1]),
        free_free,
        0.0,
    )
    return np.maximum(bound_free / equilibrium + free_free, 0.0)


def helium_three_body_cia_linear_absorption_coefficient(
    wavelength_angstrom: ArrayLike,
    temperature: ArrayLike,
    neutral_helium_density: ArrayLike,
) -> FloatArray:
    """Return dense-He three-body CIA absorption in cm^-1.

    This implements equations 3--5 of Kowalski (2014).  The published
    absorptivity per density cubed is in cm^-1 amagat^-3, so the result scales
    as ``n(He I)^3``.  The fit was calculated for 1000--10000 K; temperatures
    outside that interval are held at the nearest boundary to avoid an
    unphysical temperature extrapolation in optically thick layers. The
    published high-frequency exponential branch is continued smoothly beyond
    6000 cm^-1, rather than imposing an artificial absorption edge at 1.67
    micron. This declining tail is an extrapolation of the published fit,
    not an independently validated optical CIA calculation. No new scale or
    fitted taper is introduced; the slope is negative throughout the fitted
    temperature interval.

    The coefficient is already a true linear absorptivity.  No additional
    stimulated-emission factor should be applied.
    """

    wavelength, temperature, neutral_density = np.broadcast_arrays(
        np.asarray(wavelength_angstrom, dtype=np.float64),
        np.asarray(temperature, dtype=np.float64),
        np.asarray(neutral_helium_density, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(~np.isfinite(temperature))
        or np.any(temperature <= 0.0)
        or np.any(~np.isfinite(neutral_density))
        or np.any(neutral_density < 0.0)
    ):
        raise ValueError(
            "wavelength and temperature must be finite and positive, and "
            "neutral_helium_density must be finite and non-negative"
        )

    fitted_temperature = np.clip(temperature, 1000.0, 10_000.0)
    wavenumber = 1.0e8 / wavelength
    gamma = (
        -0.0601248 + 1.55103e-6 * fitted_temperature
    ) * fitted_temperature**-0.393053
    omega_0 = _HELIUM_THREE_BODY_BREAK_WAVENUMBER
    # Only evaluate the power law through the join. The high-frequency
    # branch is exponential, so even very short wavelengths cannot overflow
    # an unused wavenumber**2.5 expression before the exponential underflows.
    low_wavenumber = np.minimum(wavenumber, omega_0)
    per_amagat_cubed = (
        _HELIUM_THREE_BODY_BETA
        * low_wavenumber**2.5
        * np.exp(gamma * low_wavenumber)
        * np.exp((gamma + 6.25e-4) * np.maximum(wavenumber - omega_0, 0.0))
    )
    absorption = per_amagat_cubed * (
        neutral_density / _AMAGAT_NUMBER_DENSITY
    ) ** 3
    return np.maximum(absorption, 0.0)
