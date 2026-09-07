import numpy as np
import pytest
from wd_spectra._compat import trapezoid
from wd_spectra.opacity import optical_depth_from_mass_opacity
from wd_spectra.radiative_transfer import coherent_scattering_feautrier_field
from wd_spectra._stable_feautrier import (
    cancellation_safe_field as stable_field,
    cancellation_safe_response as stable_response,
)


@pytest.mark.parametrize(
    "absorption,scatter", [(-1.0, 1.0), (0.0, 0.0), (1.0, np.nan), (np.inf, 0.0)]
)
def test_invalid_opacity_is_rejected(absorption, scatter):
    with pytest.raises(ValueError, match="opacities"):
        stable_field([0.01, 1.0], [1.0, 1.0], absorption, scatter)


@pytest.mark.parametrize("chunk", [0, -1, 1.5, True])
def test_invalid_chunk_size_is_rejected(chunk):
    with pytest.raises(ValueError, match="chunk_size"):
        stable_field([0.01, 1.0], [1.0, 1.0], 1.0, 0.0, wavelength_chunk_size=chunk)


def test_zero_material_surface_depth_is_rejected():
    with pytest.raises(ValueError, match="above zero"):
        stable_field([0.0, 1.0], [1.0, 1.0], 1.0, 0.0)


def test_response_chunking_and_read_only_consumer():
    mass = np.geomspace(1e-12, 100.0, 14)
    wave = np.geomspace(1000.0, 10000.0, 7)
    b = np.broadcast_to(1.0 + mass, (7, 14)).copy()
    source, _ = stable_field(mass, b, 0.2, 0.8)
    args = (mass, wave, source, 0.2 * b, b, 0.8, mass, np.zeros_like(b))
    expected, expected_j, expected_s = stable_response(*args, wavelength_chunk_size=7)
    consumed = []

    def consume(start, stop, values):
        assert not values.flags.writeable
        consumed.append(values.copy())

    actual, mean, src = stable_response(
        *args,
        wavelength_chunk_size=2,
        return_auxiliary_response=False,
        mean_response_consumer=consume
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-11)
    np.testing.assert_allclose(np.concatenate(consumed), expected_j, rtol=1e-14)
    assert mean is None and src is None
    diagonal = np.zeros_like(expected_s)
    diagonal[:, np.arange(14), np.arange(14)] = 0.2 * b
    np.testing.assert_allclose(expected_s, 0.8 * expected_j + diagonal)


@pytest.mark.parametrize(
    "bad_index,bad_value",
    [
        (1, np.array([1.0, 1.0])),
        (3, np.ones((2, 3))),
        (5, 1.1),
        (6, np.array([0.0, 1.0])),
        (7, np.full((2, 2), np.nan)),
    ],
)
def test_invalid_response_inputs_are_rejected(bad_index, bad_value):
    ones = np.ones((2, 2))
    args = [
        np.array([0.01, 1.0]),
        np.array([1000.0, 2000.0]),
        ones,
        ones,
        ones,
        0.5,
        np.array([0.01, 1.0]),
        ones,
    ]
    args[bad_index] = bad_value
    with pytest.raises(ValueError):
        stable_response(*args)


@pytest.mark.parametrize("fraction", [0.0, 0.8, 0.999, 0.999999, 1.0])
def test_stable_blocks_reproduce_existing_well_resolved_equations(fraction):
    tau = np.geomspace(1e-3, 100, 50)
    planck = np.broadcast_to(1.0 + tau * 0.2, (2, len(tau))).copy()
    old_s, old = coherent_scattering_feautrier_field(
        tau, planck, 1.0 - fraction, fraction
    )
    new_s, new = stable_field(tau, planck, 1.0 - fraction, fraction)
    np.testing.assert_allclose(new_s, old_s, rtol=1e-9)
    np.testing.assert_allclose(new.mean_intensity, old.mean_intensity, rtol=1e-9)
    np.testing.assert_allclose(
        new.interface_flux, old.interface_flux, rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize("surface_tau", [1e-3, 1e-12])
@pytest.mark.parametrize("epsilon", [0.1, 1e-6])
@pytest.mark.parametrize("step", [1e-3, 2e-3])
def test_temperature_and_opacity_response_finite_difference(surface_tau, epsilon, step):
    mass = np.geomspace(surface_tau, 100.0, 24)
    wave = np.array([2000.0, 5000.0, 15000.0])
    base_b = np.broadcast_to(1 + mass * 0.2, (3, len(mass))).copy()
    base_abs = np.full_like(base_b, epsilon)
    base_scat = np.full_like(base_b, 1-epsilon)
    exponent_b, exponent_a, exponent_s = 4.0, 1.3, -0.2

    def field(x):
        b = base_b * np.exp(exponent_b * x)
        a = base_abs * np.exp(exponent_a * x)
        s = base_scat * np.exp(exponent_s * x)
        tau = optical_depth_from_mass_opacity(mass, a + s)
        source, result = stable_field(tau, b, a, s)
        return source, result, (tau, b, a, s)

    source, base, (tau, b, a, s) = field(np.zeros(len(mass)))
    deps = (exponent_a - exponent_s) * a * s / (a + s) ** 2
    direct = a / (a + s) * exponent_b * b + deps * (b - base.mean_intensity)
    flux_j, mean_j, _ = stable_response(
        tau,
        wave,
        source,
        direct,
        exponent_b * b,
        s / (a + s),
        mass,
        exponent_a * a + exponent_s * s,
    )
    # A 1e-5 centered probe subtracts nearly identical scattering-dominated
    # fields and amplifies platform-dependent solve roundoff. Fourth-order
    # differences at two wider steps control truncation AND cancellation;
    # retain the original tangent tolerances and all optical-depth cases.
    for index in [0, 8, 20, 23]:
        dx = np.eye(len(mass))[index] * step
        _, plus, _ = field(dx)
        _, minus, _ = field(-dx)
        _, far_plus, _ = field(2 * dx)
        _, far_minus, _ = field(-2 * dx)
        measured_mean = (
            8 * (plus.mean_intensity - minus.mean_intensity)
            - (far_plus.mean_intensity - far_minus.mean_intensity)
        ) / (12 * step)
        measured_flux = trapezoid(
            (8 * (plus.interface_flux - minus.interface_flux)
             - (far_plus.interface_flux - far_minus.interface_flux)) / (12 * step),
            wave, axis=0,
        )
        np.testing.assert_allclose(
            mean_j[:, :, index], measured_mean, rtol=3e-5, atol=2e-7
        )
        np.testing.assert_allclose(
            flux_j[:, index], measured_flux, rtol=5e-5, atol=2e-3
        )


def test_thin_cells_against_70_digit_discrete_reference():
    mp = pytest.importorskip("mpmath")
    tau = np.geomspace(1e-12, 10.0, 60)
    planck = np.ones((2, len(tau)))
    _, new = stable_field(tau, planck, 1.0, 0.0, n_angle=1)
    _, old = coherent_scattering_feautrier_field(tau, planck, 1.0, 0.0, n_angle=1)
    with mp.workdps(70):
        expanded = [mp.mpf(0)] + [mp.mpf(float(t)) for t in tau]
        h = [expanded[i + 1] - expanded[i] for i in range(len(tau))]
        n = len(expanded)
        matrix, rhs = mp.matrix(n), mp.matrix(n, 1)
        mu = mp.mpf(".5")
        matrix[0, 0], matrix[0, 1] = 1 + mu / h[0], -mu / h[0]
        for i in range(1, n - 1):
            a = 2 * mu ** 2 / (h[i - 1] * (h[i - 1] + h[i]))
            c = 2 * mu ** 2 / (h[i] * (h[i - 1] + h[i]))
            matrix[i, i - 1], matrix[i, i], matrix[i, i + 1] = -a, 1 + a + c, -c
            rhs[i] = 1
        matrix[n - 1, n - 1], rhs[n - 1] = 1, 1
        u = mp.lu_solve(matrix, rhs)
        expected = np.array([float(u[i]) for i in range(1, n)])
        expected_flux = np.array(
            [
                float(4 * mp.pi * mu ** 2 * (u[i + 1] - u[i]) / h[i])
                for i in range(n - 1)
            ]
        )
    np.testing.assert_allclose(new.mean_intensity[0], expected, rtol=3e-14)
    np.testing.assert_allclose(
        new.interface_flux[0], expected_flux, rtol=3e-12, atol=1e-14
    )
    # Ensure this regression actually exposes the original loss of precision.
    assert np.max(abs(old.mean_intensity[0] - expected)) > 1e-7
