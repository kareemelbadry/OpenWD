import numpy as np
from audit_dense_spectrum import coupled_linear_source
from wd_spectra.radiative_transfer import radiation_field


def test_independent_linear_scattering_solve_satisfies_source_equation():
    tau = np.broadcast_to(np.geomspace(1e-7, 100, 25), (3, 25)).copy()
    b = (1+np.sqrt(tau))*np.array([1., 2., .5])[:, None]
    absorption = np.broadcast_to(np.array([1., .02, 1e-5])[:, None], b.shape)
    scattering = 1-absorption
    source = coupled_linear_source(tau, b, absorption, scattering)
    j = radiation_field(tau, source).mean_intensity
    np.testing.assert_allclose(source, absorption*b+scattering*j, rtol=3e-13)
    np.testing.assert_array_equal(source[0], b[0])
