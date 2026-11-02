"""Read MAKEE HIRES products (Flux-/Err-NNN_C.fits) into per-order arrays."""
from pathlib import Path

import numpy as np
from astropy.io import fits


def order_wavelengths(header, norders, npix):
    """MAKEE WV_0_nn / WV_4_nn polynomial (pixel index starting at 0) -> wavelength per order."""
    x = np.arange(npix, dtype=float)
    wave = np.empty((norders, npix))
    for i in range(norders):
        c = [float(v) for v in header[f"WV_0_{i + 1:02d}"].split()] + \
            [float(v) for v in header[f"WV_4_{i + 1:02d}"].split()]
        wave[i] = np.polynomial.polynomial.polyval(x, c)
    return wave


def read_makee(directory):
    """Return dict with wave, flux, err (n_orders x n_pix), header and helio velocity for one MAKEE run."""
    directory = Path(directory)
    flux_path = next(directory.glob("Flux-*.fits"))
    err_path = flux_path.with_name(flux_path.name.replace("Flux-", "Err-"))
    flux, header = fits.getdata(flux_path, header=True)
    err = fits.getdata(err_path)
    wave = order_wavelengths(header, *flux.shape)
    return {"wave": wave, "flux": flux.astype(float), "err": err.astype(float), "header": header,
            "heliovel": float(header.get("HELIOVEL", np.nan)), "path": str(flux_path)}
