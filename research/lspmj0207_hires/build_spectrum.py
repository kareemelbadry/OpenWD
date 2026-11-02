#!/usr/bin/env python3
"""Combine the MAKEE extractions of LSPM J0207+3331 into one merged HIRES spectrum.

Fixes applied to the raw MAKEE Flux (flat-fielded counts, vacuum, heliocentric):
  1. Response: every science order is divided by the same order of EGGR 180
     (WD 0426+588 = Stein 2051B, a featureless DC; same setup, same trace and flat),
     after a robust Chebyshev fit to EGGR 180 so its noise and cosmic rays are not
     propagated.  This removes blaze/flat-lamp residuals and the order-edge steps.
     The result is F(LSPM)/F(EGGR 180) times a smooth function: it is NOT flux
     calibrated, and the comparison script normalizes it against the model.
  2. Cosmic rays: the two 3600 s exposures are compared pixel by pixel; a pixel
     more than 4 sigma above the scaled other exposure is masked (with neighbours).
  3. Masked pixels: MAKEE pixels with Err <= 0 (untraced order ends) are dropped,
     plus 40 pixels at each order end where the flat is faint.
  4. Merge: all orders of both exposures are resampled onto one log-lambda grid
     (1.3 km/s, the native pixel) and averaged with inverse-variance weights.

usage: python -I build_spectrum.py REDUCTIONDIR OUTDIR
"""
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hires_io import read_makee  # noqa: E402

SCIENCE = ["HI.20201008.32858", "HI.20201008.36786"]
RESPONSE = "HI.20201008.55363"
EDGE = 40
C_KMS = 299792.458

red, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)


def robust_cheb(x, y, w, degree=9, nsigma=3.0, iterations=8):
    """Iteratively clipped weighted Chebyshev fit; returns the fit evaluated at x."""
    good = w > 0
    for _ in range(iterations):
        coef = np.polynomial.chebyshev.chebfit(x[good], y[good], degree, w=np.sqrt(w[good]))
        model = np.polynomial.chebyshev.chebval(x, coef)
        resid = (y - model) * np.sqrt(w)
        scale = 1.4826 * np.median(np.abs(resid[good]))
        new = (w > 0) & (resid > -nsigma * scale) & (resid < nsigma * scale)
        if np.array_equal(new, good):
            break
        good = new
    return model, good


orders = []  # per-order products for the diagnostics
cr_count = 0
for ccd in (1, 2, 3):
    sci = [read_makee(red / "makee" / f"{f}_ccd{ccd}") for f in SCIENCE]
    resp = read_makee(red / "makee" / f"{RESPONSE}_ccd{ccd}")
    for i in range(sci[0]["flux"].shape[0]):
        npix = sci[0]["flux"].shape[1]
        x = np.linspace(-1, 1, npix)
        valid = np.ones(npix, bool)
        valid[:EDGE] = valid[-EDGE:] = False

        # response: smooth EGGR 180 order, interpolated onto the science wavelengths
        rw, rf, re = resp["wave"][i], resp["flux"][i], resp["err"][i]
        rweight = np.where((re > 0) & valid, 1.0 / np.where(re > 0, re, 1) ** 2, 0.0)
        rsmooth, _ = robust_cheb(x, rf, rweight)
        if np.mean(rweight > 0) < 0.3 or np.any(rsmooth[valid] <= 0):
            continue

        fluxes, errs = [], []
        for s in sci:
            w, f, e = s["wave"][i], s["flux"][i], s["err"][i]
            r = np.interp(w, rw, rsmooth)
            good = (e > 0) & valid & np.isfinite(f)
            fluxes.append(np.where(good, f / r, np.nan))
            errs.append(np.where(good, e / r, np.nan))
        wave = sci[0]["wave"][i]
        f2 = np.interp(wave, sci[1]["wave"][i], fluxes[1])
        e2 = np.interp(wave, sci[1]["wave"][i], errs[1])
        f1, e1 = fluxes[0], errs[0]

        # cosmic rays: compare exposures (scaled to each other)
        both = np.isfinite(f1) & np.isfinite(f2)
        scale = np.nanmedian(f1[both]) / np.nanmedian(f2[both]) if both.sum() > 100 else 1.0
        sig = np.sqrt(e1 ** 2 + (scale * e2) ** 2)
        diff = (f1 - scale * f2) / sig
        hit1 = both & (diff > 4.0)
        hit2 = both & (diff < -4.0)
        for hit, arr in ((hit1, f1), (hit2, f2)):
            grown = hit | np.roll(hit, 1) | np.roll(hit, -1)
            arr[grown] = np.nan
            cr_count += int(hit.sum())
        e1 = np.where(np.isfinite(f1), e1, np.nan)
        e2 = np.where(np.isfinite(f2), e2, np.nan)
        orders.append({"ccd": ccd, "order": i, "wave": wave, "flux": np.vstack([f1, f2 * scale]),
                       "err": np.vstack([e1, e2 * scale]), "response": np.interp(wave, rw, rsmooth),
                       "response_raw": np.interp(wave, rw, rf)})

# global log-lambda grid, 1.3 km/s
lo = min(np.nanmin(o["wave"]) for o in orders)
hi = max(np.nanmax(o["wave"]) for o in orders)
step = 1.3 / C_KMS
grid = np.exp(np.arange(np.log(lo), np.log(hi), step))
num = np.zeros_like(grid)
den = np.zeros_like(grid)
for o in orders:
    for f, e in zip(o["flux"], o["err"]):
        ok = np.isfinite(f) & np.isfinite(e) & (e > 0)
        inside = (grid >= o["wave"][EDGE]) & (grid <= o["wave"][-EDGE - 1])
        fi = np.interp(grid, o["wave"][ok], f[ok])
        wi = np.interp(grid, o["wave"], np.where(ok, 1.0 / np.where(ok, e, 1) ** 2, 0.0))
        # a grid point next to a masked pixel gets zero weight
        bad = np.interp(grid, o["wave"], (~ok).astype(float)) > 0
        wi = np.where(inside & ~bad, wi, 0.0)
        num += wi * fi
        den += wi
flux = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
err = np.where(den > 0, 1 / np.sqrt(np.where(den > 0, den, 1)), np.nan)

cols = [fits.Column("WAVE_VAC_HELIO", "D", unit="Angstrom", array=grid),
        fits.Column("FLUX", "D", array=flux), fits.Column("ERR", "D", array=err)]
header = fits.Header()
header["OBJECT"] = "LSPM J0207+3331"
header["DATE-OBS"] = "2020-10-08"
header["INSTRUME"] = "HIRES"
header["DECKER"] = "C5"
header["EXPTIME"] = (7200.0, "2 x 3600 s")
header["WAVEREF"] = ("vacuum, heliocentric", "MAKEE defaults")
header["FLUXUNIT"] = ("relative", "F(LSPM J0207)/F(EGGR 180) x smooth function")
header["RESPONSE"] = ("EGGR 180 (WD 0426+588, DC)", "HI.20201008.55363")
header["NCRREJ"] = (cr_count, "cosmic-ray pixels rejected")
fits.BinTableHDU.from_columns(cols, header=header).writeto(out / "lspmj0207_hires_merged.fits", overwrite=True)
np.savez_compressed(out / "lspmj0207_hires_orders.npz",
                    **{f"ccd{o['ccd']}_o{o['order']:02d}_{k}": o[k] for o in orders
                       for k in ("wave", "flux", "err", "response", "response_raw")})
print(f"{len(orders)} orders, {cr_count} cosmic-ray pixels, grid {grid[0]:.1f}-{grid[-1]:.1f} A, "
      f"{np.isfinite(flux).mean():.3f} filled")
