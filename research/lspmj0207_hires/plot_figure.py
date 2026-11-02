#!/usr/bin/env python3
"""LSPM J0207+3331: the two documentation figures (Keck/HIRES against one OpenWD model).

The panels cover the twelve windows of Figure 6 of Le Bourdais & Dufour (2025,
arXiv:2510.08676), six per figure. The HIRES data have gaps at about
4004-4061 and 4993-5051 A (vacuum).

Model treatment (as in compare_hires.py):
  * Gaussian LSF R = 37,000 (C5 decker), then a 1.3 km/s pixel box.
  * One line velocity, by chi^2 over 3800-5000 A.
  * Vacuum -> air; the plots use rest-frame air wavelengths.
Normalization: in each panel (plus a 25 A margin) the data are fit as
D = M_conv * P(lambda), with P a quadratic and iterative 4-sigma clipping. Plotted are
D / (P * C_conv) and M_conv / C_conv, where C is the model's true continuum. The
pseudo-continuum level is therefore the model's; the panels test line contrasts.

usage: python -I plot_figure.py SPECTRUM.npz MODELDIR OUTSTEM
writes OUTSTEM-1.png/.pdf, OUTSTEM-2.png/.pdf and OUTSTEM.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C_KMS = 299792.458
R_INSTR = 37000.0
PIX_KMS = 1.3
BIN = 3  # display binning: 3 px = 3.9 km/s, half a resolution element
MARGIN = 25.0
DEGREE = 2
SKY_AIR = [(5577.0, 5578.4), (5889.5, 5890.4), (5895.5, 5896.4)]  # observed frame, air
DATA_COLOR, MODEL_COLOR = "0.2", "#c0392b"  # as in the Xu et al. (2019) figures

# Figure 6 of Le Bourdais & Dufour (2025), rest air wavelengths
FIGURES = [
    [(3245, 3375), (3375, 3480), (3480, 3575), (3575, 3680), (3680, 3780), (3780, 3880)],
    [(3883, 3980), (4053, 4180), (4180, 4320), (4320, 4460), (4460, 4630), (4800, 4990)],
]
LABELS = {  # a few landmarks, rest air
    3933.66: "Ca II K", 3968.47: "Ca II H", 4077.71: "Sr II", 4101.74: r"H$\delta$",
    4215.52: "Sr II", 4226.73: "Ca I", 4340.47: r"H$\gamma$", 4861.33: r"H$\beta$",
}


def vac_to_air(w):
    s2 = (1e4 / w) ** 2
    return w / (1 + 8.34254e-5 + 2.406147e-2 / (130 - s2) + 1.5998e-4 / (38.9 - s2))


def smooth_loggrid(wave, flux, out_wave):
    """Gaussian LSF at R_INSTR plus a pixel box, on a uniform log-lambda grid."""
    step = 0.2 / C_KMS
    lg = np.arange(np.log(wave[0]), np.log(wave[-1]), step)
    f = np.interp(lg, np.log(wave), flux)
    sigma = (C_KMS / R_INSTR / 2.3548) / 0.2
    x = np.arange(-int(5 * sigma), int(5 * sigma) + 1)
    kern = np.convolve(np.exp(-0.5 * (x / sigma) ** 2), np.ones(int(round(PIX_KMS / 0.2))))
    kern /= kern.sum()
    g = np.convolve(np.pad(f, len(kern), mode="edge"), kern, mode="same")[len(kern):-len(kern)]
    return np.interp(np.log(out_wave), lg, g)


def poly_norm(w, d, e, m, degree):
    x = (w - w.mean()) / (np.ptp(w) / 2)
    good = np.isfinite(e)
    A = np.vstack([m * x ** k for k in range(degree + 1)]).T
    for _ in range(10):
        coef, *_ = np.linalg.lstsq(A[good] / e[good, None], d[good] / e[good], rcond=None)
        new = np.isfinite(e) & (np.abs((d - A @ coef) / e) < 4)
        if np.array_equal(new, good):
            break
        good = new
    return sum(coef[k] * x ** k for k in range(degree + 1))


spec_path, model_dir, out_stem = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

tab = np.load(spec_path)
dw, df, de = tab["WAVE_VAC_HELIO"], tab["FLUX"], tab["ERR"]
ok = np.isfinite(df) & np.isfinite(de) & (de > 0)
dw, df, de = dw[ok], df[ok], de[ok].copy()
obs_air = vac_to_air(dw)
for lo, hi in SKY_AIR:  # sky lines sit at fixed topocentric wavelengths (helio - 12 km/s)
    de[(obs_air * (1 - 12.1 / C_KMS) > lo) & (obs_air * (1 - 12.1 / C_KMS) < hi)] = np.inf

inputs = json.loads((model_dir / "inputs.json").read_text())
summary_path = model_dir / "summary.json"
if summary_path.exists() and not json.loads(summary_path.read_text()).get("verified", False):
    sys.exit(f"{model_dir}: model did not pass its convergence certificate")
mod = np.load(model_dir / "spectrum_fine.npz")
sel = (mod["wavelength"] > 3050) & (mod["wavelength"] < 6100)
lg = np.arange(np.log(mod["wavelength"][sel][0]), np.log(mod["wavelength"][sel][-1]), 0.2 / C_KMS)
model_flux = smooth_loggrid(mod["wavelength"][sel], mod["flux"][sel], np.exp(lg))
model_cont = smooth_loggrid(mod["wavelength"][sel], mod["continuum"][sel], np.exp(lg))


def model_at(velocity, wave_vac_helio):
    rest = np.log(wave_vac_helio / (1 + velocity / C_KMS))
    return np.interp(rest, lg, model_flux), np.interp(rest, lg, model_cont)


def chi2(v):
    total = 0.0
    for a in np.arange(3800, 5000, 50):
        m = (dw > a) & (dw < a + 50)
        mm, _ = model_at(v, dw[m])
        p = poly_norm(dw[m], df[m], de[m], mm, 2)
        z = (df[m] - p * mm) / de[m]
        total += np.sum(z[np.abs(z) < 4] ** 2)
    return total


vgrid = np.arange(-100, 150.1, 5.0)
v0 = vgrid[np.argmin([chi2(v) for v in vgrid])]
vfine = np.arange(v0 - 6, v0 + 6.01, 0.5)
pp = np.polyfit(vfine, [chi2(v) for v in vfine], 2)
vel = -pp[1] / (2 * pp[0])
print(f"line velocity (heliocentric) = {vel:.2f} km/s")
rest_air = obs_air / (1 + vel / C_KMS)

ab = inputs["abundances_log_n_z_over_n_h"]
abundance_text = "  ".join(f"{k} {v:.2f}" for k, v in ab.items())
summary = {"model": str(model_dir), "velocity_kms": vel, "panels": []}
for number, windows in enumerate(FIGURES, start=1):
    fig, axes = plt.subplots(len(windows), 1, figsize=(16, 19))
    fig.subplots_adjust(hspace=0.28, left=0.06, right=0.985, top=0.915, bottom=0.045)
    for ax, (lo, hi) in zip(axes, windows):
        fit = (rest_air > lo - MARGIN) & (rest_air < hi + MARGIN)
        w, d, e, ra = dw[fit], df[fit], de[fit], rest_air[fit]
        mm, mc = model_at(vel, w)
        p = poly_norm(w, d, e, mm, DEGREE)
        dn, en, mn = d / (p * mc), e / (p * mc), mm / mc
        show = (ra >= lo) & (ra <= hi)
        idx = np.where(show)[0]
        # bin only runs of consecutive pixels, so nothing is averaged across a data gap
        breaks = np.where(np.diff(ra[idx]) > 0.5)[0] + 1
        wb, db = [], []
        for run in np.split(idx, breaks):
            n = len(run) // BIN * BIN
            if n == 0:
                continue
            ivar = np.where(np.isfinite(en[run[:n]]), en[run[:n]] ** -2.0, 0.0).reshape(-1, BIN)
            vals = np.nan_to_num(dn[run[:n]]).reshape(-1, BIN)
            wb += list(ra[run[:n]].reshape(-1, BIN).mean(1)) + [np.nan]
            db += list(np.where(ivar.sum(1) > 0, (vals * ivar).sum(1) / np.maximum(ivar.sum(1), 1e-30),
                                np.nan)) + [np.nan]
        ax.plot(wb, db, color=DATA_COLOR, lw=0.6, label="HIRES")
        model_line = np.where(np.r_[np.diff(ra[idx]) > 0.5, False], np.nan, mn[idx])  # break at gaps
        ax.plot(ra[idx], model_line, color=MODEL_COLOR, lw=1.2, label="OpenWD")
        ax.axhline(1, color="0.6", ls="--", lw=0.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.35)
        for wl, text in LABELS.items():
            if lo < wl < hi:
                ax.annotate(text, (wl, 1.22), ha="center", fontsize=10)
        ax.tick_params(direction="in", top=True, right=True, labelsize=11)
        resid = ((d - p * mm) / e)[show]
        summary["panels"].append({"figure": number, "range": [lo, hi],
                                  "chi2_red": float(np.nanmean(resid[np.isfinite(resid)] ** 2)),
                                  "median_snr_per_pix": float(np.nanmedian(1 / en[show]))})
    axes[0].legend(loc="lower right", fontsize=10)
    fig.suptitle(f"LSPM J0207+3331 — Le Bourdais & Dufour (2025) parameters and abundances fixed ({number}/2)\n"
                 f"Keck/HIRES 2020-10-08 (R = 37,000), binned by {BIN} px · OpenWD default, "
                 f"Teff = {inputs['teff']:,.0f} K, log g = {inputs['logg']:.2f}, hydrogen host · "
                 f"line velocity {vel:+.1f} km/s · nothing fitted\n"
                 f"log N(Z)/N(H): {abundance_text}", fontsize=13)
    fig.supxlabel("Rest air wavelength (Å)", fontsize=13)
    fig.supylabel("Flux / model continuum", fontsize=13)
    fig.savefig(f"{out_stem}-{number}.png", dpi=110)
    from PIL import Image  # a 64-colour palette keeps the PNG near 300 kB
    Image.open(f"{out_stem}-{number}.png").convert("RGB").quantize(
        colors=64, method=Image.Quantize.MEDIANCUT).save(f"{out_stem}-{number}.png", optimize=True)
    fig.savefig(f"{out_stem}-{number}.pdf")
    plt.close(fig)
Path(f"{out_stem}.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps(summary, indent=1))
