#!/usr/bin/env python3
"""LSPM J0207+3331: merged Keck/HIRES spectrum vs the OpenWD DAZ model at the
Le Bourdais & Dufour (2025) parameters (nothing fitted except the RV and a
low-order continuum correction per window).

Model treatment (same idea as the Xu et al. 2019 HIRES diagnostics in docs/):
  * Gaussian LSF R = 37,000 (C5 decker, 1.148"), then a 1.3 km/s pixel box.
  * Shift by the measured line velocity (cross-correlation, 3800-5000 A).
  * Vacuum -> air (Morton 2000 / IAU), plots in rest-frame air wavelengths.
Normalization: in each window (plus a 25 A margin) the data are fit as
  D = M_conv * P(lambda), P a quadratic (linear for the broad-line windows),
with iterative 4-sigma clipping. Plotted: D / (P * C_conv) and M_conv / C_conv,
where C is the model's own true continuum. y = 1 is the model's true continuum,
but P absorbs any overall level, so the depressed pseudo-continuum is the
model's and is NOT tested here; only line contrasts within a window are.
(The broad blanketing is tested separately in compare_sed.py.)

usage: python -I compare_hires.py SPECTRUM.npz|.fits OUT.png MODELDIR[=label] [MODELDIR=label ...]
The first model sets the velocity; every model gets its own continuum polynomial.
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
BIN = 3  # display binning (3 px = 3.9 km/s, half a resolution element)
SKY_AIR = [(5577.0, 5578.4), (5889.5, 5890.4), (5895.5, 5896.4)]  # [O I], Na D sky (observed frame, air)

spec_path, out_png = Path(sys.argv[1]), Path(sys.argv[2])
MODEL_ARGS = [(Path(a.split("=")[0]), a.split("=")[1] if "=" in a else "OpenWD") for a in sys.argv[3:]]
COLORS = ["#c0392b", "#2471a3", "#1e8449", "#b9770e"]


def vac_to_air(w):
    s2 = (1e4 / w) ** 2
    return w / (1 + 8.34254e-5 + 2.406147e-2 / (130 - s2) + 1.5998e-4 / (38.9 - s2))


def smooth_loggrid(wave, flux, out_wave):
    """Gaussian LSF at R_INSTR plus a pixel box, on a uniform log-lambda grid."""
    step = 0.2 / C_KMS
    lg = np.arange(np.log(wave[0]), np.log(wave[-1]), step)
    f = np.interp(lg, np.log(wave), flux)
    sigma = (C_KMS / R_INSTR / 2.3548) / 0.2
    half = int(5 * sigma)
    x = np.arange(-half, half + 1)
    kern = np.exp(-0.5 * (x / sigma) ** 2)
    box = np.ones(int(round(PIX_KMS / 0.2)))
    kern = np.convolve(kern, box)
    kern /= kern.sum()
    g = np.convolve(np.pad(f, len(kern), mode="edge"), kern, mode="same")[len(kern):-len(kern)]
    return np.interp(np.log(out_wave), lg, g)


# ---- data
if spec_path.suffix == ".npz":
    tab = np.load(spec_path)
else:  # FITS table written by build_spectrum.py
    from astropy.io import fits
    tab = fits.getdata(spec_path)
dw, df, de = tab["WAVE_VAC_HELIO"], tab["FLUX"], tab["ERR"]
ok = np.isfinite(df) & np.isfinite(de) & (de > 0)
dw, df, de = dw[ok], df[ok], de[ok]
obs_air = vac_to_air(dw)
for lo, hi in SKY_AIR:  # sky lines are at fixed observed (topocentric ~ helio - 12 km/s) wavelengths
    m = (obs_air * (1 - 12.1 / C_KMS) > lo) & (obs_air * (1 - 12.1 / C_KMS) < hi)
    de[m] = np.inf

# ---- models
class Model:
    def __init__(self, directory, label):
        mod = np.load(directory / "spectrum_fine.npz")
        sel = (mod["wavelength"] > 3050) & (mod["wavelength"] < 6100)
        mw, mf, mc = mod["wavelength"][sel], mod["flux"][sel], mod["continuum"][sel]
        self.inputs = json.loads((directory / "inputs.json").read_text())
        self.label = label
        self.lg = np.arange(np.log(mw[0]), np.log(mw[-1]), 0.2 / C_KMS)
        self.f = smooth_loggrid(mw, mf, np.exp(self.lg))  # convolve once, then shift
        self.c = smooth_loggrid(mw, mc, np.exp(self.lg))

    def at(self, velocity, wave_vac_helio):
        rest = np.log(wave_vac_helio / (1 + velocity / C_KMS))
        return np.interp(rest, self.lg, self.f), np.interp(rest, self.lg, self.c)


models = [Model(d, label) for d, label in MODEL_ARGS]
inputs = models[0].inputs
model_at = models[0].at


def poly_norm(w, d, e, m, degree, mask=None):
    x = (w - w.mean()) / (np.ptp(w) / 2)
    good = np.isfinite(e) & (mask if mask is not None else True)
    for _ in range(10):
        A = np.vstack([m * x ** k for k in range(degree + 1)]).T
        coef, *_ = np.linalg.lstsq(A[good] / e[good, None], d[good] / e[good], rcond=None)
        fit = A @ coef
        z = (d - fit) / e
        new = np.isfinite(e) & (np.abs(z) < 4) & (mask if mask is not None else True)
        if np.array_equal(new, good):
            break
        good = new
    return sum(coef[k] * x ** k for k in range(degree + 1)), good


# ---- radial velocity: chi^2 over 50 A chunks, 3800-5000 A
def chi2(v):
    total = 0.0
    for a in np.arange(3800, 5000, 50):
        m = (dw > a) & (dw < a + 50)
        mm, _ = model_at(v, dw[m])
        p, g = poly_norm(dw[m], df[m], de[m], mm, 2)
        total += np.sum(((df[m] - p * mm) / de[m])[g] ** 2)
    return total


vgrid = np.arange(-100, 150.1, 5.0)
c = np.array([chi2(v) for v in vgrid])
v0 = vgrid[np.argmin(c)]
vfine = np.arange(v0 - 6, v0 + 6.01, 0.5)
cf = np.array([chi2(v) for v in vfine])
pp = np.polyfit(vfine, cf, 2)
vel = -pp[1] / (2 * pp[0])
vel_err = 1 / np.sqrt(pp[0])
print(f"line velocity (heliocentric) = {vel:.2f} +/- {vel_err:.2f} km/s")

# ---- windows (rest air)
WINDOWS = [
    ("Cu I / Fe I", 3240, 3280, 2),
    ("Ni I / Co I / Fe I", 3410, 3470, 2),
    ("Fe I / Cr I", 3565, 3615, 2),
    ("Mg I / Fe I", 3815, 3865, 2),
    ("Si I / Ca II H & K / Al I", 3895, 3990, 1),
    ("H$\\delta$ / Fe I", 4060, 4130, 1),
    ("Ca I / Cr I / Fe I", 4220, 4300, 2),
    ("H$\\gamma$", 4310, 4370, 1),
    ("H$\\beta$", 4830, 4895, 1),
    ("Mg I b / Fe I", 5160, 5195, 2),
    ("Na I D", 5875, 5905, 1),
]
MARGIN = 25.0
import os  # noqa: E402
if os.environ.get("WINDOW_SET") == "docs":  # windows of the documentation figure
    WINDOWS = [
        ("Sr II 4077.7", 4065, 4090, 2),
        ("Ni I / Co I / Fe I", 3455, 3490, 2),
        ("Fe I / Cr I", 3565, 3615, 2),
        ("Mg I / Fe I", 3815, 3865, 2),
        ("Si I / Ca II H & K / Al I", 3895, 3990, 1),
        ("Sr II 4215.5 / Ca I 4226.7", 4205, 4240, 2),
        ("Ca I / Cr I / Fe I", 4240, 4300, 2),
        ("H$\\beta$", 4830, 4895, 1),
        ("Mg I b / Fe I", 5160, 5195, 2),
        ("Na I D", 5875, 5905, 1),
        ("Fe I / Cr I / Ni I", 3800, 3815, 2),
    ]

fig = plt.figure(figsize=(16, 21))
gs = fig.add_gridspec(6, 2, hspace=0.42, wspace=0.12, left=0.06, right=0.985, top=0.955, bottom=0.04)
axes = [fig.add_subplot(gs[0, 0])] + [fig.add_subplot(gs[r, k]) for r in range(1, 6) for k in (0, 1)]
info_ax = fig.add_subplot(gs[0, 1])
rest_air_all = obs_air / (1 + vel / C_KMS)
summary = {"velocity_kms": vel, "velocity_err_kms": vel_err, "models": [m.label for m in models], "windows": []}
for ax, (title, lo, hi, deg) in zip(axes, WINDOWS):
    fitm = (rest_air_all > lo - MARGIN) & (rest_air_all < hi + MARGIN)
    w, d, e = dw[fitm], df[fitm], de[fitm]
    ra = rest_air_all[fitm]
    show = (ra >= lo) & (ra <= hi)
    entry = {"title": title, "range": [lo, hi], "chi2_red": {}}
    for k, model in enumerate(models):
        mm, mcont = model.at(vel, w)
        p, good = poly_norm(w, d, e, mm, deg)
        dn, en = d / (p * mcont), e / (p * mcont)
        mn = mm / mcont
        if k == 0:  # data shown with the first model's normalization
            n = show.sum() // BIN * BIN
            idx = np.where(show)[0][:n]
            wb = ra[idx].reshape(-1, BIN).mean(1)
            ivar = np.where(np.isfinite(en[idx]), 1 / en[idx] ** 2, 0).reshape(-1, BIN)
            db = np.where(ivar.sum(1) > 0, (np.nan_to_num(dn[idx]).reshape(-1, BIN) * ivar).sum(1)
                          / np.maximum(ivar.sum(1), 1e-30), np.nan)
            ax.plot(wb, db, color="0.15", lw=0.6, drawstyle="steps-mid", label="HIRES")
            entry["median_snr_per_pix"] = float(np.nanmedian(1 / en[show]))
            ymin = np.nanmin(mn[show])
        else:  # put this model on the first model's data normalization
            ref_m, ref_c = models[0].at(vel, w)
            p0, _ = poly_norm(w, d, e, ref_m, deg)
            mn = mn * (p * mcont) / (p0 * ref_c)
            ymin = min(ymin, np.nanmin(mn[show]))
        ax.plot(ra[show], mn[show], color=COLORS[k], lw=1.1 if k == 0 else 0.9,
                ls="-" if k == 0 else "--", label=model.label)
        resid = (d - p * mm)[show] / e[show]
        entry["chi2_red"][model.label] = float(np.nanmean(resid[np.isfinite(resid)] ** 2))
    ax.axhline(1, color="0.6", ls="--", lw=0.7)
    ax.set_xlim(lo, hi)
    top = np.nanpercentile(db, 99.5)
    ax.set_ylim(min(0, ymin - 0.05), max(1.25, top + 0.05))
    ax.set_title(title, fontsize=13)
    ax.tick_params(direction="in", top=True, right=True)
    summary["windows"].append(entry)
axes[0].legend(loc="lower right", fontsize=9 if os.environ.get("WINDOW_SET") == "docs" else 11)
info_ax.axis("off")
ab = inputs["abundances_log_n_z_over_n_h"]
abtxt = "  ".join(f"{k} {v:+.2f}" for k, v in ab.items())
lines = [
    "LSPM J0207+3331  (Keck/HIRES, 2020-10-08, 2 x 3600 s, C5)",
    f"Teff = {inputs['teff']:,.0f} K  ·  log g = {inputs['logg']:.2f}  ·  H atmosphere",
    "Le Bourdais & Dufour (2025) parameters and abundances fixed",
    "log N(Z)/N(H):",
    abtxt[:len(abtxt) // 2].rsplit("  ", 0)[0],
    abtxt[len(abtxt) // 2:],
    "Sr omitted (no OpenWD atomic data); upper limits omitted",
    "",
    f"OpenWD {inputs['openwd_version']} ({inputs['openwd_commit'][:7]}), DAZ, quality = {inputs['quality']}",
    *([f"Model {i + 1}: {m.label}" for i, m in enumerate(models)] if len(models) > 1 else []),
    f"Gaussian LSF R = 37,000 + pixel; line velocity {vel:+.1f} km/s (CC)",
    "Data scaled to model x quadratic per window (pseudo-continuum level",
    "  is the model's; line contrasts are tested); y = 1 = true continuum",
    f"HIRES binned by {BIN} px (3.9 km/s); no abundances or widths fitted",
]
if os.environ.get("WINDOW_SET") == "docs":
    lines = [
        "Keck/HIRES 2020-10-08, 2 x 3600 s, C5 decker (R = 37,000)",
        f"Teff = {inputs['teff']:,.0f} K · log g = {inputs['logg']:.2f} · hydrogen host",
        "Le Bourdais & Dufour (2025) parameters and abundances",
        "log N(Z)/N(H): " + abtxt[:len(abtxt) // 2].strip(),
        "    " + abtxt[len(abtxt) // 2:].strip() + ", Sr -10.85 (model 2 only)",
        "",
        *[f"{['Red', 'Blue dashed'][i]}: {m.label}" for i, m in enumerate(models)],
        f"Line velocity {vel:+.1f} km/s; nothing fitted",
        "Each window: data scaled to model × quadratic;",
        "y = 1 is the model continuum; HIRES binned by 3 px",
    ]
    info_ax.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=11, family="sans-serif", transform=info_ax.transAxes)
else:
    info_ax.text(0.02, 0.95, "\n".join(lines), va="top", fontsize=12.5, family="sans-serif", transform=info_ax.transAxes)
fig.suptitle("LSPM J0207+3331 — Le Bourdais & Dufour (2025) parameters and abundances fixed", fontsize=17)
fig.supxlabel("Rest air wavelength (Å)", fontsize=14)
fig.supylabel("Flux / model continuum", fontsize=14)
fig.savefig(out_png, dpi=130)
fig.savefig(out_png.with_suffix(".pdf"))
out_png.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=1))
