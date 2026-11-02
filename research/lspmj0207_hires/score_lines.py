#!/usr/bin/env python3
"""Score a list of lines against several OpenWD models.

For each line (rest air wavelength) and model: the model (R = 37,000 + pixel box,
+53.04 km/s) is scaled to the data with a clipped linear factor over +/- 3 A, and
    resid = inverse-variance mean of (model - data)/continuum within +/- 0.08 A
with its significance.  resid > 0: the model line is too weak (data deeper);
resid < 0: too strong.  Also chi2 per pixel within +/- 0.25 A.

usage: python -I score_lines.py SPECTRUM.npz LINES.csv OUT.csv MODELDIR=label [...]
       (LINES.csv needs columns lambda_air, ion)
"""
import csv
import sys
from pathlib import Path

import numpy as np

C_KMS = 299792.458
VEL = 53.04
R_INSTR = 37000.0
spec_path, lines_path, out = (Path(a) for a in sys.argv[1:4])
models = [(Path(a.split("=")[0]), a.split("=")[1]) for a in sys.argv[4:]]


def vac_to_air(w):
    s2 = (1e4 / w) ** 2
    return w / (1 + 8.34254e-5 + 2.406147e-2 / (130 - s2) + 1.5998e-4 / (38.9 - s2))


def convolve(wave, flux):
    lg = np.arange(np.log(wave[0]), np.log(wave[-1]), 0.2 / C_KMS)
    fl = np.interp(lg, np.log(wave), flux)
    sigma = (C_KMS / R_INSTR / 2.3548) / 0.2
    x = np.arange(-int(5 * sigma), int(5 * sigma) + 1)
    kern = np.convolve(np.exp(-0.5 * (x / sigma) ** 2), np.ones(int(round(1.3 / 0.2))))
    kern /= kern.sum()
    return np.exp(lg), np.convolve(np.pad(fl, len(kern), mode="edge"), kern, mode="same")[len(kern):-len(kern)]


sp = np.load(spec_path)
ok = np.isfinite(sp["FLUX"]) & np.isfinite(sp["ERR"]) & (sp["ERR"] > 0)
d_air = vac_to_air(sp["WAVE_VAC_HELIO"][ok]) / (1 + VEL / C_KMS)
dfl, der = sp["FLUX"][ok], sp["ERR"][ok]

loaded = []
for directory, label in models:
    m = np.load(directory / "spectrum_fine.npz")
    sel = (m["wavelength"] > 3150) & (m["wavelength"] < 6100)
    w, f = convolve(m["wavelength"][sel], m["flux"][sel])
    _, c = convolve(m["wavelength"][sel], m["continuum"][sel])
    loaded.append((label, vac_to_air(w), f, c))


def score(lam, mw, mf, mc):
    win = np.abs(d_air - lam) < 3.0
    w, y, e = d_air[win], dfl[win], der[win]
    mm = np.interp(w, mw, mf)
    x = w - lam
    good = np.ones_like(w, bool)
    for _ in range(8):
        A = np.vstack([mm, mm * x]).T
        coef, *_ = np.linalg.lstsq(A[good] / e[good, None], y[good] / e[good], rcond=None)
        new = np.abs((y - A @ coef) / e) < 4
        if np.array_equal(new, good):
            break
        good = new
    scale = coef[0] + coef[1] * x
    cont = np.interp(w, mw, mc)
    core, near = np.abs(x) < 0.08, np.abs(x) < 0.25
    iv = (scale[core] * cont[core] / e[core]) ** 2
    resid = np.sum((mm[core] - y[core] / scale[core]) / cont[core] * iv) / iv.sum()
    chi2 = float(np.mean(((y - scale * mm) / e)[near] ** 2))
    return resid, resid * np.sqrt(iv.sum()), chi2


rows = list(csv.DictReader(open(lines_path)))
table = []
for r in rows:
    lam = float(r["lambda_air"])
    entry = {"lambda_air": r["lambda_air"], "ion": r["ion"]}
    for label, mw, mf, mc in loaded:
        resid, snr, chi2 = score(lam, mw, mf, mc)
        entry[f"{label}:resid"], entry[f"{label}:sigma"], entry[f"{label}:chi2"] = round(resid, 3), round(snr, 1), round(chi2, 2)
    table.append(entry)
with open(out, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
labels = [label for label, *_ in loaded]
print(f"{'line':18s}" + "".join(f"{lab[:26]:>30s}" for lab in labels))
for e in table:
    print(f"{e['lambda_air']:>9s} {e['ion']:8s}" + "".join(
        f"{e[f'{lab}:resid']:>+12.3f} ({e[f'{lab}:sigma']:>+5.1f}s) chi2 {e[f'{lab}:chi2']:>5.2f}" for lab in labels))
for lab in labels:
    print(f"{lab}: median |sigma| {np.median([abs(e[f'{lab}:sigma']) for e in table]):.1f}, "
          f"sum chi2 {sum(e[f'{lab}:chi2'] for e in table):.1f}")
