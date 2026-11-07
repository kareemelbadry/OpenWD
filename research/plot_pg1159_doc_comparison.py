#!/usr/bin/env python3
"""PG 1159 documentation figures: archival spectra against OpenWD models.

    python research/plot_pg1159_doc_comparison.py --models results/pg1159/docs \
        --observations DATA --output docs/assets

MODELS/<tag>/spectrum.txt are the models of research/pg1159_doc_models.sh
(vacuum wavelength, surface F_lambda, continuum). OBSERVATIONS/<star>/ holds
the archival spectra: ``eso/ADP*.fits`` (VLT/UVES SPY, phase-3, air
wavelengths), ``sdss/spec-*.fits`` and ``fuse/*_vo.fits`` (MAST). They are not
installed with OpenWD.

Four figures are written, as PDF and PNG:

* ``pg1159-optical``: one optical panel per star, sorted by Teff, in the
  layout of the DO/DAO figure.
* ``pg1159-pg1707``, ``pg1159-pg1424``, ``pg1159-pg1159``: FUSE and optical
  (SDSS or UVES) spectra of PG 1707+427, PG 1424+535 and PG 1159-035, in the
  layout of the earlier PG 1424+535 paper figure.

Nothing is fitted. Parameters and mass fractions are the published values in
research/pg1159_doc_models.sh. The model is convolved with a Gaussian at the
instrument resolving power and shifted to the stellar velocity. That
velocity comes from a chi^2 scan over narrow photospheric lines (UVES, FUSE)
and is zero for SDSS. Both spectra are then binned to the same display pixels:
0.5 A for UVES and 0.05 A for FUSE; SDSS is shown at its native sampling.
Data and model are then normalized independently and identically in each
panel and contiguous segment. The normalization is a Chebyshev fit to the
log of the 88th-percentile envelope in ~100 A (FUSE: 8 A) bins, computed on a
3 A (FUSE: 0.3 A) running median so that noise does not bias the data
envelope. UVES exposures of the same arm are co-added
with inverse-variance weights, and the outer 30 A of each arm are dropped.
In FUSE, interstellar lines are shaded and geocoronal emission is masked.
Interstellar Ca II and Na I are labelled "IS" in the optical panels.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

C = 299792.458
PRINT_SCALE = 8.0 / 6.77
AXIS_LABEL_SIZE, TICK_SIZE, SECONDARY_SIZE = 8.5 * PRINT_SCALE, 7.5 * PRINT_SCALE, 6.5 * PRINT_SCALE
OBSERVED, OPENWD, ISM = ".30", "#d95f02", "#cfe5ee"

# tag: (name, observation directory, optical instrument, Teff, log g, He/C/O mass fractions)
STARS = {
    "pg1707": ("PG 1707+427", "pg1707+427", "SDSS", 85000, 7.5, (0.52, 0.45, 0.03)),
    "mct0130": ("MCT 0130-1937", "mct0130-1937", "UVES", 90000, 7.5, (0.7462, 0.2239, 0.0298)),
    "pg1424": ("PG 1424+535", "pg1424+535", "SDSS", 110000, 7.0, (0.52, 0.45, 0.03)),
    "pg1159": ("PG 1159-035", "pg1159-035", "UVES", 140000, 7.0, (0.33, 0.48, 0.17)),
    "abell21": ("Abell 21", "abell21", "UVES", 140000, 6.5, (0.33, 0.48, 0.17)),
    "pg1151": ("PG 1151-029", "pg1151-029", "UVES", 140000, 6.0, (0.33, 0.48, 0.17)),
    "pg1144": ("PG 1144+005", "pg1144+005", "UVES", 150000, 6.5, (0.38, 0.57, 0.016)),
    "he1429": ("HE 1429-1209", "he1429-1209", "UVES", 160000, 6.0, (0.38, 0.54, 0.06)),
    "rxj0122": ("RX J0122.9-7521", "rxj0122-7521", "UVES", 175000, 7.7, (0.626, 0.372, 0.0027)),
}
OPTICAL_RANGE = (3800.0, 6680.0)
UVES_EDGE = 30.0
# Interstellar Ca II H+K and Na I D (mean vacuum A of each pair), labelled "IS" in the optical panels.
OPTICAL_ISM = (3952.2, 5894.6)
# Geocoronal emission in FUSE (vacuum A, half width A): H I Ly beta, Ly gamma, O I 989.
AIRGLOW = ((1025.72, 1.3), (972.54, 0.7), (988.77, 0.5))
# Narrow photospheric lines for the velocity scan (vacuum A).
VELOCITY_WINDOWS = {
    "UVES": ((3805.0, 3842.0), (4650.0, 4700.0), (5795.0, 5820.0)),
    "FUSE": ((1105.0, 1110.0), (1167.0, 1171.0)),
}
FUSE_PANELS = ((912.0, 985.0), (985.0, 1050.0), (1050.0, 1120.0), (1120.0, 1188.0))
SDSS_PANELS = ((3850.0, 5000.0), (5000.0, 6700.0), (6700.0, 9150.0))
UVES_PANELS = ((3800.0, 4560.0), (4560.0, 5660.0), (5660.0, 6680.0))
FUSE_STARS = ("pg1707", "pg1424", "pg1159")
# Interstellar lines in FUSE (rest vacuum A, half width of the shaded band in A).
# H I Lyman series, then the metal lines of research/scripts compare.py.
FUSE_ISM = (
    [(1025.72, 1.0), (972.54, 0.8), (949.74, 0.7), (937.80, 0.6), (930.75, 0.5), (926.23, 0.5),
     (923.15, 0.5), (920.96, 0.5), (919.35, 0.5), (918.13, 0.5), (917.18, 0.5), (916.43, 0.5),
     (915.82, 0.5), (915.33, 0.5)]
    + [(w, 0.25) for w in (977.02, 988.77, 1031.93, 1036.34, 1037.62, 1039.23, 1048.22, 1066.66,
                           1083.99, 1084.58, 1121.97, 1122.52, 1125.45, 1134.17, 1134.42, 1134.98,
                           1142.37, 1143.23, 1144.94, 951.08, 951.30, 952.30, 952.42, 953.42,
                           953.66, 953.97, 954.10, 948.69, 950.89, 1063.18, 1096.88, 1152.82)]
)


def identity(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def air_to_vacuum(air):
    vacuum = np.asarray(air, dtype=float).copy()
    for _ in range(3):
        s2 = (1e4 / vacuum) ** 2
        vacuum = air * (1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2))
    return vacuum


def read_uves(directory):
    """Inverse-variance co-add of the flux-calibrated UVES exposures of each arm."""
    arms, files = {}, []
    for path in sorted(glob.glob(str(directory / "eso" / "*.fits"))):
        header = fits.getheader(path)
        table = fits.getdata(path, 1)
        if header.get("INSTRUME") != "UVES" or "FLUX" not in table.columns.names:
            continue
        if float(header.get("SPEC_RES", 0.0)) > 30000.0:  # SPY only, not the DIC1 high-resolution set
            continue
        wave = air_to_vacuum(table["WAVE"][0].astype(float))
        flux, error = table["FLUX"][0].astype(float), table["ERR"][0].astype(float)
        arm = "blue" if wave.mean() < 4600 else "red"
        arms.setdefault(arm, []).append((wave, flux, error, float(header["SPEC_RES"])))
        files.append(path)
    segments = []
    for arm in sorted(arms):
        reference = arms[arm][0][0]
        weight = np.zeros_like(reference)
        total = np.zeros_like(reference)
        for wave, flux, error, _ in arms[arm]:
            good = np.isfinite(flux) & (error > 0) & (flux != 0)
            inside = (reference >= wave[good].min()) & (reference <= wave[good].max())
            f = np.interp(reference, wave[good], flux[good])
            e = np.interp(reference, wave[good], error[good])
            weight += np.where(inside, 1 / e**2, 0.0)
            total += np.where(inside, f / e**2, 0.0)
        flux = np.where(weight > 0, total / np.where(weight > 0, weight, 1), np.nan)
        # The outer 30 A of each arm are poorly flux-calibrated (dichroic, order ends).
        valid = reference[np.isfinite(flux)]
        flux = np.where((reference > valid.min() + UVES_EDGE) & (reference < valid.max() - UVES_EDGE), flux, np.nan)
        segments.append((reference, flux, float(np.mean([a[3] for a in arms[arm]]))))
    return segments, files


def read_sdss(directory):
    path = sorted(glob.glob(str(directory / "sdss" / "spec-*.fits")))[0]
    table = fits.getdata(path, 1)
    flux = np.where((table["ivar"] > 0) & (table["and_mask"] == 0), table["flux"], np.nan)
    return [(10 ** table["loglam"].astype(float), flux.astype(float), 2000.0)], [path]


def read_fuse(directory):
    candidates = []
    for path in glob.glob(str(directory / "fuse" / "*_vo.fits")):
        table = fits.getdata(path, 1)
        wave, flux, sigma = (table[k][0].astype(float) for k in ("WAVE", "FLUX", "SIGMA"))
        flux = np.where(sigma > 0, flux, np.nan)
        for centre, half in AIRGLOW:
            flux[np.abs(wave - centre) < half] = np.nan
        candidates.append((np.nanmedian(flux / np.where(sigma > 0, sigma, np.inf)), wave, flux, path))
    _, wave, flux, path = max(candidates, key=lambda c: c[0])
    return [(wave, flux, 20000.0)], [path]


class Model:
    def __init__(self, directory):
        self.path = Path(directory) / "spectrum.txt"
        self.wave, self.flux = np.loadtxt(self.path, usecols=(0, 1), unpack=True)
        self._cache = {}

    def convolved(self, power):
        if power not in self._cache:
            step = 1.0  # km/s
            grid = np.arange(np.log(self.wave[0]), np.log(self.wave[-1]), step / C)
            flux = np.interp(grid, np.log(self.wave), self.flux)
            sigma = C / power / 2.3548 / step
            x = np.arange(-int(5 * sigma) - 1, int(5 * sigma) + 2)
            kernel = np.exp(-0.5 * (x / sigma) ** 2)
            kernel /= kernel.sum()
            padded = np.pad(flux, len(kernel), mode="edge")
            self._cache[power] = (grid, np.convolve(padded, kernel, mode="same")[len(kernel):-len(kernel)])
        return self._cache[power]

    def at(self, wave, power, velocity):
        grid, flux = self.convolved(power)
        return np.interp(np.log(wave / (1 + velocity / C)), grid, flux)


def local_ratio(wave, data, model, lower, upper):
    keep = (wave >= lower) & (wave <= upper) & np.isfinite(data)
    if np.count_nonzero(keep) < 20:
        return None
    x = (wave[keep] - wave[keep].mean()) / max(np.ptp(wave[keep]), 1e-9)
    design = np.vstack([model[keep], model[keep] * x]).T
    coefficients, *_ = np.linalg.lstsq(design, data[keep], rcond=None)
    residual = data[keep] - design @ coefficients
    return float(np.mean(residual**2) / np.nanmedian(data[keep]) ** 2)


def fit_velocity(segments, model, windows):
    def cost(velocity):
        total = 0.0
        for wave, flux, power in segments:
            prediction = model.at(wave, power, velocity)
            for lower, upper in windows:
                value = local_ratio(wave, flux, prediction, lower, upper)
                total += 0.0 if value is None else value
        return total

    coarse = np.arange(-150.0, 151.0, 5.0)
    best = coarse[np.argmin([cost(v) for v in coarse])]
    fine = np.arange(best - 5.0, best + 5.1, 0.5)
    return float(fine[np.argmin([cost(v) for v in fine])])


def binned(wave, flux, width):
    edges = np.arange(wave.min(), wave.max() + width, width)
    index = np.digitize(wave, edges)
    good = np.isfinite(flux)
    count = np.bincount(index[good], minlength=edges.size + 1)
    total = np.bincount(index[good], weights=flux[good], minlength=edges.size + 1)
    centre = np.bincount(index[good], weights=wave[good], minlength=edges.size + 1)
    keep = count > 0
    return centre[keep] / count[keep], total[keep] / count[keep]


def normalize(wave, flux, lower, upper, bin_angstrom, degree, smooth_angstrom, gap_angstrom=10.0):
    """Upper-envelope normalization, separately for each contiguous segment.

    In each segment, take the 88th percentile of a running median in bins of
    about ``bin_angstrom``. Fit a Chebyshev polynomial to the log of that
    envelope (degree at most ``degree``) and divide by it.
    """
    out = np.full_like(flux, np.nan)
    step = np.median(np.diff(wave))
    breaks = np.flatnonzero(np.diff(wave) > max(gap_angstrom, 5 * step)) + 1
    for index in np.split(np.arange(wave.size), breaks):
        w, f = wave[index], flux[index]
        finite = np.isfinite(f) & (f > 0)
        if np.count_nonzero(finite) < 20:
            continue
        size = max(1, int(round(smooth_angstrom / step)) | 1)
        smooth = median_filter(np.where(finite, f, np.nanmedian(f[finite])), size=size, mode="nearest")
        smooth = np.where(finite, smooth, np.nan)
        lo, hi = w[finite].min(), w[finite].max()
        bins = max(2, int(round((hi - lo) / bin_angstrom)))
        edges = np.linspace(lo, hi + 1e-9, bins + 1)
        centres, envelope = [], []
        for left, right in zip(edges[:-1], edges[1:]):
            keep = (w >= left) & (w < right) & np.isfinite(smooth) & (smooth > 0)
            if np.count_nonzero(keep) >= 5:
                centres.append(np.mean(w[keep]))
                envelope.append(np.nanpercentile(smooth[keep], 88.0))
        if not centres:
            continue
        if len(centres) == 1:
            out[index] = f / envelope[0]
            continue
        fit = np.polynomial.Chebyshev.fit(centres, np.log(envelope), min(degree, len(centres) - 1),
                                          domain=(lo, hi))
        out[index] = f / np.exp(fit(w))
    return out


def gapped(wave, flux, factor=5.0):
    """Insert NaN at gaps wider than `factor` median steps so lines are not drawn across them."""
    if wave.size < 3:
        return wave, flux
    step = np.diff(wave)
    breaks = np.flatnonzero(step > factor * np.median(step))
    return np.insert(wave, breaks + 1, wave[breaks] + 0.5 * step[breaks]), np.insert(flux, breaks + 1, np.nan)


def prepare(segments, model, velocity, lower, upper, width, bins, degree, smooth):
    """Display arrays (observed, model) for one panel, both binned and normalized identically."""
    observed_wave, observed, model_flux = [], [], []
    for wave, flux, power in segments:
        keep = (wave >= lower - 20) & (wave <= upper + 20)
        if np.count_nonzero(keep) < 10:
            continue
        w, f = wave[keep], flux[keep]
        m = model.at(w, power, velocity)
        if width:
            wb, fb = binned(w, f, width)
            _, mb = binned(w, np.where(np.isfinite(f), m, np.nan), width)
        else:
            wb, fb, mb = w, f, np.where(np.isfinite(f), m, np.nan)
        observed_wave.append(wb)
        observed.append(fb)
        model_flux.append(mb)
    wave = np.concatenate(observed_wave)
    order = np.argsort(wave)
    wave, data, prediction = wave[order], np.concatenate(observed)[order], np.concatenate(model_flux)[order]
    data = normalize(wave, data, lower, upper, bins, degree, smooth)
    prediction = normalize(wave, np.where(np.isfinite(data), prediction, np.nan), lower, upper, bins, degree,
                           smooth)
    return wave, data, prediction


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "mathtext.fontset": "dejavusans", "font.size": 10,
        "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
        "axes.linewidth": .8, "xtick.direction": "in", "ytick.direction": "in",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})


def mixture(fractions):
    return "He/C/O = " + "/".join(f"{v:.3g}" for v in fractions) + " by mass"


def optical_figure(tags, models, observations, output, report):
    style()
    panel = 1.55
    height = panel * len(tags) + 0.9
    fig, axes = plt.subplots(len(tags), 1, figsize=(8.0, height), sharex=True,
                             gridspec_kw=dict(hspace=0.0, left=.105, right=.965,
                                              bottom=.55 / height, top=1 - .12 / height))
    axes = np.atleast_1d(axes)
    lower, upper = OPTICAL_RANGE
    for ax, tag in zip(axes, tags):
        name, key, instrument, teff, logg, fractions = STARS[tag]
        reader = read_uves if instrument == "UVES" else read_sdss
        segments, files = reader(observations / key)
        model = Model(models / tag)
        velocity = fit_velocity(segments, model, VELOCITY_WINDOWS["UVES"]) if instrument == "UVES" else 0.0
        width = 0.5 if instrument == "UVES" else None
        wave, data, prediction = prepare(segments, model, velocity, lower, upper, width,
                                         bins=100.0, degree=3, smooth=3.0)
        ax.plot(*gapped(wave / (1 + velocity / C), data), color=OBSERVED, lw=.5, rasterized=True)
        ax.plot(*gapped(wave / (1 + velocity / C), prediction), color=OPENWD, lw=.7)
        for line in OPTICAL_ISM:  # interstellar lines are near 0 km/s heliocentric
            ax.annotate("IS", (line / (1 + velocity / C), 1.08), ha="center", va="bottom",
                        fontsize=SECONDARY_SIZE, color=".35")
        ax.set_xlim(lower, upper)
        ax.set_ylim(0.62, 1.38)
        ax.set_yticks((0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3))
        ax.set_yticklabels(["", "0.8", "", "1.0", "", "1.2", ""])
        ax.tick_params(top=True, right=True, length=4, direction="in", labelsize=TICK_SIZE)
        ax.text(.99, .95, rf"{name} ({instrument})   $T_{{\rm eff}}={teff:.0f}\,$K, $\log g={logg:.2f}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=TICK_SIZE)
        ax.text(.99, .78, mixture(fractions), transform=ax.transAxes, ha="right", va="top",
                fontsize=SECONDARY_SIZE)
        report["optical"][tag] = dict(name=name, instrument=instrument, velocity_km_s=velocity,
                                      resolving_power=[s[2] for s in segments],
                                      model=identity(model.path), observations=[identity(f) for f in files])
    axes[0].legend(handles=[Line2D([], [], color=OBSERVED, lw=.8, label="Observed"),
                            Line2D([], [], color=OPENWD, lw=.8, label="OpenWD")],
                   loc="lower left", ncol=2, frameon=False, fontsize=TICK_SIZE, handlelength=2.3,
                   columnspacing=1.4)
    fig.supxlabel(r"Wavelength ($\mathrm{\AA}$)", fontsize=AXIS_LABEL_SIZE, y=.12 / height)
    fig.supylabel("Normalized flux", fontsize=AXIS_LABEL_SIZE, x=.015)
    for suffix, dpi in (("pdf", 400), ("png", 220)):
        fig.savefig(output / f"pg1159-optical.{suffix}", dpi=dpi)
    plt.close(fig)


def fuse_figure(tag, models, observations, output, report):
    """FUSE and optical panels of one star, in the layout of the PG 1424+535 paper figure."""
    style()
    name, key, instrument_optical, teff, logg, fractions = STARS[tag]
    model = Model(models / tag)
    fuse, fuse_files = read_fuse(observations / key)
    reader = read_uves if instrument_optical == "UVES" else read_sdss
    optical, optical_files = reader(observations / key)
    velocity = fit_velocity(fuse, model, VELOCITY_WINDOWS["FUSE"])
    optical_velocity = (fit_velocity(optical, model, VELOCITY_WINDOWS["UVES"])
                        if instrument_optical == "UVES" else 0.0)
    panels = UVES_PANELS if instrument_optical == "UVES" else SDSS_PANELS
    rows = [("FUSE", p) for p in FUSE_PANELS] + [(instrument_optical, p) for p in panels]
    panel, spacer = 1.25, .42
    height = len(rows) * (panel + spacer) + 0.6
    fig, axes = plt.subplots(len(rows), 1, figsize=(8.0, height),
                             gridspec_kw=dict(hspace=spacer / panel, left=.105, right=.965,
                                              bottom=.55 / height, top=1 - .12 / height))
    for index, (ax, (instrument, (lower, upper))) in enumerate(zip(axes, rows)):
        if instrument == "FUSE":
            wave, data, prediction = prepare(fuse, model, velocity, lower, upper, 0.05,
                                             bins=8.0, degree=2, smooth=0.3)
            rest = wave / (1 + velocity / C)
            for line, half in FUSE_ISM:
                if lower - 2 < line < upper + 2:  # interstellar lines are near 0 km/s heliocentric
                    centre = line / (1 + velocity / C)
                    ax.axvspan(centre - half, centre + half, color=ISM, lw=0, zorder=0)
            ax.set_ylim(0, 1.5)
            ax.set_yticks((0, .5, 1))
        else:
            wave, data, prediction = prepare(optical, model, optical_velocity, lower, upper,
                                             0.5 if instrument == "UVES" else None,
                                             bins=100.0, degree=3, smooth=3.0)
            rest = wave / (1 + optical_velocity / C)
            ax.set_ylim(0.62, 1.32)
            ax.set_yticks((0.7, 0.8, 0.9, 1.0, 1.1, 1.2))
        ax.plot(*gapped(rest, data), color=OBSERVED, lw=.5, rasterized=True)
        ax.plot(*gapped(rest, prediction), color=OPENWD, lw=.7)
        ax.set_xlim(lower, upper)
        ax.tick_params(top=True, right=True, length=4, direction="in", labelsize=TICK_SIZE)
        label = f"{name} ({instrument})" if index == 0 else instrument
        ax.text(.008, .95, label, transform=ax.transAxes, ha="left", va="top",
                fontsize=TICK_SIZE if index == 0 else SECONDARY_SIZE, color="0" if index == 0 else ".35")
        if index == 0:
            ax.text(.99, .95, rf"$T_{{\rm eff}}={teff:.0f}\,$K, $\log g={logg:.2f}$", transform=ax.transAxes,
                    ha="right", va="top", fontsize=TICK_SIZE)
            ax.text(.99, .78, mixture(fractions), transform=ax.transAxes, ha="right", va="top",
                    fontsize=SECONDARY_SIZE)
    axes[2].legend(handles=[Line2D([], [], color=OBSERVED, lw=.8, label="Observed"),
                            Line2D([], [], color=OPENWD, lw=.8, label="OpenWD"),
                            Patch(color=ISM, label="ISM absorption")],
                   loc="lower right", ncol=3, frameon=False, fontsize=TICK_SIZE, handlelength=2.3,
                   columnspacing=1.4)
    fig.supxlabel(r"Wavelength ($\mathrm{\AA}$)", fontsize=AXIS_LABEL_SIZE, y=.12 / height)
    fig.supylabel("Normalized flux", fontsize=AXIS_LABEL_SIZE, x=.015)
    for suffix, dpi in (("pdf", 400), ("png", 220)):
        fig.savefig(output / f"pg1159-{tag}.{suffix}", dpi=dpi)
    plt.close(fig)
    report["fuse"][tag] = dict(fuse_velocity_km_s=velocity, optical_velocity_km_s=optical_velocity,
                               model=identity(model.path),
                               observations=[identity(f) for f in fuse_files + optical_files])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stars", nargs="+", choices=list(STARS),
                        default=["pg1707", "mct0130", "pg1424", "pg1159", "abell21", "pg1144", "he1429",
                                 "rxj0122"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(scope=__doc__.strip(), plotter=identity(Path(__file__)), optical={}, fuse={})
    optical_figure(args.stars, args.models, args.observations, args.output, report)
    for tag in FUSE_STARS:
        fuse_figure(tag, args.models, args.observations, args.output, report)
    (args.output / "pg1159-figures-provenance.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
