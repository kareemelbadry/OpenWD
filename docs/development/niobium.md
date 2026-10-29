# Niobium, zinc, copper and nickel lines: HS 0209+0832

[Contributor guide](README.md) · [DAB trace metals](../models/DAB.md)

This note describes three additions and their test on HS 0209+0832:

- niobium (Z = 41) for every LTE metal preset;
- trace metals in the homogeneous H/He (DAB) preset;
- optional E1 lines for Zn IV–V, Cu IV–VI and Ni IV–VI. The bundled Stout
  files do not have these lines.

Williams et al. (2026, [arXiv:2610.07161](https://arxiv.org/abs/2610.07161))
identified Nb III and Nb IV lines in the far-UV spectrum of HS 0209+0832. The
test uses their parameters and abundances without a fit.

## Niobium

Request `"Nb"` in the abundances, as for a Stout element. Nb has a Saha
ladder from Nb I to Nb VI, partition functions from the NIST levels, charge
donation and line opacity. The line widths are the thermal, radiative and
neutral-perturber widths, with optional classical electron-Stark widths.

`read_niobium_atomic_ion` builds the ions from checksum-pinned files in
`data/runtime/cache/metal-opacity/niobium/` (see `NIOBIUM_ATOMIC_DATA_FILES`):

| Ion | Levels | Lines | Transition probabilities |
| --- | --- | --- | --- |
| Nb I, II | 378, 353 | none | NIST has none |
| Nb III | 188 | 76 | HFR+CPOL gA, Nilsson et al. (2010) Table 7; only log gf > -0.5 |
| Nb IV | 182 | 819 | NIST ASD, mostly Tauheed & Reader (2005); grades B+ to E |
| Nb V, VI | 30, 104 | none | NIST has none |

Wavelengths are vacuum Ritz values from the NIST level energies. Each
Nilsson et al. row matches a level pair by both J values, parity and
wavelength, because NIST and Iglesias (1955) give different LS names to some
mixed Nb III levels. All 76 rows match one level pair within 0.08 Å.

`IONIZATION_ENERGY_EV` has the fourth and fifth NIST ionization energies of
Al, Ti, Co, Ni, Cu and Zn. Thus, hot models can use ladders to charge 4.

## Zn, Cu and Ni line supplements

The bundled Stout files for Zn IV–VI, Cu IV–VI and Ni IV–VI contain only
forbidden (M1 and E2) lines. At 35,800 K these stages hold most of each
element. Without E1 lines, they add charge and partition functions but no line
absorption.

`DABConfig(metal_line_supplements=("rauch-zn-cu", "kurucz-fe-ni"))` adds E1
lines. The default `()` uses Stout only.

- `"rauch-zn-cu"` adds the HFR lines of Rauch et al. for Zn IV–V (2014) and
  Cu IV–VI (2020). Each line attaches to an existing Stout level pair. No
  levels are added. Zn lines match by level energy (within 1 cm⁻¹). Cu lines
  match by J, parity and Ritz wavenumber (within 0.6 cm⁻¹), because the
  level-energy columns of the GAVO TOSS table are not reliable. Lines with no
  pair or more than one pair are rejected. The
  [data README](../../src/wd_spectra/data/runtime/cache/metal-opacity/rauch-zn-cu/README.md)
  gives the files and their terms of use.
- `"kurucz-fe-ni"` adds Kurucz measured-level lines of Fe IV, Fe VII and
  Ni IV–VI from `data/hot_daz/kurucz/`. The missing-transition mode of
  `read_kurucz_gf100_atomic_database` keeps all Stout lines. It adds only lines
  between level pairs that Stout does not connect, and adds the Kurucz levels
  that Stout does not have. Fe V, Fe VI and Ni VII are excluded, because some
  of their Kurucz energies differ from Stout by 0.5–3 cm⁻¹. A guard stops the
  merge if an added level is within 5 cm⁻¹ of a Stout level of equal weight.

A change of the default supplements is a physics change. It needs new frozen
controls and the approval of a maintainer.

## HS 0209+0832 test

The host is the [DAB preset with trace metals](../models/DAB.md):
Teff = 35,800 K, log g = 7.90, log H/He = 1.90, and the nine abundances of the
paper (C, Al, Si, Ca, Ti, Ni, Cu, Zn and Nb, with log Nb/H = -6.33). Metal lines
use classical electron-Stark widths. At standard quality, each model is a cold
start of about 5 min on one process and passes all five convergence checks.

Results on the structure:

- With all metals at log N/H = -20, the model reproduces the metal-free
  preset to |ΔT/T| < 7 × 10⁻⁴ below the top layer.
- The paper metals change the temperature by less than 0.2% for
  τ_R = 10⁻⁴–1, by up to 1% for τ_R > 1, and by up to 12% for τ_R < 10⁻⁴.
- Nb is mostly Nb IV (54–84%) in the photosphere, with Nb V (12–57%) and
  Nb III (0.4–4%).

The comparison uses the HST/STIS E140M HASP coadd (program 7473) and the
CalFUSE spectrum C0260201. The model is scaled by (R/d)² with R = 0.0145 R☉
and d = 82.6 pc. It is convolved with the STIS line-spread function or a
Gaussian (R = 20,000) for FUSE. One constant per panel (0.98–1.10) scales it
to the observed continuum. The red curves are the published model of
Williams et al. (2026), Figure 1.

![HS 0209+0832: the six windows of Williams et al. (2026), Figure 1](../assets/hs0209-niobium/hs0209-figure1-windows.png)

![HS 0209+0832: STIS regions with Zn IV, Cu IV and Ni IV lines](../assets/hs0209-niobium/hs0209-zn-cu-ni-windows.png)

![HS 0209+0832: the carbon windows of Williams et al. (2026), Extended Data Figure 2](../assets/hs0209-niobium/hs0209-carbon-windows.png)

- Nb reproduces the Nb IV 1434.14/1434.22 Å doublet and the FUSE Nb IV lines.
  The model is too strong for Nb IV 1002.76 Å and 1005.70 Å.
- The supplements add the Zn IV 1365.25 and 1369.51 Å, Cu IV 1367.52 Å and
  Ni IV 1452.22 Å lines. Their equivalent widths are near those of the
  published model. Both models are weaker than the observed Ni IV 1452.22 Å
  line.
- With the supplements, χ² per pixel decreases in eight of the ten windows,
  does not change in one and increases slightly in one.
- The STIS lines prefer a velocity of 82.8 km/s. Table 3 of the paper gives
  76–78 km/s. The cause of this offset is not known. The figures use 82.8 km/s.

## Reproduce

`RAW` holds the STIS E140M HASP coadd and x1d files, the CalFUSE file of
C0260201 and the STIS line-spread functions. `CURVES.npz` is optional.
`research/extract_williams2026_figure1.py` (PyMuPDF) makes it from the paper
PDF.

```bash
python research/hs0209_niobium.py model --output results/hs0209-niobium/standard
python research/hs0209_niobium.py model --line-supplements rauch-zn-cu kurucz-fe-ni \
    --output results/hs0209-niobium/supplemented
for MODEL in standard supplemented; do
  python research/hs0209_niobium.py windows --output results/hs0209-niobium/$MODEL
  python research/hs0209_niobium.py windows --window-set extended --output results/hs0209-niobium/$MODEL
done
for SET in figure1 metals carbon; do
  python research/plot_hs0209_regions.py --model results/hs0209-niobium/supplemented \
      --compare-model results/hs0209-niobium/standard --raw RAW --paper-model CURVES.npz \
      --window-set $SET --stis-velocity 82.8 \
      --output results/hs0209-niobium/figures/hs0209_${SET}_v82p8_kms
done
```

`--control metal-free` and `--control negligible-metals` compute the two
control models. `plot_hs0209_regions.py` writes the χ² values and continuum
scales to a JSON file next to each figure.

## Limitations

- There are no Nb photoionization data. The thresholds of the populated Nb
  ions are below 500 Å, so the far-UV continuum does not change.
- Nb III has only its stronger lines (log gf > -0.5). Nb II and Nb V have no
  lines. Thus, cool stars, where Nb II dominates, show no Nb lines.
- All lines use LTE populations.
- Rauch lines to theoretical levels and all Cu VII lines are not included.
  Rauch lines use classical widths. Kurucz lines use Kurucz widths.
