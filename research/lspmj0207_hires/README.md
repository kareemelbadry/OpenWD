# LSPM J0207+3331: Keck/HIRES and OpenWD

These scripts make the LSPM J0207+3331 figure of the
[DZ comparisons](../../docs/validation/dz-observational-comparisons.md) and
the line test of the [Sr note](../../docs/development/strontium.md). The
parameters and abundances are those of Le Bourdais & Dufour (2025,
arXiv:2510.08676): Teff = 5910 K, log g = 8.11, hydrogen host and 13 detected
metals. Nothing is fitted.

## Spectrum

`lspmj0207_hires_merged.npz` is the reduced spectrum (`WAVE_VAC_HELIO`,
`FLUX`, `ERR`; vacuum, heliocentric, relative flux). It comes from two 3600 s
Keck/HIRES exposures (2020-10-08, C5 decker, program U061), re-extracted from
the raw frames. The KOA level-1 extraction of this faint, cool star traces
only four blue-CCD orders. To rebuild it (MAKEE is required for step 3):

```bash
python -I fetch_hires.py raw                 # raw frames from KOA
python -I prep_makee.py raw/lev0 red/split   # split mosaics, combine flats and biases
MAKEE_SETUP=/path/to/setup_makee.sh bash reduce_makee.sh red   # MAKEE, one run per CCD and frame
python -I build_spectrum.py red spectrum     # response, cosmic rays, order merge
```

`build_spectrum.py` divides every order by the same order of the DC white
dwarf EGGR 180 observed on the same night, rejects cosmic rays by comparing
the two exposures, and merges all orders with inverse-variance weights on a
1.3 km/s grid.

## Models and figure

```bash
python run_model.py results/default                                  # Stout data, no Sr
python run_model.py results/sr-nist --with-sr --strong-line-data nist-asd
WINDOW_SET=docs python -I compare_hires.py lspmj0207_hires_merged.npz lspmj0207-hires.png \
  "results/default=OpenWD default (Stout atomic data, no Sr)" \
  "results/sr-nist=OpenWD with Sr and nist-asd strong-line data"
```

Each model is a cold start at standard quality (about 40 min on 8 threads)
and must pass its convergence certificate. The figure measures one line
velocity (+53.0 km/s) by χ² over 3800–5000 Å. In each window, the data are
scaled to the model times a quadratic. The windows test line contrasts, not the
pseudo-continuum level. `score_lines.py` gives the line scores of the Sr note.
