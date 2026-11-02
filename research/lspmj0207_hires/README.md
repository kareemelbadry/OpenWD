# LSPM J0207+3331: Keck/HIRES and OpenWD

These scripts make the LSPM J0207+3331 figures of the
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
python run_model.py results/default-sr --with-sr               # current default, with Sr
python -I plot_figure.py lspmj0207_hires_merged.npz results/default-sr lspmj0207-hires
```

The model is a cold start at standard quality and must pass its convergence
certificate. `plot_figure.py` writes `lspmj0207-hires-1` and `-2` (PNG and
PDF), with the twelve windows of Figure 6 of Le Bourdais & Dufour. It
measures one line velocity (+53.0 km/s) by χ² over 3800–5000 Å. In each
window, the data are scaled to the model times a quadratic, so the windows
test line contrasts, not the pseudo-continuum level.

`compare_hires.py` overlays several models in smaller diagnostic windows,
for example the earlier Stout data against the current default:

```bash
python run_model.py results/stout --strong-line-data stout     # Stout data, no Sr
python -I compare_hires.py lspmj0207_hires_merged.npz compare.png \
  "results/stout=Stout strong-line data, no Sr" "results/default-sr=Default with Sr"
```

`score_lines.py` gives the line scores of the Sr note.
