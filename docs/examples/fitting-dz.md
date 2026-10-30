# DZ fitting example

[Metal-polluted atmospheres](../models/DZ.md) · [Getting started](../getting-started.md)

The [fitting notebook](../../examples/fit_dz_spectrum.ipynb) fits the Ca, Mg
and Fe abundances of PG 1225-079. It solves one atmosphere and then scans the
abundances on that fixed thermal structure. It needs the normal OpenWD
dependencies and Jupyter.

## Run the example

1. Start Jupyter from the repository root.
2. Open `examples/fit_dz_spectrum.ipynb`.
3. Run all cells.

The cold start (`quality="production"`) takes about 30 minutes on one core.
The notebook saves it in `results/dz-fit-pg1225` and reuses it on the next run
if the request and code are unchanged. The velocity and abundance scans take
about 15 minutes more.

To fit your own spectrum, replace the data cell. Set the resolving power, the
fixed parameters and the fit windows. Use vacuum wavelengths and positive
errors.

## Observed spectrum

The [included coadd](../../examples/data/README.md) combines four 300-second
UVES blue-arm exposures from ESO programmes 165.H-0588(A) and 167.D-0407(A).
The companion JSON file records the source URLs, SHA-256 checksums and FITS
headers.

The preparation code does these steps:

1. Convert air wavelengths to vacuum (Morton 1991).
2. Apply the barycentric correction from each header.
3. Interpolate flux and errors onto a 0.03 Å grid.
4. Scale each exposure to the first one in 4150–4400 Å.
5. Form the inverse-variance mean.

The nominal resolving power is 19,540. The weights do not include the
resampling covariance. The
[ESO release description](https://www.eso.org/sci/observing/phase3/data_releases/uves_echelle_v1.2.pdf)
gives the pipeline and its flux-calibration limits.

To rebuild the coadd, install the `validation` extra (for `astropy`). Then
run this command from the repository root:

```bash
python docs/examples/prepare_pg1225.py --raw-dir results/pg1225-raw --output results/pg1225-rebuilt.npz
```

The [preparation script](prepare_pg1225.py) downloads the archive files with
certificate verification and checks their checksums. It does not overwrite
an existing output.

## What the fit does

- Teff = 10800 K, log g = 8.0, H/He and the starting metal abundances come
  from Table 5 of [Klein et al. (2011)](https://doi.org/10.1088/0004-637X/741/1/64).
  Abundances are log10 N(Z)/N(He). Teff, log g and H/He are not fitted.
- Each trial recomputes the populations, electron density and opacity on the
  fixed structure of the starting atmosphere. It does not solve a new
  atmosphere.
- The model is broadened with a Gaussian at the nominal resolving power.
- Each window has a linear continuum. Each element has its own velocity
  shift. These shifts are nuisance parameters, not a radial velocity.
- The Ca II H and K cores are masked because the model does not reproduce
  their emission.
- The printed errors are local chi-square curvature widths. They do not
  include resampling covariance, nuisance parameters or model errors.

## Result and structure check

The fit on the starting structure, and a refit on a structure solved again
near the fitted mixture, give these log10 N(Z)/N(He) values:

| Element | Starting structure | Re-solved structure | Change (dex) |
| --- | ---: | ---: | ---: |
| Ca | −7.791 | −7.810 | −0.019 |
| Mg | −7.297 | −7.301 | −0.004 |
| Fe | −7.481 | −7.500 | −0.019 |

The structure feedback is small compared with the effect of the Teff and
log g uncertainties. For a final solution, set `RUN_FINAL_COLD_START = True`
in the notebook. It solves a new atmosphere at the fitted mixture and refits.

Klein et al. (2011) used Keck/HIRES data and different models. This example
uses UVES data, the OpenWD line list and only the detected elements.
The comparison with their abundances is therefore not a calibrated bias test.
