# Observed DZ spectrum

`pg1225-uves.npz` contains the UVES coadded spectrum of PG 1225-079. The
[fitting notebook](../fit_dz_spectrum.ipynb) uses it. You do not need to
download it.

| Array | Contents |
| --- | --- |
| `wave` | Vacuum, barycentric wavelength in Angstrom |
| `flux` | Flux density in 10^-16 erg cm^-2 s^-1 Angstrom^-1 |
| `error` | Approximate pipeline error in the same units |

The errors of neighboring pixels are correlated.

`pg1225-uves.json` records the checksum, the four source products, the
processing steps and all original FITS headers.

The data are copyright ESO and are distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
They come from programmes 165.H-0588(A) and 167.D-0407(A),
[UVES archive DOI 10.18727/archive/50](https://doi.org/10.18727/archive/50).

See the [fitting and data-preparation notes](../../docs/examples/fitting-dz.md).
