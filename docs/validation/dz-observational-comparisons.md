# Observational comparisons: DZ, DBZ and DAZ

[Model usage and physics](../models/DZ.md) · [Documentation home](../README.md)

These comparisons use published effective temperatures (Teff), surface
gravities and abundances. No parameter or abundance is fitted. The commands
compute new models with the current defaults. Their spectra can differ
from the figures.

## Paper comparisons

### Seven polluted white dwarfs

Five stars have helium-dominated atmospheres and use `DZConfig`. G149-28 and
GALEX J1931+0117 have hydrogen atmospheres and use `DAZConfig`. Each panel
lists the adopted abundances relative to H or He.

| Object | Host | Teff (K) | log g | Observation | Parameter source |
| --- | --- | ---: | ---: | --- | --- |
| G149-28 | H | 8600 | 8.10 | DESI DR1 | [Zuckerman et al. (2011)](https://doi.org/10.1088/0004-637X/739/2/101) |
| PG 1225-079 | He | 10800 | 8.00 | VLT/UVES | [Klein et al. (2011)](https://doi.org/10.1088/0004-637X/741/1/64) |
| WD J1013+0259 | He | 12255 | 7.90 | VLT/X-shooter | [Izquierdo et al. (2023)](https://doi.org/10.1093/mnras/stad282) |
| SDSS J0738+1835 | He | 13950 | 8.40 | SDSS | [Dufour et al. (2012)](https://doi.org/10.1088/0004-637X/749/1/6) |
| WD J0259-0721 | He | 16390 | 8.26 | VLT/X-shooter | [Izquierdo et al. (2023)](https://doi.org/10.1093/mnras/stad282) |
| Ton 345 | He | 19780 | 8.18 | SDSS | [Wilson et al. (2015)](https://doi.org/10.1093/mnras/stv1201) |
| GALEX J1931+0117 | H | 20890 | 7.90 | VLT/UVES | [Vennes et al. (2011)](https://doi.org/10.1111/j.1365-2966.2011.18323.x) |

[![Observed spectra and OpenWD models at published parameters for seven polluted white dwarfs](../assets/dz-daz-paper.png)](../assets/dz-daz-paper.pdf)

[Download the comparison (PDF)](../assets/dz-daz-paper.pdf).
Each model atmosphere includes the metals of its panel. The models are
convolved to the resolution of each observation: R = 2000 (DESI), 19,540
(PG 1225-079 UVES), 5453 (X-shooter) and 40,970 (GALEX J1931+0117 UVES), or
a Gaussian of FWHM 2.23 Å and 2.57 Å (SDSS spectra of J0738 and Ton 345). The
data and the models are normalized independently. The plots use rest-frame
vacuum wavelengths.

The models reproduce most metal features. The largest differences are the
broad Balmer lines of GALEX J1931+0117 (probably continuum normalization) and
the Ca II H and K lines of PG 1225-079, which are weaker in the model.

### Cool DZs and a cool DAZ

This figure compares five helium-dominated DZs at 4170–7940 K and one cool
DAZ with public spectra. Only the listed elements are included.

| Object | Host | Teff (K) | log g | Observation | Parameter and abundance source |
| --- | --- | ---: | ---: | --- | --- |
| WD 2251-070 | He | 4170 | 8.06 | VLT/X-shooter | [Blouin et al. (2019)](https://doi.org/10.3847/1538-4357/ab1266) Teff, log g, Ca; [O'Brien et al. (2026)](https://doi.org/10.1093/mnras/stag1669) Mg, Na |
| SDSS J0916+2540 | He | 5378 | 8.00 | SDSS | [Hollands et al. (2017)](https://doi.org/10.1093/mnras/stx250) |
| vMa2 | He | 5830 | 8.09 | VLT/X-shooter | [O'Brien et al. (2026)](https://doi.org/10.1093/mnras/stag1669) |
| SDSS J1227+6330 | He | 7418 | 8.07 | DESI DR1 | [Coutu et al. (2019)](https://doi.org/10.3847/1538-4357/ab46b9) Teff, log g; [Swan et al. (2023)](https://doi.org/10.1093/mnras/stad2867) abundances |
| WD J0255+0237 | He | 7940 | 8.03 | VLT/X-shooter | [Izquierdo et al. (2026)](https://doi.org/10.1093/mnras/stag1353) |
| WD J1336-0337 | H | 6262 | 7.94 | VLT/X-shooter | [Kilic et al. (2026)](https://doi.org/10.3847/1538-4357/ae825f): photometric Teff, log g and Ca; other metals CI-chondritic |

[![Five cool DZs and one cool DAZ compared with OpenWD at published parameters](../assets/dz-cool-paper.png)](../assets/dz-cool-paper.pdf)

[Download the comparison (PDF)](../assets/dz-cool-paper.pdf).
Each model is convolved to the instrumental resolution, shifted to the
measured radial velocity and multiplied by one constant. There is no slope or
continuum normalization, so the panels test the spectral energy distribution
and the line profiles.

- For X-shooter, the UVB arm is put on the VIS flux scale at the dichroic
  junction, from the observation alone.
- For WD J1336-0337, a quadratic fit to the ratio of the X-shooter and DESI
  spectra corrects the X-shooter continuum shape. This step does not use the
  model. The X-shooter atmospheric dispersion correctors are disabled, which
  probably causes chromatic slit losses.
- Data and models are averaged in 2 Å bins. Gaps in the data are the
  X-shooter junction and masked telluric and instrumental features.

The models reproduce the broad continuum shapes, including the strong
suppression of the blue flux of WD 2251-070 and SDSS J0916+2540, and most
metal-line strengths and widths. The largest differences:

- WD 2251-070 has a shallow depression at 4300–4700 Å that the model does not
  have. He-induced Ca I ¹S–¹D absorption at 4576 Å, which is not included, is
  a possible cause.
- The blue wings of Mg I b and Na I D in SDSS J0916+2540 are too weak.
- The red wing of Ca II H in vMa2 is too weak.

At the Montreal parameters, the OpenWD Ca II K equivalent widths of
WD J1336-0337 and two other DAZs of the same sample are 0.91–0.98 times those
of the Montreal models.

The code for the figure is in [`research/cool_dz_paper`](../../research/cool_dz_paper).
[`objects.json`](../../research/cool_dz_paper/objects.json) lists the inputs
and the archive identifiers. To compute the figure (each model takes minutes
to half an hour; run them one at a time):

```bash
python -I research/cool_dz_paper/fetch_spectra.py data/cool-dz
for key in wd2251 j0916 vma2 j1227 j0255 j1336; do
  python research/cool_dz_paper/run_model.py $key results/cool-dz/$key
done
python research/cool_dz_paper/plot_figure.py results/cool-dz data/cool-dz cool-dz.pdf
```

The fetch script downloads ten X-shooter products from the ESO archive
(about 11 MB), one SDSS DR17 spectrum and one DESI DR1 coadd row. The plot
script refuses a model that failed its convergence checks or whose
parameters differ from `objects.json`.

## Xu et al. (2019) HIRES comparisons

These comparisons use the helium-dominated stars of Figures 11–13 of
[Xu et al. (2019), AJ, 158, 242](https://doi.org/10.3847/1538-3881/ab4cee),
at the paper's parameters and detected abundances (Tables 1 and 4). The black
curves are Keck/HIRES spectra from the
[Keck Observatory Archive](https://koa.ipac.caltech.edu/). The red curves are
OpenWD models. Reported upper limits are omitted.

The models are convolved to R = 40,000 and integrated over the detector
pixels. The data and the models are normalized independently with the same
local continuum masks. The plots use rest-frame **air** wavelengths. The
model files use rest-frame **vacuum** wavelengths. Some line depths and the
Ca II cores differ from the data.

Run the commands from a source checkout. Abundances are
`log10[N(element)/N(He)]`. The listed metals replace the default mixture.
Each command saves the surface flux in `spectrum.txt` (vacuum Å,
erg s⁻¹ cm⁻² Å⁻¹), with the atmosphere and the provenance. The output is not
convolved or normalized. `--require-convergence` rejects a model that fails
its convergence checks. Use a new output directory for each run.

### WD 1232+563

`Teff = 11787 K`, `log g = 8.30`, `log N(H)/N(He) = -5.90`. The figure covers
the blue metal-line windows and Ca II H and K of Figure 11. The rest-frame
correction uses the photospheric velocity of +19.0 km/s. Al and Ni are upper
limits and are omitted.

![WD 1232+563 HIRES data and OpenWD at the Xu et al. composition](../assets/wd1232-xu2019-hires.png)

```bash
python examples/one_shot_dz.py --teff 11787 --logg 8.30 --log-h-he -5.90 \
  --abundance O=-5.14 --abundance Mg=-6.09 --abundance Si=-6.36 \
  --abundance Ca=-7.69 --abundance Ti=-8.96 --abundance Cr=-8.16 \
  --abundance Mn=-8.54 --abundance Fe=-6.45 \
  --wavelength-min 3200 --wavelength-max 8000 --wavelength-step 0.02 \
  --quality standard --require-convergence --output results/wd1232-xu2019
```

### WD 1551+175

`Teff = 14756 K`, `log g = 8.02`, `log N(H)/N(He) = -4.45`. The photospheric
velocity is +22.9 km/s. Ni is an upper limit and is omitted. The archive has
a coverage gap at Hα. The O I 7774 Å panel shows **only the model**, because
the archive red-CCD products have no wavelength calibration there.

![WD 1551+175 HIRES data and OpenWD at the Xu et al. composition](../assets/wd1551-xu2019-hires.png)

```bash
python examples/one_shot_dz.py --teff 14756 --logg 8.02 --log-h-he -4.45 \
  --abundance O=-5.48 --abundance Mg=-6.29 --abundance Al=-6.99 \
  --abundance Si=-6.33 --abundance Ca=-6.93 --abundance Ti=-8.68 \
  --abundance Cr=-8.25 --abundance Mn=-8.74 --abundance Fe=-6.60 \
  --wavelength-min 3200 --wavelength-max 8000 --wavelength-step 0.02 \
  --quality standard --require-convergence --output results/wd1551-xu2019
```

### WD 2207+127

`Teff = 14752 K`, `log g = 7.97`, `log N(H)/N(He) = -6.32`. The paper calls
this star WD 2207+121 in its tables and WD 2207+127 in Figure 13 (RA =
332.3951°, Dec = +12.3934°). The photospheric velocity is +34.5 km/s. All ten
listed metals are detections. The archive has a coverage gap at Hα. The
O I 7774 Å panel shows **only the model**.

![WD 2207+127 HIRES data and OpenWD at the Xu et al. composition](../assets/wd2207-xu2019-hires.png)

```bash
python examples/one_shot_dz.py --teff 14752 --logg 7.97 --log-h-he -6.32 \
  --abundance O=-5.32 --abundance Mg=-6.15 --abundance Al=-7.08 \
  --abundance Si=-6.17 --abundance Ca=-7.40 --abundance Ti=-8.84 \
  --abundance Cr=-8.16 --abundance Mn=-8.50 --abundance Fe=-6.46 \
  --abundance Ni=-7.55 \
  --wavelength-min 3200 --wavelength-max 8000 --wavelength-step 0.02 \
  --quality standard --require-convergence --output results/wd2207-xu2019
```

## Ultraviolet comparisons

These comparisons use HST/COS spectra of seven helium-dominated DZ/DBZ stars
and two hydrogen-dominated DAZ stars. Teff, log g and the detected abundances
are the values of the abundance papers. Reported upper limits are omitted.
Abundances are `log10[N(element)/N(He)]` for DZ/DBZ and
`log10[N(element)/N(H)]` for DAZ.

The models are convolved with the wavelength-dependent COS line-spread
function and integrated over the detector pixels. The observations are
binned over three pixels. A photospheric velocity correction aligns each
spectrum. The data and the models are normalized independently. Lyα
(1208–1235 Å), airglow and foreground absorption are excluded from the
far-UV comparison. The [figure provenance](../assets/uv-validation-provenance.json)
records the observations, velocities, continuum fits and checksums.

The largest differences are the **Si II 1260–1265 Å wings of GD 40 and
G241−6**, which are shallower than observed.

| Object | Model | Teff (K) | log g | log N(H)/N(He) | COS mode | Parameter and abundance source | Configuration key |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| GD 40 | DZ/DBZ | 15300 | 8.00 | −5.10 | G130M | [Jura et al. (2012), Table 3](https://arxiv.org/abs/1203.2885) | `gd40` |
| G241−6 | DZ/DBZ | 15300 | 8.00 | −5.90 | G130M | [Jura et al. (2012), Table 4](https://arxiv.org/abs/1203.2885) | `g241minus6` |
| PG 1225−079 | DZ/DBZ | 10800 | 8.00 | −4.05 | G185M | [Xu et al. (2013), Table 5](https://arxiv.org/abs/1302.4799) | `pg1225` |
| GD 61 | DZ/DBZ | 17280 | 8.20 | −3.89 | G130M | [Farihi et al. (2013), Supplement Table 2](https://arxiv.org/abs/1310.3269) | `gd61` |
| GD 378 | DZ/DBZ | 15620 | 7.93 | −4.48 | G130M | [Klein et al. (2021), Table 5](https://faculty.epss.ucla.edu/~eyoung/reprints/Klein_2021.pdf) | `gd378` |
| WD 1425+540 | DZ/DBZ | 14490 | 7.95 | −4.20 | G130M | [Xu et al. (2017), Table 1, Model I](https://arxiv.org/abs/1702.02868) | `wd1425` |
| G29−38 | DAZ | 11820 | 8.40 | — | G130M | [Xu et al. (2014), Table 3](https://arxiv.org/abs/1401.4252) | `g29_38` |
| GD 133 | DAZ | 12600 | 8.10 | — | G130M | [Xu et al. (2014), Table 3](https://arxiv.org/abs/1401.4252) | `gd133` |
| GD 362 | DZ/DBZ | 10540 | 8.24 | −1.14 | G185M | [Xu et al. (2013), Table 4](https://arxiv.org/abs/1302.4799) | `gd362` |

[uv-validation-configs.json](../assets/uv-validation-configs.json) gives the
complete mixtures and numerical settings. The bundled metal data do not
include Be (GD 378) and Sr (GD 362), so these detections are omitted. DZ/DBZ
models use ML2/α = 1.25. DAZ models use ML2/α = 0.8, with H₂ Lyman and Werner
lines and molecular equilibrium. WD 1425+540 uses production resolution; the
other models use standard resolution.

`examples/uv_validation_model.py` computes **one** atmosphere and spectrum.
It requires a passed convergence check and uses one numerical thread. It
saves `spectrum.txt` (vacuum Å, erg s⁻¹ cm⁻² Å⁻¹), the atmosphere and the
provenance. The wavelength grids are 1125–1460 Å (G130M) and 1785–2055 Å
(G185M), with 0.01 Å spacing. Use a new output directory for each run.

### GD 40

The model uses the final detected abundances of Jura et al. (2012), Table 3.
The figure shows the strong Si II multiplet and weaker Si II, O, Mg and Fe
lines.

```bash
python examples/uv_validation_model.py gd40 --output results/gd40-uv
```

[![GD 40 COS spectra and OpenWD at the Jura et al. composition](../assets/uv-gd40.png)](../assets/uv-gd40.pdf)

### G241−6

The model uses the final detected abundances of Jura et al. (2012), Table 4,
with oxygen from the UV column. Teff and log g are the same as for GD 40, so
the two stars compare different metal mixtures.

```bash
python examples/uv_validation_model.py g241minus6 --output results/g241minus6-uv
```

[![G241−6 COS spectra and OpenWD at the Jura et al. composition](../assets/uv-g241minus6.png)](../assets/uv-g241minus6.pdf)

### PG 1225−079

The model uses Xu et al. (2013), Table 5, including carbon. The three near-UV
strips show Si II, C I, Mg I and iron-group lines. At the strip edges, the
normalization uncertainty is up to 16%. The figure marks these regions.

```bash
python examples/uv_validation_model.py pg1225 --output results/pg1225-uv
```

[![PG 1225−079 COS spectra and OpenWD at the Xu et al. composition](../assets/uv-pg1225.png)](../assets/uv-pg1225.pdf)

### GD 61

The model uses the adopted abundances of Farihi et al. (2013), with trace
hydrogen and O, Mg, Si, S, Ca and Fe. The figure marks foreground N I, Si II
and C II lines at about 40 km/s blueward.

```bash
python examples/uv_validation_model.py gd61 --output results/gd61-uv
```

[![GD 61 COS spectra and OpenWD at the Farihi et al. composition](../assets/uv-gd61.png)](../assets/uv-gd61.pdf)

### GD 378

The model uses Klein et al. (2021), Table 5, with nitrogen from its footnote.
Be (`log N(Be)/N(He) = −11.44`) is omitted. The figure marks the N I and C II
lines of the foreground cloud.

```bash
python examples/uv_validation_model.py gd378 --output results/gd378-uv
```

[![GD 378 COS spectra and OpenWD at the Klein et al. composition](../assets/uv-gd378.png)](../assets/uv-gd378.pdf)

### WD 1425+540

The model uses **Model I** of Xu et al. (2017), with
`log N(H)/N(He) = −4.20`. Possible N I airglow at 1199.0–1201.2 Å (observed
frame) is excluded. N I 1134 Å is the nitrogen comparison.

```bash
python examples/uv_validation_model.py wd1425 --output results/wd1425-uv
```

[![WD 1425+540 COS spectra and OpenWD at the Xu et al. Model I composition](../assets/uv-wd1425.png)](../assets/uv-wd1425.pdf)

### G29−38

The DAZ model uses the detected abundances of Xu et al. (2014), with
photospheric H₂ Lyman and Werner lines. The observations combine the usable
exposures of two visits. Foreground components are excluded. The broad
residual at 1260–1283 Å depends on the local continuum.

```bash
python examples/uv_validation_model.py g29_38 --output results/g29_38-uv
```

[![G29−38 COS spectra and OpenWD with photospheric H2 at the Xu et al. composition](../assets/uv-g29_38.png)](../assets/uv-g29_38.pdf)

### GD 133

The DAZ model uses the detected O, Mg, Si and Ca abundances of Xu et al.
(2014), with H₂ Lyman and Werner lines. O I airglow at 1302–1306 Å is
excluded.

```bash
python examples/uv_validation_model.py gd133 --output results/gd133-uv
```

[![GD 133 COS spectra and OpenWD with photospheric H2 at the Xu et al. composition](../assets/uv-gd133.png)](../assets/uv-gd133.pdf)

### GD 362

The model uses the detected mixture of Xu et al. (2013), Table 4. Sr
(`log N(Sr)/N(He) = −10.42`) is omitted. The three strips cover about
1800–1841, 1903–1943 and 2008–2047 Å, with Si II, C I, Mg I and iron-group
lines. At the strip edges, the normalization uncertainty is up to 15%. The
NUVA and NUVC strips use the published velocity of 48 km/s. The NUVB strip
uses 46.3 km/s.

```bash
python examples/uv_validation_model.py gd362 --output results/gd362-uv
```

[![GD 362 COS spectra and OpenWD at the Xu et al. composition](../assets/uv-gd362.png)](../assets/uv-gd362.pdf)
