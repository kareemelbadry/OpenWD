# Tested points

[Documentation home](README.md) · [Limitations](limitations.md) · [Model guide](models/README.md)

These are the parameters at which a fresh cold start passed all convergence
checks. A cold start begins from a gray or continuum atmosphere and uses no
saved model. The points are tested points, not guaranteed intervals. Accuracy
between and outside them is not known.

Points are given as effective temperature (Teff, K) / log g. "Canary" means
that `python tools/validate.py cold --case <name>` repeats the cold start in
the protected CI workflow. The "paper range" is the range of the paper's
comparisons.

## DA

Paper range: 3000–40,000 K. See the [DA guide](models/DA.md).

| Points | Settings | Evidence |
| --- | --- | --- |
| 3000, 4000, 5000, 20000 / 8.0 | production | Canaries `da-3000`, `da-4000`, `da-5000`, `da-20000` |
| 6000, 8000, 10000, 12000, 14000 / 8.0 | production | Cold starts, 8–31 min |
| 4250/6.0, 4500/7.0, 4750/7.75, 5000/7.25, 6500/6.0, 3000/9.5 | production | Cold starts |
| 30000/5.0, 32000/6.0, 34000/6.25, 38000/6.0 | production | Cold starts, one lower-boundary extension |
| G 76−48: 6680/7.96, convection off | production and standard | Canaries `da-radiative-g76-48`, `da-radiative-g76-48-standard` |

2500/9.0 does not pass the all-depth flux check.

## DB

Paper range: up to about 40,000 K, with a separate cool-DB comparison. See the
[DB guide](models/DB.md).

| Points | Settings | Evidence |
| --- | --- | --- |
| 10000, 22000 / 8.0 | `compute_db`, production; 22000 K also standard | Canaries `db-10000`, `db-22000`, `db-22000-standard` |
| 12000/6.0, 13000/8.0, 16000/9.0, 17000/6.0, 18000/7.0, 20000/8.0, 27500/8.125, 30000/6.25, 32000/6.0, 32000/6.25, 34000/6.0, 34000/6.5, 36000/6.5, 36000/7.5, 38000/7.5, 40000/6.0, 40000/7.0, 40000/7.75, 45000/7.0, 55000/6.5, 60000/6.25 | default `transfer_discretization="auto"`, production | Cold starts, 3–29 min |
| 7000/6.0, 7500/6.5, 8000/6.5, 9000/7.0, 9500/6.0 | column-mass discretization (selected by `"auto"` here), production | Cold starts, up to 2.5 h |
| Dense helium: 4500, 5000, 5500, 6000, 6500, 7000, 8000 / 8.0 | `run_model`, production | Cold starts with an independent spectrum audit; canaries `dense-db-5000`, `dense-db-8000` |

## DAB and DBA

Paper range: 9000–30,000 K. See the [DAB/DBA guide](models/DAB.md).

| Points | Settings | Evidence |
| --- | --- | --- |
| 7500, 8000, 9000, 10000 / 8.0, log N(H)/N(He) = −2 | `run_model` (molecular closure), production | Cold starts; canaries `molecular-dab-7500`, `molecular-dab-9000`, `molecular-dab-10000` |
| 12000/8.0, 12000/9.0, 15000/9.0, log N(H)/N(He) = 0 | `run_model` (molecular closure), production | Cold starts |
| 20000/8.0, log N(H)/N(He) = −2 | `compute_dab` (atomic), production | Canary `dab-20000` |

7250 K and 5000 K do not pass.

## DZ, DBZ and DAZ

Paper range: 4000–25,000 K. See the [DZ/DBZ/DAZ guide](models/DZ.md).

| Points | Settings | Evidence |
| --- | --- | --- |
| DZ, PG 1225−079: 10800/8.0 | production | Canary `dz-pg1225` |
| DAZ, G149-28: 8600/8.10 | standard | Canary `daz-g149-28` |
| DAZ, G29-38: 11820/8.40 | standard | Canary `daz-g29-38` |
| DAZ, GALEX J1931+0117: 20890/7.90 | standard | Canary `daz-galex1931` |
| Cool DZs and one cool DAZ: 4000–8000 K | published parameters | Paper comparison |

Each point uses the metal composition of that star.

## DQ

See the [DQ guide](models/DQ.md). Each DQ run must also pass the independent
218,520-point spectrum check.

| Points | Settings | Evidence |
| --- | --- | --- |
| J1235: 9347/8.041, log N(C)/N(He) = −4.107 | standard | Canary `dq-j1235` |
| J1225: 6294/7.924, log N(C)/N(He) = −5.33 | standard | Cold start (paper comparison) |

The other DQ paper models (4400–7256 K) started from saved atmospheres. They
are spectral comparisons, not cold-start evidence.

## DO and DAO

Paper range: 40,000–90,000 K. See the [DO/DAO guide](models/DO-DAO.md).

| Points | Settings | Evidence |
| --- | --- | --- |
| DO: 50000, 55000, 60000, 70000, 90000, 120000 / 8.0 | standard | Cold starts; canary `do-50000` |
| DAO: 40000/8.0 (log N(H)/N(He) = 0); 60000/8.0 and 100000/8.0 (log N(H)/N(He) = 2) | standard | Cold starts; canary `dao-60000` |
| Hot DA, GD 153: 40204/7.82, log N(H)/N(He) = 6 | standard; production with H8 and H20 | Cold starts |
| Trace metals, G191-B2B: 52500/7.53, N(He)/N(H) = 1e-5 | standard and production | Cold start of host and metals |

## PG 1159

Paper range: 75,000–200,000 K. See the [PG 1159 guide](models/PG1159.md).

| Points | Settings | Evidence |
| --- | --- | --- |
| PG 1707+427: 85000/7.5 | standard | Cold start, `spectrum-qualified` |
| PG 1424+535: 110000/7.0 | standard | Cold start, `spectrum-qualified`; canary `pg1159-pg1424` (He/C/O only) |
| PG 1159-035: 140000/7.0 | standard, `include_radiative_acceleration=True` | Cold start, `spectrum-qualified` |

## D6

Paper range: 10,000–30,000 K. See the [D6 guide](models/D6.md).

| Points | Settings | Evidence |
| --- | --- | --- |
| SDSS J1637+3631: 15680/6.3 | standard | Canary `d6-j1637` |
| J1235−3752: 18625/7.0 | standard | Cold start |

## DAH

Paper range: 10,500–29,316 K, dipole polar fields 6.13–437.1 MG. See the
[DAH guide](models/DAH.md).

| Points | Settings | Evidence |
| --- | --- | --- |
| J1007+1237: 18687/8.04, 6.13 MG | production | Canary `dah-j1007+1237` |
| J1254+5612: 12870/8.58, 60.43 MG | production | Canary `dah-j1254+5612` |

The certificate applies to the nonmagnetic structure. The RWA
photoionization is tested to about 200 MG.

## sdB

Paper range: 20,000–40,000 K, log g 5–6. See the [sdB guide](models/sdB.md).

| Points | Settings | Evidence |
| --- | --- | --- |
| HD 4539: 23200/5.20, log N(He)/N(H) = −2.27 | standard | Canary `sdb-hd4539` |
