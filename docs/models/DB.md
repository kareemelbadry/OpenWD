# DB module

The default solver has protected cold starts at **10000 and 22000 K**.
The explicit experimental dense-He workflow also converges at **5000 and
8000 K**, without making its trace-ion EOS a warm-star default. See
[tested temperatures](../tested-temperature-ranges.md) and
[reproduction commands](../../research/cool_models/README.md).

`compute_db` solves a homogeneous pure-helium LTE atmosphere. It combines a
Hummer--Mihalas/Q-MHD helium EOS, He I/II/III continuum opacity, corrected
Doppler-convolved Beauchamp25-LD He I Stark profiles, Schoening/SYNSPEC He II
profiles, Unsold neutral-He broadening, and ML2/alpha=1.25 convection.

```bash
python examples/one_shot_db.py --teff 20000 --logg 8.0 \
  --quality standard --output results/db-20000-8.0
```

Typical standard calculations take roughly 3--10 minutes on a current laptop;
cool neutral-line models can be slower. A same-parameter atmosphere checkpoint
can be supplied with `--restart-atmosphere` for formal synthesis only.
