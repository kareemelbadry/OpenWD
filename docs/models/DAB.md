# DAB/DBA module

The explicit molecular conservative-transport workflow is numerically
qualified down to **7500 K** at log g=8 and log10(N_H/N_He)=-2. This is a
continuation result, not a guaranteed cold-start range; 7250 and 5000 K
remain unqualified. See [tested temperatures](../tested-temperature-ranges.md)
and [reproduction commands](../../research/cool_models/README.md).

`compute_dab` defaults to one homogeneous atomic H/He layer in LTE; it is not a
stratified thin-hydrogen-layer calculation. Hydrogen and helium share the
charge-neutrality solution and nonideal occupation-probability EOS. The module
combines the DA hydrogen opacity/profile treatment with the DB helium profiles
and uses ML2/alpha=1.25 convection.

```bash
python examples/one_shot_dab.py --teff 20000 --logg 8.0 --log-h-he -2 \
  --quality standard --output results/dab-20000-8.0
```

`--log-h-he` means `log10[N(H)/N(He)]`. Standard calculations typically take
3--20 minutes, with cool mixtures and low-gravity production models slower.
Molecular H/He chemistry is not yet included, so the coolest mixtures should
be treated cautiously.
