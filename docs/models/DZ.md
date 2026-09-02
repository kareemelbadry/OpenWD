# DZ/DBZ module

`compute_dz` solves a helium-dominated polluted LTE atmosphere at fixed input
abundances. Metals contribute electrons, bound-free opacity, sampled line
blanketing, and therefore feed back on the relaxed structure. The default GD
40 mixture uses Stout levels through charge 3, evaluated NIST data for matched
strong transitions, Verner photoionization, dense-helium ionization shifts,
and available unified Mg I--He and Ca I--He profiles. Observable helium and
trace-hydrogen lines use the same policies as DB and DAB.

```bash
python examples/one_shot_dz.py --teff 15300 --logg 8.0 \
  --abundance O=-5.61 --abundance Mg=-6.24 --abundance Si=-6.76 \
  --abundance Ca=-6.88 --abundance Fe=-6.48 --log-h-he -6.16 \
  --quality standard --output results/gd40
```

Abundances are `log10[N(element)/N(He)]`; supplying any `--abundance` entries
replaces the entire default abundance dictionary. The model does not refit
`Teff`, `log g`, or composition. Standard line-rich atmospheres can take from
about 30 minutes to several hours. `dense_helium_eos="reos3"` is available in
the Python configuration as an explicitly experimental bulk-EOS option; the
validated production default remains the chemical-picture EOS.
