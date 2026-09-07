# DZ/DBZ module

[Model guide](README.md) · [Getting started](../getting-started.md)

`compute_dz` solves a helium-dominated polluted LTE atmosphere at fixed input
abundances. Metals contribute electrons, bound-free opacity, sampled line
blanketing, and therefore feed back on the relaxed structure. The default GD
40 mixture uses the Stout v3.00b4 line data used for every object in the
published DZ/DAZ comparison, with levels through charge 3, Verner
photoionization, dense-helium ionization shifts, and available unified Mg
I--He and Ca I--He profiles. Observable helium and trace-hydrogen lines use
the same policies as DB and DAB. Evaluated NIST replacements for matched
strong transitions remain available as the explicit
`strong_line_atomic_data="nist-asd"` alternative. Trace-hydrogen Lyman lines
likewise retain the figure's charged-particle Stark treatment by default;
`lyman_profile_source="allard"` explicitly selects the later unified-profile
option.

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

Paper-spectrum regression and cold-start convergence are separate checks.
See [tested points](../tested-temperature-ranges.md) and
[reference-comparison limitations](../limitations.md#spectrum-accuracy-and-reference-comparisons)
for the status of PG 1225 and SDSS J0738+1835.
