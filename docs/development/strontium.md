# Strontium and NIST strong-line coverage: LSPM J0207+3331

[Contributor guide](README.md) · [DZ model guide](../models/DZ.md)

This note describes two atomic-data additions and their test on the
heavily polluted cool DAZ LSPM J0207+3331:

- strontium (Z = 38) for every LTE metal preset;
- NIST ASD strong-line files for all neutral and singly ionized ions of the
  independent-Kurucz screen, used by `strong_line_atomic_data="nist-asd"`.

The defaults do not change. A model without Sr, with the default
`strong_line_atomic_data="stout"`, reads the same atomic data as before.

## Strontium

Request `"Sr"` in the abundances, as for a Stout element. Sr follows the
[niobium](niobium.md) pattern: `read_strontium_atomic_ion` builds Sr I–V from
checksum-pinned NIST ASD 5.12 files in
`data/runtime/cache/metal-opacity/strontium/` (see
`STRONTIUM_ATOMIC_DATA_FILES`):

| Ion | Levels | Lines | Transition probabilities |
| --- | --- | --- | --- |
| Sr I | 377 | 86 | NIST ASD |
| Sr II | 70 | 31 | NIST ASD; resonance lines grade AA |
| Sr III–V | 149, 254, 143 | none | NIST has none |

Every allowed NIST line with a transition probability is attached to its
unique NIST level pair. Rows without a transition probability, and forbidden
rows, are skipped. Wavelengths are vacuum Ritz values from the level
energies; f follows from the Einstein A. The Barklem neutral-H widths of
Sr II 4078.86 and 4216.71 Å come from the bundled table. No Sr
photoionization data are bundled.

## NIST strong-line files

`strong_line_atomic_data="nist-asd"` replaces a Stout transition by the NIST
ASD value when both level energies and statistical weights match, the NIST
accuracy grade is C or better and f ≥ 0.01. This rule is unchanged. The
bundled NIST files covered only some ions (mostly Ca, Mg, Na, O and Si in the
optical). They now cover the neutral and singly ionized ions of all 15
elements of the independent-Kurucz screen (C, N, O, Na, Mg, Al, Si, P, S, Ca,
Ti, Cr, Mn, Fe, Ni), in the 115–300 nm and 300–900 nm windows of the existing
queries. For the LSPM J0207+3331 mixture, the option now replaces 3359
transitions in 20 ions.

## LSPM J0207+3331 test

The test uses a Keck/HIRES spectrum (2020-10-08, 2 × 3600 s, C5 decker),
re-extracted from the raw frames, and the parameters and abundances of
Le Bourdais & Dufour (2025): Teff = 5910 K, log g = 8.11, hydrogen host, 13
detected metals with log Sr/H = −10.85. Nothing is fitted. Each model is a
cold start at standard quality and passes its convergence certificate.

At 27 lines, the score is the inverse-variance mean of (model − data) within
±0.08 Å of the line centre, in units of the continuum, after one linear
scaling over ±3 Å:

| Line | Default | + Sr, `nist-asd` |
| --- | ---: | ---: |
| Sr II 4077.7 | +7.0σ | −1.9σ |
| Sr II 4215.5 | +9.4σ | +1.4σ |
| Ni I 3472.5 (Stout f ten times too small) | +12.3σ | +2.6σ |
| Ni I 3807.1 (Stout f eight times too small) | +12.6σ | +2.0σ |
| Fe I 3407.5 | +4.6σ | −0.5σ |
| Fe I 3684.1 | +6.5σ | −0.4σ |
| Fe I 4184.9 (Stout f 2.6 times too large) | −3.1σ | +0.8σ |
| Six well-fitted control lines | unchanged | unchanged |

The summed χ² per pixel over the 27 lines decreases from 62.6 to 38.8. Over
eleven 35–95 Å windows, no window degrades by more than 0.001 in reduced χ².

Lines that this change does not address:

- Ti II 3685.18/3685.20 Å is absent from the Stout atom. The
  independent-Kurucz rule rejects it, because the Stout and Kurucz level
  energies differ by 0.12–0.13 cm⁻¹ (above its 0.1 cm⁻¹ tolerance) and NIST
  gives no transition probability.
- Fe I 3649.30, 4177.59 and 4216.18 Å are in the Stout atom with f < 10⁻⁴,
  below the synthesis floor.
- Fe I 3556.68, 3776.45 and 4001.66 Å have NIST f < 0.01, below the
  replacement threshold.
