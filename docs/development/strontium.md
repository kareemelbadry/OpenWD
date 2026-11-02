# Strontium and NIST strong-line coverage: LSPM J0207+3331

[Contributor guide](README.md) · [DZ model guide](../models/DZ.md)

This note describes two atomic-data additions and their test on the
heavily polluted cool DAZ LSPM J0207+3331:

- strontium (Z = 38) for every LTE metal preset;
- NIST ASD strong-line files for all neutral and singly ionized ions of the
  independent-Kurucz screen, used by `strong_line_atomic_data="nist-asd"`,
  which is the DZ and DAZ default since 2026-11-02.

`strong_line_atomic_data="stout"` restores the earlier atomic data. The
documentation figures were computed with it unless stated otherwise.

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
ASD value when both level energies (within 0.15 cm⁻¹) and statistical weights
match, the NIST accuracy grade is C or better and f ≥ 10⁻⁴, the floor of the
synthesis line list. Unmatched NIST lines are not added. (Before 2026-11-02,
DAZ models replaced only lines with f ≥ 0.01.) The
bundled NIST files covered only some ions (mostly Ca, Mg, Na, O and Si in the
optical). They now cover the neutral and singly ionized ions of all 15
elements of the independent-Kurucz screen (C, N, O, Na, Mg, Al, Si, P, S, Ca,
Ti, Cr, Mn, Fe, Ni), in the 115–300 nm and 300–900 nm windows of the existing
queries. For all 15 elements, the rule replaces 6649 transitions in 29 ions.
The median change is 0.5%; 384 transitions change by more than a factor of
two.

### Default since 2026-11-02

`DZConfig` and `DAZConfig` default to `"nist-asd"`, and the DZ and DAZ physics
revisions include `nist-asd-strong-lines-2026-11-02`. The fixed-spectrum
controls `dz-pg1225`, `dz-j0738`, `daz-g149_28` and `daz-galex1931` were
recaptured on their frozen structures with
`research/capture_nist_asd_default_controls.py`, after it checked that the
earlier setting still reproduces the earlier controls. Integrated flux changes
by at most 0.08% in any band. The largest single-pixel changes are −67% at
O I 1303.0 Å (`dz-j0738`) and +29% at 1240 Å (`dz-pg1225`) in the UV, and
2.6% in the optical DAZ controls.

## LSPM J0207+3331 test

The test uses a Keck/HIRES spectrum (2020-10-08, 2 × 3600 s, C5 decker),
re-extracted from the raw frames, and the parameters and abundances of
Le Bourdais & Dufour (2025): Teff = 5910 K, log g = 8.11, hydrogen host, 13
detected metals with log Sr/H = −10.85. Nothing is fitted. Each model is a
cold start at standard quality and passes its convergence certificate.

At 27 lines, the score is the inverse-variance mean of (model − data) within
±0.08 Å of the line centre, in units of the continuum, after one linear
scaling over ±3 Å:

| Line | `"stout"`, no Sr | Current default + Sr |
| --- | ---: | ---: |
| Sr II 4077.7 | +7.0σ | −1.9σ |
| Sr II 4215.5 | +9.4σ | +1.3σ |
| Ni I 3472.5 (Stout f ten times too small) | +12.3σ | +2.6σ |
| Ni I 3807.1 (Stout f eight times too small) | +12.6σ | +1.8σ |
| Fe I 3407.5 | +4.6σ | −0.5σ |
| Fe I 3684.1 | +6.5σ | −0.4σ |
| Fe I 4184.9 (Stout f 2.6 times too large) | −3.1σ | +0.8σ |
| Fe I 3776.5 (Stout f too large) | −4.1σ | +0.9σ |
| Fe I 4001.7 (Stout f too large) | −3.4σ | −0.3σ |
| Six well-fitted control lines | unchanged | unchanged |

The summed χ² per pixel over the 27 lines decreases from 62.6 to 56.9 with
Sr alone and to 37.5 with Sr and the NIST data (38.8 with the earlier DAZ
replacement limit f ≥ 0.01, which left Fe I 3776.5 and 4001.7 Å at their
Stout values). Over
eleven 35–95 Å windows, the summed reduced χ² decreases from 10.50 to 9.88, and no window degrades by more than 0.002.

## Other DZ stars

Every star below was computed with both `"stout"` and `"nist-asd"` at the
published parameters and abundances. Every model is a cold start at standard
quality and passes its convergence certificate.

Xu et al. (2019) HIRES stars (WD 1232+563, WD 1551+175, WD 2207+127): the
mean χ² per pixel over the panels of the
[comparison figures](../validation/dz-observational-comparisons.md#xu-et-al-2019-hires-comparisons)
changes by at most 0.001, and no panel changes by more than 0.015. The
largest model changes are Ti II 3261.6 and 3444.3 Å and Si II 3862.6 and
5041.0 Å, which become weaker by up to 0.1 of the continuum, and Ti II/Fe II
4549.6 Å, which becomes deeper by up to 0.2. Where HIRES covers them, the
data do not prefer either model. The abundances of Xu et al. were derived
with their own oscillator strengths, so these windows do not test the
atomic data independently.

Cool DZ and DAZ stars of the
[cool-DZ comparison](../validation/dz-observational-comparisons.md): the
change of the model flux, as an rms over 20 Å bins, is:

| Star | 3000–4000 Å | 4000–9000 Å | Largest line change |
| --- | ---: | ---: | --- |
| WD 2251−070 | 2.6% | ≤ 1.0% | 4% |
| vMa2 | 0.2% | ≤ 0.1% | 2% |
| SDSS J1227+6330 | 0.6% | ≤ 0.3% | 11% |
| WD J0255+0237 | 0.2% | ≤ 0.1% | 13% |
| WD J1336−0337 | 0.2% | ≤ 0.1% | 67% (Ni I 3472.5 Å) |

WD 2251−070 is the only star with a broad change: its near-UV flux is lower
by 2.6% with the NIST data. Against the observed spectra, the score of the
comparison figure (rms of log data/model over 2 Å bins) is:

| Star | `"stout"` | `"nist-asd"` |
| --- | ---: | ---: |
| WD 2251−070 | 0.082 | 0.081 |
| vMa2 | 0.068 | 0.068 |
| SDSS J1227+6330 | 0.104 | 0.104 |
| WD J0255+0237 | 0.029 | 0.029 |
| WD J1336−0337 | 0.034 | 0.034 |

WD J0255+0237 converged in 89 iterations with the NIST data and in 19
without it.

SDSS J0916+2540 converges slowly with either atomic data set. Its structure
residual stays near 0.14 for about 30 iterations before it decreases. A run
of the release commit of the cool-DZ comparison (`783908a`) follows the same
trajectory, so this is not a regression.

Lines that this change does not address:

- Ti II 3685.18/3685.20 Å is absent from the Stout atom. The
  independent-Kurucz rule rejects it, because the Stout and Kurucz level
  energies differ by 0.12–0.13 cm⁻¹ (above its 0.1 cm⁻¹ tolerance) and NIST
  gives no transition probability.
- Fe I 3649.30, 4177.59 and 4216.18 Å are in the Stout atom with f < 10⁻⁴,
  below the synthesis floor.
- Fe I 3556.68 Å remains too strong (−2.0σ) with the NIST value.
