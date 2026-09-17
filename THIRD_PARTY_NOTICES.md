# Third-party notices

The BSD-3-Clause license in `LICENSE` covers the OpenWD source code. The
scientific data distributed with the package remain independently authored
works and retain their original terms and attribution.

The checkout-only cool-model research workflows additionally use external
HITRAN H2-He CIA and ExoMol RACPPK H2 state data. Those files are not
redistributed in this checkpoint. Follow the source attribution and data
terms linked in [the research data instructions](research/cool_models/README.md).
The small bundled HNC cache and DAB checkpoints there are OpenWD-generated
research outputs, not published reference spectra; their source models,
provenance and limitations are documented alongside them.

## Allard neutral-hydrogen/proton Lyman profiles

The temperature-dependent Ly-alpha, Ly-beta, and Ly-gamma tables in
`src/wd_spectra/data/runtime/allard_data` were supplied by Nicole Allard and
are redistributed in OpenWD with the author's permission. Scientific uses
should cite the applicable Allard line-profile calculations. The historical
`laquasi.dat`, `lbquasi.dat`, and `lgquasi.dat` files are the versions
distributed with TLUSTY 205.

## Tremblay--Bergeron hydrogen Stark profiles

The Doppler-convolved Lyman, Balmer, Paschen, and Brackett profile tables in
`src/wd_spectra/data/stark` are licensed under CC BY 4.0. Cite Tremblay &
Bergeron (2009), ApJ, 696, 1755.

## Stout atomic line data

The Stout Atomic Line List files are copyright Peter van Hoof, Royal
Observatory of Belgium, and licensed under CC BY 4.0. The original license is
retained beside the tables in
`src/wd_spectra/data/runtime/cache/metal-opacity/atomic-line-list/stout`.

## Other scientific tables

The DQ constitutive data in `src/wd_spectra/data/dq` include derived ExoMol
12C2 8states cross sections and partition functions, Hornkohl/Parigger Swan
line records, and an OpenWD-computed dense-helium correction grid. They are
scientific input tables, not model atmospheres or observed spectra.
Original data authors retain their rights; these data are not relicensed as
OpenWD source code. Checksums and processing provenance are in `manifest.json`,
`report.json`, and the NPZ metadata. Scientific use should credit:

- ExoMol 8states C₂ data (Yurchenko et al.), including the updated state energies
  identified in the opacity-table provenance.
- Parigger et al. (2015), DOI [10.1016/j.sab.2015.02.018](https://doi.org/10.1016/j.sab.2015.02.018).
  The supplied dimensional strength convention was not recovered: OpenWD uses
  one rotationless-band calibration to Brooke's `A(0,0)=7.626e6 s^-1`, preserves
  relative strengths, and does not fit stellar spectra to set this scale.
- Brooke et al. (2013), [arXiv:1212.2102](https://arxiv.org/abs/1212.2102).
- Cooper (1979), [NASA TM-78574](https://ntrs.nasa.gov/citations/19790013711),
  and Nicholls (1965), [Franck–Condon factors, Table 6](https://pmc.ncbi.nlm.nih.gov/articles/PMC6716003/),
  for the historical C₂ C–A estimate in `c2-ca-historical.npz`. OpenWD uses
  the measured squared electronic moment 0.93 atomic units (reported
  uncertainty 0.18), assumes it constant across the included bands, and
  combines the published band constants with ExoMol lower-state populations.
  Its finite-bin rigid-rotor envelope is an approximation, not a reproduction
  of a modern rotational line list or a validated dense-helium pressure profile.
- Iglesias et al. (2002), DOI [10.1086/340689](https://doi.org/10.1086/340689),
  and Blouin et al. (2018), DOI [10.3847/1538-4357/aad4a9](https://doi.org/10.3847/1538-4357/aad4a9),
  for dense-continuum/refractivity prescriptions. Numerical interpolation and
  extrapolation choices remain OpenWD approximations documented in the data.

The derived ExoMol file `c2-8states-r15000.npz` is distributed under
[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/),
following the [ExoMol data licence](https://www.exomol.com/data/licence/).
OpenWD's changes comprise cross-section binning with updated state-energy
differences, explicit finite-state partition sums, separation of the Swan
component, and rotational-overlap redistribution; these contributions to
that data file use the same CC BY-SA 4.0 licence. Credit Yurchenko et al.
(2018, MNRAS 480, 3397) and McKemmish et al. (2020, MNRAS 497, 1081).
This data licence does not replace the separate BSD licence for OpenWD code.

The additional `c2-ca-historical.npz` table also uses CC BY-SA 4.0: it derives
its populations and partition function from those ExoMol data. Credit both
ExoMol papers above and Cooper/Nicholls. OpenWD's additional processing is
the historical-band envelope integration; `tools/build_dq_ca_historical.py`
and `tools/package_dq_ca.py` reproduce the physical arrays without stellar
observations. The NPZ provenance retains the original research description
and input hashes; numerical release qualification does not remove its stated
spectroscopic approximations.

The runtime data directory also contains evaluated NIST ASD strong-line data,
Verner et al. photoionization fits, CHIANTI Ca II collision strengths,
Beauchamp He I and Schoening/SYNSPEC He II profiles, Becker et al. He-REOS.3,
and published atomic/molecular continuum tables. Source identifiers and
checksums are recorded in the corresponding OpenWD readers. Redistribution of
these numerical data does not place them under the OpenWD source-code license.

## Korg.jl Stancil (1994) table transcription

The numerical Stancil (1994) H2+/He2+ opacity-table transcription used as
the source for `src/wd_spectra/helium_molecular.py` is adapted from Korg.jl.

Copyright (c) 2021, Adam Wheeler. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Original scientific data: P. C. Stancil, *Continuous absorption by He2+ and
H2+ in cool white dwarfs*, ApJ 430, 360 (1994), DOI 10.1086/174411.
