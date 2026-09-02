# Third-party notices

The BSD-3-Clause license in `LICENSE` covers the OpenWD source code. The
scientific data distributed with the package remain independently authored
works and retain their original terms and attribution.

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
