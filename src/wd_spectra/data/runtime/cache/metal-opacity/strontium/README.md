Strontium atomic data

Stout stops at Zn, so `read_strontium_atomic_ion` in `wd_spectra/metals.py`
builds Sr I--V from these checksum-pinned files (query URLs and SHA-256 in
`STRONTIUM_ATOMIC_DATA_FILES`), in the same way as Nb:

- `nist-asd-sr{1..5}-levels.tsv`: NIST ASD 5.12 energy levels, retrieved
  2026-11-01. They give the partition functions and Ritz wavelengths.
- `nist-asd-sr1-lines.tsv`, `nist-asd-sr2-lines.tsv`: all NIST lines of Sr I
  and Sr II. Every allowed (E1) row with a transition probability is attached
  to its unique NIST level pair (configuration, term, J and energy). This
  gives 86 Sr I and 31 Sr II transitions. Rows without a transition
  probability (275 Sr I, 102 Sr II) and the two forbidden Sr II rows are
  skipped. The resonance lines Sr II 4078.86 and 4216.71 A (vacuum) have
  NIST grade AA.
- `nist-asd-sr-ionization.csv`: NIST ionization energies.

NIST has no transition probabilities for Sr III, IV or V, so those stages
carry partition functions and charge but no lines. No Sr photoionization data
are bundled; Verner's fits stop at Zn. Neutral-H widths of the Sr II
resonance lines come from the bundled Barklem table.
