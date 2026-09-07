# Explicit molecular-DAB starting states

These five **OpenWD-generated**, numerically qualified atmospheres use log g=8
and log10(N_H/N_He)=-2. They are not published/reference-grid spectra. They are
inputs for explicit re-relaxation with the molecular conservative-transport
recipe, never an automatic fallback for a failed model. The ordinary public
checkpoint loader invalidates experimental chemistry claims when rebuilding
with a different partition function. A fresh measured correction and physical
checks are required even when the supplied state already has small residuals.

| File | Layers | Original run under `results/molecular-dab-conservative-20260907/` in the research workspace |
| --- | ---: | --- |
| `dab-7500.npz` | 166 | `detuning-deeper-check/7500`: continuation with explicit lower-domain extension. |
| `dab-7750.npz` | 162 | `detuning-only-continuation/7750`: temperature continuation from 8000 K. |
| `dab-8000.npz` | 80 | `detuning-cold/8000`: fresh hydrostatic seed plus pseudo-time initialization. |
| `dab-9000.npz` | 80 | `detuning-upper-continued/9000`: continuation from 10000 K. |
| `dab-10000.npz` | 80 | `detuning-upper-check/10000`: final-profile re-relaxation of the earlier molecular cold start. |

Original archives are preserved byte-for-byte, including their historical
provenance metadata. Those old source paths identify how they were made;
the replay command does not require or load those paths. Local physical data
must be provided as described in the parent README. The atmosphere files do
not include opacities, external tables or literature spectra.

SHA-256:

```
e348d7a858ba1cc46313ff64aff31f271735236c7169c0577b4d54a0c3b0104a  dab-7500.npz
c709bb47cd9b905d25a50d528bb8254dd3172ef6411f98b4dfaa95b5917aa348  dab-7750.npz
1ffca134d1889dbe4454391a5cb52a1fd4f7f08b38a39877a1b58f3db8930d4a  dab-8000.npz
9ed00d07f827b8ff00caf890d28b01bd78e12a7ca42563cd828ae180da4001d4  dab-9000.npz
b6fc3d0844f1a3e23e4ffe5b01ff148bded234fef9bf9cb22c6f89a38a507bb2  dab-10000.npz
```

See [tested temperatures](../../../docs/tested-temperature-ranges.md) for
separate convergence, quadrature and physics limitations. The 7250 and 5000 K
failures are deliberately not supplied as successful checkpoints.
