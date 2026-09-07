# Performance

[Documentation home](../README.md) · [Development](README.md)

The optional C extension accelerates formal transfer, metal lines, helium
profiles, and neutral-broadened hydrogen profiles without changing their
numerical settings. Compiled hydrogen profiles use up to eight worker threads
by default because atmospheric depths are independent.

Set `OPENWD_NUM_THREADS` to a positive integer to control that work. When
running independent models in parallel processes, use `OPENWD_NUM_THREADS=1`
and consider `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` to avoid competing
thread pools. Each model needs its own output directory.

From the repository root, the bundled data-independent benchmark exercises
the dominant Balmer-opacity path:

```bash
python benchmarks/benchmark_hot_paths.py
python benchmarks/benchmark_hot_paths.py --full
OPENWD_NUM_THREADS=1 python benchmarks/benchmark_hot_paths.py --full
```

The benchmark reports a checksum with its timing. It is a diagnostic, not a
wall-clock regression gate: runtime depends on machine and model. Lowering
numerical resolution or relaxing convergence tolerances is not an equivalent
speedup; numerical and spectral checks must still pass.
