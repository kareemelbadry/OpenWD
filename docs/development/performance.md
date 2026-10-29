# Performance

[Documentation home](../README.md) · [Contributor guide](README.md)

## C extension

The optional C extension `wd_spectra._rt` accelerates these calculations:

- formal radiative transfer;
- metal-line opacity;
- helium line profiles;
- hydrogen line profiles with neutral broadening;
- D6 Stark-manifold profiles (bin deposition, impact profiles, the
  frequency-space Voigt profile and grid interpolation).

The extension uses the same numerical settings as the Python code. Without
the extension, OpenWD uses the NumPy code. Check the extension with:

```bash
python -c "from wd_spectra import compiled_backend_available; print(compiled_backend_available())"
```

The [contributor guide](README.md#c-extension) tells how to compile it.

## Threads

The compiled hydrogen profiles use worker threads, because the atmospheric
depths are independent. The default number of threads is the smaller of 8 and
the CPU count. It is never more than the number of depths.

- Set `OPENWD_NUM_THREADS` to a positive integer to change the number of
  threads.
- To run independent models in parallel processes, set
  `OPENWD_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`.
  This prevents competing thread pools.
- Give each model its own output directory.

`run_model` sets these three variables to 1 for the cool DB/DAB worker
processes. `tools/validate.py` and CI also use one thread per process.

## Benchmarks

Run the benchmarks from the repository root. Timings depend on the machine
and the model.

The Balmer-opacity benchmark needs no model data. It prints a checksum with
each timing:

```bash
python benchmarks/benchmark_hot_paths.py
python benchmarks/benchmark_hot_paths.py --full
OPENWD_NUM_THREADS=1 python benchmarks/benchmark_hot_paths.py --full
```

The Stark-manifold benchmark compares the C kernels with the NumPy code. It
needs the bundled Mg atomic data and the C extension:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmarks/benchmark_stark_profiles.py --repeat 7
```

It reports the median speedup and the largest profile difference relative to
the peak. This speedup is not the speedup of a full atmosphere calculation.
The [benchmark README](../../benchmarks/README.md) describes the solver
benchmarks.

## Rules for speed changes

- A speed change must not change the physics, the grids or the convergence
  tolerances.
- A lower numerical resolution or a looser tolerance is not a speedup.
- Measure the full structure opacity and a broad spectrum, then run the
  cold-start canaries at unchanged tolerances.
