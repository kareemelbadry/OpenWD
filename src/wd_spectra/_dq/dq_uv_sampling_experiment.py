"""Deterministic continuum refinement around the current-energy DQ driver.

Retain the original UV-only diagnostic option; the released DQ driver also
refines the infrared with full_continuum=True. Existing line nodes, physics,
and convergence gates are unchanged. No past spectrum is needed.
"""

import numpy as np

from .provenance import digest


def augment_uv(wavelength, points, *, full_continuum=False):
    """Retain every input node and add a parameter-independent UV mesh."""
    if points < 2:
        raise ValueError('UV reference mesh requires at least two points')
    wave = np.asarray(wavelength, float)
    if (wave.ndim != 1 or len(wave) < 2 or np.any(~np.isfinite(wave))
            or np.any(wave <= 0) or np.any(np.diff(wave) <= 0)):
        raise ValueError('Expected finite ordered positive wavelength grid')
    reference = np.geomspace(1000., 100000., points)
    extra = reference[(reference >= wave[0]) & ((reference < 3800.) | full_continuum)
                      & (reference <= wave[-1])]
    return np.unique(np.r_[wave, extra])


def refined_material(base, points, *, full_continuum=False):
    class UVRefined(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.experiment_metadata.update(
                uv_structure_sampling_experiment=True,
                uv_structure_sampling_reference_points=points,
                uv_structure_sampling_rule='union existing grid with geomspace(1000,100000,N) below 3800 A',
                uv_structure_sampling_source_sha256=digest(__file__),
                uv_structure_sampling_requires_saved_spectrum=False)
            self.experiment_metadata.update(
                full_continuum_sampling=full_continuum,
                continuum_sampling_rule=('union existing grid with geomspace(1000,100000,N)'
                                         if full_continuum else 'UV only'))

        def structure_grid(self, seed, count):
            path = self.experiment_metadata.get('fixed_wavelength_grid')
            if path is not None:
                with np.load(path, allow_pickle=False) as saved:
                    if 'weights' in saved.files:
                        raise ValueError('UV refinement does not support externally weighted grids')
            original = super().structure_grid(seed, count)
            wave = augment_uv(original, points, full_continuum=full_continuum)
            self.saved_structure_grid = wave
            print(f'Continuum structure refinement: {len(original)} -> {len(wave)} wavelengths', flush=True)
            return wave
    return UVRefined


