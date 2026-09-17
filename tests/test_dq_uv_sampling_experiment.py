import numpy as np
import pytest

from wd_spectra._dq.dq_uv_sampling_experiment import augment_uv, refined_material


def test_original_nodes_and_non_uv_are_preserved():
    original = np.array([1000., 1250., 2000., 3799., 4000., 100000.])
    refined = augment_uv(original, 4000)
    assert np.all(np.isin(original, refined))
    np.testing.assert_array_equal(refined[refined >= 3800.], original[original >= 3800.])
    assert np.all(np.diff(refined) > 0)
    np.testing.assert_array_equal(augment_uv(refined, 4000), refined)


def test_no_extension_outside_input_domain():
    refined = augment_uv([2000., 3000.], 4000)
    assert refined[0] == 2000. and refined[-1] == 3000.


def test_full_continuum_preserves_nodes_and_refines_infrared():
    original=np.array([1000.,1250.,3800.,5100.,6800.,100000.])
    refined=augment_uv(original,4000,full_continuum=True)
    assert np.all(np.isin(original,refined))
    assert np.all(np.isin(np.geomspace(1000.,100000.,4000),refined))
    np.testing.assert_array_equal(augment_uv(refined,4000,full_continuum=True),refined)
    limited=augment_uv([2000.,9000.],4000,full_continuum=True)
    assert limited[0]==2000. and limited[-1]==9000.


@pytest.mark.parametrize('wave', [[1000., 999.], [1000., np.nan], [0., 1.]])
def test_invalid_grid_rejected(wave):
    with pytest.raises(ValueError):
        augment_uv(wave, 4000)


def test_material_records_the_actual_augmented_grid():
    class Base:
        def __init__(self):
            self.experiment_metadata = {'fixed_wavelength_grid': None}
        def structure_grid(self, seed, count):
            return np.array([1000., 2000., 5000., 100000.])
    material = refined_material(Base, 4000)()
    wave = material.structure_grid(None, 300)
    np.testing.assert_array_equal(wave, material.saved_structure_grid)
    assert material.experiment_metadata['uv_structure_sampling_requires_saved_spectrum'] is False
    assert len(wave) > 4
