"""Research certificates; operator tests live in OpenWD/tests/test_mass_feautrier.py."""
import pytest


def test_depth_certificate_must_not_mix_transfer_discretizations():
    from compare_dense_depth_resolution import validate_discretization
    with pytest.raises(ValueError, match='same transfer'):
        validate_discretization({'atmosphere': {}},
            {'atmosphere': {'experimental_mass_conservative_transfer': True}})
    with pytest.raises(ValueError, match='quadratures'):
        validate_discretization({'atmosphere': {'experimental_structure_n_angle': 8}},
            {'atmosphere': {'experimental_structure_n_angle': 16}})
