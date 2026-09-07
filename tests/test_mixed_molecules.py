import numpy as np
import pytest
from wd_spectra import eos, molecules
from wd_spectra.constants import BOLTZMANN
from wd_spectra._mixed_molecules import (
    molecular_hydrogen_helium_lte as solve,
    molecular_hydrogen_helium_enthalpy as enthalpy,
    molecular_hydrogen_helium_thermodynamics as thermo,
)
from wd_spectra._mixed_cia import H2HeCIATable, read_hitran_h2_he_cia


def test_mixed_molecular_conservation_and_mass_action():
    t = np.array([3000.0, 5000.0, 8000.0, 15000.0, 30000.0, 60000.0])[:, None, None]
    p = np.array([1e4, 1e8, 1e12])[None, :, None]
    ratio = np.array([-8.0, -2.0, 0.0, 2.0])[None, None, :]
    s = solve(t, p, ratio)
    h = s.hydrogen_lte_state
    he = s.helium_lte_state
    nh = (
        h.neutral_h_density
        + h.proton_density
        + 2 * h.molecular_hydrogen_density
        + 2 * h.molecular_hydrogen_ion_density
        + h.negative_hydrogen_density
        + 3 * h.trihydrogen_ion_density
    )
    nhe = (
        he.neutral_he_density
        + he.singly_ionized_he_density
        + he.doubly_ionized_he_density
    )
    positive = (
        h.proton_density
        + h.molecular_hydrogen_ion_density
        + h.trihydrogen_ion_density
        + he.singly_ionized_he_density
        + 2 * he.doubly_ionized_he_density
    )
    particles = (
        nh
        - h.molecular_hydrogen_density
        - h.molecular_hydrogen_ion_density
        - 2 * h.trihydrogen_ion_density
        + nhe
        + s.electron_density
    )
    np.testing.assert_allclose(nh, s.hydrogen_nuclei_density, rtol=3e-13)
    np.testing.assert_allclose(nhe, s.helium_nuclei_density, rtol=3e-13)
    np.testing.assert_allclose(
        nh / nhe, np.broadcast_to(10 ** ratio, nh.shape), rtol=3e-11
    )
    np.testing.assert_allclose(
        positive, s.electron_density + h.negative_hydrogen_density, rtol=3e-11
    )
    np.testing.assert_allclose(
        particles * BOLTZMANN * t, np.broadcast_to(p, nh.shape), rtol=3e-11
    )
    k = molecules.molecular_hydrogen_dissociation_constant(
        t, atomic_internal_partition_function=h.internal_partition_function
    )
    np.testing.assert_allclose(
        h.neutral_h_density ** 2 / h.molecular_hydrogen_density, k, rtol=3e-13
    )
    assert np.all(2 * h.molecular_hydrogen_density / nh <= 1)


def test_molecules_are_smooth_and_disappear_in_warm_limit():
    t = np.array([4999.0, 5000.0, 5001.0, 7999.0, 8000.0, 8001.0, 20000.0])
    p = np.full_like(t, 1e8)
    s = solve(t, p, -2.0)
    fraction = (
        2 * s.hydrogen_lte_state.molecular_hydrogen_density / s.hydrogen_nuclei_density
    )
    assert np.all(np.diff(fraction) < 0)
    assert fraction[-1] < 1e-6
    old = eos.hummer_mihalas_hydrogen_helium_lte(t[-1:], p[-1:], -2.0)
    np.testing.assert_allclose(s.mass_density[-1:], old.mass_density, rtol=1e-6)
    therm = thermo(t, p, -2.0)
    assert np.all(therm.specific_heat_constant_pressure > 0)
    assert np.max(abs(np.diff(therm.specific_heat_constant_pressure[:3]))) < 1e4


def test_explicit_chemical_precision_preserves_default_and_tightens_conservation():
    t=np.array([3000.,5000.,8000.,10000.,20000.])[:,None]
    p=np.array([1e8,1e10,1e12])[None,:]
    original=solve(t,p,-2.)
    explicit=solve(t,p,-2.,conservation_tolerance=2e-11)
    np.testing.assert_array_equal(original.mass_density,explicit.mass_density)
    np.testing.assert_array_equal(original.electron_density,explicit.electron_density)
    tight=solve(t,p,-2.,conservation_tolerance=2e-13)
    np.testing.assert_allclose(tight.mass_density,original.mass_density,rtol=5e-11)
    np.testing.assert_allclose(tight.hydrogen_nuclei_density/tight.helium_nuclei_density,.01,rtol=3e-13)
    h=tight.hydrogen_lte_state;he=tight.helium_lte_state
    charge=h.proton_density+h.molecular_hydrogen_ion_density+h.trihydrogen_ion_density
    charge=charge+he.singly_ionized_he_density+2*he.doubly_ionized_he_density
    np.testing.assert_allclose(charge,tight.electron_density+h.negative_hydrogen_density,rtol=3e-13)
    particles=tight.hydrogen_nuclei_density+ tight.helium_nuclei_density+tight.electron_density
    particles-=h.molecular_hydrogen_density+h.molecular_hydrogen_ion_density+2*h.trihydrogen_ion_density
    np.testing.assert_allclose(particles*BOLTZMANN*t,np.broadcast_to(p,t.shape[:-1]+p.shape[-1:]),rtol=3e-13)


@pytest.mark.parametrize('tolerance',[0.,-1.,np.nan,np.inf,1e-8])
def test_invalid_or_looser_chemical_precision_is_rejected(tolerance):
    with pytest.raises(ValueError,match='conservation_tolerance'):
        solve(5000.,1e10,-2.,conservation_tolerance=tolerance)


def test_reaction_enthalpy_response_matches_different_step():
    t = np.array([3500.0, 5000.0, 8000.0])
    p = np.array([1e9, 2.5e10, 1e9])
    delta = 1e-3
    cold, _ = enthalpy(t * np.exp(-delta), p, -2.0)
    hot, _ = enthalpy(t * np.exp(delta), p, -2.0)
    measured = (hot - cold) / (2 * t * np.sinh(delta))
    np.testing.assert_allclose(
        thermo(t, p, -2.0).specific_heat_constant_pressure, measured, rtol=2e-5
    )


def test_h2he_interpolation_units_nodes_zeros_and_declining_tail(tmp_path):
    path = tmp_path / "H2-He.cia"
    path.write_text(
        "H2-He 10 30 3 1000\n10 1e-45\n20 2e-45\n30 1e-45\n"
        "H2-He 10 30 3 2000\n10 2e-45\n20 4e-45\n30 2e-45\n"
    )
    table = read_hitran_h2_he_cia(path)
    np.testing.assert_allclose(
        table.evaluate(1e8 / np.array([10.0, 20.0, 30.0]), [1000.0, 2000.0]),
        table.coefficient_cm5,
        rtol=3e-14,
    )
    assert np.all(
        table.evaluate([1e8 / 40], [1500]) < table.evaluate([1e8 / 30], [1500])
    )
    assert table.evaluate([1e8 / 5], [1500])[0, 0] == 0
    c = table.coefficient_cm5.copy()
    c[0] = 0
    zero = H2HeCIATable(table.wavenumber, table.temperature, c, path)
    assert np.all(zero.evaluate([1e8 / 10], [1500]) == 0)
    with pytest.raises(ValueError):
        table.evaluate([0], [5000])


def test_invalid_mixed_input_raises_instead_of_returning_atomic_state():
    with pytest.raises(ValueError):
        solve(5000.0, -1.0, -2.0)
    with pytest.raises(ValueError):
        solve(np.nan, 1e8, -2.0)


def test_scalar_mixed_thermodynamics_matches_batch():
    single = thermo(5000.0, 1e10, -2.0)
    batch = thermo([5000.0], [1e10], [-2.0])
    np.testing.assert_allclose(
        single.specific_heat_constant_pressure,
        batch.specific_heat_constant_pressure[0],
        rtol=1e-10,
    )


def test_molecular_checkpoint_does_not_inherit_atomic_convergence(tmp_path):
    import json
    from wd_spectra.models import load_atmosphere_checkpoint
    from wd_spectra.spectrum import synthesize_helium_spectrum

    path = tmp_path / "atomic.npz"
    t = np.array([4000.0, 4500.0, 5000.0])
    p = np.array([1e6, 1e8, 1e10])
    np.savez(
        path,
        temperature=t,
        gas_pressure=p,
        column_mass=p / 1e8,
        rosseland_optical_depth=[1e-4, 0.1, 1.0],
        atmosphere_metadata_json=json.dumps(
            {
                "radiative_equilibrium_converged": True,
                "model_request_fingerprint": {"old": True},
            }
        ),
    )
    a = load_atmosphere_checkpoint(
        path, 5000.0, 8.0, "mixed", log_hydrogen_to_helium=-2.0, include_molecules=True
    )
    assert not a.metadata["radiative_equilibrium_converged"]
    assert "model_request_fingerprint" not in a.metadata
    assert a.metadata["mixed_chemical_model"] == "molecular-h-he-hm"
    with pytest.raises(ValueError, match="matching molecular_h_he"):
        synthesize_helium_spectrum(a, [4000.0, 5000.0], stark_table=None)


def test_h2he_opacity_population_scaling_and_no_extra_unit_factor():
    from types import SimpleNamespace
    from wd_spectra._mixed_cia import h2_he_cia_mass_absorption_coefficient

    table = H2HeCIATable(
        np.array([100.0, 200.0]),
        np.array([1000.0, 2000.0]),
        np.full((2, 2), 1e-45),
        "synthetic",
    )
    a = SimpleNamespace(
        temperature=np.array([1500.0]),
        mass_density=np.array([0.1]),
        hydrogen_lte_state=SimpleNamespace(molecular_hydrogen_density=np.array([1e20])),
        helium_lte_state=SimpleNamespace(neutral_he_density=np.array([1e22])),
    )
    np.testing.assert_allclose(
        h2_he_cia_mass_absorption_coefficient(a, [1e6], table), 0.01, rtol=1e-13
    )
    a.hydrogen_lte_state.molecular_hydrogen_density *= 2
    np.testing.assert_allclose(
        h2_he_cia_mass_absorption_coefficient(a, [1e6], table), 0.02, rtol=1e-13
    )


def test_external_molecular_data_changes_checkpoint_identity():
    from wd_spectra.models import DABConfig, ModelData
    from wd_spectra.models.common import model_request_fingerprint

    args = ("DAB", DABConfig(include_molecules=True), ModelData.default())
    one = model_request_fingerprint(*args, physical_data_identity={"cia_sha256": "one"})
    two = model_request_fingerprint(*args, physical_data_identity={"cia_sha256": "two"})
    assert one["sha256"] != two["sha256"]


def test_missing_molecular_data_fails_explicitly(tmp_path):
    from wd_spectra.models import DABConfig, compute_dab

    assert DABConfig().include_molecules is False
    with pytest.raises(FileNotFoundError, match="HITRAN"):
        compute_dab(
            DABConfig(include_molecules=True, h2_he_cia_path=str(tmp_path / "absent"))
        )
