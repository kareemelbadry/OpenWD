import pytest
from wd_spectra import adaptive_structure as adaptive
from mass_transfer_experiment import mass_transfer_experiment


def test_scope_sets_only_entry_option_and_restores_after_error(monkeypatch):
    calls = []
    def solve(*args, **options):
        calls.append(options)
    monkeypatch.setattr(adaptive, "solve_adaptive_lte_structure", solve)
    originals = {name: getattr(adaptive, name) for name in (
        "coherent_scattering_feautrier_field", "feautrier_radiation_field",
        "optical_depth_from_mass_opacity", "cancellation_safe_field",
        "integrated_radiative_cell_energy_state_response")}
    with pytest.raises(RuntimeError):
        with mass_transfer_experiment():
            adaptive.solve_adaptive_lte_structure()
            assert calls == [{"transfer_discretization": "column-mass"}]
            assert all(getattr(adaptive, k) is v for k, v in originals.items())
            with pytest.raises(ValueError, match="conflicting transfer"):
                adaptive.solve_adaptive_lte_structure(transfer_discretization="optical-depth")
            raise RuntimeError("scope restoration probe")
    assert adaptive.solve_adaptive_lte_structure is solve
    adaptive.solve_adaptive_lte_structure()
    assert calls[-1] == {}
