from types import SimpleNamespace
import pytest
from wd_spectra.models import stellar,DABConfig
from matched_molecular_synthesis import matched_synthesis


def test_research_synthesis_explicitly_matches_without_mutating_defaults(monkeypatch):
    def fake_synthesis(*args,**options):return options
    monkeypatch.setattr(stellar,'synthesize_hydrogen_helium_spectrum',fake_synthesis)
    def compute(config,*args,**options):
        atmosphere=SimpleNamespace(effective_temperature=config.effective_temperature)
        return config,stellar.synthesize_hydrogen_helium_spectrum(atmosphere,n_angle=4),options
    runner=SimpleNamespace(compute_dab=compute)
    config=DABConfig(effective_temperature=10000,include_molecules=True)
    with matched_synthesis(runner,8):
        selected,synthesis,options=runner.compute_dab(config,relax_atmosphere=False)
        assert selected.lyman_profile_source=='stark'
        assert synthesis==dict(transfer_discretization='column-mass',n_angle=8,
                               include_helium_ii_lines=False)
        assert options['relax_atmosphere'] is False
        with pytest.raises(ValueError,match='fixed atmosphere'):runner.compute_dab(config)
    assert runner.compute_dab is compute
    assert stellar.synthesize_hydrogen_helium_spectrum is fake_synthesis
    assert config.lyman_profile_source=='allard'
