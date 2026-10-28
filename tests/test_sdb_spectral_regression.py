"""Fixed-state sdB transfer control; never a seed for the separate cold test."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from wd_spectra import ModelData, SdBConfig
from wd_spectra.atmosphere import gray_hydrogen_helium_atmosphere
from wd_spectra.hot_nlte import population_arrays, with_departures
from wd_spectra.models import sdb


@pytest.mark.spectral
def test_hd4539_fixed_spectrum_matches_research_driver(monkeypatch):
    directory = Path(__file__).parent / 'data/sdb_regressions'
    with np.load(directory / 'hd4539-fixed-state.npz', allow_pickle=False) as saved:
        state = dict(saved)
    config = SdBConfig()
    model = sdb._nlte_model(config, ModelData.default())
    template = gray_hydrogen_helium_atmosphere(
        config.effective_temperature, config.logg, -config.log_helium_to_hydrogen,
        n_depth=state['temperature'].size)
    template = replace(template, column_mass=state['column_mass'],
                       gas_pressure=state['gas_pressure'],
                       rosseland_optical_depth=state['rosseland_optical_depth'])
    atmosphere = model.rebuild_atmosphere(template, state['temperature'])
    lte = model._rate_state(atmosphere)
    np.testing.assert_allclose(population_arrays(lte)[1], state['lte_population'], rtol=1e-10)
    populations = with_departures(lte, state['population'] / state['lte_population'])

    def forbidden(*args, **kwargs):
        pytest.fail('fixed-spectrum control must not iterate a structure or populations')

    monkeypatch.setattr(sdb, 'compute_dab', forbidden)
    monkeypatch.setattr(type(model), 'solve_populations', forbidden)
    with np.load(directory / 'hd4539-hhe.npz', allow_pickle=False) as reference:
        flux = sdb._emergent_flux(model, atmosphere, populations,
                                  reference['wavelength'], lines=True)
        np.testing.assert_allclose(flux, reference['flux'], rtol=1e-7, atol=0)
