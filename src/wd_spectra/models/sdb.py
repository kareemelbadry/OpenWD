"""Hot-subdwarf (sdB) hybrid models: LTE H/He structure, NLTE H/He populations.

The structure is a fresh public DAB (LTE, metal-free) calculation at the
requested parameters. The restricted hot H/He NLTE atom is then solved on that
fixed temperature and gas-pressure structure, with the LTE H/He electron
density, and the emergent spectrum is formed from those populations. This is
the hybrid recipe of ``research/sdb_lte_hhe.py`` and
``research/sdb_hybrid_nlte.py``; trace metals remain a research driver
(``research/sdb_trace_metals.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal
import warnings

import numpy as np

from ..atmosphere import gray_hydrogen_helium_atmosphere
from ..helium_collisions import read_tlusty_helium_collision_data
from ..helium_i_atom import read_tlusty_helium_i_atom
from ..hot_nlte import transfer_field
from ..spectrum import Spectrum
from .common import (
    AtmosphereConvergenceWarning, ModelData, ModelResult, Quality,
    atmosphere_with_model_request_fingerprint, model_request_fingerprint,
    numerical_resolution, validate_wavelength,
)
from .hot import DAOConfig, _model_from_config
from .stellar import DABConfig, compute_dab


@dataclass(frozen=True)
class SdBConfig:
    """Parameters of the hybrid sdB model.

    ``log_helium_to_hydrogen`` is log10 N(He)/N(H), the usual sdB convention
    (the opposite sign to ``DABConfig.log_hydrogen_to_helium``). The defaults
    of the NLTE atom and iteration follow the published paper models.
    """

    effective_temperature: float = 23200.
    logg: float = 5.2
    log_helium_to_hydrogen: float = -2.27
    quality: Quality = 'standard'
    hydrogen_levels: int = 16
    helium_i_atom: Literal['14', '24'] = '14'
    population_maximum_iterations: int = 600
    population_tolerance: float = 1e-4


def validate_config(config):
    if not isinstance(config, SdBConfig):
        raise TypeError('expected SdBConfig')
    for name in ('effective_temperature', 'logg', 'log_helium_to_hydrogen'):
        if not np.isfinite(getattr(config, name)):
            raise ValueError(f'{name} must be finite')
    if config.effective_temperature <= 0:
        raise ValueError('effective_temperature must be positive')
    if config.quality not in ('standard', 'production'):
        raise ValueError('sdB models support quality "standard" or "production"')
    numerical_resolution(config.quality)
    if config.helium_i_atom not in ('14', '24'):
        raise ValueError('helium_i_atom must be "14" or "24"')
    _nlte_config(config)


def _nlte_config(config):
    nlte = DAOConfig(effective_temperature=config.effective_temperature, logg=config.logg,
                     log_hydrogen_to_helium=-config.log_helium_to_hydrogen,
                     quality=config.quality, maximum_hydrogen_level=config.hydrogen_levels,
                     population_maximum_iterations=config.population_maximum_iterations,
                     population_tolerance=config.population_tolerance)
    from .hot import validate_config as validate_nlte
    validate_nlte(nlte)
    return nlte


def _nlte_model(config, data):
    model = replace(_model_from_config(_nlte_config(config), data),
                    helium_conservation_row='dominant')
    if config.helium_i_atom == '24':
        atom = data.tlusty_atoms / 'he1.dat'
        model = replace(model, helium_i_atom=read_tlusty_helium_i_atom(atom),
                        helium_i_collision_data=read_tlusty_helium_collision_data(data.tlusty_source, atom))
    return model


def _emergent_flux(model, atmosphere, populations, wavelength, *, lines, chunk=8000):
    flux = np.empty_like(wavelength)
    for start in range(0, wavelength.size, chunk):
        piece = wavelength[start:start + chunk]
        coefficients = model.transfer_coefficients(atmosphere, piece, populations, lines=lines)
        _, field, _ = transfer_field(atmosphere, coefficients, n_angle=model.n_angle,
                                     check_source=lines)
        flux[start:start + chunk] = field.interface_flux[:, 0]
    return flux


def compute_sdb(config=SdBConfig(), wavelength=None, *, data=None, initial_atmosphere=None,
                iteration_callback=None):
    """Construct a fresh hybrid sdB model and its spectrum and continuum.

    Public models are cold starts. The default wavelength grid is the paper's
    3000--9200 A vacuum grid at 0.05 A. The result is ``converged`` only when
    the LTE structure is certified and the NLTE populations converge.
    """
    validate_config(config)
    if initial_atmosphere is not None:
        raise ValueError('sdB public models require a cold start')
    wave = (np.arange(3000.0, 9200.0, 0.05) if wavelength is None
            else validate_wavelength(wavelength))
    data = ModelData.default() if data is None else data
    structure_config = DABConfig(effective_temperature=config.effective_temperature, logg=config.logg,
                                 log_hydrogen_to_helium=-config.log_helium_to_hydrogen,
                                 quality=config.quality)
    # Only the structure is used; synthesize the LTE spectrum on a token grid.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AtmosphereConvergenceWarning)
        structure = compute_dab(structure_config, np.array([4000.0, 5000.0]), data=data,
                                initial_atmosphere=None, iteration_callback=iteration_callback)
    structure_status = structure.metadata['atmosphere_convergence_status']
    lte = structure.atmosphere
    model = _nlte_model(config, data)
    template = gray_hydrogen_helium_atmosphere(
        config.effective_temperature, config.logg, -config.log_helium_to_hydrogen,
        n_depth=lte.n_depth)
    template = replace(template, column_mass=lte.column_mass, gas_pressure=lte.gas_pressure,
                       rosseland_optical_depth=lte.rosseland_optical_depth)
    atmosphere = model.rebuild_atmosphere(template, lte.temperature)
    populations = model.solve_populations(atmosphere, model._rate_state(atmosphere),
                                          accelerated_lambda=True)
    flux = _emergent_flux(model, atmosphere, populations, wave, lines=True)
    continuum = _emergent_flux(model, atmosphere, populations, wave, lines=False)
    status = ('converged' if structure_status == 'converged' and populations.converged
              else 'unconverged')
    if status != 'converged':
        warnings.warn(
            f'sdB model is not converged (LTE structure: {structure_status}; NLTE populations '
            f'converged: {bool(populations.converged)}); exploratory outputs only',
            AtmosphereConvergenceWarning, stacklevel=2)
    fingerprint = model_request_fingerprint('sdB', config, data)
    atmosphere = atmosphere_with_model_request_fingerprint(
        replace(atmosphere, metadata={**lte.metadata, **atmosphere.metadata}), fingerprint)
    spectrum = Spectrum(wave, flux, {
        'flux_unit': 'erg s^-1 cm^-2 Angstrom^-1', 'flux_convention': 'surface F_lambda',
        'wavelength_medium': 'vacuum', 'composition': 'sdB',
        'line_formation': 'restricted H/He NLTE on a fixed LTE H/He structure',
        'transfer': 'column-mass Feautrier',
        'continuum_flux': ('NLTE H/He bound-free and free-free with the converged populations, '
                           'dissolved-level pseudo-continuum, electron scattering; all lines removed'),
    }, continuum)
    return ModelResult('sdB', atmosphere, spectrum, config, {
        'preset': 'sdB-hybrid-LTE-structure-NLTE-HHe-v1', 'experimental': True,
        'cold_start': True, 'metals': False,
        'structure_preset': 'public DAB (LTE, metal-free)',
        'structure_convergence_status': structure_status,
        'structure_model_request_fingerprint': structure.metadata.get('model_request_fingerprint'),
        'nlte_populations_converged': bool(populations.converged),
        'nlte_population_iterations': int(populations.iterations),
        'nlte_population_change': float(populations.maximum_relative_population_change),
        'hydrogen_levels': config.hydrogen_levels, 'helium_i_atom': config.helium_i_atom,
        'accelerated_lambda': True, 'helium_conservation_row': 'dominant',
        'atmosphere_convergence_status': status, 'model_request_fingerprint': fingerprint,
        'independent_grid_validation': False, 'full_physics_validation': False,
    }, population_state=populations)
