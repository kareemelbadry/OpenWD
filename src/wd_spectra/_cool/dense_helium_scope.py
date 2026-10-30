"""Dense-He table/bulk EOS base and the scope that installs it in the DB runner.

REOS.3 supplies bulk density and caloric quantities; an HNC interaction table
supplies excess chemical potentials. All out-of-domain requests fail. There
is no alternate EOS or electron floor.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.interpolate import RectBivariateSpline
import wd_spectra.eos as eos
from .dense_helium_limits import DenseHeliumDomainError

PHYSICS = ('EXPERIMENTAL atomic REOS3 + HNC neutral/He+ insertion + thesis electron '
           'insertion; He2+ chemistry/bound-free ABSENT; collective He- free-free '
           'and refraction ABSENT; not validated dense-He atmosphere physics')
TABLE_VERSION = 'atomic-hnc-young1981-bruno2010-v1'


class AtomicDenseEOS:
    table_version = TABLE_VERSION
    species_count = 2
    physics = PHYSICS
    dimer_description = 'free-free only; molecular bound-free absent'
    output_stem = 'atomic-dense'

    def __init__(self, bulk, interaction_path):
        self.bulk = bulk
        self.path = Path(interaction_path)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with np.load(self.path, allow_pickle=False) as table:
            if str(table['version']) != self.table_version:
                raise ValueError('incompatible atomic HNC interaction table')
            self.temperature = table['temperature'].copy()
            self.density = table['density'].copy()
            mu = table['mu'].copy()
            # Cold isotherms whose fluid solve stops below the top density.
            self.maximum_valid_density = (table['maximum_valid_density'].copy()
                                          if 'maximum_valid_density' in table.files else None)
        if (self.temperature.ndim != 1 or self.density.ndim != 1
                or min(len(self.temperature),len(self.density)) < 4
                or np.any(~np.isfinite(self.temperature)) or np.any(~np.isfinite(self.density))
                or np.any(self.temperature <= 0) or np.any(self.density <= 0)
                or np.any(np.diff(self.temperature) <= 0) or np.any(np.diff(self.density) <= 0)
                or mu.shape != (len(self.temperature),len(self.density),self.species_count)
                or np.any(~np.isfinite(mu))):
            raise ValueError('invalid HNC interaction table contents')
        # Interpolate mu/rho: its finite second-virial limit avoids loss of
        # dilute accuracy over many orders of magnitude in density.
        self.curves = [RectBivariateSpline(np.log(self.temperature), np.log(self.density),
            mu[..., i]/self.density[None, :], s=0) for i in range(self.species_count)]

    def excess_potentials(self, temperature, density):
        t, rho = np.broadcast_arrays(np.asarray(temperature, float), np.asarray(density, float))
        if (np.any(~np.isfinite(t)) or np.any(~np.isfinite(rho))
                or np.any(t < self.temperature[0]) or np.any(t > self.temperature[-1])
                or np.any(rho < self.density[0]) or np.any(rho > self.density[-1])):
            raise DenseHeliumDomainError(f'atomic HNC table domain violation: T={t.min():g}..{t.max():g}, '
                             f'rho={rho.min():g}..{rho.max():g}; no substitution')
        if self.maximum_valid_density is not None:
            # Bicubic support spans two isotherms on each side; continued
            # (unconverged) nodes are kept two density nodes away.
            limit = self.maximum_valid_density
            node = np.clip(np.searchsorted(self.temperature, t, side='right')-1, 0, len(limit)-1)
            low, high = np.maximum(node-1, 0), np.minimum(node+3, len(limit))
            allowed = np.array([limit[a:b].min() for a, b in zip(np.ravel(low), np.ravel(high))]).reshape(np.shape(node))
            margin = (self.density[-1]/self.density[-2])**2
            allowed = np.where(allowed < self.density[-1], allowed/margin, allowed)
            if np.any(rho > allowed):
                i = np.unravel_index(np.argmax(rho > allowed), np.shape(rho))
                raise DenseHeliumDomainError(
                    f'dense He fluid solve unconverged above {allowed[i]:.4g} g/cm3 at T={t[i]:.6g} K '
                    f'(requested {rho[i]:.4g}); no substitution')
        terms = [rho*curve.ev(np.log(t), np.log(rho)) for curve in self.curves]
        return terms

    def bulk_state(self, temperature, pressure):
        rho, energy, inside = self.bulk.evaluate(pressure, temperature)
        if not np.all(inside):
            raise DenseHeliumDomainError('REOS3 domain violation; no substitution')
        return rho, energy

    def thermodynamics(self, temperature, pressure, **options):
        # As in the neutral-dominated REOS + trace-chemistry construction,
        # the tabulated bulk energy is not amended with a second ideal EOS.
        if hasattr(self.bulk, 'thermodynamics'):
            return self.bulk.thermodynamics(temperature, pressure)
        t, p = np.broadcast_arrays(np.asarray(temperature, float), np.asarray(pressure, float))
        eps = 2e-4
        cold_rho, cold_u = self.bulk_state(t*np.exp(-eps), p)
        hot_rho, hot_u = self.bulk_state(t*np.exp(eps), p)
        rho, _ = self.bulk_state(t, p)
        cp = ((hot_u+p/hot_rho)-(cold_u+p/cold_rho))/(2*t*np.sinh(eps))
        expansion = -np.log(hot_rho/cold_rho)/(2*eps)
        gradient = p*expansion/(rho*t*cp)
        if (np.any(~np.isfinite(cp)) or np.any(cp <= 0) or np.any(expansion <= 0)
                or np.any(~np.isfinite(gradient))):
            raise ValueError('invalid REOS thermal derivatives')
        return eos.HeliumThermodynamics(cp, expansion, gradient)


def transport_surface_boundary(runner, model, teff, logg, **options):
    """Construct just the Eddington surface seed, not an unnecessary gray column.

    The old seed computes a whole gray atmosphere to use only its first point,
    probing arbitrarily low P and high T outside this EOS's domain. Here solve
    P*kappa_R/g=tau_min at the same Eddington boundary temperature. This is an
    approximate INITIAL condition, not a converged atmosphere or EOS fallback.
    """
    from scipy.optimize import brentq
    tau = getattr(model, 'surface_rosseland_depth', 1e-8)
    temperature = teff*(.75*(tau+2/3))**.25
    upper = np.searchsorted(model.bulk.temperature_grid, temperature)
    if upper == 0 or upper == len(model.bulk.temperature_grid):
        raise ValueError('surface temperature outside REOS support')
    pressure_min = max(model.bulk.pressure_by_temperature[j][0] for j in (upper-1, upper))
    pressure_min = max(pressure_min, getattr(model.bulk, 'p_min', pressure_min))

    def at_pressure(logp):
        p, t = np.array([np.exp(logp)]), np.array([temperature])
        point = runner.atmosphere_at(teff, p, t, np.array([tau]), logg=logg)
        opacity = runner.rosseland_mean_helium_continuum_opacity(point, n_frequency=160)[0]
        return np.log(p[0]*opacity/(10**logg*tau)), point

    lower = np.log(pressure_min*(1+1e-8))
    if at_pressure(lower)[0] >= 0:
        raise ValueError('requested surface depth lies below REOS support')
    upper = lower+1
    while at_pressure(upper)[0] < 0:
        upper += 1
    logp = brentq(lambda lp: at_pressure(lp)[0], lower, upper, xtol=1e-11)
    return at_pressure(logp)[1]


@contextmanager
def atomic_dense_experiment(runner, model, *, opacity_factory, least_squares_merit=False):
    from wd_spectra import atmosphere, convection, helium
    from wd_spectra.models.common import _jsonable
    original_compute = runner.compute_db
    original_run = runner.run
    original_solver = runner.radiative_equilibrium_helium_atmosphere

    def run(args):
        args.experimental_physics = model.physics
        if hasattr(model,'sha256'):
            args.experimental_material_certificate_version=2
            args.experimental_material_table_sha256=model.sha256
        args.experimental_exact_dense_materials = getattr(model,'exact_dense_materials',False)
        args.experimental_dense_options = getattr(model,'experiment_options',{})
        args.experimental_positive_radiative_rates = getattr(model,'positive_radiative_rates',False)
        args.experimental_source_sha256 = getattr(model,'source_sha256',{})
        return original_run(args)

    def solve(*args, **kwargs):
        a = original_solver(*args, **kwargs)
        return replace(a, metadata={**a.metadata,
            'eos': model.physics, 'bulk_helium_eos': model.bulk.source,
            'helium_dimer_ion_continuum': model.dimer_description,
            'experimental_nonlinear_merit': 'RMS' if least_squares_merit else 'production RMS + 0.25 max',
            'experimental_dense_helium': model.physics, 'interaction_table_sha256': model.sha256})

    def compute(*args, **kwargs):
        a = kwargs['initial_atmosphere']
        kwargs['initial_atmosphere'] = replace(a, metadata={**a.metadata,
            'experimental_dense_helium': model.physics, 'interaction_table_sha256': model.sha256})
        return original_compute(*args, **kwargs)

    def save(result, path):
        path = Path(path)
        a = result.atmosphere
        extra = {}
        if hasattr(model,'chemistry'):
            chemistry=model.chemistry(a.temperature,a.mass_density)
            extra['experimental_molecular_ion_density']=chemistry.molecular_ion
            extra['experimental_atomic_ion_density']=chemistry.atomic_ion
        np.savez_compressed(path/f'experimental-{model.output_stem}-structure.npz',
            experimental_logg=a.logg, experimental_effective_temperature=a.effective_temperature,
            experimental_temperature=a.temperature, experimental_pressure=a.gas_pressure,
            experimental_density=a.mass_density, experimental_electron_density=a.electron_density,
            experimental_column_mass=a.column_mass, experimental_tau=a.rosseland_optical_depth,**extra)
        columns = [result.spectrum.wavelength_angstrom, result.spectrum.surface_flux_lambda]
        if result.spectrum.continuum_flux_lambda is not None:
            columns.append(result.spectrum.continuum_flux_lambda)
        np.savetxt(path/'experimental-spectrum.txt', np.column_stack(columns), header=model.physics)
        (path/'experimental-metadata.json').write_text(json.dumps(dict(
            effective_temperature=a.effective_temperature, logg=a.logg,
            physics=model.physics, interaction_table_sha256=model.sha256,
            compatible_with_production_checkpoint_loader=False,
            atmosphere=_jsonable(a.metadata)), indent=2)+'\n')

    with ExitStack() as stack:
        # The standalone research runner installs proposal dispatchers by
        # assignment. Restore those too if this scope is used in a longer
        # lived Python process, including an exception during a run.
        from wd_spectra import adaptive_structure
        for name in ('solve_adaptive_lte_structure', 'solve_trust_region_newton'):
            stack.enter_context(patch.object(adaptive_structure, name,
                                             getattr(adaptive_structure, name)))
        if least_squares_merit:
            from wd_spectra import nonlinear
            # Match the cheap least-squares proposal's objective. All actual
            # flux/local-energy/step convergence gates are unchanged.
            stack.enter_context(patch.object(nonlinear, '_residual_merit',
                lambda r: float(np.sqrt(np.mean(np.asarray(r)**2)))))
        for module in (eos, atmosphere, runner):
            stack.enter_context(patch.object(module, 'hummer_mihalas_helium_lte', model.lte))
        for module in (eos, convection, runner):
            stack.enter_context(patch.object(module, 'hummer_mihalas_helium_thermodynamics', model.thermodynamics))
        absorption = opacity_factory(helium.helium_continuum_mass_absorption_coefficient)
        for module in (helium,runner):
            stack.enter_context(patch.object(module,'helium_continuum_mass_absorption_coefficient',absorption))
        stack.enter_context(patch.object(runner, 'compute_db', compute))
        stack.enter_context(patch.object(runner, 'save_model_result', save))
        stack.enter_context(patch.object(runner, 'run', run))
        stack.enter_context(patch.object(runner, 'radiative_equilibrium_helium_atmosphere', solve))
        stack.enter_context(patch.object(runner, 'helium_continuum_atmosphere',
            lambda *args, **kwargs: transport_surface_boundary(runner, model, *args, **kwargs)))
        yield
