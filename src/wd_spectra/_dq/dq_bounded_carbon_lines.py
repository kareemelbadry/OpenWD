"""DQ-only LTE carbon screening with an explicit maximum-opacity bound.

The shared pseudo-Voigt width is at least its Gaussian FWHM. Both mixture
components have peaks below 1/(sqrt(2*pi)*sigma_thermal). Multiplying by the
unreduced LTE lower population bounds a line at every wavelength. Discard
only line/layer pairs whose summed bound is below 1e-10 of the continuum.
This is a numerical error budget, not a temperature/abundance cutoff.
"""
from collections import defaultdict
import numpy as np

from .dq_exact_opacity_cache import subset
from .dq_hornkohl_consistent import anchored_grid
from wd_spectra import metals
from wd_spectra.constants import BOLTZMANN, PLANCK, LIGHT_SPEED, PI, ELEMENTARY_CHARGE_ESU, ELECTRON_MASS


class BoundedCarbonLines:
    def __init__(self,database,keys,anchors):
        self.database,self.keys,self.anchors=database,tuple(keys),anchors
        requested=set(keys)
        records={}
        for ion in database.ion_stages('C'):
            levels={level.index:level for level in ion.levels}
            for line in ion.transitions:
                key=('C',ion.charge,line.lower_index,line.upper_index)
                if key in requested and line.transition_type=='E1':
                    if key in records:
                        raise ValueError('Ambiguous carbon transition key')
                    lower=levels[line.lower_index]
                    records[key]=(line.wavelength_vacuum_angstrom,line.absorption_oscillator_strength,
                        lower.energy_wavenumber,lower.statistical_weight,ion.atomic_mass_u)
        if set(records)!=requested:
            raise ValueError('Missing declared carbon transition')
        self.parameters=np.asarray([records[key] for key in self.keys]).T
        self.charges=np.asarray([key[1] for key in self.keys])

    def upper_bounds(self,a,carbon):
        center,f,energy,g,mass=self.parameters
        t=a.temperature[None,:]
        population=np.empty((len(center),a.n_depth))
        for charge in np.unique(self.charges):
            use=self.charges==charge
            population[use]=(carbon.ion_number_density['C'][charge][None,:]
                /carbon.partition_function[('C',int(charge))][None,:]
                *g[use,None]*np.exp(-energy[use,None]*PLANCK*LIGHT_SPEED/(BOLTZMANN*t)))
        sigma=center[:,None]*np.sqrt(BOLTZMANN*t/(mass[:,None]*1.66053906892e-24))/LIGHT_SPEED
        stimulated=-np.expm1(-PLANCK*LIGHT_SPEED/(center[:,None]*1e-8*BOLTZMANN*t))
        prefactor=PI*ELEMENTARY_CHARGE_ESU**2/(ELECTRON_MASS*LIGHT_SPEED)
        return (prefactor*f[:,None]*population*stimulated/a.mass_density[None,:]
            *(center[:,None]**2*1e-8/LIGHT_SPEED)/(np.sqrt(2*PI)*sigma))

    def evaluate(self,a,carbon,wavelength,background,*,relative_tolerance=1e-10):
        floor=np.min(np.asarray(background),axis=0)
        if (not 0<relative_tolerance<=1e-8 or floor.shape!=(a.n_depth,)
            or np.any(~np.isfinite(floor)) or np.any(floor<=0)):
            raise ValueError('A strictly positive continuum/error budget is required')
        bounds=self.upper_bounds(a,carbon)
        skip=bounds<=relative_tolerance*floor[None,:]/len(self.keys)
        omitted=np.sum(np.where(skip,bounds,0.),axis=0)
        groups=defaultdict(list)
        for j in range(a.n_depth):
            groups[tuple(np.flatnonzero(~skip[:,j]))].append(j)
        padded,indices=anchored_grid(wavelength,self.anchors)
        result=np.zeros((len(wavelength),a.n_depth))
        for selected,columns in groups.items():
            if not selected:
                continue
            aa=subset(a,columns,a.n_depth)
            cc=subset(carbon,columns,a.n_depth,axis=-1)
            keys=tuple(self.keys[i] for i in selected)
            result[:,columns]=metals.metal_line_mass_absorption_coefficient(aa,padded,self.database,cc,
                minimum_oscillator_strength=1e-4,maximum_lines=None,transition_keys=keys)[indices]
        return result,dict(relative_bound=float(np.max(omitted/floor)),
            retained_line_layers=int(np.count_nonzero(~skip)),total_line_layers=int(skip.size),
            omitted_bound=omitted)
