"""Dense-He chemical equilibrium with explicit, REQUIRED species data.

He, He+, He2+ (molecular singly charged), e, in a neutral-dominated dense bath.
Bulk rho and chemical potentials must be supplied by a documented provider.
This module does not invent the unavailable He2+--He excess potential.
Definitions follow Kowalski's 2006 thesis, section 4.2.3; no He++ is included.
"""
from dataclasses import dataclass
import numpy as np
from wd_spectra import eos
from wd_spectra.constants import BOLTZMANN, HELIUM_MASS
from .dense_helium_fluid import EV, KB_EV


@dataclass(frozen=True)
class ExcessPotentials:
    """Excess chemical potentials, eV/particle; molecular entry is mandatory."""
    neutral: object
    atomic_ion: object
    molecular_ion: object
    electron: object
    source: str

    def reaction_shifts(self):
        if not self.source or any(v is None for v in (
                self.neutral, self.atomic_ion, self.molecular_ion, self.electron)):
            raise ValueError('all four chemical potentials and their source are required')
        atom, ion, dimer, electron = np.broadcast_arrays(*[
            np.asarray(v, float) for v in (self.neutral, self.atomic_ion,
                                          self.molecular_ion, self.electron)])
        if any(np.any(~np.isfinite(v)) for v in (atom, ion, dimer, electron)):
            raise ValueError('chemical potentials must be finite')
        return ion+electron-atom, dimer-atom-ion


@dataclass(frozen=True)
class DenseHeliumChemistry:
    neutral: np.ndarray
    atomic_ion: np.ndarray
    molecular_ion: np.ndarray
    electron: np.ndarray
    log_ionization_constant: np.ndarray
    log_dissociation_constant: np.ndarray
    maximum_relative_charge_residual: float
    maximum_relative_nuclei_residual: float
    chemical_potential_source: str


def solve_solute_chemistry(temperature, density, potentials, bound_neighbours, *, neutral_partition=1.):
    """He/He+/e balance when the He+ solute already contains its bound He2+ pairs.

    The He+ excess potential of the HNC solute uses the full bonding-channel
    Mayer function, so its population includes ions bound to a neighbour; an
    explicit He2+ species would count those pairs twice. Here the ionization
    follows the single solute reaction He <-> He+ + e, and the ions are then
    split by the HNC bound-neighbour count N_b: a fraction 1 - exp(-N_b)
    (Poisson) has a bound partner and is reported as He2+, for its
    photodissociation opacity. The bound partner leaves the neutral pool, so
    nuclei and charge are conserved. As n -> 0, N_b -> n/K(He2+) and ideal
    association is recovered.
    """
    t, rho = np.broadcast_arrays(np.asarray(temperature, float), np.asarray(density, float))
    if (np.any(~np.isfinite(t)) or np.any(~np.isfinite(rho)) or np.any(rho <= 0)
            or np.any(t <= 0)):
        raise ValueError('chemical kernel requires positive finite T and density')
    ion_shift, _ = potentials.reaction_shifts()
    nb = np.asarray(bound_neighbours, float)
    t, rho, ion_shift, nb = np.broadcast_arrays(t, rho, ion_shift, nb)
    if np.any(~np.isfinite(nb)) or np.any(nb < 0):
        raise ValueError('bound-neighbour count must be finite and non-negative')
    q0 = np.asarray(neutral_partition, float)
    if np.any(~np.isfinite(q0)) or np.any(q0 < 1):
        raise ValueError('invalid neutral partition function')
    log_s = (np.log(4./q0)+1.5*np.log(2*np.pi*eos.ELECTRON_MASS*BOLTZMANN*t/eos.PLANCK**2)
             -(eos.HELIUM_FIRST_IONIZATION_ENERGY/EV+ion_shift)/(KB_EV*t))
    nuclei = rho/HELIUM_MASS
    # n_i^2 = S (n - n_i), solved without cancellation in the trace limit.
    s = np.exp(log_s)
    ions = 2*nuclei/(1+np.sqrt(1+4*nuclei/s))
    if np.any(ions/nuclei > 1e-3):
        from .dense_helium_limits import DenseHeliumDomainError
        raise DenseHeliumDomainError('dense trace-ion chemical model exceeded 0.1% ionization')
    bound_fraction = -np.expm1(-nb)
    molecular = ions*bound_fraction
    atomic = ions-molecular
    neutral = nuclei-atomic-2*molecular
    charge_error = float(np.max(np.abs((atomic+molecular)/ions-1)))
    nuclei_error = float(np.max(np.abs((neutral+atomic+2*molecular)/nuclei-1)))
    with np.errstate(divide='ignore'):
        log_k = np.log(atomic*neutral)-np.log(molecular)
    return DenseHeliumChemistry(neutral, atomic, molecular, ions, log_s, log_k,
                                charge_error, nuclei_error, potentials.source)
