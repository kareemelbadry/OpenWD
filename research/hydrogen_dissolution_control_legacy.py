"""Recover the pre-2026-10-27 SYNSPEC-style hydrogen dissolution cutoff.

Research/test scope only. Before 2026-10-27 the DAM pseudo-continuum kept the
full dissolved fraction only blueward of the transition to i + 3 and only
its neutral-perturber part from there to the first series member, in both the
ordinary and the magnetic (RWA) continuation. That discarded the dissolved
strength of the Tremblay--Bergeron Balmer/Paschen/Brackett profiles and left a
discontinuity at H-gamma. The two functions below are the origin/main
(56093d8) implementations verbatim, evaluated in their original module
namespaces. Current public models must never select this scope.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from types import FunctionType
import sys
from unittest.mock import patch

from wd_spectra import magnetic_continuum, opacity

# The verbatim functions below are rebound to their own modules' globals.


def hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    *,
    lower_levels: tuple[int, ...] = (1, 2, 3, 4),
) -> FloatArray:
    """Return series-wide DAM/HM dissolved-level opacity in cm^2 g^-1.

    For each populated lower level ``i``, a photon redward of its series limit
    is mapped to the fictitious upper level ``n*`` of Daeppen, Anderson &
    Mihalas (1987).  The full dissolved fraction
    ``1 - w(n*) / w(i)`` is retained through the transition to ``i + 3``, the
    default pseudo-continuum cutoff used by SYNSPEC for explicit hydrogen
    levels.  Between that cutoff and the first series member, only the
    neutral-perturber part of the dissolved fraction is retained.  This
    separates the long neutral-H tail, which is absent from the charged-only
    Tremblay--Bergeron Stark profiles, from charged dissolution that would
    otherwise become an artificial background beneath the first two members
    of a series.  Lyman, Balmer, Paschen, and Brackett series are included by
    default.  Charged and neutral perturbers use the same Q-MHD/HM occupation
    probabilities as the EOS.  The Lyman
    redistribution is cut off at 925 A by default: the first three Lyman
    profiles are supplied as ideal profiles, and extrapolating the raw DAM
    term across the whole Ly-alpha interval transfers far too much strength.

    The hydrogenic cross section is continued from the ordinary threshold of
    each lower level only across these finite intervals.  This deliberately
    avoids the unphysical indefinite optical/infrared extension of the raw
    DAM prescription discussed by Kowalski (2006).
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be a finite positive 1D array")

    if not lower_levels or any(level < 1 for level in lower_levels):
        raise ValueError("lower_levels must contain positive integers")
    opacity = np.zeros((wavelength.size, atmosphere.n_depth), dtype=np.float64)
    distribution = _atmosphere_level_distribution(
        atmosphere, max(40, max(lower_levels))
    )
    lyman_limit = (
        PLANCK * LIGHT_SPEED / HYDROGEN_IONIZATION_ENERGY * 1.0e8
    )
    lyman_frequency = HYDROGEN_IONIZATION_ENERGY / PLANCK
    for lower_level in lower_levels:
        series_limit = lyman_limit * lower_level**2
        charged_red_limit = series_limit / (
            1.0 - (lower_level / (lower_level + 3.0)) ** 2
        )
        first_line = series_limit / (
            1.0 - (lower_level / (lower_level + 1.0)) ** 2
        )
        red_limit = first_line
        if lower_level == 1:
            red_limit = min(red_limit, 925.0)
        valid = (wavelength > series_limit) & (wavelength < red_limit)
        if not np.any(valid):
            continue

        selected = wavelength[valid]
        effective_level = lower_level / np.sqrt(
            1.0 - series_limit / selected
        )
        upper_probability = hydrogen_occupation_probability(
            atmosphere.neutral_h_density[np.newaxis, :],
            atmosphere.electron_density[np.newaxis, :],
            atmosphere.temperature[np.newaxis, :],
            effective_level[:, np.newaxis],
            neutral_he_density=(
                atmosphere.helium_lte_state.neutral_he_density[np.newaxis, :]
                if atmosphere.helium_lte_state is not None
                else 0.0
            ),
            neutral_radius_scale=(
                atmosphere.hydrogen_lte_state.neutral_radius_scale
                if atmosphere.hydrogen_lte_state is not None
                else HM_NEUTRAL_HYDROGEN_RADIUS_SCALE
            ),
            helium_neutral_radius_scale=(
                atmosphere.helium_lte_state.neutral_radius_scale
                if atmosphere.helium_lte_state is not None
                else 0.5
            ),
            correlated_microfields=(
                atmosphere.hydrogen_lte_state is not None
                and atmosphere.hydrogen_lte_state.microfield_model == "qmhd"
            ),
        )
        lower_probability = distribution.occupation_probability[
            :, lower_level - 1
        ]
        total_survival = np.clip(
            upper_probability / lower_probability[np.newaxis, :],
            0.0,
            1.0,
        )
        correlated_microfields = (
            atmosphere.hydrogen_lte_state is not None
            and atmosphere.hydrogen_lte_state.microfield_model == "qmhd"
        )
        upper_charged_probability = (
            charged_particle_hydrogen_occupation_probability(
                atmosphere.electron_density[np.newaxis, :],
                effective_level[:, np.newaxis],
                (
                    atmosphere.temperature[np.newaxis, :]
                    if correlated_microfields
                    else None
                ),
            )
        )
        lower_charged_probability = (
            charged_particle_hydrogen_occupation_probability(
                atmosphere.electron_density,
                float(lower_level),
                atmosphere.temperature if correlated_microfields else None,
            )
        )
        charged_survival = np.clip(
            upper_charged_probability
            / lower_charged_probability[np.newaxis, :],
            0.0,
            1.0,
        )
        full_dissolved_fraction = 1.0 - total_survival
        neutral_dissolved_fraction = np.clip(
            charged_survival - total_survival,
            0.0,
            1.0,
        )
        dissolved_fraction = np.where(
            (selected < charged_red_limit)[:, np.newaxis],
            full_dissolved_fraction,
            neutral_dissolved_fraction,
        )
        frequency = LIGHT_SPEED / (selected[:, np.newaxis] * 1.0e-8)
        stimulated_emission = -np.expm1(
            -PLANCK
            * frequency
            / (BOLTZMANN * atmosphere.temperature[np.newaxis, :])
        )
        threshold_frequency = lyman_frequency / lower_level**2
        if lower_level == 1:
            threshold_cross_section = float(
                hydrogen_ground_state_photoionization_cross_section(
                    threshold_frequency
                )
            )
        else:
            threshold_cross_section = float(
                2.815e29
                * threshold_frequency**-3
                / lower_level**5
                * hydrogen_bound_free_gaunt_factor(
                    lower_level, threshold_frequency
                )
            )
        extrapolated_cross_section = (
            threshold_cross_section * (selected / series_limit) ** 3
        )
        lower_population = distribution.population_density[
            :, lower_level - 1
        ]
        opacity[valid] += (
            lower_population[np.newaxis, :]
            / atmosphere.mass_density[np.newaxis, :]
            * extrapolated_cross_section[:, np.newaxis]
            * dissolved_fraction
            * stimulated_emission
        )
    return opacity


def _dissolved_rwa_extinction(
    atmosphere: Atmosphere,
    levels: object,
    principal: int,
    photon_energy: FloatArray,
    state_threshold: FloatArray,
    threshold_kernel: FloatArray,
    state_population: FloatArray,
) -> FloatArray:
    """Daeppen--Anderson--Mihalas dissolved-level continuation of each edge.

    This is the magnetic counterpart of
    :func:`hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient`:
    below each substate's own threshold ``T`` a photon of energy ``E`` maps
    to the fictitious level ``n*^2 = 1/(T - E)``; the threshold cross
    section, continued as ``(T/E)^3``, is multiplied by the full dissolved
    fraction ``1 - w(n*)/w(n)`` for ``n* > n + 3`` and by its neutral part for
    ``n + 1 < n* <= n + 3`` (Lyman: only above the zero-field 925-A
    equivalent).  At zero field every threshold is ``1/n^2`` and the result is
    identical to the ordinary pseudo-continuum.
    """

    from .constants import LIGHT_SPEED as _c
    from .eos import (
        HM_NEUTRAL_HYDROGEN_RADIUS_SCALE,
        charged_particle_hydrogen_occupation_probability,
        hydrogen_occupation_probability,
    )

    state = atmosphere.hydrogen_lte_state
    correlated = state is not None and state.microfield_model == "qmhd"
    radius_scale = (
        state.neutral_radius_scale if state is not None else HM_NEUTRAL_HYDROGEN_RADIUS_SCALE
    )
    minimum_level = float(principal + 1)
    if principal == 1:
        lyman_limit = PLANCK * _c / HYDROGEN_IONIZATION_ENERGY * 1.0e8
        minimum_level = max(minimum_level, 1.0 / np.sqrt(1.0 - lyman_limit / 925.0))
    grid = np.geomspace(minimum_level, 400.0, 240)
    upper = hydrogen_occupation_probability(
        atmosphere.neutral_h_density[np.newaxis, :],
        atmosphere.electron_density[np.newaxis, :],
        atmosphere.temperature[np.newaxis, :],
        grid[:, np.newaxis],
        neutral_radius_scale=radius_scale,
        correlated_microfields=correlated,
    )
    lower = levels.occupation_probability[:, principal - 1]  # type: ignore[attr-defined]
    total_survival = np.clip(upper / lower[np.newaxis, :], 0.0, 1.0)
    upper_charged = charged_particle_hydrogen_occupation_probability(
        atmosphere.electron_density[np.newaxis, :],
        grid[:, np.newaxis],
        atmosphere.temperature[np.newaxis, :] if correlated else None,
    )
    lower_charged = charged_particle_hydrogen_occupation_probability(
        atmosphere.electron_density,
        float(principal),
        atmosphere.temperature if correlated else None,
    )
    charged_survival = np.clip(upper_charged / lower_charged[np.newaxis, :], 0.0, 1.0)
    full_dissolved = 1.0 - total_survival
    neutral_dissolved = np.clip(charged_survival - total_survival, 0.0, 1.0)
    log_grid = np.log(grid)
    result = np.zeros((photon_energy.size, atmosphere.n_depth), dtype=np.float64)
    for threshold in np.unique(state_threshold):
        members = state_threshold == threshold
        weight = threshold_kernel[members] @ state_population[members]
        deficit = threshold - photon_energy
        selected = np.flatnonzero(
            (deficit > 0.0) & (deficit < 1.0 / minimum_level**2)
        )
        if selected.size == 0 or not np.any(weight):
            continue
        effective = np.log(1.0 / np.sqrt(deficit[selected]))
        position = np.clip(np.searchsorted(log_grid, effective) - 1, 0, grid.size - 2)
        fraction = np.clip(
            (effective - log_grid[position]) / (log_grid[position + 1] - log_grid[position]),
            0.0,
            1.0,
        )[:, np.newaxis]
        def sample(table: FloatArray) -> FloatArray:
            return (1.0 - fraction) * table[position] + fraction * table[position + 1]

        local = np.where(
            (np.exp(effective) > principal + 3.0)[:, np.newaxis],
            sample(full_dissolved),
            sample(neutral_dissolved),
        )
        continuation = (threshold / photon_energy[selected]) ** 3
        result[selected] += continuation[:, np.newaxis] * local * weight[np.newaxis, :]
    return result


def _in_module(function, module):
    rebound = FunctionType(function.__code__, vars(module), function.__name__,
                           function.__defaults__, function.__closure__)
    rebound.__kwdefaults__ = function.__kwdefaults__
    rebound.__doc__ = function.__doc__
    return rebound


_PREVIOUS = {
    "hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient": _in_module(
        hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient, opacity),
    "_dissolved_rwa_extinction": _in_module(_dissolved_rwa_extinction, magnetic_continuum),
}
_CURRENT = {
    "hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient":
        opacity.hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient,
    "_dissolved_rwa_extinction": magnetic_continuum._dissolved_rwa_extinction,
}


@contextmanager
def previous_hydrogen_dissolution_cutoff():
    """Replace every wd_spectra binding of the two current functions."""
    with ExitStack() as stack:
        for module_name, module in tuple(sys.modules.items()):
            if module is None or not module_name.startswith("wd_spectra"):
                continue
            for name, current in _CURRENT.items():
                if vars(module).get(name) is current:
                    stack.enter_context(patch.object(module, name, _PREVIOUS[name]))
        yield
