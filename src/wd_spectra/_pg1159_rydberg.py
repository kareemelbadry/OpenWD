"""Rydberg-level atomic data upgrades for the PG1159 formal (line-formation) model.

Three transformations of a :class:`~wd_spectra.pg1159.PG1159NLTEModel`, applied to
the formal model only (the structure model and the other model classes are not
touched):

1. :func:`with_hydrogenic_rydberg_ladder` -- explicit hydrogenic shells for
   O VI (n = 11..18) and C IV (n = 11..14) replace the TMAD LTE reservoir levels
   they cover.  Each shell has g = 2 n^2 and E = IP - Z^2 R / n^2, and
   shell-averaged hydrogenic E1 lines (Kramers f with the Johnson 1972 Gaunt
   factor) to every explicit term below it and to the lower shells.  Shells
   nmax+1..ntop remain LTE with the continuum parent and are linked to the top
   three explicit shells by Van Regemorter collisions.  Without the ladder the
   explicit n <= 10 terms couple to the reservoir only radiatively (the TMAD
   atoms carry no CBX collision records), so the high-n ladder cannot
   thermalize to the continuum.  This flips O VI 5290 from absorption to the
   observed emission.
2. :func:`with_same_shell_electron_collisions` -- electron-impact dipole
   collisions between adjacent l states of the same n (l <-> l+1) from the
   hydrogenic line strength S = 2 (l+1) R^2, R = 3 n sqrt(n^2 - (l+1)^2)/(2Z)
   a0, in the Van Regemorter form, for O VI (n >= 6, e.g. 7d-7f, which had no
   record at all) and C IV (n >= 4).  For pairs with l >= 2 they also replace
   the placeholder TMAD records (C IV f = 1e-5 .. 7e-4), which left 5f
   under-populated relative to 5g.  PSM20/BTM/Chianti records are kept.
3. :func:`with_civ_linear_stark_profiles` -- the formal C IV and C VI lines of
   the Stout list carry no TMAD profile prescription and get the classical
   width (n_eff capped at 5) without any static ion broadening.  Lines with
   n_lower >= 5 receive the TMAD formula-4 (linear Stark) parameters that the
   TMAD C IV atom gives the same transitions.  n_lower < 5 is excluded: the
   broad C IV 4-6 wings otherwise degrade He II 1640 in the suite.
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import MappingProxyType

import numpy as np
from scipy.special import exp1

from .light_metal_nlte import ConstantEffectiveCollisionStrength, TmadLTEBoundBoundCoupling
from .metals import AtomicLevel, AtomicTransition

_RYDBERG_WAVENUMBER = 109737.316
_EV_TO_WAVENUMBER = 8065.54429
_HC_OVER_K = 1.4388  # cm K
# Van Regemorter: q = 8.629e-6 Omega / (g sqrt T) exp(-u)
_COLLISION_CONSTANT = 8.629e-6
_ANGULAR = "SPDFGHIKLMNOQ"
# TMAD C IV formula-4 Cowley parameter P as a function of the upper n.
_CIV_STARK_P = {2: 1.28, 3: 3.08, 4: 9.85, 5: 24.24, 6: 50.52, 7: 93.94,
                8: 160.69, 9: 257.94, 10: 390.64}


def _hydrogenic_oscillator_strength(p: int, n: int) -> float:
    """Shell-averaged absorption f(p -> n): Kramers with the Johnson (1972) Gaunt factor."""
    x = 1.0 - (p / n) ** 2
    if p == 1:
        g0, g1, g2 = 1.1330, -0.4059, 0.07014
    elif p == 2:
        g0, g1, g2 = 1.0785, -0.2319, 0.02947
    else:
        g0 = 0.9935 + 0.2328 / p - 0.1296 / p**2
        g1 = -(0.6282 - 0.5598 / p + 0.5299 / p**2) / p
        g2 = (0.3887 - 1.181 / p + 1.470 / p**2) / p**2
    gaunt = g0 + g1 / x + g2 / x**2
    return float(32.0 / (3.0 * np.sqrt(3.0) * np.pi) * p / n**3 / x**3 * gaunt)


def _van_regemorter_gbar(delta_wavenumber: float, temperature: float = 1.0e5) -> float:
    u = _HC_OVER_K * delta_wavenumber / temperature
    return max(0.2, 0.276 * float(np.exp(u) * exp1(u)))


def _principal_quantum_number(label: str) -> int | None:
    """n of a Stout label (``1s2.7d.(2D<5/2>)``), a TMAD label (``O608S``) or a shell (``O611``)."""
    text = str(label)
    match = re.search(r"\.(\d+)[a-z]", text)
    if match:
        return int(match.group(1))
    match = re.match(r"\s*[A-Z][A-Z]?\d(\d)(\d)", text)
    if match:
        return int(match.group(1) + match.group(2))
    return None


def _with_ion(database, key, ion):
    ions = dict(database.ions)
    ions[key] = ion
    return replace(
        database,
        ions=MappingProxyType(ions) if isinstance(database.ions, MappingProxyType) else ions,
    )


def with_hydrogenic_rydberg_ladder(
    model,
    *,
    maximum_explicit_n: dict | None = None,
    maximum_reservoir_n: dict | None = None,
):
    """Return ``model`` with explicit hydrogenic O VI and C IV Rydberg shells (see module docstring)."""

    maximum_explicit_n = maximum_explicit_n or {"C": 14, "O": 18}
    maximum_reservoir_n = maximum_reservoir_n or {"C": 22, "O": 26}
    updates = {}
    for element, symbol, charge in (("carbon", "C", 3), ("oxygen", "O", 5)):
        database = getattr(model, f"{element}_population_atomic_database", None)
        if database is None or (symbol, charge) not in database.ions:
            continue
        ion = database.ions[(symbol, charge)]
        levels_per_charge = dict(getattr(model, f"{element}_levels_per_charge"))
        explicit_count = int(levels_per_charge[charge])
        ordered = sorted(ion.levels, key=lambda level: level.energy_wavenumber)
        explicit = ordered[:explicit_count]
        n_explicit = max(_principal_quantum_number(level.label) or 0 for level in explicit)
        core = charge + 1
        limit = ion.ionization_energy_ev * _EV_TO_WAVENUMBER
        first_index = max(level.index for level in ion.levels) + 1
        shells = [
            AtomicLevel(
                first_index + k,
                limit - core * core * _RYDBERG_WAVENUMBER / n**2,
                2.0 * n * n,
                f"{symbol}{core}{n:02d}",
            )
            for k, n in enumerate(range(n_explicit + 1, maximum_explicit_n[symbol] + 1))
        ]
        if not shells:
            continue
        top_energy = shells[-1].energy_wavenumber
        remaining = ordered[explicit_count:]
        replaced = {level.index for level in remaining if level.energy_wavenumber <= top_energy}
        kept = [level for level in remaining if level.energy_wavenumber > top_energy]
        transitions = [
            transition
            for transition in ion.transitions
            if transition.lower_index not in replaced and transition.upper_index not in replaced
        ]
        for upper in shells:
            n_upper = _principal_quantum_number(upper.label)
            for lower in explicit + [shell for shell in shells if _principal_quantum_number(shell.label) < n_upper]:
                n_lower = _principal_quantum_number(lower.label)
                if n_lower is None or n_lower >= n_upper:
                    continue
                delta = upper.energy_wavenumber - lower.energy_wavenumber
                if delta <= 0.0:
                    continue
                oscillator = _hydrogenic_oscillator_strength(n_lower, n_upper)
                wavelength = 1.0e8 / delta
                einstein_a = (
                    6.670e15 * lower.statistical_weight * oscillator
                    / (upper.statistical_weight * wavelength * wavelength)
                )
                transitions.append(AtomicTransition(
                    lower.index, upper.index, einstein_a, "E1", wavelength, oscillator))
        database = _with_ion(
            database,
            (symbol, charge),
            replace(ion, levels=tuple(explicit + shells + kept), transitions=tuple(transitions)),
        )
        levels_per_charge[charge] = explicit_count + len(shells)

        reservoir = dict(getattr(model, f"{element}_lte_level_reservoir") or {})
        parent = next(key for key, entries in reservoir.items()
                      if any(entry[0] == charge for entry in entries))
        top_shells = [
            (charge, limit - core * core * _RYDBERG_WAVENUMBER / n**2, 2.0 * n * n)
            for n in range(maximum_explicit_n[symbol] + 1, maximum_reservoir_n[symbol] + 1)
        ]
        reservoir[parent] = tuple(
            entry for entry in reservoir[parent]
            if not (entry[0] == charge and entry[1] <= top_energy + 1.0)
        ) + tuple(top_shells)
        couplings = [
            coupling
            for coupling in (getattr(model, f"{element}_lte_bound_bound_couplings") or ())
            if not (coupling.lte_charge == charge and coupling.lte_energy_wavenumber <= top_energy + 1.0)
        ]
        for lower in shells[-3:]:
            n_lower = _principal_quantum_number(lower.label)
            for _, energy, weight in top_shells:
                n_upper = int(round(np.sqrt(weight / 2.0)))
                delta = energy - lower.energy_wavenumber
                oscillator = _hydrogenic_oscillator_strength(n_lower, n_upper)
                omega = (
                    19.7363 * lower.statistical_weight * oscillator * _van_regemorter_gbar(delta)
                    / (_COLLISION_CONSTANT * _HC_OVER_K * delta)
                )
                couplings.append(TmadLTEBoundBoundCoupling(
                    explicit_level_key=(symbol, charge, lower.index),
                    lte_parent_key=parent,
                    explicit_is_lower=True,
                    lte_charge=charge,
                    lte_energy_wavenumber=float(energy),
                    lte_statistical_weight=float(weight),
                    lte_level_label=f"{symbol}{core}top{n_upper}",
                    # Collisional coupling only: the shell-averaged radiative
                    # transitions are carried by the explicit ladder above.
                    transition=AtomicTransition(lower.index, 0, 1.0e-30, "E1", 1.0e8 / delta, 1.0e-12),
                    collision_record=ConstantEffectiveCollisionStrength(
                        float(omega), "hydrogenic Rydberg ladder: Van Regemorter"),
                ))
        updates.update({
            f"{element}_population_atomic_database": database,
            f"{element}_levels_per_charge": MappingProxyType(levels_per_charge),
            f"{element}_lte_level_reservoir": MappingProxyType(reservoir),
            f"{element}_lte_bound_bound_couplings": tuple(couplings),
        })
    return replace(model, **updates) if updates else model


def with_same_shell_electron_collisions(
    model,
    *,
    specifications=(("carbon", "C", 3, 4, True, 2), ("oxygen", "O", 5, 6, True, 2)),
    maximum_n: int = 10,
):
    """Return ``model`` with hydrogenic same-n (l <-> l+1) electron collisions (see module docstring).

    Each specification is ``(element, symbol, charge, minimum_n, replace_tmad, replace_minimum_l)``:
    pairs without any record always receive the hydrogenic rate; with ``replace_tmad`` the TMAD
    (tuple) records of radiative pairs with lower l >= ``replace_minimum_l`` are replaced too.
    """

    updates = {}
    for element, symbol, charge, minimum_n, replace_tmad, replace_minimum_l in specifications:
        database = getattr(model, f"{element}_population_atomic_database", None)
        if database is None or (symbol, charge) not in database.ions:
            continue
        ion = database.ions[(symbol, charge)]
        core = charge + 1
        prefix = f"{symbol.upper()}{core}"
        terms = {}
        for level in ion.levels:
            label = level.label.strip().upper()
            if not label.startswith(prefix):
                continue
            match = re.match(r"(\d{2})([SPDFGHIKLMNOQ])", label[len(prefix):])
            if match:
                terms[(int(match.group(1)), _ANGULAR.index(match.group(2)))] = level
        radiative = {frozenset((t.lower_index, t.upper_index)) for t in ion.transitions}
        collisions = dict(getattr(model, f"{element}_collision_data") or {})
        tmad_records = {
            frozenset((key[2], key[3])): key
            for key, record in collisions.items()
            if key[0] == symbol and key[1] == charge and isinstance(record, tuple)
        }
        other_records = {
            frozenset((key[2], key[3]))
            for key, record in collisions.items()
            if key[0] == symbol and key[1] == charge and not isinstance(record, tuple)
        }
        changed = False
        for (n, l), first in terms.items():
            if not minimum_n <= n <= maximum_n:
                continue
            second = terms.get((n, l + 1))
            if second is None:
                continue
            pair = frozenset((first.index, second.index))
            if pair in other_records:
                continue  # PSM20/BTM/Chianti records are kept
            if pair in tmad_records or pair in radiative:
                if not (replace_tmad and l >= replace_minimum_l):
                    continue
            lower, upper = sorted((first, second), key=lambda level: level.energy_wavenumber)
            delta = max(abs(second.energy_wavenumber - first.energy_wavenumber), 10.0)
            radius = 3.0 * n * np.sqrt(n * n - (l + 1) ** 2) / (2.0 * core)
            strength = 2.0 * (l + 1) * radius * radius
            omega = (
                19.7363 * _van_regemorter_gbar(delta) * strength
                / (3.0 * _RYDBERG_WAVENUMBER * _COLLISION_CONSTANT * _HC_OVER_K)
            )
            if pair in tmad_records:
                collisions.pop(tmad_records[pair])
            collisions[(symbol, charge, lower.index, upper.index)] = ConstantEffectiveCollisionStrength(
                float(omega), "same-n electron dipole (hydrogenic S, Van Regemorter)")
            changed = True
        if changed:
            updates[f"{element}_collision_data"] = MappingProxyType(collisions)
    return replace(model, **updates) if updates else model


def _stout_principal_n(label: str) -> int | None:
    text = label.strip()
    match = re.match(r"n=(\d+)", text)
    if match:
        return int(match.group(1))
    match = re.match(r"1s2\.(\d+)[a-z]", text)
    return int(match.group(1)) if match else None


def with_civ_linear_stark_profiles(model, *, minimum_lower_n: int = 5):
    """Return ``model`` whose formal C IV and C VI Rydberg lines use TMAD formula-4 profiles (see module docstring)."""

    database = getattr(model, "atomic_database", None)
    if database is None:
        return model
    for key, core, cowley in (
        (("C", 3), 4.0, lambda n: _CIV_STARK_P.get(n, 0.039 * n**4)),
        (("C", 5), 6.0, lambda n: 0.0202 * n**4),
    ):
        ion = database.ions.get(key)
        if ion is None:
            continue
        labels = {level.index: level.label for level in ion.levels}
        transitions = []
        for transition in ion.transitions:
            if transition.profile_formula is None:
                n_lower = _stout_principal_n(labels.get(transition.lower_index, ""))
                n_upper = _stout_principal_n(labels.get(transition.upper_index, ""))
                if (n_lower is not None and n_upper is not None
                        and n_lower >= minimum_lower_n and n_upper > n_lower):
                    transition = replace(
                        transition,
                        profile_formula=4,
                        profile_parameters=(
                            transition.absorption_oscillator_strength,
                            1.0e8,
                            cowley(n_upper),
                            core,
                            float(n_lower),
                            float(n_upper),
                        ),
                    )
            transitions.append(transition)
        database = _with_ion(database, key, replace(ion, transitions=tuple(transitions)))
    return replace(model, atomic_database=database)


def with_pg1159_rydberg_upgrades(model):
    """Apply the three formal-model upgrades in their validated order."""

    model = with_hydrogenic_rydberg_ladder(model)
    model = with_same_shell_electron_collisions(model)
    return with_civ_linear_stark_profiles(model)
