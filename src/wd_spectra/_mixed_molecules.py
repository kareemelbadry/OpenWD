"""Molecular extension of the shared HM H/He chemistry, without a Teff switch.

This is a chemical-picture closure, not a dense-fluid free-energy EOS. It
retains the existing atomic occupation prescription, and includes H2, H2+,
H-, and H3+ in nuclei, particle pressure, charge, and reaction enthalpy.
"""
import numpy as np
from . import eos
from . import molecules as mol
from .constants import BOLTZMANN, HYDROGEN_MASS, HELIUM_MASS


def molecular_hydrogen_helium_lte(
    temperature,
    gas_pressure,
    log_hydrogen_to_helium,
    *,
    maximum_level=eos.HM_MAX_BOUND_LEVEL,
    hydrogen_neutral_radius_scale=0.5,
    helium_neutral_radius_scale=eos.HM_HELIUM_NEUTRAL_RADIUS_SCALE,
    correlated_microfields=True,
    include_negative_hydrogen=True,
    trihydrogen_ion_partition_model="neale-tennyson-1995",
    conservation_tolerance=2e-11,
):
    """Solve mass action with three log densities and strict conservation.

    The atomic solution initializes the nonlinear solve; it is NEVER returned
    on failure. Rates are combined in log space, without capped molecular
    populations, electron floors, or a temperature-dependent dispatch.
    ``conservation_tolerance`` may explicitly tighten the existing log-space
    conservation criterion for derivative-accuracy studies; the default and
    equilibrium equations are unchanged.
    """
    if not np.isfinite(conservation_tolerance) or not 0 < conservation_tolerance <= 2e-11:
        raise ValueError("molecular conservation_tolerance may only tighten the default 2e-11")
    t, p, abundance = np.broadcast_arrays(
        np.asarray(temperature, float),
        np.asarray(gas_pressure, float),
        np.asarray(log_hydrogen_to_helium, float),
    )
    shape = t.shape
    t, p, abundance = t.ravel(), p.ravel(), abundance.ravel()
    options = dict(
        maximum_level=maximum_level,
        hydrogen_neutral_radius_scale=hydrogen_neutral_radius_scale,
        helium_neutral_radius_scale=helium_neutral_radius_scale,
        correlated_microfields=correlated_microfields,
    )
    initial = eos.hummer_mihalas_hydrogen_helium_lte(t, p, abundance, **options)
    target_log = np.log(p / (BOLTZMANN * t))
    x = np.log(
        np.stack(
            (
                initial.hydrogen_lte_state.neutral_h_density,
                initial.helium_lte_state.neutral_he_density,
                initial.electron_density,
            ),
            axis=-1,
        )
    )
    # Analytic H/H+/H2 balance with frozen atomic partitions provides only a
    # starting point. It avoids beginning with impossibly large trace-H2
    # abundances in a cold molecular gas; the full Newton closure follows.
    h0 = initial.hydrogen_lte_state.neutral_h_density
    ion_ratio = initial.hydrogen_lte_state.proton_density / h0
    k2 = mol.molecular_hydrogen_dissociation_constant(
        t,
        atomic_internal_partition_function=initial.hydrogen_lte_state.internal_partition_function,
    )
    linear = 1 + ion_ratio
    neutral_seed = (
        2
        * initial.hydrogen_nuclei_density
        / (linear + np.hypot(linear, np.sqrt(8 * initial.hydrogen_nuclei_density / k2)))
    )
    x[:, 0] = np.log(neutral_seed)
    if np.any(~np.isfinite(x)):
        raise ValueError(
            "Molecular H/He initial state is outside representable density range"
        )
    log_saha = np.log(eos.hydrogen_saha_constant(t))
    translation = 1.5 * np.log(
        2 * np.pi * eos.ELECTRON_MASS * BOLTZMANN * t / eos.PLANCK ** 2
    )
    log_k3 = (
        np.log(
            mol.trihydrogen_ion_dissociation_constant(
                t, partition_model=trihydrogen_ion_partition_model
            )
        )
        if trihydrogen_ion_partition_model is not None
        else None
    )

    def evaluate(v, material=False):
        h, he, ne = np.exp(v).T
        dist = eos.hydrogen_level_distribution(
            h,
            ne,
            t,
            maximum_level=maximum_level,
            neutral_he_density=he,
            neutral_radius_scale=hydrogen_neutral_radius_scale,
            helium_neutral_radius_scale=helium_neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )
        qh = dist.internal_partition_function
        q0, q1, occ, e0, e1 = eos._helium_partition_functions(
            t,
            ne,
            he,
            maximum_level=maximum_level,
            neutral_h_density=h,
            neutral_radius_scale=helium_neutral_radius_scale,
            hydrogen_neutral_radius_scale=hydrogen_neutral_radius_scale,
            correlated_microfields=correlated_microfields,
        )
        hp = v[:, 0] + log_saha - np.log(qh) - v[:, 2]
        h2 = 2 * v[:, 0] - np.log(
            mol.molecular_hydrogen_dissociation_constant(
                t, atomic_internal_partition_function=qh
            )
        )
        h2p = (
            v[:, 0]
            + hp
            - np.log(
                mol.molecular_hydrogen_ion_dissociation_constant(
                    t, atomic_internal_partition_function=qh
                )
            )
        )
        hm = (
            v[:, 0]
            + v[:, 2]
            - np.log(
                mol.negative_hydrogen_ionization_constant(
                    t, atomic_internal_partition_function=qh
                )
            )
            if include_negative_hydrogen
            else np.full_like(t, -np.inf)
        )
        h3p = h2 + hp - log_k3 if log_k3 is not None else np.full_like(t, -np.inf)
        he1 = (
            v[:, 1]
            + np.log(2.0)
            + translation
            + np.log(q1 / q0)
            - eos.HELIUM_FIRST_IONIZATION_ENERGY / (BOLTZMANN * t)
            - v[:, 2]
        )
        he2 = (
            he1
            + np.log(2.0)
            + translation
            - np.log(q1)
            - eos.HELIUM_SECOND_IONIZATION_ENERGY / (BOLTZMANN * t)
            - v[:, 2]
        )
        logs = np.stack(
            (v[:, 0], hp, h2, h2p, hm, h3p, v[:, 1], he1, he2, v[:, 2]), axis=-1
        )
        h_nuclei = np.logaddexp.reduce(
            logs[:, :6] + np.log([1, 1, 2, 2, 1, 3]), axis=-1
        )
        he_nuclei = np.logaddexp.reduce(logs[:, 6:9], axis=-1)
        positive = np.logaddexp.reduce(
            logs[:, [1, 3, 5, 7, 8]] + np.log([1, 1, 1, 1, 2]), axis=-1
        )
        negative = np.logaddexp(logs[:, 4], logs[:, 9])
        residual = np.stack(
            (
                h_nuclei - he_nuclei - np.log(10.0) * abundance,
                negative - positive,
                np.logaddexp.reduce(logs, axis=-1) - target_log,
            ),
            axis=-1,
        )
        if material:
            return logs, h_nuclei, he_nuclei, dist, (q0, q1, occ, e0, e1)
        return residual

    # The finite differences include the response of every occupation factor.
    # Converged members of a batch remain unchanged while other depths finish.
    for iteration in range(40):
        r = evaluate(x)
        norm = np.max(abs(r), axis=-1)
        if np.all(norm < conservation_tolerance):
            break
        jac = np.empty((t.size, 3, 3))
        eps = 2e-5
        for j in range(3):
            shift = np.zeros_like(x)
            shift[:, j] = eps
            jac[:, :, j] = (evaluate(x + shift) - evaluate(x - shift)) / (2 * eps)
        # Explicit vector RHS: NumPy 2 otherwise interprets this batched 2-D
        # array as a matrix RHS (and can silently add a dimension at size 3).
        step = np.linalg.solve(jac, -r[..., None])[..., 0]
        step *= np.minimum(1.0, 2.5 / np.maximum(np.max(abs(step), axis=-1), 1e-300))[
            :, None
        ]
        moved = np.zeros(t.size, bool)
        best = x.copy()
        for damping in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015_625):
            trial = x + damping * step
            # Each independent species is bounded by the target particle
            # density at the solution; allow derivative probes just above it.
            admissible = np.all(trial <= target_log[:, None] + 1.0, axis=-1)
            trial = np.minimum(trial, target_log[:, None] + 1.0)
            trial_norm = np.max(abs(evaluate(trial)), axis=-1)
            accept = admissible & ~moved & (norm >= conservation_tolerance) & (trial_norm < norm)
            best[accept] = trial[accept]
            moved |= accept
        if np.any((norm >= conservation_tolerance) & ~moved):
            bad = int(np.flatnonzero((norm >= conservation_tolerance) & ~moved)[0])
            raise ValueError(
                f"Molecular H/He chemistry failed its conservation line search "
                f"at T={t[bad]:g}, P={p[bad]:g}, log(H/He)={abundance[bad]:g}; residual={norm[bad]:.6g}"
            )
        x = best
    else:
        raise ValueError(
            "Molecular H/He chemistry exhausted its conservation iterations"
        )
    logs, lh, lhe, dist, (q0, q1, occ, e0, e1) = evaluate(x, True)
    n = np.exp(logs)
    h, hp, h2, h2p, hm, h3p, he, he1, he2, ne = n.T
    nh, nhe = np.exp(lh), np.exp(lhe)

    def shaped(a):
        return np.asarray(a).reshape(shape + np.asarray(a).shape[1:])

    hydrogen = eos.HydrogenLTEState(
        mass_density=shaped(HYDROGEN_MASS * nh),
        hydrogen_nuclei_density=shaped(nh),
        neutral_h_density=shaped(h),
        proton_density=shaped(hp),
        electron_density=shaped(ne),
        ionization_fraction=shaped(hp / nh),
        internal_partition_function=shaped(dist.internal_partition_function),
        level_occupation_probability=shaped(dist.occupation_probability),
        level_population_density=shaped(dist.population_density),
        microfield_model="qmhd" if correlated_microfields else "holtsmark",
        molecular_hydrogen_density=shaped(h2),
        molecular_hydrogen_ion_density=shaped(h2p),
        negative_hydrogen_density=shaped(hm),
        trihydrogen_ion_density=shaped(h3p),
        trihydrogen_ion_partition_model=trihydrogen_ion_partition_model,
        chemical_model="molecular-h-he-hm",
        neutral_radius_scale=hydrogen_neutral_radius_scale,
    )
    low = (
        eos.HELIUM_I_LOW_TERM_STATISTICAL_WEIGHT
        * occ
        * np.exp(-eos.HELIUM_I_LOW_TERM_ENERGY / (BOLTZMANN * t[:, None]))
    )
    helium = eos.HeliumLTEState(
        shaped(HELIUM_MASS * nhe),
        shaped(nhe),
        shaped(he),
        shaped(he1),
        shaped(he2),
        shaped(ne),
        shaped((he1 + 2 * he2) / nhe),
        shaped(q0),
        shaped(q1),
        shaped(occ),
        shaped(he[:, None] * low / q0[:, None]),
        microfield_model=hydrogen.microfield_model,
        neutral_radius_scale=helium_neutral_radius_scale,
    )
    state = eos.HydrogenHeliumLTEState(
        shaped(HYDROGEN_MASS * nh + HELIUM_MASS * nhe),
        shaped(nh),
        shaped(nhe),
        shaped(ne),
        shaped(nh / nhe),
        hydrogen,
        helium,
        microfield_model=hydrogen.microfield_model,
        chemical_model="molecular-h-he-hm",
    )
    return state


def molecular_hydrogen_helium_enthalpy(temperature, pressure, abundance, **options):
    """Specific enthalpy from the identical species and partition functions."""
    temperature, pressure, abundance = np.broadcast_arrays(
        np.asarray(temperature, float),
        np.asarray(pressure, float),
        np.asarray(abundance, float),
    )
    state = molecular_hydrogen_helium_lte(temperature, pressure, abundance, **options)
    h, he = state.hydrogen_lte_state, state.helium_lte_state
    t = np.asarray(temperature, float)
    level = np.arange(1, h.level_population_density.shape[-1] + 1.0)
    excitation = np.sum(
        h.level_population_density
        * eos.HYDROGEN_IONIZATION_ENERGY
        * (1 - 1 / level ** 2),
        axis=-1,
    )
    _, _, _, e0, e1 = eos._helium_partition_functions(
        t,
        state.electron_density,
        he.neutral_he_density,
        maximum_level=level.size,
        neutral_h_density=h.neutral_h_density,
        neutral_radius_scale=options.get(
            "helium_neutral_radius_scale", eos.HM_HELIUM_NEUTRAL_RADIUS_SCALE
        ),
        hydrogen_neutral_radius_scale=options.get("hydrogen_neutral_radius_scale", 0.5),
        correlated_microfields=options.get("correlated_microfields", True),
    )
    internal = (
        1.5 * np.asarray(pressure)
        + excitation
        + eos.HYDROGEN_IONIZATION_ENERGY
        * (
            h.proton_density
            + h.molecular_hydrogen_ion_density
            + h.trihydrogen_ion_density
        )
        + h.molecular_hydrogen_density
        * (mol.molecular_hydrogen_rovibrational_energy(t) - mol.H2_DISSOCIATION_ENERGY)
        + h.molecular_hydrogen_ion_density
        * (
            mol.molecular_hydrogen_ion_rovibrational_energy(t)
            - mol.H2_PLUS_DISSOCIATION_ENERGY
        )
        - h.negative_hydrogen_density * mol.H_MINUS_DETACHMENT_ENERGY
        - h.trihydrogen_ion_density
        * (mol.H2_DISSOCIATION_ENERGY + mol.H3_PLUS_DISSOCIATION_ENERGY)
        + he.neutral_he_density * e0
        + he.singly_ionized_he_density * (eos.HELIUM_FIRST_IONIZATION_ENERGY + e1)
        + he.doubly_ionized_he_density
        * (eos.HELIUM_FIRST_IONIZATION_ENERGY + eos.HELIUM_SECOND_IONIZATION_ENERGY)
    )
    if h.trihydrogen_ion_partition_model is not None:
        internal += (
            h.trihydrogen_ion_density
            * mol.trihydrogen_ion_rovibrational_energy(
                t, partition_model=h.trihydrogen_ion_partition_model
            )
        )
    return (internal + pressure) / state.mass_density, state


def molecular_hydrogen_helium_thermodynamics(
    temperature, pressure, abundance, **options
):
    t, p, a = np.broadcast_arrays(
        np.asarray(temperature, float),
        np.asarray(pressure, float),
        np.asarray(abundance, float),
    )
    eps = 2e-4
    cold, sc = molecular_hydrogen_helium_enthalpy(t * np.exp(-eps), p, a, **options)
    hot, sh = molecular_hydrogen_helium_enthalpy(t * np.exp(eps), p, a, **options)
    central = molecular_hydrogen_helium_lte(t, p, a, **options)
    cp = (hot - cold) / (2 * t * np.sinh(eps))
    q = -np.log(sh.mass_density / sc.mass_density) / (2 * eps)
    ad = p * q / (central.mass_density * t * cp)
    if np.any(~np.isfinite(cp)) or np.any(cp <= 0) or np.any(q <= 0):
        raise ValueError("Invalid molecular H/He thermal derivatives")
    return eos.HydrogenHeliumThermodynamics(cp, q, ad)
