"""Cold-start homogeneous PG 1159 models with the shared nonlinear solver."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Mapping
import hashlib
import warnings
import numpy as np
from .common import (
    ModelData,
    ModelResult,
    Quality,
    validate_wavelength,
    numerical_resolution,
    AtmosphereConvergenceWarning,
)
from ..pg1159_presets import PG1424_PUBLISHED_MASS_FRACTIONS
from .._pg1159_builder import (
    PG1159_HYDROGENIC_STARK_COMPONENTS,
    build_model,
    structure_wavelength,
)
from .._pg1159_structure import solve_pg1159_atmosphere, PG1159Equations
from .._pg1159_transfer import transfer_field
from ..atmosphere import helium_continuum_atmosphere
from ..spectrum import Spectrum


@dataclass(frozen=True)
class PG1159Config:
    effective_temperature: float = 110000.0
    logg: float = 7.0
    mass_fractions: Mapping[str, float] = field(
        default_factory=lambda: dict(PG1424_PUBLISHED_MASS_FRACTIONS)
    )
    target_name: str = "PG 1424+535"
    quality: Quality = "standard"
    oxygen_atom: str = "extended54-complete"
    include_radiative_acceleration: bool = False
    population_maximum_iterations: int = 120
    population_tolerance: float = 1e-2
    # Converge the upper atmosphere (tau_Ross < 1e-2) to local radiative
    # equilibrium after certification and certify the refined structure.
    # Default since 2026-10-14: without it the layers above tau_Ross ~ 1e-3
    # stay near the gray seed. Their heating residual passes the gate but
    # barely depends on temperature, so they are not in local radiative
    # equilibrium. Either way the certificate scope is spectrum-qualified.
    refine_upper_atmosphere: bool = True
    # Rosseland depth above which the refinement imposes local radiative
    # equilibrium; Unsold-Lucy flux corrections apply below it. With 1e-2 the
    # line-dominated mid-layers (1e-3 < tau_Ross < 1e-2) are ill-conditioned
    # under local balance, and the refinement stagnated or diverged in several
    # published-parameter models.
    refinement_local_maximum_rosseland_depth: float = 1e-3
    # Shallowest Rosseland depth at which the coupled solve gates local energy
    # balance. If the first gated cell stalls the Newton step, the solve
    # restarts cold with the gate ten times deeper, up to
    # ``maximum_local_energy_minimum_rosseland_depth``. Set both equal to
    # disable the fallback.
    local_energy_minimum_rosseland_depth: float = 1e-5
    maximum_local_energy_minimum_rosseland_depth: float = 1e-3
    # Formal-model atomic upgrades (see _pg1159_rydberg): explicit hydrogenic
    # O VI/C IV Rydberg shells, same-shell electron collisions, and linear
    # Stark profiles for high C IV/C VI lines. Enabling them also tightens
    # the formal population tolerance to 1e-3.
    rydberg_upgrades: bool = True


@dataclass(frozen=True)
class PG1159PopulationResult:
    """Keep structural and trace line-formation states with their own material."""

    structure: object
    line_formation: object
    line_formation_atmosphere: object

    @property
    def converged(self):
        return bool(self.structure.converged and self.line_formation.converged)


def validate_config(config):
    if not isinstance(config, PG1159Config):
        raise TypeError("expected PG1159Config")
    if (
        not np.isfinite(config.effective_temperature)
        or config.effective_temperature <= 0
        or not np.isfinite(config.logg)
    ):
        raise ValueError("Teff must be positive and finite; logg must be finite")
    numerical_resolution(config.quality)
    if config.oxygen_atom not in ("compact14", "extended54-complete"):
        raise ValueError("unknown oxygen atom")
    if not isinstance(config.include_radiative_acceleration, bool):
        raise ValueError("include_radiative_acceleration must be boolean")
    if not isinstance(config.refine_upper_atmosphere, bool):
        raise ValueError("refine_upper_atmosphere must be boolean")
    if not isinstance(config.rydberg_upgrades, bool):
        raise ValueError("rydberg_upgrades must be boolean")
    for name in (
        "refinement_local_maximum_rosseland_depth",
        "local_energy_minimum_rosseland_depth",
        "maximum_local_energy_minimum_rosseland_depth",
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must lie in (0, 1]")
    if (
        config.maximum_local_energy_minimum_rosseland_depth
        < config.local_energy_minimum_rosseland_depth
    ):
        raise ValueError(
            "maximum_local_energy_minimum_rosseland_depth must not be shallower "
            "than local_energy_minimum_rosseland_depth"
        )
    if (
        isinstance(config.population_maximum_iterations, bool)
        or not isinstance(config.population_maximum_iterations, int)
        or config.population_maximum_iterations < 1
    ):
        raise ValueError("population_maximum_iterations must be a positive integer")
    if (
        not np.isfinite(config.population_tolerance)
        or not 0 < config.population_tolerance <= 1e-2
    ):
        raise ValueError("population_tolerance must lie in (0, 1e-2]")
    if any(not np.isfinite(v) or v < 0 for v in config.mass_fractions.values()) or any(
        config.mass_fractions.get(e, 0) <= 0 for e in ("He", "C", "O")
    ):
        raise ValueError(
            "mass fractions must be finite and nonnegative with positive He, C, and O"
        )


def required_atomic_files(data, oxygen_atom):
    cache = data.cache
    paths = [
        data.ccc_hydrogen_collisions,
        data.tlusty_source,
        data.tlusty_helium_atom,
        cache / "helium-stark/Tremblay26.txt",
        data.helium_ii_stark,
        data.verner_photoionization,
    ]
    paths += [
        cache / "tlusty-atoms" / n
        for n in ("c3.dat", "c4_35+2lev.dat", "o4.dat", "o5.dat", "o6.dat")
    ]
    paths += [
        cache / "tmad-atoms" / n
        for n in (
            "C_III-V",
            "O_III-VII",
            "C_IV_syn",
            "O_III_syn",
            "O_IV_syn",
            "O_V_syn",
            "O_VI_syn",
            "O_VII_syn",
        )
    ]
    if oxygen_atom == "extended54-complete":
        paths += [
            cache / "sirocco-atomic/o_6_levels.dat",
            cache / "sirocco-atomic/o_6_phot.dat",
            cache / "chianti/o_6/o_6.scups",
        ]
    return tuple(paths)


class _LocalEnergyGateStall(Exception):
    """The first gated cell stalled the coupled Newton solve."""


def _gate_stall_monitor(
    callback,
    *,
    minimum_residual=2e-2,
    maximum_step=5e-3,
    crawl_iterations=8,
    converged_flux=2e-2,
):
    """Wrap an iteration callback; raise when the gate-boundary cell stalls.

    Two failure modes seen in published-parameter models, both with the worst
    local-energy cell at the shallowest gated cell and its relative residual
    above twice the 1e-2 relaxation tolerance (``minimum_residual``):

    * stall: the boundary cell is the worst on two consecutive iterations and
      the trust region shrinks the accepted step below ``maximum_step``
      (HE 1429-1209 at 1e-5 and 1e-4, PG 1144+005 and RX J0122.9-7521 at 1e-5);
    * crawl: the all-depth flux has already converged below
      ``converged_flux``, yet the boundary cell stays the worst for
      ``crawl_iterations`` consecutive iterations with full-size steps
      (PG 1151-029 at 1e-5, whose residual fell from 9% to 1.7% over 30
      iterations and about 6 h).

    On the logged published-parameter runs these fired only on those models.
    Among the converging runs, the longest flux-converged boundary streak was
    four iterations.
    """
    boundary_iterations = [0]
    crawl = [0]

    def monitor(i, a, p, d):
        if callback is not None:
            callback(i, a, p, d)
        gate = d.get("local_energy_minimum_rosseland_depth")
        index = d.get("maximum_local_energy_balance_depth_index")
        residual = d.get("maximum_relative_cell_energy_balance_residual")
        step = d.get("nonlinear_step")
        flux = d.get("maximum_all_depth_total_flux_residual")
        tau = getattr(a, "rosseland_optical_depth", None)
        if gate is None or index is None or residual is None or tau is None:
            boundary_iterations[0] = 0
            crawl[0] = 0
            return
        tau = np.asarray(tau)
        gated = np.flatnonzero(tau >= gate)
        at_boundary = (
            gated.size > 0 and int(index) == int(gated[0]) and residual > minimum_residual
        )
        boundary_iterations[0] = boundary_iterations[0] + 1 if at_boundary else 0
        crawl[0] = (
            crawl[0] + 1 if at_boundary and flux is not None and flux < converged_flux else 0
        )
        location = f"local-energy gate boundary (tau_Ross={tau[int(index)]:.3g}, residual {residual:.3g})"
        if boundary_iterations[0] >= 2 and step is not None and step < maximum_step:
            raise _LocalEnergyGateStall(f"{location} stalled the Newton step ({step:.3g})")
        if crawl[0] >= crawl_iterations:
            raise _LocalEnergyGateStall(
                f"{location} held the converged-flux solve for {crawl[0]} iterations"
            )

    return monitor


def compute_pg1159(
    config=PG1159Config(),
    wavelength=None,
    *,
    data=None,
    iteration_callback=None,
    output=None,
):
    """Build a fresh He/C/O atmosphere; return the common ModelResult.

    Additional listed elements are included in the final line-formation
    calculation, as in the development model. Their opacity is not fed back
    into the structural solve. No previous model/checkpoint is loaded.
    """
    validate_config(config)
    wave = validate_wavelength(wavelength)
    data = ModelData.default() if data is None else data
    files = required_atomic_files(data, config.oxygen_atom)
    missing = [str(p) for p in files if not p.is_file()]
    if not (data.stout / "stout").is_dir():
        missing.append(str(data.stout / "stout"))
    if missing:
        raise FileNotFoundError(
            "PG1159 requires atomic data at ModelData/OPENWD_DATA; see docs/models/PG1159.md. Missing: "
            + ", ".join(missing)
        )
    full = {
        e: float(v) / sum(config.mass_fractions.values())
        for e, v in config.mass_fractions.items()
        if v > 0
    }
    heco = {
        e: full[e] / sum(full[k] for k in ("He", "C", "O")) for e in ("He", "C", "O")
    }
    resolution = numerical_resolution(config.quality)
    # The PG1159 validation model is converged with the same 120-point,
    # two-angle quadrature used by the development calculation.  Reusing the
    # package-wide 300/3 standard atmosphere grid increased transfer work by
    # nearly fourfold without changing the requested stellar model. Reserve
    # the denser quadrature for production-quality resolution studies.
    structure_continuum, structure_angles = {
        "quick": (120, 2),
        "standard": (120, 2),
        "production": (300, 3),
    }[config.quality]
    model, _ = build_model(
        data,
        heco,
        structure_only=True,
        population_iterations=config.population_maximum_iterations,
        oxygen_atom_preset=config.oxygen_atom,
    )
    model = replace(
        model,
        coupled_population_acceleration_depth=80,
        metal_population_relative_tolerance=config.population_tolerance,
        helium_model=replace(model.helium_model, population_n_angle=structure_angles),
    )
    seed = helium_continuum_atmosphere(
        config.effective_temperature,
        config.logg,
        n_depth=resolution.n_depth,
        rosseland_frequency_points=structure_continuum,
        # The validated legacy PG1159 structures end at tau_Ross=100.  The
        # extra public decade to 1000 altered the hydrostatic column-mass grid
        # without contributing to the emergent spectrum and introduced the
        # deep, ill-conditioned residuals seen during migration.
        tau_max=100.0,
    )
    seed = model.rebuild_atmosphere(seed, seed.temperature, None)
    from .._pg1159_reference import gray_opacity_seed

    seed = gray_opacity_seed(
        model, seed, wavelength_points=structure_continuum
    )
    refinement = None
    if config.refine_upper_atmosphere:
        from .._pg1159_ali_temperature import REFINEMENT_LINE_VELOCITY_SAMPLES_KMS

        refinement_model = replace(
            model,
            metal_rate_line_velocity_samples_kms=REFINEMENT_LINE_VELOCITY_SAMPLES_KMS,
        )
        # The refinement holds the pressure structure fixed; radiation
        # pressure, when requested, enters the coupled Newton
        # re-certification that follows (a separate pressure update inside
        # the temperature refinement does not converge).
        refinement = (
            refinement_model,
            structure_wavelength(refinement_model, structure_continuum),
            {
                "include_radiative_acceleration": False,
                "local_maximum_rosseland_depth": (
                    config.refinement_local_maximum_rosseland_depth
                ),
            },
        )
    user_callback = (
        None
        if iteration_callback is None
        else lambda i, a, p, d: iteration_callback(i, a, d)
    )
    gate = config.local_energy_minimum_rosseland_depth
    gate_restarts = []
    while True:
        gated_model = replace(
            model, atmosphere_local_energy_minimum_rosseland_depth=gate
        )
        fallback = gate * 10.0 <= (
            config.maximum_local_energy_minimum_rosseland_depth * (1 + 1e-9)
        )
        try:
            result = solve_pg1159_atmosphere(
                seed,
                gated_model,
                structure_wavelength(gated_model, structure_continuum),
                maximum_iterations=resolution.maximum_iterations,
                cold_start=True,
                refinement=(
                    None
                    if refinement is None
                    else (
                        replace(
                            refinement[0],
                            atmosphere_local_energy_minimum_rosseland_depth=gate,
                        ),
                        *refinement[1:],
                    )
                ),
                include_radiative_acceleration=config.include_radiative_acceleration,
                iteration_callback=(
                    _gate_stall_monitor(user_callback) if fallback else user_callback
                ),
            )
        except _LocalEnergyGateStall as stall:
            gate_restarts.append({"gate": gate, "reason": str(stall)})
            warnings.warn(
                f"PG1159 {stall}; restarting cold with the local-energy gate at "
                f"tau_Ross={gate * 10.0:.3g}",
                AtmosphereConvergenceWarning,
                stacklevel=2,
            )
            gate *= 10.0
            continue
        break
    model = gated_model
    from .._convergence import recorded_equilibrium_status

    status = recorded_equilibrium_status(result.atmosphere.metadata)
    formal_model = model
    state = result.population_state
    atmosphere = result.atmosphere
    formal_tolerance = (
        1.0e-3 if config.rydberg_upgrades else max(config.population_tolerance, 1.0e-2)
    )
    if set(full) - {"He", "C", "O"}:
        formal_model, _ = build_model(
            data,
            full,
            structure_only=False,
            population_iterations=config.population_maximum_iterations,
            oxygen_atom_preset=config.oxygen_atom,
        )
        formal_model = replace(
            formal_model,
            # Match the validated structural closure: the 80-state weighted
            # SVD history resolves weak population modes, followed by the
            # undamped physical map near the target. The builder's 0.25
            # damping made the expanded formal atom spend tens of costly
            # iterations applying quarter-sized updates.
            coupled_population_acceleration_depth=(
                model.coupled_population_acceleration_depth
            ),
            metal_population_damping=1.0,
            helium_population_damping=1.0,
            # The expanded trace-element atom only affects the formal
            # spectrum.  Its fixed point develops a roughly 0.6% numerical
            # floor while the optical spectrum is already stable, so do not
            # force the formal solve below one percent.  The He/C/O structure
            # is independently certified at config.population_tolerance.
            # With the Rydberg upgrades the explicit high shells start from
            # the structural anchor and need a real fixed point: 1e-3 closed
            # in 2-74 iterations across the published-parameter suite.
            metal_population_relative_tolerance=formal_tolerance,
            atmosphere_local_energy_minimum_rosseland_depth=gate,
            helium_model=replace(
                formal_model.helium_model, population_n_angle=structure_angles
            ),
        )
    elif config.rydberg_upgrades:
        # He/C/O only: the formal solve runs on the structural atom, with the
        # same undamped map as the trace-element branch. The builder's 0.25
        # metal damping left RX J0122.9-7521 short of the 1e-3 tolerance
        # after 400 iterations.
        formal_model = replace(
            model,
            metal_population_relative_tolerance=formal_tolerance,
            metal_population_damping=1.0,
            helium_population_damping=1.0,
        )
    if config.rydberg_upgrades:
        from .._pg1159_rydberg import with_pg1159_rydberg_upgrades

        formal_model = with_pg1159_rydberg_upgrades(
            replace(
                formal_model,
                # The explicit Rydberg shells start from the structural anchor;
                # RX J0122.9-7521 needed 393 undamped iterations to reach 1e-3.
                metal_population_iterations=max(
                    formal_model.metal_population_iterations, 800
                ),
            )
        )
    if formal_model is not model:
        formal_equations = PG1159Equations(
            atmosphere,
            formal_model,
            structure_wavelength(formal_model, structure_continuum),
            radiative_acceleration=False,
        )
        formal_equations.anchor = state
        atmosphere, state = formal_equations.material(formal_equations.initial_state())
    c = formal_model.transfer_coefficients(atmosphere, wave, state)
    _, radiation, closure = transfer_field(atmosphere, c, n_angle=structure_angles)
    # Same NLTE populations without any bound-bound terms.
    _, continuum_radiation, _ = transfer_field(
        atmosphere,
        formal_model.transfer_coefficients(atmosphere, wave, state, lines=False),
        n_angle=structure_angles, check_source=False)
    spectrum = Spectrum(
        wave,
        radiation.interface_flux[:, 0],
        {
            "flux_unit": "erg s^-1 cm^-2 Angstrom^-1",
            "flux_convention": "surface F_lambda",
            "wavelength_medium": "vacuum",
            "composition": "PG1159",
            "line_formation": "He/C/O term-resolved NLTE with trace ion-stage NLTE",
            "source_closure_residual": closure,
            "continuum_flux": (
                "NLTE He and metal bound-free/free-free with the converged "
                "populations, dissolved-level pseudo-continuum, electron "
                "scattering; all lines removed"
            ),
        },
        continuum_radiation.interface_flux[:, 0],
    )
    qualified_status = status if state.converged and closure < 1e-6 else "unconverged"
    if qualified_status != "converged":
        detail = (
            "; the spectrum passed its declared qualification profile"
            if qualified_status == "spectrum-qualified"
            else ""
        )
        warnings.warn(
            "PG1159 model has not passed the full equilibrium certificate"
            + detail,
            AtmosphereConvergenceWarning,
            stacklevel=2,
        )
    # Record only the Stout elements actually used by this requested mixture.
    stout_files = tuple(
        p
        for element in full
        if element != "He"
        for p in sorted((data.stout / "stout" / element.lower()).glob("*/*"))
        if p.suffix in (".nrg", ".tp")
    )

    def data_key(path):
        try:
            return str(path.relative_to(data.root))
        except ValueError:
            return str(path.resolve())

    identity = {
        data_key(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (*files, *stout_files)
    }
    value = ModelResult(
        "PG1159",
        result.atmosphere,
        spectrum,
        config,
        {
            "experimental": True,
            "cold_start": True,
            "stellar_parameters_fixed": True,
            "mass_fractions": full,
            "structure_mass_fractions": heco,
            "trace_opacity_in_structure": False,
            "oxygen_atom": config.oxygen_atom,
            "structure_continuum_points": structure_continuum,
            "structure_angle_points": structure_angles,
            "structure_maximum_rosseland_depth": 100.0,
            "carbon_atom_counts": dict(model.carbon_levels_per_charge),
            "oxygen_atom_counts": dict(model.oxygen_levels_per_charge),
            "helium_ii_collision_model": model.helium_model.hydrogenic_collision_model,
            "ccc_maximum_shell": model.helium_model.collision_data.maximum_level,
            "helium_ii_collision_fallback": "Mihalas above CCC shell limit",
            "atmosphere_convergence_status": qualified_status,
            "structure_convergence_status": status,
            "spectral_qualification": qualified_status == "spectrum-qualified",
            "atomic_data_sha256": identity,
            "formal_population_converged": state.converged,
            "formal_population_defect": getattr(state, "metadata", {}).get(
                "undamped_population_defect"
            ),
            "formal_electron_closure_residual": getattr(state, "metadata", {}).get(
                "electron_closure_residual"
            ),
            "population_state_layout": "structure, line_formation, line_formation_atmosphere",
            "independent_grid_validation": False,
            "full_physics_validation": False,
            "hydrogenic_linear_stark_components": PG1159_HYDROGENIC_STARK_COMPONENTS,
            "refinement_local_maximum_rosseland_depth": (
                config.refinement_local_maximum_rosseland_depth
                if config.refine_upper_atmosphere
                else None
            ),
            "local_energy_minimum_rosseland_depth": gate,
            "local_energy_gate_restarts": gate_restarts,
            "rydberg_upgrades": config.rydberg_upgrades,
            "formal_population_tolerance": (
                formal_tolerance if formal_model is not model else None
            ),
        },
        population_state=PG1159PopulationResult(
            result.population_state, state, atmosphere
        ),
    )
    if output is not None:
        from .common import save_model_result

        save_model_result(value, output)
    return value
