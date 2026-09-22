"""Planck-field reference for PG 1159, without a transfer iteration."""
from dataclasses import replace
import numpy as np
from .pg1159 import PG1159NLTEState, pg1159_composition_atmosphere
from .helium_nlte import solve_coupled_helium_statistical_equilibrium
from .light_metal_nlte import (
    solve_light_metal_ionization_nlte,
    solve_reduced_light_metal_levels_nlte,
)
from .spectrum import planck_lambda_angstrom


# Dimensionless temperature shape from the accepted legacy PG 1424+535
# He/C/O structure.  Keeping the small table in code makes initialization
# deterministic and independent of a user checkpoint.  Scaling by Teff gives
# the central solver a line-blanketed surface rather than the Eddington-gray
# plateau that left optical He II cores tens of percent away from the legacy
# spectrum after the nominal residual gates had passed.
_REFERENCE_COLUMN_MASS = np.asarray(
    (
        4.995373961238447e-08, 1.049871600607537e-07,
        2.206432549634116e-07, 4.636778547630472e-07,
        9.742756199182956e-07, 2.046547493194278e-06,
        4.296380353989110e-06, 9.008627827144111e-06,
        1.884473405005902e-05, 3.925172344160376e-05,
        8.119596672717270e-05, 1.664219626147216e-04,
        3.377292154565651e-04, 6.789406365961740e-04,
        1.351101017457602e-03, 2.650951016754623e-03,
        5.094226533516068e-03, 9.521868511609206e-03,
        1.722343360710535e-02, 3.006446014992972e-02,
        5.057801745307360e-02, 8.248011209620722e-02,
        1.329442111479233e-01, 2.130574218660367e-01,
        3.444698515889259e-01, 5.732533357988648e-01,
        9.777574571781544e-01, 1.711443210807157e00,
        3.169027276024599e00, 6.637307239113452e00,
        1.610484004300765e01, 4.228820981840455e01,
    ),
    dtype=float,
)
_REFERENCE_TEMPERATURE_OVER_TEFF = np.asarray(
    (
        .8373298783608293, .8373030664928965, .8372547954724631,
        .8371810824078457, .8370901939850252, .8369984255609451,
        .8369218945540892, .8368667749222433, .8368334697984371,
        .8368108912976464, .8367647577692846, .7549100788580022,
        .6699086695690858, .5907841386353923, .5642189573367120,
        .5983838110422716, .6344474507612490, .6630203293751973,
        .6958041985492877, .7348653318891967, .7806126811985029,
        .8346453945418758, .9003149016877334, .9793873336321080,
        1.0760968792057237, 1.1995242691347838, 1.3570289006305707,
        1.5592452394004470, 1.7032777408127562, 2.0350229243745740,
        2.4410349547507930, 2.9337955481678870,
    ),
    dtype=float,
)
_REFERENCE_MIGRATED_FLUX_NORMALIZATION = 1.0885


def reference_temperature_seed(model, atmosphere):
    """Apply a checkpoint-free, Teff-scaled PG1159 non-gray temperature seed."""
    column_mass = np.asarray(atmosphere.column_mass, dtype=float)
    ratio = np.exp(
        np.interp(
            np.log(column_mass),
            np.log(_REFERENCE_COLUMN_MASS),
            np.log(_REFERENCE_TEMPERATURE_OVER_TEFF),
        )
    )
    # The reference ends in the diffusion regime. Continue its local
    # T proportional to m^(1/4) slope on deeper public grids.
    deep = column_mass > _REFERENCE_COLUMN_MASS[-1]
    ratio[deep] = _REFERENCE_TEMPERATURE_OVER_TEFF[-1] * (
        column_mass[deep] / _REFERENCE_COLUMN_MASS[-1]
    ) ** 0.25
    seeded = model.rebuild_atmosphere(
        atmosphere,
        atmosphere.effective_temperature
        * _REFERENCE_MIGRATED_FLUX_NORMALIZATION
        * ratio,
        None,
    )
    return replace(
        seeded,
        metadata={
            **seeded.metadata,
            "pg1159_reference_temperature_initialization": {
                "reference": "legacy PG 1424+535 He/C/O T(column-mass) shape",
                "scaled_by_effective_temperature": True,
                "migrated_flux_normalization": (
                    _REFERENCE_MIGRATED_FLUX_NORMALIZATION
                ),
                "runtime_checkpoint_loaded": False,
                "equilibrium_certified": False,
            },
        },
    )


def planck_state(model, atmosphere, wavelength):
    a, metal = pg1159_composition_atmosphere(
        atmosphere, model.atomic_database, model.mass_fractions
    )
    h = model.helium_model
    helium = solve_coupled_helium_statistical_equilibrium(
        a,
        h.collision_data,
        maximum_helium_ii_level=h.maximum_helium_ii_level,
        helium_i_collision_data=h.helium_i_collision_data,
        hydrogenic_collision_model=h.hydrogenic_collision_model,
        neutral_collision_strength_scale=h.neutral_collision_strength_scale,
    )
    mean = planck_lambda_angstrom(wavelength[:, None], a.temperature[None, :])
    light = solve_light_metal_ionization_nlte(
        a,
        model.atomic_database,
        metal,
        model.photoionization_database,
        wavelength,
        mean,
        elements=model.nlte_metal_elements,
        photoionization_threshold_data=model.trace_photoionization_threshold_data,
    )
    levels = []
    for element, prefix in [("C", "carbon"), ("O", "oxygen")]:
        kwargs = {
            key: getattr(model, prefix + "_" + key)
            for key in (
                "photoionization_threshold_data",
                "formal_level_mapping",
                "formal_lte_parent_mapping",
                "continuum_parent_mapping",
                "lte_level_reservoir",
                "lte_bound_bound_couplings",
                "effective_dielectronic_couplings",
                "collision_data",
            )
        }
        levels.append(
            solve_reduced_light_metal_levels_nlte(
                a,
                getattr(model, prefix + "_population_atomic_database"),
                metal,
                model.photoionization_database,
                wavelength,
                mean,
                element,
                getattr(model, prefix + "_levels_per_charge"),
                **kwargs
            )
        )
    return (
        a,
        PG1159NLTEState(
            helium,
            metal,
            light,
            *levels,
            metadata={
                "metal_population_converged": True,
                "metal_population_iterations": 1,
                "metal_population_relative_change": 0.0,
                "undamped_population_defect": 0.0,
                "electron_closure_residual": 0.0,
                "population_reference": "local Planck field",
            }
        ),
    )


def gray_opacity_seed(model, atmosphere, *, iterations=4, wavelength_points=300):
    """Precondition a fresh continuum seed with its own He/C/O opacity.

    Only the provisional temperature is changed. The requested Teff, gravity,
    mixture and hydrostatic mass/pressure grid stay fixed. A sparse opacity
    quadrature suffices for this gray estimate; it never certifies equilibrium.
    The common solver subsequently evaluates the complete structural grid.
    """
    from dataclasses import replace
    import logging
    import numpy as np
    from ._rosseland import rosseland_mean_from_opacity_grid

    logger = logging.getLogger(__name__)
    wavelength = np.geomspace(10., 200000., wavelength_points)
    current = atmosphere
    for iteration in range(iterations):
        current, population = planck_state(model, current, wavelength)
        coefficients = model.transfer_coefficients(current, wavelength, population)
        rosseland = rosseland_mean_from_opacity_grid(
            wavelength, coefficients.total_extinction, current.temperature)
        surface = rosseland[0] * current.column_mass[0]
        tau = np.r_[surface, surface + np.cumsum(
            .5 * (rosseland[1:] + rosseland[:-1]) * np.diff(current.column_mass))]
        gray_temperature = current.effective_temperature * (.75 * (tau + 2./3.)) ** .25
        temperature = np.sqrt(current.temperature * gray_temperature)
        logger.info('PG1159 gray opacity initialization %d: maximum log-T change %.4g',
            iteration + 1, np.max(abs(np.log(temperature / current.temperature))))
        current = model.rebuild_atmosphere(current, temperature, None)
    return replace(current, metadata={**current.metadata,
        'pg1159_gray_opacity_initialization': {
            'iterations': iterations, 'wavelength_points': wavelength_points,
            'mass_and_pressure_grid_preserved': True,
            'equilibrium_certified': False,
        }})
