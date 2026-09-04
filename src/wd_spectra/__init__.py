"""OpenWD: open white-dwarf atmosphere and spectrum calculations."""

from .models import (
    DAConfig,
    DABConfig,
    DBConfig,
    DZConfig,
    ModelData,
    ModelResult,
    compute_da,
    compute_dab,
    compute_db,
    compute_dz,
    load_atmosphere_checkpoint,
    save_model_result,
)
from .atmosphere import (
    Atmosphere,
    gray_helium_atmosphere,
    gray_hydrogen_atmosphere,
    gray_hydrogen_helium_atmosphere,
    helium_continuum_atmosphere,
    hydrogen_continuum_atmosphere,
    hydrogen_helium_continuum_atmosphere,
    radiative_equilibrium_helium_atmosphere,
    radiative_equilibrium_hydrogen_atmosphere,
    radiative_equilibrium_hydrogen_helium_atmosphere,
)
from .cool_da import (
    TREMBLAY_2013_LOG_ROSSELAND_DEPTH,
    TREMBLAY_2013_MEAN_3D_MINUS_1D_TEMPERATURE_K,
    atmosphere_with_tremblay_2013_mean_3d_temperature_difference,
    hydrostatic_atmosphere_with_tremblay_2013_mean_3d_temperature_difference,
    tremblay_2013_mean_3d_temperature_difference,
)
from .gaunt import hydrogen_bound_free_gaunt_factor, hydrogen_free_free_gaunt_factor
from .helium import (
    HELIUM_I_LINES,
    HELIUM_II_LINES,
    helium_continuum_mass_absorption_coefficient,
    helium_dissolved_level_pseudocontinuum_mass_absorption_coefficient,
    helium_ii_bound_free_linear_absorption_coefficient,
    helium_ii_dissolved_level_pseudocontinuum_mass_absorption_coefficient,
    helium_ii_line_mass_absorption_coefficient,
    helium_ii_rayleigh_scattering_cross_section,
    helium_i_line_mass_absorption_coefficient,
    helium_i_resonance_hwhm_angstrom,
    helium_i_resonance_line_mass_absorption_coefficient,
    helium_i_resonance_stark_hwhm_angstrom,
    helium_i_rydberg_bound_free_linear_absorption_coefficient,
    helium_minus_free_free_coefficient,
    helium_rayleigh_scattering_cross_section,
    helium_rayleigh_scattering_mass_coefficient,
    neutral_helium_photoionization_cross_section,
)
from .helium_ii_stark import read_helium_ii_stark_table
from .helium_molecular import (
    helium_dimer_ion_continuum_coefficient,
    helium_three_body_cia_linear_absorption_coefficient,
)
from .helium_stark import read_helium_stark_table
from .helium_neutral import deridder_van_rensbergen_helium_hwhm_angstrom
from .hydrogen_self import (
    BarklemSelfBroadeningTable,
    read_barklem_self_broadening_table,
)
from .mixture import hydrogen_helium_continuum_mass_absorption_coefficient
from .opacity import hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient
from .radiative_transfer import compiled_backend_available, emergent_flux
from .spectrum import (
    Spectrum,
    planck_lambda_angstrom,
    synthesize_balmer_spectrum,
    synthesize_gray_spectrum,
    synthesize_helium_spectrum,
    synthesize_hydrogen_helium_spectrum,
    synthesize_hydrogen_spectrum,
)
from .stark import (
    default_balmer_stark_table,
    default_brackett_stark_table,
    default_lyman_stark_table,
    default_paschen_stark_table,
)

__version__ = "0.1.1"

__all__ = [
    "Atmosphere",
    "BarklemSelfBroadeningTable",
    "DAConfig",
    "DABConfig",
    "DBConfig",
    "DZConfig",
    "HELIUM_I_LINES",
    "HELIUM_II_LINES",
    "ModelData",
    "ModelResult",
    "Spectrum",
    "atmosphere_with_tremblay_2013_mean_3d_temperature_difference",
    "compiled_backend_available",
    "compute_da",
    "compute_dab",
    "compute_db",
    "compute_dz",
    "convection",
    "default_balmer_stark_table",
    "default_brackett_stark_table",
    "default_lyman_stark_table",
    "default_paschen_stark_table",
    "deridder_van_rensbergen_helium_hwhm_angstrom",
    "emergent_flux",
    "gray_helium_atmosphere",
    "gray_hydrogen_atmosphere",
    "gray_hydrogen_helium_atmosphere",
    "helium_continuum_atmosphere",
    "hydrogen_bound_free_gaunt_factor",
    "hydrogen_continuum_atmosphere",
    "hydrogen_free_free_gaunt_factor",
    "hydrogen_helium_continuum_atmosphere",
    "hydrostatic_atmosphere_with_tremblay_2013_mean_3d_temperature_difference",
    "load_atmosphere_checkpoint",
    "planck_lambda_angstrom",
    "radiative_equilibrium_helium_atmosphere",
    "radiative_equilibrium_hydrogen_atmosphere",
    "radiative_equilibrium_hydrogen_helium_atmosphere",
    "read_barklem_self_broadening_table",
    "save_model_result",
    "synthesize_balmer_spectrum",
    "synthesize_gray_spectrum",
    "synthesize_helium_spectrum",
    "synthesize_hydrogen_helium_spectrum",
    "synthesize_hydrogen_spectrum",
    "tremblay_2013_mean_3d_temperature_difference",
]
