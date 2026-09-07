"""Stable one-shot interfaces for the initial OpenWD release."""

from .common import (
    AtmosphereConvergenceWarning,
    ModelData,
    ModelResult,
    NumericalResolution,
    Quality,
    atmosphere_convergence_status,
    default_wavelength_grid,
    load_atmosphere_checkpoint,
    numerical_resolution,
    save_model_result,
)
from .stellar import (
    DAConfig,
    DABConfig,
    DBConfig,
    DZConfig,
    compute_da,
    compute_dab,
    compute_db,
    compute_dz,
)
from .selection import PhysicsSelection, PhysicsSelectionPolicy, select_physics
from .automatic import ModelRun, run_model

__all__ = [
    "AtmosphereConvergenceWarning",
    "DAConfig",
    "DABConfig",
    "DBConfig",
    "DZConfig",
    "ModelData",
    "ModelResult",
    "ModelRun",
    "PhysicsSelection",
    "PhysicsSelectionPolicy",
    "select_physics",
    "run_model",
    "NumericalResolution",
    "Quality",
    "atmosphere_convergence_status",
    "compute_da",
    "compute_dab",
    "compute_db",
    "compute_dz",
    "default_wavelength_grid",
    "load_atmosphere_checkpoint",
    "numerical_resolution",
    "save_model_result",
]
