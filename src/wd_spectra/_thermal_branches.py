"""Frozen-field thermal branches of thin LTE layers.

Where opacity sources switch with temperature, a thin layer's radiative
energy balance in a fixed radiation field,
``Q(T) = integral kappa(T, P) (J - B(T)) dlambda``, can have several roots.
Cool DA upper layers have a hot root (H- bound-free heating against
free-free cooling) and a cold root (H2 collision-induced cooling against
H-H2 Lyman-alpha-wing heating), separated by an unstable one.  A layer that
a Newton solve leaves between them, or beside a root that has merged away,
has no nearby equation root.  These helpers locate the stable root on the
side its residual points to, as a restart seed.  They solve no structure
equations and certify nothing.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ._compat import trapezoid


def radiative_layers(convective_flux_fraction_by_interface, n_depth: int) -> np.ndarray:
    """Layers whose upper and lower interfaces carry no convective flux.

    Interface ``k`` is the upper face of layer ``k``; the last layer's lower
    face is the bottom boundary.
    """

    convective = np.asarray(convective_flux_fraction_by_interface, dtype=np.float64)
    if convective.shape != (n_depth,):
        raise ValueError("one convective flux fraction per interface is required")
    faces = np.append(np.abs(convective), 0.0)
    return (faces[:-1] == 0.0) & (faces[1:] == 0.0)


def stable_branch_temperature(
    temperature_grid, balance, temperature: float, heating: bool
) -> float | None:
    """Nearest stable root of a tabulated balance in the residual's direction.

    ``balance[k]`` is the net radiative heating at ``temperature_grid[k]``
    (increasing).  A stable root is a sign change from heating to cooling
    with increasing temperature.  With ``heating`` the lowest stable root
    above ``temperature`` is returned, otherwise the highest below.  None if
    there is none.
    """

    grid = np.asarray(temperature_grid, dtype=np.float64)
    values = np.asarray(balance, dtype=np.float64)
    if grid.ndim != 1 or values.shape != grid.shape or np.any(np.diff(grid) <= 0.0):
        raise ValueError("balance must be tabulated on an increasing temperature grid")
    roots = []
    for k in range(grid.size - 1):
        lower, upper = values[k], values[k + 1]
        if np.isfinite(lower) and np.isfinite(upper) and lower > 0.0 >= upper:
            roots.append(grid[k] + lower * (grid[k + 1] - grid[k]) / (lower - upper))
    if heating:
        above = [root for root in roots if root > temperature]
        return min(above) if above else None
    below = [root for root in roots if root < temperature]
    return max(below) if below else None


def frozen_field_branch_seed(
    temperature,
    wavelength,
    mean_intensity,
    cell_residual,
    stuck,
    temperature_grid,
    absorption_at: Callable[[np.ndarray], np.ndarray],
    planck: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, list[int]]:
    """Move each ``stuck`` layer to its stable frozen-field root.

    ``absorption_at(T)`` returns the true absorption (wavelength x depth) of
    the structure with every layer at the temperatures ``T`` and its own
    pressure; ``planck(wavelength, T)`` the matching Planck function.  The
    sign of ``cell_residual`` (net heating per layer, one fewer than the
    layers) chooses the direction.  Layers without such a root keep their
    temperature.  Returns the seed and the changed layers.
    """

    temperature = np.asarray(temperature, dtype=np.float64)
    residual = np.asarray(cell_residual, dtype=np.float64)
    stuck = np.asarray(stuck, dtype=bool)
    grid = np.asarray(temperature_grid, dtype=np.float64)
    mean = np.asarray(mean_intensity, dtype=np.float64)
    n_depth = temperature.size
    if residual.shape != (n_depth - 1,) or stuck.shape != (n_depth,):
        raise ValueError("residual has one value per cell and stuck one per layer")
    seed = temperature.copy()
    layers = np.flatnonzero(stuck[:-1])
    if layers.size == 0:
        return seed, []
    balance = np.empty((grid.size, layers.size))
    for k, trial in enumerate(grid):
        uniform = np.full(n_depth, trial)
        absorption = np.asarray(absorption_at(uniform), dtype=np.float64)[:, layers]
        source = np.asarray(planck(wavelength, uniform), dtype=np.float64)[:, layers]
        balance[k] = trapezoid(absorption * (mean[:, layers] - source), wavelength, axis=0)
    changed = []
    for column, layer in enumerate(layers):
        root = stable_branch_temperature(
            grid, balance[:, column], temperature[layer], residual[layer] > 0.0
        )
        if root is not None:
            seed[layer] = root
            changed.append(int(layer))
    return seed, changed
