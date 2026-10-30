"""Research-only pseudo-time proposal with exact algebraic ML2 elimination.

This cheap-model helper changes neither the transient equation nor its units.
It proposes temperatures on the model's compatible convection manifold; a
caller must still evaluate full physics and enforce all acceptance gates.
"""
import numpy as np


def compatible_thermal_model(model, state, thermal_rows):
    """Return (full state, residual, reduced Jacobian) for a temperature step."""
    n = model.n
    cache = [None, None]

    def evaluated(delta):
        if cache[0] is None or not np.array_equal(cache[0], delta):
            logt = state[:n]+delta
            physical = model.physical(logt, True)
            compatible = model.state_from_temperature(logt, physical)
            full = model.evaluate(compatible, True)
            residual, jacobian = thermal_rows(compatible, full)
            mapping = model.compatible_state_jacobian(logt, physical)
            if mapping.shape != (state.size, n):
                raise ValueError('Compatible-state tangent has the wrong shape')
            # Compatibility is enforced algebraically, not downweighted.
            if np.max(abs(residual[n:])) > 1e-9:
                raise ValueError('Compatible thermal trial violates algebraic closure')
            cache[:] = [delta.copy(), (compatible, residual, jacobian[:n]@mapping)]
        return cache[1]

    return evaluated
