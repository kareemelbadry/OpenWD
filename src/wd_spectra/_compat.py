"""Small compatibility helpers for supported dependency versions."""

from __future__ import annotations

import numpy as np


try:
    trapezoid = np.trapezoid
except AttributeError:  # NumPy < 2.0
    trapezoid = np.trapz
