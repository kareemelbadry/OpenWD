"""Historical, independently normalized C2 C–A opacity (no fitted scale).

Cooper 1979 electronic strength and Nicholls 1965 Franck–Condon factors,
with ExoMol lower-state populations and the unchanged 8states partition.
The finite-bin rigid-rotor envelope is approximate, not a modern line list.
No Swan pressure-shift law is silently assigned to this different system.
"""
import numpy as np
from wd_spectra.carbon_molecular import read_c2_cross_section_table
from .data import data_root


def load_ca_table(parent):
    table = read_c2_cross_section_table(data_root()/'c2-ca-historical.npz')
    for key in ('partition_temperature_K', 'partition_function'):
        if not np.array_equal(getattr(table, key), getattr(parent, key)):
            raise ValueError('C–A and chemistry partition conventions differ')
    return table
