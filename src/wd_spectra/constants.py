"""Physical constants in cgs units.

Keeping the constants local and explicit makes it possible to reproduce a
calculation without depending on a particular version of an external units
package.
"""

from __future__ import annotations

import math

BOLTZMANN = 1.380_649e-16  # erg K^-1
BOHR_RADIUS = 5.291_772_105_44e-9  # cm
FINE_STRUCTURE_CONSTANT = 7.297_352_5643e-3
PLANCK = 6.626_070_15e-27  # erg s
LIGHT_SPEED = 2.997_924_58e10  # cm s^-1
ELECTRON_MASS = 9.109_383_7139e-28  # g
ELEMENTARY_CHARGE_ESU = 4.803_204_712_570_263e-10  # statcoulomb
HYDROGEN_MASS = 1.673_557_5e-24  # g (neutral atomic hydrogen)
HYDROGEN_IONIZATION_ENERGY = 13.598_434_599_702 * 1.602_176_634e-12  # erg
HELIUM_MASS = 6.646_476_989_051e-24  # g (neutral 4He atom)
HELIUM_FIRST_IONIZATION_ENERGY = 24.587_389_011 * 1.602_176_634e-12  # erg
HELIUM_SECOND_IONIZATION_ENERGY = 54.417_765_528_2 * 1.602_176_634e-12  # erg
STEFAN_BOLTZMANN = 5.670_374_419e-5  # erg cm^-2 s^-1 K^-4
THOMSON_CROSS_SECTION = 6.652_458_7051e-25  # cm^2
PI = math.pi
