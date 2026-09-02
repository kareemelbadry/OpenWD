This package includes the Stark broadening tables described in Tremblay &
Bergeron (2009), ApJ, 696, 1755. The original package (March 2009) included
the first 10 Lyman and Balmer lines.

In this new version (16 November 2015), I have computed 10 extra Balmer lines
and 10 extra Lyman lines, as well as 19 Paschen lines and 10 Brackett lines.

The format of the tables is the same as that of Lemke (1997) so that they can
easily replace previous tables in the codes. Versions with and without Doppler
convolution are provided.

Non-zero goodness flags indicate that the code did not converge. The goodness
flags are defined as :

0   profile ok
-1  copied from next higher T
-2  copied from 2nd next higher T
+1  copied from next lower T
-10 copied from next lower ne
-22 copied from 2nd next lower ne, and 2nd next higher T
ect

Note that the non-ideal effects are not implemented in the profiles for the
first three Lyman lines (however, I have included new calculations for these
three lines as the original Lemke profiles has small glitches.) As explained
in Tremblay & Bergeron (2009), this is to be consistent with the
pseudo-continuum cutoff that is currently used in most atmosphere codes
(e.g. TLUSTY) at these wavelengths (occupation probability equal to unity).

Downloaded from 
https://warwick.ac.uk/fac/sci/physics/research/astro/people/tremblay/modelgrids/

Pier-Emmanuel Tremblay (P-E.Tremblay@warwick.ac.uk)
Pierre Bergeron (bergeron@astro.umontreal.ca)
March 2009, Updated November 2015
