Electron thermal conductivity data
==================================

``potekhin_condtab21wd.dat`` is a verbatim copy of ``condtab21wd.dat``,
version 23 April 2021, published by Alexander Potekhin's
group at the Ioffe Institute:

  https://www.ioffe.ru/astro/conduct/condint.html

The current code reads its first, Z=1 (pure-hydrogen) block.  The entries are
log10 of the electron thermal conductivity in
erg cm^-1 s^-1 K^-1 on a log10 density / log10 temperature grid.  The ``wd``
table uses the weakly damped correction of Blouin et al. (2020) in the
partially degenerate regime.  The underlying treatment and interpolation are
described by Cassisi et al. (2007) and the references on the Ioffe page.

The original table assumes full ionization.  The atmosphere code therefore
uses it only as microscopic electron-transport data; its contribution is
negligible in the partially ionized photospheric layers of the cool-DA models
for which it is currently enabled.
