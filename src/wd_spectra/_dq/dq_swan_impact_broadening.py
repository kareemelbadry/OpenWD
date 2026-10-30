"""Neutral-He impact broadening of the resolved Swan lines (EMPIRICAL opt-in).

Off by default (``DQConfig.swan_helium_broadening="none"``). Because the
Lorentzian is truncated at the Weisskopf detuning and the truncated area is
omitted, this removes 60-95% of the Swan strength at n_He = 1e22-1e23 cm^-3.
That improves cool/dense DQ band depths against SDSS, but it is not
strength-conserving line-shape physics; treat it as an effective model.


Each line keeps its exact thermal Gaussian and the cell-integrated Blouin
shift interval. It is further convolved with a He impact Lorentzian of
half width gamma = 0.07 cm^-1/bar (296 K/T)^0.5 at the neutral-He partial
pressure. These are the defaults in the checksum-pinned ExoMol 8states .def
file (``c2_profiles.helium_impact_hwhm(..., 'exomol_default')``); they are
generic, not a measured C2-He coefficient, and no factor is fitted.

The Lorentzian is restricted to the validity domain of impact theory,
|Delta omega| <= v/rho_W, exactly as for the DQ carbon lines
(``metals.metal_line_mass_absorption_coefficient(impact_validity_support=True)``).
Here rho_W follows from the line's own width, Gamma = 2 pi rho_W^2 v n_He, with the
C2-He mean relative speed v. The truncated area is omitted, not
redistributed. Red detunings carry the ground-state Boltzmann factor
exp(-h c |Delta nu|/kT), as ``red_wing_boltzmann`` does for carbon lines.

Numerics: [WINDOW_LOW, WINDOW_HIGH] cm^-1 is split into SEGMENTS log-spaced
segments (Doppler sigma varies by <=3% within one). For each, the lines that
can reach its queries are deposited on a uniform grid of step sigma/4 with
linear (cloud-in-cell) weights. They are convolved by FFT with the exact
Gaussian (segment-centre sigma), box and shift transforms and the transform of
the cell-integrated, truncated Lorentz kernel, then linearly interpolated. Lines outside the window (<3e-4 of the Swan strength at
3000-12000 K) keep the unbroadened exact thermal cell average.
"""
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len, rfftfreq

from wd_spectra.c2_profiles import helium_impact_hwhm
from wd_spectra.constants import BOLTZMANN, PI, LIGHT_SPEED
from .swan_strength_cache import CachedSwan

WINDOW_LOW, WINDOW_HIGH = 12000., 32000.
SEGMENTS = 16
PRESCRIPTION = 'exomol_default'
SECOND_RADIATION_CONSTANT_CM_K = 1.438776877
C2_HE_REDUCED_MASS_G = 24.0*4.002602/28.002602*1.66053906892e-24
DESCRIPTION = ('EMPIRICAL opt-in: He impact Lorentz (ExoMol 8states .def default 0.07 cm-1/bar, (296/T)^0.5) '
               'on exact thermal Swan lines; Weisskopf-limited support, area omitted; '
               'red side x exp(-hc|dnu|/kT); cell-integrated Blouin shift retained')


def impact_half_width(temperature, neutral_helium_density):
    """Lorentz HWHM in cm^-1."""
    return helium_impact_hwhm(temperature, neutral_helium_density, PRESCRIPTION)


def weisskopf_detuning(temperature, neutral_helium_density):
    """Impact-validity detuning v/rho_W in cm^-1 from the line's own width."""
    t = np.asarray(temperature, float)
    n = np.asarray(neutral_helium_density, float)
    gamma = impact_half_width(t, n)
    per_perturber = np.divide(4*PI*LIGHT_SPEED*gamma, n, out=np.zeros_like(gamma), where=n > 0)
    speed = np.sqrt(8*BOLTZMANN*t/(PI*C2_HE_REDUCED_MASS_G))
    with np.errstate(divide='ignore'):
        omega = np.sqrt(2*PI)*speed**1.5/np.sqrt(per_perturber)
    return omega/(2*PI*LIGHT_SPEED)


class ImpactBroadenedSwan(CachedSwan):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        inside = (self.nu >= WINDOW_LOW) & (self.nu <= WINDOW_HIGH)
        self._inside = inside
        # The out-of-window lines keep the production exact thermal cell
        # average (prefix sums), evaluated on a restricted copy of the list.
        self._outside_swan = CachedSwan(*args, **kwargs)
        for name in ('nu', 'ag', 'lower_energy'):
            values = np.ascontiguousarray(getattr(self, name)[~inside])
            values.setflags(write=False)
            setattr(self._outside_swan, name, values)
        self.broadened_calls = 0

    def set_strength_cache_bytes(self, budget):
        super().set_strength_cache_bytes(budget)
        if hasattr(self, '_outside_swan'):
            self._outside_swan.set_strength_cache_bytes(budget)

    def broadened_cell_average(self, wavelength, temperature, shift_left, shift_right,
                               neutral_helium_density):
        w = np.asarray(wavelength, dtype=float)
        t = float(temperature)
        if (w.ndim != 1 or np.any(~np.isfinite(w)) or np.any(w <= 0) or not np.isfinite(t)
                or t < self.table.temperature_K[0] or t > self.maximum_temperature
                or not np.isfinite(shift_left) or not np.isfinite(shift_right)
                or not np.isfinite(neutral_helium_density) or neutral_helium_density < 0):
            raise ValueError('Invalid broadened Swan state')
        self.broadened_calls += 1
        low, high = sorted((float(shift_left), float(shift_right)))
        span, middle = high - low, .5*(low + high)
        q = 1e8/w
        gamma = float(impact_half_width(t, neutral_helium_density))
        support = float(weisskopf_detuning(t, neutral_helium_density)) if gamma > 0 else 0.
        velocity = self._velocity_fraction(t)
        strength_all = self._line_strength(t)
        nu_in, s_in = self.nu[self._inside], strength_all[self._inside]
        result = np.zeros_like(q)
        edges = WINDOW_LOW*(WINDOW_HIGH/WINDOW_LOW)**(np.arange(SEGMENTS + 1)/SEGMENTS)
        reach_max = support + 10*WINDOW_HIGH*velocity
        segment = np.searchsorted(edges, q, side='right') - 1
        segment[(q < edges[0]) & (q >= edges[0] + low - reach_max)] = 0
        segment[(q >= edges[-1]) & (q <= edges[-1] + high + reach_max)] = SEGMENTS - 1
        for k in range(SEGMENTS):
            use = np.flatnonzero(segment == k)
            if len(use) == 0:
                continue
            sigma = np.sqrt(edges[k]*edges[k + 1])*velocity
            step = .25*sigma
            reach = support + 10*sigma
            qlo, qhi = q[use].min(), q[use].max()
            first = np.searchsorted(nu_in, qlo - high - reach - 2*step, side='left')
            last = np.searchsorted(nu_in, qhi - low + reach + 2*step, side='right')
            if last <= first:
                continue
            origin = min(qlo, nu_in[first]) - 2*step
            size = int(np.ceil((max(qhi, nu_in[last - 1]) - origin)/step)) + 3
            position = (nu_in[first:last] - origin)/step
            index = np.floor(position).astype(np.int64)
            frac = position - index
            strength = s_in[first:last]
            stick = (np.bincount(index, strength*(1 - frac), minlength=size)
                     + np.bincount(index + 1, strength*frac, minlength=size))[:size]/step
            margin = int(np.ceil((reach + abs(low) + abs(high) + span)/step)) + 4
            nfft = next_fast_len(size + 2*margin)
            f = rfftfreq(nfft, d=step)
            transform = (np.exp(-2*(PI*sigma*f)**2)*np.sinc(span*f)*np.exp(-2j*PI*f*middle))
            m = int(np.ceil(support/step)) if gamma > 0 else 0
            if m > 0:
                x = np.arange(-m, m + 1)*step
                kernel = (np.arctan((x + .5*step)/gamma) - np.arctan((x - .5*step)/gamma))/PI
                kernel = np.where(x < 0, kernel*np.exp(-SECOND_RADIATION_CONSTANT_CM_K*np.abs(x)/t), kernel)
                kernel = np.where(np.abs(x) <= support, kernel, 0.)
                wrapped = np.zeros(nfft)
                wrapped[:m + 1] = kernel[m:]
                wrapped[nfft - m:] = kernel[:m]
                transform = transform*rfft(wrapped)
            spectrum = irfft(rfft(stick, nfft)*transform, nfft)[:size]
            grid = origin + np.arange(size)*step
            result[use] = np.interp(q[use], grid, np.maximum(spectrum, 0.), left=0., right=0.)
        return result + self._outside_swan.cell_average(w, t, low, high)
