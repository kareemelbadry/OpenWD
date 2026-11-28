#!/usr/bin/env python3
"""Build the motional Stark table of the H-alpha and H-beta Zeeman components (n = 2 -> 3, 4).

An atom moving with v_perp across B feels the electric field F = v_perp B / c
in its rest frame. In a strong field the levels are non-degenerate, so the
shift is quadratic, dE = -alpha_perp F^2 / 2, and a thermal (two-dimensional
Maxwellian) v_perp gives each component a one-sided exponential profile with
mean shift s = Delta alpha kT beta^2 / M in atomic units, Delta alpha =
alpha_up - alpha_low, beta = B / 2.350518e9 G, M = m_p + m_e (Pavlov &
Meszaros 1993, ApJ 416, 752, Eqs. 2.12b, 2.23, 3.27-3.31; Rosato 2023, JQSRT
306, 108628, Eq. 3). This script tabulates Delta alpha for every H2db
H-alpha and H-beta component on 50-650 MG every 5 MG with the clean-room
solver ``hydrogen_magnetic.py`` (textbook Hamiltonian, infinite nuclear mass).

Delta alpha is NaN, and the zero-field Stark kernel is kept, where
  * a level lies above the lowest Landau threshold of an m +- 1 block: the
    motional field couples it to a continuum, there is no second-order shift
    (``flag_continuum``);
  * second order fails for a level of the pair: exact diagonalisation of
    H_B + F x (150 lowest states per block, m = -6..5) at F^2 = 3 <F^2> and
    20000 K moves the level by more than 20 % from second order or keeps less
    than 0.8 of it in one eigenstate; evaluated every 25 MG and taken from the
    nearest such field (``flag_second_order``);
  * the upper level lies below the lower one (``flag_negative``).

Exact profiles. Where second order fails (within one profile step, at least 5 MG, of a flagged
field, not above a Landau threshold, not negative, F = 0 wavelength in
3000-11000 A) the component gets the exact F-averaged profile instead, on a
coarser field grid (``--bstep``): eigenstates of H_B + F x in the centred
basis ``hydrogen_magnetic.coupled`` (|m| <= 60) for F^2 = x <F^2>, the two-
dimensional Maxwellian weighting each x by e^-x dx; the absorption from the
perturbed lower level into every eigenstate is |<k|r_q|low(F)>|^2 relative to
the F = 0 line (Pavlov & Meszaros 1993, Sect. 3). Lines are computed once at
T0 = 17 kK on x_ref cells (dx 0.05 to 6, 0.25 to 20) and reweighted for each
temperature node with x = x_ref T0 / T (F^2 is proportional to T). A component
owns the strength that falls inside its window: the Voronoi cell (halfway
points in wavelength) between its F = 0 line and the neighbouring F = 0 lines
from the same lower state in the same polarization (other table components
and every line with at least 1e-3 of the strongest); the outermost line of a
group gets a symmetric cell; no window reaches farther than 0.12 lambda0 from
its line (WINDOW_MAX; ~540 A at H-beta). The windows tile the spectrum, so no
strength is counted twice. Without the cap a window could take in a distant
chunk of another line's strength once a neighbouring line fell below the 1e-3
cut (H-beta sigma- 2s0-4f-1: +0.10 at +1590 A between 425 and 430 MG, which
broke the interpolation in B); with it, far strength is left to its own
line. The table stores each profile as 96 quantiles of its blue
(dlambda <= 0) and 96 of its red part, dense in the tails (levels (1 -
cos(pi j/95))/2), plus the strength share of each part, at six temperatures
(8-40 kK; ``--temperatures`` chooses others at packing);
the shares sum to the window share (strength in the window over the F = 0
line). The lines themselves (dlambda, strength, x cell) go to
``<output>_lines.npz`` (not shipped); ``--from-lines`` repacks the table from
it with another ``--quantiles`` without diagonalising again, and merges the
profile fields of several lines files (a local refinement in B). Caveat: a far,
weak part of a window (for H-beta sigma- 2s0-4f-1 at 366 MG, 0.026 of the
line at +600-1450 A) is spread across the quantile gap to it, which moves
the red mean of the whole window (0.4 A at the nodes, ~2 A after
interpolation in B) but no feature. A component
whose lower level mixes with the n = 3-4 manifold (2p+1 above ~300 MG: less
than 0.8 of it in one eigenstate at some x_ref <= 3) is flagged
(``flag_lower_mixed``) and keeps the zero-field kernel. Quantiles and shares
are interpolated linearly in B and T at run time. At 360-370 MG a 5 MG step
halves the interpolation error of a 10 MG step (H-beta sigma- 2s0-4f-1 red
mean 1.2 vs 2.4 A; the blue/red split of H-alpha pi 2p0-3d0, whose line sits
on dlambda = 0, 0.04 vs 0.08); window shares interpolate to 0.002 at both.

Labels are H2db's (lower m, z-parity, nu, upper m, z-parity, nu) with the
physical sign of m, as stored by ``read_h2db_transition_database``.

Usage::

    python tools/build_motional_stark_table.py --jobs 12 --bstep 5 \
        src/wd_spectra/data/runtime/cache/motional-stark/balmer_motional_stark.npz
    python tools/build_motional_stark_table.py OUT.npz --from-lines A_lines.npz B_lines.npz

Writes the table, ``<output>_lines.npz`` and ``<output>.json`` (summary
numbers). Deterministic.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np  # noqa: E402
import scipy.linalg as la  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as sl  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hydrogen_magnetic as hm  # noqa: E402

SECOND_ORDER_STEP, SECOND_ORDER_TEMPERATURE, TOLERANCE = 25.0, 20000.0, 0.2
MESH = dict(N=60, h=0.6, lmax=60)
M_ATOM = hm.M_P + 1.0
HC = hm.HC_EV_A / hm.HARTREE_EV  # h c [Hartree A]
MMAX, T0, NQ = 60, 17000.0, 96
TEMPERATURES = np.array([8000.0, 11000.0, 15000.0, 20000.0, 28000.0, 40000.0])
X_EDGES = np.concatenate((np.arange(0.0, 6.0, 0.05), np.arange(6.0, 20.0 + 1e-9, 0.25)))
PROFILE_WAVELENGTHS = (3000.0, 11000.0)
WINDOW_MAX = 0.12  # window half width at most WINDOW_MAX lambda0 (module docstring)


def states(nmax):
    """{(n, l, m): (m, z-parity, k)} of every state with n <= nmax (k = H2db nu - 1)."""
    out = {}
    for m in range(-nmax + 1, nmax):
        for par in (1, -1):
            for k, (n, l) in enumerate(hm.labels(m, par, nmax)):
                out[n, l, m] = (m, par, k)
    return out


def components():
    """H-alpha and H-beta (lower, upper) state pairs allowed in electric-dipole order."""
    st = states(4)
    low = [v for (n, _, _), v in st.items() if n == 2]
    return [(a, b) for nu in (3, 4) for a in low for b in [v for (n, _, _), v in st.items() if n == nu]
            if abs(b[0] - a[0]) == 1 and b[1] == a[1] or b[0] == a[0] and b[1] == -a[1]]


def second_order_failures(task):
    """Levels whose shift at F^2 = 3 <F^2> departs from second order (see the module docstring)."""
    B, levels = task
    beta = B * 1e6 / hm.B0_G
    H = hm.Hydrogen(beta, **MESH)
    frms = np.sqrt(2 * SECOND_ORDER_TEMPERATURE / hm.K_HARTREE * beta ** 2 / M_ATOM)
    bad = set()
    for par in (1, -1):
        ms = list(range(-6, 6))
        E0 = np.concatenate([H.block(m, par)[0][:150] for m in ms])
        offs = np.cumsum([0] + [150] * len(ms))
        X = np.zeros((E0.size, E0.size))
        for i, m in enumerate(ms[:-1]):
            Xm = H.xmat(m, par)[:150, :150].copy()
            H.xmats.clear()  # memory: one full xmat at a time
            X[offs[i + 1]:offs[i + 2], offs[i]:offs[i + 1]] = Xm
            X[offs[i]:offs[i + 1], offs[i + 1]:offs[i + 2]] = Xm.T
        f = np.sqrt(3.0) * frms
        w, V = np.linalg.eigh(np.diag(E0) + f * X)
        for m, p, k in levels:
            if p != par:
                continue
            j = offs[ms.index(m)] + k
            dE = E0 - E0[j]
            alpha = 2 * np.sum(X[dE != 0, j] ** 2 / dE[dE != 0])
            overlap = V[j, :] ** 2
            kk = int(np.argmax(overlap))
            pert = -0.5 * alpha * f * f
            if pert == 0 or abs((w[kk] - E0[j]) / pert - 1) > TOLERANCE or overlap[kk] < 0.8:
                bad.add((m, p, k))
    return bad


def second_order_field(task):
    """F = 0 wavelengths [A], Delta alpha and continuum flags of every pair at one field."""
    B, pairs = task
    beta = B * 1e6 / hm.B0_G
    H = hm.Hydrogen(beta, **MESH)
    threshold = lambda mm: 0.5 * beta * (abs(mm) + mm + 1)  # lowest Landau level of block mm [Hartree]  # noqa: E731
    out = []
    for a, b in pairs:
        Ea, Eb = H.block(*a[:2])[0][a[2]], H.block(*b[:2])[0][b[2]]
        out.append((hm.HC_EV_A / ((Eb - Ea) * hm.HARTREE_EV), H.alpha(*b) - H.alpha(*a),
                    any(E > min(threshold(s[0] - 1), threshold(s[0] + 1)) for s, E in ((a, Ea), (b, Eb)))))
    return out


def _eigsh(H, k, sigma):
    """Shift-invert eigenpairs nearest sigma; fixed start vector, so the result is deterministic."""
    return sl.eigsh(H, k=min(k, H.shape[0] - 2), sigma=sigma, v0=np.ones(H.shape[0]))


def windows(C, ops, pairs, need):
    """Voronoi windows (lo, hi) [A] at F = 0 and the F = 0 line data of the needed pairs (module docstring)."""
    idx = lambda s: C[s[1]]['off'][C[s[1]]['ms'].index(s[0])] + s[2]  # noqa: E731
    out = {}
    for i in need:
        a, b = pairs[i]
        q, ja, jb = b[0] - a[0], idx(a), idx(b)
        z0 = ops[a[1], q][:, [ja]].toarray().ravel() ** 2
        e0 = C[b[1]]['E0'] - C[a[1]]['E0'][ja]
        own = {idx(bb) for aa, bb in pairs if aa == a and bb[0] - aa[0] == q}
        ok = e0 > 0
        lines = np.flatnonzero(ok & ((z0 >= 1e-3 * z0[ok].max()) | np.isin(np.arange(z0.size), list(own))))
        lam = np.sort(HC / e0[lines])
        lam0 = HC / e0[jb]
        k = int(np.argmin(np.abs(lam - lam0)))
        lo = 0.5 * (lam[k - 1] + lam0) if k > 0 else None
        hi = 0.5 * (lam[k + 1] + lam0) if k + 1 < lam.size else None
        if lo is None and hi is None:
            continue
        lo, hi = (lo if lo is not None else 2 * lam0 - hi), (hi if hi is not None else 2 * lam0 - lo)
        lo, hi = max(lo, (1.0 - WINDOW_MAX) * lam0), min(hi, (1.0 + WINDOW_MAX) * lam0)
        out[i] = dict(ja=ja, jb=jb, pl=a[1], pu=b[1], q=q, d0=z0[jb], lam0=lam0, lo=lo, hi=hi)
    return out


def exact_profiles(task):
    """Quantile profiles of the needed pairs at one field (module docstring)."""
    B, pairs, need = task
    t0 = time.perf_counter()
    beta = B * 1e6 / hm.B0_G
    C = {p: hm.coupled(beta, p, MMAX) for p in (1, -1)}
    O = hm.dipole_operators(C)
    for c in C.values():  # sparse from here on (block tridiagonal in m)
        c['Xs'] = sp.csr_matrix(c.pop('X'))
        del c['V']
    rows = windows(C, O, pairs, need)
    frms = np.sqrt(2 * T0 / hm.K_HARTREE * beta ** 2 / M_ATOM)
    xr = 0.5 * (X_EDGES[1:] + X_EDGES[:-1])
    lines = {i: [] for i in rows}
    mixed = {i: False for i in rows}
    kept = {}  # eigenpairs found in each merged window in the previous x cell: start k from there
    for cell, x in enumerate(xr):
        f = np.sqrt(x) * frms
        H = {p: (sp.diags(c['E0']) + f * c['Xs']).tocsc() for p, c in C.items()}
        low = {}
        for i, r in rows.items():  # perturbed lower level: the eigenstate with most of it
            if (r['pl'], r['ja']) not in low:
                w, V = _eigsh(H[r['pl']], 6, C[r['pl']]['E0'][r['ja']])
                k = int(np.argmax(V[r['ja']] ** 2))
                low[r['pl'], r['ja']] = (w[k], V[:, k], V[r['ja'], k] ** 2)
        for p in (1, -1):
            iv = sorted([low[r['pl'], r['ja']][0] + HC / r['hi'], low[r['pl'], r['ja']][0] + HC / r['lo']]
                        for r in rows.values() if r['pu'] == p)
            if not iv:
                continue
            merged = [iv[0]]
            for lo, hi in iv[1:]:
                if lo <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            ws, Vs = [], []
            for j, (lo, hi) in enumerate(merged):  # every eigenpair inside [lo, hi]: the k nearest the centre c
                E0 = C[p]['E0']  # contain them all once one lies farther from c than the half width
                c, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
                k = max(int(np.sum((E0 > lo - 2e-3) & (E0 < hi + 2e-3))), kept.get((p, j), 0)) + 10
                while k <= 300:  # above ~300 pairs the dense solver is faster (shift-invert cost grows as k^2)
                    w, V = _eigsh(H[p], k, c)
                    if np.abs(w - c).max() > half or k >= E0.size - 2:
                        break
                    k *= 2
                if k > 300:
                    w, V = la.eigh(H[p].toarray(), subset_by_value=(lo, hi))
                keep = (w >= lo) & (w <= hi)
                kept[p, j] = int(keep.sum())
                ws.append(w[keep]), Vs.append(V[:, keep])
            w, V = np.concatenate(ws), np.hstack(Vs)
            for i, r in rows.items():
                if r['pu'] != p:
                    continue
                wl, vl, overlap = low[r['pl'], r['ja']]
                mixed[i] |= bool(x <= 3.0 and overlap < 0.8)
                e = w - wl
                ok = e > 0
                lam = HC / e[ok]
                amp = ((V.T @ (O[r['pl'], r['q']] @ vl)) ** 2 / r['d0'])[ok]
                inside = (lam > r['lo']) & (lam < r['hi'])
                lines[i].append((lam[inside] - r['lam0'], amp[inside], np.full(inside.sum(), cell)))
    out = {i: dict(mixed=mixed[i], lines=[np.concatenate([c[j] for c in lines[i]]) for j in range(3)]) for i in rows}
    print('B = %g MG: %d exact profiles in %.0f s' % (B, len(out), time.perf_counter() - t0), flush=True)
    return out


def pack(lines, nq, temperatures):
    """Quantile table from the lines file: per component, field and temperature node, nq quantiles of the blue
    (dlambda <= 0) and of the red part at levels (1 - cos(pi j/(nq - 1)))/2 (midpoint cumulative sums), and the
    strength share of each part; each line weighted by e^-x dx of its x cell, x = x_ref T0 / T, normalised, and by
    omega/omega0 = lam0/(lam0 + dlambda): the lines file holds |d|^2 over the F = 0 line's, the table oscillator strengths."""
    levels = 0.5 * (1.0 - np.cos(np.pi * np.arange(nq) / (nq - 1)))
    edges = lines['x_edges']
    xr, count = 0.5 * (edges[1:] + edges[:-1]), lines['line_count']
    at = np.searchsorted(lines['field_megagauss'], lines['profile_field_megagauss'])  # profile field in the 5 MG grid
    end = np.cumsum(count).reshape(count.shape)
    q = np.full(count.shape + (temperatures.size, 2, nq), np.nan, np.float32)
    share = np.full(count.shape + (temperatures.size, 2), np.nan, np.float32)
    for i, k in zip(*np.nonzero(count)):
        part = slice(end[i, k] - count[i, k], end[i, k])
        dl, amp, cell = lines['line_dl'][part], lines['line_amp'][part], lines['line_cell'][part]
        lam0 = lines['wavelength_infinite_mass_angstrom'][i, at[k]]
        keep = np.abs(dl) <= WINDOW_MAX * lam0  # lines files built before the cap
        dl, cell = dl[keep], cell[keep]
        amp = amp[keep] * lam0 / (lam0 + dl)  # |d|^2 ratio -> oscillator-strength ratio, f ~ omega |d|^2
        for t, T in enumerate(temperatures):
            wx = np.exp(-xr * T0 / T) * np.diff(edges)
            a = amp * (wx / wx.sum())[cell]
            for side, sel in enumerate((dl <= 0.0, dl > 0.0)):
                share[i, k, t, side] = a[sel].sum()
                if share[i, k, t, side] > 0.0:
                    o = np.argsort(dl[sel], kind='stable')
                    c = (np.cumsum(a[sel][o]) - 0.5 * a[sel][o]) / a[sel].sum()
                    q[i, k, t, side] = np.interp(levels, c, dl[sel][o])
                else:
                    q[i, k, t, side] = 0.0
    table = dict({key: lines[key] for key in TABLE_KEYS}, profile_temperature_kelvin=temperatures)
    np_ = dict(profile_quantile_levels=levels, profile_blue_angstrom=q[..., 0, :], profile_red_angstrom=q[..., 1, :],
               profile_blue_share=share[..., 0], profile_red_share=share[..., 1])
    return dict(table, **np_)


TABLE_KEYS = ('field_megagauss', 'labels', 'wavelength_infinite_mass_angstrom', 'delta_alpha', 'delta_alpha_unflagged',
              'flag_continuum', 'flag_second_order', 'flag_negative', 'profile_field_megagauss',
              'flag_lower_mixed')


def merge(parts):
    """One lines file from several builds with the same components and x cells (e.g. extra
    profile fields to refine B locally); the second-order arrays are those of the first."""
    out = dict(parts[0])
    nodes = [(j, k) for j, p in enumerate(parts) for k in range(p['profile_field_megagauss'].size)]
    fields = np.array([parts[j]['profile_field_megagauss'][k] for j, k in nodes])
    order = np.argsort(fields, kind='stable')
    if np.any(np.diff(fields[order]) == 0.0):
        raise ValueError('a profile field appears in two lines files')
    ends = [np.cumsum(p['line_count']).reshape(p['line_count'].shape) for p in parts]
    take = [(j, slice(ends[j][i, k] - parts[j]['line_count'][i, k], ends[j][i, k]))
            for i in range(out['labels'].shape[0]) for j, k in (nodes[g] for g in order)]
    for key in ('line_dl', 'line_amp', 'line_cell'):
        out[key] = np.concatenate([parts[j][key][part] for j, part in take])
    for key in ('line_count', 'flag_lower_mixed'):
        out[key] = np.concatenate([p[key] for p in parts], axis=1)[:, order]
    out['profile_field_megagauss'] = fields[order]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('output')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--bstep', type=float, default=10.0, help='field step of the exact profiles [MG]')
    ap.add_argument('--fields', default='50:650', help='field range LO:HI [MG] (second order every 5 MG, profiles every --bstep)')
    ap.add_argument('--quantiles', type=int, default=NQ, help='quantiles per side in the table')
    ap.add_argument('--temperatures', default=','.join('%g' % T for T in TEMPERATURES),
                    help='temperature nodes of the table [K], comma separated')
    ap.add_argument('--from-lines', nargs='+', help='only pack the table from these lines files (merged)')
    args = ap.parse_args()
    lines_path = os.path.splitext(args.output)[0] + '_lines.npz'
    t0 = t1 = t2 = time.perf_counter()
    if args.from_lines:
        parts = []
        for path in args.from_lines:
            with np.load(path) as values:
                parts.append(dict(values))
        lines = merge(parts)
    else:
        lo, hi = (float(v) for v in args.fields.split(':'))
        fields = np.arange(lo, hi + 1e-9, 5.0)
        pfields = np.arange(lo, hi + 1e-9, args.bstep)
        pairs = components()
        so_fields = np.unique([SECOND_ORDER_STEP * round(B / SECOND_ORDER_STEP) for B in fields])
        levels = {v for pair in pairs for v in pair}
        with Pool(args.jobs) as pool:
            failures = dict(zip(so_fields, pool.map(second_order_failures, [(B, levels) for B in so_fields], chunksize=1)))
            res = np.array(pool.map(second_order_field, [(B, pairs) for B in fields], chunksize=1))  # [field, pair, 3]
            lam, dalpha, cont = res[:, :, 0].T, res[:, :, 1].T, res[:, :, 2].T.astype(bool)
            second = np.array([[a in failures[so_fields[np.argmin(np.abs(so_fields - B))]] or
                                b in failures[so_fields[np.argmin(np.abs(so_fields - B))]] for B in fields]
                               for a, b in pairs])
            negative = ~(lam > 0)
            t1 = time.perf_counter()
            near = np.abs(pfields[:, None] - fields[None, :]) <= max(args.bstep, 5.0) + 1e-9  # [profile field, field]
            at = np.argmin(np.abs(fields[None, :] - pfields[:, None]), axis=1)  # nearest second-order field
            need = (second.astype(float) @ near.T > 0) & ~cont[:, at] & ~negative[:, at] & \
                (lam[:, at] > PROFILE_WAVELENGTHS[0]) & (lam[:, at] < PROFILE_WAVELENGTHS[1])
            tasks = [(B, pairs, list(np.flatnonzero(need[:, k]))) for k, B in enumerate(pfields)]
            exact = pool.map(exact_profiles, tasks, chunksize=1)
        t2 = time.perf_counter()
        mixed = np.zeros((len(pairs), pfields.size), bool)
        count = np.zeros((len(pairs), pfields.size), np.int64)
        parts = []
        for i in range(len(pairs)):  # lines in (component, field) order, counted per cell of count
            for k, ex in enumerate(exact):
                if i in ex:
                    mixed[i, k] = ex[i]['mixed']
                    if not mixed[i, k]:
                        count[i, k] = ex[i]['lines'][0].size
                        parts.append(ex[i]['lines'])
        cat = [np.concatenate([p[j] for p in parts]) if parts else np.zeros(0) for j in range(3)]
        lines = dict(field_megagauss=fields, labels=np.array([(a[0], a[1], a[2] + 1, b[0], b[1], b[2] + 1)
                                                               for a, b in pairs], dtype=np.int64),
                     wavelength_infinite_mass_angstrom=lam,
                     delta_alpha=np.where(cont | second | negative, np.nan, dalpha), delta_alpha_unflagged=dalpha,
                     flag_continuum=cont, flag_second_order=second, flag_negative=negative,
                     profile_field_megagauss=pfields, flag_lower_mixed=mixed,
                     x_edges=X_EDGES, line_count=count, line_dl=cat[0].astype(np.float32),
                     line_amp=cat[1].astype(np.float32), line_cell=cat[2].astype(np.uint8))
        np.savez_compressed(lines_path, **lines)
    table = pack(lines, args.quantiles, np.array([float(T) for T in args.temperatures.split(',')]))
    np.savez_compressed(args.output, **table)
    share = table['profile_blue_share'] + table['profile_red_share']
    labels, mixed = table['labels'], table['flag_lower_mixed']
    summary = dict(
        fields_megagauss=[float(table['field_megagauss'][0]), float(table['field_megagauss'][-1])],
        profile_fields_megagauss=table['profile_field_megagauss'].tolist(),
        temperatures_kelvin=table['profile_temperature_kelvin'].tolist(), quantiles_per_side=args.quantiles,
        components=int(labels.shape[0]), second_order_valid_cells=int(np.isfinite(table['delta_alpha']).sum()),
        exact_profiles=int(np.isfinite(share[..., 0]).sum()), lower_mixed=int(mixed.sum()),
        window_share_range=[float(np.nanmin(share)), float(np.nanmax(share))] if np.isfinite(share).any() else None,
        components_with_profiles=[' '.join(map(str, labels[i])) for i in np.flatnonzero(np.isfinite(share).any((1, 2)))],
        components_lower_mixed=[' '.join(map(str, labels[i])) for i in np.flatnonzero(mixed.any(1))],
        table_megabytes=round(os.path.getsize(args.output) / 1e6, 2),
        wall_seconds=dict(second_order=round(t1 - t0, 1), exact=round(t2 - t1, 1)), jobs=args.jobs)
    with open(os.path.splitext(args.output)[0] + '.json', 'w') as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
