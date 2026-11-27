"""Rebuild the HELIUM_I_IMPACT_LINES literal in wd_spectra/helium_impact.py.

Inputs (downloaded, not bundled):
  NIST_TSV     NIST ASD v5.12 He I lines 2500-400000 A, vacuum wavelengths,
               tab-delimited (lines1.pl?spectra=He+I&low_w=2500&upp_w=400000
               &unit=0&format=3&show_av=3&f_out=on&conf_out=on&term_out=on
               &enrg_out=on&J_out=on&g_out=on&allowed_out=1&line_out=0).
  STARKB_HTML  STARK-B result page for its "He I, SCP" dataset: all
               transitions and temperatures at a perturber density of 1e16.

Every STARK-B transition redward of 7300 A (air) becomes one line: the NIST
fine-structure components of its LS multiplet, plus the spin-changing
multiplet from the same lower term to the same upper configuration when it
carries more than 1% of the gf (the singlet-triplet mixed nf terms).
Components closer than 1.5 km/s are merged.

usage: python tools/build_helium_impact_lines.py NIST_TSV STARKB_HTML > literal.py
"""

import csv
import html
import re
import sys
from collections import defaultdict

import numpy as np


_LOW_TERMS = {"1s.2s 3S": 1, "1s.2s 1S": 2, "1s.2p 3P*": 3, "1s.2p 1P*": 4}


def _number(text):
    try:
        return float(text.strip("[]() "))
    except ValueError:
        return np.nan


def _term_weight(term):
    match = re.match(r"(\d)([SPDFGHIK])", term)
    return int(match.group(1)) * (2 * "SPDFGHIK".index(match.group(2)) + 1)


def nist_multiplets(path):
    """LS multiplets keyed by ("conf term", "conf term") from the NIST table."""

    groups = defaultdict(list)
    with open(path) as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader)
        for row in reader:
            d = {h: v.strip().strip('"') for h, v in zip(header, row)}
            if d.get("Type", ""):
                continue  # electric-dipole lines only
            f = _number(d["fik"])
            wavelength = _number(d["ritz_wl_vac(A)"])
            if not np.isfinite(wavelength):
                wavelength = _number(d["obs_wl_vac(A)"])
            if not (np.isfinite(f) and np.isfinite(wavelength)):
                continue
            if not (d["conf_i"].startswith("1s.") and d["conf_k"].startswith("1s.")):
                continue
            key = (f"{d['conf_i']} {d['term_i']}", f"{d['conf_k']} {d['term_k']}")
            groups[key].append(dict(
                wavelength=wavelength, gf=int(d["g_i"]) * f,
                Ji=d["J_i"], Jk=d["J_k"], gi=int(d["g_i"]), gk=int(d["g_k"]),
                Ei=_number(d["Ei(cm-1)"]), Ek=_number(d["Ek(cm-1)"]),
            ))
    multiplets = {}
    for (lower, upper), components in groups.items():
        lower_levels = {c["Ji"]: (c["gi"], c["Ei"]) for c in components}
        upper_levels = {c["Jk"]: (c["gk"], c["Ek"]) for c in components}
        multiplets[(lower, upper)] = dict(
            g_lower=_term_weight(lower.split()[1]),
            E_lower=sum(g * e for g, e in lower_levels.values())
            / sum(g for g, _ in lower_levels.values()),
            E_upper=sum(g * e for g, e in upper_levels.values())
            / sum(g for g, _ in upper_levels.values()),
            components=[(c["wavelength"], c["gf"]) for c in components],
        )
    return multiplets


def starkb_rows(path):
    """(lower, upper) -> list of per-temperature STARK-B rows."""

    text = open(path, encoding="utf-8", errors="replace").read()
    start = text.find("Line widths and shifts")
    body = text[start:text.find("</table>", start)]
    table = defaultdict(list)
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, flags=re.S):
        cells = [
            html.unescape(re.sub(r"<[^>]+>", "", cell)).strip()
            for cell in re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.S)
        ]
        if len(cells) < 16 or not re.match(r"^\d", cells[0]):
            continue
        lower = cells[1].split()
        upper = cells[2].split()
        key = (
            f"{lower[0]} {lower[1].replace('o', '*')}",
            f"{upper[0]} {upper[1].replace('o', '*')}",
        )
        # electron w, d are cells 9, 11; He II (third perturber) 17, 19.
        table[key].append(dict(
            air=_number(cells[4]), C=_number(cells[5]), T=_number(cells[6]),
            e_w=_number(cells[9]), e_d=_number(cells[11]),
            ion_w=_number(cells[17]) if len(cells) > 19 else np.nan,
            ion_d=_number(cells[19]) if len(cells) > 19 else np.nan,
        ))
    return table


def _principal(term):
    return int(term.split()[0].split(".")[1][:-1])


def _resonance_level(term):
    configuration, multiplet = term.split()
    is_p = configuration.split(".")[1].endswith("p")
    return _principal(term) if (multiplet == "1P*" and is_p) else None


def _tuple(values):
    return "(" + ", ".join(
        'float("nan")' if not np.isfinite(v) else repr(float(v)) for v in values
    ) + ")"


def emit(multiplets, stark):
    blocks = []
    for (lower, upper), rows in sorted(stark.items(), key=lambda kv: kv[1][0]["air"]):
        if rows[0]["air"] < 7300.0:
            continue
        main = multiplets[(lower, upper)]
        configuration, multiplet = upper.split()
        other = f"{configuration} {('1' if multiplet[0] == '3' else '3') + multiplet[1:]}"
        components = list(main["components"])
        partner = multiplets.get((lower, other))
        main_gf = sum(gf for _, gf in components)
        if partner is not None and sum(gf for _, gf in partner["components"]) > 0.01 * main_gf:
            components += partner["components"]
        total_gf = sum(gf for _, gf in components)
        merged = []
        for wavelength, gf in sorted(components):
            if merged and abs(wavelength - merged[-1][0]) < 5.0e-6 * wavelength:
                w0, g0 = merged[-1]
                merged[-1] = (1.0 / ((g0 / w0 + gf / wavelength) / (g0 + gf)), g0 + gf)
            else:
                merged.append((wavelength, gf))
        rows = [r for r in rows if np.isfinite(r["e_w"])]
        ion = {r["T"]: r for r in rows}
        with_ions = all(
            t in ion and np.isfinite(ion[t]["ion_w"]) for t in (10_000.0, 20_000.0)
        )
        name = f"HeI{int(round(rows[0]['air']))}"
        lines = [
            "    HeliumImpactLine(",
            f"        {name!r}, {lower!r}, {upper!r},",
            f"        lower_term_index={_LOW_TERMS.get(lower)!r}, "
            f"lower_principal_quantum_number={_principal(lower)}, "
            f"upper_principal_quantum_number={_principal(upper)},",
            f"        lower_wavenumber={main['E_lower']:.6f}, "
            f"lower_statistical_weight={float(main['g_lower'])!r}, "
            f"upper_wavenumber={main['E_upper']:.6f},",
            f"        absorption_oscillator_strength={total_gf / main['g_lower']:.6g},",
            "        components=(" + ", ".join(
                f"({w:.6f}, {g / total_gf:.6f})" for w, g in merged
            ) + ",),",
            f"        lower_resonance_level={_resonance_level(lower)!r}, "
            f"upper_resonance_level={_resonance_level(upper)!r},",
            f"        stark_temperature={_tuple(r['T'] for r in rows)},",
            f"        electron_fwhm_1e16={_tuple(r['e_w'] for r in rows)},",
            f"        electron_shift_1e16={_tuple(r['e_d'] for r in rows)},",
        ]
        if with_ions:
            lines += [
                f"        ion_fwhm_1e16={_tuple(r['ion_w'] for r in rows)},",
                f"        ion_shift_1e16={_tuple(r['ion_d'] for r in rows)},",
            ]
        lines += [f"        isolated_line_parameter={rows[0]['C']:.3g},", "    ),"]
        blocks.append("\n".join(lines))
    return "HELIUM_I_IMPACT_LINES = (\n" + "\n".join(blocks) + "\n)"


if __name__ == "__main__":
    print(emit(nist_multiplets(sys.argv[1]), starkb_rows(sys.argv[2])))
