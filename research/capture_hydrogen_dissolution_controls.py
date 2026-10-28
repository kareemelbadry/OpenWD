"""Offline capture of the full hydrogen dissolved fraction (2026-10-27).

The DAM pseudo-continuum previously kept the full dissolved fraction only
blueward of the transition to ``i + 3`` (the SYNSPEC/TLUSTY default; H-gamma
for the Balmer series) and only its neutral-perturber part from there to the
first series member. The Tremblay--Bergeron Balmer/Paschen/Brackett profiles
include the occupation-probability reduction of every line, so the full
fraction is now kept to the first member, in both the ordinary and the
magnetic (RWA) continuation.

Every fixed-state control that synthesizes hydrogen is recomputed on its
stored structure, configuration and wavelengths. Under
``previous_hydrogen_dissolution_cutoff`` each must first recover its current
approved reference (and the DAH paper spectra) at that test's tolerance.
Existing controls are never overwritten; tests and CI never run this script.
No atmosphere is reconverged or certified by this fixed-state capture.

Each completed case is written immediately; rerunning with the same OUTPUT
skips completed cases.

Usage: PYTHONPATH=src python research/capture_hydrogen_dissolution_controls.py OUTPUT
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import warnings

import numpy as np
import scipy

from wd_spectra import (
    DAHConfig, compiled_backend_available, compute_dah, paper_dah_config,
)
from wd_spectra._compat import trapezoid
from wd_spectra.models import (
    AtmosphereConvergenceWarning, DABConfig, DAConfig, DAZConfig, DZConfig,
    compute_da, compute_dab, compute_daz, compute_dz, load_atmosphere_checkpoint,
)
from wd_spectra.models.common import ModelData, model_request_fingerprint
from hydrogen_dissolution_control_legacy import previous_hydrogen_dissolution_cutoff

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'tests/data'
APPROVED = DATA / 'approved_regressions'
BANDS = ((1000, 3000), (3000, 3650), (3650, 4100), (4100, 4340), (4340, 4860),
         (4860, 6560), (6560, 8200), (8200, 100000))
DAH_TARGETS = json.loads((DATA / 'dah_paper/manifest.json').read_text())['targets']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def difference(wavelength, actual, expected):
    relative = actual / expected - 1.
    significant = wavelength * expected > 0.01 * np.max(wavelength * expected)
    index = int(np.argmax(np.where(significant, np.abs(relative), 0.)))
    return dict(
        maximum_relative_change=float(np.max(np.abs(relative))),
        maximum_significant_relative_change=float(abs(relative[index])),
        maximum_significant_change_wavelength_angstrom=float(wavelength[index]),
        integrated_flux_change=float(trapezoid(actual - expected, wavelength)
                                     / trapezoid(expected, wavelength)),
        band_fractional_changes={f'{lo}-{hi}': float(
            trapezoid((actual - expected)[take], wavelength[take])
            / trapezoid(expected[take], wavelength[take]))
            for lo, hi in BANDS
            if np.count_nonzero(take := (wavelength >= lo) & (wavelength < hi)) > 1})


def _spectral_regression(case, previous):
    path = DATA / 'spectral_regressions' / (case + '.npz')
    with np.load(path) as saved:
        kind = str(saved['spectral_type'])
        config = {'DA': DAConfig, 'DAB': DABConfig, 'DZ': DZConfig}[kind](
            **json.loads(str(saved['config_json'])))
        wavelength = saved['wavelength'].copy()
    molecular = kind == 'DA' and config.effective_temperature <= 12000
    atmosphere = load_atmosphere_checkpoint(
        path, config.effective_temperature, config.logg,
        {'DA': 'hydrogen', 'DAB': 'mixed', 'DZ': 'helium'}[kind],
        include_molecules=molecular, include_negative_hydrogen=molecular,
        log_hydrogen_to_helium=getattr(config, 'log_hydrogen_to_helium', None),
        trihydrogen_ion_partition_model='neale-tennyson-1995' if kind == 'DA' else None)
    compute = {'DA': compute_da, 'DAB': compute_dab, 'DZ': compute_dz}[kind]
    with np.load(previous) as prior:
        expected = prior['surface_flux'].copy()
    return dict(kind=kind, config=config, wavelength=wavelength, structure=path,
                previous=previous, expected=expected, rtol=2e-6,
                run=lambda: compute(config, wavelength, initial_atmosphere=atmosphere,
                                    relax_atmosphere=False))


def _da_public(case):
    path = DATA / 'da_cubic_regressions' / (case + '.npz')
    with np.load(path) as saved:
        config = DAConfig(**json.loads(str(saved['config_json'])))
        wavelength = saved['wavelength'].copy()
        expected = saved['original_surface_flux'].copy()
    molecular = config.effective_temperature <= 12000
    atmosphere = load_atmosphere_checkpoint(
        path, config.effective_temperature, config.logg, 'hydrogen',
        include_molecules=molecular, include_negative_hydrogen=molecular,
        trihydrogen_ion_partition_model='neale-tennyson-1995')
    return dict(kind='DA', config=config, wavelength=wavelength, structure=path,
                previous=path, expected=expected, rtol=1e-3,
                run=lambda: compute_da(config, wavelength, initial_atmosphere=atmosphere,
                                       relax_atmosphere=False))


def _daz(case):
    path = DATA / 'daz_regressions' / (case + '.npz')
    previous = APPROVED / 'independent_kurucz_2026_10_14/fixed' / ('daz-' + case + '.npz')
    with np.load(path) as saved:
        config = DAZConfig(**json.loads(str(saved['config_json'])))
        wavelength = saved['wavelength'].copy()
    with np.load(previous) as prior:
        expected = prior['surface_flux'].copy()
    molecular = config.effective_temperature <= 12000
    atmosphere = load_atmosphere_checkpoint(
        path, config.effective_temperature, config.logg, 'hydrogen',
        include_molecules=molecular, include_negative_hydrogen=molecular,
        trihydrogen_ion_partition_model='neale-tennyson-1995')
    return dict(kind='DAZ', config=config, wavelength=wavelength, structure=path,
                previous=previous, expected=expected, rtol=1e-3,
                run=lambda: compute_daz(config, wavelength, initial_atmosphere=atmosphere,
                                        relax_atmosphere=False))


def _dah(key, prescription):
    original = DAH_TARGETS[key]['config']
    names = ('effective_temperature', 'logg', 'magnetic_field_megagauss', 'quality',
             'field_geometry', 'dipole_inclination_deg', 'dipole_offset_radius')
    values = {name: original[name] for name in names}
    values['dipole_offset_radius'] = tuple(values['dipole_offset_radius'])
    config = (paper_dah_config if prescription == 'paper' else DAHConfig)(**values)
    structure = DATA / 'dah_paper' / DAH_TARGETS[key]['reference_file']
    if prescription == 'paper':
        previous = structure
        with np.load(previous) as prior:
            wavelength, expected = prior['wavelength'].copy(), prior['flux'].copy()
    else:
        previous = APPROVED / 'dah_rwa_default_2026_10_25/fixed' / f'{key}.npz'
        with np.load(previous) as prior:
            wavelength, expected = prior['wavelength'].copy(), prior['surface_flux'].copy()
    atmosphere = load_atmosphere_checkpoint(
        structure, config.effective_temperature, config.logg, 'hydrogen',
        include_molecules=False, include_negative_hydrogen=False,
        trihydrogen_ion_partition_model='neale-tennyson-1995')
    return dict(kind='DAH', config=config, wavelength=wavelength, structure=structure,
                previous=previous, expected=expected, rtol=2e-8,
                run=lambda: compute_dah(config, wavelength, initial_atmosphere=atmosphere,
                                        relax_atmosphere=False))


def cases():
    fixed = APPROVED / 'fixed'
    yield 'da-20000', lambda: _spectral_regression('da-20000', fixed / 'da-20000.npz')
    yield 'dab-9000', lambda: _spectral_regression(
        'dab-9000', APPROVED / 'stark_density_floor_2026_10_05/fixed/dab-9000.npz')
    yield 'dab-20000', lambda: _spectral_regression('dab-20000', fixed / 'dab-20000.npz')
    for case in ('dz-pg1225', 'dz-j0738'):
        yield case, lambda case=case: _spectral_regression(
            case, APPROVED / 'metal_populations_2026_10_24/fixed' / (case + '.npz'))
    for case in ('da-12000-public', 'da-20000'):
        yield 'public-' + case, lambda case=case: _da_public(case)
    for case in ('g149_28', 'galex1931'):
        yield 'daz-' + case, lambda case=case: _daz(case)
    for key in DAH_TARGETS:
        for prescription in ('default', 'paper'):
            yield f'dah-{prescription}-{key}', (
                lambda key=key, prescription=prescription: _dah(key, prescription))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    (args.output / 'fixed').mkdir(parents=True, exist_ok=True)
    assert compiled_backend_available(), 'use the compiled backend for the saved controls'
    partial = args.output / 'manifest.partial.json'
    records = json.loads(partial.read_text()) if partial.exists() else {}
    for name, build in cases():
        if name in records:
            continue
        case = build()
        outputs = {}
        for label, scope in (('previous', previous_hydrogen_dissolution_cutoff), ('current', nullcontext)):
            with scope(), warnings.catch_warnings():
                warnings.simplefilter('ignore', AtmosphereConvergenceWarning)
                result = case['run']()
            flux = result.spectrum.surface_flux_lambda
            assert np.all(np.isfinite(flux)) and np.all(flux > 0.)
            outputs[label] = flux.copy()
        recovery = difference(case['wavelength'], outputs['previous'], case['expected'])
        # The 0.1% controls apply their bound where wavelength*flux exceeds 1% of its peak.
        checked = (case['wavelength'] * case['expected']
                   > 0.01 * np.max(case['wavelength'] * case['expected'])
                   if case['rtol'] >= 1e-3 else slice(None))
        np.testing.assert_allclose(outputs['previous'][checked], case['expected'][checked],
                                   rtol=case['rtol'], atol=1e-12 * np.max(case['expected']))
        out = args.output / 'fixed' / (name + '.npz')
        np.savez_compressed(out, wavelength=case['wavelength'], surface_flux=outputs['current'])
        config = case['config']
        records[name] = dict(
            spectral_type=case['kind'],
            structure_input=str(case['structure'].relative_to(ROOT)),
            structure_sha256=digest(case['structure']),
            previous_approved=str(case['previous'].relative_to(ROOT)),
            previous_sha256=digest(case['previous']),
            fixture_sha256=digest(out),
            config=json.loads(json.dumps(asdict(config), default=list)),
            request_fingerprint=model_request_fingerprint(case['kind'], config, ModelData.default()),
            change_from_previous=difference(case['wavelength'], outputs['current'], case['expected']),
            previous_recovery=recovery, recovery_tolerance=case['rtol'],
            atmosphere_reconverged=False, cold_start_qualification=False,
            tolerances_changed=False)
        partial.write_text(json.dumps(records, indent=2) + '\n')
        change = records[name]['change_from_previous']
        print(f"{name}: max significant change {100*change['maximum_significant_relative_change']:.3f}% "
              f"at {change['maximum_significant_change_wavelength_angstrom']:.1f} A; "
              f"recovery {recovery['maximum_relative_change']:.1e}", flush=True)
    manifest = dict(description=__doc__, date='2026-10-27',
        reference_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                       sorted((ROOT / 'src/wd_spectra').rglob('*.py'))},
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        cases=records)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    partial.unlink()


if __name__ == '__main__':
    main()
