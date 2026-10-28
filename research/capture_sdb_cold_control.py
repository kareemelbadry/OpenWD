"""Capture a spectrum-only control from completed, independent sdB drivers.

First run sdb_lte_hhe.py at 23200 K, log g 5.20, log He/H -2.27, then run
sdb_hybrid_nlte.py with --hydrogen-levels 16 --accelerated-lambda and
--population-iterations 600. This script never calls the public sdB adapter.
It refuses to replace an existing control or accept an unconverged run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lte_model', type=Path)
    parser.add_argument('hybrid_model', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('choose a new control directory')
    lte = json.loads((args.lte_model / 'run-summary.json').read_text())
    hybrid = json.loads((args.hybrid_model / 'run-summary.json').read_text())
    saved_lte = json.loads((args.lte_model / 'metadata.json').read_text())
    certificate = saved_lte['atmosphere_metadata']['equilibrium_certificate']
    expected = dict(teff=23200., logg=5.20, log_he_h=-2.27, quality='standard')
    assert lte['config'] == hybrid['config'] == expected
    assert lte['atmosphere_convergence_status'] == 'converged'
    assert certificate['verified'] is True
    for name in ('all_depth_flux', 'local_energy', 'temperature_stationarity',
                 'source_closure', 'boundary_screening'):
        assert certificate['checks'][name]['measured'] is True
        assert certificate['checks'][name]['passed'] is True
    assert hybrid['populations_converged'] is True
    assert hybrid['population_change'] < 1e-4
    assert hybrid['hydrogen_levels'] == 16
    assert hybrid['helium_i_atom'] == '14'
    assert hybrid['helium_conservation_row'] == 'dominant'
    assert hybrid['accelerated_lambda'] is True
    with np.load(args.hybrid_model / 'spectrum.npz', allow_pickle=False) as result:
        wavelength, flux = result['wavelength_vacuum'], result['flux']
    with np.load(args.hybrid_model / 'populations.npz', allow_pickle=False) as result:
        fixed_state = dict(result)
    np.testing.assert_array_equal(wavelength, np.arange(3000., 9200., .05))
    assert np.all(np.isfinite(flux)) and np.all(flux > 0)
    root = Path(__file__).resolve().parents[1]
    drivers = ['research/sdb_lte_hhe.py', 'research/sdb_hybrid_nlte.py']
    subprocess.run(['git', 'diff', '--exit-code', '--', *drivers], cwd=root, check=True)
    args.output.mkdir(parents=True)
    control = args.output / 'hd4539-hhe.npz'
    np.savez_compressed(control, wavelength=wavelength, flux=flux)
    state = args.output / 'hd4539-fixed-state.npz'
    np.savez_compressed(state, **fixed_state)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = dict(
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
        source_changes=subprocess.check_output(['git', 'diff', '--name-only'], cwd=root).decode().splitlines(),
        recipe='separate sdb_lte_hhe.py + sdb_hybrid_nlte.py, no compute_sdb call',
        driver_sha256={name: digest(root / name) for name in drivers},
        config=expected, hydrogen_levels=16, helium_i_atom='14', accelerated_lambda=True,
        helium_conservation_row='dominant',
        lte_convergence_status=lte['atmosphere_convergence_status'],
        lte_equilibrium_certificate=certificate,
        nlte_populations_converged=hybrid['populations_converged'],
        nlte_population_iterations=hybrid['population_iterations'],
        nlte_population_change=hybrid['population_change'],
        control_sha256=digest(control), fixed_state_sha256=digest(state),
        absolute_flux_rtol=1e-3, fixed_spectrum_rtol=1e-7,
        limitations='Spectrum-only regression of the metal-free hybrid recipe; not a new observation, grid or coupled-NLTE equilibrium qualification.',
    )
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
