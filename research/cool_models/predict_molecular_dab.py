"""Explicit temperature continuation seed; never inherits convergence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--source-teff',type=float,required=True)
    p.add_argument('--target-teff',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('refusing overwrite')
    if not np.all(np.isfinite([a.source_teff,a.target_teff])) or min(a.source_teff,a.target_teff)<=0:
        p.error('temperatures must be finite positive')
    blob=a.source.read_bytes()
    import io
    with np.load(io.BytesIO(blob)) as old:
        values={k:old[k].copy() for k in ('temperature','gas_pressure','column_mass','rosseland_optical_depth')}
        meta=json.loads(str(old['atmosphere_metadata_json']))
        stored_teff=float(old['effective_temperature'])
        logg=float(old['logg'])
    if stored_teff!=a.source_teff:
        raise ValueError('source Teff does not match the saved atmosphere')
    if meta.get('mixed_chemical_model')!='molecular-h-he-hm':
        raise ValueError('temperature continuation requires a declared molecular H/He source')
    if meta.get('radiative_equilibrium_converged') is not True:
        raise ValueError('continuation parent must be a converged molecular atmosphere, not a failed prediction')
    provenance=dict(temperature_continuation_source=str(a.source),source_sha256=hashlib.sha256(blob).hexdigest(),
        source_teff=a.source_teff,target_teff=a.target_teff,source_reported_converged=True,
        prediction_method='uniform temperature scaling; new equilibrium checks required',
        radiative_equilibrium_converged=False,initialization_only=True,
        includes_molecular_equilibrium=True,mixed_chemical_model='molecular-h-he-hm',
        experimental_h2_partition=meta.get('experimental_h2_partition'),
        log_hydrogen_to_helium=meta.get('log_hydrogen_to_helium'))
    values['temperature']*=a.target_teff/a.source_teff
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,**values,effective_temperature=a.target_teff,logg=logg,
                        atmosphere_metadata_json=json.dumps(provenance))
    print(json.dumps(provenance),flush=True)


if __name__=='__main__':main()
