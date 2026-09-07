import json
import sys
import numpy as np
import pytest
from predict_molecular_dab import main


def inputs(tmp_path,monkeypatch,*,converged=True,source_teff=8000):
    source=tmp_path/'source.npz';output=tmp_path/'prediction.npz'
    fields=dict(temperature=np.array([6000.,9000.]),gas_pressure=np.array([1e5,1e9]),
        column_mass=np.array([.001,10.]),rosseland_optical_depth=np.array([1e-5,100.]))
    np.savez_compressed(source,**fields,effective_temperature=8000.,logg=8.,
        atmosphere_metadata_json=json.dumps(dict(mixed_chemical_model='molecular-h-he-hm',
            radiative_equilibrium_converged=converged,log_hydrogen_to_helium=-2.)))
    monkeypatch.setattr(sys,'argv',['predict',str(source),'--source-teff',str(source_teff),
        '--target-teff','7750','--output',str(output)])
    return fields,output


def test_prediction_preserves_grid_and_invalidates_convergence(tmp_path,monkeypatch):
    fields,output=inputs(tmp_path,monkeypatch)
    main()
    with np.load(output) as saved:
        for k in ('gas_pressure','column_mass','rosseland_optical_depth'):
            np.testing.assert_array_equal(saved[k],fields[k])
        np.testing.assert_array_equal(saved['temperature'],fields['temperature']*7750/8000)
        assert float(saved['effective_temperature'])==7750
        meta=json.loads(str(saved['atmosphere_metadata_json']))
        assert meta['radiative_equilibrium_converged'] is False
        assert meta['source_reported_converged'] is True


@pytest.mark.parametrize('converged,teff,message',[(False,8000,'converged molecular'),
                                                  (True,9000,'does not match')])
def test_invalid_continuation_parent_is_not_used(tmp_path,monkeypatch,converged,teff,message):
    _,output=inputs(tmp_path,monkeypatch,converged=converged,source_teff=teff)
    with pytest.raises(ValueError,match=message):main()
    assert not output.exists()
