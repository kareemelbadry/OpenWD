from types import SimpleNamespace
import numpy as np
import full_ml2_manifold_projection as module


def test_full_projection_uses_actual_gradient_and_preserves_inputs(monkeypatch):
    mapping=np.diag([1.,2.,3.]);scale=np.array([2.,3.])
    temperatures=np.array([3000.,4000.,6000.]);pressure=np.array([1e6,1e7,1e8])
    trial=np.r_[np.linalg.solve(mapping,np.log(temperatures)),[-.1,.3]]
    saved=trial.copy();options=dict(with_temperature=lambda t:SimpleNamespace(temperature=t,
        gas_pressure=pressure,effective_temperature=8000.),thermodynamics=None,rosseland_opacity=None,
        mixing_length_alpha=1.25)
    def transport(candidate,desired,runner,options,**kw):
        np.testing.assert_allclose(desired,[0.,0.,.9**3])
        assert kw['project_stable']
        return options['with_temperature'](candidate.temperature*np.exp(.001)),0.
    class Material:
        def __init__(self,*args):pass
        def __call__(self,lt):return np.ones((3,3))
    gradients=[]
    def velocity(g,*args):gradients.append(g.copy());return np.array([-.2,.5])
    monkeypatch.setattr(module,'transport_profile',transport)
    monkeypatch.setattr(module,'MaterialCoefficients',Material)
    monkeypatch.setattr(module,'ml2_auxiliary_from_gradient',velocity)
    projected=module.project_full_trial(trial,trial,scale=scale,mapping=mapping,options=options)
    np.testing.assert_array_equal(trial,saved)
    np.testing.assert_allclose(mapping@projected[:3],np.log(temperatures)+.001)
    np.testing.assert_allclose(projected[3:]*scale,[-.2,.5])
    np.testing.assert_allclose(gradients[0],np.diff(np.log(temperatures))/np.diff(np.log(pressure)))
