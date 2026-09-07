from types import SimpleNamespace
import numpy as np
import pytest
import exact_ml2_trial_projector as module


@pytest.mark.parametrize('fail',[False,True])
def test_every_backtracking_trial_uses_actual_material_projection_and_restores(monkeypatch,fail):
    calls=[]
    velocity=np.array([.2,-.1])
    proposed=np.array([.4,.2])
    direction=np.array([.03,-.02,.01])
    mapping=np.diag([1.,2.,3.])
    state=np.log([3000.,4000.,5000.])/np.diag(mapping)
    payload=dict(energy_balance_is_physical_flux=True,
        convection_transport={k:np.ones(3) for k in
            ('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient')},
        atmosphere=SimpleNamespace(effective_temperature=8000.),
        temperature_gradient=np.ones(3),log_temperature_from_state=mapping)
    def builder(*args):
        payload['diagnostic_augmented_proposed_velocity']=proposed
        return direction
    def newton(initial,evaluate,**options):
        assert options['allow_initial_convergence'] is False
        options['step_builder'](initial,evaluate(initial,True),np.eye(3),.1)
        for factor in (1.,.5,.125):
            trial=initial+factor*direction
            projected=options['trial_projector'](initial,trial)
            np.testing.assert_allclose(mapping@projected,mapping@trial+.001)
        if fail:raise RuntimeError('test scoped restoration')
        return 'done'
    def adaptive(seed,wave,**options):
        assert options['metadata']['experimental_trial_projection_changes_flux'] is False
        return module.adaptive.solve_trust_region_newton(state,
            lambda *a:SimpleNamespace(payload=payload),step_builder=builder)
    def transport(seed,desired,runner,options,**kwargs):
        calls.append(desired.copy())
        assert kwargs['project_stable'] is True
        assert options['rosseland_opacity'] is actual_opacity
        return SimpleNamespace(temperature=seed.temperature*np.exp(.001)),0.
    actual_opacity=object()
    monkeypatch.setattr(module.adaptive,'solve_adaptive_lte_structure',adaptive)
    monkeypatch.setattr(module.adaptive,'solve_trust_region_newton',newton)
    monkeypatch.setattr(module,'ml2_auxiliary_from_gradient',lambda *args:velocity.copy())
    monkeypatch.setattr(module,'transport_profile',transport)
    def run():
        with module.exact_trial_projection():
            return module.adaptive.solve_adaptive_lte_structure(None,None,
                with_temperature=lambda t:SimpleNamespace(temperature=t),
                rosseland_opacity=actual_opacity)
    if fail:
        with pytest.raises(RuntimeError,match='scoped restoration'):run()
    else:assert run()=='done'
    for factor,desired in zip((1.,.5,.125),calls):
        np.testing.assert_allclose(desired,np.r_[0.,np.maximum(velocity+factor*(proposed-velocity),0.)**3])
    assert len(calls)==3
    assert module.adaptive.solve_adaptive_lte_structure is adaptive
    assert module.adaptive.solve_trust_region_newton is newton
