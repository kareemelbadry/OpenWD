"""Reuse the shared atmosphere evaluation's actual ML2 material responses.

The physical Jacobian already evaluates the base/hot/cold EOS and opacity,
including centered even/odd interface coefficient responses. Reuse those
responses for the augmented compatibility block instead of evaluating three
more nearly identical constitutive states. Accepted residuals and physics do
not change. Qualify the resulting finite-window tangent independently; this
does not assume equivalence to analytic differentiation of the coefficients.
"""
from unittest.mock import patch
from . import dq_augmented_current_norm as current


class ProbeReuseSystem(current.CurrentNormSystem):
    def __init__(self,first,evaluate,material):
        def physical_probes(logt,need=False):
            p=self.physical(logt,need).payload
            values=tuple(p['convection_transport'][key] for key in
                ('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient'))
            if not need:return values
            if p.get('material_temperature_response_domain_limited',False):
                raise ValueError('DQ shared-probe experiment requires centered material responses')
            return values,p['ml2_coefficient_log_temperature_responses']
        super().__init__(first,evaluate,physical_probes)


def probe_reuse_material(wavelengths=None):
    class SharedProbeDQ(current.current_norm_material(wavelengths)):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            from .provenance import digest
            self.experiment_metadata.update(
                coupled_material_response='reuse shared centered even/odd ML2 coefficient probes',
                coupled_probe_reuse_source_sha256=digest(__file__))
        def solve(self,*args,**kwargs):
            with patch.object(current,'CurrentNormSystem',ProbeReuseSystem):
                return super().solve(*args,**kwargs)
    return SharedProbeDQ


