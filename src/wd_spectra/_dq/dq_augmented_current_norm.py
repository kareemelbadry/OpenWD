"""Coupled DQ equations with a differentiated current ML2 compatibility norm.

The fixed initial stability gap is not an appropriate error scale after a
layer becomes convective. Reuse the C1 norm already tested in the cool-model
experiments: stellar-flux gradient excess on the convective branch, and that
excess plus the current stability gap on the stable branch. Differentiate
the normalization, not just the numerator. Physical equations, fluxes and
qualification gates are unchanged. This remains research-only.
"""
from .dq_finite_material_coordinates import ExactMaterialCache
from . import dq_augmented_material_scaling as scaling
from .dq_augmented_conditioned_cold import conditioned_material


class CurrentNormSystem(scaling.MaterialScaledSystem):
    def __init__(self,first,evaluate,material):
        # The parent and new compatibility block share the exact same
        # constitutive derivative request. Do not double the opacity work.
        super().__init__(first,evaluate,ExactMaterialCache(material))


def current_norm_material(wavelengths=None, *, system_class=None):
    system_class = CurrentNormSystem if system_class is None else system_class
    class CurrentNormDQ(conditioned_material(wavelengths, system_class=system_class)):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            from .provenance import digest
            self.experiment_metadata.update(
                coupled_compatibility_norm='C1 current ML2 stellar-flux excess / stable-gap; fully differentiated',
                coupled_norm_source_sha256=digest(__file__))
    return CurrentNormDQ


