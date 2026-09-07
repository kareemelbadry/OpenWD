import numpy as np
import pytest
from wd_spectra.nonlinear import NonlinearEvaluation
from equilibrated_temperature_step import equilibrated_step


@pytest.mark.parametrize('radius',[.04,.01])
def test_weak_global_mode_is_not_regularized_away(radius):
    n=40;uniform=np.ones((n,n))/n
    j=uniform+1e8*(np.eye(n)-uniform)
    mapping=np.tril(np.ones((n,n)))
    desired=np.full(n,.02)
    ev=NonlinearEvaluation(-j@desired,j@mapping,{'log_temperature_from_state':mapping})
    result=equilibrated_step(np.zeros(n),ev,ev.jacobian,radius,None)
    np.testing.assert_allclose(mapping@result,min(radius,.02),rtol=1e-8,atol=1e-9)
