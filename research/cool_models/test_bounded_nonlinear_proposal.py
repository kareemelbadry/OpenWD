import numpy as np
import pytest
from bounded_nonlinear_proposal import solve_bounded_model


def test_stiff_nonlinear_uniform_mode_keeps_physical_box():
    n=40
    uniform=np.ones((n,n))/n
    difference=1e6*(np.eye(n)-uniform)
    expected=np.full(n,.025)
    def model(x):
        return (difference@(x-expected)+np.exp(uniform@x)-np.exp(uniform@expected),
                difference+np.exp(uniform@x)[:,None]*uniform)
    result=solve_bounded_model(model,n,.04,200)
    assert result.root_solved
    np.testing.assert_allclose(result.x,expected,atol=3e-9)
    assert np.max(abs(result.x))<=.04


def test_unreachable_root_is_not_certified():
    result=solve_bounded_model(lambda x:(x-.2,np.eye(3)),3,.04,200)
    assert not result.root_solved
    np.testing.assert_allclose(result.x,.04,atol=1e-15)


def test_nonlinear_model_requires_repeated_actual_model_evaluations():
    result=solve_bounded_model(lambda x:(np.expm1(30*x)-1.5,
        np.diag(30*np.exp(30*x))),3,.04,200)
    assert result.root_solved and result.nfev>2
    np.testing.assert_allclose(result.x,np.log(2.5)/30,atol=1e-9)


@pytest.mark.parametrize('radius',[0.,-1.,np.inf,np.nan])
def test_invalid_box_rejected(radius):
    with pytest.raises(ValueError):solve_bounded_model(lambda x:(x,np.eye(2)),2,radius,10)
