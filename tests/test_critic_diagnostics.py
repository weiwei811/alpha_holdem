import pytest
from tools.diagnose_critic import summarize


def test_calibration_uses_realized_scaled_return_variance():
    perfect=summarize([{'value':v,'return':v} for v in (-1,0,1)])
    constant=summarize([{'value':0,'return':v} for v in (-1,0,1)])
    assert perfect['explained_variance']==pytest.approx(1)
    assert perfect['rmse']==0
    assert constant['explained_variance']==pytest.approx(0)
    assert constant['correlation'] is None
    assert constant['return_std']==pytest.approx((2/3)**.5)


def test_constant_payoffs_do_not_claim_explained_variance():
    result=summarize([{'value':.2,'return':1}]*3)
    assert result['explained_variance'] is None
    assert result['rmse']==pytest.approx(.8)
