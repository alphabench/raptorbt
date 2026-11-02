"""Every streaming indicator refuses a period below one, as the vectorised ones do."""

import numpy as np
import pytest

import raptorbt
from raptorbt import Indicator
from raptorbt.errors import InvalidParameter

SINGLE = ["sma", "ema", "wilder_ma", "wma", "roc", "stddev", "rsi", "atr", "donchian"]


@pytest.mark.parametrize("name", SINGLE)
@pytest.mark.parametrize("period", [0, -3])
def test_a_period_below_one_is_refused(name, period):
    with pytest.raises(InvalidParameter) as err:
        getattr(Indicator, name)(period)
    assert err.value.code == "invalid_parameter"
    assert err.value.parameter == "period"
    assert err.value.value == period


def test_bollinger_and_macd_refuse_the_same_inputs_the_array_forms_do():
    with pytest.raises(InvalidParameter):
        Indicator.bollinger(0)
    with pytest.raises(InvalidParameter):
        Indicator.bollinger(20, k=0.0)
    with pytest.raises(InvalidParameter):
        Indicator.macd(0, 26, 9)
    with pytest.raises(InvalidParameter):
        Indicator.macd(26, 12, 9)
    with pytest.raises(InvalidParameter):
        Indicator.macd(12, 26, 0)


def test_the_array_form_refuses_it_too():
    with pytest.raises(ValueError):
        raptorbt.sma(np.array([1.0, 2.0, 3.0]), 0)


def test_period_one_is_valid():
    ind = Indicator.sma(1)
    ind.update_bar(0.0, 0.0, 0.0, 5.0)
    assert ind.value == 5.0
