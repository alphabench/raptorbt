"""Instrument, option-chain and resampler refusals are structured and point at the right skill."""

import numpy as np
import pytest

import raptorbt
from raptorbt import InstrumentSpec, OptionChain, Resampler
from raptorbt.errors import InvalidParameter, RaptorBTError


def test_a_bad_option_right_is_a_structured_error():
    with pytest.raises(InvalidParameter) as err:
        InstrumentSpec.option("X", 100.0, "straddle", 10**18, 50.0)
    e = err.value
    assert e.code == "invalid_option_right"
    assert e.parameter == "right" and e.value == "straddle"
    assert e.valid == ["call", "put"]
    assert e.skill == "instruments"
    assert isinstance(e, ValueError)


@pytest.mark.parametrize(
    "build, parameter",
    [
        (lambda: InstrumentSpec.option("X", 0.0, "call", 10**18, 50.0), "strike"),
        (lambda: InstrumentSpec.index("NIFTY", price_increment=-1.0), "price_increment"),
    ],
)
def test_other_factory_refusals_name_the_parameter(build, parameter):
    with pytest.raises(InvalidParameter) as err:
        build()
    assert err.value.parameter == parameter
    assert err.value.skill == "instruments"


def test_a_bad_expiry_rule_is_a_structured_error():
    chain = OptionChain([])
    with pytest.raises(InvalidParameter) as err:
        chain.pick_expiry("NIFTY", 0, "fortnightly")
    e = err.value
    assert e.code == "invalid_expiry_pick" and e.parameter == "which"
    assert e.skill == "instruments" and "monthly" in e.valid
    with pytest.raises(InvalidParameter) as err:
        chain.pick_expiry("NIFTY", 0, "date")
    assert err.value.parameter == "date_ns"


def test_resampler_refusals_point_at_the_timeframes_skill():
    with pytest.raises(RaptorBTError) as err:
        Resampler(60 * 10**9, closed="sideways")
    assert err.value.skill == "timeframes"
    ones = np.ones(2)
    with pytest.raises(RaptorBTError) as err:
        raptorbt.resample_bars(
            np.array([1, 2], dtype=np.int64), ones, ones, ones, ones, np.ones(1), 60 * 10**9
        )
    assert err.value.skill == "timeframes"
