"""A NaN or infinite price is refused, naming the column and the first index."""

import numpy as np
import pytest

import raptorbt
from raptorbt import Strategy
from raptorbt.errors import InvalidData

N = 30


def _arrays(column=None, index=7, value=np.nan):
    c = np.linspace(100.0, 110.0, N)
    d = {
        "timestamps": np.arange(N, dtype=np.int64),
        "open": c.copy(),
        "high": c + 1.0,
        "low": c - 1.0,
        "close": c.copy(),
        "volume": np.ones(N),
    }
    if column is not None:
        d[column][index] = value
    return d


def _signals():
    entries = np.zeros(N, dtype=bool)
    exits = np.zeros(N, dtype=bool)
    entries[2] = True
    exits[20] = True
    return entries, exits


def run_array(d):
    entries, exits = _signals()
    return raptorbt.run_single_backtest(
        d["timestamps"], d["open"], d["high"], d["low"], d["close"], d["volume"],
        entries, exits, symbol="SOLO",
    )


class Flat(Strategy):
    def on_bar(self, ctx):
        pass


def run_class(d):
    return raptorbt.run_strategy_backtest(Flat(), **d, bar_stamp="close")


def run_many(d):
    return raptorbt.run_strategies([Flat()], {"AAA": d}, bar_stamp="close")


RUNNERS = [
    pytest.param(run_array, id="array"),
    pytest.param(run_class, id="strategy"),
    pytest.param(run_many, id="strategies"),
]


@pytest.mark.parametrize("runner", RUNNERS)
@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
def test_nan_is_refused_with_column_and_index(runner, column):
    with pytest.raises(InvalidData) as err:
        runner(_arrays(column, index=7))
    assert err.value.code == "non_finite_price"
    assert column in str(err.value)
    assert "index 7" in str(err.value)
    assert err.value.parameter == column
    assert err.value.value == 7


@pytest.mark.parametrize("runner", RUNNERS)
def test_infinity_is_refused(runner):
    with pytest.raises(InvalidData, match="close"):
        runner(_arrays("close", index=3, value=np.inf))


@pytest.mark.parametrize("runner", RUNNERS)
def test_finite_data_still_runs(runner):
    runner(_arrays())
