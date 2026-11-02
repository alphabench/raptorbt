"""Array runners refuse decreasing timestamps, naming the instrument or leg.

The engine assumes ascending time everywhere. A series that steps backwards is
refused with a ``ValueError`` that names the series and the offending index.
Equal timestamps stay legal. Each case below also runs once on sorted input
as a control, so the refusal is the timestamps and not the other arguments.
"""

import numpy as np
import pytest

import raptorbt as r

N = 6
GOOD = np.arange(N, dtype=np.int64)
BAD = np.array([0, 1, 2, 1, 4, 5], dtype=np.int64)  # index 3 is 1 after 2
TIES = np.array([0, 1, 1, 2, 2, 3], dtype=np.int64)
MESSAGE = r"timestamps must not decrease, but index 3 is 1 after 2"


def _px():
    close = 100.0 + np.arange(N, dtype=np.float64)
    return close, close + 1.0, close - 1.0, np.full(N, 1_000.0)


def _sig():
    entries = np.zeros(N, dtype=bool)
    entries[1] = True
    exits = np.zeros(N, dtype=bool)
    exits[4] = True
    return entries, exits


def _instrument(ts, sym):
    c, h, l, v = _px()
    entries, exits = _sig()
    return (ts, c, h, l, c, v, entries, exits, 1, 1.0, sym)


def basket(ts):
    return r.run_basket_backtest([_instrument(ts, "AAA"), _instrument(GOOD, "BBB")])


def basket_second_instrument(ts):
    return r.run_basket_backtest([_instrument(GOOD, "AAA"), _instrument(ts, "BBB")])


def portfolio(ts):
    return r.run_portfolio_backtest([_instrument(ts, "AAA"), _instrument(GOOD, "BBB")])


def multi(ts):
    c, h, l, v = _px()
    entries, exits = _sig()
    return r.run_multi_backtest(ts, c, h, l, c, v, [(entries, exits, 1, 1.0, "s1")])


def pairs_leg1(ts):
    return _pairs(ts, GOOD)


def pairs_leg2(ts):
    return _pairs(GOOD, ts)


def _pairs(ts1, ts2):
    c, h, l, v = _px()
    entries, exits = _sig()
    return r.run_pairs_backtest(
        ts1, c, h, l, c, v, ts2, c * 2, h * 2, l * 2, c * 2, v, entries, exits,
        1, "PAIR",
    )


def options(ts):
    c, h, l, v = _px()
    entries, exits = _sig()
    return r.run_options_backtest(
        ts, c, h, l, c, v, np.full(N, 5.0), entries, exits, 1, "OPT"
    )


def _spread_args(ts):
    entries, exits = _sig()
    return ts, np.linspace(100, 105, N), [np.full(N, 5.0)], [("CE", 100.0, 1, 1)], entries, exits


def spread(ts):
    return r.run_spread_backtest(*_spread_args(ts))


def batch_spread(ts):
    t, under, legs, cfgs, entries, exits = _spread_args(ts)
    item = r.BatchSpreadItem("s1", legs, cfgs, entries, exits)
    return r.batch_spread_backtest(t, under, [item])


def batch_single(ts):
    c, h, l, v = _px()
    entries, exits = _sig()
    return r.batch_single_backtest(ts, c, h, l, c, v, [r.BatchSingleItem("i1", entries, exits)])


def tick(ts):
    c, _, _, _ = _px()
    entries, exits = _sig()
    zeros = np.zeros(N)
    return r.run_tick_backtest(
        ts, c, c - 0.05, c + 0.05, zeros, zeros, zeros, entries, exits, symbol="TICKER"
    )


def single(ts):
    c, h, l, v = _px()
    entries, exits = _sig()
    return r.run_single_backtest(ts, c, h, l, c, v, entries, exits, symbol="SOLO")


# (runner, the label the error must carry)
CASES = [
    pytest.param(single, "SOLO bars", id="single"),
    pytest.param(basket, "AAA bars", id="basket-first"),
    pytest.param(basket_second_instrument, "BBB bars", id="basket-second"),
    pytest.param(portfolio, "AAA bars", id="portfolio"),
    pytest.param(multi, "multi-strategy bars", id="multi"),
    pytest.param(pairs_leg1, "PAIR leg 1 bars", id="pairs-leg1"),
    pytest.param(pairs_leg2, "PAIR leg 2 bars", id="pairs-leg2"),
    pytest.param(options, "OPT bars", id="options"),
    pytest.param(spread, "spread underlying bars", id="spread"),
    pytest.param(batch_spread, "batch spread underlying bars", id="batch-spread"),
    pytest.param(batch_single, "batch single shared price series", id="batch-single"),
    pytest.param(tick, "TICKER ticks", id="tick"),
]


@pytest.mark.parametrize("runner,label", CASES)
def test_decreasing_timestamps_are_refused_with_the_series_named(runner, label):
    with pytest.raises(ValueError, match=f"{label}: {MESSAGE}"):
        runner(BAD)


@pytest.mark.parametrize("runner,label", CASES)
def test_sorted_and_tied_timestamps_still_run(runner, label):
    runner(GOOD)
    runner(TIES)
