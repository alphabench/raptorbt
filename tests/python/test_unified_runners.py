"""The basket, pairs, options, spread and tick runners trade on the kernel's rules."""

import numpy as np
import pytest

import raptorbt as r
from raptorbt import BacktestConfig

N = 12
TS = np.arange(N, dtype=np.int64) * 60_000_000_000


def _bars(price):
    c = np.asarray(price, dtype=float)
    return TS, c, c, c, c, np.full(N, 1e5)


def _sig(entry=1, exit_=8):
    e = np.zeros(N, bool)
    x = np.zeros(N, bool)
    e[entry] = True
    x[exit_] = True
    return e, x


def test_a_short_option_earns_when_the_premium_falls():
    prem = np.concatenate([np.full(4, 100.0), np.linspace(100.0, 60.0, N - 4)])
    ts, o, h, l, c, v = _bars(np.full(N, 1000.0))
    e, x = _sig()
    cfg = BacktestConfig(initial_capital=500_000.0, fees=0.0)
    kw = dict(config=cfg, option_type="call", strike_selection="atm", size_type="contracts",
              size_value=1, lot_size=50, strike_interval=50.0)
    short = r.run_options_backtest(ts, o, h, l, c, v, prem, e, x, -1, "O", **kw)
    long_ = r.run_options_backtest(ts, o, h, l, c, v, prem, e, x, 1, "O", **kw)
    assert short.trades()[0].pnl > 0 > long_.trades()[0].pnl
    assert short.trades()[0].pnl == pytest.approx(-long_.trades()[0].pnl)


def test_a_pair_is_two_leg_trades_that_add_up_to_the_account():
    p1 = np.linspace(100, 110, N)
    p2 = np.linspace(50, 51, N)
    e, x = _sig()
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0)
    res = r.run_pairs_backtest(*_bars(p1), *_bars(p2), e, x, 1, "P", cfg, 1.0, False)
    trades = res.trades()
    assert [t.symbol for t in trades] == ["P_LEG1", "P_LEG2"]
    assert sum(t.pnl for t in trades) == pytest.approx(res.equity_curve()[-1] - 100_000.0)
    assert trades[0].pnl > 0 > trades[1].pnl


def test_a_hedged_pair_never_spends_more_than_the_account():
    p1 = np.full(N, 100.0)
    p2 = np.full(N, 50.0)
    e, x = _sig()
    cfg = BacktestConfig(initial_capital=100_000.0, fees=0.0)
    res = r.run_pairs_backtest(*_bars(p1), *_bars(p2), e, x, 1, "P", cfg, 3.0, False)
    notional = sum(t.size * t.entry_price for t in res.trades())
    assert notional <= 100_000.0 + 1e-6


def test_a_spread_that_cannot_be_paid_for_opens_no_leg():
    prem = [np.full(N, 400.0), np.full(N, 400.0)]
    e, x = _sig()
    cfg = BacktestConfig(initial_capital=1_000.0, fees=0.0)
    res = r.run_spread_backtest(TS, np.full(N, 1000.0), prem,
                                [("CE", 100.0, 1, 75), ("PE", 100.0, 1, 75)], e, x, cfg)
    assert res.trades() == []
    assert set(np.asarray(res.equity_curve())) == {1_000.0}


def test_nan_premiums_and_ticks_are_refused():
    e, x = _sig()
    bad = np.full(N, 5.0)
    bad[3] = np.nan
    with pytest.raises(ValueError, match="premium"):
        r.run_spread_backtest(TS, np.full(N, 100.0), [bad], [("CE", 100.0, 1, 1)], e, x)
    zeros = np.zeros(N)
    ltp = np.full(N, 100.0)
    ltp[2] = np.inf
    with pytest.raises(ValueError, match="ltp"):
        r.run_tick_backtest(TS, ltp, ltp, ltp, zeros, zeros, zeros, e, x, symbol="T")


def test_full_report_defaults_are_not_one_exchanges():
    ts = (np.arange(50) * 60_000_000_000).tolist()
    eq = (100.0 + np.arange(50)).tolist()
    explicit = r.full_report(ts, eq, session_minutes=375.0, tz_offset_ns=r.IST_OFFSET_NS)
    default = r.full_report(ts, eq)
    utc = r.full_report(ts, eq, session_minutes=None, tz_offset_ns=0)
    assert default["report"] == utc["report"]
    assert default["report"] != explicit["report"]
