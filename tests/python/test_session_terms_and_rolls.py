"""Terms an instrument's venue implies, a bar stamped at the session close, and a roll.

* F4: an instrument that names a known Indian venue and no session has that venue's session, so
  its orders are day orders and its composites sit on the session grid; a venue no preset covers
  has no session to guess, and a day order on it is refused.
* F9: a bar stamped at the session close that is the first bar of its session (a daily bar) is
  that session's bar: a same_bar_close decision on it fills at that close, a next_bar_open one at
  the next session's open.
* F10: a roll's open is sent on the close's fill and meets the bar the close filled in, in every
  mode: the position is never flat between the two.
"""

import numpy as np
import pytest

import raptorbt as rb
from raptorbt import BacktestConfig, SessionSchedule, Strategy, run_strategies
from raptorbt.strategy import Future, Holding, Requirements, Underlying
from raptorbt.strategy import orders as O
from raptorbt.strategy.continuous import RollEvents

from _market_harness import MIN, Script, bars, fmt, ist, run_paths
from test_market_sessions_and_defaults import _composites

MODES = ("batch", "node", "paper")

# -- F4 ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("venue", ["NSE", "BSE", "NFO", "BFO", "MCX", "CDS", "BCD"])
def test_a_known_venue_gives_an_instrument_with_no_session_its_sessions_calendar(venue):
    # "GOLD" is a name MCX knows the class of; "X" would be refused there (mcx_session_unknown)
    assert rb.InstrumentSpec.equity("GOLD", venue=venue).session is not None
    spec = rb.InstrumentSpec.equity("GOLD")
    assert spec.session is None
    spec.venue = venue
    assert spec.session is not None, "set after the fact, the venue implies it as well"


def test_a_session_given_beside_the_venue_is_the_one_used():
    four_pm = ist(2026, 11, 2, 16, 0)  # open for the currency session, closed for NSE's
    spec = rb.InstrumentSpec.equity("X", venue="NSE", session=SessionSchedule.cds())
    assert spec.session.is_open(four_pm) and not rb.InstrumentSpec.equity("X", venue="NSE").session.is_open(four_pm)


def test_an_instrument_that_names_no_venue_still_trades_at_any_time():
    assert rb.InstrumentSpec.equity("X").session is None


def test_an_nse_limit_on_an_instrument_with_a_venue_and_no_session_is_a_day_order():
    stamps = [ist(2026, 11, 2, 15, 28), ist(2026, 11, 2, 15, 29), ist(2026, 11, 3, 9, 16), ist(2026, 11, 3, 9, 17)]
    rows = [(100,) * 4, (100,) * 4, (98, 99, 97.5, 98.5), (98.5,) * 4]
    place = {1: lambda s, c: s.submit_order(O.Limit(side="buy", price=99.0, units=10.0))}
    runs = run_paths(
        place, {"X": bars(rows, stamps)}, lambda: BacktestConfig(initial_capital=1e5, fees=0.0),
        instruments={"X": rb.InstrumentSpec.equity("X", venue="NSE")},
    )
    for path, (strategy, res) in runs.items():
        assert res.result.trades() == [], f"{path}: a resting NSE limit filled at the next morning's open"
        assert strategy.events[-1] == "order_expired", path


def test_composites_of_an_nse_instrument_with_no_session_sit_on_the_session_grid():
    got = _composites(venue="NSE")
    first_day = [g[0] for g in got["1h"] if g[0].startswith("11-02")]
    assert first_day[:2] == ["11-02 10:15", "11-02 11:15"], "09:15-10:15, not 09:00-10:00"
    assert got == _composites(session=SessionSchedule.nse_equity())


def test_a_day_order_on_a_venue_no_preset_covers_and_no_session_is_refused():
    start = ist(2026, 11, 2, 10, 0)
    place = {1: lambda s, c: s.submit_order(O.Limit(side="buy", price=99.0, units=1.0, tif="day"))}
    with pytest.raises(rb.errors.InvalidOrder) as e:
        run_paths(
            place, {"F": bars([(100,) * 4] * 4, start)}, lambda: BacktestConfig(initial_capital=1e6, fees=0.0),
            instruments={"F": rb.InstrumentSpec.equity("F", venue="NCO")}, paths=("batch",),
        )
    assert e.value.code == "order_validity_unsupported" and "session" in str(e.value)


# -- F9 ---------------------------------------------------------------------------------------------

DAYS = [(11, 2), (11, 3), (11, 4), (11, 5)]
DAILY_ROWS = [(100, 101, 99, 100), (102, 103, 101, 102), (104, 105, 103, 104), (106, 107, 105, 106)]


@pytest.mark.parametrize("timing, want", [
    ("same_bar_close", [(100.0, "11-02 15:30")]),
    ("next_bar_open", [(102.0, "11-03 15:30")]),
])
def test_a_daily_bar_stamped_at_the_close_is_that_sessions_bar(timing, want):
    """Decided on the 11-02 bar: filled at that bar's close, or at the open of the 11-03 bar."""
    data = {"X": bars(DAILY_ROWS, [ist(2026, m, d, 15, 30) for m, d in DAYS])}
    runs = run_paths(
        {0: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0))}, data,
        lambda: BacktestConfig(initial_capital=1e4, fees=0.0, fill_timing=timing),
        instruments={"X": rb.InstrumentSpec.equity("X", venue="NSE", session=SessionSchedule.nse_equity())},
    )
    for path, (strategy, res) in runs.items():
        assert [(t.entry_price, fmt(t.entry_time)) for t in res.result.trades()] == want, path
        assert "order_rejected" not in strategy.events, path


def test_a_minute_bar_stamped_at_the_close_is_not_a_session_bar():
    """The last minute of a day of minute bars cannot take a decision: the market has closed."""
    stamps = [ist(2026, 11, 2, 15, 28), ist(2026, 11, 2, 15, 29), ist(2026, 11, 2, 15, 30)]
    data = {"X": bars([(100,) * 4] * 3, stamps)}
    runs = run_paths(
        {2: lambda s, c: s.submit_order(O.Market(side="buy", units=10.0))}, data,
        lambda: BacktestConfig(initial_capital=1e4, fees=0.0),
        instruments={"X": rb.InstrumentSpec.equity("X", venue="NSE", session=SessionSchedule.nse_equity())},
    )
    for path, (strategy, res) in runs.items():
        assert res.result.trades() == [] and strategy.rejections == [("11-02 15:30", "market_closed")], path


# -- F10 --------------------------------------------------------------------------------------------

T0 = ist(2026, 11, 2, 9, 15)
EXP_A, EXP_B = T0 + 20 * MIN, T0 + 400 * MIN


def _roll_run(mode, timing):
    specs = {
        "FA": rb.InstrumentSpec.futures_contract("FA", EXP_A, 1.0, underlying="IDX"),
        "FB": rb.InstrumentSpec.futures_contract("FB", EXP_B, 1.0, underlying="IDX"),
    }

    class Carry(RollEvents, Strategy):
        requirements = Requirements(
            roles={"index": Underlying(asset="index"), "fut": Future(of="index", contract="continuous", roll_days=5 / 1440)},
            timeframes=["1m"], holding=Holding.overnight(), data=["bar"],
        )
        auto_roll = True

        def __init__(self):
            super().__init__()
            self.n, self.fills = 0, []

        def on_bar(self, ctx, bar):
            if ctx.symbol == "IDX":
                self.n += 1
                if self.n == 1:
                    self.submit_order(O.Market(side="buy", units=3.0, instrument=ctx.role("fut")))

        def on_order_event(self, ctx, e):
            if e.kind == "order_filled":
                self.fills.append((e.client_order_id, fmt(ctx.event_ns), e.price))

    n = 30
    data = {}
    for k, s in enumerate(("IDX", "FA", "FB")):
        c = 100.0 + 10 * k + np.arange(n) * 0.1
        data[s] = {"timestamps": T0 + (np.arange(n, dtype=np.int64) + 1) * MIN, "open": c - 0.05,
                   "high": c + 1, "low": c - 1, "close": c, "volume": np.full(n, 100.0)}
    cfg = BacktestConfig(initial_capital=1e6, fees=0.0, fill_timing=timing)
    s = Carry()
    if mode == "batch":
        res = run_strategies([s], data, cfg, bar_stamp="close", instruments=specs, bindings={"index": "IDX"})[0]
    else:
        node = rb.LiveNode([s], list(data), cfg, bar_stamp="close", instruments=specs, bindings={"index": "IDX"},
                           execution=rb.execution.PaperExecution() if mode == "paper" else None)
        for i in range(n):
            node.push_group(int(data["IDX"]["timestamps"][i]),
                            {x: tuple(float(data[x][c][i]) for c in ("open", "high", "low", "close", "volume")) for x in data})
        res = node.finish()[0]
    return s, res


@pytest.mark.parametrize("timing", ["same_bar_close", "next_bar_open"])
def test_a_rolls_open_follows_the_closes_fill_in_the_same_step_in_every_mode(timing):
    runs = {mode: _roll_run(mode, timing) for mode in MODES}
    for mode, (s, res) in runs.items():
        close, open_ = (f for f in s.fills if f[0].startswith("fut:roll0"))
        assert close[1] == open_[1], f"{mode}: the position was flat between {close[1]} and {open_[1]}"
        trades = [(t.symbol, t.entry_idx, t.exit_idx, t.entry_price, t.exit_price) for t in res.result.trades()]
        assert trades == [(t.symbol, t.entry_idx, t.exit_idx, t.entry_price, t.exit_price)
                          for t in runs["batch"][1].result.trades()], mode
    if timing == "next_bar_open":
        # the close and the open both fill at the open of the 09:31 bar
        s, _ = runs["batch"]
        assert [(f[1]) for f in s.fills[1:]] == ["11-02 09:31", "11-02 09:31"]
        assert s.fills[2][2] == pytest.approx(121.45)


def test_composites_on_a_venue_no_preset_covers_and_no_session_are_refused():
    with pytest.raises(rb.errors.InvalidParameter) as e:
        _composites(venue="NCO")
    assert "session" in str(e.value) and "NCO" in str(e.value)
    assert _composites(session=None), "an instrument with no venue keeps the clock grid"
