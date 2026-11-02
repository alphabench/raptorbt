"""TWAP schedules in every strategy runner.

A schedule submitted from a portfolio, tick, stream or multi-strategy run
releases slices exactly as it does in the single-instrument runner, and its
lifecycle events reach the same hooks.
"""

import numpy as np
import pytest

import raptorbt
from raptorbt import BacktestConfig, Strategy, TickStrategyStream
from raptorbt.strategy import orders


def _cfg():
    return BacktestConfig(initial_capital=100_000.0, fees=0.0, slippage=0.0)


def _bars(n, price=100.0, step=1):
    p = np.full(n, price)
    return {
        "timestamps": np.arange(0, n * step, step, dtype=np.int64),
        "open": p,
        "high": p,
        "low": p,
        "close": p,
        "volume": np.ones(n),
    }


class TwapOnce(Strategy):
    """Submit one schedule on the first event, recording every order event."""

    def __init__(self, config=None, units=30.0, slices=3, symbol=None):
        super().__init__(config)
        self.units, self.slices, self.symbol = units, slices, symbol
        self.sent = False
        self.fills = []
        self.algos = []

    def _go(self):
        if not self.sent:
            self.sent = True
            self.submit_order(
                orders.Twap(side="buy", units=self.units, slices=self.slices, every=1),
                symbol=self.symbol,
            )

    def on_bar(self, ctx):
        self._go()

    def on_trade_tick(self, ctx, tick):
        self._go()

    def on_order_filled(self, ctx, event):
        self.fills.append(("fill", event.idx, event.size))

    def on_algo_started(self, ctx, event):
        self.algos.append(("started", event.idx))

    def on_algo_completed(self, ctx, event):
        self.algos.append(("completed", event.idx))


def _trades(result):
    return [(t.entry_idx, t.exit_idx, t.size) for t in result.result.trades()]


def test_run_strategies_matches_run_portfolio_strategy():
    data = {"A": _bars(6)}
    legacy, shared = TwapOnce(), TwapOnce()
    a = raptorbt.run_portfolio_strategy(legacy, data, config=_cfg(), oms_type="hedging")
    (b,) = raptorbt.run_strategies([shared], data, config=_cfg(), oms_type="hedging")
    assert legacy.fills == [("fill", 0, 10.0), ("fill", 1, 10.0), ("fill", 2, 10.0)]
    assert (legacy.fills, legacy.algos) == (shared.fills, shared.algos)
    assert _trades(a) == _trades(b)
    np.testing.assert_array_equal(a.result.equity_curve(), b.result.equity_curve())


def test_run_strategies_with_several_strategies():
    data = {"A": _bars(8), "B": _bars(8, price=50.0)}
    alone = TwapOnce(units=30.0, slices=3, symbol="A")
    raptorbt.run_portfolio_strategy(alone, data, config=_cfg(), oms_type="hedging")
    items = [
        (TwapOnce(units=30.0, slices=3, symbol="A"), ["A", "B"]),
        (TwapOnce(units=8.0, slices=4, symbol="B"), ["A", "B"]),
        (TwapOnce(units=5.0, slices=1), ["B"]),
    ]
    results = raptorbt.run_strategies(items, data, config=_cfg(), oms_type="hedging")
    first, second, third = (s for s, _ in items)
    # Strategies do not see each other's schedules.
    assert first.fills == alone.fills
    assert [size for _, _, size in second.fills] == [2.0, 2.0, 2.0, 2.0]
    assert third.fills == [("fill", 0, 5.0)]
    assert [len(first.algos), len(second.algos), len(third.algos)] == [2, 2, 2]
    assert len(results) == 3


def test_batch_tick_runner_slices_twap():
    ticks = {"A": {"timestamps": np.arange(6, dtype=np.int64), "ltp": np.full(6, 100.0)}}
    s = TwapOnce()
    raptorbt.run_tick_strategy(s, ticks, config=_cfg(), oms_type="hedging")
    assert [size for _, _, size in s.fills] == [10.0, 10.0, 10.0]
    assert [kind for kind, _ in s.algos] == ["started", "completed"]


def test_tick_stream_slices_twap_on_push_tick():
    s = TwapOnce()
    stream = TickStrategyStream(s, ["A"], config=_cfg(), oms_type="hedging")
    for ts in range(6):
        stream.push_tick("A", ts, 100.0)
    stream.finish()
    assert [size for _, _, size in s.fills] == [10.0, 10.0, 10.0]
    assert [kind for kind, _ in s.algos] == ["started", "completed"]
    batch = TwapOnce()
    ticks = {"A": {"timestamps": np.arange(6, dtype=np.int64), "ltp": np.full(6, 100.0)}}
    raptorbt.run_tick_strategy(batch, ticks, config=_cfg(), oms_type="hedging")
    assert (s.fills, s.algos) == (batch.fills, batch.algos)


# ---------------------------------------------------------------------------
# cancel by client id
# ---------------------------------------------------------------------------
CANCEL_AT = 2  # the event on which the strategy cancels; the schedule has 6 slices


class TwapThenCancel(Strategy):
    """Schedule 6 slices on event 0 and cancel the schedule on event 2."""

    def __init__(self, config=None):
        super().__init__(config)
        self.n = 0
        self.client_id = None
        self.fills = []
        self.algos = []

    def _step(self):
        if self.n == 0:
            self.client_id = self.submit_order(
                orders.Twap(side="buy", units=60.0, slices=6, every=1)
            )
        elif self.n == CANCEL_AT:
            self.cancel_order(self.client_id)
        self.n += 1

    def on_bar(self, ctx):
        self._step()

    def on_trade_tick(self, ctx, tick):
        self._step()

    def on_order_filled(self, ctx, event):
        self.fills.append(event.size)

    def on_algo_started(self, ctx, event):
        self.algos.append("started")

    def on_algo_completed(self, ctx, event):
        self.algos.append("completed")


N_EVENTS = 10


def _run_single(s):
    raptorbt.run_strategy_backtest(s, **_bars(N_EVENTS), config=_cfg(), oms_type="hedging")


def _run_portfolio(s):
    raptorbt.run_portfolio_strategy(s, {"A": _bars(N_EVENTS)}, config=_cfg(), oms_type="hedging")


def _run_shared(s):
    raptorbt.run_strategies([s], {"A": _bars(N_EVENTS)}, config=_cfg(), oms_type="hedging")


def _ticks():
    return {"A": {"timestamps": np.arange(N_EVENTS, dtype=np.int64), "ltp": np.full(N_EVENTS, 100.0)}}


def _run_tick(s):
    raptorbt.run_tick_strategy(s, _ticks(), config=_cfg(), oms_type="hedging")


def _run_stream(s):
    stream = TickStrategyStream(s, ["A"], config=_cfg(), oms_type="hedging")
    for ts in range(N_EVENTS):
        stream.push_tick("A", ts, 100.0)
    stream.finish()


RUNNERS = [_run_single, _run_portfolio, _run_shared, _run_tick, _run_stream]


@pytest.mark.parametrize("run", RUNNERS, ids=lambda f: f.__name__[5:])
def test_cancel_by_client_id_stops_the_schedule(run):
    s = TwapThenCancel()
    run(s)
    # Slices went out on events 0 and 1 only; the rest of the 6 never did.
    assert s.fills == [10.0, 10.0]
    # Started once; the cancelled schedule is reported finished exactly once.
    assert s.algos == ["started", "completed"]


def test_cancel_by_client_id_is_the_same_on_every_runner():
    seen = []
    for run in RUNNERS:
        s = TwapThenCancel()
        run(s)
        seen.append((s.fills, s.algos))
    assert all(x == seen[0] for x in seen)


@pytest.mark.parametrize("run", RUNNERS, ids=lambda f: f.__name__[5:])
def test_cancelling_a_finished_schedule_is_a_no_op(run):
    class Late(TwapThenCancel):
        def _step(self):
            if self.n == 0:
                self.client_id = self.submit_order(
                    orders.Twap(side="buy", units=20.0, slices=2, every=1)
                )
            elif self.n == 5:
                self.cancel_order(self.client_id)
            self.n += 1

    s = Late()
    run(s)
    assert s.fills == [10.0, 10.0]
    assert s.algos == ["started", "completed"]
