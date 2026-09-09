"""Every order the run placed, including the ones that never filled.

A backtest reports the trades a strategy made. Without an order log it
cannot report the trades it *tried* to make and could not — an entry
refused for margin, a limit that rested all day and expired, a stop
rejected because a position was already open. None of those produce a
trade, so a run that never got into the market looks exactly like one
whose idea was wrong.

These tests pin the two halves that matter: an order that filled is
described by what it actually filled at, and an order that did NOT fill
still appears, with the reason it did not.
"""

import numpy as np

from raptorbt import Strategy, run_strategy_backtest
from raptorbt.strategy import orders as O

SEC_NS = 1_000_000_000


def _bars(close, high=None, low=None):
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    return {
        "timestamps": np.arange(n, dtype=np.int64) * SEC_NS,
        "open": close,
        "high": np.asarray(high if high is not None else close * 1.002, dtype=np.float64),
        "low": np.asarray(low if low is not None else close * 0.998, dtype=np.float64),
        "close": close,
        "volume": np.full(n, 1_000_000.0),
    }


class _MarketOnce(Strategy):
    """One market order on the first bar, then nothing."""

    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.submit_order(O.Market(side="buy", units=10.0), client_id="entry")


class _UnreachableLimit(Strategy):
    """A buy limit far below the market: it rests and never fills."""

    def on_bar(self, ctx):
        if ctx.idx == 0:
            self.submit_order(
                O.Limit(side="buy", units=10.0, price=1.0), client_id="never"
            )


def test_a_filled_order_reports_what_it_actually_filled():
    result = run_strategy_backtest(_MarketOnce, **_bars(np.full(20, 100.0)))
    log = result.orders()
    assert log, "an order was placed; the log must not be empty"

    order = next(o for o in log if o.client_id == "entry")
    assert order.status == "filled"
    assert order.side == "buy"
    assert order.kind == "market"
    assert order.requested_qty == 10.0
    assert order.filled_qty == 10.0
    assert order.avg_fill_price is not None and order.avg_fill_price > 0
    assert order.fill_slices >= 1
    # Not rejected, so the reason is absent rather than an empty string.
    assert order.reject_reason is None


def test_an_order_that_never_filled_is_still_in_the_log():
    """The whole point: silence is not the same as nothing having happened."""
    result = run_strategy_backtest(_UnreachableLimit, **_bars(np.full(20, 100.0)))
    log = result.orders()
    assert log, "a resting order that never filled must still be reported"

    order = next(o for o in log if o.client_id == "never")
    assert order.filled_qty == 0.0
    assert order.avg_fill_price is None, "never filled, so there is no fill price"
    assert order.status != "filled"
    assert order.limit_price == 1.0


def test_a_run_that_placed_no_typed_orders_reports_an_empty_log():
    class _NoOrders(Strategy):
        def on_bar(self, ctx):
            pass

    result = run_strategy_backtest(_NoOrders, **_bars(np.full(10, 100.0)))
    assert result.orders() == []


def test_the_log_reads_in_submission_order():
    class _Three(Strategy):
        def on_bar(self, ctx):
            if ctx.idx == 0:
                for i in range(3):
                    self.submit_order(
                        O.Limit(side="buy", units=1.0, price=1.0),
                        client_id=f"o{i}",
                    )

    result = run_strategy_backtest(_Three, **_bars(np.full(10, 100.0)))
    ids = [o.client_id for o in result.orders()]
    assert ids == ["o0", "o1", "o2"], ids
