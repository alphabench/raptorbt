"""A multi-leg group fills all legs or none, funded by its combined margin.

In plain words: a sold call whose cost is only bearable because a long call
hedges it (a bear call spread) must not fill without the hedge, and must not
have to wait for it. The legs are linked with ``PortfolioSession.link_group``;
they fill together on the event where every leg is marketable, funded as one
group, or each is rejected and nothing opens.
"""

import numpy as np
import pytest

from raptorbt import BacktestConfig, InstrumentSpec, MarginModel, PortfolioSession

UNDERLYING = "NIFTY"
SOLD, HEDGE = "NIFTY24000CE", "NIFTY24200CE"
LOT = 50.0
N = 6


def _spec(symbol, strike):
    return InstrumentSpec.option(
        symbol,
        strike=strike,
        right="call",
        expiration_ns=10_000_000,
        lot_size=LOT,
        underlying=UNDERLYING,
    )


def _bars(premium):
    closes = np.full(N, float(premium))
    return dict(
        timestamps=np.arange(0, N * 10, 10, dtype=np.int64),
        open=closes.copy(), high=closes + 1.0, low=closes - 1.0, close=closes,
        volume=np.full(N, 1_000.0),
    )


def _run(capital, order=(SOLD, HEDGE), request_order=(SOLD, HEDGE), group=True):
    """Sell one lot of SOLD, buy one lot of HEDGE, from bar 1. Returns trades by symbol,
    the entry timestamps, and the rejection reasons of the orders."""
    session = PortfolioSession(
        BacktestConfig(initial_capital=capital, fees=0.0, margin_model=MarginModel.defined_risk()),
        account_type="margin",
        leverage=1.0,
    )
    spec = {SOLD: _spec(SOLD, 24_000.0), HEDGE: _spec(HEDGE, 24_200.0)}
    premium = {SOLD: 100.0, HEDGE: 40.0}
    index = {}
    for symbol in order:
        index[symbol] = session.add_instrument(
            symbol, direction=1, instrument=spec[symbol], oms_type="netting"
        )
        session.set_bars(index[symbol], **_bars(premium[symbol]))
    session.seal()
    side = {SOLD: "sell", HEDGE: "buy"}
    ids = {}
    submitted = False
    events = []
    while (cur := session.current_event()) is not None:
        _, _, _, ts, *_ = cur
        if not submitted and ts >= 10:
            for symbol in request_order:
                ids[symbol] = session.submit_order(
                    index[symbol], side[symbol], "market", 0, ts, f"leg-{symbol}", units=LOT
                )
            if group:
                session.link_group([(index[s], ids[s]) for s in request_order])
            submitted = True
        events.extend(session.apply_current())
    result = session.finish()
    trades = {t.symbol: (t.entry_idx, t.entry_price, t.size, t.direction) for t in result.result.trades()}
    reasons = sorted(o.reject_reason for o in result.result.orders() if o.reject_reason)
    return trades, reasons, result


def test_the_hedged_pair_fills_together():
    # 20,000 carries the spread's worst loss but not the sold call's naked deposit.
    trades, reasons, _ = _run(20_000.0)
    assert set(trades) == {SOLD, HEDGE}
    assert trades[SOLD][0] == trades[HEDGE][0], "both legs fill on the same bar"
    assert reasons == []


def test_an_unfundable_group_leaves_no_leg_filled():
    # The hedge alone is affordable (40 x 50 = 2,000); the group is not.
    trades, reasons, result = _run(3_000.0)
    assert trades == {}
    assert reasons == ["group_unfunded", "group_unfunded"]
    assert result.unfunded_entries >= 1 or result.result.unfunded_entries >= 1


@pytest.mark.parametrize("request_order", [(SOLD, HEDGE), (HEDGE, SOLD)])
@pytest.mark.parametrize("register", [(SOLD, HEDGE), (HEDGE, SOLD)])
def test_the_order_of_legs_does_not_change_the_result(request_order, register):
    reference, _, ref_result = _run(20_000.0)
    trades, _, result = _run(20_000.0, order=register, request_order=request_order)
    assert trades == reference
    assert result.result.metrics.end_value == ref_result.result.metrics.end_value
