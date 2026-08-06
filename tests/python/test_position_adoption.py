"""Position adoption (0.6.2) — guard tests.

Plain words: a strategy attached to a stock the user ALREADY OWNS must start
out knowing it holds those shares, at the price the user really paid, without
the engine pretending a buy happened.

The tempting shortcut — submit a fake buy at the average price — is wrong in
three ways that each cost real money or produce a false signal:

  1. it charges brokerage nobody ever paid, so every metric downstream of
     equity is off by that fee;
  2. it writes a trade into the log that never happened, so trade counts and
     win-rate describe a history the user did not live; and
  3. it emits an entry event, and a live deployment that diffs positions
     around each push reads that as "open a new position" and sends the
     broker an order to buy shares the user already holds.

So adoption opens a ledger position directly: no order, no fill, no fees, no
trade record, no Entered event. Cash drops by the cost basis so equity reads
as initial + unrealized, exactly like an account that bought earlier.

Each test below pins one of those properties. If one fails, the shortcut has
crept back in and the failure names which of the three consequences returned.

Deliberate refusals, also pinned here: adoption is cash-account-only and
long-only, and a malformed seed is rejected rather than skipped. Margin
adoption is refused because the margin already posted against a broker-held
position cannot be derived from quantity and average price — inventing a
figure there would misstate free capital, which is the number that decides
whether the next entry is allowed.
"""

import numpy as np
import pytest

from raptorbt import PyBacktestConfig, Strategy, TickStrategyStream
from raptorbt._raptorbt import PyPortfolioSession

SYMBOL = "RELIANCE"
QUANTITY = 100.0
AVG_PRICE = 90.0
INITIAL_CAPITAL = 100_000.0
COST_BASIS = QUANTITY * AVG_PRICE  # 9_000.0
DAY_NS = 86_400_000_000_000


class Passive(Strategy):
    """Never trades, so anything in the results came from adoption alone."""

    def on_bar(self, ctx):
        pass

    def on_trade_tick(self, ctx, tick):
        pass


def _config(**kwargs):
    kwargs.setdefault("initial_capital", INITIAL_CAPITAL)
    kwargs.setdefault("fees", 0.001)
    return PyBacktestConfig(**kwargs)


def _bars(closes, start_ts=0):
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    return {
        "timestamps": np.arange(start_ts, start_ts + n * DAY_NS, DAY_NS, dtype=np.int64),
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": np.ones(n, dtype=np.float64),
    }


def _sealed_session(closes, config=None, account_type="cash"):
    """A sealed one-instrument session, ready to adopt into."""
    session = PyPortfolioSession(config=config or _config(), account_type=account_type)
    instrument = session.add_instrument(SYMBOL, direction=1)
    bars = _bars(closes)
    session.set_bars(
        instrument,
        bars["timestamps"],
        bars["open"],
        bars["high"],
        bars["low"],
        bars["close"],
        bars["volume"],
    )
    session.seal()
    return session, instrument


def _drain(session):
    while session.current_event() is not None:
        session.apply_current()
    return session.finish()


# --- The three properties that make adoption worth having -------------------


def test_adoption_charges_no_fees():
    """Consequence 1: a fake buy would charge brokerage nobody paid."""
    session, instrument = _sealed_session([100.0, 102.0, 104.0, 106.0])
    session.adopt_position(instrument, 0, AVG_PRICE, QUANTITY)
    result = _drain(session)

    assert result.metrics.total_fees_paid == pytest.approx(0.0, abs=1e-12)


def test_adoption_writes_no_closed_trade():
    """Consequence 2: a fake buy would invent a trade the user never made.

    The adopted holding is an OPEN position, not a completed round trip, so
    it must not appear among closed trades or shift win-rate arithmetic.
    """
    session, instrument = _sealed_session([100.0, 102.0, 104.0, 106.0])
    session.adopt_position(instrument, 0, AVG_PRICE, QUANTITY)
    result = _drain(session)

    assert result.metrics.total_closed_trades == 0
    assert result.metrics.total_open_trades == 1


def test_adopted_position_is_visible_before_any_event_is_applied():
    """Consequence 3: the position must never look like a fresh entry.

    A live deployment turns engine state into broker orders by diffing
    positions around each push. If the holding appeared only AFTER the first
    event, that diff would read it as a new entry and buy shares the user
    already owns. So it has to be present in the very first snapshot.
    """
    session, instrument = _sealed_session([100.0, 102.0, 104.0])
    session.adopt_position(instrument, 0, AVG_PRICE, QUANTITY)

    snapshot = session.position(instrument)
    assert snapshot is not None, "adopted position missing before the first event"
    assert snapshot.size == pytest.approx(QUANTITY)
    assert snapshot.entry_price == pytest.approx(AVG_PRICE)
    assert snapshot.direction == 1


# --- Cash and equity arithmetic ---------------------------------------------


def test_cash_drops_by_cost_basis_and_equity_holds():
    """Equity must read as an account that bought earlier, not one funded extra.

    Cash falls by exactly the cost basis — the money became shares. Equity is
    marked to market by the event loop, so it picks the holding up on the
    first applied event; priced at the adoption price the holding is worth
    exactly what it cost, so equity returns to the initial capital rather
    than showing the cash outflow as a loss.
    """
    session, instrument = _sealed_session([AVG_PRICE, AVG_PRICE])
    session.adopt_position(instrument, 0, AVG_PRICE, QUANTITY)

    # Cash is reduced immediately, at adoption time.
    assert session.cash() == pytest.approx(INITIAL_CAPITAL - COST_BASIS)

    session.apply_current()  # first mark-to-market

    assert session.cash() == pytest.approx(INITIAL_CAPITAL - COST_BASIS)
    assert session.equity() == pytest.approx(INITIAL_CAPITAL)


def test_open_trade_pnl_marks_against_the_current_price():
    """Unrealized PnL is measured from the real average cost, not the first bar."""
    last_close = 106.0
    session, instrument = _sealed_session([100.0, 102.0, 104.0, last_close])
    session.adopt_position(instrument, 0, AVG_PRICE, QUANTITY)
    result = _drain(session)

    expected = (last_close - AVG_PRICE) * QUANTITY
    assert result.metrics.open_trade_pnl == pytest.approx(expected)


# --- Deliberate refusals ----------------------------------------------------


def test_margin_adoption_is_refused_not_guessed():
    """Margin posted against a broker-held position is not derivable here.

    Guessing it would misstate free capital, which gates every later entry —
    so this must stay a refusal rather than becoming a silent estimate.
    """
    session, instrument = _sealed_session(
        [100.0, 102.0], config=_config(), account_type="margin"
    )
    with pytest.raises(ValueError, match="cash accounts only"):
        session.adopt_position(instrument, 0, AVG_PRICE, QUANTITY)


@pytest.mark.parametrize(
    "price, size",
    [
        (0.0, QUANTITY),  # no average cost
        (AVG_PRICE, 0.0),  # no shares
        (-AVG_PRICE, QUANTITY),  # negative cost
        (AVG_PRICE, -QUANTITY),  # short: adoption is long-only
    ],
)
def test_malformed_seed_is_refused(price, size):
    """A bad seed must raise, never be skipped.

    Skipping would leave the strategy believing it is flat while the user
    holds shares — the exact confusion adoption exists to prevent.
    """
    session, instrument = _sealed_session([100.0, 102.0])
    with pytest.raises(ValueError, match="positive price and size"):
        session.adopt_position(instrument, 0, price, size)


def test_adopting_over_an_existing_position_is_refused():
    """One holding per instrument: a second adoption must not stack silently.

    Stacking would double-count the cost basis against cash and leave the
    strategy holding more than the broker reports.
    """
    session, instrument = _sealed_session([100.0, 102.0, 104.0])
    session.adopt_position(instrument, 0, AVG_PRICE, QUANTITY)

    with pytest.raises(ValueError, match="already open"):
        session.adopt_position(instrument, 0, AVG_PRICE, QUANTITY)


def test_adoption_ordering_is_a_convention_not_an_enforced_guard():
    """Documents a real gap: adopting mid-run is ALLOWED, and should not be.

    Callers are told to adopt after seal() and before the first
    apply_current(), because a position adopted mid-run is priced into an
    equity curve that already ran without it. The engine does not enforce
    that ordering today — this test pins the current permissive behaviour so
    the gap stays visible rather than being mistaken for a guarantee.

    If adoption is later restricted to pre-run only, this test SHOULD fail:
    replace it with a `pytest.raises` assertion at that point.
    """
    session, instrument = _sealed_session([100.0, 102.0, 104.0])
    session.apply_current()

    position_id = session.adopt_position(instrument, 0, AVG_PRICE, QUANTITY)

    assert position_id == 0
    assert session.cash() == pytest.approx(INITIAL_CAPITAL - COST_BASIS)


# --- The ergonomic path: TickStrategyStream(initial_positions=...) ----------


def test_stream_seeds_the_position_before_the_first_push():
    """The documented entry point must land the same state as the primitive."""
    stream = TickStrategyStream(
        Passive(),
        symbols=[SYMBOL],
        config=_config(),
        initial_positions={SYMBOL: {"quantity": QUANTITY, "avg_price": AVG_PRICE}},
    )

    snapshot = stream.ctx.position_for(SYMBOL)
    assert snapshot is not None, "seeded position missing before the first push"
    assert snapshot.size == pytest.approx(QUANTITY)
    assert snapshot.entry_price == pytest.approx(AVG_PRICE)
    assert snapshot.direction == 1
    assert stream.ctx.equity == pytest.approx(INITIAL_CAPITAL - COST_BASIS)


def test_stream_adoption_survives_pushes_without_becoming_an_entry():
    """Pushing events must not re-open, duplicate, or re-cost the holding."""
    stream = TickStrategyStream(
        Passive(),
        symbols=[SYMBOL],
        config=_config(),
        initial_positions={SYMBOL: {"quantity": QUANTITY, "avg_price": AVG_PRICE}},
    )
    before = stream.ctx.position_for(SYMBOL)

    base = 1_700_000_000_000_000_000
    for step, price in enumerate([101.0, 103.0, 102.0]):
        stream.push_tick(SYMBOL, base + step * 1_000_000_000, price)

    after = stream.ctx.position_for(SYMBOL)
    assert after is not None
    assert after.size == pytest.approx(before.size)
    assert after.entry_price == pytest.approx(before.entry_price)

    result = stream.finish()
    assert result.metrics.total_fees_paid == pytest.approx(0.0, abs=1e-12)
    assert result.metrics.total_closed_trades == 0


@pytest.mark.parametrize(
    "seed",
    [
        {"quantity": 0, "avg_price": AVG_PRICE},
        {"quantity": QUANTITY, "avg_price": 0},
    ],
)
def test_stream_refuses_a_malformed_seed(seed):
    with pytest.raises(ValueError, match="positive quantity and avg_price"):
        TickStrategyStream(
            Passive(),
            symbols=[SYMBOL],
            config=_config(),
            initial_positions={SYMBOL: seed},
        )


def test_stream_refuses_an_unknown_symbol():
    """A typo must fail loudly, not silently seed nothing."""
    with pytest.raises(ValueError, match="unknown symbol"):
        TickStrategyStream(
            Passive(),
            symbols=[SYMBOL],
            config=_config(),
            initial_positions={"NOT_REGISTERED": {"quantity": 1, "avg_price": 1.0}},
        )


def test_stream_refuses_margin_adoption():
    with pytest.raises(ValueError, match="cash accounts only"):
        TickStrategyStream(
            Passive(),
            symbols=[SYMBOL],
            config=_config(),
            account_type="margin",
            leverage=2.0,
            initial_positions={SYMBOL: {"quantity": QUANTITY, "avg_price": AVG_PRICE}},
        )
