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

Ordering is enforced, not merely documented: adoption must happen before the
first equity sample. Adopting mid-run leaves the curve flat for the
pre-adoption stretch, which holds the running peak down and makes the later
decline measure against the wrong high-water mark — a real 0.495% drawdown
reporting as 0.199%. The curve is written streaming, so it cannot be repaired
afterwards. Quote and depth events sample no equity, so adopting after one is
still allowed; the gate is the equity curve, not the event cursor.

Deliberate refusals, also pinned here: adoption is long-only, a malformed seed
is rejected rather than skipped, and a LEVERAGED book is refused. Fully funded
books (leverage 1.0) are supported and fund the holding by locking the notional
rather than debiting cash, which is what the margin equity formula requires —
that path is what lets a long/short strategy, which must run under a margin
account for its short to transact at all, be seeded. Above leverage 1.0 the
margin a broker has already posted against a position it holds cannot be
derived from quantity and average price, and inventing a figure there would
misstate free capital, the number that decides whether the next entry is
allowed.
"""

import numpy as np
import pytest

from raptorbt import BacktestConfig, Strategy, TickStrategyStream
from raptorbt._raptorbt import PortfolioSession

SYMBOL = "RELIANCE"
QUANTITY = 100.0
AVG_PRICE = 90.0
INITIAL_CAPITAL = 100_000.0
COST_BASIS = QUANTITY * AVG_PRICE  # 9_000.0
DAY_NS = 86_400_000_000_000

# A steadily falling market, so the equity curve has a real peak-to-trough
# decline to measure. Adopted pre-run this reports a 0.495% max drawdown.
FALLING_CLOSES = [100.0, 99.0, 98.0, 97.0, 96.0, 95.0]


class Passive(Strategy):
    """Never trades, so anything in the results came from adoption alone."""

    def on_bar(self, ctx):
        pass

    def on_trade_tick(self, ctx, tick):
        pass


def _config(**kwargs):
    kwargs.setdefault("initial_capital", INITIAL_CAPITAL)
    kwargs.setdefault("fees", 0.001)
    return BacktestConfig(**kwargs)


def _bars(closes, start_ts=0):
    closes = np.asarray(closes, dtype=np.float64)
    n = len(closes)
    return {
        "timestamps": np.arange(
            start_ts, start_ts + n * DAY_NS, DAY_NS, dtype=np.int64
        ),
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": np.ones(n, dtype=np.float64),
    }


def _sealed_session(closes, config=None, account_type="cash", leverage=1.0):
    """A sealed one-instrument session, ready to adopt into."""
    session = PortfolioSession(
        config=config or _config(), account_type=account_type, leverage=leverage
    )
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


def test_leveraged_adoption_is_refused_not_guessed():
    """Under leverage the broker's posted margin is genuinely not derivable.

    Quantity and average price do not tell you what margin the broker has
    already posted against a position it holds, and inventing a figure would
    misstate free capital — the number that gates every later entry. So this
    stays a refusal rather than becoming a silent estimate.
    """
    session, instrument = _sealed_session(
        [100.0, 102.0], config=_config(), account_type="margin", leverage=2.0
    )
    with pytest.raises(ValueError, match="fully funded"):
        session.adopt_position(instrument, 0, AVG_PRICE, QUANTITY)


def test_fully_funded_margin_adoption_is_allowed():
    """A leverage-1.0 book locks the whole notional, so margin IS the cost basis.

    This is the case the old blanket cash-only refusal got wrong. A strategy
    with any short leg must run under a margin account or the short's P&L
    never reaches equity, so refusing margin outright meant a seeded
    long/short book could not be deployed at all.
    """
    session, instrument = _sealed_session(
        [AVG_PRICE, AVG_PRICE], config=_config(), account_type="margin", leverage=1.0
    )

    position_id = session.adopt_position(instrument, 0, AVG_PRICE, QUANTITY)

    assert position_id == 0
    snapshot = session.position(instrument)
    assert snapshot.size == pytest.approx(QUANTITY)
    assert snapshot.entry_price == pytest.approx(AVG_PRICE)


def test_margin_adoption_locks_rather_than_debiting():
    """Margin funds an open by locking the notional, not by debiting cash.

    Margin equity is `balance + unrealized` — there is no position-value
    term — so a cash-style debit would never be offset and would understate
    equity by the cost basis for the whole run. Free capital must still fall
    by the cost basis; if this ever reads the full initial capital, the
    session stopped reconciling the locked delta and portfolio risk limits
    are being computed against money that is not available.
    """
    session, instrument = _sealed_session(
        [AVG_PRICE, AVG_PRICE], config=_config(), account_type="margin", leverage=1.0
    )
    session.adopt_position(instrument, 0, AVG_PRICE, QUANTITY)

    # The balance is untouched — in margin mode cash() includes locked margin.
    assert session.cash() == pytest.approx(INITIAL_CAPITAL)
    # But the capital that can fund a new position has dropped.
    assert session.free_capital() == pytest.approx(INITIAL_CAPITAL - COST_BASIS)


def test_cash_and_fully_funded_margin_report_the_same_numbers():
    """Fully funded margin is economically identical to cash for a long hold.

    Any divergence means the funding arm and the mode's equity formula
    disagree. Note this compares free_capital(), NOT cash(): in margin mode
    cash() returns the balance including locked margin, so the two modes
    legitimately differ there. That asymmetry is the design, and a test
    asserting cash() equality would fail for the wrong reason.
    """
    results = {}
    for mode in ("cash", "margin"):
        session, instrument = _sealed_session(
            FALLING_CLOSES, config=_config(), account_type=mode, leverage=1.0
        )
        session.adopt_position(instrument, 0, AVG_PRICE, QUANTITY)
        free_capital = session.free_capital()
        session.apply_current()
        equity = session.equity()
        result = _drain(session)
        results[mode] = (free_capital, equity, result.metrics)

    cash_free, cash_equity, cash_metrics = results["cash"]
    margin_free, margin_equity, margin_metrics = results["margin"]

    assert cash_free == pytest.approx(margin_free)
    assert cash_equity == pytest.approx(margin_equity)
    assert cash_metrics.open_trade_pnl == pytest.approx(margin_metrics.open_trade_pnl)
    assert cash_metrics.total_fees_paid == pytest.approx(margin_metrics.total_fees_paid)
    assert cash_metrics.total_closed_trades == margin_metrics.total_closed_trades
    assert cash_metrics.max_drawdown_pct == pytest.approx(
        margin_metrics.max_drawdown_pct
    )


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


def test_adoption_after_the_first_event_is_refused():
    """Adopting mid-run understates max drawdown, in the flattering direction.

    The equity curve is written streaming, one sample per applied event,
    against a running peak seeded to the initial capital. A position adopted
    after the run started leaves that curve FLAT for the pre-adoption
    stretch, which holds the peak down; the decline that follows is then
    measured against a high-water mark lower than the truth.

    Measured on the falling fixture below (6 bars 100 -> 95, adopting 100
    shares at 90): adopting pre-run reports a 0.495% max drawdown, adopting
    after three applied events reports 0.199%. Total return and
    open_trade_pnl are IDENTICAL in both — only the risk number moves, and it
    moves to look safer than reality, which is the worst direction for a risk
    metric to be wrong in.

    Because the curve is written streaming, the samples are already wrong by
    the time metrics are computed; there is no repairing it afterwards. So
    the bad ordering is refused instead.
    """
    session, instrument = _sealed_session(FALLING_CLOSES)
    session.apply_current()

    with pytest.raises(ValueError, match="before the first applied event"):
        session.adopt_position(instrument, 0, AVG_PRICE, QUANTITY)


def test_pre_run_adoption_reports_the_true_drawdown():
    """Pins the correct number, not merely the refusal.

    Without this, the ordering guard could be removed and no test would
    notice max drawdown drifting. This is what makes that guard load-bearing
    rather than decorative.
    """
    session, instrument = _sealed_session(FALLING_CLOSES)
    session.adopt_position(instrument, 0, AVG_PRICE, QUANTITY)
    result = _drain(session)

    assert result.metrics.max_drawdown_pct == pytest.approx(0.495, abs=0.002)


def test_adoption_is_still_allowed_after_a_quote_only_event():
    """The gate is the equity curve, not the event cursor.

    A quote (and a depth snapshot) advances the schedule cursor but samples
    NO equity — deliberately, since marking on a quote would append a zero
    return per quote and distort annualized metrics by how chatty the feed
    is. A live feed routinely delivers quotes before the first trade print,
    and a broker's holdings callback can easily return after them.

    So adopting once a quote has been applied corrupts nothing and must keep
    working. Gating on the cursor instead would reject this valid sequence —
    this test is the regression a future refactor is most likely to
    reintroduce.
    """
    session = PortfolioSession(config=_config())
    instrument = session.add_instrument(SYMBOL, direction=1)
    session.seal()

    # ltp = 0 means "no trade print": this pushes a quote and nothing else.
    appended = session.push_tick(instrument, 1_000, 0.0, 99.0, 101.0)
    assert appended == 1, "expected exactly one quote event"
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


def test_stream_refuses_leveraged_adoption():
    """A leveraged book still cannot be seeded, and says why."""
    with pytest.raises(ValueError, match="fully funded"):
        TickStrategyStream(
            Passive(),
            symbols=[SYMBOL],
            config=_config(),
            account_type="margin",
            leverage=2.0,
            initial_positions={SYMBOL: {"quantity": QUANTITY, "avg_price": AVG_PRICE}},
        )


def test_stream_seeds_a_long_short_book():
    """The production scenario that could not deploy at all before.

    A strategy with any short leg is given `account_type="margin"` (at
    leverage 1.0, so the book stays fully funded), because a short only
    transacts as a short under a margin account. Adoption used to refuse
    margin outright, so a seeded long/short strategy raised at construction
    and could never be deployed — the seed and the short were mutually
    exclusive.
    """
    stream = TickStrategyStream(
        Passive(),
        symbols=["LONGSYM", "SHORTSYM"],
        config=_config(),
        directions={"LONGSYM": 1, "SHORTSYM": -1},
        account_type="margin",
        leverage=1.0,
        initial_positions={"LONGSYM": {"quantity": QUANTITY, "avg_price": AVG_PRICE}},
    )

    seeded = stream.ctx.position_for("LONGSYM")
    assert seeded is not None, "the seeded holding must exist before the first push"
    assert seeded.size == pytest.approx(QUANTITY)
    assert seeded.direction == 1
    # The short leg is registered but flat: seeding one symbol must not
    # fabricate a position on another.
    assert stream.ctx.position_for("SHORTSYM") is None
