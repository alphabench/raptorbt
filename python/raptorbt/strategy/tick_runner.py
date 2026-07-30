"""Tick-driven driver for class-based strategies.

Feeds trade prints and quotes through the same event session the bar runner
uses, so orders, positions, risk gates and the shared account behave
identically — only the resolution changes.

Two things are deliberately asymmetric with the bar runner:

- **Quotes are observation only.** They fire ``on_quote`` and update
  ``ctx.best_bid`` / ``ctx.best_ask``, but do not fill orders, move trailing
  stops, or mark equity. An order submitted from ``on_quote`` rests and
  matches against the next print, which is the first evidence of a trade at
  that price.
- **Bars are a view, not a venue.** ``primary_bars=(step, unit)`` aggregates
  prints into bars that fire ``on_bar`` and feed indicators. Orders still
  match against ticks; nothing executes on these bars. A strategy that wants
  bar execution should pre-aggregate and use ``run_portfolio_strategy``.
"""

from __future__ import annotations

import numpy as np

from raptorbt._raptorbt import (
    BarAggregator,
    PyBacktestConfig,
    PyInstrumentConfig,
    PyPortfolioResult,
    PyPortfolioSession,
)
from raptorbt.strategy.base import Strategy
from raptorbt.strategy.context import Bar, BookSnapshot, QuoteTick, TradeTick
from raptorbt.strategy.portfolio_runner import (
    PortfolioContext,
    apply_commands_on,
    drain_intents,
)
from raptorbt.strategy.runner import dispatch_events
from raptorbt.strategy.streams import StreamState

_TICK_FIELDS = ("timestamps", "ltp", "bid", "ask", "buy_qty_delta", "sell_qty_delta")


class TickContext(PortfolioContext):
    """Portfolio context plus the tick-specific views."""

    def __init__(self, session, symbols, data):
        super().__init__(session, symbols, data)
        self._tick: TradeTick | None = None
        self._quote: QuoteTick | None = None
        self._book: dict[str, BookSnapshot] = {}
        self._best_bid: dict[str, float] = {}
        self._best_ask: dict[str, float] = {}
        self._last_price: dict[str, float] = {}

    @property
    def tick(self) -> TradeTick | None:
        """The print being handled, or ``None`` outside ``on_trade_tick``."""
        return self._tick

    @property
    def quote(self) -> QuoteTick | None:
        """The quote being handled, or ``None`` outside ``on_quote``."""
        return self._quote

    @property
    def book(self) -> BookSnapshot | None:
        """Last seen book for the current symbol, or ``None``.

        Unlike ``ctx.quote``/``ctx.tick`` this persists outside its hook, so
        a strategy can read the book while handling a print.
        """
        return self._book.get(self.symbol)

    @property
    def best_bid(self) -> float | None:
        """Last observed bid for the current symbol."""
        return self._best_bid.get(self.symbol)

    @property
    def best_ask(self) -> float | None:
        """Last observed ask for the current symbol."""
        return self._best_ask.get(self.symbol)

    @property
    def last_price(self) -> float | None:
        """Last trade print for the current symbol."""
        return self._last_price.get(self.symbol)


def _as_tick_arrays(arrays: dict) -> dict[str, np.ndarray]:
    if "timestamps" not in arrays or "ltp" not in arrays:
        raise ValueError("tick data needs at least 'timestamps' and 'ltp'")
    out = {"timestamps": np.ascontiguousarray(arrays["timestamps"], dtype=np.int64)}
    n = len(out["timestamps"])
    for key in _TICK_FIELDS[1:]:
        value = arrays.get(key)
        out[key] = (
            np.ascontiguousarray(value, dtype=np.float64)
            if value is not None
            else np.zeros(n, dtype=np.float64)
        )
        if len(out[key]) != n:
            raise ValueError(f"{key} has length {len(out[key])}, expected {n}")
    return out


def setup_tick_strategy(strategy, ctx, symbols, primary_bars):
    """Reset a strategy and build the per-symbol runner state.

    Returns ``(clocks, streams, primary)``: one clock per symbol, the
    stream/indicator routing declared in ``on_start``, and the primary bar
    aggregators (empty when ``primary_bars`` is ``None``).
    """
    strategy.drain_orders()
    strategy.drain_commands()
    strategy._bar_subscriptions = []
    strategy._indicators = []
    from raptorbt.strategy.cache import Cache
    from raptorbt.strategy.clock import Clock

    strategy.clock = Clock()
    strategy.cache = Cache()
    strategy.on_start(ctx)

    # One clock per symbol: a timer set in on_start belongs to every symbol,
    # not to whichever one's event happens to cross the threshold first.
    clocks = {symbol: strategy.clock.clone_schedule() for symbol in symbols}

    streams = StreamState(strategy, symbols)
    # One primary aggregator per symbol, feeding on_bar and the indicators
    # registered without a stream_id. Bars from ticks are a view only.
    primary = (
        {symbol: BarAggregator(*primary_bars) for symbol in symbols}
        if primary_bars is not None
        else {}
    )
    return clocks, streams, primary


def drive_tick_events(strategy, ctx, session, symbols, clocks, streams, primary, apply_commands):
    """Drain every pending schedule event through the strategy's hooks.

    Shared by the batch tick runner and the live stream: both produce the
    same schedule shapes, so one dispatch loop keeps their semantics
    identical. Returns once the session has no pending event.
    """
    while True:
        current = session.current_event()
        if current is None:
            break
        kind, instrument, local_idx, ts, a, b, c, d, e = current
        symbol = symbols[instrument]
        ctx.symbol = symbol
        ctx.idx = local_idx

        # Clock first: scheduled times precede the data revealing them.
        strategy.clock = clocks[symbol]
        for time_event in strategy.clock._advance(ts):
            strategy.on_time_event(ctx, time_event)

        if kind == "bar":
            # Real bars in a tick session: warmup history or a pushed live
            # bar. These execute — same semantics as the portfolio runner.
            ctx._bar = Bar(ts, a, b, c, d, e)
            ctx._last_price[symbol] = d
            streams.push(strategy, ctx, ts, a, b, c, d, e, symbol=symbol)
            strategy.on_bar(ctx)
            apply_commands(instrument, local_idx, ts)
            events = session.apply_current(**drain_intents(strategy, symbol, local_idx))
            dispatch_events(strategy, ctx, events)
            continue

        if kind == "book":
            bids, asks = session.current_depth() or ((), ())
            snapshot = BookSnapshot(ts, tuple(bids), tuple(asks), symbol)
            ctx._book[symbol] = snapshot
            if snapshot.best_bid is not None:
                ctx._best_bid[symbol] = snapshot.best_bid
            if snapshot.best_ask is not None:
                ctx._best_ask[symbol] = snapshot.best_ask
            strategy.on_order_book(ctx, snapshot)
            apply_commands(instrument, local_idx, ts)
            # A book cannot carry an entry: nothing traded at it.
            dispatch_events(strategy, ctx, session.apply_current())
            continue

        if kind == "quote":
            quote = QuoteTick(ts, a, b, symbol)
            ctx._best_bid[symbol] = a
            ctx._best_ask[symbol] = b
            ctx._quote = quote
            strategy.on_quote(ctx, quote)
            ctx._quote = None
            apply_commands(instrument, local_idx, ts)
            # A quote cannot carry an entry: nothing traded at it.
            dispatch_events(strategy, ctx, session.apply_current())
            continue

        tick = TradeTick(ts, a, b, symbol)

        # A bar that completed on this print closed strictly before it, so it
        # dispatches first — the same rule composite bars follow.
        aggregator = primary.get(symbol)
        if aggregator is not None:
            completed = aggregator.push_trade(ts, a, b)
            if completed is not None:
                ctx._bar = Bar(*completed)
                for indicator in streams.primary_indicators(symbol):
                    indicator.update_bar(
                        ctx._bar.open, ctx._bar.high, ctx._bar.low, ctx._bar.close
                    )
                strategy.on_bar(ctx)
                apply_commands(instrument, local_idx, ts)

        streams.push_trade(strategy, ctx, ts, a, b, symbol=symbol)

        ctx._last_price[symbol] = a
        ctx._tick = tick
        strategy.on_trade_tick(ctx, tick)
        ctx._tick = None
        apply_commands(instrument, local_idx, ts)

        events = session.apply_current(**drain_intents(strategy, symbol, local_idx))
        dispatch_events(strategy, ctx, events)


def run_tick_strategy(
    strategy: Strategy | type[Strategy],
    ticks: dict[str, dict],
    config: PyBacktestConfig | None = None,
    primary_bars: tuple[int, str] | None = None,
    depth: dict[str, dict] | None = None,
    directions: dict[str, int] | None = None,
    instruments: dict | None = None,
    instrument_configs: dict[str, PyInstrumentConfig] | None = None,
    oms_type: str = "netting",
    account_type: str = "cash",
    leverage: float = 1.0,
) -> PyPortfolioResult:
    """Run one strategy over N instruments' tick streams.

    ``ticks`` maps symbol -> dict of arrays: ``timestamps`` and ``ltp`` are
    required; ``bid``/``ask``/``buy_qty_delta``/``sell_qty_delta`` are
    optional. A row with ``ltp > 0`` prints a trade; a row with both
    ``bid > 0`` and ``ask > 0`` yields a quote, dispatched *after* that row's
    print.

    ``primary_bars=(step, unit)`` aggregates prints into bars that fire
    ``on_bar`` and feed indicators registered without a ``stream_id``. These
    bars are a view: order matching still happens against ticks. Leave it
    ``None`` for a pure-tick strategy, where ``on_bar`` never fires.

    Everything else — account type, leverage, risk limits, OMS type — behaves
    as in :func:`run_portfolio_strategy`.
    """
    if isinstance(strategy, type):
        strategy = strategy()
    if not isinstance(strategy, Strategy):
        raise ValueError(
            f"strategy must be a Strategy instance or subclass, got {type(strategy).__name__}"
        )
    if not ticks:
        raise ValueError("ticks must contain at least one symbol")

    symbols = list(ticks.keys())
    arrays = {symbol: _as_tick_arrays(ticks[symbol]) for symbol in symbols}

    session = PyPortfolioSession(
        config=config, account_type=account_type, leverage=leverage
    )
    for symbol in symbols:
        session.add_instrument(
            symbol,
            direction=(directions or {}).get(symbol, 1),
            instrument_config=(instrument_configs or {}).get(symbol),
            instrument=(instruments or {}).get(symbol),
            oms_type=oms_type,
        )
    for i, symbol in enumerate(symbols):
        a = arrays[symbol]
        session.set_ticks(
            i,
            a["timestamps"],
            a["ltp"],
            a["bid"],
            a["ask"],
            a["buy_qty_delta"],
            a["sell_qty_delta"],
        )
    for i, symbol in enumerate(symbols):
        levels = (depth or {}).get(symbol)
        if levels is not None:
            session.set_depth(
                i,
                np.ascontiguousarray(levels["timestamps"], dtype=np.int64),
                np.ascontiguousarray(levels["bid_prices"], dtype=np.float64),
                np.ascontiguousarray(levels["bid_sizes"], dtype=np.float64),
                np.ascontiguousarray(levels["ask_prices"], dtype=np.float64),
                np.ascontiguousarray(levels["ask_sizes"], dtype=np.float64),
            )
    session.seal()

    ctx = TickContext(session, symbols, arrays)
    clocks, streams, primary = setup_tick_strategy(strategy, ctx, symbols, primary_bars)

    id_map: dict[str, tuple[int, int]] = {}
    apply_commands = apply_commands_on(strategy, session, ctx, symbols, id_map)

    drive_tick_events(
        strategy, ctx, session, symbols, clocks, streams, primary, apply_commands
    )

    strategy.on_stop(ctx)
    return session.finish()
