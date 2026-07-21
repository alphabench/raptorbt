"""Multi-instrument driver for class-based strategies.

One strategy instance trades N instruments against a single shared account.
Bars from all instruments are merged into one deterministic schedule
(by timestamp, then registration order); ``on_bar`` fires once per event
with ``ctx.symbol`` naming the instrument whose bar just closed. Orders and
closes route to the current symbol by default, or explicitly via
``symbol=``.

With ``account_type="margin"`` the instruments also share one pool of locked
initial margin, so leverage applies portfolio-wide and a margin call halts
every instrument at once.

Indicators and composite-bar subscriptions are per symbol: one
``subscribe_bars`` declaration yields one aggregated stream per symbol, and
``register_indicator(..., symbol=...)`` routes an indicator to one of them.
A symbol's composite bar dispatches before the ``on_bar`` of that symbol's
primary bar that completed it; across symbols, order follows the merged
schedule.
"""

from __future__ import annotations

import warnings

import numpy as np

from raptorbt._raptorbt import (
    PyBacktestConfig,
    PyInstrumentConfig,
    PyPortfolioResult,
    PyPortfolioSession,
)
from raptorbt.strategy.base import Strategy
from raptorbt.strategy.context import Bar
from raptorbt.strategy.orders import ClosePosition, MarketOrder
from raptorbt.strategy.runner import dispatch_events
from raptorbt.strategy.streams import StreamState


class PortfolioContext:
    """Read/query surface handed to strategy hooks in portfolio runs."""

    def __init__(self, session: PyPortfolioSession, symbols: list[str], data: dict):
        self._session = session
        self._symbols = symbols
        self._index_of = {s: i for i, s in enumerate(symbols)}
        self._data = data
        # Current event state, maintained by the runner.
        self.symbol: str = symbols[0]
        self.idx: int = 0  # local bar index within the current symbol
        self._bar: Bar | None = None

    # -- current event -------------------------------------------------------

    @property
    def bar(self) -> Bar:
        """The bar being processed (belongs to ``ctx.symbol``)."""
        return self._bar

    @property
    def timestamp(self) -> int:
        return self._bar.timestamp

    # -- data access ---------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    def series(self, symbol: str | None = None) -> dict[str, np.ndarray]:
        """Full OHLCV arrays of a symbol (default: the current one)."""
        return self._data[symbol or self.symbol]

    # -- portfolio state -----------------------------------------------------

    def position(self, symbol: str | None = None):
        """Earliest open position of a symbol, or ``None``."""
        return self._session.position(self._index_of[symbol or self.symbol])

    def positions(self, symbol: str | None = None):
        """All open positions of a symbol, in opening order."""
        return self._session.positions(self._index_of[symbol or self.symbol])

    def net_position(self, symbol: str | None = None) -> float:
        """Signed unit total across a symbol's open positions."""
        return sum(p.size * p.direction for p in self.positions(symbol))

    def is_net_long(self, symbol: str | None = None) -> bool:
        return self.net_position(symbol) > 0.0

    def is_net_short(self, symbol: str | None = None) -> bool:
        return self.net_position(symbol) < 0.0

    def is_flat(self, symbol: str | None = None) -> bool:
        return not self.positions(symbol)

    @property
    def equity(self) -> float:
        """Portfolio equity: shared cash plus all instruments' marks."""
        return self._session.equity()

    @property
    def cash(self) -> float:
        """Uncommitted shared cash."""
        return self._session.cash()

    def _instrument_index(self, symbol: str | None) -> int:
        return self._index_of[symbol or self.symbol]


def _as_arrays(arrays: dict) -> dict[str, np.ndarray]:
    out = {
        "timestamps": np.ascontiguousarray(arrays["timestamps"], dtype=np.int64),
    }
    for key in ("open", "high", "low", "close", "volume"):
        out[key] = np.ascontiguousarray(arrays[key], dtype=np.float64)
    n = len(out["timestamps"])
    for key, arr in out.items():
        if len(arr) != n:
            raise ValueError(f"{key} has length {len(arr)}, expected {n}")
    return out


def run_portfolio_strategy(
    strategy: Strategy | type[Strategy],
    data: dict[str, dict],
    config: PyBacktestConfig | None = None,
    directions: dict[str, int] | None = None,
    instruments: dict | None = None,
    instrument_configs: dict[str, PyInstrumentConfig] | None = None,
    oms_type: str = "netting",
    account_type: str = "cash",
    leverage: float = 1.0,
) -> PyPortfolioResult:
    """Run one strategy over N instruments sharing a capital pool.

    ``data`` maps symbol -> dict of OHLCV arrays (``timestamps``/``open``/
    ``high``/``low``/``close``/``volume``). ``instruments`` optionally maps
    symbol -> :class:`InstrumentSpec`. Returns a ``PyPortfolioResult``:
    portfolio-level curves/metrics plus per-instrument summaries.

    ``account_type`` is ``"cash"`` (fully funded, the default) or
    ``"margin"``, which locks initial margin per position instead of the
    full notional and marks equity with direction-aware unrealized PnL.
    Under margin the account is shared: ``leverage`` applies across all
    instruments, and a margin call halts every one of them, surfacing as
    ``on_margin_call`` plus ``halted``/``halted_at`` on the result.

    Risk limits on ``config`` are portfolio-wide: ``max_positions`` counts
    open positions across all symbols, and ``max_drawdown_pct`` trips on
    portfolio equity and halts entries on every symbol. Capital allocation
    is the strategy's own: each entry is offered the full free balance, so
    size it with ``size_frac``.
    """
    if isinstance(strategy, type):
        strategy = strategy()
    if not isinstance(strategy, Strategy):
        raise ValueError(
            f"strategy must be a Strategy instance or subclass, got {type(strategy).__name__}"
        )
    if not data:
        raise ValueError("data must contain at least one symbol")

    symbols = list(data.keys())
    arrays = {symbol: _as_arrays(data[symbol]) for symbol in symbols}

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
        session.set_bars(
            i, a["timestamps"], a["open"], a["high"], a["low"], a["close"], a["volume"]
        )
    session.seal()

    ctx = PortfolioContext(session, symbols, arrays)
    strategy.drain_orders()
    strategy.drain_commands()
    strategy._bar_subscriptions = []
    strategy._indicators = []
    from raptorbt.strategy.cache import Cache
    from raptorbt.strategy.clock import Clock

    strategy.clock = Clock()
    strategy.cache = Cache()
    strategy.on_start(ctx)

    # Subscriptions and indicators are declared in on_start, so the per-symbol
    # aggregators and indicator routing are built once it returns.
    streams = StreamState(strategy, symbols)
    if streams.has_unrouted_indicators:
        warnings.warn(
            "register_indicator() without symbol= in a portfolio run feeds the "
            "indicator every symbol's bars interleaved; pass symbol= to route it",
            stacklevel=2,
        )

    # client order id -> (instrument index, engine order id).
    id_map: dict[str, tuple[int, int]] = {}

    def apply_commands(current_instrument: int, local_idx: int, ts: int) -> None:
        for command in strategy.drain_commands():
            if command[0] == "submit":
                _, client_id, order, parent, symbol = command
                instrument = (
                    ctx._instrument_index(symbol) if symbol else current_instrument
                )
                parent_engine_id = None
                if parent:
                    mapped = id_map.get(parent)
                    if mapped is None:
                        raise ValueError(f"unknown parent order {parent!r}")
                    if mapped[0] != instrument:
                        raise ValueError("parent order belongs to a different symbol")
                    parent_engine_id = mapped[1]
                engine_id = session.submit_order(
                    instrument,
                    side=order.side,
                    kind=order.kind,
                    submitted_idx=local_idx,
                    submitted_ts=ts,
                    client_id=client_id,
                    units=order.units,
                    size_frac=order.size_frac,
                    limit_price=getattr(order, "price", None),
                    trigger_price=getattr(order, "trigger", None),
                    tif=order.tif,
                    expire_ns=order.expire_ns,
                    stop_price=order.stop_price,
                    target_price=order.target_price,
                    offset=getattr(order, "offset", None),
                    offset_kind=getattr(order, "offset_kind", "price"),
                    limit_offset=getattr(order, "limit_offset", 0.0),
                    post_only=getattr(order, "post_only", False),
                    reduce_only=order.reduce_only,
                    parent_id=parent_engine_id,
                )
                id_map[client_id] = (instrument, engine_id)
            elif command[0] == "cancel":
                mapped = id_map.get(command[1])
                if mapped is not None:
                    session.cancel_order(mapped[0], local_idx, mapped[1])
            elif command[0] == "cancel_all":
                for i in range(len(symbols)):
                    session.cancel_all_orders(i, local_idx)
            elif command[0] == "link_oco":
                mapped = [id_map[c] for c in command[1] if c in id_map]
                if len(mapped) >= 2:
                    instruments_involved = {m[0] for m in mapped}
                    if len(instruments_involved) != 1:
                        raise ValueError("one-cancels-other links cannot span symbols")
                    session.link_oco(mapped[0][0], [m[1] for m in mapped])
            elif command[0] == "close":
                _, position_id, symbol = command
                session.request_close(
                    ctx._instrument_index(symbol) if symbol else current_instrument,
                    position_id,
                )
            elif command[0] == "close_all_for":
                instrument = ctx._instrument_index(command[1])
                for snapshot in session.positions(instrument):
                    session.request_close(instrument, snapshot.position_id)
            elif command[0] == "modify":
                # The id map carries the owning instrument, so a modify
                # routes without the caller naming a symbol.
                mapped = id_map.get(command[1])
                if mapped is not None:
                    session.modify_order(mapped[0], mapped[1], **command[2])

    while True:
        current = session.current()
        if current is None:
            break
        instrument, local_idx, ts, o, h, l, c, v = current
        ctx.symbol = symbols[instrument]
        ctx.idx = local_idx
        ctx._bar = Bar(ts, o, h, l, c, v)

        for time_event in strategy.clock._advance(ts):
            strategy.on_time_event(ctx, time_event)

        # Composite bars and indicators for THIS symbol only, before on_bar
        # sees the bar. Orders queued from on_composite_bar are drained by
        # this bar's apply_commands and route to the completing symbol.
        streams.push(strategy, ctx, ts, o, h, l, c, v, symbol=ctx.symbol)

        strategy.on_bar(ctx)
        apply_commands(instrument, local_idx, ts)

        entry = False
        exit_ = False
        size_mult = None
        stop_override = None
        target_override = None
        for intent in strategy.drain_orders():
            if isinstance(intent, MarketOrder):
                if entry:
                    raise ValueError(f"duplicate entry intents on {ctx.symbol} bar {local_idx}")
                entry = True
                size_mult = intent.size_frac
                stop_override = intent.stop_price
                target_override = intent.target_price
            elif isinstance(intent, ClosePosition):
                if exit_:
                    raise ValueError(f"duplicate close intents on {ctx.symbol} bar {local_idx}")
                exit_ = True
            else:
                raise ValueError(f"unknown order intent: {intent!r}")

        events = session.apply_current(
            entry=entry,
            exit=exit_,
            atr=0.0,
            size_mult=size_mult,
            stop_price=stop_override,
            target_price=target_override,
        )
        dispatch_events(strategy, ctx, events)

    strategy.on_stop(ctx)
    return session.finish()
