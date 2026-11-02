"""A long-running node that hosts many strategies on one pushed feed.

:class:`~raptorbt.strategy.tick_stream.TickStrategyStream` drives one strategy
from events pushed one at a time. :class:`LiveNode` does the same for many
strategies at once: push a tick, bar or depth snapshot, and every strategy that
trades that symbol, in the order the strategies were given, sees exactly what
its own ``TickStrategyStream`` would show it. Each strategy has its own
account; nothing is shared between them but the market data, unless strategies
are put on a shared account with ``accounts=`` (see below).

The node runs on the same Rust core as :func:`~raptorbt.run_strategies`. What
differs is the storage. A batch run reads an immutable copy of the whole
dataset. A node holds only the most recent rows of each symbol in fixed-size
rings, so its memory does not grow with the length of the feed (see
``history`` below, and ``curve_samples`` for the one other thing that grows).

Typical shape::

    node = LiveNode(
        [MeanReversion(), Breakout(), (Hedger(), ["NSE:TCS", "NSE:INFY"])],
        ["NSE:TCS", "NSE:INFY", "NSE:SBIN"],
        warmup_bars={"NSE:TCS": bars},
        bar_stamp="close",
        primary_bars=(1, "m"),
    )
    for tick in feed:                       # days-long, open-ended
        node.push_tick(tick.symbol, tick.ts_ns, tick.ltp, tick.bid, tick.ask)
    results = node.finish()                 # once, when the session ends
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np

from raptorbt._raptorbt import (
    AccountResult,
    BacktestConfig,
    InstrumentConfig,
    MarketData,
    PortfolioResult,
    Runtime,
)
from raptorbt.errors import (
    InvalidData,
    InvalidParameter,
    StateError,
    UnknownSymbol,
)
from raptorbt.strategy.accounts import (
    AccountSpec,
    check_accounts,
    check_group_sizes,
    unpack_entries,
)
from raptorbt.strategy._commands import dispatch_events, reject_atr_stops, session_tz_offset_ns
from raptorbt.strategy._stamps import BarStamps
from raptorbt.strategy.base import Strategy
from raptorbt.strategy.context import (
    Bar,
    BookSnapshot,
    CompositeBar,
    QuoteTick,
    TradeTick,
    WorkingOrder,
)
from raptorbt.strategy.multi_runner import _add_strategy, _normalise, _primary_bars

#: Rows kept per symbol and kind (bars, tick rows, depth snapshots) when
#: ``history`` is not given. About 1 MB per active symbol for bars and ticks,
#: another 2 MB if it also gets depth.
DEFAULT_HISTORY = 10_000

_BAR_FIELDS = ("open", "high", "low", "close", "volume")


class LiveNode:
    """Many strategies over the symbols of one pushed feed.

    ``strategies`` holds, per strategy, a :class:`Strategy` instance, a
    ``Strategy`` subclass (instantiated without arguments), a
    ``(strategy, symbols)`` pair or either with an account, as in
    :func:`~raptorbt.run_strategies`. A
    strategy without symbols trades every symbol of the node. A strategy sees
    its symbols in the node's ``symbols`` order, whatever order it names them
    in. Strategies are numbered by their position in ``strategies``.

    ``warmup_bars`` maps symbol -> dict of OHLCV arrays (``timestamps``,
    ``open``, ``high``, ``low``, ``close``, ``volume``) replayed through every
    strategy that trades the symbol during construction. Warmup bars execute:
    orders match and equity is marked, exactly as for a pushed bar.

    ``initial_positions`` seeds positions the account already holds, per
    strategy: a dict keyed by the strategy's index in ``strategies`` or by the
    strategy object itself, whose value maps symbol -> ``{"quantity",
    "avg_price", "timestamp_ns"?}``. Each seed is adopted, with no order, fill
    or fee, before that strategy's ``on_start``, so ``on_start`` sees it. Seeds
    are long-only, need a positive quantity and price, and are refused unless
    the account is cash or fully funded margin (``leverage=1.0``).

    ``bar_stamp`` / ``bar_ns`` say what the timestamps of warmup and pushed
    bars mean, as in :class:`~raptorbt.TickStrategyStream`. Ticks are instants.

    ``history`` is the number of rows each symbol's rings hold, for bars, tick
    rows and depth snapshots separately (default :data:`DEFAULT_HISTORY`). It
    is raised to what the strategies' indicators read (their longest period).
    ``ctx.series(symbol)`` returns read-only arrays of the rows still held, up
    to the latest the node has reached: a symbol's bars once it has had any,
    otherwise its ticks (``timestamps``, ``ltp``, ``bid``, ``ask``,
    ``buy_qty_delta``, ``sell_qty_delta``, ``ltq``, ``bid_qty``, ``ask_qty``,
    ``oi``). Older rows are overwritten. A strategy that needs more than the
    default can call ``ctx.keep_history(rows, symbol=None)``; rows it asked
    for are never overwritten, and a push that would have to overwrite one
    raises ``RuntimeError`` and changes nothing. That is the only way a node
    refuses data, and it never drops any silently.

    ``curve_samples`` bounds the equity, drawdown and returns curves each
    strategy accumulates (one sample per distinct timestamp), which otherwise
    grow with the feed. With it, a strategy keeps between ``curve_samples`` and
    twice that many of the latest samples, and the curves and metrics
    ``finish`` returns cover that window only; trades and orders are complete.
    Leave it ``None`` for the exact curves of the whole session, and accept
    about 32 bytes per sample per strategy.

    ``accounts`` maps a name to an :class:`~raptorbt.AccountSpec`; strategies
    that name it, as ``(strategy, symbols, {"account": name})`` or
    ``{"strategy": s, "symbols": [...], "account": name}``, share that
    account's cash, margin and account-level limits, and step in the order they
    are listed. The rules, including how a strategy's equity is attributed, are
    those of :mod:`raptorbt.strategy.accounts`. With ``accounts``,
    :meth:`finish` returns ``(results, account_results)`` and
    :meth:`halt_account` halts a whole account. ``initial_positions`` is
    adopted from the account's cash, in strategy order. ``curve_samples`` also
    bounds an account's curves.

    Everything else (account type, leverage, risk limits, OMS type) behaves as
    in :class:`~raptorbt.TickStrategyStream`. ATR-based stops are refused, as
    there.
    """

    def __init__(
        self,
        strategies: Iterable,
        symbols: list[str],
        config: BacktestConfig | None = None,
        *,
        primary_bars: tuple[int, str] | None = None,
        warmup_bars: dict[str, dict] | None = None,
        directions: dict[str, int] | None = None,
        instruments: dict | None = None,
        instrument_configs: dict[str, InstrumentConfig] | None = None,
        oms_type: str = "netting",
        account_type: str = "cash",
        leverage: float = 1.0,
        accounts: dict[str, AccountSpec] | None = None,
        initial_positions: dict | None = None,
        bar_stamp: Literal["open", "close"] | None = None,
        bar_ns: int | dict[str, int] | None = None,
        history: int | None = None,
        curve_samples: int | None = None,
        _via: str | None = None,
    ):
        # ``_via`` names the wrapper (``TickStrategyStream``) that builds this
        # node: its name leads every warning and error, its extra frame is
        # skipped when attributing them, and it calls itself a stream.
        name = _via or "LiveNode"
        extra = 1 if _via else 0
        self._name = name
        self._extra = extra
        if isinstance(symbols, str) or not symbols:
            raise InvalidParameter(
                "symbols must name at least one instrument",
                code="no_symbols",
                operation="LiveNode",
                parameter="symbols",
                value=symbols,
            )
        symbols = list(symbols)
        if len(set(symbols)) != len(symbols):
            raise InvalidParameter(
                "symbols must be unique",
                code="duplicate_symbols",
                operation="LiveNode",
                parameter="symbols",
                value=symbols,
            )
        history = DEFAULT_HISTORY if history is None else history
        if isinstance(history, bool) or not isinstance(history, int) or history < 1:
            raise InvalidParameter(
                f"history must be a positive number of rows, got {history!r}",
                code="invalid_history",
                operation="LiveNode",
                parameter="history",
                value=history,
                valid=["a positive integer"],
            )
        if curve_samples is not None and (
            isinstance(curve_samples, bool) or not isinstance(curve_samples, int) or curve_samples < 1
        ):
            raise InvalidParameter(
                f"curve_samples must be a positive integer or None, got {curve_samples!r}",
                code="invalid_curve_samples",
                operation="LiveNode",
                parameter="curve_samples",
                value=curve_samples,
                valid=["a positive integer", None],
            )
        primary_bars = _primary_bars(primary_bars)
        accounts = check_accounts(accounts, name)
        strategies, names = unpack_entries(strategies, accounts, name)
        check_group_sizes(names, name)
        entries = _normalise(strategies, dict.fromkeys(symbols), name)
        seeds = _seeds(initial_positions, entries)
        reject_atr_stops(config, instrument_configs, symbols)

        self._symbols = symbols
        self._index_of = {s: i for i, s in enumerate(symbols)}
        self._finished = False
        self._stamps = BarStamps.resolve(
            bar_stamp, bar_ns, symbols, name, warn=bool(warmup_bars), stacklevel=4 + extra
        )
        warmup = _warmup(warmup_bars, self._index_of, self._stamps)

        market = MarketData.live(symbols, [self._stamps.offset(s) for s in symbols], history)
        rt = Runtime(
            market,
            [(directions or {}).get(s, 1) for s in symbols],
            [(instrument_configs or {}).get(s) for s in symbols],
            [(instruments or {}).get(s) for s in symbols],
            config,
            oms_type,
            account_type,
            leverage,
            session_tz_offset_ns(config),
            self._stamps.aggregator_stamp,
            True,
            True,
            Bar,
            CompositeBar,
            WorkingOrder,
            dispatch_events,
            TradeTick,
            QuoteTick,
            BookSnapshot,
        )
        self._rt = rt
        try:
            if curve_samples is not None:
                rt.set_curve_window(curve_samples)
            if primary_bars is not None:
                rt.set_primary_bars(*primary_bars)
            self._strategies: list[Strategy] = []
            self._sids: list[int] = []
            self._ctxs = []
            self._views = []
            self._symbols_of: list[list[str]] = []
            self._accounts = accounts
            self._gids: dict[str, int] = {}
            for account in dict.fromkeys(n for n in names if n is not None):
                spec = accounts[account]
                self._gids[account] = rt.add_account(
                    spec.initial_capital,
                    spec.account_type,
                    spec.leverage,
                    sum(1 for n in names if n == account),
                    max_drawdown_pct=spec.max_drawdown_pct,
                    max_daily_loss=spec.max_daily_loss,
                    max_daily_loss_pct=spec.max_daily_loss_pct,
                    max_gross_exposure=spec.max_gross_exposure,
                    max_positions=spec.max_positions,
                )
            for k, (strategy, syms) in enumerate(entries):
                sid, ctx, view = _add_strategy(
                    rt,
                    strategy,
                    syms,
                    adopt=_adopter(rt, syms, seeds[k]),
                    stacklevel=3 + extra,
                    account=self._gids.get(names[k]),
                )
                self._strategies.append(strategy)
                self._sids.append(sid)
                self._ctxs.append(ctx)
                self._views.append(view)
                self._symbols_of.append(syms)
            rt.finalize()
            # Prime indicators and strategy state from the warmup history now,
            # so the node is warm before the first live event.
            for push in warmup:
                rt.push_bar(*push)
        except BaseException:
            rt.close()
            raise

    # -- feeding -------------------------------------------------------------

    def push_tick(
        self,
        symbol: str,
        timestamp: int,
        ltp: float,
        bid: float = 0.0,
        ask: float = 0.0,
        buy_qty_delta: float = 0.0,
        sell_qty_delta: float = 0.0,
        ltq: float = 0.0,
        bid_qty: float = 0.0,
        ask_qty: float = 0.0,
        oi: float = 0.0,
    ) -> int:
        """Feed one tick row; every hook it triggers fires before this returns.

        ``ltp > 0`` prints a trade; a two-sided book yields a quote after it.
        ``ltq`` is the print's own size when the feed has it (else the flow
        deltas stand in); ``bid_qty``/``ask_qty`` are the displayed L1 sizes;
        ``oi`` the open interest. Returns how many events were appended (0-2).

        A row that yields no event still carries a timestamp: alerts and timers
        due by it fire before this returns, for each strategy that trades
        ``symbol``. Orders they submit are routed to their instrument at once
        and match from its next print; intents they queue are held for this
        symbol's next print.

        Timestamps must not decrease per symbol (equal ones are allowed);
        a decreasing one raises ``ValueError`` and changes nothing.
        """
        self._check_open()
        return self._rt.push_tick(
            self._index_of[symbol],
            timestamp,
            ltp,
            bid,
            ask,
            buy_qty_delta,
            sell_qty_delta,
            ltq,
            bid_qty,
            ask_qty,
            oi,
        )

    def push_bar(
        self,
        symbol: str,
        timestamp: int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        """Feed one closed bar. It executes: orders match, equity marks.

        ``timestamp`` is stamped as the node's ``bar_stamp`` declares.
        """
        self._check_open()
        self._stamps.warn_missing(f"{self._name}.push_bar", stacklevel=3 + self._extra)
        self._rt.push_bar(self._index_of[symbol], timestamp, open, high, low, close, volume)

    def push_depth(
        self,
        symbol: str,
        timestamp: int,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
    ) -> None:
        """Feed one depth snapshot: ``(price, size)`` lists, best first."""
        self._check_open()
        self._rt.push_depth(self._index_of[symbol], timestamp, bids, asks)

    def advance_time(self, timestamp: int) -> None:
        """Move time to ``timestamp`` with no data.

        Fires the alerts and timers that are due for every strategy, on each
        of its symbols' clocks, in strategy order. Orders they submit are
        routed at once and match from their instrument's next print.
        """
        self._check_open()
        self._rt.advance_time(timestamp)

    # -- state ---------------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def strategies(self) -> list[Strategy]:
        """The strategies, in the order given."""
        return list(self._strategies)

    def ctx(self, strategy_index: int):
        """The context handed to strategy ``strategy_index``'s hooks."""
        return self._ctxs[strategy_index]

    def equity(self, strategy_index: int) -> float:
        self._check_open()
        return self._views[strategy_index].equity()

    def is_halted(self, strategy_index: int) -> bool:
        """Whether a margin call or drawdown kill-switch has latched."""
        self._check_open()
        return self._views[strategy_index].is_halted()

    def halt(
        self, strategy_index: int, flatten: bool = False, reason: str = "manual"
    ) -> None:
        """Halt one strategy from outside it, like ``ctx.halt``: its new
        entries are refused with ``killed``, its working orders are cancelled
        and, with ``flatten=True``, its positions are closed at the last
        prices seen. ``is_halted`` reads True at once; the cancels and closes
        happen when that strategy's next event is applied."""
        self._check_open()
        self._views[strategy_index].halt(reason, flatten)

    def halt_account(
        self, account: str, flatten: bool = False, reason: str = "manual"
    ) -> None:
        """Halt every strategy on ``account`` from outside, as ``halt`` halts
        one: their new entries are refused with ``killed``, their working
        orders are cancelled and, with ``flatten=True``, their positions are
        closed at the last prices seen. Each strategy's cancels and closes
        happen when its next event is applied. The account's ``halted_at`` is
        the number of events applied to its strategies so far. Strategies not
        on the account are untouched."""
        self._check_open()
        if account not in self._gids:
            raise InvalidParameter(
                f"{self._name}: no account named {account!r} with strategies on it",
                code="unknown_account",
                operation=f"{self._name}.halt_account",
                parameter="account",
                value=account,
                valid=list(self._gids),
            )
        self._rt.halt_account(self._gids[account], reason, flatten)

    def positions(self, strategy_index: int, symbol: str | None = None):
        """Open positions of one strategy: a symbol's list, or, without a
        symbol, ``{symbol: list}`` for every symbol the strategy trades."""
        self._check_open()
        ctx = self._ctxs[strategy_index]
        if symbol is not None:
            return ctx.positions(symbol)
        return {s: ctx.positions(s) for s in self._symbols_of[strategy_index]}

    def memory_report(self) -> dict[str, int]:
        """Heap bytes held by the Rust side, by category; ``live_rings`` is the
        fixed size of the history rings. Python objects are not counted."""
        self._check_open()
        return self._rt.memory_report()

    def finish(
        self,
    ) -> list[PortfolioResult] | tuple[list[PortfolioResult], dict[str, AccountResult]]:
        """Close out every strategy and compute metrics, in strategy order.

        Calls each strategy's ``on_stop`` first. With ``accounts`` the return
        value is ``(results, account_results)``, ``account_results`` mapping
        each account that has strategies to its :class:`~raptorbt.AccountResult`.
        The node is unusable afterwards: any further call raises ``RuntimeError``.
        """
        self._check_open()
        self._finished = True
        try:
            results = []
            for strategy, sid, ctx in zip(self._strategies, self._sids, self._ctxs):
                strategy.on_stop(ctx)
                results.append(self._rt.finish(sid))
            if self._accounts is None:
                return results
            owned = {}
            for name, gid in self._gids.items():
                account = self._rt.finish_account(gid, name)
                # Strategies are numbered by their position in the node.
                account.strategies = [self._sids.index(sid) for sid in account.strategies]
                owned[name] = account
            return results, owned
        finally:
            self._rt.close()

    # -- internals -----------------------------------------------------------

    def _check_open(self) -> None:
        if self._finished:
            what = "stream" if self._extra else "node"
            raise StateError(
                f"{what} is finished; create a new one",
                code="node_finished",
                operation="TickStrategyStream" if self._extra else "LiveNode",
            )


def _warmup(warmup_bars, index_of, stamps):
    """Warmup bars as pushes in feed order: ``(index, ts, o, h, l, c, v)``.

    Bars of different symbols merge by close time, ties by symbol order, as a
    session merges its bar streams.
    """
    rows = []
    for symbol, bars in (warmup_bars or {}).items():
        if symbol not in index_of:
            raise UnknownSymbol(
                f"warmup_bars names unknown symbol {symbol!r}",
                code="unknown_symbol",
                operation="LiveNode",
                parameter="warmup_bars",
                value=symbol,
                valid=list(index_of),
            )
        i = index_of[symbol]
        ts = np.ascontiguousarray(bars["timestamps"], dtype=np.int64)
        cols = [np.ascontiguousarray(bars[k], dtype=np.float64) for k in _BAR_FIELDS]
        for col in cols:
            if len(col) != len(ts):
                raise InvalidData(
                    f"warmup_bars[{symbol!r}]: columns differ in length",
                    code="length_mismatch",
                    operation="LiveNode",
                    parameter=f"warmup_bars[{symbol!r}]",
                    value=len(col),
                    valid=[len(ts)],
                )
        closes = stamps.visible(symbol, ts)
        rows.extend(
            (int(closes[j]), i, j, (i, int(ts[j]), *(float(c[j]) for c in cols)))
            for j in range(len(ts))
        )
    rows.sort(key=lambda r: r[:3])
    return [r[3] for r in rows]


def _seeds(initial_positions, entries):
    """Per strategy, ``{symbol: (quantity, avg_price, timestamp_ns)}``."""
    out = [dict() for _ in entries]
    if not initial_positions:
        return out
    by_object = {id(strategy): k for k, (strategy, _) in enumerate(entries)}
    claimed: dict[int, object] = {}
    for key, per_symbol in initial_positions.items():
        if isinstance(key, bool):
            raise InvalidParameter(
                f"initial_positions key {key!r} is not a strategy index or strategy",
                code="invalid_initial_positions",
                operation="LiveNode",
                parameter="initial_positions",
                value=key,
                valid=["a strategy index", "a strategy of the node"],
            )
        if isinstance(key, int):
            if not 0 <= key < len(entries):
                raise InvalidParameter(
                    f"initial_positions names strategy index {key}, but there are "
                    f"{len(entries)} strategies",
                    code="invalid_initial_positions",
                    operation="LiveNode",
                    parameter="initial_positions",
                    value=key,
                    valid=list(range(len(entries))),
                )
            k = key
        elif id(key) in by_object:
            k = by_object[id(key)]
        else:
            raise InvalidParameter(
                f"initial_positions key {key!r} is neither a strategy index nor one of "
                "the node's strategies",
                code="invalid_initial_positions",
                operation="LiveNode",
                parameter="initial_positions",
                value=key,
                valid=["a strategy index", "a strategy of the node"],
            )
        if k in claimed:
            raise InvalidParameter(
                f"initial_positions names strategy {k} twice",
                code="invalid_initial_positions",
                operation="LiveNode",
                parameter="initial_positions",
                value=k,
            )
        claimed[k] = key
        syms = entries[k][1]
        for symbol, seed in per_symbol.items():
            if symbol not in syms:
                raise UnknownSymbol(
                    f"initial_positions for strategy {k} names unknown symbol {symbol!r}",
                    code="unknown_symbol",
                    operation="LiveNode",
                    parameter="initial_positions",
                    value=symbol,
                    valid=list(syms),
                )
            quantity = float(seed.get("quantity") or 0)
            avg_price = float(seed.get("avg_price") or 0)
            if quantity <= 0 or avg_price <= 0:
                raise InvalidParameter(
                    f"initial_positions[{k}][{symbol!r}] needs positive quantity "
                    f"and avg_price (got {quantity} @ {avg_price})",
                    code="invalid_initial_positions",
                    operation="LiveNode",
                    parameter=f"initial_positions[{k}][{symbol!r}]",
                    value=f"{quantity} @ {avg_price}",
                )
            out[k][symbol] = (quantity, avg_price, int(seed.get("timestamp_ns") or 0))
    return out


def _adopter(rt, syms, seeds):
    """The callback that adopts a strategy's seeds once its account exists."""
    if not seeds:
        return None

    def adopt(sid):
        for symbol, (quantity, avg_price, ts) in seeds.items():
            rt.adopt_position(sid, syms.index(symbol), ts, avg_price, quantity)

    return adopt
