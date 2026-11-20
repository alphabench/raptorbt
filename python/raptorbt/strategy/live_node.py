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

import logging
import weakref
from math import isfinite as _isfinite
from typing import Iterable, Literal, Sequence

import numpy as np

from raptorbt._raptorbt import (
    AccountResult,
    ExecutionReport,
    BacktestConfig,
    CorporateAction,
    InstrumentConfig,
    MarketData,
    PortfolioResult,
    Runtime,
)
from raptorbt.execution import ExecutionAdapter, PaperExecution, SimulatedExecution
from raptorbt import errors
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
from raptorbt.strategy._validate import SessionCoverage, ensure_prices_finite, ensure_pushed_finite
from raptorbt.strategy._commands import (
    config_for_holding,
    dispatch_events,
    reject_atr_stops,
    reject_bar_only_settings,
    session_tz_offset_ns,
)
from raptorbt.strategy._stamps import BarStamps
from raptorbt.strategy import api, context_host
from raptorbt.strategy import order_check as order_check_mod
from raptorbt.strategy.base import Strategy
from raptorbt.strategy.context import (
    Bar,
    BookSnapshot,
    CompositeBar,
    QuoteTick,
    TradeTick,
    WorkingOrder,
)
from raptorbt.strategy.multi_runner import _add_strategy, _lifecycle, _normalise, _primary_bars

#: Rows kept per symbol and kind (bars, tick rows, depth snapshots) when
#: ``history`` is not given. About 1 MB per active symbol for bars and ticks,
#: another 2 MB if it also gets depth.
DEFAULT_HISTORY = 10_000

_BAR_FIELDS = ("open", "high", "low", "close", "volume")


_log = logging.getLogger(__name__)

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
    "avg_price", "timestamp_ns"?, "margin_minor"?, "cost_minor"?}``. Each seed
    is adopted, with no order, fill, fee, trade record or entry event, before
    that strategy's ``on_start``, so ``on_start`` sees it. ``quantity`` is
    signed: negative is a short, which needs a margin account (a cash account
    cannot hold one). ``avg_price`` is the position's entry price, so a short
    gains as the price falls from it. The seed's money is the account's own
    fact, stated in minor units of its money scale (``money_decimals``,
    paise for two decimals) as an ``int``:

    * margin account: ``margin_minor`` is the margin the account has posted
      for this position. The position is adopted with exactly that locked,
      whatever the leverage, margin model or instrument (options, shorts
      included); the balance is not debited, as for any position on margin.
      Without it only a fully funded account (``leverage=1.0``, or a model
      that demands the whole cost basis) is adopted, locking the notional, and
      anything else is refused: the margin a broker holds against a position
      cannot be derived from quantity and price. An option leg margined by a
      built-in SPAN-style deposit is re-priced with its group after the first
      event applied, like any leg; the stated margin holds until then.
    * cash account: ``cost_minor`` is the cash the position cost, debited in
      place of ``avg_price * quantity`` (exact where the broker's figure
      differs by rounding). Longs only.

    A seed needs a non-zero quantity and a positive price. Positions are
    adopted before any event is applied (the equity curve starts with them).

    ``bar_stamp`` / ``bar_ns`` say what the timestamps of warmup and pushed
    bars mean, as in :class:`~raptorbt.TickStrategyStream`. Ticks are instants.
    ``bar_stamp`` is required for bars: warmup bars, ``push_bar``, ``push_group``
    and ``warm_up`` raise ``InvalidParameter`` (``missing_bar_stamp``) without it.

    ``history`` is the number of rows each symbol's rings hold, for bars, tick
    rows and depth snapshots separately (default :data:`DEFAULT_HISTORY`). It
    is raised to what the strategies' indicators read (their longest period).
    ``ctx.series(symbol)`` returns read-only arrays of the rows still held, up
    to the latest the node has reached: a symbol's bars once it has had any,
    otherwise its ticks (``timestamps``, ``ltp``, ``bid``, ``ask``,
    ``buy_qty_delta``, ``sell_qty_delta``, ``ltq``, ``bid_qty``, ``ask_qty``,
    ``oi``). The bars the node builds from a symbol's prints (``primary_bars``)
    once it has had a bar join its bars, in the order ``on_bar`` hears them and
    stamped as a pushed bar of the same close would be, as in
    :func:`~raptorbt.run_strategies`; the series then shows at most ``history``
    rows. Older rows are overwritten. A strategy that needs more than the
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

    ``execution`` says where orders go. ``None`` (the default) or
    :class:`~raptorbt.execution.SimulatedExecution` matches them in-process,
    as a backtest does. :class:`~raptorbt.execution.PaperExecution` sends them
    to a simulated exchange that answers with execution reports, by the route
    a real venue's reports take; fills are those a backtest of the same
    stream would make. Any object with ``submit`` / ``cancel`` / ``modify``
    (:class:`~raptorbt.execution.ExecutionAdapter`) is a venue of your own: the
    node sends it requests and learns what became of them only from
    :meth:`push_report`. ``venue_account`` names the venue account on every
    request. ``protective`` says who holds the stops and targets of positions
    (configured, attached to an order or an entry): ``"venue"`` sends them as
    protective orders (a stop, a target, linked one-cancels-other) that the
    venue holds, ``"local"`` has the engine watch the levels on market data
    and send a market order when one is reached. A leg the venue says it
    cannot hold is watched by the engine in either mode. Read the mode back
    as :attr:`protective`; the orders carry ids starting ``prot-`` in their
    ``tags["client_id"]``. See :mod:`raptorbt.execution` for the parity of
    each mode with a backtest.

    ``corporate_actions`` (:class:`~raptorbt.CorporateAction`) and
    ``adjustment_policy`` are as in :func:`~raptorbt.run_strategies`, except
    that a live node takes ``"raw"`` only: prices are as traded and positions
    are rescaled, and dividends paid, at the ex-date. Declare them for every
    action the node will see; without a policy there must be none.

    ``order_check`` is the host's pre-submit order check, the same callable
    :func:`~raptorbt.run_strategies` takes: every typed order passes it before
    the engine or the venue sees it (see :mod:`raptorbt.strategy.order_check`).
    ``order_check_scope="typed"`` declares that the check judges typed orders only, so
    ``enter()`` and ``close_position()`` are not refused for it (the default ``"all"`` refuses
    them with ``order_check_intent``).

    ``bindings`` maps a requirements role name to the symbol it means, or, for a
    :class:`~raptorbt.strategy.requirements.Universe` role, to the list of symbols chosen for it;
    with ``instruments=`` it is what ``ctx.role``, ``ctx.instrument`` and
    ``ctx.option_chain`` answer from, identically to
    :func:`~raptorbt.run_strategies` (see :mod:`raptorbt.strategy.context_host`).

    ``warmup`` says what warm-up bars do. ``"live"`` replays them as bars:
    orders match and equity is marked, as before. ``"muted"`` runs the
    strategies' hooks on them so indicators, aggregators and the strategies'
    own state are primed, but the orders and intents the strategies queue are
    discarded: no order is made, no position is held (positions adopted with
    ``initial_positions`` stay), no fill or fee happens, and nothing reaches
    the venue. ``None`` (the default) is ``"muted"`` for a node with your own
    venue (an :class:`~raptorbt.execution.ExecutionAdapter`), which a
    real exchange must never be sent history orders by, and ``"live"``
    otherwise. In either mode ``ctx.is_warmup`` is true in every hook that runs
    on a warm-up bar, and :meth:`warm_up` primes the node from history pushed
    after construction, always muted.

    ``processing_latency_ns`` declares how long the engine takes to act on an
    event after it arrives. Every event has three times (see
    ``docs/architecture/event-semantics.md``): the exchange's, the arrival and
    the processing time, read in hooks as ``ctx.event_ns``,
    ``ctx.arrival_ns`` and ``ctx.processing_ns``. The push methods take
    ``arrival_ns``; without it arrival is the event's own time.

    ``journal`` (a :class:`raptorbt.journal.Journal`) records everything the
    node is told, decides and sends, before it acts (see
    :mod:`raptorbt.journal`); without it nothing is recorded and nothing is
    paid. ``epoch`` is the node's execution epoch: an adapter with a
    ``fence(epoch)`` method raises :class:`~raptorbt.execution.Fenced` when a
    newer epoch owns the account, after which the node sends nothing (see
    :meth:`heartbeat`). ``strategy_states`` hands each strategy the plain data
    its ``state()`` returned in an earlier run, before ``on_start``.
    :meth:`checkpoint` and :meth:`restore` rebuild a node from its journal.

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
        execution: ExecutionAdapter | PaperExecution | SimulatedExecution | None = None,
        venue_account: str | None = None,
        protective: Literal["venue", "local"] = "venue",
        corporate_actions: Iterable[CorporateAction] | None = None,
        adjustment_policy: Literal["raw", "adjusted"] | None = None,
        processing_latency_ns: int = 0,
        journal=None,
        epoch: int | None = None,
        strategy_states: list | None = None,
        order_check=None,
        order_check_scope: Literal["all", "typed"] = "all",
        bindings: dict[str, str | Sequence[str]] | None = None,
        warmup: Literal["live", "muted"] | None = None,
        _via: str | None = None,
        _replay=None,
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
        primary_bars = _primary_bars(primary_bars, name)
        execution_kind = _execution_kind(execution, name)
        if oms_type == "hedging" and execution_kind == "adapter" and getattr(execution, "supports_hedging", True) is False:
            raise InvalidParameter(
                f"oms_type='hedging' keeps several positions per instrument, and {type(execution).__name__} "
                "sends its orders to a venue that nets them: a close that names a position would "
                "reduce the net one instead, and the node's book and the venue's would disagree. "
                "Use oms_type='netting' with this adapter, or an adapter whose venue keeps "
                "positions apart (supports_hedging = True)",
                code="hedging_unsupported_by_venue",
                operation=name,
                parameter="oms_type",
                value=oms_type,
                valid=["netting"],
            )
        if isinstance(processing_latency_ns, bool) or not isinstance(processing_latency_ns, int) or processing_latency_ns < 0:
            raise InvalidParameter(
                f"processing_latency_ns must be a non-negative integer, got {processing_latency_ns!r}",
                code="invalid_execution",
                operation=name,
                parameter="processing_latency_ns",
                value=processing_latency_ns,
                valid=["a non-negative integer"],
            )
        accounts = check_accounts(accounts, name)
        strategies, names = unpack_entries(strategies, accounts, name)
        check_group_sizes(names, name)
        entries = _normalise(strategies, dict.fromkeys(symbols), name)
        # A strategy given no symbols trades every symbol of the node, those added later too.
        self._open_ended = [
            not (isinstance(item, tuple) and item[1] is not None) for item in strategies
        ]
        self._added: list[str] = []
        self._init_recovery(journal, epoch, execution, execution_kind, entries, strategy_states, _replay)
        seeds = _seeds(initial_positions, entries)
        reject_atr_stops(config, instrument_configs, symbols)
        self._bar_config = config
        # Pushed events cannot be put on one later basis (docs/architecture/lifecycle.md,
        # "Live nodes"), so every symbol counts as a tick symbol: only the policy 'raw'
        # (prices as traded, positions rescaled at the ex-date) is open to a live node.
        corporate_actions, adjustment_policy = _lifecycle(
            corporate_actions, adjustment_policy, set(symbols), set(symbols), name, live=True
        )

        if warmup not in (None, "live", "muted"):
            raise InvalidParameter(
                f"warmup must be 'live', 'muted' or None, got {warmup!r}",
                code="invalid_warmup",
                operation=name,
                parameter="warmup",
                value=warmup,
                valid=["live", "muted", None],
            )
        self._warmup_muted = (warmup == "muted") or (warmup is None and execution_kind == "adapter")
        self._host = context_host.build(instruments, bindings, entries, symbols, name)
        self._muted_now = False
        if execution_kind == "adapter":
            # The adapter learns which broker key is which of this node's symbols, so a position
            # the broker reports under its own key reads as the engine's symbol after a restart.
            know = getattr(execution, "know_symbols", None)
            if callable(know):
                know(symbols)
        if execution_kind == "adapter" and callable(getattr(execution, "place_exit_rule", None)):
            self._host.venue = _VenueExits(self, execution)
            self._host.index_of = lambda strategy: next(
                (k for k, s in enumerate(self._strategies) if s is strategy), None
            )
        self._host.node = weakref.ref(self)
        #: What strategies asked the node to add; the host appends, the node drains (in place).
        self._requests = self._host.pending
        self._protective = protective
        self._symbols = symbols
        #: Per symbol, the check that a pushed bar falls on a date its session schedule covers.
        self._coverage = {
            s: c
            for s in symbols
            if (c := SessionCoverage.of(s, (instruments or {}).get(s), f"{name}.push_bar"))
        }
        self._index_of = {s: i for i, s in enumerate(symbols)}
        self._last_push = None
        #: Per symbol, the last bar delivered: a retried push of it is refused, not delivered twice.
        self._last_bar: dict[str, tuple] = {}
        #: What stopped the call that poisoned the node, as ``(code, first line)``.
        self._poison: tuple[str | None, str] | None = None
        self._ungrouped = 0
        self._retired: dict[str, int] = {}
        self._holders: dict[str, list[int]] = {}
        self._finished = False
        self._stamps = BarStamps.resolve(
            bar_stamp, bar_ns, symbols, name, require=bool(warmup_bars)
        )
        warmup = _warmup(warmup_bars, self._index_of, self._stamps)

        market = MarketData.live(symbols, [self._stamps.offset(s) for s in symbols], history)
        config = config_for_holding(config, [strategy for strategy, _ in entries], name)
        rt = errors.call_as(
            name,
            Runtime,
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
        if _replay is not None:
            rt.set_replay(True)
        self._strategies: list[Strategy] = []
        try:
            order_check_mod.install(rt, order_check, name, order_check_scope)
            if execution_kind is not None:
                rt.set_execution(
                    execution_kind,
                    None if execution_kind == "paper" else execution,
                    venue_account,
                    protective,
                )
            if processing_latency_ns:
                rt.set_processing_latency(processing_latency_ns)
            if curve_samples is not None:
                rt.set_curve_window(curve_samples)
            if primary_bars is not None:
                rt.set_primary_bars(*primary_bars)
            # The journal is attached, and its first record written, before any
            # strategy code runs: what ``on_init`` / ``on_start`` do (an order
            # queued, a halt, an adopted position) is on the record from its
            # first act.
            if journal is not None or epoch is not None:
                rt.set_journal(journal, epoch)
            self._journal_start(symbols, len(entries), strategy_states)
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
                    host=self._host,
                )
                self._strategies.append(strategy)
                self._sids.append(sid)
                self._ctxs.append(ctx)
                self._views.append(view)
                self._symbols_of.append(syms)
            if adjustment_policy is not None:
                rt.set_corporate_actions(corporate_actions, adjustment_policy)
            rt.finalize()
            # Prime indicators and strategy state from the warmup history now,
            # so the node is warm before the first live event. Muted history
            # discards what hooks queue; what ``on_start`` queued is not
            # history's, so it waits and goes with the first live event.
            if warmup and self._warmup_muted:
                rt.hold_boot()
            rt.set_warmup(True, self._warmup_muted)
            self._muted_now = self._warmup_muted
            try:
                # History is replayed in group form: the bars of one close time
                # are applied before any is decided.
                rt.begin_group(None, False)
                try:
                    for push in warmup:
                        rt.push_bar(*push)
                finally:
                    rt.flush(False)
            finally:
                self._muted_now = False
            rt.set_warmup(False)
            if warmup and self._warmup_muted:
                rt.release_boot()
        except BaseException:
            rt.close()
            for strategy in self._strategies:
                api.release(strategy)
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
        *,
        arrival_ns: int | None = None,
    ) -> int:
        """Feed one tick row; every hook it triggers fires before this returns.

        ``ltp > 0`` prints a trade; a two-sided book yields a quote after it.
        ``ltq`` is the print's own size when the feed has it (else the flow
        deltas stand in); ``bid_qty``/``ask_qty`` are the displayed L1 sizes;
        ``oi`` the open interest. Returns how many events were appended (0-2).

        ``arrival_ns`` is when the row reached the engine, if that differs from
        ``timestamp`` (see the class docs).

        A row that yields no event still carries a timestamp: alerts and timers
        due by it fire before this returns, for each strategy that trades
        ``symbol``. Orders they submit are routed to their instrument at once
        and match from its next print; intents they queue are held for this
        symbol's next print.

        Timestamps must not decrease per symbol (equal ones are allowed);
        a decreasing one raises ``ValueError`` and changes nothing.
        """
        # The checks of ``_check_open``, ``ensure_pushed_finite``, ``_note_push`` and ``_index``,
        # in their order, each taken whole only when its quick test fails: a push is per tick.
        if (
            self._finished
            or self._replaying
            or self._state != "live"
            or (self._epoch is not None and self._rt.fenced)
        ):
            self._check_open(push=True)
        if not (_isfinite(ltp) and _isfinite(bid) and _isfinite(ask)):
            ensure_pushed_finite(symbol, f"{self._name}.push_tick", ltp=ltp, bid=bid, ask=ask)
        last = self._last_push
        self._last_push = (symbol, timestamp)
        if not (last is None or last[1] != timestamp or last[0] == symbol):
            self._note_ungrouped(last, symbol, timestamp)
        try:
            index = self._index_of.get(symbol)
            return self._rt.push_tick(
                self._index(symbol, "push_tick") if index is None else index,
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
                arrival_ns,
            )
        except BaseException as e:
            self._note_failure(e)
            raise

    def push_bar(
        self,
        symbol: str,
        timestamp: int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        *,
        arrival_ns: int | None = None,
    ) -> None:
        """Feed one closed bar. It executes: orders match, equity marks.

        ``timestamp`` is stamped as the node's ``bar_stamp`` declares.
        """
        self._check_open(push=True)
        self._check_bar(symbol, timestamp, open, high, low, close, volume, "push_bar")
        self._push_bar(symbol, timestamp, open, high, low, close, volume, arrival_ns)

    def _push_bar(self, symbol, timestamp, open, high, low, close, volume, arrival_ns) -> None:
        """``push_bar`` once the node is open and the bar checked (``_check_bar``)."""
        self._stamps.require_given(f"{self._name}.push_bar")
        self._note_push(symbol, timestamp)
        index = self._index(symbol, "push_bar")
        try:
            self._rt.push_bar(index, timestamp, open, high, low, close, volume, arrival_ns)
        except BaseException as e:
            self._note_failure(e)
            raise
        finally:
            # A bar the runtime took is delivered, whatever a hook then raised: the host's retry
            # of it is refused as a duplicate rather than delivered and journaled twice.
            taken = self._rt.delivered_bar(index)
            if taken is not None:
                self._last_bar[symbol] = taken

    def warm_up(self, bars: dict[str, dict]) -> None:
        """Prime the strategies from history pushed after construction.

        ``bars`` is ``warmup_bars``: symbol -> OHLCV arrays, stamped as the
        node's ``bar_stamp`` declares. The bars of different symbols merge by
        close time, ties by symbol order, and are replayed muted: hooks run and
        indicators, aggregators and strategy state update, with
        ``ctx.is_warmup`` true, but no order is made and no position held. They
        are ordinary bars for the node's clock, so they must not be older than
        what was already pushed for a symbol (that raises and changes nothing
        for the rest).
        """
        self._check_open()
        self._stamps.require_given(f"{self._name}.warm_up")
        self._warm_up(_warmup(bars, self._index_of, self._stamps))

    def _warm_up(self, pushes) -> None:
        """Push ``pushes`` (``(index, ts, o, h, l, c, v)``) as ``warm_up`` does: muted, in one
        group kept off the journal. A restore passes the bars one ``warm_up`` journaled."""
        self._rt.set_warmup(True, True)
        self._muted_now = True
        try:
            self._rt.begin_group(None, False)
            try:
                for push in pushes:
                    self._rt.push_bar(*push)
            finally:
                self._rt.flush(False)
        finally:
            self._muted_now = False
            self._rt.set_warmup(False)

    def begin_group(self, timestamp: int | None = None) -> None:
        """Open a timestamp group: the pushes that follow are held and
        delivered together, as a batch run delivers the events that share a
        timestamp.

        By default each push is delivered on arrival, one event at a time, as
        a group of one. A host that has the bars of several instruments for
        one timestamp opens a group, pushes them, and calls :meth:`flush`
        (or uses :meth:`push_group`): every event is then applied before any
        is decided, and every decision is taken before any is matched, so what
        a decision sees of another instrument, and the bar a cross-instrument
        order meets, do not depend on the order the pushes arrived in.
        Events that share a timestamp and a kind (trade, quote, depth, bar)
        are one group. A push stamped later than what the group holds,
        :meth:`advance_time`, :meth:`push_report`, :meth:`add_instrument`,
        :meth:`retire_instrument` and :meth:`finish` deliver what is complete
        or held. A node reproduces a batch run exactly when it delivers the
        same groups.

        ``timestamp`` is the group's, if you know it; it is only recorded. The
        journal writes ``group_begin`` here and ``group_end`` at :meth:`flush`;
        a journal without them replays every push alone, as it ran.
        """
        self._check_open(push=True)
        try:
            self._rt.begin_group(timestamp)
        except BaseException as e:
            self._note_failure(e)
            raise
        self._group_ts = timestamp

    def flush(self) -> None:
        """Close the open timestamp group and deliver what it holds. With none
        open, nothing happens."""
        self._check_open(push=True)
        try:
            self._rt.flush()
        except BaseException as e:
            self._note_failure(e)
            raise

    def push_group(
        self,
        timestamp: int,
        bars: dict[str, "tuple[float, float, float, float, float] | dict[str, float]"],
        *,
        arrival_ns: int | None = None,
    ) -> None:
        """Feed the bars of several symbols for one timestamp as one group.

        ``bars`` maps each symbol to ``(open, high, low, close, volume)`` or to
        a dict with those keys. ``timestamp`` is stamped as the node's
        ``bar_stamp`` declares, as for :meth:`push_bar`. The symbols are
        delivered in ascending symbol order whatever order the mapping lists
        them in; see :meth:`begin_group` for the rule.
        """
        # Every bar is checked before the group opens, so a bad one delivers nothing and a
        # retry delivers each good bar once. What can still fail inside the group (the history
        # ring full, the journal's sink) leaves the group open with the pushes it took, and
        # nothing is flushed: ``open_group`` says what it holds, and the host sends the rest.
        self._check_open(push=True)
        self._stamps.require_given(f"{self._name}.push_group")
        rows = []
        for symbol, bar in bars.items():
            o, h, l, c, v = (
                (bar[k] for k in ("open", "high", "low", "close", "volume"))
                if isinstance(bar, dict)
                else bar
            )
            self._check_bar(symbol, timestamp, o, h, l, c, v, "push_group")
            rows.append((symbol, o, h, l, c, v))
        # A subclass that overrides push_bar is called for each bar, as it always was.
        own = type(self).push_bar is LiveNode.push_bar
        self.begin_group(timestamp)
        for symbol, o, h, l, c, v in rows:
            if own:
                # Each bar was checked above, and nothing a check reads changes while the group
                # fills: the group's timestamp is the bars', and each symbol appears once.
                self._check_open(push=True)
                self._push_bar(symbol, timestamp, o, h, l, c, v, arrival_ns)
            else:
                self.push_bar(symbol, timestamp, o, h, l, c, v, arrival_ns=arrival_ns)
        self.flush()

    def _check_bar(self, symbol, timestamp, open, high, low, close, volume, call) -> None:
        """What ``push_bar`` refuses before it delivers anything, in its order."""
        # Each check is taken whole only when its quick test fails: a bar is checked per push.
        config = self._bar_config
        if config is not None and (config.order_latency_ns != 0 or config.partial_fills):
            reject_bar_only_settings(config, [symbol], f"{self._name}.{call}")
        if not (_isfinite(open) and _isfinite(high) and _isfinite(low) and _isfinite(close)):
            ensure_pushed_finite(
                symbol, f"{self._name}.{call}", open=open, high=high, low=low, close=close
            )
        group_ts = getattr(self, "_group_ts", None)
        if group_ts is not None and self._rt.group_open and timestamp != group_ts:
            # A group declared at a timestamp holds that timestamp's bars: one stamped otherwise
            # is not part of it (it would be delivered as if it were).
            raise StateError(
                f"the open group is at {group_ts}: a bar for {symbol!r} stamped {timestamp} is not part of it",
                code="group_timestamp_mismatch",
                operation=f"{self._name}.{call}",
                parameter="timestamp",
                value=timestamp,
                valid=[group_ts],
            )
        last = self._last_bar.get(symbol)
        if last is not None and last[0] == timestamp:
            # A bar is one per symbol and timestamp. A retry of a push that was delivered is
            # refused, not delivered twice (a second delivery would be a second bar for every
            # indicator, aggregator and decision); the code says whether it repeats the bar or
            # contradicts it.
            same = last[1:] == (open, high, low, close, volume)
            if same:
                raise StateError(
                    f"{symbol!r} already has a bar at {timestamp}: this one repeats it, drop the retry",
                    code="duplicate_bar",
                    operation=f"{self._name}.{call}",
                    parameter="timestamp",
                    value=timestamp,
                )
            raise StateError(
                f"{symbol!r} already has a bar at {timestamp} with other values: a bar is not revised",
                code="conflicting_bar",
                operation=f"{self._name}.{call}",
                parameter="timestamp",
                value=timestamp,
            )
        covered = self._coverage.get(symbol)
        if covered is not None:
            covered.check(timestamp)
        if call == "push_group":
            self._rt.check_bar(self._index(symbol, call), timestamp)

    def _note_failure(self, error: BaseException) -> None:
        """Keep the error of the call that poisoned the node, for ``health()``."""
        if self._poison is None and self._rt.poisoned():
            text = str(error).splitlines()
            self._poison = (getattr(error, "code", None), text[0] if text else type(error).__name__)

    def _note_push(self, symbol: str, timestamp: int) -> None:
        """Count a push that shares its timestamp with the previous push of another instrument
        while no group is open: a batch run delivers those together, this node delivers them one
        at a time, so what a decision sees can differ. The first one is logged (WARNING, ``raptorbt.strategy.live_node``)."""
        last = self._last_push
        self._last_push = (symbol, timestamp)
        if last is None or last[1] != timestamp or last[0] == symbol:
            return
        self._note_ungrouped(last, symbol, timestamp)

    def _note_ungrouped(self, last, symbol: str, timestamp: int) -> None:
        """The part of ``_note_push`` for a push at the previous push's timestamp."""
        if self._rt.group_open:
            return
        self._ungrouped += 1
        if self._ungrouped == 1:
            _log.warning(
                f"{self._name}: pushes for {last[0]!r} and {symbol!r} share timestamp {timestamp} "
                "but were not grouped, so they are delivered one at a time, not together as a batch "
                "run delivers them, and a decision may see the other instrument's old bar. "
                "Use push_group(...) or begin_group()/flush(); health() counts these pushes "
                "(ungrouped_same_timestamp_pushes)."
            )

    @property
    def open_group(self) -> dict | None:
        """The timestamp group left open, if any, and what it holds: ``{"timestamp": ns,
        "symbols": [...]}``. A node restored from a journal that ends inside a group has one:
        the replay delivered nothing of it, exactly as before the crash, so the host sends the
        symbols not listed here and calls :meth:`flush`."""
        if not self._rt.group_open:
            return None
        held = self._rt.held_pushes()
        names = {i: s for s, i in self._index_of.items()}
        symbols = sorted({names[i] for i, _ in held})
        return {"timestamp": min((t for _, t in held), default=None), "symbols": symbols}

    def push_depth(
        self,
        symbol: str,
        timestamp: int,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        *,
        arrival_ns: int | None = None,
    ) -> None:
        """Feed one depth snapshot: ``(price, size)`` lists, best first."""
        self._check_open(push=True)
        ensure_pushed_finite(
            symbol,
            f"{self._name}.push_depth",
            **{f"{side}[{i}]": price for side, levels in (("bids", bids), ("asks", asks)) for i, (price, _) in enumerate(levels)},
        )
        self._note_push(symbol, timestamp)
        try:
            self._rt.push_depth(self._index(symbol, "push_depth"), timestamp, bids, asks, arrival_ns)
        except BaseException as e:
            self._note_failure(e)
            raise

    def advance_time(self, timestamp: int) -> None:
        """Move time to ``timestamp`` with no data.

        Fires the alerts and timers that are due for every strategy, on each
        of its symbols' clocks, in strategy order. Orders they submit are
        routed at once and match from their instrument's next print.
        """
        self._check_open(push=True)
        try:
            self._rt.advance_time(timestamp)
        except BaseException as e:
            self._note_failure(e)
            raise

    def push_report(
        self, client_order_id: str, report: ExecutionReport, *, arrival_ns: int
    ) -> str:
        """Hand the node what the venue said about an order.

        ``client_order_id`` is the ``client_order_id`` of the
        :class:`~raptorbt.execution.OrderRequest` the order was sent as.
        ``arrival_ns`` is when the report reached you; the report's own
        ``ts_ns`` is the venue's time, and the node's processing time is the
        arrival plus ``processing_latency_ns``. The three are what hooks read
        as ``ctx.event_ns``, ``ctx.arrival_ns`` and ``ctx.processing_ns``.

        The report is applied now, as an event in the node's single stream:
        in the order calls are made, between pushes of market data, never
        from another thread. When a simulated venue answers a market event,
        its reports come after that event's hooks and before the next event.
        Reports may repeat, arrive late or arrive out of order; applying them
        again, or in another order, leaves the same orders, positions and
        cash. Returns ``"applied"``, ``"duplicate"`` (already recorded),
        ``"late"`` (no longer matters, or cannot matter yet), ``"deferred"`` (the venue filled
        the order and the book could not book the fill yet, for instance an early reduce-only fill
        before the entry it reduces, or for want of margin: the fill is kept and books itself when
        the book can, and until then the position at the venue and the book disagree; the journal
        holds an ``unbooked_fill`` record and ``health()["unbooked_fills"]`` lists it with its
        reason) or ``"unbooked"`` (an execution on an order the node no longer holds: refused,
        journaled as ``unbooked_fill`` with reason ``order_evicted``, never booked; reconcile the
        position with the venue).

        Raises when the node has no execution adapter, when no order was sent
        under ``client_order_id``, or when the report contradicts the order.
        """
        self._check_open(push="report")
        if not isinstance(report, ExecutionReport):
            raise InvalidParameter(
                f"report must be a raptorbt.execution.ExecutionReport, got {type(report).__name__}",
                code="invalid_execution",
                operation=f"{self._name}.push_report",
                parameter="report",
                value=type(report).__name__,
            )
        try:
            return self._rt.push_report(client_order_id, report, arrival_ns)
        except BaseException as e:
            self._note_failure(e)
            raise

    # -- the instrument set ------------------------------------------------------

    def add_instrument(
        self,
        symbol: str,
        spec=None,
        *,
        strategies: list[int] | None = None,
        history: int | None = None,
        direction: int = 1,
        bar_ns: int | None = None,
    ) -> None:
        """Add an instrument to the running node, between pushes.

        ``symbol`` becomes a symbol of the node (``node.symbols``, ``ctx.symbols``) for
        ``strategies``, a list of strategy indices, or by default every strategy that was given
        no symbols of its own. ``spec`` is its :class:`~raptorbt.InstrumentSpec` (also what
        ``ctx.instrument`` and ``ctx.option_chain`` then answer from); it is tradable, or an
        index (a data-only reference: its orders are refused, ``not_tradable``), and names
        ``symbol``. ``history`` is how many rows its rings hold (default: the node's
        ``history``), always at least what the node's indicators read. ``bar_ns`` is its bar
        width under ``bar_stamp="open"`` when the node's symbols do not share one.

        The instrument gets the highest id and is each chosen strategy's last local instrument,
        so nothing that exists moves: every strategy's other symbols, order ids, positions and
        event ordinals are as they were. The composite subscriptions and the symbol-less
        indicators a strategy declared in ``on_start`` follow the new instrument, a clock copy
        of the schedule it set there is started for it, a ``PaperExecution`` venue's shadow book
        and the order engine take orders on it, and a strategy that has ``Requirements``
        resolves it through ``ctx.role`` / ``ctx.option_chain``. An indicator registered with
        ``symbol=`` belongs to that symbol and is not created again for the new one.

        A symbol that was retired comes back with the same id: fresh rings, its kernels flat as
        they were retired, fresh aggregators, for ``strategies`` that held it (default: those
        that did), with the spec it had. The record says ``revive``. An execution adapter that
        has ``add_instrument(symbol, spec)`` is told before any order for the symbol can reach it
        (it may refuse: nothing changes); ``retire_instrument(symbol)`` likewise. Each strategy
        then hears ``on_instrument_added(ctx, symbol)``.

        The add is journaled (kind ``instrument_add``, with the spec) before the engine acts,
        stamped at the node's clock, and :meth:`restore` replays it. Raises on a symbol the node
        already trades, and in ``recovering`` state; changes nothing then.
        """
        self._check_open(push="instrument")
        name = f"{self._name}.add_instrument"
        if not isinstance(symbol, str) or not symbol:
            raise InvalidParameter(
                f"symbol must be a non-empty string, got {symbol!r}",
                code="duplicate_symbols", operation=name, parameter="symbol", value=symbol,
            )
        revive = symbol in self._retired
        if symbol in self._index_of:
            raise InvalidParameter(
                f"{symbol!r} is already an instrument of this node",
                code="duplicate_symbols", operation=name, parameter="symbol", value=symbol,
            )
        if revive:
            have = self._host.specs.get(symbol)
            if spec is None:
                spec = have
            elif have is not None and spec.to_json() != have.to_json():
                raise InvalidParameter(
                    f"{symbol!r} comes back with the spec it was retired with",
                    code="invalid_instrument_spec", operation=name, parameter="spec", value=symbol,
                )
            if strategies is None:
                strategies = list(self._holders[symbol])
        if spec is not None and getattr(spec, "symbol", symbol) != symbol:
            raise InvalidParameter(
                f"spec is for {spec.symbol!r}, not {symbol!r}",
                code="invalid_instrument_spec", operation=name, parameter="spec", value=symbol,
            )
        if strategies is None:
            strategies = [k for k, open_ in enumerate(self._open_ended) if open_]
            if not strategies:
                raise InvalidParameter(
                    "every strategy of this node names its own symbols: say which should trade "
                    f"{symbol!r} with strategies=[...]",
                    code="invalid_strategies", operation=name, parameter="strategies",
                )
        else:
            strategies = sorted(set(strategies))
        offset = self._stamps.offsets[symbol] if revive else self._stamps.offset_for_new(symbol, bar_ns, name)
        index = self._rt.add_instrument(symbol, spec, direction, strategies, history, offset)
        self._requests[:] = [r for r in self._requests if r[0] != symbol]
        if not self._requests and self._state == "adding":
            self._state = "live"
        if revive:
            self._retired.pop(symbol)
        else:
            self._stamps.add(symbol, offset)
            self._added.append(symbol)
        self._holders[symbol] = list(strategies)
        self._symbols.append(symbol)
        self._index_of[symbol] = index
        for k in strategies:
            self._symbols_of[k] = [*self._symbols_of[k], symbol]
            run = getattr(self._strategies[k], "_api", None)
            if run is not None:
                run.add_symbol(symbol)
        if spec is not None:
            self._host.specs[symbol] = spec
        self._host.changed()
        for k in strategies:
            self._strategies[k].on_instrument_added(self._ctxs[k], symbol)

    @property
    def retired_instruments(self) -> tuple[str, ...]:
        """The symbols retired with :meth:`retire_instrument` (:meth:`add_instrument` revives them)."""
        return tuple(self._retired)

    def holders_of(self, symbol: str) -> tuple[int, ...]:
        """The strategy indices that traded ``symbol`` when it was last added or revived."""
        return tuple(self._holders.get(symbol, ()))

    @property
    def added_instruments(self) -> tuple[str, ...]:
        """The symbols added with :meth:`add_instrument`, in the order added (retired ones too)."""
        return tuple(self._added)

    @property
    def instruments(self) -> dict:
        """The ``InstrumentSpec`` of every symbol the host gave one for (``instruments=`` and
        :meth:`add_instrument`), listed contracts the node does not trade included. A copy."""
        return dict(self._host.specs)

    @property
    def bindings(self) -> dict[str, str | tuple[str, ...]]:
        """The role bindings the node was built with (a Universe role's symbols as a tuple). A copy."""
        return dict(self._host.bindings)

    @property
    def now_ns(self) -> int:
        """Event time of the latest input the node took (``0`` before the first)."""
        return self._rt.last_event_ns()

    def last_prices(self, strategy_index: int) -> dict[str, float]:
        """The last price of each symbol strategy ``strategy_index`` trades, as it reads them: the
        last close, else the last traded ``ltp``. A symbol with no price yet (no data, or only
        quote-only rows in view) is not in the dict."""
        return self._host._prices(self._ctxs[strategy_index])

    def apply_plan(self, plan) -> None:
        """Apply a :func:`~raptorbt.strategy.requirements.plan_instruments` plan: retire what it
        says to retire, then add what it says to add, each as its own journaled step. Stops at
        the first refusal; what was applied stays applied."""
        for symbol in plan.retire:
            self.retire_instrument(symbol)
        specs = self._host.specs
        for symbol, strategies in plan.add.items():
            self.add_instrument(symbol, specs.get(symbol), strategies=list(strategies))

    def _requested(self) -> None:
        """A strategy queued a request in ``self._requests``: the next input adds it."""
        if self._state == "live":
            self._state = "adding"

    def _add_requested(self) -> None:
        """Add what strategies asked for (``ctx`` hosts a request when an ``auto_roll`` needs a
        contract the node does not trade) before the next input is taken."""
        self._state = "live"
        wanted: dict[str, list[int]] = {}
        for symbol, k in self._requests:
            wanted.setdefault(symbol, []).append(k)
        for symbol, strategies in wanted.items():
            self.add_instrument(symbol, self._host.specs[symbol], strategies=sorted(set(strategies)))

    def retire_instrument(self, symbol: str) -> None:
        """Stop trading ``symbol`` and free its history, between pushes.

        Refused with ``instrument_in_use`` while any strategy holds a position or a working
        order on it (or a protective level the engine watches, or intents held for its next
        step); nothing changes then, and the error says which. A retired symbol leaves
        ``node.symbols`` and ``ctx.symbols``, takes no more data, and its rings, aggregators and
        indicator references are freed; its id and its trades and orders stay for
        :meth:`finish` and the node's results, and the id is never reused. The spec stays in
        ``instruments`` so ``ctx.option_chain`` still lists the contract. Journaled
        (``instrument_retire``) and replayed like :meth:`add_instrument`.
        """
        self._check_open(push="instrument")
        name = f"{self._name}.retire_instrument"
        if symbol not in self._index_of:
            raise UnknownSymbol(
                f"{symbol!r} is not an active instrument of this node",
                code="unknown_symbol", operation=name, parameter="symbol", value=symbol,
                valid=list(self._symbols),
            )
        self._rt.retire_instrument(symbol)
        self._requests[:] = [r for r in self._requests if r[0] != symbol]
        self._retired[symbol] = self._index_of.pop(symbol)
        self._symbols.remove(symbol)
        holders = [k for k, symbols in enumerate(self._symbols_of) if symbol in symbols]
        for k, symbols in enumerate(self._symbols_of):
            if symbol in symbols:
                self._symbols_of[k] = [s for s in symbols if s != symbol]
                run = getattr(self._strategies[k], "_api", None)
                if run is not None:
                    run.drop_symbol(symbol)
        self._host.changed()
        for k in holders:
            self._strategies[k].on_instrument_retired(self._ctxs[k], symbol)

    # -- state ---------------------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def strategies(self) -> list[Strategy]:
        """The strategies, in the order given."""
        return list(self._strategies)

    @property
    def protective(self) -> str:
        """Who holds the stops and targets of positions: ``"venue"`` or ``"local"``."""
        return self._protective

    def ctx(self, strategy_index: int):
        """The context handed to strategy ``strategy_index``'s hooks. Refused once the node
        has finished: the finish clears the contexts."""
        self._check_open()
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
        self._journal_host(
            "halt", {"account": account, "reason": reason, "flatten": flatten}
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
        self._check_open(push=True)
        self._rt.flush()
        self._journal_host("finish", None)
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
            for strategy in self._strategies:
                api.release(strategy)
            if self._journal is not None:
                self._journal.sync()

    # -- journal, checkpoint and recovery -------------------------------------

    @property
    def journal(self):
        """The node's :class:`~raptorbt.journal.Journal`, or ``None``."""
        return self._journal

    @property
    def state(self) -> str:
        """``"live"``, ``"recovering"`` after :meth:`restore` until :meth:`resume`, or
        ``"poisoned"`` once a call stopped part way through a booking (every push is then refused;
        :meth:`health` says what stopped)."""
        if self._rt.poisoned():
            return "poisoned"
        return "live" if self._state == "adding" else self._state

    @property
    def epoch(self) -> int | None:
        return self._epoch

    @property
    def fenced(self) -> bool:
        """Whether a newer execution epoch owns the account."""
        return self._epoch is not None and self._rt.fenced

    def annotate(self, ts: int, kind: str, detail=None) -> None:
        """Write a note of your own into the journal at event time ``ts``
        (an adoption, an unattributed fill, an operator command): record kind
        ``annotation``, payload ``{"kind", "detail"}``. Refused without a
        journal."""
        self._check_open()
        if self._journal is None:
            raise InvalidParameter(
                "annotate() needs a node built with journal=",
                code="invalid_execution",
                operation=f"{self._name}.annotate",
                parameter="journal",
            )
        self._journal.append(
            "annotation", ts, {"kind": kind, "detail": detail}, source="host"
        )

    def checkpoint(self, include_log: bool = True) -> bytes:
        """A versioned snapshot to rebuild this node from: the journal head,
        a digest of the state to check the rebuild against, each strategy's
        ``state()``, the epoch and (``include_log``) every journal record since
        the journal's ``seq_base``. Taken between pushes. See
        :meth:`restore`."""
        self._check_open()
        self._no_open_group()
        from raptorbt.strategy import _recovery

        return _recovery.checkpoint(self, include_log)

    @classmethod
    def restore(cls, checkpoint, strategies, symbols, config=None, **kwargs):
        """Rebuild a node from its journal after a crash.

        ``checkpoint`` is what :meth:`checkpoint` returned (or ``None`` with
        ``journal=`` and ``records=`` from
        :func:`raptorbt.journal.open_for_recovery`); ``strategies``, ``symbols``,
        ``config`` and the other arguments are those the node was built with
        (the same ``warmup_bars`` too), plus ``execution=`` the adapter to
        reconnect. The journal is replayed into the fresh node with the adapter
        muted: every record the replay makes must equal the one the journal
        holds, or ``replay_diverged`` is raised. A request whose result was
        never written is *not* sent again; its order stays as it was.

        Then, if there is an adapter, the node asks the venue what it holds
        (:meth:`reconcile`). With no discrepancy it resumes. With some, it stays
        ``state == "recovering"`` (market data is refused; reports, ``resend``
        and ``reconcile`` are not) until you :meth:`resume`, accepting them.
        ``reconcile="skip"`` skips the venue and says so in the journal.
        ``cash_tolerance`` is the largest difference between the node's cash
        and the venue's balance that still counts as equal (default ``0.0``;
        a broker's balance rarely matches to the paisa). ``on_discrepancy``
        is called with each :class:`~raptorbt.journal.Discrepancy`.
        """
        from raptorbt.strategy import _recovery

        return _recovery.restore(cls, checkpoint, strategies, symbols, config, **kwargs)

    def reconcile(self, cash_tolerance: float = 0.0):
        """Ask the adapter what the venue holds (``open_orders()``,
        ``positions()``, ``balances()``, each optional) and compare it with
        the node. Returns the list of :class:`~raptorbt.journal.Discrepancy`
        and delivers each to the strategy's ``on_discrepancy``.

        An adapter with ``catch_up()`` (a ``BrokerAdapter``) is read first and
        the node takes the reports it returns, so an order that filled while
        no node was listening is booked and compared as filled, not reported
        as ``missing_order``. ``cash_tolerance`` is the largest difference
        between the node's cash and the venue's balance that counts as equal."""
        self._check_open()
        if self._state != "recovering":
            self._no_open_group()
        from raptorbt.strategy import _recovery

        return _recovery.reconcile(self, cash_tolerance)

    def resume(self, accept=()) -> None:
        """Leave ``recovering``. Every discrepancy of the last reconcile must be
        in ``accept`` (``Discrepancy`` objects, their ``.key``, or ``"all"``);
        otherwise ``recovering`` is raised and nothing changes."""
        self._check_open()
        if self._state != "recovering":
            # Only a node restored from a journal that ended inside a group may resume with it open.
            self._no_open_group()
            return
        if isinstance(accept, str):
            if accept != "all":
                raise InvalidParameter(
                    "accept is 'all' or a list of discrepancies",
                    code="invalid_execution", operation=f"{self._name}.resume",
                    parameter="accept", value=accept,
                )
            keys = set(self._pending)
        else:
            keys = {getattr(a, "key", None) or tuple(a) for a in accept}
        left = [d for k, d in self._pending.items() if k not in keys]
        if left:
            raise StateError(
                f"{len(left)} discrepancies are neither resolved nor accepted: "
                + "; ".join(f"{d.kind} {d.client_order_id or d.symbol}" for d in left[:5]),
                code="recovering",
                operation=f"{self._name}.resume",
            )
        self._resume_record({"accepted": [list(k) for k in sorted(map(str, keys & set(self._pending)))]})
        self._pending = {}
        self._state = "adding" if self._requests else "live"

    def resend(self, client_order_id: str) -> None:
        """Send an order again under its first request (same client order id).
        Only for an order :meth:`reconcile` reported ``missing_order`` after
        the venue was asked and does not hold it."""
        self._check_open()
        self._rt.resend(client_order_id)

    def heartbeat(self) -> dict:
        """Check the execution epoch with the adapter now and return
        :meth:`health`. Raises ``fenced`` when a newer epoch owns the account.
        Call it on a timer; the node also checks before every request."""
        self._check_open()
        if self._epoch is not None:
            self._rt.heartbeat()
        return self.health()

    def health(self) -> dict:
        """State for a supervisor: ``state``, ``epoch``, ``fenced``, the
        journal's ``count`` and ``chain``, the orders with no confirmed
        outcome, ``halted`` (the indices of halted strategies) and, when the
        adapter has a ``health()``, its own state under ``adapter``
        (``BrokerAdapter``: ``paused``, ``outbox``, ``faults``,
        ``recent_faults`` and its executor's ``alive``). ``margin_faults`` lists each strategy
        stopped for holding a sold option with no hedge left; it is why the strategy stopped, and
        a stop cannot be undone, so it stays listed after the positions are closed and the
        strategy cannot buy the hedge back."""
        if not self._finished:
            self._no_open_group()
        out = {
            "state": self.state,
            "epoch": self._epoch,
            "fenced": self.fenced,
            "finished": self._finished,
            "ungrouped_same_timestamp_pushes": self._ungrouped,
            "poisoned": self._rt.poisoned(),
        }
        if not self._finished and not out["poisoned"]:
            # venue fills no book could take: (client order id, quantity, price, reason)
            out["unbooked_fills"] = self._rt.unbooked_fills()
            # strategies stopped for a sold option with no hedge left: (strategy index, reason)
            out["margin_faults"] = self._rt.margin_faults()
            # strategies halted for any cause (a follow-up loop, a margin call, the drawdown
            # kill switch, a halt from outside): their entries are refused, the node runs on
            out["halted"] = [i for i, v in enumerate(self._views) if v.is_halted()]
        if out["poisoned"]:
            # the error of the call that stopped, as ``(code, message)``
            out["poison_reason"] = self._poison
        if self._journal is not None:
            out["journal_count"] = self._journal.count
            out["chain"] = self._journal.chain
        if not self._finished:
            out["unconfirmed"] = sorted(self._rt.unconfirmed())
            out["clock_ns"] = self._rt.clock()[0]
            # an adapter that can say how it is doing (a paused outbox, a dead worker thread,
            # jobs that raised) is asked, so a supervisor sees it here
            adapter_health = getattr(self._execution, "health", None)
            if callable(adapter_health):
                out["adapter"] = adapter_health()
        return out

    def state_digest(self) -> str:
        """A hash of the books, orders and cash. Equal digests mean equal
        account state."""
        self._check_open()
        self._no_open_group()
        from raptorbt.strategy import _recovery

        return _recovery.state_digest(self)

    # -- internals -----------------------------------------------------------

    def _init_recovery(self, journal, epoch, execution, execution_kind, entries, states, replay):
        if epoch is not None and (isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0):
            raise InvalidParameter(
                f"epoch must be a non-negative integer, got {epoch!r}",
                code="invalid_execution", operation=self._name, parameter="epoch", value=epoch,
            )
        if replay is not None and journal is None:
            raise InvalidParameter(
                "a replay needs the journal to check it against",
                code="invalid_checkpoint", operation=self._name, parameter="journal",
            )
        self._namespace = None
        if execution_kind == "adapter":
            # The engine names its orders s{k}-{n} in every run, so two runs on one
            # account send the same ids. The adapter's namespace is what keeps them
            # apart, and what lets a rebuilt node recognise its own working orders.
            namespace = getattr(execution, "namespace", None)
            if not isinstance(namespace, str) or not namespace:
                raise InvalidParameter(
                    "an execution adapter must expose a non-empty `namespace` (a string that "
                    "marks the orders of one node): the engine's client ids restart at s0-0 "
                    "every run, and a rebuilt node finds its own orders by the namespace",
                    code="adapter_namespace_missing",
                    operation=self._name,
                    parameter="execution",
                    value=type(execution).__name__,
                )
            self._namespace = namespace
            if epoch is not None and not getattr(execution, "can_fence", False):
                raise InvalidParameter(
                    "epoch= fences a stale node through the adapter's fence(epoch), and only an "
                    f"adapter with can_fence true enforces it; {type(execution).__name__} does not "
                    "(a BrokerAdapter needs a host lease=), so the epoch would protect nothing",
                    code="fence_unsupported",
                    operation=self._name,
                    parameter="epoch",
                    value=epoch,
                )
        self._journal = journal
        self._epoch = epoch
        self._execution = execution
        self._adapter_kind = execution_kind
        self._state = "live"
        self._replaying = replay is not None
        self._pending: dict = {}
        self._on_discrepancy = None
        if states is not None:
            if len(states) != len(entries):
                raise InvalidParameter(
                    f"strategy_states holds {len(states)} entries for {len(entries)} strategies",
                    code="invalid_checkpoint", operation=self._name, parameter="strategy_states",
                )
            for (strategy, _), state in zip(entries, states):
                if state is not None:
                    strategy.load_state(state)
        if replay is not None:
            journal.expect(replay)

    def _journal_start(self, symbols, strategies, states) -> None:
        if self._journal is None:
            return
        import json

        self._journal.append(
            "start",
            0,
            {
                "node": self._name,
                "symbols": list(symbols),
                "strategies": strategies,
                "execution": self._adapter_kind or "in_process",
                "protective": self._protective,
                "namespace": self._namespace,
                "strategy_states": None
                if states is None
                else [None if st is None else json.dumps(st, sort_keys=True) for st in states],
            },
            source="engine",
        )
        # The epoch is a record of its own, not part of what a replay compares in
        # `start`: a node that takes over (restore with a newer epoch) must be
        # able to replay the journal of the node it replaces.
        if self._epoch is not None:
            self._journal.append("epoch", 0, {"epoch": self._epoch}, source="engine")

    def _claim_epoch(self, epoch: int, recorded: int | None = None) -> None:
        """Take ``epoch`` after a replay: tell the venue now (an older node is
        fenced at its next request) and write it down. Raises ``fenced`` if a
        newer epoch already owns the account."""
        if not getattr(self._execution, "can_fence", False):
            raise InvalidParameter(
                "epoch= fences a stale node through the adapter's fence(epoch), and only an "
                f"adapter with can_fence true enforces it; {type(self._execution).__name__} does not "
                "(a BrokerAdapter needs a host lease=), so the epoch would protect nothing",
                code="fence_unsupported", operation=self._name, parameter="epoch", value=epoch,
            )
        previous = self._epoch
        self._rt.set_epoch(epoch)
        self._epoch = epoch
        try:
            self._rt.heartbeat()
        except BaseException:
            self._epoch = previous
            raise
        if epoch != recorded:
            self._journal_host("epoch", {"epoch": epoch})

    def _register_exit_orders(self, request) -> None:
        """Make the orders a venue-held exit rule will fire known to the books."""
        k = request.strategy_index
        rows = [
            (cid, self._index_of[sym], side, float(qty))
            for cid, (sym, side, qty) in zip(request.exit_client_ids, request.legs)
        ]
        if rows:
            self._rt.register_venue_orders(k, rows)

    def _journal_host(self, kind: str, payload) -> None:
        if self._journal is not None:
            self._journal.append(kind, self._rt.clock()[0], payload, source="host")

    def _resume_record(self, payload) -> None:
        self._journal_host("resume", payload)

    def _no_open_group(self) -> None:
        if self._rt.group_open:
            raise StateError(
                "a timestamp group is open: flush() it before reconcile, resume, health, a checkpoint or a state digest",
                code="group_open",
                operation=self._name,
            )

    def _guard_push(self, what) -> None:
        if self._epoch is not None and self._rt.fenced:
            raise StateError(
                "a newer execution epoch owns the account; this node sends nothing more",
                code="fenced",
                operation=self._name,
            )
        if self._state != "live" and what != "report":
            if self._state == "adding":
                # A strategy asked for an instrument (see ``_add_requested``): one test of the
                # state the guard already makes, so a node that is never asked pays nothing.
                if what != "instrument":
                    self._add_requested()
                return
            raise StateError(
                "the node was rebuilt from its journal and has not resumed: reconcile with "
                "the venue, then resume()",
                code="recovering",
                operation=self._name,
            )

    def _index(self, symbol: str, call: str) -> int:
        """The node's index of ``symbol``; a symbol it does not trade is a structured error."""
        try:
            return self._index_of[symbol]
        except KeyError:
            raise UnknownSymbol(
                f"{self._name}.{call}: unknown symbol {symbol!r}",
                code="unknown_symbol",
                operation=f"{self._name}.{call}",
                parameter="symbol",
                value=symbol,
                valid=list(self._index_of),
            ) from None

    def _check_open(self, push: bool = False) -> None:
        if self._finished:
            what = "stream" if self._extra else "node"
            raise StateError(
                f"{what} is finished; create a new one",
                code="node_finished",
                operation="TickStrategyStream" if self._extra else "LiveNode",
            )
        if push and not self._replaying:
            self._guard_push(push)
        elif push and self._state == "adding" and push != "instrument":
            self._add_requested()  # replaying: the guard above is not run


class _VenueExits:
    """The adapter's optional exit-rule calls (``place_exit_rule``, ``cancel_exit_rule``,
    ``replace_exit_rule``) as a node offers them to a strategy's exit rules: through the node's
    fence, its journal and its replay.

    Each request and what became of it are journaled like any adapter call (the request before
    the adapter is called). While a journal is replayed, or warm-up bars are replayed muted, the
    adapter is not called: a replay answers with the result the journal holds. A rule the venue
    takes gets its exit orders registered at the node, so the fills of the exits the venue fires
    are booked as the orders they are; a rule it drops or changes ends them.

    An answer that is neither yes nor no (an error after the request left, or a request whose
    result was never written) is *unconfirmed*: the rule may be held. Its exit orders are
    registered anyway, so a fill is booked rather than refused as unknown, and the order ids are
    marked unconfirmed for the next reconciliation."""

    def __init__(self, node, adapter) -> None:
        self._node = node
        self._adapter = adapter

    # -- the three calls ----------------------------------------------------------------

    def place_exit_rule(self, request) -> str:
        """Returns ``"held"`` or ``"unconfirmed"``; raises ``Unsupported``/``OrderRefused`` when
        the venue will not hold it (the engine then watches the rule)."""
        from raptorbt.strategy.exits import ExitRuleUnconfirmed

        try:
            status = self._call(
                "place_exit_rule", _exit_request_value(request), request.rule_id,
                lambda: self._adapter.place_exit_rule(request),
            )  # fmt: skip
        except ExitRuleUnconfirmed:
            self._register_unconfirmed(request)  # a fill of the exit it may hold must be booked
            raise
        if status == "ok":
            self._node._register_exit_orders(request)
            return "held"
        self._register_unconfirmed(request)
        return "unconfirmed"

    def _needs(self, name: str) -> None:
        from raptorbt.execution import Unsupported

        if not callable(getattr(self._adapter, name, None)):
            raise Unsupported(f"{type(self._adapter).__name__} has no {name}: a held exit rule cannot be changed through it")

    def cancel_exit_rule(self, request) -> None:
        self._needs("cancel_exit_rule")
        self._call(
            "cancel_exit_rule", {"rule_id": request.rule_id, "exit_client_ids": list(request.exit_client_ids)},
            request.rule_id, lambda: self._adapter.cancel_exit_rule(request),
        )  # fmt: skip
        self._end_orders(request)

    def replace_exit_rule(self, old, new) -> None:
        from raptorbt.strategy.exits import ExitRuleUnconfirmed

        self._needs("replace_exit_rule")

        try:
            status = self._call(
                "replace_exit_rule",
                {"old": _exit_request_value(old), "new": _exit_request_value(new)},
                new.rule_id, lambda: self._adapter.replace_exit_rule(old, new),
            )  # fmt: skip
        except ExitRuleUnconfirmed:
            # Either rule may be the one the venue holds: both stay registered, both unconfirmed.
            self._register_unconfirmed(new)
            self._node._rt.mark_unconfirmed(list(old.exit_client_ids))
            raise
        self._end_orders(old)
        if status == "ok":
            self._node._register_exit_orders(new)
        else:
            self._register_unconfirmed(new)

    # -- the journaled call ------------------------------------------------------------

    def _call(self, op: str, value: dict, rule_id: str, call) -> str:
        """Fence, journal the request, call (or replay) the adapter, journal the result.

        Returns ``"ok"``, or ``"unconfirmed"`` when a replay cannot say how a request ended.
        Raises ``Unsupported``/``OrderRefused`` for a definite no, and
        :class:`~raptorbt.strategy.exits.ExitRuleUnconfirmed` when a live call failed in a way
        that leaves the venue's state unknown."""
        from raptorbt.execution import OrderRefused, Unsupported
        from raptorbt.strategy.exits import ExitRuleUnconfirmed

        node = self._node
        if node._muted_now:
            raise Unsupported("warm-up bars hold no exit at the venue")
        if node._epoch is not None:
            node._rt.heartbeat()  # raises `fenced` when a newer epoch owns the account
        j = node._journal
        ts = node._rt.clock()[0]
        if j is not None:
            j.append(
                "adapter_request", ts, {"op": op, "request": value},
                source="engine", client_order_id=rule_id,
            )  # fmt: skip
        error = None
        if node._replaying:
            status, message = _replayed_result(j)
        else:
            try:
                call()
                status, message = "ok", ""
            except Unsupported as e:
                status, message = "unsupported", str(e)
            except OrderRefused as e:
                status, message = "refused", str(e)
            except Exception as e:  # noqa: BLE001 - the venue could not be asked, or did not say
                status, message, error = "error", str(e), e
        if j is not None:
            j.append(
                "adapter_result", ts, {"op": op, "status": status, "message": message},
                source="engine", client_order_id=rule_id,
            )  # fmt: skip
        if status == "ok":
            return "ok"
        if status == "unsupported":
            raise Unsupported(message)
        if status == "refused":
            raise OrderRefused(message)
        # error, or a result that was never written: the venue may have acted
        if node._replaying:
            return "unconfirmed"
        raise ExitRuleUnconfirmed(message or "the venue did not answer") from error

    def _register_unconfirmed(self, request) -> None:
        node = self._node
        node._register_exit_orders(request)
        node._rt.mark_unconfirmed(list(request.exit_client_ids))

    def _end_orders(self, request) -> None:
        """The venue no longer holds ``request``: the orders it would have fired are over."""
        node = self._node
        k = request.strategy_index
        if k is not None and request.exit_client_ids:
            node._rt.end_venue_orders(k, list(request.exit_client_ids))
            for cid in request.exit_client_ids:
                node._rt.confirm(cid)


def _exit_request_value(request) -> dict:
    value = {
        "rule_id": request.rule_id,
        "kind": request.kind,
        "legs": [[sym, side, float(qty)] for sym, side, qty in request.legs],
        "rule": {k: float(v) for k, v in request.rule.items()},
        "exit_client_ids": list(request.exit_client_ids),
        "strategy_index": request.strategy_index,
    }
    if request.generation:
        value["generation"] = request.generation  # written only for a replaced rule: older journals replay
    return value


def _replayed_result(journal):
    """The result the journal holds for the request just replayed. A request whose result was
    never written (the process died between the call and the record) has an outcome nobody
    knows: ``unknown_outcome``, which the venue may have acted on."""
    nxt = journal.peek() if journal is not None else None
    if nxt is not None and nxt.kind == "adapter_result":
        payload = nxt.payload
        status = payload.get("status")
        if not isinstance(status, str):
            raise InvalidData(
                "the journal holds an adapter result with no status",
                code="invalid_checkpoint", operation="LiveNode.restore", parameter="records",
            )
        return status, payload.get("message", "")
    return "unknown_outcome", "the request was written; its result was not"


def _execution_kind(execution, name):
    """The runtime's name for the node's execution: ``None`` for in-process,
    ``"paper"`` or ``"adapter"``."""
    if execution is None or isinstance(execution, SimulatedExecution):
        return None
    if isinstance(execution, PaperExecution):
        return "paper"
    if isinstance(execution, ExecutionAdapter):
        return "adapter"
    raise InvalidParameter(
        "execution must be None, SimulatedExecution(), PaperExecution() or an object with "
        f"submit, cancel and modify, got {type(execution).__name__}",
        code="invalid_execution",
        operation=name,
        parameter="execution",
        value=type(execution).__name__,
        valid=["None", "SimulatedExecution()", "PaperExecution()", "an ExecutionAdapter"],
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
        ensure_prices_finite(f"{symbol} warmup bars", "LiveNode", dict(zip(_BAR_FIELDS, cols)))
        closes = stamps.visible(symbol, ts)
        rows.extend(
            (int(closes[j]), i, j, (i, int(ts[j]), *(float(c[j]) for c in cols)))
            for j in range(len(ts))
        )
    rows.sort(key=lambda r: r[:3])
    return [r[3] for r in rows]


def _seeds(initial_positions, entries):
    """Per strategy, ``{symbol: (quantity, avg_price, timestamp_ns, margin_minor, cost_minor)}``."""
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
            if not np.isfinite(quantity) or quantity == 0 or not avg_price > 0:
                raise InvalidParameter(
                    f"initial_positions[{k}][{symbol!r}] needs a non-zero quantity "
                    f"and a positive avg_price (got {quantity} @ {avg_price})",
                    code="invalid_initial_positions",
                    operation="LiveNode",
                    parameter=f"initial_positions[{k}][{symbol!r}]",
                    value=f"{quantity} @ {avg_price}",
                )
            minor = {}
            for field in ("margin_minor", "cost_minor"):
                value = seed.get(field)
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                    raise InvalidParameter(
                        f"initial_positions[{k}][{symbol!r}][{field!r}] must be a "
                        f"non-negative int of minor units, got {value!r}",
                        code="invalid_initial_positions",
                        operation="LiveNode",
                        parameter=f"initial_positions[{k}][{symbol!r}][{field!r}]",
                        value=value,
                        valid=["a non-negative int"],
                    )
                minor[field] = int(value)
            out[k][symbol] = (
                quantity,
                avg_price,
                int(seed.get("timestamp_ns") or 0),
                minor.get("margin_minor"),
                minor.get("cost_minor"),
            )
    return out


def _adopter(rt, syms, seeds):
    """The callback that adopts a strategy's seeds once its account exists."""
    if not seeds:
        return None

    def adopt(sid):
        for symbol, (quantity, avg_price, ts, margin_minor, cost_minor) in seeds.items():
            rt.adopt_position(
                sid, syms.index(symbol), ts, avg_price, quantity, margin_minor, cost_minor
            )

    return adopt
