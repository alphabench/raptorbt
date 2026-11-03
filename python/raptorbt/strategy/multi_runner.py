"""Run many strategies over one shared copy of the market data.

:func:`run_strategies` runs a list of strategies: each one gets its own
account, trades only its own symbols, and produces exactly the result its own
:func:`~raptorbt.strategy.portfolio_runner.run_portfolio_strategy` call
would (that function is this one with a single strategy), while the bars, the
aggregated bars and the indicator values they share are computed once. The
per-event loop runs in Rust; this module does the setup and assembles the
results.

Trade ticks, quotes and order-book depth run the same way: ``ticks=`` and
``depth=`` take the arrays :func:`~raptorbt.strategy.tick_runner.run_tick_strategy`
takes, and each strategy gets exactly the result its own ``run_tick_strategy``
call would, hooks included (``on_trade_tick``, ``on_quote``, ``on_order_book``
and the ``ctx.tick`` / ``quote`` / ``book`` / ``best_bid`` / ``best_ask`` /
``last_price`` views).

``ctx.series()`` returns read-only views cut at the latest bar (or, for a
symbol with ticks and no bars, the latest print or quote) the run has reached,
not the full arrays, so a strategy cannot read data from its own future.
``run_portfolio_strategy``, ``run_tick_strategy`` and ``TickStrategyStream``
are thin wrappers over this runtime and behave the same.
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable, Literal

import numpy as np

from raptorbt.errors import (
    InvalidData,
    InvalidParameter,
    InvalidSubscription,
    UnknownSymbol,
)
from raptorbt._raptorbt import (
    AccountResult,
    BarAggregator,
    BacktestConfig,
    CorporateAction,
    InstrumentConfig,
    MarketData,
    PortfolioResult,
    Runtime,
    adjust_history,
)
from raptorbt.strategy._commands import (
    CommandPath,
    PortfolioTarget,
    dispatch_events,
    reject_atr_stops,
    reject_bar_only_settings,
    session_tz_offset_ns,
    config_for_holding,
)
from raptorbt.strategy import api
from raptorbt.strategy import context_host
from raptorbt.strategy import order_check as order_check_mod
from raptorbt.strategy.accounts import (
    AccountSpec,
    check_accounts,
    check_group_sizes,
    plan_batches,
    unpack_entries,
)
from raptorbt.strategy._stamps import BarStamps
from raptorbt.strategy._validate import ensure_prices_finite, ensure_sessions_cover
from raptorbt.strategy.base import Strategy
from raptorbt.strategy.cache import Cache
from raptorbt.strategy.clock import Clock
from raptorbt.strategy.context import (
    Bar,
    BookSnapshot,
    CompositeBar,
    QuoteTick,
    TradeTick,
    WorkingOrder,
)

# Strategies per batch when ``batch_size`` is None; see ``run_strategies``.
_AUTO_BATCH = 250

# Tick columns a caller may pass besides the required timestamps and ltp.
_TICK_OPTIONAL = (
    "bid", "ask", "buy_qty_delta", "sell_qty_delta", "ltq", "bid_qty", "ask_qty", "oi",
)
_DEPTH_FIELDS = ("bid_prices", "bid_sizes", "ask_prices", "ask_sizes")


def _as_arrays(arrays: dict, symbol: str, call: str) -> dict[str, np.ndarray]:
    out = {
        "timestamps": np.ascontiguousarray(arrays["timestamps"], dtype=np.int64),
    }
    for key in ("open", "high", "low", "close", "volume"):
        out[key] = np.ascontiguousarray(arrays[key], dtype=np.float64)
    n = len(out["timestamps"])
    for key, arr in out.items():
        if len(arr) != n:
            raise ValueError(f"{key} has length {len(arr)}, expected {n}")
    ensure_prices_finite(f"{symbol} bars", call, out)
    return out


def _tick_arrays(symbol, arrays):
    """One symbol's tick arrays, coerced; every error names the symbol and field."""
    where = f"ticks[{symbol!r}]"
    if not isinstance(arrays, dict):
        raise InvalidData(
            f"{where}: expected a dict of arrays, got {type(arrays).__name__}",
            code="invalid_input_type",
            operation="run_strategies",
            parameter=where,
            value=type(arrays).__name__,
            valid=["dict"],
        )
    out = {}
    for key in ("timestamps", "ltp", *_TICK_OPTIONAL):
        value = arrays.get(key)
        if value is None:
            if key in ("timestamps", "ltp"):
                raise InvalidData(
                    f"{where}: missing {key!r}; ticks need 'timestamps' and 'ltp'",
                    code="missing_field",
                    operation="run_strategies",
                    parameter=f"{where}[{key!r}]",
                    valid=["timestamps", "ltp"],
                )
            continue
        try:
            col = np.ascontiguousarray(value, dtype=np.int64 if key == "timestamps" else np.float64)
        except (TypeError, ValueError) as ex:
            raise InvalidData(
                f"{where}: {key!r} is not a numeric array ({ex})",
                code="invalid_array",
                operation="run_strategies",
                parameter=f"{where}[{key!r}]",
            ) from None
        if col.ndim != 1:
            raise InvalidData(
                f"{where}: {key!r} must be 1-D, got {col.ndim}-D",
                code="invalid_array",
                operation="run_strategies",
                parameter=f"{where}[{key!r}]",
                value=col.ndim,
                valid=[1],
            )
        out[key] = col
    n = len(out["timestamps"])
    for key, col in out.items():
        if len(col) != n:
            raise InvalidData(
                f"{where}: {key!r} has length {len(col)}, expected {n} (len of 'timestamps')",
                code="length_mismatch",
                operation="run_strategies",
                parameter=f"{where}[{key!r}]",
                value=len(col),
                valid=[n],
            )
    return out


def _depth_arrays(symbol, arrays):
    """One symbol's depth arrays, coerced; every error names the symbol and field."""
    where = f"depth[{symbol!r}]"
    if not isinstance(arrays, dict):
        raise InvalidData(
            f"{where}: expected a dict of arrays, got {type(arrays).__name__}",
            code="invalid_input_type",
            operation="run_strategies",
            parameter=where,
            value=type(arrays).__name__,
            valid=["dict"],
        )
    out = {}
    for key in ("timestamps", *_DEPTH_FIELDS):
        value = arrays.get(key)
        if value is None:
            raise InvalidData(
                f"{where}: missing {key!r}",
                code="missing_field",
                operation="run_strategies",
                parameter=f"{where}[{key!r}]",
                valid=["timestamps", *_DEPTH_FIELDS],
            )
        try:
            col = np.ascontiguousarray(value, dtype=np.int64 if key == "timestamps" else np.float64)
        except (TypeError, ValueError) as ex:
            raise InvalidData(
                f"{where}: {key!r} is not a numeric array ({ex})",
                code="invalid_array",
                operation="run_strategies",
                parameter=f"{where}[{key!r}]",
            ) from None
        want = 1 if key == "timestamps" else 2
        if col.ndim != want:
            raise InvalidData(
                f"{where}: {key!r} must be {want}-D, got {col.ndim}-D",
                code="invalid_array",
                operation="run_strategies",
                parameter=f"{where}[{key!r}]",
                value=col.ndim,
                valid=[want],
            )
        out[key] = col
    n = len(out["timestamps"])
    for key in _DEPTH_FIELDS:
        if out[key].shape[0] != n:
            raise InvalidData(
                f"{where}: {key!r} has {out[key].shape[0]} rows, expected {n} (len of 'timestamps')",
                code="length_mismatch",
                operation="run_strategies",
                parameter=f"{where}[{key!r}]",
                value=out[key].shape[0],
                valid=[n],
            )
    return out


def _primary_bars(primary_bars):
    if primary_bars is None:
        return None
    ok = (
        isinstance(primary_bars, (tuple, list))
        and len(primary_bars) == 2
        and isinstance(primary_bars[0], int)
        and not isinstance(primary_bars[0], bool)
        and isinstance(primary_bars[1], str)
    )
    if not ok:
        raise InvalidSubscription(
            f"primary_bars must be (step, unit), e.g. (5, 'm'), got {primary_bars!r}",
            code="invalid_primary_bars",
            parameter="primary_bars",
            value=primary_bars,
        )
    try:
        BarAggregator(*primary_bars)
    except ValueError as ex:
        raise InvalidSubscription(
            f"primary_bars {tuple(primary_bars)!r}: {ex}",
            code="invalid_primary_bars",
            parameter="primary_bars",
            value=tuple(primary_bars),
        ) from None
    return tuple(primary_bars)


def _normalise(strategies, data, call="run_strategies"):
    """``[(strategy, symbols)]`` with symbols in ``data`` order."""
    out = []
    seen: set[int] = set()
    for item in strategies:
        symbols = None
        if isinstance(item, tuple):
            if len(item) != 2:
                raise InvalidParameter(
                    "a strategy entry is a strategy or a (strategy, symbols) pair",
                    code="invalid_strategy_entry",
                    operation=call,
                    parameter="strategies",
                    value=item,
                )
            item, symbols = item
        if isinstance(item, type):
            item = item()
        if not isinstance(item, Strategy):
            raise InvalidParameter(
                f"strategy must be a Strategy instance or subclass, got {type(item).__name__}",
                code="invalid_strategy",
                operation=call,
                parameter="strategies",
                value=type(item).__name__,
                valid=["Strategy instance", "Strategy subclass"],
            )
        if id(item) in seen:
            raise InvalidParameter(
                "the same strategy instance cannot run twice in one call",
                code="duplicate_strategy",
                operation=call,
                parameter="strategies",
                value=type(item).__name__,
            )
        seen.add(id(item))
        if symbols is None:
            wanted = list(data)
        else:
            wanted = [symbols] if isinstance(symbols, str) else list(symbols)
            unknown = [s for s in wanted if s not in data]
            if unknown:
                raise UnknownSymbol(
                    f"{call}: no data for " + ", ".join(repr(s) for s in unknown),
                    code="unknown_symbol",
                    operation=call,
                    parameter="symbols",
                    value=unknown,
                    valid=list(data),
                )
            keep = set(wanted)
            wanted = [s for s in data if s in keep]
        if not wanted:
            raise InvalidParameter(
                "data must contain at least one symbol",
                code="no_symbols",
                operation=call,
                parameter="data",
            )
        out.append((item, wanted))
    if not out:
        raise InvalidParameter(
            f"{call}: strategies is empty; pass at least one strategy",
            code="empty_strategies",
            operation=call,
            parameter="strategies",
            value=[],
        )
    return out


def _add_strategy(rt, strategy, symbols, adopt=None, stacklevel=4, account=None, host=None):
    """Reset ``strategy``, register it with ``rt`` over ``symbols`` and run its
    ``on_start``; returns ``(sid, ctx, view)``.

    ``adopt(sid)`` runs after the strategy's account exists and before
    ``on_start``, so a position it adopts is there for ``on_start`` to see.
    ``account`` is the runtime's id of the account group the strategy joins.
    """
    strategy.drain_orders()
    strategy.drain_commands()
    strategy._bar_subscriptions = []
    strategy._indicators = []
    strategy.clock = Clock()
    strategy.cache = Cache()
    sid, ctx, view = rt.add_strategy(symbols, account)
    api.begin(strategy, ctx, symbols, host=host)
    if adopt is not None:
        adopt(sid)
    strategy.on_start(ctx)
    mask = api.finish(strategy)
    # One clock per symbol: a timer set in on_start belongs to every
    # symbol, not to whichever one's event crosses the threshold first.
    clocks = [strategy.clock.clone_schedule() for _ in symbols]
    # The Python command path stays reachable while the native one settles.
    commands = (
        CommandPath(strategy, PortfolioTarget(view, ctx))
        if os.environ.get("RAPTORBT_PY_COMMANDS") == "1"
        else None
    )
    unrouted = rt.register_streams(
        sid,
        strategy,
        ctx,
        commands,
        clocks,
        mask,
        [(s[0], s[1], s[2] if len(s) > 2 else 0.0) for s in strategy._bar_subscriptions],
        list(strategy._indicators),
    )
    if unrouted:
        warnings.warn(
            "register_indicator() without symbol= in a portfolio run feeds the "
            "indicator every symbol's bars interleaved; pass symbol= to route it",
            stacklevel=stacklevel,
        )
    return sid, ctx, view


def _run_group(
    entries,
    market,
    used,
    stamps,
    config,
    *,
    directions,
    instruments,
    instrument_configs,
    oms_type,
    account_type,
    leverage,
    keep_curves,
    share,
    want_report,
    primary_bars=None,
    stacklevel=4,
    names=None,
    accounts=None,
    corporate_actions=(),
    adjustment_policy=None,
    order_check=None,
    host=None,
):
    """One runtime over ``entries``; returns ``(results, report, account_results)``.

    ``names`` holds the account of each entry (or ``None``) and ``accounts``
    the specs; every strategy of an account is in ``entries``. The
    ``strategies`` of an account result are positions in ``entries``.
    """
    config = config_for_holding(config, [strategy for strategy, _ in entries], "run_strategies")
    rt = Runtime(
        market,
        [(directions or {}).get(s, 1) for s in used],
        [(instrument_configs or {}).get(s) for s in used],
        [(instruments or {}).get(s) for s in used],
        config,
        oms_type,
        account_type,
        leverage,
        session_tz_offset_ns(config),
        stamps.aggregator_stamp,
        share,
        keep_curves,
        Bar,
        CompositeBar,
        WorkingOrder,
        dispatch_events,
        TradeTick,
        QuoteTick,
        BookSnapshot,
    )
    try:
        order_check_mod.install(rt, order_check, "run_strategies")
        if primary_bars is not None:
            rt.set_primary_bars(*primary_bars)
        gids = {}
        for name in dict.fromkeys(n for n in names or () if n is not None):
            spec = accounts[name]
            gids[name] = rt.add_account(
                spec.initial_capital,
                spec.account_type,
                spec.leverage,
                sum(1 for n in names if n == name),
                max_drawdown_pct=spec.max_drawdown_pct,
                max_daily_loss=spec.max_daily_loss,
                max_daily_loss_pct=spec.max_daily_loss_pct,
                max_gross_exposure=spec.max_gross_exposure,
                max_positions=spec.max_positions,
            )
        setups = []
        for k, (strategy, symbols) in enumerate(entries):
            gid = gids.get(names[k]) if names else None
            sid, ctx, _ = _add_strategy(
                rt, strategy, symbols, stacklevel=stacklevel, account=gid, host=host
            )
            setups.append((strategy, sid, ctx))
        if adjustment_policy is not None:
            rt.set_corporate_actions(list(corporate_actions), adjustment_policy)
        rt.finalize()
        rt.run()
        report = rt.memory_report() if want_report else None
        results = []
        for strategy, sid, ctx in setups:
            strategy.on_stop(ctx)
            results.append(rt.finish(sid))
        account_results = {name: rt.finish_account(gid, name) for name, gid in gids.items()}
        return results, report, account_results
    finally:
        rt.close()


def _lifecycle(actions, policy, universe, tick_symbols, call, live=False):
    """The declared corporate actions and policy, checked against the data."""
    actions = list(actions or ())
    if policy is not None and policy not in ("raw", "adjusted"):
        raise InvalidParameter(
            f"adjustment_policy must be 'raw' or 'adjusted', got {policy!r}",
            code="invalid_adjustment_policy",
            operation=call,
            parameter="adjustment_policy",
            value=policy,
            valid=["raw", "adjusted"],
        )
    for action in actions:
        if not isinstance(action, CorporateAction):
            raise InvalidParameter(
                f"corporate_actions holds CorporateAction objects, got {type(action).__name__}",
                code="invalid_corporate_action",
                operation=call,
                parameter="corporate_actions",
                value=type(action).__name__,
            )
        if action.instrument not in universe:
            raise UnknownSymbol(
                f"{call}: corporate action on {action.instrument!r} but there is no data for it",
                code="unknown_symbol",
                operation=call,
                parameter="corporate_actions",
                value=action.instrument,
                valid=list(universe),
            )
    if actions and policy is None:
        raise InvalidParameter(
            f"{call}: corporate_actions need an adjustment_policy: 'raw' (prices as traded, "
            "positions rescaled at the ex-date) or 'adjusted' (history rescaled onto the "
            "basis after the last split)",
            code="invalid_adjustment_policy",
            operation=call,
            parameter="adjustment_policy",
            valid=["raw", "adjusted"],
        )
    if policy == "adjusted":
        for action in actions:
            if action.instrument in tick_symbols:
                why = (
                    "bars already pushed, and the indicator state, composite bars and orders "
                    "built on them, are on the old basis when the ex-date arrives; restating "
                    "them would mean replaying the whole feed. Use 'raw' (positions are "
                    "rescaled at the ex-date), or push history already adjusted and declare "
                    "no action inside it"
                    if live
                    else "cannot put the ticks on one basis; use 'raw'"
                )
                raise InvalidParameter(
                    f"{call}: adjustment_policy='adjusted' is refused for "
                    f"{action.instrument!r}: {why}",
                    code="invalid_adjustment_policy",
                    operation=call,
                    parameter="adjustment_policy",
                    value=policy,
                    valid=["raw"],
                )
    return actions, policy


def _adjusted(symbol, actions, arrays, call):
    """``arrays`` with the bars before each split or bonus on the later basis."""
    if not any(a.instrument == symbol and a.ratio is not None for a in actions):
        return arrays
    return _as_arrays(adjust_history(symbol, actions, arrays), symbol, call)


def _merge_reports(reports, strategies, size):
    """One report for a tiled run: the store once, the rest from the peak batch."""
    peak = max(reports, key=lambda r: r["total"] - r["shared_store"])
    merged = dict(peak)
    merged["strategies"] = strategies
    merged["events"] = sum(r["events"] for r in reports)
    merged["batches"] = len(reports)
    merged["batch_size"] = max(r["strategies"] for r in reports)
    return merged


def run_strategies(
    strategies: Iterable,
    data: dict[str, dict] | None = None,
    config: BacktestConfig | None = None,
    *,
    ticks: dict[str, dict] | None = None,
    depth: dict[str, dict] | None = None,
    primary_bars: tuple[int, str] | None = None,
    directions: dict[str, int] | None = None,
    instruments: dict | None = None,
    instrument_configs: dict[str, InstrumentConfig] | None = None,
    oms_type: str = "netting",
    account_type: str = "cash",
    leverage: float = 1.0,
    accounts: dict[str, AccountSpec] | None = None,
    bar_stamp: Literal["open", "close"] | None = None,
    bar_ns: int | dict[str, int] | None = None,
    keep_curves: bool = True,
    share: bool = True,
    batch_size: int | None = None,
    memory_report: bool = False,
    corporate_actions: Iterable[CorporateAction] | None = None,
    adjustment_policy: Literal["raw", "adjusted"] | None = None,
    order_check=None,
    bindings: dict[str, str] | None = None,
    _via: str | None = None,
) -> (
    list[PortfolioResult]
    | tuple[list[PortfolioResult], dict]
    | tuple[list[PortfolioResult], dict[str, AccountResult]]
    | tuple[list[PortfolioResult], dict, dict[str, AccountResult]]
):
    """Run many strategies over shared market data.

    ``strategies`` holds, per strategy, a :class:`Strategy` instance, a
    ``Strategy`` subclass (instantiated without arguments) or a
    ``(strategy, symbols)`` pair, optionally with a third item
    ``{"account": name}``, or a dict ``{"strategy": s, "symbols": [...],
    "account": name}`` (``symbols`` and ``account`` are optional). Without symbols a strategy trades every
    symbol in ``data`` and ``ticks``, in that order: the symbols of ``data``
    first, then those that only have ticks. Results come back in input order.

    ``ticks`` maps symbol to tick arrays, in the format of ``run_tick_strategy``:
    ``timestamps`` and ``ltp`` are required; ``bid``, ``ask``,
    ``buy_qty_delta``, ``sell_qty_delta``, ``ltq``, ``bid_qty``, ``ask_qty`` and
    ``oi`` are optional. ``depth`` maps a symbol that has ticks to its
    snapshots (``timestamps`` and ``(n, levels)`` arrays ``bid_prices``,
    ``bid_sizes``, ``ask_prices``, ``ask_sizes``). ``data`` and ``ticks`` may
    both be given, and a symbol may be in either or both; a symbol with both
    sees its bars and its ticks on one tape, in time order. Quotes and books are
    observation only; ``primary_bars=(step, unit)`` aggregates the prints of
    every symbol with ticks into bars that fire ``on_bar`` and feed indicators
    registered without a ``stream_id`` (orders still match against ticks). Bad
    tick or depth input raises ``ValueError`` naming the symbol and the field.
    ATR-based stops and targets are refused on a symbol that has ticks or depth,
    as ``run_tick_strategy`` refuses them: tick events carry no bar series to
    compute ATR from. ``bar_stamp`` and ``bar_ns`` apply to the symbols that
    have bars.

    The contract: for every strategy ``s`` over ``syms``,
    ``run_strategies([(s, syms)], data, **kw)[0]`` equals
    ``run_portfolio_strategy(s, {k: data[k] for k in data if k in syms}, **kw)``,
    and ``run_strategies([(s, syms)], ticks=ticks, depth=depth, primary_bars=pb, **kw)[0]``
    equals ``run_tick_strategy(s, {k: ticks[k] for k in ticks if k in syms}, depth=..., primary_bars=pb, **kw)``
    in every observable (trades, orders, curves, metrics, halting, per-instrument
    summaries and the sequence and values of hook calls), and so does the entry
    for ``s`` when other strategies run in the same call. Strategies are
    isolated: each has its own account, so no cash, margin, risk limit or halt
    is shared, unless it is put on an account (see ``accounts``). ``directions``, ``instruments``, ``instrument_configs``,
    ``oms_type``, ``account_type``, ``leverage``, ``config``, ``bar_stamp`` and
    ``bar_ns`` apply to every strategy, as they would in separate calls.

    ``ctx.series(symbol)`` returns read-only views truncated to the latest bar
    the run has reached (empty in ``on_start``), not the full arrays.

    ``accounts`` maps a name to an :class:`~raptorbt.AccountSpec`; strategies
    that name it share one account: its cash, margin and account-level risk
    limits. A strategy with no account keeps its own, as without ``accounts``.
    Unknown names raise ``unknown_account``. The strategies of an account step
    in the order they appear in ``strategies``, which decides who gets scarce
    cash; a strategy on an account ignores the call's ``initial_capital``,
    ``account_type``, ``leverage`` and account-level limits, and its equity
    curve is attributed to it (see :mod:`raptorbt.strategy.accounts` for the
    rules, and for why a group of one is bit-identical to an isolated
    strategy). With ``accounts`` the return value is ``(results,
    account_results)``, where ``account_results`` maps each account that has
    strategies to its :class:`~raptorbt.AccountResult`; without it the return
    value is as it was. With ``memory_report=True`` as well it is ``(results,
    report, account_results)``.

    ``share=True`` (the default) computes aggregated bars once per distinct
    aggregation and links indicators registered through
    ``register_indicator`` to one shared state per distinct definition.
    ``share=False`` gives every strategy its own runtime, with private
    aggregators and indicators; results are identical, only slower.

    ``batch_size`` runs the strategies in consecutive batches of that many, in
    input order. An account is never split across batches: its strategies run
    together, in the batch of its first strategy, and that batch grows to hold
    them (an account holds at most 500 strategies; a larger one raises
    ``account_too_large``). With ``share=False`` an account is one runtime. The market data is built once and shared by every batch; each
    batch gets its own aggregators, indicator states and books, runs the whole
    tape, and is finished and freed before the next one starts. Batching
    changes neither results nor what is shared: aggregated bars and
    indicators are shared among the strategies of a batch, the columns among
    all batches, and strategies stay isolated either way. What it buys is
    memory locality: the loop visits every strategy subscribed to an event, so
    when all of them are live at once their state outgrows the CPU caches and
    the cost per strategy-event climbs. Each batch's ``on_start`` runs when
    the batch starts, not before the first event of the call. ``None`` picks
    250 strategies per batch: in ``benches/runtime/density.py`` sweeps (K =
    3000, 5 instruments per strategy) the cost per strategy-event was lowest
    and flat from 250 to 500 and rose steeply above 1000, where the live state
    (about 11 KB per strategy) no longer fits the CPU caches. It is a fixed
    count, measured at 5 instruments per strategy. ``batch_size`` must be at least 1 and is moot with
    ``share=False``, which is a batch of one.

    ``corporate_actions`` are the splits, bonuses and cash dividends of the
    run (:class:`~raptorbt.CorporateAction`), data you supply and the engine
    never infers. They need ``adjustment_policy``, which says what the bars
    are: ``"raw"`` (prices as traded: a split shows as a jump, indicators see
    it, and the engine rescales each position at the ex-date, so cash stays
    exact) or ``"adjusted"`` (the engine puts every bar before a split or
    bonus on the basis after it, so indicators see a continuous series; it
    trades that one basis and does not rescale positions again; a dividend's
    per-unit amount is restated to it). The policy is part of an indicator's
    identity. An action on a symbol that has no data raises ``unknown_symbol``.
    ``"adjusted"`` needs bars (it refuses a symbol with ticks that has actions).
    A dividend is credited on its ``pay_ts`` (default the ex-date) to the
    holder at the ex-date; a short pays it. See ``docs/architecture/lifecycle.md``.

    ``order_check`` is the host's pre-submit order check, shared with
    :class:`~raptorbt.LiveNode` so one rule refuses the same orders in a
    backtest and live; see :mod:`raptorbt.strategy.order_check`.

    ``bindings`` maps a requirements role name to the symbol it means
    (``{"index": "NSE:NIFTY"}``); with ``instruments=`` it is what
    ``ctx.role``, ``ctx.instrument`` and ``ctx.option_chain`` answer from, as on a
    :class:`~raptorbt.LiveNode` (see :mod:`raptorbt.strategy.context_host`).

    ``keep_curves=False`` drops the equity, drawdown and returns curves from
    the results (the final equity value stays as a one-element curve); trades,
    orders and metrics are unchanged. The curves are still sampled during the
    run, since the metrics are computed from them.

    With ``memory_report=True`` the return value is ``(results, report)``
    where ``report`` maps category to heap bytes held by the Rust runtime
    (``shared_store``, ``aggregators``, ``indicators``, ``index``, ``atr``,
    ``engine_per_strategy``, ``portfolio_per_strategy``, ``curves``) plus
    ``*_bytes_per_strategy`` figures and sharing counts. Python objects are
    not counted. ``shared_store`` is counted once. The other categories are those of
    the batch with the most live state (the peak), measured after its last
    event and before its results are built, so ``total`` is the peak
    footprint; ``strategies`` is the number of strategies in the call, and
    ``batches`` and ``batch_size`` (the largest batch) describe the tiling.
    """
    # ``_via`` names the legacy entry point (``run_portfolio_strategy``,
    # ``run_tick_strategy``) that wraps this call: its name leads every
    # warning and error, and its extra frame is skipped when attributing them.
    call = _via or "run_strategies"
    extra = 1 if _via else 0
    if batch_size is not None and (
        isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
    ):
        raise InvalidParameter(
            f"batch_size must be a positive integer or None, got {batch_size!r}",
            code="invalid_batch_size",
            operation="run_strategies",
            parameter="batch_size",
            value=batch_size,
            valid=["a positive integer", None],
        )
    for name, given in (("data", data), ("ticks", ticks), ("depth", depth)):
        if given is not None and not isinstance(given, dict):
            raise InvalidData(
                f"{name} must be a dict of symbol -> arrays, got {type(given).__name__}",
                code="invalid_input_type",
                operation="run_strategies",
                parameter=name,
                value=type(given).__name__,
                valid=["dict"],
            )
    if not data and not ticks:
        which = "ticks" if ticks is not None and data is None else "data"
        raise InvalidParameter(
            f"{which} must contain at least one symbol",
            code="no_symbols",
            operation="run_strategies",
            parameter=which,
        )
    accounts = check_accounts(accounts, call)
    strategies, names = unpack_entries(strategies, accounts, call)
    check_group_sizes(names, call)
    data, ticks, depth = data or {}, ticks or {}, depth or {}
    unknown = [s for s in depth if s not in ticks]
    if unknown:
        raise InvalidData(
            "depth needs ticks for the same symbol: "
            + ", ".join(f"depth[{s!r}] has no ticks[{s!r}]" for s in unknown),
            code="depth_without_ticks",
            operation="run_strategies",
            parameter="depth",
            value=unknown,
            valid=list(ticks),
        )
    primary_bars = _primary_bars(primary_bars)
    if primary_bars is not None and not ticks:
        raise InvalidSubscription(
            "primary_bars aggregates trade prints, but no ticks were given",
            code="primary_bars_without_ticks",
            operation="run_strategies",
            parameter="primary_bars",
            value=primary_bars,
        )
    # Bar symbols first, in data order, then symbols that only have ticks.
    universe = {**dict.fromkeys(data), **dict.fromkeys(ticks)}
    entries = _normalise(strategies, universe, call)
    if not entries:
        return ([], {}) if memory_report else []
    used = [s for s in universe if any(s in syms for _, syms in entries)]
    tick_symbols = [s for s in used if s in ticks]
    reject_atr_stops(config, instrument_configs, tick_symbols)
    bar_symbols = [s for s in used if s in data]
    reject_bar_only_settings(config, [s for s in bar_symbols if s not in ticks], call)
    stamps = BarStamps.resolve(
        bar_stamp, bar_ns, bar_symbols, call, warn=bool(bar_symbols), stacklevel=4 + extra
    )
    arrays = {s: _as_arrays(data[s], s, call) for s in bar_symbols}
    ensure_sessions_cover(arrays, instruments, call)
    corporate_actions, adjustment_policy = _lifecycle(
        corporate_actions, adjustment_policy, universe, tick_symbols, call
    )
    if adjustment_policy == "adjusted":
        for s in bar_symbols:
            arrays[s] = _adjusted(s, corporate_actions, arrays[s], call)
    tick_arrays = {s: _tick_arrays(s, ticks[s]) for s in tick_symbols}
    depth_arrays = {s: _depth_arrays(s, depth[s]) for s in tick_symbols if s in depth}
    market = MarketData(
        used,
        [arrays.get(s) for s in used],
        [stamps.offset(s) for s in used],
        [tick_arrays.get(s) for s in used],
        [depth_arrays.get(s) for s in used],
    )
    kwargs = dict(
        directions=directions,
        instruments=instruments,
        instrument_configs=instrument_configs,
        oms_type=oms_type,
        account_type=account_type,
        leverage=leverage,
        keep_curves=keep_curves,
        share=share,
        want_report=memory_report,
        primary_bars=primary_bars,
        stacklevel=4 + extra,
        corporate_actions=corporate_actions,
        adjustment_policy=adjustment_policy,
        order_check=order_check,
        host=context_host.build(instruments, bindings, entries, universe, call),
    )
    size = 1 if not share else (batch_size or _AUTO_BATCH)
    results: list = [None] * len(entries)
    reports, account_results = [], {}
    for batch in plan_batches(names, size):
        done, report, owned = _run_group(
            [entries[i] for i in batch],
            market,
            used,
            stamps,
            config,
            names=[names[i] for i in batch],
            accounts=accounts,
            **kwargs,
        )
        for i, result in zip(batch, done):
            results[i] = result
        for name, account in owned.items():
            account.strategies = [batch[k] for k in account.strategies]
            account_results[name] = account
        if report is not None:
            reports.append(report)
    out: tuple = (results,)
    if memory_report:
        out += (_merge_reports(reports, len(entries), size),)
    if accounts is not None:
        out += (account_results,)
    return out if len(out) > 1 else results
