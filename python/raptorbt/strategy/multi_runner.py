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

:func:`run_strategies_windowed` runs the same strategies over the same tape fed in windows, so
that what is held in memory is bounded by the window and not by the run, and gives the results
:func:`run_strategies` gives.
"""

from __future__ import annotations

import os
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np

from raptorbt import errors
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
    where = f"data[{symbol!r}]"
    if not isinstance(arrays, dict):
        raise InvalidData(
            f"{where}: expected a dict of arrays, got {type(arrays).__name__}",
            code="invalid_input_type",
            operation=call,
            parameter=where,
            value=type(arrays).__name__,
            valid=["dict"],
        )
    out = {}
    for key in ("timestamps", "open", "high", "low", "close", "volume"):
        value = arrays.get(key)
        if value is None:
            raise InvalidData(
                f"{where}: missing {key!r}; bars need timestamps, open, high, low, close and volume",
                code="missing_field",
                operation=call,
                parameter=f"{where}[{key!r}]",
                valid=["timestamps", "open", "high", "low", "close", "volume"],
            )
        try:
            col = np.ascontiguousarray(value, dtype=np.int64 if key == "timestamps" else np.float64)
        except (TypeError, ValueError) as ex:
            raise InvalidData(
                f"{where}: {key!r} is not a numeric array ({ex})",
                code="invalid_array",
                operation=call,
                parameter=f"{where}[{key!r}]",
            ) from None
        if col.ndim != 1:
            raise InvalidData(
                f"{where}: {key!r} must be 1-D, got {col.ndim}-D",
                code="invalid_array",
                operation=call,
                parameter=f"{where}[{key!r}]",
                value=col.ndim,
                valid=[1],
            )
        out[key] = col
    n = len(out["timestamps"])
    for key, arr in out.items():
        if len(arr) != n:
            raise InvalidData(
                f"{where}: {key!r} has length {len(arr)}, expected {n} (len of 'timestamps')",
                code="length_mismatch",
                operation=call,
                parameter=f"{where}[{key!r}]",
                value=len(arr),
                valid=[n],
            )
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


def _primary_bars(primary_bars, call="run_strategies"):
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
            operation=call,
            parameter="primary_bars",
            value=primary_bars,
            valid=["(step, unit) with step >= 1 and unit a bar unit, e.g. (5, 'm')"],
        )
    try:
        BarAggregator(*primary_bars)
    except ValueError as ex:
        raise InvalidSubscription(
            f"primary_bars {tuple(primary_bars)!r}: {ex}",
            code="invalid_primary_bars",
            operation=call,
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
    try:
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
    except BaseException:
        # the run never gets to release it: its context would wait for the collector, on any thread
        api.release(strategy)
        raise
    return sid, ctx, view


def _open_group(
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
    primary_bars=None,
    stacklevel=4,
    names=None,
    accounts=None,
    corporate_actions=(),
    adjustment_policy=None,
    order_check=None,
    order_check_scope="all",
    host=None,
    trade_from=None,
    call="run_strategies",
):
    """One runtime over ``entries``, set up (every ``on_start`` has run) and not yet fed;
    returns a :class:`_Group`.

    ``names`` holds the account of each entry (or ``None``) and ``accounts``
    the specs; every strategy of an account is in ``entries``.
    """
    config = config_for_holding(config, [strategy for strategy, _ in entries], call)
    rt = errors.call_as(
        call,
        Runtime,
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
    group = _Group(rt)
    try:
        order_check_mod.install(rt, order_check, call, order_check_scope)
        if trade_from is not None:
            rt.set_trade_from(trade_from)
        if primary_bars is not None:
            rt.set_primary_bars(*primary_bars)
        for name in dict.fromkeys(n for n in names or () if n is not None):
            spec = accounts[name]
            group.gids[name] = rt.add_account(
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
        for k, (strategy, symbols) in enumerate(entries):
            gid = group.gids.get(names[k]) if names else None
            sid, ctx, view = _add_strategy(
                rt, strategy, symbols, stacklevel=stacklevel, account=gid, host=host
            )
            group.setups.append((strategy, sid, ctx))
        if adjustment_policy is not None:
            rt.set_corporate_actions(list(corporate_actions), adjustment_policy)
        rt.finalize()
        return group
    except BaseException as exc:
        # the frames the exception carries hold contexts: keep the lines, drop the locals
        traceback.clear_frames(exc.__traceback__)
        group.close()
        raise


class _Group:
    """One runtime and the strategies on it: what a run feeds and then finishes."""

    def __init__(self, rt):
        self.rt = rt
        self.setups: list[tuple[Any, int, Any]] = []
        self.gids: dict = {}

    def finish(self, want_report):
        """``(results, report, account_results)``; the runtime is closed either way."""
        try:
            rt = self.rt
            report = rt.memory_report() if want_report else None
            results = []
            for strategy, sid, ctx in self.setups:
                strategy.on_stop(ctx)
                results.append(rt.finish(sid))
            account_results = {
                name: rt.finish_account(gid, name) for name, gid in self.gids.items()
            }
            return results, report, account_results
        except BaseException as exc:
            traceback.clear_frames(exc.__traceback__)
            raise
        finally:
            self.close()

    def close(self):
        """Release the runtime and the strategies' contexts, which would otherwise wait for the
        collector, on any thread."""
        self.rt.close()
        for strategy, _, _ in self.setups:
            api.release(strategy)
        # this frame outlives the call in an exception's traceback
        self.setups.clear()


def _run_group(entries, market, used, stamps, config, *, want_report, **kwargs):
    """One runtime over ``entries``; returns ``(results, report, account_results)``.

    ``names`` holds the account of each entry (or ``None``) and ``accounts``
    the specs; every strategy of an account is in ``entries``. The
    ``strategies`` of an account result are positions in ``entries``.
    """
    group = _open_group(entries, market, used, stamps, config, **kwargs)
    try:
        group.rt.run()
    except BaseException as exc:
        traceback.clear_frames(exc.__traceback__)
        group.close()
        raise
    return group.finish(want_report)


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


def _check_trade_from(trade_from, call):
    if trade_from is not None and (isinstance(trade_from, bool) or not isinstance(trade_from, int)):
        raise InvalidParameter(
            f"trade_from must be an event time in integer nanoseconds since the epoch or None, "
            f"got {trade_from!r}",
            code="invalid_trade_from",
            operation=call,
            parameter="trade_from",
            value=trade_from,
            valid=["integer nanoseconds since the epoch", None],
        )


def _check_batch_size(batch_size, call):
    if batch_size is not None and (
        isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
    ):
        raise InvalidParameter(
            f"batch_size must be a positive integer or None, got {batch_size!r}",
            code="invalid_batch_size",
            operation=call,
            parameter="batch_size",
            value=batch_size,
            valid=["a positive integer", None],
        )


def _prepare(call, strategies, accounts, universe, primary_bars):
    """What ``run_strategies`` and ``run_strategies_windowed`` check the same way before they
    look at data: the accounts, the strategy entries over ``universe`` and the primary bars.
    Returns ``(accounts, entries, names, primary_bars, used)``; ``used`` is the symbols of
    ``universe`` some strategy trades, in its order."""
    accounts = check_accounts(accounts, call)
    strategies, names = unpack_entries(strategies, accounts, call)
    check_group_sizes(names, call)
    primary_bars = _primary_bars(primary_bars, call)
    entries = _normalise(strategies, universe, call)
    used = [s for s in universe if any(s in syms for _, syms in entries)]
    return accounts, entries, names, primary_bars, used


def _merge_concurrent(reports, strategies, size):
    """One report for runtimes that are all alive at once (a windowed run feeds every batch each
    window): what each holds adds up, so ``batch_size`` does not lower the peak here."""
    merged: dict = {}
    for r in reports:
        for key, value in r.items():
            if isinstance(value, int) and not isinstance(value, bool):
                merged[key] = merged.get(key, 0) + value
    merged["strategies"] = strategies
    for key, base in (
        ("engine_bytes_per_strategy", "engine_per_strategy"),
        ("portfolio_bytes_per_strategy", "portfolio_per_strategy"),
        ("curve_bytes_per_strategy", "curves"),
    ):
        merged[key] = merged[base] // max(strategies, 1)
    merged["batches"] = len(reports)
    merged["batch_size"] = max(r["strategies"] for r in reports)
    merged["concurrent"] = True
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
    order_check_scope: Literal["all", "typed"] = "all",
    bindings: dict[str, str | Sequence[str]] | None = None,
    trade_from: int | None = None,
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
    have bars; ``bar_stamp`` is required when any symbol has them
    (``missing_bar_stamp``).

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
    250 strategies per batch: in density sweeps (K =
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
    backtest and live; see :mod:`raptorbt.strategy.order_check`. With a check installed,
    ``enter()`` and ``close_position()`` are refused (``order_check_intent``), since they carry
    no order to show it; ``order_check_scope="typed"`` declares that the check judges typed
    orders only, and intents then pass unseen by it. Use it when the check judges forms an
    intent cannot have (an order kind, a validity, ``post_only``).

    ``bindings`` maps a requirements role name to the symbol it means
    (``{"index": "NSE:NIFTY"}``), or, for a :class:`~raptorbt.strategy.requirements.Universe`
    role, to the list of symbols chosen for it (``{"basket": ["A", "B", "C"]}``); with
    ``instruments=`` it is what ``ctx.role``, ``ctx.instrument`` and ``ctx.option_chain`` answer
    from, as on a :class:`~raptorbt.LiveNode` (see :mod:`raptorbt.strategy.context_host`).

    ``trade_from`` (ns since the epoch, an event time) is the first moment the strategies may
    trade. The bars before it are run in full, so the strategies' hooks and indicators warm up on
    them, but every order decided at an event before it is refused with the reason ``warmup``:
    a typed order is recorded as rejected (the strategy hears ``on_order_rejected`` with
    ``reject_reason == "warmup"``), an ``enter()`` intent is a refused entry, and the refusal is
    journaled and counted like any other. Orders decided at ``trade_from`` or later are not
    affected. ``None`` (the default) trades from the first bar. A host that wants the history
    without the trades uses this instead of an ``order_check`` that also refuses ``enter()``
    and ``close_position()``.

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
    _check_trade_from(trade_from, call)
    _check_batch_size(batch_size, "run_strategies")
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
    if primary_bars is not None and not ticks:
        # checked once the primary bars themselves are
        _primary_bars(primary_bars, call)
        raise InvalidSubscription(
            "primary_bars aggregates trade prints, but no ticks were given",
            code="primary_bars_without_ticks",
            operation="run_strategies",
            parameter="primary_bars",
            value=primary_bars,
        )
    # Bar symbols first, in data order, then symbols that only have ticks.
    universe = {**dict.fromkeys(data), **dict.fromkeys(ticks)}
    accounts, entries, names, primary_bars, used = _prepare(
        call, strategies, accounts, universe, primary_bars
    )
    if not entries:
        return ([], {}) if memory_report else []
    tick_symbols = [s for s in used if s in ticks]
    bar_symbols = [s for s in used if s in data]
    reject_atr_stops(config, instrument_configs, tick_symbols)
    reject_bar_only_settings(config, [s for s in bar_symbols if s not in ticks], call)
    stamps = BarStamps.resolve(
        bar_stamp, bar_ns, bar_symbols, call, require=bool(bar_symbols)
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
        [stamps.width(s) for s in used],
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
        order_check_scope=order_check_scope,
        host=context_host.build(instruments, bindings, entries, universe, call),
        trade_from=trade_from,
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


# ---------------------------------------------------------------------------
# A run fed in windows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """One piece of the tape of :func:`run_strategies_windowed`.

    ``bars``, ``ticks`` and ``depth`` map a symbol to the arrays :func:`run_strategies` takes for
    it (``data``, ``ticks`` and ``depth``), holding the rows of the window only. The first window
    names every symbol, with arrays that may be empty; that declares what each carries (bars,
    ticks with their columns, depth), and every later window carries the same. ``end_ns`` is the
    last event time the window covers (event time: a bar stamped at its open counts at its
    close); every row of the window is stamped at or before it. Left out, it is the time of the
    window's last event.

    Windows come in ascending order and never cut a timestamp: every event of a window is
    stamped strictly after the last event of the one before it, and a window that does not is
    refused with ``window_overlap``.
    """

    bars: Mapping[str, dict] | None = None
    ticks: Mapping[str, dict] | None = None
    depth: Mapping[str, dict] | None = None
    end_ns: int | None = None


@dataclass(frozen=True)
class WindowDone:
    """What ``on_window`` is told once a window has been run.

    ``index`` counts windows from 0. ``end_ns`` is the window's ``end_ns``, or the time of its
    last event (the end of the window before, for a window with no event). ``events`` is how many
    events the window delivered, summed over the runtimes of the call (one, unless
    ``batch_size`` splits the strategies); ``total_events`` the same since the first window.
    """

    index: int
    end_ns: int | None
    events: int
    total_events: int


def _window_error(call, message, code="invalid_window", parameter="windows", value=None, valid=None):
    cls = InvalidParameter if code == "invalid_window" else InvalidData
    return cls(
        message,
        code=code,
        operation=call,
        parameter=parameter,
        value=value,
        valid=valid or ["a Window with the arrays of run_strategies, in ascending time"],
    )


_UNIT_NS = {"ms": 10**6, "s": 10**9, "m": 60 * 10**9, "h": 3600 * 10**9, "d": 86_400 * 10**9}


def _primary_widths(used, bar_symbols, tick_symbols, offsets, stamps, primary_bars):
    """The width of the primary bars of each symbol, in nanoseconds (``0`` unknown), for an
    expression on a coarser clock. A windowed run has no stored bars to read it from:

    - bars stamped at their open: the stamp's own width;
    - bars stamped at their close: ``bar_ns``, which is required for such an expression;
    - prints cut into ``primary_bars=(step, unit)``: that span.
    """
    span = 0
    if primary_bars is not None:
        step, unit = primary_bars
        span = step * _UNIT_NS.get(unit, 0)
    widths = []
    for s in used:
        width = offsets.get(s, 0) or stamps.width(s)
        if not width and s in tick_symbols and s not in bar_symbols:
            width = span
        widths.append(width)
    return widths


def _event_span(arrays, offsets, ticks, depth):
    """The first and last event time a window's arrays hold, or ``(None, None)``."""
    first, last = [], []
    for sym, a in arrays.items():
        ts = a["timestamps"]
        if len(ts):
            first.append(int(ts[0]) + offsets.get(sym, 0))
            last.append(int(ts[-1]) + offsets.get(sym, 0))
    for group in (ticks, depth):
        for a in group.values():
            ts = a["timestamps"]
            if len(ts):
                first.append(int(ts[0]))
                last.append(int(ts[-1]))
    return (min(first), max(last)) if first else (None, None)


def run_strategies_windowed(
    strategies: Iterable,
    windows: Iterable[Window],
    config: BacktestConfig | None = None,
    *,
    symbols: Sequence[str],
    history: int | Literal["declared", "all"] = "declared",
    on_window: Callable[[WindowDone], None] | None = None,
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
    order_check_scope: Literal["all", "typed"] = "all",
    bindings: dict[str, str | Sequence[str]] | None = None,
    trade_from: int | None = None,
) -> (
    list[PortfolioResult]
    | tuple[list[PortfolioResult], dict]
    | tuple[list[PortfolioResult], dict[str, AccountResult]]
    | tuple[list[PortfolioResult], dict, dict[str, AccountResult]]
):
    """Run many strategies over market data fed in windows.

    The same strategies over the same tape as :func:`run_strategies`, fed a window at a time:
    the books, indicators, aggregated bars, orders, fills and equity curves are the whole run's,
    and what is dropped is each window's rows once it is run. For a strategy whose logic reads
    at most the rows it declared (below), the trades, orders, curves, metrics, halts, ``on_*``
    calls and ``ctx.idx`` are those of one :func:`run_strategies` call with all the data, bit for
    bit, whatever the cuts. The curves grow with the run (about 32 bytes per timestamp per
    strategy), as in a batch run.

    ``windows`` yields :class:`Window` objects in ascending time, none cutting a timestamp (a
    window that does is refused, ``window_overlap``). ``symbols`` lists every symbol the windows
    carry, in the order of ``data`` in :func:`run_strategies` (a strategy that names none trades
    all of them); a window that carries a symbol outside it is refused (``window_schema``), a
    symbol of it that no strategy trades is ignored. The first window says what each carries
    (bars, ticks with their columns, depth) and the rest match it. ``on_window`` is called with a
    :class:`WindowDone` after each window.

    ``history`` says which trailing rows of an instrument ``ctx.series`` shows a strategy:

    - ``"declared"`` (the default): a strategy declares how many rows it reads of a symbol with
      ``ctx.keep_history(N, symbol=...)``, and ``ctx.series`` shows it exactly the last ``N``
      (fewer at the start), whatever else is held. A strategy that declares nothing for a
      symbol gets every row of it, so what it reads is the batch run's; that is decided before
      the first window and costs the memory of the run for that symbol. ``ctx.data.bar``,
      ``ctx.data.history`` and the structure exits read through ``ctx.series`` and so count as
      readers. ``memory_report=True`` lists under ``"history"`` which symbols were kept whole,
      for which strategies, and which were declared.
    - ``"all"``: every row of every symbol, for every strategy, whatever it declared, and
      whatever it declares later.
    - an integer ``N``: every strategy that declared nothing for a symbol is given the last
      ``N`` rows of it, which can differ from the batch run's series if it reads further back.

    What a strategy declared is its own contract: ``ctx.series`` shows it exactly the last ``N``
    rows, and ``ctx.data.history`` refuses a count above ``N`` (``invalid_history``), but code that
    reads further back through ``ctx.series`` itself gets the ``N`` rows and no error, and can
    decide differently from the batch run. Declare what the strategy reads. ``ctx.keep_history``
    called after the first window is refused (``window_state``) if it asks for more rows than are
    still held, since released rows do not come back; declare in ``on_start``. A declaration made
    with ``subscribe(symbol, data_type="quote"|"trade"|"book", history=N)`` is for that symbol's
    ticks and depth and does not cap its bars.

    ``trade_from`` is as in :func:`run_strategies`: the bars before it are run in full and every
    order decided before it is refused with the reason ``warmup``.

    Not taken, because it needs rows the run has not been given yet: ATR-based stops and targets
    (``invalid_config``), ``adjustment_policy="adjusted"`` with a split or a bonus
    (``invalid_adjustment_policy``), and an expression that uses ``at(...)`` without the width
    of the primary bars, which no stored bars give here: pass ``bar_ns`` (``expr_unsupported_here``).
    The width of the primary bars comes from the stamp (open), from
    ``bar_ns`` (close, as in :func:`run_strategies`, which uses it too when given and else
    reads the most common gap of the stored bars) or from ``primary_bars`` (prints).

    The other keywords are those of :func:`run_strategies`. ``batch_size`` runs the strategies
    in batches whose runtimes are all alive and fed every window, so it does not lower the peak
    memory here: each batch holds its own history of every symbol it trades, and the memory report
    adds the batches up (``live_rings``).
    """
    call = "run_strategies_windowed"
    _check_trade_from(trade_from, call)
    _check_batch_size(batch_size, call)
    if not (
        history in ("declared", "all")
        or (isinstance(history, int) and not isinstance(history, bool) and history >= 1)
    ):
        raise InvalidParameter(
            f"history must be 'declared', 'all' or a positive number of rows, got {history!r}",
            code="invalid_history",
            operation=call,
            parameter="history",
            value=history,
            valid=["a positive integer", "declared", "all"],
        )
    if on_window is not None and not callable(on_window):
        raise _window_error(
            call, f"on_window must be a callable or None, got {type(on_window).__name__}",
            parameter="on_window", value=type(on_window).__name__, valid=["a callable", None],
        )
    if isinstance(symbols, str) or not symbols or len(set(symbols)) != len(list(symbols)):
        raise InvalidParameter(
            f"{call}: symbols lists each symbol the windows carry once",
            code="no_symbols",
            operation=call,
            parameter="symbols",
            value=symbols if isinstance(symbols, str) else list(symbols or ()),
            valid=["a list of symbols, each once"],
        )
    symbols = list(symbols)
    universe = dict.fromkeys(symbols)
    accounts, entries, names, primary_bars, used = _prepare(
        call, strategies, accounts, universe, primary_bars
    )
    if not entries:
        return ([], {}) if memory_report else []

    windows_iter = iter(windows)
    first = next(windows_iter, None)
    if first is None:
        raise _window_error(
            call, "windows is empty: a run needs at least one window", parameter="windows", value=[]
        )
    first = _checked_window(first, 0, call)

    # What each symbol carries is what the first window says.
    bar_symbols = [s for s in used if s in (first.bars or {})]
    tick_symbols = [s for s in used if s in (first.ticks or {})]
    depth_symbols = [s for s in used if s in (first.depth or {})]
    bare = [s for s in used if s not in bar_symbols and s not in tick_symbols]
    if bare:
        raise _window_error(
            call,
            "the first window names no bars or ticks for "
            + ", ".join(repr(s) for s in bare)
            + ": it declares what each symbol carries, and an empty array will do",
            code="window_schema",
            parameter="windows",
            value=bare,
            valid=["every symbol of the run in the first window, with arrays that may be empty"],
        )
    nodepth = [s for s in depth_symbols if s not in tick_symbols]
    if nodepth:
        raise InvalidData(
            "depth needs ticks for the same symbol: "
            + ", ".join(f"depth[{s!r}] has no ticks[{s!r}]" for s in nodepth),
            code="depth_without_ticks",
            operation=call,
            parameter="depth",
            value=nodepth,
            valid=tick_symbols,
        )
    if primary_bars is not None and not tick_symbols:
        raise InvalidSubscription(
            "primary_bars aggregates trade prints, but no ticks were given",
            code="primary_bars_without_ticks",
            operation=call,
            parameter="primary_bars",
            value=primary_bars,
        )
    reject_atr_stops(config, instrument_configs, tick_symbols)
    reject_bar_only_settings(config, [s for s in bar_symbols if s not in tick_symbols], call)
    stamps = BarStamps.resolve(bar_stamp, bar_ns, bar_symbols, call, require=bool(bar_symbols))
    corporate_actions, adjustment_policy = _lifecycle(
        corporate_actions, adjustment_policy, universe, tick_symbols, call
    )
    offsets = {s: stamps.offset(s) for s in bar_symbols}
    known = set(symbols)

    def arrays_of(window):
        for kind, given in (("bars", window.bars), ("ticks", window.ticks), ("depth", window.depth)):
            stray = [s for s in (given or {}) if s not in known]
            if stray:
                raise _window_error(
                    call,
                    f"{kind} for {stray!r}, which are not in symbols",
                    code="window_schema",
                    parameter=kind,
                    value=stray,
                    valid=list(symbols),
                )
        bars = {s: _as_arrays(a, s, call) for s, a in (window.bars or {}).items() if s in used}
        late = [s for s in bars if s not in bar_symbols]
        if late:
            raise _window_error(
                call,
                f"bars for {late!r}, which the first window declared without bars",
                code="window_schema",
                parameter="bars",
                value=late,
                valid=["what the first window declared for each symbol"],
            )
        ensure_sessions_cover(bars, instruments, call)
        ticks = {s: _tick_arrays(s, a) for s, a in (window.ticks or {}).items() if s in used}
        depth = {s: _depth_arrays(s, a) for s, a in (window.depth or {}).items() if s in used}
        return bars, ticks, depth

    first_arrays = arrays_of(first)
    tick_columns = {s: list(a) for s, a in first_arrays[1].items()}
    widths = _primary_widths(used, bar_symbols, tick_symbols, offsets, stamps, primary_bars)
    index = {s: i for i, s in enumerate(used)}

    # The ring is as long as the run's `history` says, or two rows; what a strategy declares with
    # `ctx.keep_history` grows it, and what none declared is kept whole.
    market = MarketData.windowed(
        used,
        [stamps.offset(s) for s in used],
        # at least two rows: a fill hook that is blind to its instant reads the instrument's
        # previous bar, which is then still held
        max(2, history) if isinstance(history, int) else 2,
        [s in bar_symbols for s in used],
        [tick_columns.get(s) for s in used],
        [s in depth_symbols for s in used],
        widths,
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
        primary_bars=primary_bars,
        stacklevel=4,
        corporate_actions=corporate_actions,
        adjustment_policy=adjustment_policy,
        order_check=order_check,
        order_check_scope=order_check_scope,
        host=context_host.build(instruments, bindings, entries, universe, call),
        trade_from=trade_from,
        call=call,
    )
    size = 1 if not share else (batch_size or _AUTO_BATCH)
    groups: list[tuple[list[int], _Group]] = []
    plan: dict = {"mode": history, "kept_whole": {}, "declared": {}}
    try:
        for batch in plan_batches(names, size):
            group = _open_group(
                [entries[i] for i in batch],
                market,
                used,
                stamps,
                config,
                names=[names[i] for i in batch],
                accounts=accounts,
                **kwargs,
            )
            groups.append((batch, group))
            for k, (_, sid, _) in enumerate(group.setups):
                if isinstance(history, int):
                    group.rt.declare_rows(sid, history)
                group.rt.keep_all_rows(sid, history == "all")
                for symbol, rows in group.rt.history_plan(sid):
                    who = batch[k]
                    if rows is None:
                        plan["kept_whole"].setdefault(symbol, []).append(who)
                    else:
                        plan["declared"].setdefault(symbol, {})[who] = rows

        total, prev_end, k = 0, None, 0
        window, arrays = first, first_arrays
        first = first_arrays = None  # what a window holds is freed once it is run
        while True:
            done, prev_end = _feed_window(call, window, arrays, k, offsets, prev_end, index, groups)
            total += done
            if on_window is not None:
                on_window(WindowDone(k, prev_end, done, total))
            window = arrays = None
            window = next(windows_iter, None)
            if window is None:
                break
            k += 1
            window = _checked_window(window, k, call)
            arrays = arrays_of(window)
    except BaseException as exc:
        traceback.clear_frames(exc.__traceback__)
        if isinstance(exc, errors.RaptorBTError) and str(exc.operation).startswith("Runtime."):
            exc.operation = call  # the native method names itself; the caller called this
        for _, group in groups:
            group.close()
        raise

    results: list = [None] * len(entries)
    reports, account_results = [], {}
    try:
        for batch, group in groups:
            done, report, owned = group.finish(memory_report)
            for i, result in zip(batch, done):
                results[i] = result
            for name, account in owned.items():
                account.strategies = [batch[k] for k in account.strategies]
                account_results[name] = account
            if report is not None:
                reports.append(report)
    except BaseException:
        for _, group in groups:
            group.close()
        raise
    out: tuple = (results,)
    if memory_report:
        merged = _merge_concurrent(reports, len(entries), size)
        merged["history"] = plan
        out += (merged,)
    if accounts is not None:
        out += (account_results,)
    return out if len(out) > 1 else results


def _checked_window(window, k, call):
    if not isinstance(window, Window):
        raise _window_error(
            call,
            f"windows yields Window objects, got {type(window).__name__} at index {k}",
            value=type(window).__name__,
        )
    if window.end_ns is not None and (
        isinstance(window.end_ns, bool) or not isinstance(window.end_ns, int)
    ):
        raise _window_error(
            call,
            f"window {k}: end_ns is an event time in integer nanoseconds, got {window.end_ns!r}",
            parameter="end_ns",
            value=window.end_ns,
            valid=["integer nanoseconds since the epoch", None],
        )
    return window


def _feed_window(call, window, arrays, k, offsets, prev_end, index, groups):
    """Check one window against the one before it and feed it to every runtime; returns
    ``(events delivered, the window's end: its end_ns, else its last event time)``."""
    bars, ticks, depth = arrays
    first_ts, last_ts = _event_span(bars, offsets, ticks, depth)
    end = window.end_ns if window.end_ns is not None else last_ts
    if prev_end is not None and (
        (first_ts is not None and first_ts <= prev_end) or (end is not None and end <= prev_end)
    ):
        raise _window_error(
            call,
            f"window {k} starts at {first_ts} and ends at {end}, but the window before it ended "
            f"at {prev_end}: a window may not cut a timestamp or overlap the one before it",
            code="window_overlap",
            parameter="windows",
            value=k,
            valid=["windows in ascending time, each starting after the end of the one before"],
        )
    if window.end_ns is not None and last_ts is not None and last_ts > window.end_ns:
        raise _window_error(
            call,
            f"window {k} holds an event at {last_ts}, after its end_ns {window.end_ns}",
            code="window_overlap",
            parameter="end_ns",
            value=window.end_ns,
            valid=["an end_ns at or after every event of the window"],
        )
    delivered = 0
    for _, group in groups:
        try:
            delivered += group.rt.push_window(
                {index[s]: a for s, a in bars.items()},
                {index[s]: a for s, a in ticks.items()},
                {index[s]: a for s, a in depth.items()},
            )
        except errors.RaptorBTError as error:
            # the native method names itself; the caller called this
            error.operation = call
            raise
    return delivered, end if end is not None else prev_end
