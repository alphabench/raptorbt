"""Multi-instrument driver for class-based strategies.

:func:`run_portfolio_strategy` runs one strategy on the shared runtime: it is
:func:`~raptorbt.run_strategies` with a single strategy.

One strategy instance trades N instruments against a single shared account.
Bars from all instruments are merged into one deterministic schedule
(by timestamp, then registration order); ``on_bar`` fires once per event
with ``ctx.symbol`` naming the instrument whose bar just closed. Orders and
closes route to the current symbol by default, or explicitly via
``symbol=``.

Position state follows the single-instrument context's style:
``ctx.position`` / ``ctx.is_flat`` / ``ctx.is_net_long`` /
``ctx.is_net_short`` / ``ctx.net_position`` are properties reading the
CURRENT symbol, so a strategy written against ``StrategyContext``
(``if ctx.position is None``) behaves identically here. Cross-symbol
lookups are explicit methods: ``position_for(symbol)``,
``positions(symbol=None)``, ``net_position_for(symbol)``.

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

from typing import Literal

from raptorbt._raptorbt import BacktestConfig, InstrumentConfig, PortfolioResult
from raptorbt.strategy.base import Strategy
from raptorbt.strategy.multi_runner import run_strategies

def run_portfolio_strategy(
    strategy: Strategy | type[Strategy],
    data: dict[str, dict],
    config: BacktestConfig | None = None,
    directions: dict[str, int] | None = None,
    instruments: dict | None = None,
    instrument_configs: dict[str, InstrumentConfig] | None = None,
    oms_type: str = "netting",
    account_type: str = "cash",
    leverage: float = 1.0,
    bar_stamp: Literal["open", "close"] | None = None,
    bar_ns: int | dict[str, int] | None = None,
) -> PortfolioResult:
    """Run one strategy over N instruments sharing a capital pool.

    ``data`` maps symbol -> dict of OHLCV arrays (``timestamps``/``open``/
    ``high``/``low``/``close``/``volume``). ``instruments`` optionally maps
    symbol -> :class:`InstrumentSpec`. Returns a ``PortfolioResult``:
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

    ``bar_stamp`` says what each symbol's ``timestamps`` mean: ``"close"`` (a
    bar is stamped when it ends) or ``"open"`` (when it starts; ``bar_ns`` is
    then required: the bar width in nanoseconds, or a ``{symbol: ns}`` dict
    covering every symbol). The engine runs on bar close time: the merged
    schedule, fills, ``DAY`` expiry, session square-off and the strategy
    clock use ``ctx.bar_close_ts``, while ``ctx.timestamp`` and
    ``ctx.bar.timestamp`` stay the stamps you passed. Composite bars are
    stamped at their window end, a close time. ``None`` keeps the pre-1.0
    behaviour (composites built as if stamps were open stamps, event order as
    if they were close stamps) and emits a ``DeprecationWarning``; it will be
    required in 1.0.

    This is :func:`~raptorbt.run_strategies` with one strategy, on the shared
    runtime. ``ctx.series()`` returns read-only views cut at the bars that
    have closed, not the full arrays, so a strategy cannot read data from its
    own future.
    """
    return run_strategies(
        [strategy],
        data,
        config,
        directions=directions,
        instruments=instruments,
        instrument_configs=instrument_configs,
        oms_type=oms_type,
        account_type=account_type,
        leverage=leverage,
        bar_stamp=bar_stamp,
        bar_ns=bar_ns,
        _via="run_portfolio_strategy",
    )[0]
