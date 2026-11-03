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

from typing import Literal

from raptorbt._raptorbt import BacktestConfig, InstrumentConfig, PortfolioResult
from raptorbt.strategy.base import Strategy
from raptorbt.strategy.multi_runner import run_strategies


def run_tick_strategy(
    strategy: Strategy | type[Strategy],
    ticks: dict[str, dict],
    config: BacktestConfig | None = None,
    primary_bars: tuple[int, str] | None = None,
    depth: dict[str, dict] | None = None,
    directions: dict[str, int] | None = None,
    instruments: dict | None = None,
    instrument_configs: dict[str, InstrumentConfig] | None = None,
    oms_type: str = "netting",
    account_type: str = "cash",
    leverage: float = 1.0,
    corporate_actions=None,
    adjustment_policy: Literal["raw", "adjusted"] | None = None,
) -> PortfolioResult:
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

    ``corporate_actions`` and ``adjustment_policy`` are those of
    :func:`~raptorbt.run_strategies`: splits, bonuses and cash dividends you
    give, and what the prints mean. Only ``"raw"`` applies to ticks (an
    ``"adjusted"`` policy is refused for a symbol with prints, which cannot be
    restated).

    Everything else — account type, leverage, risk limits, OMS type — behaves
    as in :func:`run_portfolio_strategy`.

    This is :func:`~raptorbt.run_strategies` with one strategy, on the shared
    runtime. ``ctx.series()`` returns read-only views cut at the ticks that
    have arrived, not the full arrays.
    """
    return run_strategies(
        [strategy],
        ticks=ticks,
        depth=depth,
        primary_bars=primary_bars,
        config=config,
        directions=directions,
        instruments=instruments,
        instrument_configs=instrument_configs,
        oms_type=oms_type,
        account_type=account_type,
        leverage=leverage,
        corporate_actions=corporate_actions,
        adjustment_policy=adjustment_policy,
        _via="run_tick_strategy",
    )[0]
