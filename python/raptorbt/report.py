"""The full report over a run: every figure a desk reads, each one present or explained.

``result.full_report(equity_ts)`` on a ``BacktestResult`` or a ``PortfolioResult`` (or
``raptorbt.full_report`` on bare arrays) returns a dict:

``report``
    the figures; a figure that cannot be computed is ``None`` here and named in
    ``report_unavailable`` as ``"CODE: reason"`` (never a NaN or a silent zero). Every ratio
    is a fraction (0.0123 is 1.23%).
``drawdown_curve`` / ``drawdowns`` / ``monthly`` / ``yearly`` / ``costs`` / ``benchmark`` / ``counters``
    the curve of drawdowns, the deepest episodes, calendar returns, costs by component,
    the run against a benchmark, and counts.
``attribution`` and ``trades``
    (with ``attribution=True``, which ``result.full_report`` sets) who made the money, by
    rule when ``rule_metrics`` is given and by instrument, and one row per closed trade;
    ``open_at_end`` is true on the row of a position still open when the data ended (closed
    by the run at the last mark), which the trade statistics leave out.

The figures are computed once, in Rust (``metrics::report``), from the same
``StreamingMetrics`` and ``DrawdownTracker`` that fill ``BacktestMetrics``; nothing is
recomputed here. Money is ``float``. A trade's ``realised`` is gross (``pnl + fees``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from raptorbt._raptorbt import BacktestResult, PortfolioResult
from raptorbt._raptorbt import full_report as _full_report
from raptorbt.errors import InvalidData

__all__ = ["full_report"]

def full_report(
    *args: Any, unverified_fee_rows: Sequence[str] | None = None, **kwargs: Any
) -> dict[str, Any]:
    """``raptorbt._raptorbt.full_report`` plus ``unverified_fee_rows``: the fee rows the costs
    were priced on whose rates are not matched to an official publication (a result carries them
    as ``result.unverified_fee_rows``, which ``result.full_report`` passes). Left out, the
    key is absent. The costs of a run that lists any are not
    exchange-accurate."""
    out = _full_report(*args, **kwargs)
    if unverified_fee_rows is not None:
        out["unverified_fee_rows"] = list(unverified_fee_rows)
    return out


def _result_full_report(
    self: BacktestResult, equity_ts: Sequence[int], **kwargs: Any
) -> dict[str, Any]:
    """The full report over this result's equity curve, trades and orders, with the
    ``attribution`` block and the per-trade ``trades`` rows.

    ``equity_ts`` are the curve's timestamps in ns (a result does not carry them: the bar
    timestamps of a single-instrument run, or ``PortfolioSession.equity_timestamps()``).
    ``rule_metrics={"rule_fires": ..., "noops_by_source": ...}`` makes
    ``attribution.by_rule``; without it the run is not rule based and the block says so.
    Every other keyword is ``raptorbt.full_report``'s."""
    equity = [float(x) for x in self.equity_curve()]
    ts = [int(t) for t in equity_ts]
    if len(ts) != len(equity):
        raise InvalidData(f"{len(ts)} timestamps for an equity curve of {len(equity)} samples",
                          code="length_mismatch", operation="full_report", parameter="equity_ts",
                          value=len(ts))
    kwargs.setdefault("attribution", True)
    kwargs.setdefault("unverified_fee_rows", self.unverified_fee_rows)
    return full_report(ts, equity, self.trades(), self.orders(), **kwargs)


def _portfolio_full_report(
    self: PortfolioResult, equity_ts: Sequence[int], **kwargs: Any
) -> dict[str, Any]:
    """``BacktestResult.full_report`` over the portfolio's combined result."""
    return _result_full_report(self.result, equity_ts, **kwargs)


BacktestResult.full_report = _result_full_report  # type: ignore[attr-defined]
PortfolioResult.full_report = _portfolio_full_report  # type: ignore[attr-defined]
