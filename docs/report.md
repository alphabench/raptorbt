# The full report

`raptorbt.full_report(equity_ts, equity, trades, orders, ...)` and `result.full_report(equity_ts)`
(on a `BacktestResult` or `PortfolioResult`) return every figure a desk reads, each one present or
explained. See `raptorbt.report` for the shape of the returned dict.

A figure that cannot be computed is `None` in `report` and named in `report_unavailable` as
`"CODE: reason"`. The codes: `RUINED`, `TOO_FEW_SAMPLES`, `ZERO_VARIANCE`, `NO_DOWNSIDE`,
`NO_DRAWDOWN`, `NO_TRADES`, `NO_WINS`, `NO_LOSSES`, `NO_BENCHMARK`, `BENCHMARK_SPARSE`,
`NOT_RULE_BASED`, `POOLED_POSITION`, `NO_GROSS_PNL`. A NaN or a silent zero never stands for a
missing figure. Every ratio is a fraction.

## One definition of each metric

The report does not carry its own mean, deviation or drawdown. Volatility, Sharpe and Sortino come
from `StreamingMetrics`, and the maximum drawdown and its length from `DrawdownTracker`: the
accumulators that fill `BacktestMetrics`. What the report adds is what they do not hold: drawdown
episodes, calendar returns, cost shares, exposure, a benchmark's beta, and attribution.

Two things to know when comparing the report with `result.metrics`:

- the metrics average over `result.returns()`, which carries a leading zero for the first bar; the
  report averages over the curve's `n - 1` returns. They agree to the weight of that one zero.
- `total_return` in the report is over the curve (first sample to last); the metrics' is over the
  initial capital. The first sample of a run is already marked after the first bar's costs.

## Differences from the reference report (intended)

- **Sortino** divides the downside deviation by `n - 1` returns (`StreamingMetrics`), where the reference
  report divided by `n`. There is one Sortino in the engine, and it is the metrics' one. The new
  value is the old times `sqrt((n - 1) / n)`.
- **Maximum drawdown of a curve that starts at or below zero** no longer divides by the zero peak
  (`DrawdownTracker` returned NaN there); such a drawdown is 0 until a positive peak exists.

Everything else is equal to 1e-9 relative on the differential fixture
(`tests/python/differential/test_report_differential.py`), which covers daily and intraday curves,
fees by component, a benchmark, exposure, attribution, and the short, flat and ruined curves.

## Attribution

`attribution=True` (the default on `result.full_report`) adds `attribution` (by instrument, and by
rule when `rule_metrics={"rule_fires": ..., "noops_by_source": ...}` is given) and `trades` (one
row per closed trade, with `open_at_end` true on the row of a position still open when the data
ended, which the run closed at the last mark and the trade statistics leave out). A trade belongs
to the origin of the order that closed it: the filled order on the trade's instrument, on the
closing side, whose last fill is the trade's exit bar. Fees go whole to the closing origin; fills
and turnover to each order's own origin. Slippage is inside the fill price and is not split out.
