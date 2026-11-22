---
name: raptorbt-results
description: >
    Performance analysis, trades and diagnostics.
version: 1.0
api_version: 1.x
depends_on:
    - raptorbt-core
status: stable
---

# RaptorBT Results

## 1. When to use this skill

Load it after any run, to read what happened and to decide whether to believe
it: the headline metrics, the equity and drawdown curves, the trades, the
orders, and the full report with its explained gaps. It is also the place to
learn which figures are in percent, which are fractions, and which population
each statistic counts.

## 2. Domain overview

Every runner returns a `BacktestResult` (or a `PortfolioResult` that holds
one in `.result`, with `per_instrument` summaries and `rejected_entries`).

- `result.metrics`: a `BacktestMetrics`, the headline numbers.
- `result.equity_curve()`: one equity value per bar, after the bar's costs.
  `result.drawdown_curve()`: percent below the running peak, per bar.
  `result.returns()`: per-bar returns as fractions, with a leading `0.0`.
- `result.trades()`: round trips. `pnl` is **net of fees**; `fees` is the whole
  round trip; `exit_reason` says how it ended (`Signal`, `Order`, a stop or
  target, `Settlement`, `EndOfData`, `WindowEnd`). Under a fee schedule, `entry_fee_breakdown`
  and `exit_fee_breakdown` give each leg's charge by the schedule's own
  components (`brokerage`, `stt`, `exchange_txn`, `sebi_fee`, `stamp_duty`,
  `gst`, `ipft`, `dp_charge`) and `total`, the booked cost of the leg; `total`
  differs from the sum of the parts by the rounding of the fee row, so state
  that difference as its own line. The `full_report` trade rows carry the same
  as `entry_fee_parts` and `exit_fee_parts`. They are `None` (empty in the
  rows) when the schedule does not itemise, and for the entry of a position
  built from several fills.
- `result.orders()`: what became of each *typed* order, including those that
  never filled (`status`, `reject_reason`). Intents from `enter()` and
  `close_position()` do not appear here; a refused entry shows in
  `rejected_entries` and `on_order_rejected`.
- `result.full_report(equity_ts)`: every figure a desk reads, each one present
  or explained, as a dict.

**Units.** `BacktestMetrics` and `Trade.return_pct` use **percent**
(`total_return_pct = 7.9` is 7.9%; `max_drawdown_pct`, `win_rate_pct`,
`exposure_pct` likewise). Money is in your account currency. `full_report`
gives **fractions** (`0.079`). Ratios (Sharpe, Sortino, Calmar) are
annualised from bar spacing and the session length, so they depend on
`session_minutes` and `periods_per_year`.

**Event time.** A run is a sequence of bar-close events. Equity sample `i` is
the account marked at the close of bar `i`; trades carry `entry_idx`,
`exit_idx` (bar indices) and `entry_time`, `exit_time` (the engine's close
times, in nanoseconds). A position still open at the last bar is closed there
with `exit_reason == "EndOfData"`, marked at the final close and charged no exit
fee. It appears in `trades()`. `config.with_close_open_at_end()` instead trades
that position out at the same close as `exit_reason == "WindowEnd"`, paying the
exit fee and slippage; it is then a closed trade (for a run that is one window
of a longer history).

## 3. API reference

| Name | What it is |
| --- | --- |
| `BacktestResult.metrics`, `.equity_curve()`, `.drawdown_curve()`, `.returns()`, `.trades()`, `.orders()`, `.halted`, `.halted_at` | The result. The curves are NumPy arrays. |
| `BacktestMetrics` | Fields include `total_return_pct`, `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `max_drawdown_pct`, `max_drawdown_duration`, `win_rate_pct`, `profit_factor`, `expectancy`, `sqn`, `total_trades`, `total_closed_trades`, `total_open_trades`, `open_trade_pnl`, `total_fees_paid`, `total_turnover`, `exposure_pct`, `start_value`, `end_value`, `return_skew`, `return_kurtosis` and more. `.to_dict()` gives 31 of them under display labels. |
| `Trade` | `symbol`, `entry_idx`, `exit_idx`, `entry_price`, `exit_price`, `size`, `direction` (1 or -1), `pnl`, `return_pct`, `fees`, `entry_fees`, `exit_fees`, `exit_reason`, and excursions `mae_pnl`, `mfe_pnl`, `mae_price`, `mfe_price` (`None` where not measured). |
| `Order` records from `orders()` | `client_id`, `symbol`, `side`, `kind`, `status`, `requested_qty` (`None` for a capital fraction), `filled_qty`, `avg_fill_price`, `submitted_idx`, `last_fill_idx`, `reject_reason`. |
| `result.full_report(equity_ts, ...)` | The report dict. `equity_ts` are the curve's timestamps in nanoseconds (the bar timestamps of a single-instrument run). |
| `raptorbt.full_report(equity_ts, equity, trades, orders, ...)` | The same on bare arrays and lists. |
| `raptorbt.report` | The module documenting the dict's shape. |

`full_report` returns `report` (the figures), `report_unavailable`
(`{figure: "CODE: reason"}`), `drawdown_curve`, `drawdowns` (the deepest
episodes), `monthly`, `yearly`, `costs`, `benchmark`, `counters`, `attribution`
and `trades` (one row per trade, with gross `realised` and `net`). The codes are
`RUINED`, `TOO_FEW_SAMPLES`, `ZERO_VARIANCE`, `NO_DOWNSIDE`, `NO_DRAWDOWN`,
`NO_TRADES`, `NO_WINS`, `NO_LOSSES`, `NO_BENCHMARK`, `BENCHMARK_SPARSE`,
`NOT_RULE_BASED`, `POOLED_POSITION`, `NO_GROSS_PNL`.

## 4. Standard workflow

1. Check that the run did what you meant: `total_trades`, the exit reasons, and
   `halted`.
2. Read the headline: `total_return_pct`, `max_drawdown_pct`, `sharpe_ratio`,
   `total_fees_paid`, `exposure_pct`.
3. Reconcile (section 9) before reading ratios.
4. For a desk-grade view call `full_report`; read `report_unavailable` for what
   is missing and why.
5. Look at the trades: excursions (`mae_pnl`, `mfe_pnl`), the worst few, and the
   share of profit from the best few.
6. Re-run with realistic costs, `fill_timing="next_bar_open"` and a different
   period, and compare. A result that does not survive is not a result.

## 5. Examples

Read a run and reconcile it. The pieces must add up, and the units are percent:

```python
import numpy as np
import raptorbt
from raptorbt import BacktestConfig, run_single_backtest

MIN = 60 * 10**9
T0 = 1_767_225_600 * 10**9
n = 600
rng = np.random.default_rng(7)
close = 100.0 + np.cumsum(rng.normal(0.02, 0.4, n))
open_ = np.concatenate([[close[0]], close[:-1]])
ts = T0 + np.arange(1, n + 1, dtype=np.int64) * MIN
high, low, volume = np.maximum(open_, close) + 0.2, np.minimum(open_, close) - 0.2, np.full(n, 1000.0)
fast, slow = raptorbt.sma(close, 5), raptorbt.sma(close, 20)

result = run_single_backtest(
    ts, open_, high, low, close, volume, fast > slow, fast < slow,
    symbol="AAPL", config=BacktestConfig(initial_capital=100_000.0, fees=0.0005),
)
m, trades = result.metrics, result.trades()
equity, dd, returns = result.equity_curve(), result.drawdown_curve(), result.returns()

assert len(equity) == len(dd) == len(returns) == n and returns[0] == 0.0
assert np.isclose(equity[-1], m.end_value)
assert np.isclose(m.total_return_pct, 100 * (m.end_value / m.start_value - 1))   # percent
assert np.isclose(sum(t.pnl for t in trades), m.end_value - m.start_value)       # pnl is net
assert np.isclose(sum(t.fees for t in trades), m.total_fees_paid)
assert np.isclose(dd.max(), m.max_drawdown_pct)
assert np.isclose(np.max(1 - equity / np.maximum.accumulate(equity)) * 100, m.max_drawdown_pct)
assert np.isclose((1 + returns).prod(), equity[-1] / equity[0])

# The last trade was still open: it is in trades(), marked at the final close, no exit fee.
last = trades[-1]
assert last.exit_reason == "EndOfData" and last.exit_fees == 0.0 and last.exit_idx == n - 1
assert m.total_trades == len(trades) == m.total_closed_trades + m.total_open_trades
```

The full report: fractions, and every missing figure explained. With no trades
the metrics and the report agree: `sharpe_ratio`, `profit_factor` and
`win_rate_pct` are `None` (the report adds the reason):

```python
import numpy as np
from raptorbt import run_single_backtest

MIN = 60 * 10**9
T0 = 1_767_225_600 * 10**9
n = 200
close = 100.0 + np.cumsum(np.random.default_rng(7).normal(0.02, 0.4, n))
ts = T0 + np.arange(1, n + 1, dtype=np.int64) * MIN
none = np.zeros(n, dtype=bool)

result = run_single_backtest(ts, close, close + 0.2, close - 0.2, close, np.full(n, 1000.0), none, none)
m0 = result.metrics
assert m0.total_trades == 0 and m0.profit_factor is None and m0.win_rate_pct is None and m0.sharpe_ratio is None

report = result.full_report(ts)
assert report["report"]["profit_factor"] is None and report["report"]["win_rate"] is None
assert report["report_unavailable"]["win_rate"].startswith("NO_TRADES")
assert report["report_unavailable"]["profit_factor"].startswith("NO_LOSSES")
assert report["report_unavailable"]["sharpe"].startswith("ZERO_VARIANCE")
```

Annualisation. The same equity curve gives different Sharpe ratios under
different session lengths, scaling by the square root of the periods per year.
Intraday bars annualise on the data's own trading day (the median observations
per day x 252) unless you name the market's session with `session_minutes`; no
session (NSE or any other) is assumed, and only `session_minutes=0` counts
calendar time:

```python
import numpy as np
import raptorbt
from raptorbt import BacktestConfig, run_single_backtest

MIN = 60 * 10**9
T0 = 1_767_225_600 * 10**9
n = 600
rng = np.random.default_rng(7)
close = 100.0 + np.cumsum(rng.normal(0.02, 0.4, n))
open_ = np.concatenate([[close[0]], close[:-1]])
ts = T0 + np.arange(1, n + 1, dtype=np.int64) * MIN
high, low, volume = np.maximum(open_, close) + 0.2, np.minimum(open_, close) - 0.2, np.full(n, 1000.0)
fast, slow = raptorbt.sma(close, 5), raptorbt.sma(close, 20)

def sharpe(**config):
    r = run_single_backtest(ts, open_, high, low, close, volume, fast > slow, fast < slow,
                            config=BacktestConfig(fees=0.0, **config))
    return r.metrics.sharpe_ratio

unnamed = sharpe()                                      # no session named: the data's own day, 600 bars x 252
nse = sharpe(session_minutes=raptorbt.SESSION_NSE)      # a 375 minute session, 252 days
day = sharpe(session_minutes=1440.0)                    # 1440 bars a day, 252 days
around_the_clock = sharpe(session_minutes=raptorbt.SESSION_CONTINUOUS)   # 365 days
assert np.isclose(unnamed, sharpe(periods_per_year=600 * 252.0)) and not np.isclose(nse, day)
assert np.isclose(around_the_clock / day, np.sqrt(365 / 252))
assert np.isclose(sharpe(periods_per_year=252.0 * 1440), day)   # the explicit form
```

Typed orders in a strategy run. Only typed orders appear in `orders()`, and a
refused one is recorded with its reason. Here a reduce-only buy while long
cannot reduce anything and is refused (`reduce_only`), and the `enter()` intent
leaves no record:

```python
import numpy as np
from raptorbt import BacktestConfig, Strategy, run_strategy_backtest
from raptorbt.strategy import orders

MIN = 60 * 10**9
T0 = 1_767_225_600 * 10**9
n = 30
close = 100.0 + np.arange(n) * 0.1
bars = dict(timestamps=T0 + np.arange(1, n + 1, dtype=np.int64) * MIN,
            open=close, high=close + 0.2, low=close - 0.2, close=close, volume=np.full(n, 1000.0))

class OneOfEach(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 3:
            self.enter(size_frac=0.1)                   # an intent: no Order record
        if ctx.idx == 10:
            self.submit_order(orders.Market(side="buy", units=10.0, reduce_only=True))   # typed, but already long
        if ctx.idx == 20:
            self.close_position()

result = run_strategy_backtest(OneOfEach, **bars, symbol="A", bar_stamp="close",
                               config=BacktestConfig(fees=0.0))
rows = result.orders()
assert len(rows) == 1 and rows[0].submitted_idx == 10
assert (rows[0].status, rows[0].reject_reason) == ("rejected", "reduce_only")
assert rows[0].requested_qty == 10.0                    # units asked for (None for a capital fraction)
assert len(result.trades()) == 1                        # the enter() trade, closed at index 20
```

## 6. Composition

- To trust a result, stress it: costs and fill timing in [execution](../execution/SKILL.md);
  multiple testing and bootstrap in [optimization](../optimization/SKILL.md).
- Several accounts or instruments: `PortfolioResult` is in [portfolio](../portfolio/SKILL.md).
- A run that behaves oddly: [troubleshooting](../troubleshooting/SKILL.md).
- Runnable examples: `examples/single-instrument/sma_cross.py`.

## 7. Constraints and correctness

- **Percent versus fraction.** Metrics and `Trade.return_pct` are percent; the
  full report and the sweep rows of `raptorbt.research.row_from` are fractions. Convert before
  comparing.
- **Net versus gross.** `Trade.pnl` and `Trade.return_pct` are net of fees.
  The report's `trades[i].realised` is gross and `net` equals `pnl`.
- **One definition of a win.** `BacktestMetrics` and `full_report` compute win
  rate, profit factor, payoff and expectancy over **closed** trades only (a
  position still open when the data ended, exit reason `EndOfData`, is not a
  closed trade; a `WindowEnd` exit is), and a trade wins or loses on its P&L **net** of fees. They
  agree on the same run. Sharpe differs slightly: the metrics average over
  `returns()` including its leading zero, the report over the curve's `n - 1`
  returns.
- **Undefined is `None`, not zero.** With no closed trade `win_rate_pct` is
  `None`; with no losing closed trade `profit_factor` is `None`; with no
  variation in the returns (a run that never traded, or one whose trades left
  the curve flat) or fewer than two returns, `sharpe_ratio` is `None`; with no
  downside deviation `sortino_ratio` is `None`. `expectancy` and `sqn` still read `0.0` with no trades, and
  `avg_win_pct` and `avg_loss_pct` are `None` for an empty population. Use the
  report to see the reason for an absent figure.
- **Annualisation.** Sharpe, Sortino and annual figures use `periods_per_year`
  if set, else bars per session times 252 trading days (or 365 for
  `SESSION_CONTINUOUS`). Daily bars of a session market (no weekend bars) count
  252 trading days; only `session_minutes=0` or a seven-day series counts 365. The default session is 375 minutes; bars spanning more
  than that per day inflate the ratio. Set `session_minutes`.
- **Short samples.** A ratio from tens of bars is noise. The report returns
  `None` with `TOO_FEW_SAMPLES` below its minimum.
- **Excursions** (`mae_*`, `mfe_*`) are measured at bar resolution and are
  `None` for synthesised trades (spreads, baskets, pairs). Read
  `mae_mfe_coverage_pct` first.
- **Open positions.** `total_trades` counts the open trade; `total_closed_trades`
  does not. Return and equity include its mark; `open_trade_pnl` isolates it.
- **Orders are typed orders only.** `orders()` is empty for a run that used
  `enter()` and `close_position()` alone; `fills` in the report is then 0.
- **Curves** are per bar, indexed like your input. `drawdown_curve()` is percent
  below the running peak; `returns()[0]` is `0.0`.

## 8. Common mistakes

A result is a result of one run: indexing a trade that does not exist fails.
Check the count first:

<!-- skill: invalid -->
```python
import numpy as np
from raptorbt import run_single_backtest

n = 50
close = 100.0 + np.arange(n) * 0.1
ts = np.arange(1, n + 1, dtype=np.int64) * 60 * 10**9
none = np.zeros(n, dtype=bool)
result = run_single_backtest(ts, close, close, close, close, np.ones(n), none, none)
print(result.trades()[0].pnl)    # IndexError: a run with no trades has no first trade
```

- Reading `total_return_pct = 0.05` as 5%. It is 0.05%.
- Calling a Sharpe of 3 on two weeks of 1 minute bars a finding. Look at the
  sample count and the trade count first.
- Comparing two runs that differ in costs, fill timing or session length as if
  only the strategy changed.
- Summing `realised` from the report and calling it P&L; it is before fees.
- Reading `profit_factor == 0.0` from `BacktestMetrics` as "all losers".
- Taking `max_drawdown_pct` as the worst a strategy can do. It is the worst
  this path did.
- Ignoring the `EndOfData` trade, or dropping it and then not reconciling to
  the equity curve.

## 9. Validation

1. Reconcile: `equity[-1] == end_value`; `sum(t.pnl) == end_value - start_value`;
   `sum(t.fees) == total_fees_paid`; `max(drawdown_curve) == max_drawdown_pct`.
2. Re-run with `fees=0` and `slippage=0` for the gross edge, then with realistic
   costs. If net is negative, there was no edge after costs.
3. Re-run with `fill_timing="next_bar_open"`. A collapse means the signal used
   prices it could not have reached.
4. Count: trades, bars in market (`exposure_pct`), and bars. Fewer than about 30
   trades, or a few trades making most of the profit, is thin evidence.
5. Split the period in halves; both should be broadly alike.
6. Call `full_report` and read `report_unavailable`. Every `None` is something
   you do not know.
7. Run twice and compare trades; they must be identical.

## 10. Related skills

- [Core](../core/SKILL.md)
- [Optimization](../optimization/SKILL.md)
- [Execution](../execution/SKILL.md)
- [Troubleshooting](../troubleshooting/SKILL.md)
