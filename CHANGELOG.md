# Changelog

All notable changes to raptorbt are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-07-21

This release fixes four defects where the engine silently returned wrong
numbers, adds a shared-capital portfolio runner, and introduces the
class-based strategy contract — an event-driven alternative to precomputed
signal arrays.

**Read "Migrating from 0.4.x" before upgrading** — several metrics change
value. Setting `apply_slippage=False, legacy_annualization=True` reproduces
0.4.1 results bit-identically, so stored backtests stay reproducible while you
migrate.

### Fixed

- **Configured slippage was ignored.** `PyBacktestConfig` accepted `slippage`
  and `BacktestConfig` carried it, but `PortfolioEngine::new` hardcoded
  `SlippageModel::None`, so the model was applied as a no-op. Every bar-level
  backtest run with slippage configured executed at zero slippage. A 0.2%
  slippage on a 9-trade fixture now costs ~3.4% of capital; before it cost
  exactly nothing. `run_tick_backtest` was never affected — it reads `slippage`
  directly and always honored it.

- **Sharpe and Sortino were computed from different quantities per runner.**
  `run_single_backtest` annualized *per-bar* returns at 365, while
  `run_basket_backtest`, `run_pairs_backtest`, `run_options_backtest` and
  `run_multi_backtest` annualized *per-trade* returns at 252. The return basis
  is the more serious half: annualizing trade returns assumes one trade per
  trading day, inflating the ratio by roughly `sqrt(n_bars / n_trades)`. On a
  2-trade/500-bar basket, Sharpe drops from 1.175 to 0.218 once corrected. All
  five runners now share one estimator fed per-bar returns.

- **Calmar was meaningless on intraday data.** Years were derived from bar
  count over 365.25, so an 11k-bar 1-minute backtest was scored against ~31
  "years" of compounding. Years now come from elapsed wall-clock timestamps.

- **Undefined ratios crossed to Python as `inf`.** `profit_factor`,
  `payoff_ratio`, `recovery_factor`, `calmar_ratio`, `sortino_ratio` and
  `omega_ratio` divide by a denominator that can legitimately be zero — a
  strategy with no losing trades has an *undefined* profit factor, not an
  infinite one. `json.dumps` writes `float('inf')` as a bare `Infinity` token,
  which is not valid JSON and which `allow_nan=False` rejects outright. These
  are now `Optional[float]` and return `None`.

- **`__version__` had drifted.** `python/raptorbt/__init__.py` hardcoded
  `"0.4.0"` while the crate was at `0.4.1`, so any version check read a value
  that was never released. It now derives from installed package metadata.

- **`cargo test` could not link, so CI never ran it.** pyo3's
  `extension-module` feature was unconditional; it leaves Python symbols to be
  resolved by the host interpreter at import time, which a test binary cannot
  satisfy. The feature is now opt-in-by-default and tests run with
  `--no-default-features`. CI previously ran only an import smoke test; it now
  runs 198 Rust tests and 16 Python behavioral tests across
  ubuntu/macos × Python 3.10–3.12.

### Added

- **`run_portfolio_backtest`** — simulates N instruments against **one** cash
  pool, with `max_positions` and a drawdown kill-switch gating each entry
  *before* it opens, so reported metrics describe the constrained run.

  This is materially different from running one backtest per symbol and summing
  the equity curves, which gives every symbol its own private copy of the
  capital. On 5 symbols with a 500k account, the summed approach reports
  2,381,392 final equity having deployed 2.5m — 5× the account. The shared-pool
  runner reports 478,537 for the same signals.

  Reports `rejected_entries`, `halted`, `halted_at`, and per-instrument
  attribution, so a constrained run is distinguishable from one with no signals.

- **`max_positions` and `max_drawdown_pct`** on `PyBacktestConfig`, enforced
  in-loop rather than by filtering trades afterwards. The kill-switch latches:
  once tripped it stays tripped, since a switch that re-arms on recovery is a
  materially less conservative policy.

- **Itemized Indian transaction costs** via `fee_segment` (`"NSE-INTRADAY"`,
  `"NFO-OPT"`, `"MCX-FUT"`, …), covering brokerage, STT, exchange transaction,
  SEBI turnover, stamp duty and GST across NSE/BSE equity, NFO/BFO, MCX and
  CDS. `trade.fee_breakdown["total"]` equals `trade.fees`, and their sum equals
  `metrics.total_fees_paid` — the itemized costs and the equity curve are now
  the same money.

  Charges land on the leg that owes them: STT on the sell, stamp duty on the
  buy, keyed off `(direction, is_entry)`. GST applies to brokerage, exchange
  and SEBI charges only, never to STT or stamp duty.

  Note: for options, STT and exchange charges are levied on *premium*, not
  contract notional.

- **`session_minutes`** with exported constants `SESSION_NSE` (375),
  `SESSION_MCX` (870), `SESSION_CDS` (480) and `SESSION_CONTINUOUS` (24×7).
  Intraday annualization scales with session length, so assuming NSE hours on
  MCX data understates Sharpe by `sqrt(870/375)` ≈ 1.52×.

- **`periods_per_year`** to override annualization explicitly, and
  **`risk_free_rate`**, wired into Sharpe and Sortino as excess return.

- **`EngineKernel`** — the per-bar simulation body extracted into a steppable
  core (`step(bar) -> Vec<EngineEvent>`). Batch backtests loop it; a live feed
  can drive the same code, which is the groundwork for backtest/live parity.

- **Class-based strategy contract.** Strategies can now be written as Python
  classes with lifecycle hooks instead of precomputed signal arrays:

  ```python
  class SmaCross(raptorbt.Strategy):
      def on_start(self, ctx): ...   # precompute indicators on ctx.close etc.
      def on_bar(self, ctx):
          if crossed_up:   self.enter()
          if crossed_down: self.close_position()
  result = raptorbt.run_strategy_backtest(SmaCross(), timestamps, o, h, l, c, v)
  ```

  Hooks: `on_start`, `on_bar`, `on_stop`, `on_order_filled`,
  `on_order_rejected`, `on_position_opened`, `on_position_closed`. Order
  intents (`enter(size_frac=..., stop_price=..., target_price=...)`,
  `close_position()`) are applied through the same execution core as the
  array runners — `SingleRunner`, extracted from the batch engine loop — so
  identical decisions produce bit-identical trades, curves, and metrics
  (pinned by the equivalence tests in `tests/python/test_strategy.py`).
  `ctx` exposes the OHLCV arrays, current index/bar, position snapshot,
  equity/cash, and programmatic `set_stop_price`/`set_target_price`.

  Entries whose computed size rounds to zero units (size fraction below one
  lot, or insufficient capital at the fill price) now emit
  `EntryRejected { reason: ZeroSize }` instead of being silently skipped;
  class strategies receive it via `on_order_rejected` with
  `reject_reason="ZeroSize"`. Array-runner results are unchanged — the batch
  path ignores rejection events.

  Rust/PyO3 surface: `PyKernelSession` (per-bar `step` driving the engine
  with scalar inputs), `PyEngineEvent`, `PyPositionSnapshot`,
  `resolve_atr_period`, kernel `set_stop_price`/`set_target_price`/
  `position_snapshot`, and `StepInput.stop_price_override`/
  `target_price_override` for per-entry explicit exit levels.

  The array-based runners are unchanged and remain fully supported; they are
  the fast path for vectorized workloads. New strategies should prefer the
  class contract.

- **Type stubs.** `_raptorbt.pyi` and a `py.typed` marker now ship in the
  wheel. The `Typing :: Typed` classifier was previously inaccurate.

### Changed

- Stop and take-profit fills route through `FillModel`, which handles
  gap-through for all four `(direction, is_entry)` cases; the engine previously
  inlined a long/short-only copy. Behavior is unchanged.
- `compute_backtest_metrics` gained a `timestamps` parameter, and
  `compute_backtest_metrics_with_config` was added for callers that have a full
  config.
- `PositionManager::close_position` takes an `ExitDetails` struct.
- `StepInput` gained `stop_price_override` and `target_price_override`
  fields. Rust rlib consumers constructing it as a struct literal without
  `..Default::default()` must add the new fields; the Python API is
  unaffected.

### Removed

- `signals::expression` (456 lines). It had no parser or AST despite its module
  documentation, was never re-exported, never bound to Python, and had zero
  references anywhere in the crate. Its role is superseded by the class-based
  strategy contract shipped in this release. This is a breaking change only for
  Rust consumers of the rlib that referenced `raptorbt::signals::expression::`
  directly.
- The checked-in `_raptorbt.cpython-311-darwin.so` build artifact and
  `libraptorbt.dylib.dSYM/`. Compiled extensions are now gitignored.

## Migrating from 0.4.x

### Reproducing old results

```python
cfg = raptorbt.PyBacktestConfig(
    initial_capital=100_000.0,
    fees=0.001,
    slippage=0.002,
    apply_slippage=False,        # restores the 0.4.1 no-op
    legacy_annualization=True,   # restores 365/252 and bar-count Calmar
)
```

Verified bit-identical to 0.4.1 across 13 single-instrument scenarios and for
the basket runner, where legacy mode reproduces the old Sharpe to the last
digit. The only unconditional difference is `inf` → `None` on undefined ratios.

### Required code changes

**Optional metrics.** Six metrics are now `Optional[float]`:

```python
pf = metrics.profit_factor
if pf is not None:
    ...
```

A `getattr(metrics, "profit_factor", 0.0)` guard does **not** help — the
attribute exists, so the default never fires and the call returns `None`.
Arithmetic on these fields needs an explicit check.

### Expected metric changes (defaults)

| Metric | Change |
|---|---|
| Sharpe / Sortino (daily, single) | unchanged — daily data still resolves to 365 |
| Sharpe / Sortino (intraday) | increases; annualized on session count, not calendar days |
| Sharpe / Sortino (basket/pairs/options/multi) | **decreases substantially**; per-bar rather than per-trade returns |
| Calmar (daily) | shifts slightly; elapsed time rather than bar count |
| Calmar (intraday) | changes substantially |
| Any run with `slippage > 0` | returns decrease; slippage is now actually charged |
| Everything else | unchanged |

### Recommended

- Pin `raptorbt>=0.5.0,<0.6.0`. A floating `>=` lower bound means an
  unattended upgrade silently picks up behavior changes.
- Pass `session_minutes` for MCX and CDS strategies.
- Replace `hasattr` feature-sniffing with a `__version__` check; it is now
  accurate.

## [0.4.1] - earlier

Releases before 0.5.0 were tracked in commit messages only.
