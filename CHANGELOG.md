# Changelog

All notable changes to raptorbt are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-07-21

This release fixes five defects where the engine silently returned wrong
numbers, adds a shared-capital portfolio runner, and introduces the
class-based strategy contract — an event-driven alternative to precomputed
signal arrays.

**Read "Migrating from 0.4.x" before upgrading** — several metrics change
value. Setting `apply_slippage=False, legacy_annualization=True` reproduces
0.4.1 results bit-identically, so stored backtests stay reproducible while you
migrate.

### Fixed

- **Timers and alerts fired for only one symbol in multi-symbol runs.**
  The clock was global, so a timer set in `on_start` fired once for
  whichever symbol's event happened to cross the threshold first and never
  for the rest — a two-symbol heartbeat delivered half its beats, silently.
  Each symbol now has its own clock, carrying whatever `on_start`
  scheduled. Single-symbol runs are unchanged.

- **Options never settled to intrinsic value.** `settle_expiry` called
  `settlement_value(close, None)`, and the `None` meant the option branch
  could never match, so every option settled at its own last close no
  matter how far from intrinsic that was. The strategy can now supply an
  underlying via `ctx.set_underlying_price(...)` — routed per symbol in
  portfolio runs — and without one, contracts still settle at their own
  close, since an option's bars carry the option's price and the engine has
  no second series to read.

- **`TimeInForce::Day` expired on the UTC date, not the trading date.**
  A session whose local hours cross UTC midnight would see DAY orders die
  while the trading date was still running. `session_tz_offset_ns` on
  `PyBacktestConfig` sets the offset — e.g. `IST_OFFSET_NS` — and defaults
  to `0`, which is arithmetically identical to the old behavior.

  This is a latent fix rather than a live bug for NSE users: 09:15–15:30 IST
  does not cross UTC midnight, so the common case was already correct. It
  follows the trading *date*, not the trading *session* — a DAY order still
  survives past the session close to the next session of the same date.

- **Indicator registration was a silent no-op in portfolio runs.**
  `register_indicator` appended to the strategy's list but
  `run_portfolio_strategy` never updated anything, so `.value` stayed `None`
  and `indicators_initialized()` never became true. Indicators now update,
  routed per symbol. Registrations also reset per run, so re-running one
  strategy instance no longer accumulates duplicates (matching
  `run_strategy_backtest`).

- **`modify_order` raised `NotImplementedError` in portfolio runs.** The id
  map already carried the owning instrument; the routed binding was missing.
  Modifies now route without the caller naming a symbol.

- **`max_positions` was per-instrument in portfolio strategy runs.**
  `EventSession` gave each instrument its own copy of the risk gate, and
  `RiskGate` is `Copy`, so every kernel checked the limit against its own
  ledger: `max_positions=1` across three symbols allowed three concurrent
  positions. It is now counted across all instruments, as the array runner
  (`run_portfolio_backtest`) has always done, and is enforced on the
  resting-order path as well as on signal entries. Runs that set
  `max_positions` on `run_portfolio_strategy` will open fewer positions than
  before — the previous behavior did not match the documented meaning of the
  setting or the sibling API.

- **Portfolio session results reported stubbed halt/rejection fields.**
  `PyPortfolioSession::finish` hardcoded `rejected_entries: 0`,
  `halted: false`, and `halted_at: None`, so a `run_portfolio_strategy` run
  could refuse entries or trip its drawdown kill-switch and still report a
  clean, unhalted result. All three now carry real values:
  `rejected_entries` sums the per-instrument counters (already reported
  correctly on `per_instrument`), and `halted`/`halted_at` cover both the
  drawdown kill-switch and the new portfolio margin call.

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

- **Four deferred execution knobs**, all default-off:

  `limit_slippage` applies an adverse adjustment to limit fills, which
  previously always printed exactly at the limit. It is suppressed when
  `queue_fill_model` granted the fill: volume observed trading ahead of an
  order is evidence it genuinely held that price, so slipping it too would
  double-penalize.

  `liquidate_on_margin_call` force-closes positions when a margin call
  fires, instead of only latching a halt. Unlike expiry settlement or
  end-of-data finalization — both of which close free — a liquidation is a
  real trade-out: it prices through the fill model and pays exit costs, and
  reports the new `ExitReason::Liquidation`.

  `InstrumentSpec.settlement_fee` charges a fee on the settled notional at
  expiry. It sits alongside `maker_fee`/`taker_fee` rather than on the
  config because exercise and assignment are commonly priced differently
  from a trade-out, and a portfolio run needs per-instrument rates.

  `EngineKernel::set_underlying_price` lets options settle to intrinsic
  value — see below.

- **TWAP execution schedules.** `orders.Twap(side=..., units=..., slices=N,
  every=<ns>)` releases N equal slices at a fixed interval, each an ordinary
  order reporting its own fill with a client id of `"<parent>#<n>"`. New
  `on_algo_started` / `on_algo_completed` hooks bracket the schedule;
  "completed" means fully released, not necessarily fully filled.

  A schedule is deliberately not an order. Modelling it as a parent would
  deadlock its slices — the one-triggers-other gate holds a child until its
  parent *fills*, and a schedule never fills — so slices carry only a
  back-pointer and the matcher needs to know nothing about them.

  The interval is a duration, not a bar count: `idx` is a bar ordinal in a
  bar session and an event ordinal in a tick session, so "every 1 bar" would
  silently mean "every 1 print" on a tick feed and collapse a five-slice
  TWAP into a burst. Pass `every_bars` with `bar_ns` for bar-shaped
  ergonomics. Slices release one per step even after a data gap, since
  dumping a backlog defeats the point of spreading an order.

  Only explicit `units` can be sliced — `size_frac` resolves against equity
  at fill time, so each slice would size against a different account.
  Cancelling a schedule stops the remaining slices; it does not unwind what
  already traded. Only TWAP ships in 0.5.x; VWAP needs a volume forecast and
  POV needs partial fills, both still deferred.

- **Renko and signed-flow bar units.** Every variant declared in
  `AggregationUnit` now builds; none returns `Unimplemented`.

  `"renko"` emits a brick per full brick-height price move, ignoring time
  and volume entirely. Height comes from a new `brick_size` argument
  (`BarAggregator`, `aggregate_bars`, `bars_from_ticks`, `subscribe_bars`),
  falling back to `step` read as whole price units. Because one move can
  complete several bricks, `BarBuilder` gains `next_pending()` and its
  Python mirror — **drain it after every push or those bricks are lost**.
  Bricks carry no wicks and a partial brick is discarded, not flushed: an
  incomplete brick is not a brick.

  The six information-driven units — `{tick,volume,value}_imbalance` and
  `{tick,volume,value}_runs` — sample by signed order flow. Imbalance closes
  on net flow, so balanced two-sided trading never closes a bar however
  heavy; runs closes on the larger one-sided accumulation, so it does. The
  threshold is `step`, fixed rather than the literature's adaptive EWMA:
  deterministic, reproducible, and consistent with how `step` already reads
  for tick/volume/value bars.

  Direction comes from the feed when known — `TradeTick` and `SourceRecord`
  gain a signed field, populated from the buy/sell quantity deltas that
  `tick_data_to_events` previously summed away. The unsigned `size` is
  unchanged, so no existing bar moves. Without a split, direction falls back
  to the tick rule, which is what lets these units work over plain bars.

- **Order book state and queue-position limit fills.** `OrderBook` tracks
  the visible book from quotes (L1) or depth snapshots (L2, five levels),
  exposed to strategies through a new `on_order_book(ctx, book)` hook,
  `ctx.book`, and a `depth=` argument to `run_tick_strategy`.

  With `queue_fill_model=True` (opt-in, default off), resting limits fill
  from observed queue position rather than `fill_prob_limit`'s coin flip.
  The size ahead is estimated once, when the order rests, then consumed by
  print volume at that price; a print *through* the level fills
  unconditionally. Unlike the probability model, progress is monotone — an
  order passed over repeatedly genuinely gets closer to the front.

  The model claims no real queue rank: without an order-by-order feed there
  is no way to know your position, nor to tell size that executed ahead of
  you from size that was cancelled. It therefore falls back to
  `fill_prob_limit` rather than guessing — on bar events (a bar's volume is
  not volume *at* the limit price) and on a quote-only book (a quote gives
  the price but not the size). A level outside the visible five reports
  "unknown", never "empty".

  Book updates are observation only, like quotes: they never fill an order,
  move a trailing stop, mark equity, or sample the equity curve. They do
  change *future* fills by sizing the queue a new order joins.

- **Tick-driven class contract** — `run_tick_strategy(strategy, ticks, ...)`
  drives the same event session from trade prints and quotes, so orders,
  positions, risk gates and the shared account behave as they do on bars;
  only the resolution changes. New `on_trade_tick(ctx, tick)` and
  `on_quote(ctx, quote)` hooks, `TradeTick`/`QuoteTick` payloads, and
  `ctx.best_bid`/`ctx.best_ask`/`ctx.last_price`.

  Three semantics worth knowing before using it:

  - **Quotes are observation only.** They do not fill orders, move trailing
    stops, or mark equity. Filling against a quote would assert a
    counterparty the engine has no evidence for; the print that follows is
    that evidence. An order submitted from `on_quote` rests and matches on
    the next print.
  - **`ctx.best_bid`/`ctx.best_ask` inside `on_trade_tick` are the book
    observed *before* that print** — the quote from the same feed row
    arrives in the following `on_quote`. Reading it earlier would be a
    lookahead onto a book the print itself moved.
  - **`primary_bars=(step, unit)` builds bars from prints as a view**: they
    fire `on_bar` and feed indicators, but nothing executes on them. Orders
    match against ticks only.

  `AT_OPEN`/`AT_CLOSE` market orders keep resting on a print, since a print
  has no bar phase to queue against. Trailing stops ratchet off every print,
  so a tick run and a bar run over the same data legitimately differ there —
  a bar can trigger a stop against a low that preceded the high which set
  the watermark, and prints cannot.

- **Per-symbol indicators and composite bars in portfolio runs.**
  `register_indicator(indicator, stream_id=None, symbol=None)` gains
  `symbol=` to route an indicator to one instrument, and
  `register_indicators(factory, symbols)` builds one per symbol. One
  `subscribe_bars` declaration now yields one aggregated stream per symbol,
  each built only from that symbol's bars, and `CompositeBar` gains a
  trailing `symbol` field (`None` outside portfolio runs) naming the
  instrument that completed it. A symbol's composite bar dispatches before
  that symbol's `on_bar` which completed it; across symbols, order follows
  the merged schedule.

  Note: an indicator registered *without* `symbol=` in a portfolio run is
  fed every symbol's bars interleaved — rarely meaningful, since one
  indicator cannot track N series — and now warns. It previously did
  nothing at all, so no working strategy changes behavior.

- **Shared margin accounts in portfolio runs** — `run_portfolio_strategy`
  accepts `account_type="margin"` and `leverage`, previously available only
  to single-instrument runs. One account funds every instrument: leverage
  applies portfolio-wide, sizing draws on the portfolio's free capital
  (balance less all locked margin), and equity marks the balance plus
  direction-aware unrealized PnL, so a winning short raises portfolio equity
  instead of lowering it. The maintenance requirement is the sum of each
  instrument's own requirement, so per-symbol `margin_maint` rates apply
  rather than one blended rate; a breach fires `on_margin_call` once and
  halts new entries on every instrument, including symbols that never
  traded. `PyPortfolioSession` gains `free_capital()` and `is_halted()`.
  Cash-account runs are unchanged and remain pinned by the golden fixtures.

  Note: in portfolio runs `halted_at` is a **schedule-event ordinal**, since
  the session interleaves N instrument streams; the array runners'
  `halted_at` remains a bar index.

- **Portfolio drawdown halts now record `halted_at` on the shared account**,
  so margin-call and drawdown halts report identically. A drawdown halt
  keeps its own reject reason (`DrawdownHalt`) rather than borrowing the
  margin-call switch. One consequence: once any halt has latched, a later
  margin-maintenance breach no longer emits a second `MarginCall` event —
  halts are latch-once.

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
