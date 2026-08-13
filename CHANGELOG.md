# Changelog

All notable changes to raptorbt are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.2] - 2026-08-13

Patch, but a consequential one: intraday backtests can now be told to close
their positions before the market shuts, and a position left open at the end of
a run is finally reported as a trade.

**In plain words: a backtest could report profit earned overnight, while the
market was closed — money no trader could have made, because their broker would
have closed the position at the end of the day. There was no setting to stop
it. Now there is, and the results change materially.**

### Added

- **`BacktestConfig(squareoff_time="15:25")`** — force-closes open positions at
  the first bar at or after that local time in each trading day. `None` (the
  default) keeps the old behaviour, so no existing result moves unless you ask
  it to.

  The time is **local**, interpreted through `session_tz_offset_ns`, so it is
  market-agnostic: `"15:25"` with an IST offset is NSE's squareoff, `"16:00"`
  with a zero offset is a UTC-quoted market. Setting `squareoff_time` and
  leaving the offset at its `0` default is the one easy mistake — 15:29 IST
  then reads as 09:59 UTC and nothing fires. Unreadable values raise
  `ValueError` rather than silently disabling squareoff.

  Positions closed this way carry the new `ExitReason::Squareoff` (`"Squareoff"`
  from Python), distinct from `EndOfData`: it is a real trade-out at a real
  in-session price, paying normal exit costs. The engine will not re-enter on
  the squareoff bar itself.

- `core::session::squareoff_flags` — the shared session-boundary helper behind
  it, usable by any strategy path.

### Fixed

- **A position still open when the data ends is now recorded as a trade.** The
  spread path settled it into cash without pushing a `Trade` or calling
  `record_trade`, so its P&L reached `end_value`, `total_return_pct` and the
  equity curve while `trades()` returned empty and `total_closed_trades` read
  zero.

  This is the most dangerous shape a reporting defect can take: every
  trade-level audit passes, because there is nothing to audit. It was found by
  a run whose entire return came from one position opened on the first morning
  and never closed — visible in the equity curve, invisible in the trade book.

- **`BacktestConfig.set_session_config` no longer appears in the type stub.**
  It was declared in `_raptorbt.pyi` and never existed in the engine. Callers
  guarding with `hasattr(config, "set_session_config")` took the else-branch
  every time, so `session_aware=True` was silently dropped and every intraday
  backtest ran with no squareoff. A type checker reading the stub agreed with
  the call throughout.

  A new guard (`TestStubDeclaresNothingFictional`) pins the stub -> runtime
  direction. The existing `TestStubCompleteness` only checked runtime -> stub,
  which is why this was never caught. Verified by injection: reinstating the
  fictional declaration fails the guard.

### Changed

- **`SpreadConfig.close_at_eod` is removed.** It was declared, defaulted to
  `false`, hardcoded to `false` at both binding sites, and read by nothing —
  dead since it shipped. `squareoff_time` supersedes it and is actually
  honoured. A field that looks like a working setting and does nothing is what
  this release exists to stop; leaving it in place would repeat the defect.

  Rust callers constructing `SpreadConfig` literally should drop the field;
  those using `..Default::default()` need no change. No Python API used it.

### Measured

On a real NIFTY option corpus (7 sessions, one expiry), enforcing a 15:25
squareoff moved net-of-cost P&L by:

| Strategy | No squareoff | With squareoff | Change |
| --- | --- | --- | --- |
| Short ATM straddle | ₹18,405 | ₹13,934 | −24% |
| Short strangle | ₹7,736 | ₹4,523 | −42% |
| Long ATM straddle | −₹18,627 | −₹15,590 | +16% |

The long straddle moving the *other* way is the important one: the defect does
not add a constant bias, it **amplifies whichever direction a position already
points** — making winners look better and losers worse. On that corpus one
boundary (the night into expiry day) carried 47.1% of all overnight P&L, so the
direction is robust but the magnitude is corpus-specific.

### Upgrading

Existing results are unchanged unless you set `squareoff_time`, with one
exception: a backtest that ended with a position still open now reports one
more trade than it did before. The P&L was always in `end_value`; it is now
also in `trades()`, so trade counts and per-trade statistics will differ for
those runs. That is the fix, not a regression.

If you run intraday strategies, set `squareoff_time` **and**
`session_tz_offset_ns` together — the first without the second silently does
nothing.

## [0.7.1] - 2026-08-12

Patch. The 0.7.0 deprecated names resolved but could not be enumerated.

### Fixed

- **Deprecated `Py*` names now appear in `dir(raptorbt._raptorbt)`.** They
  resolved through `__getattr__` in 0.7.0, but never showed up in `dir()`, so
  they were invisible to autocomplete and to any tool that enumerates a module.

  That is not merely cosmetic. A consumer guard comparing `_raptorbt.pyi`
  against `dir(_raptorbt)` read the stub's alias block as 21 declarations for
  symbols the engine had dropped -- precisely the "type-checks clean,
  `AttributeError` in production" drift such a guard exists to catch. The
  aliases are real, so they are listed.

- **The stub-completeness test in this repo had two blind spots** and so never
  flagged the above. It recognised `class X`, `def X(` and `X:` but not the
  `X = Y` alias form; and it matched declarations by substring, so `class Foo`
  matched `class FooBar` and a renamed-away class left the guard green. Both
  are anchored now, verified by deletion.

### Upgrading

No API change. If you consume the stub in CI, this is the release that makes
the 0.7.0 alias block agree with the runtime module.

## [0.7.0] - 2026-08-12

Two things: the public class names lose a prefix that never belonged in Python,
and five places where the engine quietly guessed now refuse instead.

Plain words on the second half, because it matters more. When raptorbt was
handed something it could not interpret -- an option type it could not parse, a
direction that was neither long nor short, a correlation matrix that is not
mathematically valid -- it picked a default and returned numbers that looked
completely normal. Not a crash, not an obviously silly figure: a smooth,
well-formed result computed from something other than what you asked for. No
metric, equity curve, or risk check downstream could tell.

### Changed

- **Every public class drops its `Py` prefix.** `PyBacktestConfig` is now
  `BacktestConfig`, `PyTrade` is `Trade`, `PyRiskModel` is `RiskModel`, and so
  on for 21 classes. The old spellings still work and emit a
  `DeprecationWarning` naming the replacement; **they are removed in 0.8.0.**

  The prefix was a Rust-side disambiguator -- the crate has its own
  `BacktestConfig`, `Trade` and `BacktestResult` in `src/core`, and two Rust
  types cannot share a name -- that was never stripped on the way out.
  `BarAggregator`, `Indicator` and `InstrumentSpec` already reached Python
  clean; this finishes the other 21. Rust struct names are unchanged.

  Deep imports keep working too: `from raptorbt._raptorbt import PyX` warns and
  resolves, because `PortfolioSession` had never been re-exported at top level
  and a deep import was the only way to reach it. It is exported properly now.

- **`max_trades` no longer defaults to 50.** It defaults to unlimited. This is
  a **behaviour change to existing tick backtests**: any run that relied on the
  implicit cap will now return different -- correct -- numbers.

  `max_trades` is a hard early exit, not a filter. The tick loop `break`s and
  the result is reported as if the tape ended there. On a 1,000,000-tick input
  the old default produced 50 trades covering **0.81% of the data**: a total
  return of -0.12% where the true figure was -14.13%, and a max drawdown of
  0.124% against a true 14.13%. That is a 114-fold understatement of the single
  number a risk check reads. The knob remains for anyone who explicitly wants a
  truncated run.

- **`run_options_backtest` string arguments are case-insensitive and closed.**
  `option_type`, `strike_selection` and `size_type` used a catch-all match arm,
  so `option_type="PUT"` selected a long **call** -- the mirror image of the
  intended payoff -- while the identical string was accepted by
  `run_spread_backtest`. The same call meant two different things depending on
  which function you entered through. Unknown values now raise `ValueError`;
  the documented defaults are unchanged.

### Fixed

- **`BarAggregator` ignored `brick_size`.** The constructor accepted the
  argument and then called a helper that hard-coded `0.0`, which
  `resolved_brick` reads as "fall back to `step`". Asking for 5-point Renko
  bricks gave you `step`-point bricks -- a 10-point move produced 10 bars
  instead of 2. **Every Renko backtest built through the streaming aggregator
  was wrong; re-run any stored Renko results.** The batch `aggregate_bars` path
  was always correct. Every pre-existing test used `step=1, brick_size=1.0`,
  where the fallback returns the number you asked for and the bug is invisible.

- **A correlation matrix that is not positive definite is refused, not
  repaired.** Cholesky patched a negative pivot with `sqrt(|diag|)` and a zero
  pivot with `0.0`, then returned success. On an indefinite 3-asset matrix
  (smallest eigenvalue -0.8) `simulate_portfolio_mc` returned `var_95 = 0` and
  `probability_of_loss = 0` -- a risk model reporting no risk at all, from
  input it should have rejected. The identity-matrix fallback beneath it was
  dead code and is gone; substituting one would have made every asset
  independent, the most optimistic assumption available to a risk model.

- **An unparseable option-type code no longer becomes a Call.**
  `OptionType::from_code` documents that "defaulting an unrecognised code to
  Call would price a put as a call", and both PyO3 call sites did exactly that.
  An iron condor whose put legs failed to parse became a four-leg call
  structure. `batch_spread_backtest` multiplied it across an entire sweep.

- **`direction` must be 1 or -1.** Six call sites fell back to long, so a book
  encoded `0`/`1` instead of `-1`/`1` backtested entirely long, flipping the
  sign of the P&L on every short behind a well-formed equity curve. In the
  basket and portfolio runners the parse runs per instrument, so one bad row
  turned a leg of a market-neutral book into a doubled long.

- **`simulate_portfolio_mc` validates its shapes.** Passing an
  `(n_obs, n_assets)` matrix where a per-asset list of series was expected
  indexed past the end of `weights` inside a Rayon worker, surfacing as
  `PanicException` -- not catchable as `ValueError`, thrown from a thread with
  no user code in the traceback. It now raises a `ValueError` naming the
  mistake.

- **A test that never ran now runs.** A duplicated `#[test]` attribute left the
  following function without one, so `day_expires_on_utc_date_rollover` --
  DAY-order expiry across UTC midnight -- silently never executed. It passes.

### Internal

- Build and lint are silent: `cargo clippy --all-targets -- -D warnings` passes
  and the library build emits no warnings, down from 89 diagnostics. Not by
  suppression -- 8 manual `Default` impls became derives, shift loops became
  `copy_from_slice`, the `w'Σw` quadratic form was deduplicated between the
  optimizer and risk contributions, and `OptionType::from_str` (which shadowed
  the `FromStr` trait, so `"CE".parse()` did not work) became `from_code` with
  a real `FromStr` impl beside it.

  Six `#[allow]`s remain, each with its reasoning in a comment. The load-bearing
  one is `adopt_position`, which guards with `!(price > 0.0)` rather than
  `price <= 0.0` because the negated form is also false for NaN. Clippy's
  suggestion would let a NaN price become a position's cost basis, turning
  cash, equity and every drawdown figure into NaN with no error raised.

  The optimizer index-math refactors were verified against a captured baseline
  of 18 numeric surfaces -- covariance, optimizer weights, risk contributions,
  MACD/RSI/ADX/VWAP, Monte Carlo, and a full backtest's metrics, equity curve
  and drawdown curve. Bit-identical before and after.

- **`benches/python/` ships the benchmark harness** behind every published
  performance figure, so a claim can be re-run rather than trusted.

### Upgrading

Nothing breaks on import. Old class names work for this release.

Three behaviour changes to be aware of:

1. **Renko backtests through `BarAggregator` were wrong** and are now correct.
   Re-run any stored Renko results.
2. **Tick backtests that used the default `max_trades`** were truncated and are
   now complete. Their numbers will change, substantially.
3. **Input that used to be guessed is now refused.** If you were passing
   `direction=0`, an option-type string outside `CE/CALL/C/PE/PUT/P`, an
   unrecognised `strike_selection`, or a non-positive-definite correlation
   matrix, you will now get a `ValueError` naming the argument. Those calls were
   already producing wrong answers; they were just not saying so.

The published performance numbers moved because the harness changed, not the
engine. The 0.6.4 wheel and this build measure at 71.0 µs and 70.5 µs on 1,000
bars on the harness now in `benches/`, with identical results.

## [0.6.4] - 2026-08-10

One defect, one character, and it inverted every multi-leg options backtest.

Plain words: if you backtested a spread — any structure with more than one
option leg, like a credit spread, a straddle, or an iron condor — the result
was reported backwards. A structure that made money showed a loss, and one
that lost money showed a profit. Worse than the wrong number: the automatic
stop-loss read the same backwards figure, so it closed positions that were
*winning*, and the profit target closed positions that were *losing*.

Nothing that trades real money was affected. Paper and live deployments price
their leg groups through a different code path entirely, which was always
correct and is pinned by test. The damage was to research: a user could have
discarded a profitable strategy or deployed a losing one on the strength of an
inverted backtest.

### Fixed

- **Spread P&L is no longer negated.** `LegPosition::unrealized_pnl` computed
  `-quantity * premium_change * lot_size`. `LegConfig.quantity` is already
  signed (`+1` long, `-1` short), so the leading minus applied the direction
  convention a second time and flipped the result:

      short (-1) + premium falls (-30) -> (-1) * (-30) * 75 = +2250, a gain
      long  (+1) + premium falls (-30) -> (+1) * (-30) * 75 = -2250, a loss

  Everything downstream reads that one function, so `pnl`, the equity curve,
  the drawdown curve, and every derived metric — `sharpe_ratio`,
  `profit_factor`, `win_rate`, `expectancy`, `best_trade_pct` — inherit the
  correction.

- **`max_loss` and `target_profit` fire on the right side now.** Both compare
  against the same figure, so through 0.6.3 a max-loss threshold closed
  structures that had gained and a target-profit booked wins on structures
  that had lost. This changes *when positions close*, so a backtest re-run
  under 0.6.4 with either threshold set will differ from its 0.6.3 result by
  more than a sign.

### Upgrading

**Any stored spread backtest produced by 0.6.3 or earlier is wrong and should
be re-run.** `pnl` and every metric derived from it are inverted. Results with
`max_loss` or `target_profit` set differ further, because the exit timing
itself was wrong. Single-leg backtests are unaffected — they never went
through this code path.

### Added

- Nine Rust regression tests covering all four short/long x win/lose cases and
  all four stop/target x winner/loser cases, plus a Python behavior suite
  (`tests/python/test_spread_backtest.py`) exercising the same contract
  against the built wheel. The defect survived because neither existed: the
  Rust tests asserted only that a trade was recorded, and no Python test
  called `run_spread_backtest` at all.

## [0.6.3] - 2026-08-06

Two defects in position adoption, both on the path that seeds a strategy with
shares a user already owns. Neither was firing in production — the supported
entry point adopts before the run starts, and today's seeded strategies are
long-only — but both were reachable.

This release also teaches the portfolio optimizer to hold short
positions — by explicit configuration only. Plain words: until now the
optimizer could only say "buy, hold, trim, or sit in cash." With
`short_cap > 0` it may also propose NEGATIVE weights: positions that
profit when a price falls. Nothing changes for existing callers — the
default (`short_cap = 0`) poses the byte-identical long-only problem it
always has, pinned by test.

### Added

- **An update notice.** raptorbt now writes one `INFO` log line, at most once
  a day, when the installed version is behind the newest release on PyPI:
  `raptorbt 0.6.2 is behind the latest release 0.6.3. Install the latest
  version: pip install -U raptorbt`. Plain words: it tells you to upgrade,
  and does nothing else.

  It cannot slow or break an import, and **it fails silently by design**: the
  request runs on a daemon thread with a 2s timeout, every failure path is
  swallowed at two independent layers plus a guard inside the thread body, and
  the answer is cached on disk for 24h so a restarting fleet is not a burst of
  requests. An unreachable PyPI is indistinguishable from the check never
  running — no traceback, no stderr output, nothing on the log at all.

  The notice is `INFO` rather than `WARNING` deliberately. With no logging
  configured, Python's `logging.lastResort` prints `WARNING` and above to
  stderr; `INFO` stays under that bar, so the line appears for anyone who asked
  for INFO logs and is invisible to everyone else. A library telling you to
  upgrade has not detected a problem with your program.

  Set `RAPTORBT_NO_VERSION_CHECK=1` to disable it; continuous-integration
  environments are skipped automatically, since a pinned wheel there is
  deliberate. Versions that cannot be parsed as plain dotted releases — a
  pre-release, a local build, or the `unknown` of a source checkout — produce
  no message rather than a wrong one.

- **Long/short mode on `optimize_portfolio`** (`PyOptimizerConfig`):
  `short_cap` (per-name short bound, default 0 = long-only), `gross_max`
  (`sum |w| <= gross_max` — the total size of all bets), and `net_min` /
  `net_max` (bounds on `sum(w)` — the directional tilt; `net_min = net_max
  = 0` is a dollar-neutral book). Gross exposure and the gross sector caps
  are linearized with auxiliary variables (`u_i >= |w_i|`), the same
  epigraph device the turnover term already uses; the variables and their
  rows exist only when shorting is enabled.
- **`gross_exposure` / `net_exposure` on `PyOptimizationResult`.** `cash`
  remains `1 - sum(w)` (net-based) and is documented as such — for a
  long/short book read the exposure fields, not the cash residual.
- **`optimize_book`** as the honest name for the Rust entry point;
  `optimize_long_only` remains as a delegating alias so existing callers
  keep compiling.

### Changed

- **Sector caps are GROSS in long/short mode** (`sum_{i in k} |w_i| <=
  cap`): a cap bounds the size of a sector's bets, not their direction.
  For a long-only book the gross and signed sums coincide, so the two
  modes agree exactly where they overlap.
- A negative `w_current` is accepted when `short_cap > 0` (it is the
  book being rebalanced); still refused in long-only mode.

### Deferred, pinned

- **Short position adoption stays refused** (`short_adoption_stays_refused_
  by_construction`): posted broker collateral is not derivable from
  quantity x average price, and no supported flow seeds a short. The
  refusal is structural — adoption has no direction parameter.

### Fixed

- **A position adopted mid-run made the strategy look less risky than it was.**
  The equity curve is written as the run proceeds, one sample per event,
  against a running peak that starts at the initial capital. Adopting after the
  run began left that curve flat for the stretch before the adoption, which
  held the peak down, so the decline that followed was measured against a
  high-water mark lower than the truth.

  On a 6-bar 100→95 fixture adopting 100 shares at 90, a **0.495%** max
  drawdown reported as **0.199%**. Total return and `open_trade_pnl` were
  identical either way, so nothing in the headline numbers hinted at it — only
  the risk metric moved, and it moved to look safer.

  Because the samples are written as the run proceeds, they are already wrong
  by the time metrics are computed; there is no repairing it afterwards. So
  `adopt_position` now returns an error once any equity sample has been taken,
  raising `ValueError` from Python.

  The gate is the equity curve, not the event cursor: quote and depth events
  advance the schedule without sampling equity (marking on a quote would append
  a zero return per quote and distort annualized metrics by how chatty the feed
  is), and a live feed routinely delivers quotes before the first trade print.
  Adopting after one corrupts nothing and stays allowed.

  `TickStrategyStream(initial_positions=...)` is unaffected — it adopts before
  warmup replay and before the first push, which is why this never fired in
  production. `EngineKernel::adopt_position` holds no equity curve and cannot
  check this itself; a Rust consumer driving the kernel directly owns the
  ordering, and its doc comment now says so.

- **A seeded long/short strategy could not be deployed at all.** A short leg
  only transacts as a short under a margin account — in cash mode its P&L never
  reaches equity — so a strategy holding one runs under margin at leverage 1.0,
  which keeps the book fully funded. Adoption refused margin outright, so the
  seed and the short were mutually exclusive: construction raised and the
  deploy died before it began.

  Fully funded margin books (initial margin rate ≥ 1.0) are now adopted by
  **locking** the cost basis as initial margin rather than debiting cash. That
  is not a cosmetic difference: margin equity is `balance + unrealized`, with
  no position-value term, so a cash-style debit would never be offset and would
  understate equity by the cost basis for the entire run. Relaxing the account
  check without fixing the funding arm would have replaced a loud failure with
  a silent wrong number.

  Leveraged books stay refused, and the original reasoning is why: the margin a
  broker has already posted against a position it holds cannot be derived from
  quantity and average price, and inventing a figure would misstate free
  capital, which gates every later entry. At a rate of 1.0 the whole notional is
  locked and the posted margin simply *is* the cost basis, so the objection
  lapses there and only there.

  **The error message changed** from `"adopt_position supports cash accounts
  only"` to one naming the fully-funded requirement. Callers matching on that
  string need updating.

  The portfolio session now also reconciles the locked delta into its shared
  account, where it previously passed a hardcoded zero. Left as it was, the
  account would never learn about the adopted margin and portfolio free capital
  would read high by the whole cost basis.

  Adoption remains **long-only**: an existing short cannot be seeded. That is
  separate scope — direction-aware cost basis, short proceeds, borrow — and is
  stated here so the boundary is explicit rather than accidental.

## [0.6.2] - 2026-08-05

A strategy attached to a stock the user **already owns** can now start out
knowing it holds those shares, at the price the user actually paid — without
the engine pretending a buy happened.

### Added

- **Position adoption.** `EngineKernel::adopt_position` opens a ledger
  position with no order, no fill, no fees, no trade record and no `Entered`
  event; cash is reduced by the cost basis, so equity reads as
  initial + unrealized exactly like an account that bought earlier. Without
  this, seeding a holding meant faking an entry — which charged fees that were
  never paid and left a phantom trade in the log.

  `PortfolioSession::adopt_position` applies the same lend/drain pool
  discipline as `apply_current`, so the cost basis comes out of the shared
  cash pool rather than appearing from nowhere.

  Exposed to Python as `PyPortfolioSession.adopt_position(...)` and as
  `TickStrategyStream(initial_positions={symbol: {"quantity", "avg_price",
  "timestamp_ns"?}})`. Adoption runs **before** warmup replay and before the
  first push, so the position is present in every before-snapshot: a caller
  diffing `positions()` around a push can never mistake it for a fresh entry.

  Cash accounts only — margin adoption is **refused, not guessed**, since the
  margin already posted against a broker-held position is not derivable from
  quantity and average price. A seed with non-positive quantity or price is
  rejected rather than silently skipped.

  Design reference: NautilusTrader's position adoption in live-execution
  reconciliation, where adopted state coexists with the order lifecycle
  without synthetic fills. Ported as a design, not as code.

## [0.6.1] - 2026-07-31

The significance score on factor measurement was overstated, because
overlapping test windows were counted as if they were independent. This
release reports the corrected number alongside the old one.

### Added

- **Overlap-deflated rank-IC t-statistic.** `rank_ic` previously reported only
  the naive IID t-stat, `mean / (stdev / sqrt(n))`. With a 21-day forward
  window on daily dates, consecutive ICs share 20 of their 21 days, so
  `n_dates_scored` overstates the independent sample by ~21× and inflates the
  t-stat by ~sqrt(21).

  Plain words: the same three weeks of market movement was being counted
  twenty-one times over as if it were twenty-one separate pieces of evidence.

  `RankIc` / `PyRankIc` gain three fields, all additive:

  - `t_stat_deflated` — `t_stat / sqrt(horizon)`; **the number to decide on**
  - `n_independent` — `n_dates_scored / horizon`; the sample actually behind it
  - `overlap_days` — the window, so a stored result is self-describing

  The naive `t_stat` is deliberately kept, so the inflation stays auditable
  rather than being quietly corrected away. `n_independent` is `0.0` (not NaN)
  on a panel that scores nothing, so callers can gate on sample size without an
  `isnan` dance.

  This is not theoretical. Measured on a live 2023-02..2026-07 vendor panel
  (1045 names), momentum 12-1 scores IC +0.0386 with a naive t of **+7.15** and
  a deflated t of **+1.56**, over 17.3 independent forward windows — real, but
  a materially smaller claim than the naive figure suggests. The same
  correction took the Indian fund cross-section from t=+4.78 to +1.04, which is
  the measurement that retired funds from the equity model; reporting equities
  on the naive statistic while funds were judged on the deflated one would have
  been exactly that double standard.

  Purely additive — no existing field changes meaning.

## [0.6.0] - 2026-07-25

An order's `side` now decides the direction a position opens in, so a single
run can hold long and short legs and a leg can flip side once it is flat.
This makes a cross-sectional long/short book — long the winners, short the
losers, rebalanced — expressible in one run against one capital pool.

This release also adds the portfolio-construction maths: how much risk a book
carries, what weights to hold, and what rebalancing actually costs.

### Added

- `Strategy.enter_long()` / `Strategy.enter_short()`, and `enter(side=...)`.
  Without `side`, `enter()` opens in the session's configured direction
  exactly as before.

  `enter()` could previously open only in the session's configured direction,
  so the sided order types were the only way to short — a nine-field kw-only
  dataclass that is easy to fill wrong, and unreachable from a sandboxed
  strategy at all. `enter_long()` / `enter_short()` take no side argument to
  mis-spell. A sided entry passes an explicit `size_frac`, because omitting
  both sizing kwargs means "close the whole position", which an opening order
  refuses — a sided entry that silently rejected itself would be worse than no
  feature.

- **Covariance estimation** (`estimate_covariance` → `PyRiskModel`).
  Ledoit-Wolf shrinkage against a constant-correlation target — plain words:
  a covariance matrix estimated from a few hundred days of returns is mostly
  noise, so it is pulled part-way toward a simpler, steadier matrix. Carries
  `periods_per_year` and the asset ordering structurally, so a risk model
  cannot be silently applied to a differently-ordered basket.

- **Constrained portfolio optimizer** (`optimize_portfolio`,
  `batch_optimize_portfolios` → `PyOptimizationResult`). Long-only quadratic
  program via Clarabel (new dependency, pure Rust) with an L1 turnover
  penalty, per-position and per-sector caps, and explicit cash. Post-solve, a
  no-trade band and a minimum-trade-value rule snap tiny trades away.

  If *all* trades snap away, the result is the status-quo book with turnover
  0 — a legitimate "do nothing" answer. If only *some* snap, leaving weights
  that no longer sum correctly, it **refuses with arithmetic** rather than
  returning a book that does not add up. `batch_optimize_portfolios` runs via
  Rayon and is deterministic: batch results are bit-identical to serial.

- **Factor panels** (`winsorize_panel`, `zscore_panel`, `rank_panel`,
  `momentum_panel`, `composite_scores`). Row-major panel transforms — trim
  outliers, standardize, rank, compute past-return momentum, and blend several
  signals into one score. `NaN` means *absent* and is handled; infinity is a
  hard error, never a silent maximum. No factor list is hard-coded in Rust —
  the caller decides what to score.

- **Rank-IC factor validation** (`rank_ic` → `PyRankIc`). Per-date Spearman
  rank correlation between a factor panel and forward returns at a chosen
  horizon — plain words: does yesterday's ranking of stocks predict tomorrow's
  ordering of returns? Returns the mean IC, the naive t-stat, and an
  overlap-deflated t-stat (see 0.6.1, which added the deflated fields to the
  Python surface). `PyRankIc` carries the panel span and name count, so the
  number is reproducible rather than a constant with a citation.

  First use caught a real artifact: fund momentum on 67 NSE funds read t=+4.78
  naive but +1.04 deflated, and collapsed to +0.016 once the 25 precious-metal
  funds were removed — a metal rally, not a factor. The fund ranking was
  therefore not shipped.

- **Risk contributions** (`compute_risk_contributions` →
  `PyRiskContributions`). Euler decomposition of portfolio volatility, so
  contributions sum exactly to sigma — it says which holdings the risk is
  actually coming from, not merely which are largest.

- **Rebalance policy simulation** (`simulate_rebalance_policy` →
  `PyRebalanceSimResult`, and `indian_cost_schedule`). Simulates a rebalancing
  policy on the Indian delivery settlement schedule, including the flat DP
  sell charge — ₹15.34 per ISIN per day on any day with a sell. That flat fee
  is the cost that dominates small books, and a percentage-only cost model
  misses it entirely. Reports turnover, regulatory / brokerage / DP costs
  separately, and annualized cost drag.

- **Maintenance margin for fully funded positions** — a position covered
  entirely by posted cash no longer contributes a maintenance requirement it
  cannot breach.

### Changed

- **Netting: an order opposing a FLAT instrument now opens** in the order's
  own side, where it was previously read as a close, found no position, and
  was discarded. An order opposing an *open* position still closes it, so
  bracket legs and take-profit orders are unaffected. `reduce_only` orders
  route to the closing branch unconditionally and never open.
- `submit_bracket` marks its stop and target legs `reduce_only`, so a leg
  still working after the position closed by another route cannot open a
  fresh position on the opposite side.
- The kernel's per-instrument `direction` now governs the signal path only
  (`enter()` and the signal arrays). Runs using `direction=` / `directions=`
  without submitting sided orders are bit-identical to 0.5.0.

### Fixed

- Every refused order counts against `rejected_entries`. Previously
  `no_position`, `position_open`, `reduce_only` and `invalid_qty` rejections
  were invisible, so a discarded order looked like an order never placed.
  Sizing refusals (`zero_size`) and unfillable *closes* stay uncounted: they
  are not refused entries.
- An order-path open honors an ATR stop/target config instead of computing
  levels from a hardcoded zero ATR, which silently produced no stop at all.

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

- **Incremental (live) session feed.** `PyPortfolioSession` gains
  `push_tick`, `push_bar`, `push_depth` and `remaining()`: events append to
  the schedule tail in arrival order after `seal()` (idempotent — batch
  warmup data merges ahead of the first push), and the existing
  `current_event()`/`apply_current()` loop drives them. A batch replay and
  a push-per-row stream of the same rows produce identical results.

- **`TickStrategyStream`** — a Python driver for open-ended live feeds.
  Construct with symbols and optional `warmup_bars`, then `push_tick` /
  `push_bar` / `push_depth` as events arrive; every strategy hook a push
  triggers fires before it returns. `finish()` closes out and computes
  metrics. Shares its dispatch loop with `run_tick_strategy`, with one
  addition in streaming sessions: real bars (warmup or pushed) *execute* —
  they match orders and mark equity — unlike bars aggregated from prints
  via `primary_bars`, which remain a view.

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

- **`PortfolioContext` position state matches `StrategyContext`.**
  `position`, `positions` reads for the current symbol, plus `is_flat` /
  `is_net_long` / `is_net_short` / `net_position`, are now PROPERTIES on
  the portfolio and tick contexts, exactly as on the single-instrument
  context — so a bar-style strategy (`if ctx.position is None`) behaves
  identically on the live stream instead of silently seeing a truthy bound
  method and never entering. Cross-symbol lookups are explicit methods:
  `position_for(symbol)`, `positions(symbol=None)`,
  `net_position_for(symbol)`.
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
