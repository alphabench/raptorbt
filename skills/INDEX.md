# RaptorBT Skill Registry

Each public API domain has one `SKILL.md` that teaches an agent that domain
and nothing else. Load only what the task needs: read the entry below, load the
skills it names, and follow their `depends_on` first.

From Python, with the package installed (no repository needed):

```python
import raptorbt.skills as skills

skills.for_task("multi-timeframe options strategy with ATR stops")  # skill names, best first
print(skills.load("raptorbt-core"))                                   # a skill's text
skills.manifest()                                                      # the machine registry
```

`manifest.json` is the machine-readable form of this file. Start with
`raptorbt-core`; it teaches the lifecycle every other skill assumes.

`status: draft` means the skill is a skeleton and its examples are not yet
verified. Treat a draft as a pointer to the domain, not as documentation.

## Registry

<!-- registry:start -->

## Core

Engine concepts, the backtest lifecycle (data in, configuration, run, result) and the settings every run shares. Load this first.

Name: `raptorbt-core`  
Path: [core/SKILL.md](core/SKILL.md)  
Status: stable  
API version: 1.x  
Depends on: none

Load when:

- Starting a RaptorBT project or choosing which API to use.
- Learning the engine lifecycle: data in, config, run, result.
- Setting capital, fees, slippage, fill timing or session length.

## Strategy

Writing a class-based strategy: the Strategy class, its lifecycle hooks, StrategyConfig parameters, the requirements manifest, the context and state, and running one or many strategies over the same data.

Name: `raptorbt-strategy`  
Path: [strategy/SKILL.md](strategy/SKILL.md)  
Status: stable  
API version: 1.x  
Depends on: raptorbt-core

Load when:

- Writing a class-based strategy or choosing its hooks (on_init, on_start, on_bar, on_fill, on_stop).
- Giving a strategy parameters (StrategyConfig, param bounds) or declaring what data it needs (requirements).
- Running one strategy or many strategies over the same data.

## Instruments

Instrument definitions, asset classes, options and multi-instrument runs.

Name: `raptorbt-instruments`  
Path: [instruments/SKILL.md](instruments/SKILL.md)  
Status: stable  
API version: 1.x  
Depends on: raptorbt-core, raptorbt-strategy

Load when:

- Trading several symbols or asset classes.
- Defining futures lot size, tick size, multiplier, expiry and settlement.
- Picking option strikes and expiries from a chain; pricing options, Greeks and implied volatility.
- Building iron condors, straddles and other multi-leg option structures.
- Market hours, trading sessions and holidays.
- Trading option contracts while deciding on the index; building a synthetic chain and setting span_pct and exposure_pct for sold options.

## Market data

Bars, trades, quotes and order-book inputs, and tick-level features.

Name: `raptorbt-market-data`  
Path: [market-data/SKILL.md](market-data/SKILL.md)  
Status: stable  
API version: 1.x  
Depends on: raptorbt-core, raptorbt-instruments

Load when:

- Preparing OHLCV arrays, trade ticks, quotes and order-book depth.
- Handling timestamps, bar_stamp, bar open versus close stamps and ordering.
- Checking what a strategy can see at an event (no lookahead).
- Computing tick features or running tick-level simulation.

## Subscriptions

Declaring which streams and bar histories a strategy receives.

Name: `raptorbt-subscriptions`  
Path: [subscriptions/SKILL.md](subscriptions/SKILL.md)  
Status: stable  
API version: 1.x  
Depends on: raptorbt-core, raptorbt-strategy, raptorbt-market-data

Load when:

- Subscribing a strategy to a symbol's trades, quotes or order books.
- Subscribing to 5 minute, hourly or daily bars of a symbol.
- Reading closed-bar history and the last quote or book with ctx.data.

## Timeframes

Multi-timeframe aggregation and composite bars.

Name: `raptorbt-timeframes`  
Path: [timeframes/SKILL.md](timeframes/SKILL.md)  
Status: stable  
API version: 1.x  
Depends on: raptorbt-core, raptorbt-strategy, raptorbt-market-data

Load when:

- Combining 1m, 5m, 15m, hourly or daily data.
- Turning 1 minute bars into 5, 15 or 60 minute bars, or into daily bars.
- Building bars from ticks: time, tick-count, volume, value or Renko bars.
- Anchoring bars at the market open; resampling, aggregating and bar grids.
- Choosing which edge of a bar a boundary record belongs to.

## Indicators

Built-in vectorised and streaming indicators.

Name: `raptorbt-indicators`  
Path: [indicators/SKILL.md](indicators/SKILL.md)  
Status: stable  
API version: 1.x  
Depends on: raptorbt-core, raptorbt-strategy

Load when:

- Computing SMA, EMA, RSI, ATR, MACD, Bollinger bands, ADX, VWAP or Supertrend.
- Registering a streaming indicator on a strategy and waiting for warm-up.
- Writing a custom indicator, or an indicator on a higher timeframe.

## Expressions

Writing a signal as one formula (close > sma(close, 200)), as text or from Python operators, and evaluating it over whole columns or one bar at a time with the same numbers. Covers unknown values (NaN and Kleene logic), warm-up, no-lookahead and the function list.

Name: `raptorbt-expressions`  
Path: [expressions/SKILL.md](expressions/SKILL.md)  
Status: stable  
API version: 1.x  
Depends on: raptorbt-core, raptorbt-indicators

Load when:

- Writing a condition or derived series as one formula, as text or from Python operators.
- Evaluating a formula over a stored history and bar by bar with identical numbers.
- Looking up which expression functions exist, their warm-up and how they treat unknown values.

## Orders

Order construction and lifecycle: the typed order kinds, sizing forms, time in force, brackets, OCO links, modify and cancel, option structures, and the reasons an order is refused.

Name: `raptorbt-orders`  
Path: [orders/SKILL.md](orders/SKILL.md)  
Status: stable  
API version: 1.x  
Depends on: raptorbt-core, raptorbt-strategy, raptorbt-instruments

Load when:

- Submitting or managing orders: limit, stop, stop-limit, trailing, bracket, OCO, modify, cancel.
- Implementing a stop-loss, target or time in force (day, gtd, ioc, fok).
- Explaining why an order was rejected, or placing a multi-leg option structure.
- Putting an ATR stop or target on an option leg: the premium or the index.

## Execution

Where a strategy's orders go: the ExecutionAdapter protocol, requests and execution reports, the simulated and paper venues, how a venue's fills are booked as facts, the host's order check and order tags, and what differs between paper and the in-process matcher.

Name: `raptorbt-execution`  
Path: [execution/SKILL.md](execution/SKILL.md)  
Status: stable  
API version: 1.x  
Depends on: raptorbt-core, raptorbt-orders

Load when:

- Connecting a strategy to a venue: writing an ExecutionAdapter, or turning a venue's messages into execution reports.
- Understanding what a fill, cancel or replace report does to the book, including fills that breach a limit or arrive out of order.
- Validating on PaperExecution before real money, or explaining why paper and a backtest differ.
- Installing the host's order check or attaching tags to orders.

## Portfolio

The account a strategy trades against: positions, cash, equity and free capital, netting and hedging, margin accounts, shared account groups, booked fixed-point money, and portfolio construction math.

Name: `raptorbt-portfolio`  
Path: [portfolio/SKILL.md](portfolio/SKILL.md)  
Status: stable  
API version: 1.x  
Depends on: raptorbt-core, raptorbt-strategy, raptorbt-instruments

Load when:

- Reading positions, cash, equity or free capital; netting versus hedging; margin versus cash accounts.
- Running several strategies on one shared account (AccountSpec) or several instruments against one capital pool.
- Booked money (exact minor units), or turning forecasts into target weights (optimizer, factor panels, rebalancing).

## Risk

Risk limits, kill switches, margin models and position sizing helpers: what refuses an order, what halts a strategy, and how many lots to trade.

Name: `raptorbt-risk`  
Path: [risk/SKILL.md](risk/SKILL.md)  
Status: stable  
API version: 1.x  
Depends on: raptorbt-core, raptorbt-portfolio

Load when:

- Capping order size, position, exposure, open positions, daily loss or drawdown; adding a kill switch.
- Sizing a position in lots from capital, risk, volatility or margin.
- Modelling margin, or measuring covariance and risk contributions.

## Backtesting

Historical simulation: choosing a runner, configuring fills, fees, slippage and latency, expiry and corporate actions in a backtest, and keeping a run deterministic.

Name: `raptorbt-backtesting`  
Path: [backtesting/SKILL.md](backtesting/SKILL.md)  
Status: stable  
API version: 1.x  
Depends on: raptorbt-core, raptorbt-instruments

Load when:

- Running a historical simulation and choosing between single, basket, multi, pairs, options and spread runners.
- Setting fees, exchange charges (fee_segment, RateCard), slippage, fill timing, latency or stochastic fills.
- Expiry, settlement and corporate actions (splits, dividends) in a backtest; making a run deterministic.

## Optimization

Parameter sweeps, batch runs and Monte Carlo validation.

Name: `raptorbt-optimization`  
Path: [optimization/SKILL.md](optimization/SKILL.md)  
Status: stable  
API version: 1.x  
Depends on: raptorbt-core, raptorbt-backtesting, raptorbt-results

Load when:

- Sweeping parameters across many runs; grid and random designs.
- Walk-forward testing, the deflated Sharpe ratio and avoiding overfitting.
- Checking robustness with a bootstrap or a portfolio Monte Carlo.

## Live trading

Running strategies on a pushed feed with LiveNode: arrival times, warm-up, adopted positions and corporate actions, the journal, checkpoint, restore, reconciliation and fencing, broker adapters and their conformance kit, percent and group exits, continuous futures, and the instruments a context answers for.

Name: `raptorbt-live-trading`  
Path: [live-trading/SKILL.md](live-trading/SKILL.md)  
Status: stable  
API version: 1.x  
Depends on: raptorbt-core, raptorbt-strategy, raptorbt-market-data, raptorbt-risk, raptorbt-execution

Load when:

- Driving a strategy from a live feed with LiveNode, on paper or against a broker.
- Warming up, adopting positions an account already holds (signed seeds with posted margin) or declaring corporate actions on a node.
- Making a process survive a crash: the journal, checkpoint, restore, reconcile, resume, resend and fencing.
- Connecting Kite, Upstox or Dhan through raptorbt.adapters, or testing an adapter with the conformance kit.
- Percent and group exits, continuous futures and rolls, ctx.instrument, ctx.option_chain and ctx.role on a node.

## Results

Performance analysis, trades and diagnostics.

Name: `raptorbt-results`  
Path: [results/SKILL.md](results/SKILL.md)  
Status: stable  
API version: 1.x  
Depends on: raptorbt-core

Load when:

- Reading metrics, equity and drawdown curves, trades, orders and the full report.
- Judging whether a result is trustworthy; why a Sharpe ratio looks huge (annualisation).
- Reconciling a run: fees, P&L, drawdown, open trades.

## Deployment

What a host must provide and decide to run a validated strategy for real: admitting a strategy from its requirements manifest, resolving its roles into symbols, building the node with bindings, a venue adapter, a journal and an epoch, supervising it, recovering it, and what to verify before real money.

Name: `raptorbt-deployment`  
Path: [deployment/SKILL.md](deployment/SKILL.md)  
Status: stable  
API version: 1.x  
Depends on: raptorbt-core, raptorbt-strategy, raptorbt-live-trading, raptorbt-execution, raptorbt-risk

Load when:

- Moving a validated strategy toward paper or real money, or running strategies other people wrote.
- Admitting a strategy from its requirements manifest: check_requirements, params_schema, requirements_hash, resolve and role bindings.
- Deciding what a worker must hold: instruments, bindings, journal and fsync, epoch and fencing, warm-up history, health and heartbeat.
- Writing the recovery runbook, or the checklist of what to verify before live trading with real money.
- Pinning versions and the once-a-day release check.

## Troubleshooting

Structured errors by symptom: how to read a RaptorBT error, the codes a run, an order, a live node, a recovery and an adapter raise or report, how to read a journal, the mistakes that recur across skills, and the engine gaps known today.

Name: `raptorbt-troubleshooting`  
Path: [troubleshooting/SKILL.md](troubleshooting/SKILL.md)  
Status: stable  
API version: 1.x  
Depends on: raptorbt-core

Load when:

- A call raised a RaptorBT error: read code, parameter, valid and skill, then find the code by symptom.
- An order was rejected, a run was refused, or results look wrong.
- A live node raised fenced, recovering, replay_diverged, node_poisoned, venue_unavailable or journal_failed, or an adapter reported unconfirmed or session_expired.
- Reading a journal to reconstruct what happened to an order.
- Checking the known engine gaps before relying on a behaviour.

<!-- registry:end -->

## Changing an API and its skill together

A skill is a contract, not prose: `python -m raptorbt.skills check` runs in CI
and fails when a skill stops matching the package. A change that touches the
public Python API is not done until its skill is.

1. Change the code. Change the public names in `raptorbt/__init__.py` (or the
   submodule's `__all__`) in the same commit.
2. Find the skill: the `modules` lists in `manifest.json` say which skill covers
   which symbol. A new public symbol or module goes into some skill's
   `modules`. `python -m raptorbt.skills check --strict` lists what is not
   covered yet.
3. Update that skill's text and examples. Every `python` block runs, with the
   network blocked and `DeprecationWarning` as an error. To show invalid
   usage, put `<!-- skill: invalid -->` on the line above the fence; that block
   must then fail when run (and not by a syntax error).
4. Deprecating a name: say so on the line that mentions it, and remove it from
   the examples. The checker rejects a deprecated name in an example.
5. A new skill: copy `_template/SKILL.md` to `<domain>/SKILL.md`, add an entry to
   `manifest.json` (`name` is `raptorbt-<domain>`, `path` is
   `<domain>/SKILL.md`), then run `python -m raptorbt.skills index --write` to
   refresh the registry above. If errors should point at it, add the domain to
   `raptorbt.errors.SKILLS`.
6. Run `python -m raptorbt.skills check` and
   `pytest tests/skill_validation`, and add a `CHANGELOG.md` line under
   `[Unreleased]`.

Frontmatter fields are fixed: `name`, `description`, `version`, `api_version`,
`depends_on`, `status` (`draft` or `stable`). They must equal the manifest
entry. Keep skills free of any reference to a particular broker, platform or
deployment; examples use generic symbols.

## Examples

Complete, runnable scripts in `examples/`, one folder per kind of
strategy. Each runs offline on deterministic synthetic data, passes `bar_stamp`
explicitly, prints a short result, and names the skills it uses in its header.
`tests/python/test_examples.py` runs every one on each test run. Find the task, read
the skills it names, then copy the script.

| Task | Example |
| --- | --- |
| A single-instrument strategy with indicators, fees, a stop and a target, and its results | `examples/single-instrument/sma_cross.py` |
| Two instruments read together in one decision (the equal-timestamp group rule) | `examples/multi-instrument/relative_value.py` |
| 1-minute entries confirmed by a 15-minute trend (composite bars) | `examples/multi-timeframe/trend_confirmation.py` |
| An options strategy on a chain: strike and expiry picks, ATR stops, sizing, account limits | `examples/options/nifty_momentum.py` |
| A hedge-first multi-leg structure (bear call spread) with a defined-risk margin model | `examples/options/bear_call_spread.py` |
| Several strategies on a shared account with drawdown, daily-loss and halt limits | `examples/portfolio/shared_account.py` |
| A basket strategy over instruments the deployer picks (a `Universe` role), with a warm-up session before `trade_from` | `examples/portfolio/top_n_basket.py` |
| A tick strategy with quotes and resting limit orders | `examples/tick-level/passive_quotes.py` |
| The same strategy on a paper `LiveNode`, matching the batch run, with a journal checkpoint and restore | `examples/live-trading/paper_node.py` |
