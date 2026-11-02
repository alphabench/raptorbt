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
API version: 2.x  
Depends on: none

Load when:

- Starting a RaptorBT project or choosing which API to use.
- Learning the engine lifecycle: data in, config, run, result.
- Setting capital, fees, slippage, fill timing or session length.

## Strategy

Strategy creation, hooks and callback semantics.

Name: `raptorbt-strategy`  
Path: [strategy/SKILL.md](strategy/SKILL.md)  
Status: draft  
API version: 2.x  
Depends on: raptorbt-core

Load when:

- Writing a class-based strategy.
- Choosing hooks (on_start, on_bar, on_stop, order and position events).
- Running one strategy or many over the same data.

## Instruments

Instrument definitions, asset classes, options and multi-instrument runs.

Name: `raptorbt-instruments`  
Path: [instruments/SKILL.md](instruments/SKILL.md)  
Status: stable  
API version: 2.x  
Depends on: raptorbt-core, raptorbt-strategy

Load when:

- Trading several symbols or asset classes.
- Defining futures lot size, tick size, multiplier, expiry and settlement.
- Picking option strikes and expiries from a chain; pricing options, Greeks and implied volatility.
- Building iron condors, straddles and other multi-leg option structures.
- Market hours, trading sessions and holidays.

## Market data

Bars, trades, quotes and order-book inputs, and tick-level features.

Name: `raptorbt-market-data`  
Path: [market-data/SKILL.md](market-data/SKILL.md)  
Status: stable  
API version: 2.x  
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
API version: 2.x  
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
API version: 2.x  
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
API version: 2.x  
Depends on: raptorbt-core, raptorbt-strategy

Load when:

- Computing SMA, EMA, RSI, ATR, MACD, Bollinger bands, ADX, VWAP or Supertrend.
- Registering a streaming indicator on a strategy and waiting for warm-up.
- Writing a custom indicator, or an indicator on a higher timeframe.

## Orders

Order construction, stops, targets and lifecycle.

Name: `raptorbt-orders`  
Path: [orders/SKILL.md](orders/SKILL.md)  
Status: draft  
API version: 2.x  
Depends on: raptorbt-core, raptorbt-strategy, raptorbt-instruments

Load when:

- Submitting or managing orders.
- Implementing stop-losses, targets or brackets.

## Execution

Matching, fill timing, slippage and fee models.

Name: `raptorbt-execution`  
Path: [execution/SKILL.md](execution/SKILL.md)  
Status: draft  
API version: 2.x  
Depends on: raptorbt-core, raptorbt-orders

Load when:

- Choosing fill timing, slippage or partial-fill behaviour.
- Setting fees or a custom rate card.

## Portfolio

Positions, capital, accounts and portfolio construction math.

Name: `raptorbt-portfolio`  
Path: [portfolio/SKILL.md](portfolio/SKILL.md)  
Status: draft  
API version: 2.x  
Depends on: raptorbt-core, raptorbt-strategy, raptorbt-instruments

Load when:

- Running several instruments against one capital pool.
- Using account groups, optimizers, factor panels or rebalancing.

## Risk

Risk limits, margin models and safety controls.

Name: `raptorbt-risk`  
Path: [risk/SKILL.md](risk/SKILL.md)  
Status: draft  
API version: 2.x  
Depends on: raptorbt-core, raptorbt-portfolio

Load when:

- Capping order size, exposure, positions or daily loss.
- Modelling margin.
- Estimating covariance or risk contributions.

## Backtesting

Historical simulation runners.

Name: `raptorbt-backtesting`  
Path: [backtesting/SKILL.md](backtesting/SKILL.md)  
Status: draft  
API version: 2.x  
Depends on: raptorbt-core, raptorbt-instruments

Load when:

- Running a historical simulation from arrays.
- Choosing between single, basket, pairs, spread, options and multi runners.

## Optimization

Parameter sweeps, batch runs and Monte Carlo validation.

Name: `raptorbt-optimization`  
Path: [optimization/SKILL.md](optimization/SKILL.md)  
Status: stable  
API version: 2.x  
Depends on: raptorbt-core, raptorbt-backtesting, raptorbt-results

Load when:

- Sweeping parameters across many runs; grid and random designs.
- Walk-forward testing, the deflated Sharpe ratio and avoiding overfitting.
- Checking robustness with a bootstrap or a portfolio Monte Carlo.

## Live trading

Live runtime and broker-facing integration.

Name: `raptorbt-live-trading`  
Path: [live-trading/SKILL.md](live-trading/SKILL.md)  
Status: draft  
API version: 2.x  
Depends on: raptorbt-core, raptorbt-strategy, raptorbt-market-data, raptorbt-risk

Load when:

- Driving a strategy from a live feed.
- Preparing paper or live operation.

## Results

Performance analysis, trades and diagnostics.

Name: `raptorbt-results`  
Path: [results/SKILL.md](results/SKILL.md)  
Status: stable  
API version: 2.x  
Depends on: raptorbt-core

Load when:

- Reading metrics, equity and drawdown curves, trades, orders and the full report.
- Judging whether a result is trustworthy; why a Sharpe ratio looks huge (annualisation).
- Reconciling a run: fees, P&L, drawdown, open trades.

## Deployment

Runtime configuration, versioning and deployment checks.

Name: `raptorbt-deployment`  
Path: [deployment/SKILL.md](deployment/SKILL.md)  
Status: draft  
API version: 2.x  
Depends on: raptorbt-core, raptorbt-live-trading, raptorbt-risk

Load when:

- Moving a validated strategy to a runtime.
- Pinning versions and checking compatibility.

## Troubleshooting

Structured errors, common failures and debugging.

Name: `raptorbt-troubleshooting`  
Path: [troubleshooting/SKILL.md](troubleshooting/SKILL.md)  
Status: draft  
API version: 2.x  
Depends on: raptorbt-core

Load when:

- A call raised a RaptorBT error.
- Results look wrong or a run was refused.

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
