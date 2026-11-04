---
name: raptorbt-portfolio
description: >
    The account a strategy trades against: positions, cash, equity and free
    capital, netting and hedging, margin accounts, shared account groups,
    booked fixed-point money, and portfolio construction math.
version: 1.0
api_version: 1.x
depends_on:
    - raptorbt-core
    - raptorbt-strategy
    - raptorbt-instruments
status: stable
---

# RaptorBT Portfolio

## 1. When to use this skill

Load it when you need to know what an account holds and can still spend, run
several instruments or several strategies against one pool of capital, choose
a cash or margin account, read exact booked money, or turn return forecasts
into target weights. The limits that refuse orders are in [risk](../risk/SKILL.md);
how an order reaches the account is in [orders](../orders/SKILL.md).

## 2. Domain overview

**An account** holds cash, open positions and (on a margin account) locked
margin. Every strategy has its own account unless you put several on a shared
one. From inside a hook you read it through the context:

| Read | Meaning |
| --- | --- |
| `ctx.cash` | Booked cash. On a cash account a long position has already taken its cost out of it. |
| `ctx.equity` | Cash plus the mark-to-market of open positions, as of the last completed bar. |
| `ctx.free_capital` | What can still fund a new position: cash less margin locked (all of cash on a cash account). |
| `ctx.position`, `ctx.is_flat` | The current instrument's earliest open position, or `None`; `is_flat` is true when it holds none. |
| `ctx.net_position` | Signed units of the current instrument (hedged positions net out). |

In a portfolio run (`run_strategies`) `ctx.positions(symbol=None)`,
`ctx.position_for(symbol)`, `ctx.net_position_for(symbol)` and `ctx.symbols`
address any instrument, and `ctx.portfolio` gives the strategy's whole book:
`positions` (a dict by symbol), `cash`, `equity`, `available_capital`,
`locked_margin`, `gross_exposure`, `net_exposure`, `realized_pnl`,
`unrealized_pnl`. `ctx.positions()` is a method in every runner; in a
single-instrument run the `symbol` argument is ignored. Using the attribute as
a list there (`for p in ctx.positions`) still works and warns.

**Cash and margin accounts.** `account_type="cash"` (default) pays for a
position in full. `account_type="margin", leverage=N` locks initial margin
instead of notional, so cash stays and `free_capital` falls by the lock; an
equity breach of the maintenance requirement fires `on_margin_call` and halts
new entries. A `MarginModel` (in `BacktestConfig(margin_model=...)`) replaces
the built-in rule: `fraction_of_notional(fraction)` (`0.2` is 20 percent; a
value above 1 is refused; the older `percent_of_notional(pct)` takes percent
and warns), `leverage({"CASH": 5})`, `cash_only()`, `no_margin()`,
`from_spec(leverage)`, `option_simple`, `defined_risk`, `spread_credit` and
others. A model needs an instrument definition (`InstrumentSpec`) so it can
classify the line; without one the entry is refused as `margin_unpriced`.

**Netting and hedging.** `oms_type="netting"` (default) keeps one position per
instrument and moves it like a broker's net position: an order on its side
adds (average entry price), an order against it reduces, closes or flips it;
`"hedging"` lets every order open its own independent position, each with its
own stop and target, listed in `ctx.positions()`. See [orders](../orders/SKILL.md)
for how orders behave under each.

**A basket.** One strategy over N instruments chosen at deploy time declares
`Universe(asset=, min=, max=)` in its `requirements` and reads the chosen list
with `ctx.role("basket")`; the run binds it with
`run_strategies(..., bindings={"basket": [...]})`. `trade_from=<ns>` runs the
bars before it without trading them (orders decided earlier are refused
`warmup`), so momentum and other indicators are primed when trading starts.
`examples/portfolio/top_n_basket.py` runs both.

**Account groups.** Strategies can share an account:
`run_strategies(..., accounts={"desk": AccountSpec(...)})`, with the strategy
entry `(strategy, symbols, {"account": "desk"})`. They share cash, margin and
the account-level limits (`max_positions`, `max_gross_exposure`, daily loss,
drawdown). Within one event they step in the order of the list, which decides
who gets scarce cash. A group of one is bit-identical to an isolated strategy.
A strategy on an account has an *attributed* equity curve: an equal share of
the account's capital, moving only with its own fills. `ctx.cash` and
`ctx.free_capital` are the account's, since that is what it can spend.

**Booked money.** Cash, fees and realised P&L are integer counts of minor
units, at `BacktestConfig(money_decimals=2)` by default (100 minor units to the
unit; range 0 to 9), rounded half to even where an amount enters the books.
Prices, quantities, marks, equity, margin and every statistic stay floating
point. A `Trade` carries the exact companions `pnl_minor`, `fees_minor`,
`entry_notional_minor`, `exit_notional_minor` and `money_decimals`, with
`pnl == pnl_minor / 10**money_decimals`. Cash cannot be overdrawn by
rounding. `docs/quantitative-correctness/numerics.md` lists every conversion.

**Portfolio construction.** Separate from the event engine: estimate a
shrunk covariance (`estimate_covariance`), then `optimize_portfolio` turns
alphas, current weights and constraints into target weights; factor panels
(`momentum_panel`, `zscore_panel`, `rank_panel`, `winsorize_panel`,
`composite_scores`, `rank_ic`) build cross-sectional scores; and
`simulate_rebalance_policy` prices a rebalancing rule.

## 3. API reference

| Name | What it is |
| --- | --- |
| `raptorbt.AccountSpec(initial_capital, account_type="cash", leverage=1.0, max_drawdown_pct=None, max_daily_loss=None, max_daily_loss_pct=None, max_gross_exposure=None, max_positions=None)` | One shared account. Raises `invalid_account_spec` on a bad field. Up to 500 strategies per account. |
| `raptorbt.run_strategies(..., accounts={name: AccountSpec})` | With `accounts` the return is `(results, {name: AccountResult})`. |
| `raptorbt.run_portfolio_strategy(strategy, data, ...)` | One strategy across many symbols; the same runtime as `run_strategies`. |
| `raptorbt.run_portfolio_backtest(instruments, config=None, allocation="equal_weight", instrument_configs=None)` | Array-based portfolio of instruments with entry and exit arrays; `allocation` is `"equal_weight"` or `"full"`. |
| `raptorbt.PortfolioResult` | `.result` (a `BacktestResult`), `.metrics`, `.per_instrument` (`InstrumentSummary`: `symbol`, `trades`, `pnl`, `rejected_entries`), `.rejected_entries`, `.halted`, `.halted_at`. |
| `raptorbt.AccountResult` | `name`, `strategies`, `initial_capital`, `balance`, `halted`, `halted_at`, and the curves `equity_curve()`, `drawdown_curve()`, `locked_margin()`, `timestamps()`. |
| `raptorbt.InstrumentSummary` | One row of `per_instrument`. |
| `raptorbt.margin.MarginModel`, `raptorbt.MarginModel` | Margin rules; `.required(spec, quantity, price)`, `.required_legs(...)`, `.describe()`. |
| `raptorbt.estimate_covariance(returns, asset_ids, periods_per_year)` | A shrunk covariance `RiskModel` (see [risk](../risk/SKILL.md)). |
| `raptorbt.OptimizerConfig(risk_aversion, turnover_penalty, position_cap, sector_ids, sector_caps, no_trade_band=, min_trade_value=, portfolio_value=, cash_max=, max_iter=, tolerance=, short_cap=, gross_max=, net_min=, net_max=)` | Optimizer settings. A `short_cap > 0` enables long/short. |
| `raptorbt.optimize_portfolio(model, alpha, w_current, asset_ids, config)` | `OptimizationResult`: `weights()`, `trades()`, `turnover`, `vol_annualized`, `objective`, `solver_status`, `snapped`, `cash`, `gross_exposure`, `net_exposure`. |
| `raptorbt.batch_optimize_portfolios(model, items, config)`, `raptorbt.OptimizeItem(item_id, alpha, w_current, portfolio_value=None)` | Many optimizations over one model; results in input order. |
| `raptorbt.momentum_panel(prices, lookback, skip)`, `zscore_panel(values, min_names)`, `rank_panel(values, min_names)`, `winsorize_panel(values, pct)`, `composite_scores(factors, weights)`, `rank_ic(factor, prices, horizon, min_names)` | Cross-sectional factor tools over a `(dates, names)` array. `rank_ic` returns `RankIC` (`mean_ic`, `t_stat`, `t_stat_deflated`, `n_dates_scored`, ...). |
| `raptorbt.simulate_rebalance_policy(prices, target_weights, initial_capital, policy, policy_param, ...)` | Cost and turnover of a rebalancing rule; returns `RebalanceSimResult`. |

## 4. Standard workflow

1. Choose the account: cash or margin, netting or hedging, isolated or shared.
2. Give each instrument a definition (`InstrumentSpec`) when margin, lots,
   ticks or expiry matter ([instruments](../instruments/SKILL.md)).
3. Size from the account, not from a constant: `ctx.free_capital` for what can
   be spent, `ctx.equity` for what to allocate against.
4. Run; read `PortfolioResult.per_instrument` and `rejected_entries` to see
   what the account refused.
5. For shared accounts, read the `AccountResult` as well as each strategy's
   attributed result.
6. For target weights, estimate the covariance, optimize, then trade towards
   the weights in the strategy.

## 5. Examples

Cash against margin. The same 500-unit buy: a cash account pays 50,000 out of
cash; a margin account at 5x keeps its cash and locks a tenth of the notional.
A model changes the lock, and an order bigger than the account can fund is
refused with a reason:

```python
import numpy as np
from raptorbt import (BacktestConfig, InstrumentSpec, MarginModel, Strategy,
                      run_strategy_backtest)
from raptorbt.strategy import orders

n = 10
close = np.full(n, 100.0)
tape = dict(timestamps=np.arange(n, dtype=np.int64) * 60_000_000_000, open=close,
            high=close + 1.0, low=close - 1.0, close=close, volume=np.full(n, 1e6))


class Buy(Strategy):
    def __init__(self, quantity):
        super().__init__()
        self.quantity, self.books, self.rejected = quantity, [], []

    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.submit_order(orders.Market(side="buy", quantity=self.quantity))
        if ctx.idx == 3:
            self.books.append((ctx.cash, ctx.equity, ctx.free_capital))

    def on_order_rejected(self, ctx, event):
        self.rejected.append(event.reject_reason)


def run(quantity, model=None, **account):
    s = Buy(quantity)
    run_strategy_backtest(s, **tape, instrument=InstrumentSpec.equity("X"), bar_stamp="close",
                          config=BacktestConfig(fees=0.0, initial_capital=100_000.0, margin_model=model),
                          **account)
    return s


assert run(500).books == [(50_000.0, 100_000.0, 50_000.0)]                       # paid in full
assert run(500, account_type="margin", leverage=5.0).books == [(100_000.0, 100_000.0, 90_000.0)]
five_x = MarginModel.leverage({"CASH": 5})
assert run(4_000, five_x, account_type="margin", leverage=2.0).books[0][2] == 20_000.0
assert run(6_000, five_x, account_type="margin", leverage=2.0).rejected == ["insufficient_capital"]

spec = InstrumentSpec.equity("X")
assert MarginModel.fraction_of_notional(0.2).required(spec, 500, 100.0) == 10_000.0   # 20 percent of 50,000
```

Two strategies on one account share its cash and its position cap. The second
strategy's entry is refused because the account already holds a position:

```python
import numpy as np
from raptorbt import AccountSpec, BacktestConfig, Strategy, run_strategies


def bars(seed, base, n=40):
    rng = np.random.default_rng(seed)
    close = base + np.cumsum(rng.normal(0, 1, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    ts = (np.arange(n, dtype=np.int64) + 1) * 60_000_000_000 + 1_700_000_000_000_000_000
    return dict(timestamps=ts, open=open_, high=np.maximum(open_, close) + 0.5,
                low=np.minimum(open_, close) - 0.5, close=close, volume=np.full(n, 1000.0))


class Buy(Strategy):
    def __init__(self, symbol):
        super().__init__()
        self.symbol = symbol

    def on_bar(self, ctx):
        if ctx.symbol == self.symbol and ctx.idx == 3:
            self.enter_long(size_frac=0.2)


data = {"AAA": bars(1, 100.0), "BBB": bars(2, 50.0)}
results, accounts = run_strategies(
    [(Buy("AAA"), ["AAA"], {"account": "desk"}), (Buy("BBB"), ["BBB"], {"account": "desk"})],
    data, BacktestConfig(fees=0.0), bar_stamp="close",
    accounts={"desk": AccountSpec(initial_capital=100_000.0, max_positions=1)},
)
assert [r.result.metrics.total_trades for r in results] == [1, 0]          # first stepped wins the slot
assert results[1].per_instrument[0].rejected_entries == 1
desk = accounts["desk"]
assert desk.strategies == [0, 1] and desk.initial_capital == 100_000.0 and not desk.halted
assert len(desk.equity_curve()) == 40
```

Booked money. A trade's P&L is an integer number of minor units and the float
is its view; `money_decimals` sets the scale:

```python
import numpy as np
from raptorbt import BacktestConfig, Strategy, run_strategy_backtest
from raptorbt.strategy import orders

close = np.array([100.0, 100.0, 100.333, 101.777, 101.777, 101.777])
tape = dict(timestamps=np.arange(6, dtype=np.int64) * 60_000_000_000, open=close,
            high=close + 1.0, low=close - 1.0, close=close, volume=np.full(6, 1e6))


class RoundTrip(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.submit_order(orders.Market(side="buy", quantity=33))
        if ctx.idx == 4:
            self.submit_order(orders.Market(side="sell"))


for decimals in (0, 2, 4):
    r = run_strategy_backtest(RoundTrip, **tape, bar_stamp="close",
                              config=BacktestConfig(fees=0.00123, money_decimals=decimals))
    t = r.trades()[0]
    assert t.money_decimals == decimals
    assert t.pnl == t.pnl_minor / 10**decimals                      # the float is the integer's view
    assert isinstance(t.pnl_minor, int) and isinstance(t.fees_minor, int)
```

Netting, hedging and the book view. Under hedging a buy and a sell are two
positions, and in a portfolio run `ctx.portfolio` totals them:

```python
import numpy as np
from raptorbt import BacktestConfig, Strategy, run_strategies
from raptorbt.strategy import orders

n = 8
close = np.full(n, 100.0)
data = {"AAA": dict(timestamps=np.arange(1, n + 1, dtype=np.int64) * 60_000_000_000, open=close,
                    high=close + 1.0, low=close - 1.0, close=close, volume=np.full(n, 1e6))}


class Both(Strategy):
    def on_bar(self, ctx):
        if ctx.idx == 1:
            self.submit_order(orders.Market(side="buy", quantity=10))
            self.submit_order(orders.Market(side="sell", quantity=4))
        if ctx.idx == 3:
            self.seen = (len(ctx.positions()), ctx.net_position, ctx.portfolio.gross_exposure,
                         ctx.portfolio.net_exposure)


s = Both()
run_strategies([s], data, BacktestConfig(fees=0.0), oms_type="hedging", bar_stamp="close")
assert s.seen == (2, 6.0, 1_400.0, 600.0)          # two positions; net 6 units; 14 x 100 gross, 6 x 100 net
```

Portfolio construction: shrink a covariance, then ask the optimizer for
weights under a position cap and sector caps. The weights respect the caps and
sum to one:

```python
import numpy as np
import raptorbt as rb

ids = ["A", "B", "C", "D"]
rng = np.random.default_rng(11)
returns = rng.normal(0, 0.01, (400, 1)) * np.linspace(0.2, 1.0, 4) + rng.normal(0, 0.012, (400, 4))
model = rb.estimate_covariance(returns, ids, 252.0)

config = rb.OptimizerConfig(risk_aversion=5.0, turnover_penalty=0.0, position_cap=0.4,
                            sector_ids=[0, 0, 1, 1], sector_caps=[0.7, 0.7], cash_max=0.0)
alpha = np.array([0.04, 0.02, 0.0, -0.02])
result = rb.optimize_portfolio(model, alpha, np.full(4, 0.25), ids, config)

w = result.weights()
assert result.solver_status == "Solved"
assert abs(w.sum() - 1.0) < 1e-6 and w.max() <= 0.4 + 1e-6
assert w[0] + w[1] <= 0.7 + 1e-6 and w[2] + w[3] <= 0.7 + 1e-6
assert w[0] > w[3]                                  # the higher alpha gets the larger weight
```

## 6. Composition

- A strategy reads the account through the context; its orders are in
  [orders](../orders/SKILL.md).
- Limits applied at the account (`max_positions`, daily loss, drawdown,
  gross exposure), halts and sizing helpers: [risk](../risk/SKILL.md).
- Instrument definitions that make margin, lots and expiry real:
  [instruments](../instruments/SKILL.md).
- Exchange charges that hit the booked cash: [backtesting](../backtesting/SKILL.md).
- Reading P&L and the curves: [results](../results/SKILL.md).
- Runnable examples: `examples/portfolio/shared_account.py`, `examples/portfolio/top_n_basket.py`, `examples/multi-instrument/relative_value.py`.

## 7. Constraints and correctness

- `ctx.equity` is as of the last completed bar and marks open positions at
  their last price; it does not move inside a bar.
- On a shared account the account-level checks run after each strategy's step,
  on the marks every strategy has so far; a strategy stepped later in the same
  event is still on its previous mark when an earlier one's check runs.
- On a shared account, the call's `initial_capital`, `account_type`,
  `leverage` and account-level limits are ignored for its strategies; the
  `AccountSpec`'s apply. `flatten_on_daily_loss` and
  `liquidate_on_margin_call` are the config's and apply to the account.
- A margin model on a cash account is refused (`invalid_margin_model`); a
  margin account's entry that needs more than is free is refused with
  `insufficient_margin` (capital-fraction sizing) or `insufficient_capital`
  (an explicit quantity).
- Margin requirements, marks and equity are not booked; only cash, fees and
  realised P&L are. A position still open at the end is marked at its exact
  price, so the last equity point can differ from booked cash by under half a
  minor unit per amount.
- The optimizer refuses mismatched asset order (`asset ids mismatch`),
  infeasible caps (4 assets each capped at 0.2 cannot sum to 1) and non-finite
  alphas. Panels accept `nan` (missing) but refuse infinities.
- A factor score is a number on a date. Impose your own lag between a score
  and the trade it drives.

## 8. Common mistakes

A margin model on a cash account:

<!-- skill: invalid -->
```python
import numpy as np
from raptorbt import BacktestConfig, MarginModel, Strategy, run_strategy_backtest

x = np.full(5, 100.0)
run_strategy_backtest(Strategy, np.arange(5, dtype=np.int64), x, x, x, x, x, bar_stamp="close",
                      config=BacktestConfig(margin_model=MarginModel.leverage({"CASH": 5})))
```

Optimizer assets in a different order from the model:

<!-- skill: invalid -->
```python
import numpy as np
import raptorbt as rb

ids = ["A", "B"]
model = rb.estimate_covariance(np.random.default_rng(1).normal(0, 0.01, (100, 2)), ids, 252.0)
cfg = rb.OptimizerConfig(5.0, 0.0, 1.0, [0, 0], [1.0])
rb.optimize_portfolio(model, np.zeros(2), np.full(2, 0.5), ["B", "A"], cfg)   # ValueError
```

- Reading `ctx.positions` as a list: call `ctx.positions()`, in every runner.
- Passing `percent_of_notional(0.2)` meaning 20 percent: it is 0.2 percent.
  Use `fraction_of_notional(0.2)`, which refuses `20`.
- Expecting `ctx.free_capital` to equal `ctx.equity` on a margin account with
  positions open: it is cash less the lock.
- Giving `run_strategies` a shared account and expecting the config's
  `initial_capital` to apply.
- Comparing `pnl` from two runs with different `money_decimals`.
- Using a margin model without `instrument=`/`instruments=`: every entry is
  refused as `margin_unpriced`.

## 9. Validation

1. After a run, check that `sum(t.pnl for t in trades)` matches the change in
   cash once flat: `final_cash == initial + sum(pnl)` for booked amounts.
2. Compare a shared-account strategy of one against the same strategy on its
   own account: results must be identical.
3. Print `PortfolioResult.per_instrument` and `rejected_entries`: a high count
   means sizing or margin, not luck.
4. For an optimizer, assert weights sum to one (or to `1 - cash`), respect
   every cap, and turnover against the current book is what you expect.
5. Before live use, reconcile positions and cash with the venue
   ([live-trading](../live-trading/SKILL.md)).

## 10. Related skills

- [Strategy](../strategy/SKILL.md)
- [Orders](../orders/SKILL.md)
- [Risk](../risk/SKILL.md)
- [Instruments](../instruments/SKILL.md)
- [Results](../results/SKILL.md)
