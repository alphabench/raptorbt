# Numerics: how prices, quantities and money are converted

One rule per conversion boundary. Each names the code that enforces it.

## What is booked, and what is f64

Money that moves is **booked**: the cash balance, the notional of a fill, a
fee, and realised P&L are integer counts of minor units (`Money`, an `i128`,
in `src/accounts/money.rs`) at a scale declared once per run,
`BacktestConfig(money_decimals=2)`. Two decimals is a rupee or a dollar at 100
minor units; zero suits a whole-unit currency, four a crypto quote. The range
is 0 to 9 decimals. `i128` leaves room for 1e36 whole units at two decimals,
so a run on garbage prices books garbage amounts without wrapping; a sum that
would overflow panics with a message rather than wrapping, and an amount past
2^120 minor units, or one that is not finite, is refused at the boundary.

Everything else stays `f64`: prices, quantities (fractional for crypto and FX),
contract multipliers, percentage rates, marks and the unrealised side of
equity, margin requirements, every indicator, and every statistic (Sharpe,
drawdown, the equity curve itself). They are measurements or inputs, not
amounts the account holds. Timestamps are `i64` nanoseconds and counts
`usize`.

An amount crosses from `f64` to booked exactly once, through
`MoneyScale::money` (`src/accounts/money.rs`), which calls `to_minor`
(`src/execution/costs/rounding.rs`, the one place a number is rounded to
money). It crosses back through `MoneyScale::to_f64` (`minor / 10^places`,
the double nearest the exact decimal). Python-visible results keep their float
types; the exact integers ride alongside as `Trade.pnl_minor`, `fees_minor`,
`entry_notional_minor`, `exit_notional_minor` and `money_decimals`, with
`pnl == pnl_minor / 10**money_decimals`.

### Every boundary, and its rule

Rounding is half to even unless the row says otherwise. A value within 16 ulps
of a tie counts as that tie (`round_scaled`), so `1.005` with a representation
of `100.49999999999999` rounds as the decimal `100.5` would.

| Boundary | What is booked | Rule |
| --- | --- | --- |
| Initial capital | `Money` cash | rounded to the scale, half even |
| Entry fill | notional `price * size * multiplier` (long and short) | rounded half even; this is the position's cost basis |
| Entry cost | the fee model's charge for the fill | rounded half even once, after the model (a rate card row's own rounding runs first, and booking rounds the result again) |
| Add to a position | the slice's notional and fee | each rounded on its own; the cost basis is their integer sum |
| Partial close | the closed slice's share of cost basis and entry fees | `round_half_even(booked * qty / size)`; the open remainder keeps `booked - share`, so the parts add back exactly |
| Exit fill | notional `exit_price * size * multiplier` | rounded half even |
| Exit cost | the fee model's charge | rounded half even |
| Realised P&L | integer: `exit - cost` for a long, `cost - exit` for a short, less booked entry and exit fees | no rounding: a difference of booked integers |
| Cash, cash account | long: `-cost - entry_fee`, then `+exit - exit_fee`. Short: collateral `-cost` on entry, `+(2*cost - exit) - exit_fee` on close (collateral back plus the gain) | integer sums |
| Cash, margin account | entry debits the fee only, close adds `pnl + entry_fees` (gross P&L less the exit fee) | integer sums; the lock is not cash |
| Settlement at expiry | the position closes as an exit fill at the settlement price: notional `settle_price * size * multiplier`, rounded half even | the price is data (declared, the underlying's, or the contract's last), never rounded; a physically settled option closes at `0` and the exercise value is what is delivered against |
| Settlement charge | the settlement row's charge on `settle_value * multiplier * size` (closing side), else the spec's `settlement_fee * settled value` | rounded half even once; the row's own rounding runs first, as for any fee row |
| Physical delivery | the underlying's notional `strike * units * multiplier` as an entry fill (or the exit of an opposite line), and the fill's fee | the same entry/exit rules; a delivery is not an order, so no slippage and no risk gate |
| Split or bonus | none for the held quantity: the position's cost basis (`Money`) is untouched, size multiplies by the ratio and prices divide by it in `f64` | cost basis is invariant, so cash and realised P&L reconcile exactly across the action |
| Cash in lieu of a fraction | the fraction closes as a trade slice at `mark / ratio`: its notional rounded half even, its cost the share `round_half_even(cost * fraction / size)`, no fee | the same partial-close rule; only when the instrument trades in whole units (`size_increment == 1` on a cash instrument) |
| Cash dividend | `round_half_even(amount * signed units * multiplier)` | one conversion (`to_minor`) at the ex-date from the signed holding; credited as an integer on the pay date; a short's is negative |
| Adopted position | cost basis, no fee | rounded half even; an amount that cannot be booked is an `Err` |
| Cash lent to a kernel by a portfolio | the pool's booked balance, less other kernels' locks in margin mode | the lock is `f64`, so the lent amount is rounded half even; the movement is taken back as an integer difference, so the pool is exact |
| `f64` views (`cash()`, `Trade.pnl`, `fees`) | none | `minor / 10^places` |
| Equity | `cash` view plus the `f64` mark of open positions | not booked: a position still open is marked at its exact price |
| Margin requirement and locks | none | `f64`, never rounded: it gates entries and marks maintenance, it is not a charge |
| Statistics and indicators | none | `f64` |

The invariants these give, tested with integer equality
(`tests/accounting_money.rs`, random fill sequences with partial fills, fees,
shorts, flips and margin accounts):

- cash account: `cash = initial + sum(realised P&L) - (cost basis + entry fees
  of the open positions)`;
- margin account: `cash = initial + sum(realised P&L) - entry fees of the open
  positions`;
- each trade: `pnl = gross - entry_fees - exit_fees` on its booked integers,
  and `Trade.pnl` and `fees` are the views of those integers;
- once flat, `cash = initial + sum(realised P&L)`, exactly, after any number
  of fills (a million fills in the same file);
- across a lifecycle (`tests/lifecycle.rs`, random runs of fills, expiries,
  settlements, physical deliveries, splits, bonuses and dividends in cash and
  margin accounts): `balance = initial + sum(realised P&L, settlements,
  deliveries and cash in lieu included) + sum(dividends) - open cost basis
  and entry fees` (cash) or `- open entry fees` (margin), as integers; and
  `tests/accounting_money.rs` holds the same for a kernel held through its
  expiry with a settlement fee.

A sized entry is checked against the booked debit: sizing fits the budget in
`f64`, but the account is debited the notional and the fee each rounded, which
can come to a unit more, so a cash-funded entry whose booked debit would
exceed what is free is sized again on a budget a unit tighter (up to four
times) and refused as `zero_size` if none fits. Cash cannot be overdrawn by
rounding.

### What booking changes in a result

A booked amount is within half a minor unit of the unrounded one, so a trade's
P&L is within about a unit of what unrounded arithmetic gives (three roundings
for a long: cost, exit and the fees; the golden set's worst trade moved 0.9
units). Cash and equity after many fills differ from an unrounded replay by
the accumulated roundings, growing roughly with the square root of the number
of fills. Position sizes move by the same order divided by price. A last
equity sample that marks a position still open at the end of the data uses its
exact price, so it can differ from the booked cash after the closing fill by
up to half a unit per rounded amount (a short's cost basis enters its marked
value twice).

`docs/quantitative-correctness/` states the size of each result change in the
CHANGELOG ("Results change"); `tests/python/golden/drift_report.py` measures
it against the committed golden baselines.

The batch simulators that are not on the kernel yet (baskets, pairs, options
legs, spreads, ticks) still account in `f64` and set `Trade.money` to `None`.

## Price: the tick grid

Applies when the instrument spec has `price_increment > 0`
(`src/instruments/spec.rs`). With no increment every rule here is a no-op and
results are bit-identical to an engine without the grid.

- **Executed fills** snap to the grid *against the trader*: a buy rounds up, a
  sell rounds down (`InstrumentSpec::snap_fill`, called from
  `EngineKernel::executed_price` in `src/portfolio/kernel.rs`). Slippage is
  applied first, then the snap, so percentage slippage cannot leave an
  off-grid fill. A buy is a long entry or a short cover; a sell is a long exit
  or a short entry.
- **Limit and target fills** are already at their own price, so slippage is
  skipped. That covers `Limit` and every kind that rests as a limit once
  triggered: stop-limit, limit-if-touched and trailing-stop-limit
  (`is_limit_fill`, `apply_match_outcome`). `MarketToLimit` fills at the open
  like a market order and takes slippage. Such a price is assumed to be on the grid. If it is not, it snaps
  *in the trader's favour* (buy down, sell up), so a fill never crosses its
  limit.
- **Derived stops and targets** (percent, ATR) snap against the trader too:
  a long's stop and target round down, a short's round up
  (`quantize_protective`). Explicit prices you pass are used as given.
- **Float noise.** Prices are converted to ticks with a division. A price
  within `1e-9` ticks (relative) of a whole tick counts as on the grid and is
  left untouched, so `100.05 / 0.05 = 2000.9999...` never loses a tick.

The tick runner (`src/strategies/tick.rs`) has no instrument spec and does not
snap.

## Quantity: lot floor, then size step

Capital-fraction sizing is computed in fractional units, then
`EngineKernel::quantize_units` floors it to a multiple of `lot_size`
(`(size / lot).floor() * lot`) and then to `size_increment`
(`InstrumentSpec::quantize_size`, `spec.rs`). The order matters: the step is
applied to an already lot-aligned number. Size always rounds down, so
quantizing can only reduce the amount spent. A result of zero rejects the entry
(`ZeroSize`, or `InsufficientMargin` when a margin model set the requirement).
Explicit unit counts are floored the same way and then refused, not shrunk, if
they exceed available funds.

## Money: accumulation

Cash is one `Money` updated by integer addition at every fill (`open_at`,
`close_at`, `reduce_at`, `add_at`, `credit_slice` in `kernel.rs`). Integer
sums are exact, so there is no accumulation error to bound and compensated
summation is not needed: the replay in `tests/python/test_cash_drift.py`
agrees with the booked sum of P&L to the unit, not to a tolerance. Tests that
compare the engine with an independent `f64` model of equity allow the booking
grain (half a unit per open position, a short three halves) and nothing more.

## Fees, per model

`FeeModel` (`src/execution/fees.rs`) receives the *per-contract currency price*
(price times multiplier) and the raw contract count.

| Model | Fee |
| --- | --- |
| `Percentage(r)` | `price * size * r` |
| `Fixed(a)` | `a` per order, whatever the size |
| `PerShare(r)` | `size * r` |
| `Tiered` | `price * size * rate`, rate of the highest threshold the trade value reaches |
| `Custom` | `base + size * per_share` |
| `Indian` | itemised costs per side (STT on the sell, stamp duty on the buy), the built-in rows of `FeeSchedule.exchange_presets()` selected by `fee_segment` |
| `Rates` | the same itemised charge on rates from a loaded `FeeSchedule` (`BacktestConfig(fee_schedule=...)`) |

**Sizing uses the active model.** For `Percentage` (and no fees) the size is the
closed form `budget / (price * (1 + rate))`. For every other model
(`EngineKernel::size_within_fees`) the size starts from the flat estimate, is
refined with the rate the model actually charges at that size, then steps down
until funding plus the model's real entry fee fits in the budget. The result
never exceeds the budget after fees and can be one step below the optimum.

### Itemised costs: one computation, rounded in one place

`Indian` and `Rates` both charge through `FeeRates::charge_fill`
(`src/execution/costs/rates.rs`). The itemised `FeeBreakdown` it returns is what
the kernel debits (`total()`), so a trade's reported costs and its effect on
equity are the same numbers. The kernel books that total to the run's scale
and stores the booked amount as the breakdown's `exact_total`, so
`fee_breakdown.total()` equals `Trade.fees` to the last digit; the parts stay
as computed and can differ from their sum by up to half a minor unit. A row declares: brokerage (flat, percent, cap,
minimum, per order or per fill), the transaction tax by side, the exchange
transaction charge, the regulatory fee, stamp duty on the buy side, the GST
rate and the components it is levied on, an optional investor-fund charge, and
an optional flat fee owed once per instrument per day (the caller says when it
falls due, because one fill cannot know whether an earlier fill that day paid
it). The kernel does not charge the per-day fee or settlement rows; schedule
callers (`FeeSchedule.charge(..., day_fee_owed=True)`) do.

Where money is rounded (`src/execution/costs/rounding.rs`, the only place):

| Boundary | Rule |
| --- | --- |
| Row `rounding = "none"` (the built-in presets) | no rounding in the row: the exact `f64` products. The kernel then books the total to the run's scale (see above), so a charge reaches cash as a whole minor unit |
| `paise_each` | each component to the paisa, half up, once computed; GST on the rounded parts; total the rounded sum |
| `contract_note` (the default for a rate card document) | stamp duty to a whole rupee, half up; GST on the unrounded brokerage, exchange and regulatory charges; each part to the paisa, half even; `exact_total` the rounded sum of the unrounded parts (it can differ by a paisa from the sum of the shown parts) |
| Accumulating charges over fills and trades | booked integers: a trade's fees are the sum of its entry and exit charges, each already a whole minor unit |
| Margin | never rounded: it gates entries and marks maintenance, it is not a charge |

Money is `f64`, so a value that is exactly half a paisa in decimal (0.025% of
1,500.00 is 0.375) is rarely exactly half a paisa in binary. `round_places`
treats a value within 16 ulps of a tie as that tie before applying the rule,
instead of tipping on representation noise. Fixed-point accounting changed
only `rounding.rs` at the boundary of a charge: `to_minor` is the same rounding
returning an integer.

Rates in a rate card document are percent text (`"0.03"`); they are read two
places down as text (`"0.03e-2"`), so the stored fraction is the double nearest
0.0003 and not 0.03 divided by 100.

**Differences from the reference cost model, on purpose.** (a) The built-in rates are
the exchange presets (the numbers this engine always charged), not the reference
"illustrative" ones; the reference mechanisms (contract-note and paise-each rounding,
brokerage per order with a cap, the per-day depository fee and its GST) are
reproduced exactly and checked fill by fill against that engine
(`tests/python/differential`). (b) A rate card may give a percent brokerage
without a cap, a minimum, or both a flat and a percent amount; the reference refused those.
(c) A zero-value fill is charged no brokerage only under `contract_note`, as in
the reference; elsewhere it is charged the flat amount, as this engine always did.
(d) Charges are computed in `f64` and booked in minor units: a result can differ
from the reference's exact decimal arithmetic only by a representation tie, which
`round_places` and `to_minor` resolve as the decimal would.

## Contract multiplier

`spec.multiplier` (default 1, non-positive values count as 1;
`EngineKernel::multiplier`) turns a quoted price into currency per contract. It
scales notional (`contract_value = price * multiplier`), the cash credited on
close (`marked_value`, `src/portfolio/ledger.rs`), PnL and fee inputs. It does
not scale the contract count or the tick grid, which stay in quote units.

## Margin

In margin mode an entry debits only fees from cash and locks
`contract_value * size * rate` (`EngineKernel::entry_funding`).

- `rate` is `spec.margin_init` when set, otherwise `1 / leverage`.
- Maintenance is `spec.margin_maint`, otherwise half the initial rate. At an
  initial rate of 1.0 or more the position is fully funded and nothing is
  maintained.
- A *sold* option with `span_pct` or `exposure_pct` set locks a flat deposit,
  `(span_pct + exposure_pct) * strike * multiplier` per contract, instead of a
  rate; the premium stays in cash.

Closing releases the lock and books realised PnL; entry fees are not charged
twice.

These rules are `MarginModel.from_spec(leverage)` (`src/accounts/margin_model.rs`):
the kernel takes its rate from `initial_rate` and the deposit from
`InstrumentSpec::short_option_margin_per_contract`, and the model calls the same
two, so a figure is defined once. The rest of the models (`cash_only`,
`leverage`, `percent_of_notional`, `option_simple`, `table`, `by_type`,
`by_product`, `defined_risk`, `spread_credit`) are checked against the reference
engine figure by figure (`tests/python/differential`). `leverage`/`by_type` keys
are instrument classes (`CASH`, `CURRENCY_PAIR`, `CONTRACT`, `OPTION`; `EQUITY`,
`ETF`, `FUTURE`, `PERPETUAL` and `CURRENCY` read as the first three). A class a
model has no rule for is an error, never a zero.

### Choosing the model: `BacktestConfig(margin_model=...)`

Unset, the account applies the rule above, bit for bit. Set (and the account
`margin`; a cash account with a model is refused as `invalid_margin_model`),
every entry is funded and refused by that model through one path in the kernel
(`EngineKernel::entry_funding`):

- The requirement of one unit at the fill price is `model.required(line)`; the
  lock is that per-unit figure times the size, so a model must be linear in the
  quantity of a single line (every named model is).
- Sizing by a fraction of capital divides the budget by the per-unit
  requirement plus the fee, as for a built-in deposit.
- **`no_margin`** requires nothing and locks nothing, so a fraction of capital
  cannot be sized against a zero. Rule: it sizes as a fully funded account
  would, at the lot's full notional, so a fraction of capital never buys more
  than the same fraction in cash; the entry debits its fee only and the
  position carries no lock and no maintenance. Explicit unit counts are limited
  only by the size limits and the fee.
- An infinite requirement (`cash_only` on a short) refuses the entry as
  `insufficient_margin`; a requirement above the free capital does the same
  (a zero-size entry, or an explicit size, which is `insufficient_capital`); a
  model with no rule for the instrument's class, or an instrument without a
  market definition, refuses it as `margin_unpriced`. All reach
  `on_order_rejected` with that `reject_reason`.
- Maintenance is half the model's requirement re-read at the mark, and nothing
  when that requirement covers the whole notional.
- The built-in grouping of sold option legs of one underlying and expiry does
  not run under a chosen model: each line locks its own requirement, so a
  `defined_risk` or `spread_credit` model prices a leg alone on entry.
- Adopting a broker-held position needs a model that demands at least the
  whole cost basis, for the reason leveraged adoption is refused above.

## Partial-fill completion

An order is complete when `filled + 1e-9 >= requested`
(`record_fill`, `src/portfolio/kernel_orders.rs`). The absolute `1e-9` absorbs
the rounding of slices computed from the same arithmetic as the request, so the
last slice completes the order instead of leaving a dust remainder. It is in
units, so it is meaningful for lot sizes well above `1e-9`.
