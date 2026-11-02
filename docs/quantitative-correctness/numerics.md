# Numerics: how prices, quantities and money are converted

One rule per conversion boundary. Each names the code that enforces it.

## What is f64, and why

Prices, sizes, cash, fees and PnL are all `f64`. Order quantities are fractional
(crypto, FX), contract multipliers and percentage fees are not integers, and
every reported metric (Sharpe, drawdown) is a float ratio, so a fixed-point
money type would need converting back at each of those edges. The exceptions
are timestamps (`i64` nanoseconds) and counts (`usize`).

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

## Money: accumulation and tolerance

Cash is one `f64` updated by `+=`/`-=` at every fill (`open_at`, `close_at`,
`credit_close` in `kernel.rs`). Plain summation is used on purpose: replaying
100k round trips against an exactly rounded `math.fsum` of the trade list
differs by about `4e-14` relative (`tests/python/test_cash_drift.py`), far
below any displayed precision, so compensated summation is not used.

Cash can end a bar a few ulps off the mathematically exact value, never
meaningfully. Equity checks in tests use `pytest.approx`, not equality.

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
equity are the same numbers. A row declares: brokerage (flat, percent, cap,
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
| Row `rounding = "none"` (the built-in presets) | no rounding: the exact `f64` products, as the engine always charged |
| `paise_each` | each component to the paisa, half up, once computed; GST on the rounded parts; total the rounded sum |
| `contract_note` (the default for a rate card document) | stamp duty to a whole rupee, half up; GST on the unrounded brokerage, exchange and regulatory charges; each part to the paisa, half even; `exact_total` the rounded sum of the unrounded parts (it can differ by a paisa from the sum of the shown parts) |
| Accumulating charges over fills and trades | not rounded: plain `f64` addition, as for cash above |
| Margin | never rounded: it gates entries and marks maintenance, it is not a charge |

Money is `f64`, so a value that is exactly half a paisa in decimal (0.025% of
1,500.00 is 0.375) is rarely exactly half a paisa in binary. `round_places`
treats a value within 16 ulps of a tie as that tie before applying the rule,
instead of tipping on representation noise. When accounting moves to fixed
point, only `rounding.rs` changes.

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
(d) Values are `f64` throughout: a result can differ from 1.x's exact decimal
arithmetic only by a representation tie, which `round_places` resolves as the
decimal would.

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

## Partial-fill completion

An order is complete when `filled + 1e-9 >= requested`
(`record_fill`, `src/portfolio/kernel_orders.rs`). The absolute `1e-9` absorbs
the rounding of slices computed from the same arithmetic as the request, so the
last slice completes the order instead of leaving a dust remainder. It is in
units, so it is meaningful for lot sizes well above `1e-9`.
