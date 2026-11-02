# Contract lifecycle and corporate actions

What happens to a position when its contract expires, and when the instrument
it holds splits or pays a dividend. The code is the source of truth:
`src/portfolio/kernel_lifecycle.rs` (one instrument),
`src/portfolio/book/lifecycle.rs` (what spans instruments),
`src/instruments/lifecycle.rs` (the data types).

## Expiry and settlement

Declare how a contract settles on its `InstrumentSpec`:

```python
fut = InstrumentSpec.futures_contract("FUT", expiration_ns=..., lot_size=50.0, underlying="IDX")
fut.settlement_style = "cash"          # "cash" (default) or "physical"
fut.settlement_price = "underlying"    # "auto" (default), "last_price", "underlying", or a number
```

### Where the settlement price comes from

A settlement price is never guessed. It is one of:

| `settlement_price` | Price | `event.source` |
| --- | --- | --- |
| a number | the price you declared for the contract (an exchange's final settlement price). For a physical option it is the *underlying's* final price, which decides exercise | `declared` |
| `"last_price"` | the contract's own last price: the close of the bar at or after the expiry, or the last price seen when the expiry passes with no bar | `last_price` |
| `"underlying"` | the underlying's price: an option settles to intrinsic value against it, a future at the price itself. Taken from `set_underlying_price` when you supplied one, else from the underlying's last close in the same run. Without either the settlement is refused | `underlying` |
| `"auto"` (default) | an option settles to intrinsic value when you supplied the underlying's price (`set_underlying_price`), otherwise at its last price; every other contract at its last price | `underlying` or `last_price`, whichever was used |

A price the terms need but the run never gave refuses the settlement: the
position stays open, `on_contract_event` hears `lifecycle_refused` once with a
stable reason (`underlying_price_missing`,
`physical_settlement_needs_underlying_price`,
`physical_settlement_needs_a_portfolio`, `physical_settlement_unsupported`),
and the next step tries again. It is not closed at a guess.

### What settling does

The position closes with exit reason `Settlement` at the settlement price (a
physically settled option at nothing). The strategy hears, in order, the
position's close (`on_position_event`, with the trade) and then
`on_contract_event` with `event.kind == "contract_settled"`:

- `outcome`: `settled` (cash), or for physical settlement `exercised` (a long
  option in the money), `assigned` (a short one), `expired` (out of the
  money; nothing is delivered) or `delivered` (a future);
- `price`, `source`, `style`, and for a delivery `quantity` (underlying units
  taken in, negative given away) and `amount` (the delivery price).

Orders and entries on an expired contract are rejected with reason `expired`
(`on_order_rejected`).

### Physical delivery

The underlying must be registered in the same run (`run_strategies`,
`run_portfolio_strategy`); a physical settlement without it is refused before
the contract closes. An option in the money delivers `units * multiplier`
underlying units at the strike (a call: long takes in, short gives away; a
put the reverse). A future closes at its settlement price and delivers the
underlying at that price. The delivery is booked on the underlying's kernel
like a fill: an opposite line closes first (exit reason `Delivery`), the rest
opens or adds to a line, and the fill is charged as any fill is. No slippage
and no risk gate apply: it is the contract's obligation, not an order. The
single-instrument runner (`run_strategy_backtest`) has no underlying to
deliver on, so it refuses physical settlement.

### Settlement charges

Priced on the fee schedule the run uses (`BacktestConfig(fee_schedule=...,
fee_segment=...)`): the schedule's `SETTLEMENT` row for the venue and segment
in force on the settlement's date, on the closing side, on the settlement
value. A schedule with no settlement row for the segment falls back to the
instrument's `settlement_fee` fraction, and otherwise charges nothing. A
physical option is charged on its intrinsic (exercise) value, not on the
nothing it closes at.

## Fee rows by date, and the per-instrument-day fee

With a fee schedule, a fill on day D is charged by the row whose
`effective_from` is the latest at or before D (the venue's trading date, from
`session_tz_offset_ns`). A fill earlier than every row takes the earliest. A
row that names an underlying (an agricultural commodity) beats the general row
for instruments declaring that underlying. The per-instrument-day fee (the
depository charge on a delivery sell) is owed by the first fill of a trading
date that carries it, once per instrument and date.

## Corporate actions

```python
actions = [
    CorporateAction.split("ABC", ts, 2.0),             # 2 new units per old
    CorporateAction.bonus("ABC", ts2, 2.0),            # 1:1 bonus
    CorporateAction.dividend("ABC", ts3, 3.5, pay_ts=ts3 + 2 * DAY),
]
run_strategies([s], data, corporate_actions=actions, adjustment_policy="raw")
```

`ts` is the ex-date's first instant. Events are data you give: a list without
`adjustment_policy` is refused (`invalid_adjustment_policy`), an action on a
symbol without data is refused (`unknown_symbol`).

### Adjustment policy

| Policy | History the strategy sees | The position |
| --- | --- | --- |
| `raw` | prices as traded: a split shows as a jump, and indicators see it | rescaled at the ex-date: size times the ratio, entry price, stop, target and running extremes divided by it; the cost basis does not move, so cash and equity do not change |
| `adjusted` | every bar before a split or bonus is put on the basis after it (prices divided and volume multiplied by the product of the later ratios), so indicators see a continuous series | not rescaled again: the run trades that one basis. A dividend's per-unit amount is restated to it (divided by the later splits) |

`adjust_history(instrument, actions, bars)` is the same function the runner
applies, for your own pipeline. Dividends are never applied to prices. The
policy is part of every indicator's identity: strategies
on different bases never share one. `adjusted` refuses a symbol that has ticks
and actions.

### What a split does

- Working orders on the instrument are cancelled (their prices are on the old
  scale), as an exchange cancels them.
- An instrument that trades in whole units (a cash instrument with
  `size_increment == 1`) keeps each direction's whole units; the fraction is
  paid in cash at the last price divided by the ratio, as a trade slice with
  exit reason `CorporateAction`, its share of the cost basis, and no fee. A
  split that leaves a fraction with no last price is refused
  (`cash_in_lieu_needs_a_last_price`).
- A contract's lot size and freeze quantity are not rescaled.
- `on_corporate_action` hears `split`/`bonus` with the ratio and the cash in
  lieu.

### What a dividend does

The holder at the ex-date (before any trading at the ex-date) is entitled to
`amount * signed units * multiplier`; a short pays. The cash is credited on
`pay_ts` (default the ex-date), booked once in minor units, and
`on_corporate_action` hears `dividend` then. It is cash, not P&L of a trade:
it moves the equity curve and `session.dividends_received`, not the trade list.

## Where each runner stands

| Runner | Expiry and settlement | Physical delivery | Corporate actions |
| --- | --- | --- | --- |
| `run_strategies`, `run_portfolio_strategy` | yes: kernel branch, book timer, waits for the underlying | yes | yes (`corporate_actions=`, `adjustment_policy=`) |
| `run_tick_strategy` | yes, as above | yes | not exposed; pass ticks to `run_strategies` (`raw` only) |
| `PortfolioSession` (event session) | yes | yes | yes (`set_corporate_actions`) |
| `LiveNode` | yes, on its events; `advance_to` for a clock | yes | `Runtime.set_corporate_actions` exists; the node does not expose it yet |
| `run_strategy_backtest`, `KernelSession` | cash settlement on the first bar at or after the expiry | refused (no underlying) | no |
| array runners (`run_single_backtest`, basket, pairs, spreads, options) | the kernel branch where they use a kernel; their own settlement rules otherwise | no | no |

## Against the reference implementation

`tests/python/differential/test_lifecycle_differential.py` replays 18 scenarios
that a reference implementation ran (futures and options
settled in cash and physically, short and long, splits, a bonus, a fraction of
a share, dividends, settlement rows, and fills across two cards). Cash,
positions, settlement prices, closed trades and the charges on them agree
exactly. The differences that are intended:

- **The default settlement price.** The reference settled a contract at its underlying's
  mark whenever it had one. Here `settlement_price` is a stated term: the
  scenarios set `"underlying"` (the reference rule) or a declared price (the reference
  `final_prices`), and the default `"auto"` settles at the contract's last
  price unless you supplied the underlying's price.
- **A physical option's charge base.** The reference levied the settlement row on the
  nothing a physical option closes at; here it is levied on the exercise
  (intrinsic) value, which is what an exchange's tax on exercise is based on.
- **Cash in lieu is a trade slice.** The cash is the same; here the fraction
  also appears in the trade list (exit reason `CorporateAction`, no fee).
- **A dividend can be paid later** than its ex-date (`pay_ts`).
- **Not ported.** Symbol changes, rescaling of lot and freeze quantity at a
  split, keeping resting orders across a split (`policy="adjust"`), and the
  `SETTLEMENT`/`Lifecycle` decision records of the reference intent engine (the
  settlement is an event and a trade here).
