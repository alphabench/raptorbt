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
| `"auto"` (default) | an option settles to intrinsic value against the underlying's price when the run has one — `set_underlying_price`, else the underlying's print at the expiry instant when the underlying is registered in the run (the contract waits for that print if its own bar comes first, so the price never depends on which symbol is listed first) — and at its own last price when it has neither; every other contract at its last price | `underlying` or `last_price`, whichever was used |

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
instrument's `settlement_fee` fraction, and otherwise charges nothing.

The statutory events stay separate:

- **Exercise STT** is the settlement row of a cash-settled option. The built-in
  NFO and BFO schedule carries it: 0.15% of the *intrinsic* value, charged to
  the buyer (the line closing a long) and to no one else. An option that expires
  worthless settles at an intrinsic value of zero and owes nothing; a writer
  owes no exercise STT.
- **Physical delivery** (a stock future, or an exercised stock option) is a
  delivery event of its own on the underlying: the stock leg is charged at the
  cash market's delivery rates (STT on both sides, stamp duty on the buyer; NFO
  maps to NSE, BFO to BSE), whatever segment the derivative is priced in. The
  derivative's own settlement is not charged by the schedule's settlement row,
  so its STT is not stacked on the delivery's. The depository charge, owed when
  delivered stock is later sold, belongs to that later sale.
- **Index options and futures** are cash-settled at the underlying's print at
  the expiry instant. For an index that print is the official closing value the
  caller supplies for the instant; the engine takes no average or VWAP of
  anything.

## Fee rows: by instrument, by date

`FeeSchedule.exchange_presets()` is the dated statutory schedule, stored in
rupees per crore of each row's base (`row["per_crore"]`; a fraction would carry
the conversion's rounding) with every component separate and the base, side,
`effective_from` and `effective_until` on the row: exchange transaction charge,
investor protection fund contribution (IPFT, the `ipft` part), SEBI fee,
transaction tax (STT, CTT) on the side that owes it, stamp duty on the buyer, and
GST at 18% of exchange + IPFT + SEBI (IPFT is included conservatively). The base
is turnover, premium (an option's trade), intrinsic value (an exercised option's
settlement row) or delivery value.

| Rows | Figure | From | Source / status |
| --- | --- | --- | --- |
| NSE cash, NFO futures, NFO options | exchange 297 / 173 / 3,503 + IPFT 10 / 10 / 50 per crore; from 2026-03-01 306.99 / 182.99 / 3,552 + 0.01 | 2024-10-01 / 2026-03-01 | NSE/FA/73061, confirmed |
| NSE cash, NFO futures, NFO options, before 2024-10-01 | slab-era exchange 322 / 188 / 4,950 per crore, no IPFT (not known for the period); STT futures 0.0125% and options 0.0625% on the sell | 2023-04-01 | `unverified`: "slab-era representative rate" |
| STT | delivery 0.1% both sides, intraday 0.025% sell; futures 0.0125%, 0.02% from 2024-10-01, 0.05% from 2026-04-01, options 0.0625%, 0.1%, 0.15% on the sell; exercised options 0.125% then 0.15% on the buyer's intrinsic value | 2023-04-01, 2024-10-01, 2026-04-01 | Finance Act, confirmed |
| CTT | non-agri futures 0.01% sell, options 0.05% sell, a commodity option exercised 0.0001% on intrinsic value, agricultural futures none | 2024-10-01 | Finance Act, confirmed (the MCX exchange charges are not) |
| SEBI fee, stamp duty | 10 per crore (agricultural commodities 1); stamp 1,500 / 300 / 200 / 300 / 10 per crore | 2020 and 2019 | confirmed |
| BSE currency | futures 45, options 100 per crore of premium | 2024-10-01 | BSE Notice 20240927-36, confirmed |
| BSE cash by scrip group, every BFO row, NSE currency, MCX and agricultural exchange charges | BSE A/B 375; BFO Sensex and Bankex options 3,250, Sensex 50 and stock options 500, futures nil; CDS futures 35, options 3,110; MCX futures 210, options 4,180 per crore | 2024-10-01 | `unverified` |

A row that is not matched to an official publication is `unverified`, and its
figure is stored with its source rather than as a zero; where the sources
publish no IPFT for a segment the row says it charges none. Every result
(`BacktestResult.unverified_fee_rows`, `PortfolioResult.result`, `full_report`)
lists the unverified rows it priced fills on, as `VENUE:SEGMENT[:KEY]`, so no one
mistakes the costs for exchange-accurate. BSE cash charges depend on the scrip
group: set `InstrumentSpec.scrip_group` (`"A"` or `"B"` are priced); an
instrument without one, or with another, is refused (`fee_schedule_gap`) naming
the groups. The earliest row is 2023-04-01 for NSE cash and NFO and 2024-10-01 for the other
venues: a fill or settlement before a venue and segment's first row is refused
(`fee_schedule_gap`, naming the earliest date), never priced on that row; the same holds for a
rate card and a date before its `effective_from`. Cash equity bought and sold on one trading
date is priced on the intraday row even in a positional run (entry and exit both, the entry
charge repriced when the exit lands; a slice carried overnight stays on the delivery rows). The
presets carry no brokerage and no depository charge: those come from a broker's
`RateCard`, where the card says per order.
`tests/python/fixtures/statutory_fees.json` holds the schedule and
`tests/python/test_statutory_fees.py` proves the presets equal it on every date.

With a fee schedule, an instrument is priced on the row its own venue and kind
select: a `Contract` the futures row, an `Option` the options row, cash equity
the delivery row (the intraday row when the run squares positions off the same
day). The venue comes from `InstrumentSpec.venue`; an instrument without one,
or a venue the schedule does not price, is refused (`fee_schedule_gap`) when it
is registered. `fee_segment` is an explicit override: one venue and segment for
every instrument.

A fill on day D is charged by the row whose `effective_from` is the latest at
or before D (the venue's trading date, in the instrument's session zone, else
`session_tz_offset_ns`). A row that names an underlying (an agricultural
commodity) beats the general row for instruments declaring that underlying.
The per-instrument-day fee (the depository charge on a delivery sell, from a
card) is owed by the first fill of a trading date that carries it, once per
instrument and date. A brokerage that a card says is per order is charged by
the kernel on each fill; an order that fills in several slices
(`partial_fills`) therefore pays the flat amount per slice, which is open.

## Same-day cash equity

A cash-equity buy is charged as delivery at entry (the exit is not known), as a broker's CNC buy
is. A sale on the same trading date restates the round trip to the intraday row and refunds the
difference. Until then equity, drawdown, day-loss limits and capital-edge sizing carry the delivery
charge, so they can be slightly tighter than a pure intraday run would show. Only the components
that differ between the two rows are restated: the brokerage stays as it was booked, per order, so
a position sold in several slices pays each exit order once.

## Sessions

`SessionSchedule` decides what trades on a local date in this order: a
per-date override (a special session, Muhurat trading, a half day; empty
windows close the date), the holiday set, the latest dated window change that
covers the date, a standing rule that follows the US clocks (MCX), the base weekday windows. Presets (`nse_equity`, `bse_equity`,
`nse_fo`, `bse_fo`, `mcx`, `cds`, `bcd`) carry no holidays and no square-off;
the equity-derivatives close is 15:30 until 2026-07-31 and 15:40 from
2026-08-03 (the Closing Auction Session for F&O stocks); MCX has a day session to
17:00 and an evening session whose close follows a standing rule with no end:
23:55, and 23:30 while the US is on daylight saving, switching on the first
trading day after the US clocks change (the Monday after the second Sunday of
March and after the first Sunday of November: 2024-11-04, 2025-03-10,
2025-11-03, 2026-03-09, 2026-11-02, ...); the select agricultural
contracts (`mcx_select_agri()`: Cotton, Kapas, Cotton Seed Wash Oil
`COTWASOIL`) close at 21:00 all year and every other agricultural contract
(`mcx_other_agri()`, CPO and RBD palmolein among them) at 17:00. An
`InstrumentSpec` on `venue="MCX"` takes the session of its class from its
underlying or symbol, for the classes the table names (those agricultural
contracts, and bullion, base metals and energy on the evening session). A name
in none of them is not guessed: the spec raises `mcx_session_unknown` and the
caller passes `session=`. Dates before 2024-11-04 are
closed, not guessed (`schedule.coverage`), and need a calendar document. NSE currency on INR pairs trades 09:00-17:00
(`cds()`), on cross-currency pairs 09:00-19:30 (`cds_cross()`). Muhurat trading on
2026-11-08 has no preset: its times were not notified, so add them with
`with_session_override(20261108, [(open, close)])`. A bar on such a date raises `session_not_covered`
(naming the date and the covered ranges) in `run_strategies`,
`run_strategy_backtest` and `LiveNode.push_bar`, so it cannot be read as a
holiday.

A dated calendar document loads a venue's schedule:
`SessionSchedule.from_calendars(venue, [json, ...])`.

```json
{"name": "calendar-2026-q4", "utc_offset": "+05:30",
 "valid_from": "2026-09-28", "valid_to": "2026-12-31",
 "sessions": {"NSE": {"open": "09:15", "close": "15:30"},
              "MCX": {"windows": [["09:00", "17:00"], ["17:00", "23:55"]]}},
 "holidays": [{"date": "2026-10-02", "venues": ["NSE", "MCX"]}],
 "overrides": [{"venue": "MCX", "date": "2026-10-20", "open": "17:00", "close": "23:55"},
               {"venue": "NSE", "date": "2026-11-08", "windows": [["18:15", "19:15"]]}]}
```

`sessions` windows apply Monday to Friday unless `"weekdays"` says otherwise;
an override opens a date whatever the weekday or holiday set says
(`"closed": true` closes it). Holidays and overrides are the caller's own data:
the crate ships none. Several documents share one `utc_offset`; the later one
answers for the dates it covers, and a date no document covers is closed, not
guessed. Intraday composite bars of an instrument with a schedule start at the
open of the window they fall in and end at its close.

## Defaults that follow the instrument

Order validity is a capability of the exchange segment, not a global enum. NSE
equity cash holds good-till-cancelled and good-till-date orders itself; NSE
commodity does too, with a date no later than the contract's maturity; NSE
equity derivatives, currency (NSE and BSE: BSE currency is end of session), BSE
and MCX are established as day orders only. An order that names no time in force
(`tif="auto"`) is `day` on an instrument with a session or on a day-only
segment, and `gtc` otherwise. An explicit `gtc` or `gtd` on a segment without
native support is refused (`order_validity_unsupported`, naming `day`, `ioc`,
`fok`); a broker that keeps an order alive by resubmitting it every day is the
host's business and not the exchange's validity. The segment comes from
`InstrumentSpec.venue`; an instrument with no venue allows everything.

A position is squared off only when the run asks (`squareoff_time`, read in the
instrument's session zone, else `session_tz_offset_ns`; refused with neither) or
the strategy's `requirements.holding` is `Holding.intraday(flatten_before=...)`.

Annualisation counts trading periods, one rule everywhere (`full_report`, the
metrics of every runner, the tick path): an explicit `periods_per_year` wins;
else `session_minutes` (bars per session times 252); else the instrument's
session schedule when it has one (its trading minutes per day times 252); else
what the data shows, the median number of observations per local day times 252.
Nights and weekends add no periods; only `session_minutes=0`, naming a market
that never closes, counts calendar time. `session_tz_offset_ns` defaults to none
(days are UTC dates).

Tick size comes from the `InstrumentSpec` (`price_increment`), which the host
loads from the exchange's contract master. It is not a constant: NSE stock-option
ticks are 0.01 below an underlying price of 250 and 0.05 at or above it
(NSE/FAOP/70382, 23 September 2025, in force from 3 November 2025), reviewed
monthly on the underlying's close of the last trading day, so a host that
trades stock options must reload the master each month. Index options and
futures follow their contract specifications. An off-tick price is refused
against the spec's tick.

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

A live node takes `raw` only. `adjusted` puts every bar before a split on the
basis after it, which a batch run can do because it holds the whole series. A
node is fed bar by bar: when the ex-date arrives, the bars already pushed, the
indicator state and composite bars built on them (an EMA carries all its
history), and the orders priced from them are on the old basis, and restating
them would mean replaying the whole feed. So `adjusted` is refused
(`invalid_adjustment_policy`) with that reason. Run `raw` (positions are
rescaled at the ex-date), or push history already adjusted and declare no
action inside it.

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
| `run_tick_strategy` | yes, as above | yes | yes (`corporate_actions=`, `adjustment_policy=`; `raw` only for prints) |
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
