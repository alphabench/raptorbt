# Execution adapters

How an order leaves the engine, and how the answer gets back. The Python
guide is the docstring of `raptorbt.execution`; the Rust types are in
`src/execution/adapter/`.

## Three ways to run the same strategy

| execution | who matches | how a fill reaches the book |
| --- | --- | --- |
| in-process (`None`, `SimulatedExecution()`; what a backtest does) | the kernel, inside the book's step | the matcher books it directly |
| `PaperExecution()` | a shadow copy of the book, stepped with the same events | an `ExecutionReport`, applied by `Order::apply_report`, then booked |
| your adapter | the venue | an `ExecutionReport` you push with `LiveNode.push_report` |

The default is not an adapter. It is the direct path the backtest loop takes,
so a run that picks no adapter pays nothing for any of this: the hot path
tests one flag per step and allocates nothing.

## The contract

```rust
pub trait ExecutionAdapter {
    fn submit(&mut self, req: &OrderRequest) -> Result<(), AdapterError>;
    fn cancel(&mut self, client_order_id: &str) -> Result<(), AdapterError>;
    fn modify(&mut self, client_order_id: &str, change: &OrderChange) -> Result<(), AdapterError>;
    fn link_oco(&mut self, ids: &[String]) -> Result<(), AdapterError>; // optional
}
```

```python
class ExecutionAdapter(Protocol):
    def submit(self, request: OrderRequest) -> None: ...
    def cancel(self, client_order_id: str) -> None: ...
    def modify(self, client_order_id: str, *, quantity=None, limit_price=None, trigger_price=None) -> None: ...
```

* **Fire and forget.** A method returns when the request is handed over, not
  when the venue acts. `Ok` means "sent". Outcomes come back only as reports.
* **A namespace.** An adapter exposes a non-empty `namespace` string that marks the
  orders of one node; a node refuses an adapter without one (`journal-and-recovery.md`).
* **Optional capabilities.** `open_orders()`, `positions()` and `balances()` let a
  recovery ask the venue what it holds (shapes in `journal-and-recovery.md`);
  `fence(epoch)` raises `Fenced` when another node owns the account, and a node given
  `epoch=` refuses an adapter without it. `place_exit_rule(request)` holds an exit rule at
  the venue: the node calls it through its fence, journal and replay (never during a replay
  or muted warm-up), and registers the rule's `exit_client_ids` as orders, so the fills of
  the exits the venue fires are booked.
* **One way in.** Reports enter through `LiveNode.push_report(client_order_id,
  report, *, arrival_ns)`, applied at once as an event in the node's stream
  (`event-semantics.md`, "Execution reports in the stream"). An adapter never
  touches engine state, and must not call into the node from inside its
  methods; the node is single-threaded and refuses a push made from a hook.
* **Refusals have names.** `OrderRefused` (or `AdapterError::Refused`) means
  the venue certainly will not take it: the order is rejected. `Unsupported`
  means the venue has no equivalent: the order is rejected as unsupported. Any
  other exception means the request may or may not have arrived: it propagates
  to the caller unchanged, and the order stays `Submitted` until the venue
  speaks or a reconciliation settles it.
* **Identity.** `client_order_id` is `s{strategy}-{seq}` (`client_order_id()`
  / `parse_client_order_id()`): the strategy's index and a count of the orders
  it has sent. The same decisions in the same order send the same ids on a
  replay. An id is never sent twice: a strategy command repeated with the same
  order id and the same terms is the same order, and is not sent again; the
  same id for different terms is refused (`duplicate_client_order_id`). A
  cancel asked for twice is asked of the venue once.

## What the engine sends

An `OrderRequest` carries the venue-facing id, symbol, side, quantity in
units, the order kind with its limit and trigger (trailing offsets too), the
time in force, the strategy index, the venue account (`venue_account=`),
`reduce_only` and `post_only`, the parent order to wait for, free-form tags
(`client_id` is the id the strategy gave) and the event time and per-instrument
ordinal of the decision.

The book creates the order venue-driven from the start, and the in-process
matcher never sees it. A fill report is booked on the position ledger and the
account through the code the matcher uses (`open_at`, `add_at`, `reduce_at`),
at the quantity and price the venue reported; the venue's price is final and
no slippage is added to it. The fee the book charges is its own fee model's,
rounded to the money scale; the venue's reported fee is kept on the order and
does not move cash. That is the rule the reconciliation corrects against: the
cash the book holds after a fill is the model's, and the difference to the
venue's fee is booked there.

## Everything a backtest accepts becomes orders

A strategy that passes its backtest must run on a venue, so each form the
in-process path accepts is turned into real venue orders, deterministically,
by the runtime (`src/python/runtime_bindings/venue.rs`). The positions of a
venue-driven book are *venue-managed*: the kernel does not act on stops,
targets, schedules or the square-off itself; it leaves requests for the
runtime, which sends orders.

| in a backtest | on a venue |
| --- | --- |
| `size_frac`, or no size ("the whole position") | resolved to units when the order is sent, from the book on that event: the position, for an order that closes one; else the fraction of the capital on hand, sized as the matcher sizes an entry (lots, fees, margin) at the price the order is expected to trade at (the fill-price model for a market order, the limit or trigger otherwise). A leg held for a parent takes the parent's quantity. Sized to nothing: rejected with the matcher's reason |
| `enter()` | a market order for the sized quantity, id `enter-n`, when the instrument is flat and no entry is working |
| `close_position()` | a reduce-only market order for each position's size, id `close-n` |
| stop and target on an order or an entry, or configured on the position (fixed, risk-reward, ATR) | protective orders sent when the position opens, ids `prot-n`: a stop-market (a trailing-stop-market for a trailing stop) and a limit, both reduce-only, linked one-cancels-other |
| session square-off | at the cutoff the engine cancels what is working to add exposure and sends reduce-only market orders for every position |
| `Twap` | the schedule stays in the book; each slice is sent as a market order when it is due |
| `halt` | working orders that add exposure are cancelled, new entries are refused, and with `flatten` reduce-only market orders close the positions |
| `link_oco` | `link_oco` on the adapter; where it answers `Unsupported` the engine keeps the group and cancels the others when one fills |
| `submit_structure` | ordinary orders, hedge-first: the bought legs first, each sold leg when the hedges covering it have filled (the engine sends it on the fill report), with identical results to `run_strategies` (tested). The venue needs no group or atomic capability |

### Protective orders: `protective="venue"` or `"local"`

`LiveNode(protective=...)` (read back as `node.protective`) says who holds
the stops and targets:

* **`"venue"` (default).** The legs are orders the venue holds. A leg the
  adapter answers `Unsupported` for is watched by the engine instead, so the
  position is always protected. Parity: a stop fills as the matcher fills a
  stop-market order (gap-through included) and a target as a limit, at the same
  bar and price as a backtest's position-level exit, given the entry filled
  whole. Not matched: a trailing config is a trailing-stop-market order, whose
  ratchet is the order engine's; only the first fill slice of a partly filled
  entry is protected; protection is cancelled when the instrument is flat, not
  when one of several hedged positions closes.

  **Day-only segments.** The engine's own orders (entries, closes, legs,
  slices, the square-off) carry `gtc` only where the segment holds it (NSE
  cash, NSE commodity, a venue the engine does not know) and `day` elsewhere
  (NFO, BFO, CDS, BCD, BSE, MCX). A `day` leg ends at the session's close, while
  a backtest keeps the level across sessions. When the venue reports a leg
  ended that way (`expired`, or `cancel_ack` without the engine having asked,
  between the close and the next open: a venue such as Kite has no expiry
  status) with its position still open, the engine journals the lapse and, at
  the first event of the next session the instrument's schedule holds, sends
  the leg again for what the position then holds and not already covered by a
  working leg (a trailing stop from the extreme it had reached), linked
  one-cancels-other with its partner as before. On `PaperExecution` the new
  leg meets that bar before the strategy's hooks for it, so a bar that opened
  through the level fills it at the open, as in a backtest. On a broker venue
  it is sent at the bar's close; when the bar has already traded through the
  level the engine closes at market there instead. A re-armed leg rejected (at
  sending or by a later report) is watched by the engine; one the venue could
  not be reached for is sent again each event, and given up when the session
  closes. A leg is not sent again for a
  position closed meanwhile, one the strategy or the engine cancelled, or while
  a close of the position is working (it waits for the close's outcome). A
  venue that cannot hold it leaves the level watched by the engine. The same
  holds for a `gtc` leg an adapter sends as a day order (`gtc="day"`).
* **`"local"`.** The engine watches the levels on each bar or print and, when
  one is reached, sends a reduce-only market order. It reacts on the event, so
  the exit fills at that event's market price, not at the stop level: the same
  bar as a backtest, a different price (worse by the gap through the level).
  Use it for venues with no stop orders.

### What still differs from `run_strategies`, and the tests that pin it

`tests/python/test_execution.py` runs the frozen stream and portfolio scenario
corpus and the option-structure and contract-end tests through a paper node and
compares the whole result digest with `run_strategies` (or the in-process
node). Equal: legacy intents, config stops and targets, attached levels,
brackets, OCO, trailing orders, `Twap`, timers, risk limits, drawdown and
margin halts, adopted positions, warmup, order latency, the session
square-off, structures, and a contract's expiry, cash settlement, physical
delivery and assignment. `enter()` under `fill_timing="next_bar_open"` is sized
at the fill, as a capital-fraction order is (the simulated venue amends the
order's quantity with a replace acknowledgement; a real adapter works the
quantity estimated when the order was sent). Hedging (`oms_type="hedging"`) is
equal too: a close the engine makes names the position it ends
(`OrderRequest.position_id`, below). Two differences remain, each
pinned by a test that asserts exactly what differs:

1. **Excursions of an entry intent filled at the next open.** The trade's
   `mae_*` and `mfe_*` (and the metric derived from them, `MFE Capture Ratio`)
   start from the fill's price on a paper node; in-process they include the
   entry bar's own high and low. A fill a venue reports carries a price, not
   the range of the bar it happened in (`pf_fill_next_bar_open`).
2. **Order of an entry intent.** An `enter()` held for the next print and a
   typed market order made in the same step: in-process the engine handles a
   step's intents (signals) before it sweeps orders, so the entry fills first;
   at a venue they fill in the order they were sent, and the typed order was
   sent first (`st_refused_rows_timers`). The in-process order is the specified
   one: `enter()` and `close_position()` are signals the engine resolves in a
   phase of their own, before any order in the sweep, in every mode and on every
   data cadence, and the result does not depend on where in the hook the call
   sits. A venue cannot reproduce that, because it only sees orders in the order
   they were sent. The two agree unless one step mixes a signal and a typed order
   for one instrument that fill at the same moment; a strategy that does so asks
   for an ordering it has not stated. State the order by using one form. (Under
   netting `enter()` means "be in this position" and never adds: whichever order
   finds the position held does nothing for an entry, so a venue and a backtest
   differ only in which of the two opened it.)

**Hedging on a venue.** Under `oms_type="hedging"` every order opens a position
and a close names the position it ends. The engine sets
`OrderRequest.position_id` on a close, a reduction or a protective order it
makes of a named position; the kernel, `PaperExecution` and a `BrokerAdapter`
honour it, and an order without it opens (reduce-only, it takes the oldest
position). A venue that nets positions per instrument cannot close one by name:
a `BrokerAdapter` says so with `supports_hedging = False` (all the broker
mappings do), and a `LiveNode` with `oms_type="hedging"` on such an adapter is
refused when it is built (`hedging_unsupported_by_venue`). Nothing is sent that
the venue would read as a different book. A custom adapter that keeps positions
apart is hedging-capable unless it sets `supports_hedging = False`.

Closes and protection exit with reason `Order`, not `Signal`, `StopLoss` or
`TakeProfit`; the engine's own orders are extra order records.

Refused up front: ATR-based stops on a live node (it has no bar series, with
or without a venue), `liquidate_on_margin_call`, and **shared account groups.**
A group's cash is one pool that `AccountGroup::run_on` swaps into each member's
book on every step. Booking a venue fill, and the paper venue's shadow books,
bypass that, so a fill would credit a member's copy rather than the pool;
supporting groups means routing both through the group and giving the paper
venue the same group. At a real venue each account's orders already carry
`account_id`; the missing piece is the book side.

## The paper venue

`PaperVenue` holds a shadow `PortfolioBook` for each strategy, built with the
same configuration, instruments, account and adopted positions, and steps it
with the events the strategy sees. Its orders are matched, priced, charged and
sized by the kernel a backtest runs, so the fills of a run on the same stream
are the fills of a backtest: prices, quantities, times, fees, the equity curve
and every metric are identical (`tests/python/test_execution.py` compares the
whole result digest on the frozen stream fixtures, with queue model, partial
fills, order latency, fees, slippage and margin accounts).

The fills leave the shadow book as reports, and the strategy's book learns of
them the way it would from a broker. The differences from running in-process
that remain are all about what a venue can say:

* a close by `close_position` is a reduce-only market order, so it is subject
  to latency and partial fills like any order, and its trade is closed with
  reason `Order`;
* an order the matcher would reject after it had partly filled is cancelled
  (a venue does not reject what it has traded), and an exit larger than the
  position ends with a cancel of the surplus;
* an order's one-cancels-other link and its size resolved from "the whole
  position" are held by the venue, not the order record.

The shadow book may never change a position without an order. Its stops,
targets, schedules and square-off are venue-managed like the strategy's book
(the runtime sends the orders); a position it still changes on its own would be
a fault of the engine, not of the caller's input, and stops the run with an
internal `RuntimeError`. A contract's end is not an order's doing
but the instrument's declared terms: a position settled at expiry, delivered
against or assigned is settled alike by the strategy's book and the shadow's
copy, from the same events, and is not a change without an order.

## Recovery, the journal and fencing

Written in `journal-and-recovery.md`: every request is journaled before the adapter
is called, a request with no answer is reconciled and never blindly resent, an
adapter may offer `open_orders()`, `positions()`, `balances()` and `fence(epoch)`, and
corrections and busts are applied to the book. A fill the book cannot take
(`venue_fill_not_booked`) leaves the order with the fill and the position without it:
that is a divergence to reconcile, and it is raised, not hidden.
