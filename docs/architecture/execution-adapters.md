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
| `submit_structure` | its legs are ordinary orders: identical results to `run_strategies` (tested) |

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
* **`"local"`.** The engine watches the levels on each bar or print and, when
  one is reached, sends a reduce-only market order. It reacts on the event, so
  the exit fills at that event's market price, not at the stop level: the same
  bar as a backtest, a different price (worse by the gap through the level).
  Use it for venues with no stop orders.

### What still differs from `run_strategies`, and the tests that pin it

`tests/python/test_execution.py` runs the frozen stream and portfolio scenario
corpus and the option-structure tests through a paper node and compares the
whole result digest with `run_strategies` (or the in-process node). Equal:
legacy intents, config stops and targets, attached levels, brackets, OCO,
trailing orders, `Twap`, timers, risk limits, drawdown and margin halts,
adopted positions, warmup, order latency, the session square-off and
structures. Not equal, each a strict `xfail` naming its reason:

1. **Sizing time of an entry intent.** `enter()` under `fill_timing="next_bar_open"`
   is sized from the capital on the signal bar and fills at the next open;
   in-process it is sized there (`pf_fill_next_bar_open`). (A capital-fraction
   *order* is sized by the simulated venue as it fills, which amends the
   order's quantity with a replace acknowledgement, so resting and sliced
   fraction orders equal in-process; a real adapter works the quantity
   estimated when the order was sent.)
2. **Hedging.** Under `oms_type="hedging"` every order opens, so no order can
   close one position: closes, protection and flattening by order are not
   representable (`pf_margin_hedging`, `pf_cash_hedging`, `st_margin_hedging`).
3. **Order of an entry intent.** An `enter()` held for the next print and a
   typed market order made earlier in the same step: in-process the entry fills
   first; at a venue they fill in the order they were sent
   (`st_refused_rows_timers`).
4. Closes and protection exit with reason `Order`, not `Signal`, `StopLoss` or
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
(the runtime sends the orders); a position it still changes on its own raises
`paper_unsolicited_position_change`.

## Not done here

Recovery, the journal and reconciliation (what to do with a venue order the
book does not know, or the reverse) belong to the journal work; this layer
gives them deterministic ids, a record of every request in tags and ids, and a
`Submitted` order that stays `Submitted` when a request may be lost. A fill the
book cannot take (`venue_fill_not_booked`) leaves the order with the fill and
the position without it: that is a divergence to reconcile, and it is
raised, not hidden.
