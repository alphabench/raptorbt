# Order lifecycle

An order is driven in one of two ways, and never both:

- **In-process.** The matcher decides fills against bars or ticks and moves
  the order with `Order::transition`. An illegal transition is a bug in the
  engine and panics in every build.
- **Venue-driven.** An external venue (a live exchange connection, a paper
  venue) answers with `ExecutionReport`s, and `Order::apply_report` folds
  them in. The reports can repeat, arrive late or arrive out of order, and
  a bad one returns an `OrderEventError`; the order is left unchanged.

How an order gets to a venue, and what the book does with a fill, is in
`execution-adapters.md`.

An order the matcher has already moved (filled, ended, or with a filled
quantity) refuses reports (`NotVenueDriven`), so the two modes cannot mix.

## States

```
                 request_cancel
   +--------------------------------------------+
   |                                            v
Submitted --Accepted--> Accepted --stop touched--> Triggered      PendingCancel --CancelAck--> Canceled
   |  \                    |  \                       |              ^   |  \--CancelRejected--> (back to
   |   \--Rejected         |   \--Fill(part)--> PartiallyFilled      |   |                        the working state)
   |       v               |        |  \                |            |   \--Fill / Expired / Rejected
   |    Rejected           |        |   \--Fill(rest)--> Filled      |
   |                       |        |                                |
   |                       +--request_cancel-----------------------+
   |                       |
   |                       +--request_replace--> PendingReplace --ReplaceAck------> (working state, new terms)
   |                                                    |    \--ReplaceRejected--> (working state, old terms)
   |                                                    +--request_cancel--> PendingCancel
   +--Expired (time in force) from any working state --> Expired
```

Working states: `Submitted`, `Accepted`, `Triggered`, `PartiallyFilled`, and
the two pending states. Terminal: `Filled`, `Canceled`, `Expired`,
`Rejected`. `Triggered` is set by the in-process matcher only.

`OrderStatus::can_transition_to` is the table. It adds, to the in-process
transitions:

| from | to |
|---|---|
| `Submitted`, `Accepted`, `Triggered`, `PartiallyFilled`, `PendingReplace` | `PendingCancel` |
| `Accepted`, `Triggered`, `PartiallyFilled`, `PendingCancel` | `PendingReplace` |
| `PendingCancel`, `PendingReplace` | any working state, `Filled`, `Canceled`, `Expired`, `Rejected` |

A request is not an acknowledgement. `request_cancel` and `request_replace`
only move the order to a pending state. It becomes `Canceled` on
`CancelAck`, and takes the new terms on `ReplaceAck`; until then it can
still fill, and every such fill is real. A negative answer
(`CancelRejected`, `ReplaceRejected`) returns the order to the state its
facts imply.

## Reports

| report | effect |
|---|---|
| `Accepted` | records the venue order id; a different id later is an error |
| `Rejected` | `Rejected`; an error if any fill was ever reported for the order, busted or not |
| `Fill` | adds an execution, keyed by `exec_id`; implies acceptance; an error if the order would be over-filled, or if `exec_id` was already reported with a different quantity, price or fee (`ConflictingDuplicate`: the id is the idempotency key, so a repeat must be identical) |
| `CancelAck` | `Canceled` (also for a cancel the venue started itself); `Late` once the order expired |
| `CancelRejected` | clears the pending cancel |
| `ReplaceAck` | applies the new total quantity, limit and trigger to a pending replace, also when the order has since filled, been cancelled or expired; an error if the new quantity is below what has filled, or a price does not exist for the order kind |
| `ReplaceRejected` | clears the pending replace |
| `Expired` | `Expired` (outranks a cancel) |
| `Correction` | restates one execution's quantity and price; the latest `ts_ns` wins |
| `Bust` | removes one execution |

`apply_report` returns a `ReportOutcome`:

- `Applied`: the report changed the order.
- `Duplicate`: already recorded (same execution id, same acknowledgement);
  nothing changed.
- `Late`: it no longer matters, or cannot matter yet. A fill that arrives
  while a cancel is pending or after the cancel was acknowledged is `Late`
  and still counts, because the venue executed it; a cancel
  acknowledgement for an order that already filled is `Late` and leaves it
  `Filled`; a correction or bust for an execution not yet seen is held until
  the fill arrives.

## Convergence

Status is derived from recorded facts (accepted, rejected, cancel
acknowledged, expired, outstanding requests, the execution ledger), not
advanced by whatever arrived last. Facts only accumulate, so any reordering
or repetition of the same reports gives the same status, filled quantity,
average price and fees. Ledger sums run in `exec_id` order so the floating
point result is bit-identical. Two consequences follow:

- A fill that completes a `Canceled` or `Expired` order makes it `Filled`:
  it traded in full.
- A bust that takes a `Filled` order below its quantity reopens it
  (`PartiallyFilled` or `Accepted`), unless a cancel or expiry was already
  acknowledged, in which case it ends as that.

The rule for each pair that can disagree:

| facts | the order ends as |
|---|---|
| `CancelAck` and `Expired` | `Expired`, whichever arrives first (a cancel already reported is superseded) |
| `Rejected` and `CancelAck` / `Expired` | `Rejected`, whichever arrives first |
| `Rejected` and a `Fill` (even one busted) | an error on whichever of the two arrives second |
| `ReplaceAck` and a `Fill`, `CancelAck` or `Expired` | the replaced terms apply in every order; the status is derived from them |
| `Correction` and `Bust` of one execution | the bust; the latest correction by `ts_ns` otherwise |

The pending flags (`request_cancel`, `request_replace`) are local requests
and are not cleared by a rejection or expiry, so they cannot make a later
acknowledgement behave differently from an earlier one.

Limits of the guarantee: it covers reports that agree with each other. A
fill after a reject is an error, not a reordering. An over-fill is refused,
never clamped, so a venue that restates an execution (bust then re-fill)
must not be delivered in an order that over-fills the order for a moment.
A replace that changes the quantity must be acknowledged before fills that
rely on the new quantity are applied.

## One-cancels-other and partial fills

Legs of an OCO group are alternative ways out of the same quantity. When one
leg partly fills, each working sibling with an explicit unit size is cut to
the quantity it has already filled plus the quantity still unfilled on the
leg that traded. It is never raised. A sibling sized by capital fraction or
whole position is left alone, since it resolves when it fills. When a leg
fills completely, the working siblings are cancelled, as before. The rule
runs only with `partial_fills` on, since without it no leg fills in part.

Each cut is an `order_resized` event (`size` the units the sibling offers now,
`previous_size` what it offered), delivered to `on_order_event` alone, in every
runner, right after the fill that caused it. A live node with a journal writes an
`order_resized` record for it (`from`, `to`), so a change to a working order is never
silent. An execution adapter's own venue cuts its own siblings: the node hears of
that through the reports the venue sends.

## Asking for an order that is not there

A strategy cancels or modifies an order by client id. Two cases differ.

- **A client id the run never submitted** (a typo, or an order from another
  run) is a mistake, and is refused with `unknown_order`, naming the id and
  the ids the run did submit. Modifying a `Twap` schedule is refused the
  same way: a schedule can only be cancelled.
- **An order that was submitted and has since finished** (filled, cancelled,
  expired or rejected) is a race the strategy cannot always win. Asking for
  it again changes nothing and raises nothing. This is the same answer a
  venue run gives: a repeated or late request is a no-op there, and a venue
  that refuses a cancel answers with a `CancelRejected` report that returns
  the order to its working state.

## Orders that wait for a hedge

A sold option that its account refuses for want of capital or margin, while
another option in its margin group has a buying order working, stays
working instead of being rejected. Instruments are stepped in a fixed order,
so the sold leg's event can come before the event that fills its hedge; the
leg is funded, with the group's credit, at its next event once the hedge is
held. If the hedge never fills, the sold leg is refused when it is next
stepped without a pending hedge, as before.

## Netting: adds, reductions and flips

Under `oms_type="netting"` an instrument has one position, and an order moves it as a
broker's net position moves. The same rules apply in a batch run, a `LiveNode`, a paper
node and an adapter node (a venue's fills are facts and follow the same book):

| Order | What the book does |
|---|---|
| on the side held | **adds**. Size adds; the entry price becomes the size-weighted average of the position and the fill. The fill pays its entry fee on its own; the position's entry fees are the sum. Each fill is a record of its own (an `OrderFilled` and an `Entered`, the order log, the journal). |
| against the position, smaller | **reduces**. One trade is booked for the reduced size at the **average** entry price (the way Indian brokers' net positions book realised P&L); the rest stays open at that average. |
| against the position, equal | **closes** it. |
| against the position, larger | **flips**. It closes what is held (a trade) and opens the rest on its own side at the same fill price, as one order. The close and the open are two fills with a fee each; a per-order brokerage cap is shared by both (`fee_model.begin_order`). |

Only a unit-quantity order that is neither `reduce_only` nor tied to a position flips. A
`reduce_only` order, an order that names a position (`position_id`), and an order sized as a
fraction of capital or as the whole position stop at flat. If the rest of a flip cannot open
(halted, no margin, a limit), the order ends `Filled` at what it closed: its quantity is cut to
that (an `order_resized` event) and the refusal is counted, so no order ends rejected with fills.
With volume-sliced fills (`partial_fills`) a flip closes on the print that has the volume and the
rest opens on the following prints.

An add is gated like an entry, once, on the order's first fill: a halt (`killed`, `drawdown_halt`,
`daily_loss`, `margin_call`), the order and position size limits, funding. `max_positions` does not
count it, since the position already counts; the position and exposure caps judge the position
after the add.

**Protective levels belong to the position.** A level derived from the entry (percent, ATR) follows
the new average. A `stop_price` / `target_price` the added order names becomes the position's level
(the other is kept): the whole net position is protected at it. Under a venue that holds the
protection (`protective="venue"`) the protective orders keep the size they were placed with.
**Brackets** are `reduce_only` legs of the full position, so a bracket made with an added slice
protects the net position, earlier slices included; when one bracket's leg fires and the position
is flat, the other brackets' legs are refused as any reduce-only order with nothing to reduce.
Legs with an explicit quantity close at most that quantity, never more than is held.

Hedging is unchanged: every order opens a position of its own and a close names its position.
