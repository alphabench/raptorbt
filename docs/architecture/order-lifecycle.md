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
| `Rejected` | `Rejected`; an error if anything has filled |
| `Fill` | adds an execution, keyed by `exec_id`; implies acceptance; an error if the order would be over-filled |
| `CancelAck` | `Canceled` (also for a cancel the venue started itself) |
| `CancelRejected` | clears the pending cancel |
| `ReplaceAck` | applies the new total quantity, limit and trigger; an error if the new quantity is below what has filled, or a price does not exist for the order kind |
| `ReplaceRejected` | clears the pending replace |
| `Expired` | `Expired` |
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
