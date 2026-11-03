# Broker adapters

`raptorbt.adapters` is the layer between the engine's `ExecutionAdapter`
contract (`execution-adapters.md`) and a retail broker's HTTP and websocket
API. It is pure Python and has no dependency beyond the standard library.

```
LiveNode ──submit/cancel/modify──▶ BrokerAdapter ──outbox──▶ executor ──HTTP──▶ broker
   ▲                                    │ ▲                                         │
   └── node.push_report ◀── report queue┘ └──── handle_payload / handle_update ◀────┘ (feed, postbacks, polls)
```

## What is common and what is a mapping

`BrokerAdapter` (`adapters/base.py`) owns everything that is the same for
every broker. A mapping (`adapters/kite.py`, `upstox.py`, `dhan.py`) is thin:
it translates an `OrderRequest` into the broker's parameters, performs the
three calls and the reads, and turns the broker's rows into
`OrderStatusUpdate`s. Nothing else.

| common (base) | mapping |
| --- | --- |
| outbox, executor, FIFO order of requests | `translate`, `check_modify` (pure, local) |
| client id to tag codec (`IdCodec`) | `place`, `cancel_remote`, `modify_remote` |
| duplicate-order guard, lookup before any repeat | `fetch_orders`, `fetch_trades`, `fetch_positions`, `fetch_balances` |
| retries, backoff, per-endpoint token buckets | `parse_order_update` (feed message to row) |
| cumulative fills to deterministic fill deltas | error classification (`VenueRefusal`, `Transient`, `Ambiguous`, `AuthExpired`) |
| report queue, events, reconnect and resync hooks | broker limits (`default_limiter`), id length and charset |
| session hooks, state export and import, fence | |

## Lifecycle of a request

1. **Node thread.** `submit(request)` validates and translates, with no I/O:
   `Unsupported` for anything the broker has no equivalent of (order type,
   time in force, `post_only`, an unmapped symbol), `OrderRefused` for a local
   fact (lot size, tick size, a different order under a used id). Then it
   appends a job to the outbox and calls `executor.kick()`. A repeat of the
   same id and terms is a no-op.
2. **Executor.** Takes the job, waits for the rate limiter, calls the broker.
   * answered yes: the venue order id is recorded; the order's `accepted`
     report comes from the feed (or the next resync), when the broker says it
     is working;
   * answered no (`VenueRefusal`): a `rejected` report with the broker's code
     and message (for cancel and modify: `cancel_rejected`, `replace_rejected`);
   * not processed (`Transient`: 429, connection refused, 503 before the
     OMS): retried with backoff, then `rejected` ("venue unreachable"), since
     it certainly never left;
   * **may have been processed** (`Ambiguous`: timeout, reset, 5xx from the
     gateway): never repeated. The adapter looks the order up by its tag. Found:
     adopted. Absent: an `unconfirmed` event and the order stays `Submitted`
     for reconciliation; `resend_if_absent=True` sends it once more, only after
     `resend_delay` seconds and a second lookup that again finds nothing. Brokers offer no idempotency key, so this is
     the only way never to hold two live orders for one decision;
   * token no longer valid (`AuthExpired`): `Session(refresh=...)` is called
     once and the request repeated; with no refresh the outbox pauses, a
     `session_expired` event is raised and `set_token()` resumes it.
3. **Feed.** The host feeds every broker message to `handle_payload` (or a
   polled row to `handle_update`). The base resolves the order from the tag,
   folds the row into reports in a fixed order (accepted, replace_ack, fills,
   terminal) and queues them. Repeated, stale and reordered messages add
   nothing: fills are derived from the *cumulative* quantity and average price
   (`delta = cum - seen`, at the price that makes the running average right,
   execution id `{venue_order_id}:{cum_qty}`), and a terminal report is queued
   once.
4. **Host.** On the node's thread, after the call that caused them has
   returned: `adapter.deliver_to(node)` (or `drain_reports()` and
   `node.push_report(cid, report, arrival_ns=...)`).

## Threading

Three kinds of thread meet in the adapter; one re-entrant lock guards its
records.

| thread | does | never does |
| --- | --- | --- |
| node | `submit`, `cancel`, `modify`; drains reports into the node | I/O, sleeps |
| executor | drains the outbox: rate-limit waits, HTTP, retries | touch the node |
| feed | `handle_payload`, `handle_trade`, `connection_*` | touch the node |

Executors: `InlineExecutor` (default; runs the outbox before `kick` returns,
for tests and scripts), `ThreadExecutor` (one daemon worker; one thread keeps
a cancel behind its submit), `ManualExecutor` (the host calls `pump()`; also
what a test uses to look at the outbox). Any object with `bind`, `kick` and
`close` works.

## Fills: order updates or trades

`fill_source="order_updates"` (default) derives fills from the cumulative
figures of order rows, as above. The broker's own execution ids are not in
those rows, so ids are the deterministic ones. `fill_source="trades"` books
fills only from `handle_trade(exec_id=..., ...)` or `sync_trades()` (which reads
the trade book), with the broker's trade ids; order rows then carry status
only. Do not mix them for one order. The mappings report no fee: the book
charges its own model, and the venue's fee is for reconciliation.

## Ids and the namespace

The engine's `s0-12` restarts at `s0-0` every run, and the broker's day book keeps
every order of the day. So a tag alone never identifies an order of this run:

* `namespace` is **required** (empty is refused). Pass `new_namespace()` (four random
  letters and digits) at each start; a recovered run keeps its own, which
  `export_state()` carries and `import_state()` checks. It is the tag's prefix, so
  two runs, or two nodes on one account, do not read each other's orders, and it fits
  inside each broker's limit (Kite 20 chars, Upstox 40, Dhan 30).
* A venue row is adopted only if its terms (symbol, side, quantity, type, limit) match
  the order's; a lookup that finds several fitting rows adopts none (`ambiguous`).

The tag is the engine id behind the namespace: unchanged where hyphens are allowed,
`s0x12` where only letters and digits are. Any other id is hashed to a fixed-length
tag and remembered in a registry that `export_state()` carries.

## Authentication of push channels

`handle_payload` refuses to turn a message into a report unless the channel is
authenticated: build the adapter with `trusted=True` (the host opened the websocket
with the account's token itself), or with `verifier=` (for Kite postbacks
`kite.postback_verifier(api_secret)`, a constant-time checksum check), or pass
`authenticated=True` for one call. Anything else raises; a payload the verifier
refuses is dropped with a `payload_rejected` event. A message missing a quantity or
price is dropped with `malformed_payload`; reads (`fetch_*`, `orders()`, `positions()`,
`balances()`) raise `MalformedRow` instead of defaulting a missing number to zero.
`handle_update` takes typed rows: it is for the host's own polls and trusted code.

## Cancels and modifies that cannot be confirmed

A cancel or modify that runs out of retries triggers a resync and one more try. Only a
**definite** answer becomes `cancel_rejected` / `replace_rejected`: the venue said no, or
certainly never received the request. If the second try is ambiguous too (a timeout, a
gateway error) the request may have landed, so nothing is reported: the cancel stays
requested or the modify stays pending (`rec.quantity` is not changed, and a row carrying
either the old or the new terms is accepted), with a `cancel_pending` / `modify_pending`
event. The next resync, a feed message or `settle_pending()` decides against the venue's
actual book: a change that shows is acknowledged, a book that still shows the old terms or a
working order is answered as rejected. A broker that reports the failure itself (Upstox
`not modified` / `not cancelled`) is answered from the feed. Call `settle_pending()` after a
few seconds (it reads the book: from a worker or a timer, not the node's thread).

## Units, lots and signs

Everything the adapter hands out is in **units**, never lots, and everything it sends is in
the broker's count. A mapping builds its rows with the broker's own numbers; the base
converts them once with `units()` before anything else sees them: order rows (book reads,
`find_by_tag`, postbacks and stream frames), trades (`sync_trades`; `handle_trade(...,
wire=True)` for a host that feeds trades itself) and positions. Outbound, `wire_qty()` turns
units into lots for an instrument declared `Instrument(lot_size=..., quantity_in_lots=True)`
(a quantity that is not a whole number of lots is refused locally). The declaration is the
only input: the mapping's per-broker code never multiplies. A position or order row on an
instrument the map does not know cannot be converted; a position is returned as the broker
counted it with an `unmapped_instrument` event, an order row is not ours and is ignored.

Signs: `VenuePosition.net_qty` is long positive, short negative; flat rows are not listed;
`avg_price` is per unit and positive. `VenueBalances` amounts are in `currency`'s main unit;
`used` and `total` are `None` when the broker does not say. Prices are per unit everywhere.

A row whose status the mapping cannot read is never dropped silently: an `unmapped_status`
event carries a JSON detail (`order_id`, `status`, `tag`), once per order and status.

## Retention

Finished orders are forgotten `retention_s` (default one hour; `None` keeps all) after they
end. Reports for an order finish within seconds of its end, so an hour is ample. The ids of
forgotten orders are remembered (last 100 000), so a repeated submit is still refused and a
late message ignored. `sweep()` runs after each drain of the outbox.

## Recovery hooks (for the journal)

* `orders()`, `open_orders()` : `list[VenueOrder]` = `OrderStatusUpdate`
  (`venue_order_id`, `client_order_id` when the tag is ours, `status` in
  `received|open|filled|cancelled|rejected|expired`, `symbol`, `side`,
  `quantity`, cumulative `filled_qty`, `avg_price`, `limit_price`,
  `trigger_price`, `order_type`, `product`, `reason`, `ts_ns`).
* `positions()` : `list[VenuePosition(symbol, net_qty signed, avg_price, product)]`.
* `balances()` : `VenueBalances(available, currency, used, total)` (the mapping names its venue's currency; there is no default).
* `fence(epoch)` : monotonic; unsent jobs of an older epoch are dropped
  (`fenced_drop` event). **Without a lease it only guards this process**: it compares
  a number held in memory, so two processes that were each given an epoch both keep
  sending. It cannot recall a request in flight. `can_fence` is true only with a lease
  (below); a host or node that needs a fence across processes must refuse to run when it
  is false.
* `lease=` : the host's proof that this process may still write for the current epoch,
  `callable(epoch) -> bool` or an object with `holds(epoch)`. It is consulted before every
  outbound write (submit, cancel, modify, and each retry or resend), not for reads. A `False`
  answer, or an exception from the check, fails closed: the write is refused with `Fenced`,
  every queued write is dropped with a `fenced_drop` event (no report: the new owner
  reconciles from the venue), a `fenced` event is raised, and `submit`/`cancel`/`modify` raise
  `Fenced` at once, with no call to the lease, until `fence()` is given a higher epoch after
  the host has re-acquired it. The host owns the lease: for example a Redis key set with
  `SET key owner NX PX ttl` and renewed by the owner, where `holds(epoch)` is "the key still
  holds my owner token and epoch" (a Lua compare-and-check), or a database row updated with
  `UPDATE ... WHERE owner = me AND epoch = ?` that must affect one row. The check should be
  fast and local when it can (a cached lease with an expiry shorter than the renewal), because
  it runs on the executor before each write. A lease can lapse between the check and the
  send, which is why the venue-side tag lookup (above) still guards against a repeat.
* `namespace` : read-only property, the run's tag prefix (the node journals it and refuses
  a restore under a different one).
* `export_state()` / `import_state()` : JSON-able namespace, id maps, fill progress,
  pending modifies and unsent jobs (an order whose place request was started is looked up by
  tag before anything is sent again); no credentials.
* `resync()` : queue a read of the order book (also what
  `connection_restored()` does).

## Rate limits, reconnect, sessions

Each mapping builds a `RateLimiter` of token buckets per endpoint class
(`orders`, `modify`, `cancel`, `read`) from its broker's published limits; pass
`limiter=` to change them. `reconnect_loop(connect, on_lost=adapter.connection_lost,
on_connected=adapter.connection_restored)` reconnects with `Backoff`, and the
adapter reads the order book after every gap. The adapter holds a `Session`
(token, optional API key, optional refresh callable) in `Secret`s that print as
`Secret(****)` and refuse to pickle; nothing it logs or exports contains one.
Log in from one process only: a new login revokes the previous token on some
brokers.

## Conformance

`raptorbt.adapters.conformance` runs any adapter through a `LiveNode` against a
scripted venue that duplicates, delays, reorders and drops its messages and
restarts the adapter mid-script, and asserts the node ends with the same
orders, fills, average prices and positions as with an ideal venue, with no
order placed twice. Scenarios: fill, partial fills, reject, refusal at place,
cancel, cancel losing a race to a fill, cancel refused, replace, replace
refused, expiry, interleaved orders, a rate-limited place, an answer lost after
the venue took the order, a request lost on the way, and two with a lot size of 25
(partial fills, replace then fill: the broker counts lots, the node sees units). A scenario
with `lot_size > 1` passes `options["lot_size"]` to the binding, which must declare its
instruments in lots and render every row in lots. Profiles: `in_order`,
`duplicate`, `delay`, `reorder`, `drop`, `restart`, `chaos`.

```python
from raptorbt.adapters.conformance import run_conformance, ReferenceBinding
assert not run_conformance(ReferenceBinding)
```

For your own adapter write a binding (see the module docstring and
`ReferenceBinding`): it builds your adapter over a fake transport to the
`ScriptedVenue`, renders each `VenueEvent` as your venue would say it, and
drains the adapter's reports. The broker mappings are tested the same way over
synthetic servers in `tests/python/adapters/`.

## Support matrix

"No" means `Unsupported` is raised at `submit`; nothing is dropped silently.

| | Kite Connect v3 | Upstox v3 (writes) / v2 (reads) | DhanHQ v2 |
| --- | --- | --- | --- |
| market, limit | yes | yes | yes |
| stop-limit (SL), stop-market (SL-M) | yes | yes | yes |
| trailing, if-touched, market-to-limit | no | no | no |
| time in force | day, ioc | day, ioc | day, ioc |
| gtc (engine default) | only with `gtc="day"` | only with `gtc="day"` | only with `gtc="day"` |
| gtd, fok, at-open, at-close | no | no | no |
| `post_only` | no | no | no |
| `reduce_only` | only with `reduce_only="best_effort"` | same | same |
| products (tag `product`) | MIS, CNC, NRML, MTF | I, D, MTF | INTRADAY, CNC, MARGIN, MTF |
| after-market | `variety="amo"` | no (broker infers) | no |
| CO, BO, iceberg, auction, slicing | no | no (slice off) | no |
| modify | quantity, price, trigger | quantity, price, trigger | quantity, price, trigger |
| client id field | `tag` (20, letters+digits) | `tag` (40) | `correlationId` (30, letters digits `_-`) |
| quantity unit | units | units; lots if `quantity_in_lots` | units |
| expiry status | none (reads as cancelled) | none (reads as cancelled) | `EXPIRED` |
| feed | postback / order websocket | portfolio stream | live order update websocket |
| fills from | cumulative figures (`trades` optional) | same | same |
| order limits used | 10/s, 400/min | 50/s, 500/min, 2000/30 min | 25/s, 250/min, 1000/h |
| notes | 25 modifications per order | writes from the registered static IP; closed 00:00-05:30 IST | writes from the registered static IP; modification cap per order |

The mappings were written from the brokers' public documentation (URLs in each
module docstring) and are tested on synthetic payloads in those shapes only.
Field names that the pages do not state (Upstox status strings and stream
frames, Dhan's order-book `filledQty`/`averageTradedPrice` and trade-book
fields) are marked in the module docstrings; confirm them against a paper or
small live account before relying on them.
