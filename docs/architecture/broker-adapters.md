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
     the lookup returned an answer. Brokers offer no idempotency key, so this is
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

## Ids

The engine's `s0-12` becomes a tag by `IdCodec`: unchanged where hyphens are
allowed, `s0x12` where only letters and digits are. Any other id is hashed to
a fixed-length tag and remembered in a registry that `export_state()` carries.
`namespace="AB"` prefixes the tags of one node so two nodes on one account do
not read each other's orders.

## Recovery hooks (for the journal)

* `orders()`, `open_orders()` : `list[VenueOrder]` = `OrderStatusUpdate`
  (`venue_order_id`, `client_order_id` when the tag is ours, `status` in
  `received|open|filled|cancelled|rejected|expired`, `symbol`, `side`,
  `quantity`, cumulative `filled_qty`, `avg_price`, `limit_price`,
  `trigger_price`, `order_type`, `product`, `reason`, `ts_ns`).
* `positions()` : `list[VenuePosition(symbol, net_qty signed, avg_price, product)]`.
* `balances()` : `VenueBalances(available, used, total, currency="INR")`.
* `fence(epoch)` : monotonic; unsent jobs of an older epoch are dropped
  (`fenced_drop` event). It stops this process sending stale work; it cannot
  recall a request in flight and does not stop another process.
* `export_state()` / `import_state()` : JSON-able id maps, fill progress,
  pending modifies and unsent jobs; no credentials.
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
the venue took the order, a request lost on the way. Profiles: `in_order`,
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
