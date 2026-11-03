# The journal, checkpoints, recovery and fencing

What a live node writes down, how a node that died is rebuilt, and how a
stale node is stopped. The Python guide is the docstring of `raptorbt.journal`;
the Rust is in `src/journal/`; the recovery driver is
`python/raptorbt/strategy/_recovery.py`.

## The journal

```python
from raptorbt.journal import Journal
j = Journal({"run": "r-17"}, file="run.journal", fsync="orders", memory=True)
node = LiveNode(strategies, symbols, config, execution=venue, journal=j, epoch=3)
```

Off unless given: a node with no `journal=` and no `epoch=` writes nothing, does
not wrap its adapter, and pays one flag test per push (measured below).

A record is **written before the engine acts on it**: an input before it is
applied, a strategy's batch of commands before the engine routes it, an adapter
request before the adapter is called. A process killed anywhere leaves a journal
from which the node and the unanswered requests can be rebuilt.

The journal is attached, and `start` is its first record, before any strategy
code runs. What `on_init` and `on_start` do (a halt, a position adopted from
`initial_positions`) follows `start` as records. The orders `on_start` queues are
not discarded by muted warm-up (it discards what history's hooks queue): they are
routed with the first live event and journaled then like any other batch.

Everything that changes an order, a position or the cash is either an input (a
bar, a tick, a report, a halt, an annotation: replaying it re-derives the change)
or has a record of its own. `tests/python/test_journal_completeness.py` drives a
node through each change and fails if one writes nothing, and replays every
journal into a fresh node that must write the same records.

| kind | source | holds |
| --- | --- | --- |
| `start` | engine | node shape: symbols, strategy count, execution, the adapter's `namespace`, strategies' initial states |
| `epoch` | engine / host | the execution epoch the node runs under: one right after `start`, one more each time a restore takes a newer epoch |
| `tick` `bar` `depth` `time` | feed | one push, event and arrival time; `bar` says if it was warmup |
| `report` | venue | an execution report, with its `client_order_id`: written first, before the node decides anything about it (a report for an unknown or evicted order is on the record too) |
| `unbooked_fill` | engine | an execution the book could not take: a fill it could not book yet, a restated fill it could not book again, or one for an order the route no longer holds (`order_evicted`) |
| `command` | strategy | a batch of commands (submit, cancel, modify, close, link) |
| `refusal` | engine | the code a batch was refused with |
| `adapter_request` | engine | `submit` / `cancel` / `modify` / `link_oco` / `place_exit_rule` and its request, **before** the call |
| `adapter_result` | engine | `ok`, `refused`, `unsupported`, `error` (an exception: may or may not have arrived), `fenced`, `unknown_outcome`, `not_sent` |
| `halt` | host / strategy / engine | `node.halt`, `halt_account`, or `ctx.halt`; from the engine: the day's loss limit (`daily_loss`), the drawdown limit (`drawdown`), a margin call (`margin_call`), written before the strategy hears of it and before anything reaches a venue |
| `annotation` | host | `node.annotate(ts, kind, detail)`; a `resend` |
| `instrument_add` `instrument_retire` | host | an instrument added to or retired from the running node (the spec as JSON); replayed by `restore` |
| `restatement` | engine | a correction or bust the book followed |
| `unbooked_fill` | engine | a venue fill the order took and the book could not book yet. At most 1024 are kept at once: the next one raises `venue_fill_not_booked` instead of joining a list nobody reads. They are tried again after every booking and every restatement (a bust or correction frees cash, margin and room) |
| `order_resized` | engine | the engine cut a working order's units (a one-cancels-other sibling of a partly filled leg): `from`, `to` |
| `rejection` | engine | the book refused an order or entry it had accepted from the strategy (`position_open`, a limit, sizing), or booked a venue fill past a limit (`event: fill_breach`): `event`, `reason`. A venue's own refusal is its `report` record: the reason and the client order id, whose terms are in the `adapter_request` written before the send |
| `engine_exit` | engine | a position the engine closed by a rule of its own: `stop_loss`, `take_profit`, `trailing_stop`, `liquidation`, `time_exit`, `squareoff`; on a node with a venue the decision (`squareoff`, `watch_stop`, `watch_target`, `venue: true`) comes before the close order's `adapter_request` |
| `lifecycle` | engine | a contract settled (`what: settlement`), or a corporate action applied or refused |
| `simulated_report` | engine | what the simulated venue (`PaperExecution`) answered, stated like a `report` (reason included); derived, never replayed as an input |
| `adoption` | host | a position the account already held was given to the book (`initial_positions`) |
| `error` | engine | a push that failed after it was journaled, so a replay expects the failure |
| `reconcile` `discrepancy` `resume` `fence` `finish` | host / engine | recovery, fencing, the end |

**Encoding.** Canonical JSON: keys sorted, no whitespace, integers as integers,
text ASCII-escaped (byte for byte `json.dumps(sort_keys=True,
separators=(",", ":"))`). **No float:** a float is a string of its shortest
round-trip decimal (`"100.5"`, `"NaN"`, `"-0"`), so a record's bytes do not
depend on who wrote them. `record.canonical()` returns exactly the bytes the
chain is taken over; `record.payload` returns the decoded tree (floats as those
strings).

**Chain.** `h_0 = sha256(canonical header)`, or the head of the run being
continued; `h_n = sha256(hex(h_{n-1}) || canonical_n)`. `raptorbt.journal.chain_step(prev_hex,
canonical)` is one link and is pure. `Journal(header, seq_base=N, chain=prev)`
continues a run: sequence numbers and the chain go on. `journal.head()` is
`(count, chain, header)`, where `count` is also the next sequence number.

**Reading.** `journal.since(seq)` returns records with `seq >= seq`, oldest
first; `Record` has `seq`, `ts_event`, `ts_arrival`, `kind`, `source`,
`client_order_id`, `payload`, `chain` and `canonical()`. A bounded memory
(`memory=N`) raises `journal_evicted` for what it dropped. `raptorbt.journal.read(path)`
verifies every checksum and the chain.

**Sinks.** A file (frames of `u32 length, u32 crc32, canonical bytes`; a cut last
frame is reported as `torn_bytes` and dropped on reopen, a corrupt frame with
whole ones after it is an error), memory, and a host callback that receives each
`Record` before the node acts on it, so the host can be the durable store. A
sink that fails closes the journal (a gap in a chain cannot be repaired) and the
node refuses to go on (`journal_failed`) *before* the engine acts.

**fsync** (file sink): `"every"`; `"orders"` (default; after commands, requests,
results, reports, halts, refusals, restatements, resume and finish); `N` (every
Nth record); `"never"`. A record always reaches the operating system when it is
appended, so a killed process loses nothing under any policy; the policy decides
what a power loss can take.

## Checkpoint and restore

The engine is deterministic: the same inputs in the same order give the same
books, orders, indicators and ids. A node is therefore rebuilt by **replaying its
journal** into a fresh node with the adapter muted, checking every record the
replay makes against the one the journal holds. There is no serialised heap, so
there is nothing to go stale across versions; the cost is that restore time grows
with the journal since `seq_base`, so a host that runs for days starts a new
journal per session (`seq_base` and `chain` continue it) and carries strategy
state across with `state()` / `load_state()`.

`node.checkpoint(include_log=True) -> bytes` (version 1): the journal's header,
`seq_base`, chain start, head (count, chain), the epoch, a digest of the books
and orders (`node.state_digest()`), each strategy's `state()` (JSON), and the
log. With `include_log=False` it is a small blob and the host supplies the
records at restore.

`LiveNode.restore(checkpoint, strategies, symbols, config, *, journal=None,
records=None, execution=..., reconcile=True, accept=None,
on_discrepancy=None, **the arguments the node was built with)`.
`raptorbt.journal.open_for_recovery(path)` gives `(journal, records)` from a file. `symbols` and
`instruments` are those the node was *built* with: the instruments it added later come back from the
journal's `instrument_add` records (with their specs), in the order they were added.

## Recovery

1. **Load** the checkpoint, or the journal's records.
2. **Replay** them into a fresh node. The adapter is muted: a request already in
   the journal is checked against the one the replay makes and gets the answer it
   got then. A replay that makes a different record raises `replay_diverged` (a
   different feed, symbols, or a strategy that is not deterministic).
   The state digest is checked at the checkpoint's offset.
3. **Finish the step the crash interrupted.** Records past the end of the journal
   are written fresh. A request the crashed node had written but not answered
   gets result `unknown_outcome`; one the recovery would have made gets `not_sent`.
   Neither is sent. The order stays `Submitted`.
4. **Reconcile.** If the adapter has any of `open_orders()`, `positions()` or
   `balances()`, the node compares. The shapes are fixed, and the types a
   `BrokerAdapter` returns are accepted as they are:

   | call | returns | read as |
   | --- | --- | --- |
   | `open_orders()` | items with `client_order_id` (`None` for an order that is not this node's), `venue_order_id`, optional `filled_qty`, `symbol`, `quantity` | a foreign order is keyed `venue:<venue_order_id>`, so two of them are two discrepancies |
   | `positions()` | a list of rows with `symbol` and `net_qty` (`VenuePosition`; `quantity` also reads), or a `{symbol: signed qty}` mapping | the net quantity per symbol; rows of one symbol add |
   | `balances()` | an object with `available` and `currency` (`VenueBalances`), or `{"cash": x}` | `available` (or `cash`) is compared with the node's cash; the journal keeps `currency`, `used` and `total` |

   Discrepancies: `missing_order` (the node has it working or unconfirmed; the venue
   does not list it), `unknown_order`, `order_mismatch` (filled quantity),
   `position_mismatch`, `cash_mismatch` (to `cash_tolerance`). Each `Discrepancy` is
   journaled, handed to `Strategy.on_discrepancy(ctx, d)` and to `on_discrepancy=`. An
   adapter with none of the three is refused (`no_reconcile`) unless
   `reconcile="skip"`, which is journaled.
5. **Resume only when consistent.** With no discrepancy the node resumes. With some
   it stays `state == "recovering"`: market data and `finish` are refused
   (`recovering`); `push_report`, `resend`, `reconcile` and `annotate` are not. The
   host pushes the reports it has, `resend(client_order_id)`s an order the venue
   does not hold (same client id, so a venue that deduplicates takes it once), and
   `resume(accept=...)`s the rest explicitly (`"all"` or the discrepancies).

An order whose request is journaled but whose outcome is unknown is **never sent
again by recovery**: reconciliation asks the venue first.

## Fencing

`LiveNode(epoch=E)`. An adapter with `fence(epoch)` raises
`raptorbt.execution.Fenced` when a newer epoch owns the account. The node calls it
before every request (an exit rule placed at the venue included) and on
`node.heartbeat()`. After a `Fenced` the node writes a `fence` record, sends nothing
more, and every push raises `fenced`; `node.fenced` and `node.health()` say so. A node
given `epoch=` with an adapter whose `can_fence` is not true (a `BrokerAdapter` needs a host
`lease=`; a bare adapter has no such attribute) is **refused** (`fence_unsupported`): the epoch
would protect nothing. There is no orchestrator dependency: the epoch is a
number the host assigns and the venue enforces.

An adapter's requests that raised, or whose result was never written, are tracked as
unconfirmed in memory, journal or not (`node.health()["unconfirmed"]`). Replaying a journal
whose push raised reproduces the recorded `error` and does not call the adapter.

**Takeover.** `LiveNode.restore(..., epoch=E+1)` replays the journal under the epoch it
was written with (the epoch is the `epoch` record after `start`, not part of what a
replay compares), then claims the new epoch with the venue (`fence(E+1)`, so the old node
is fenced at its next request) and journals an `epoch` record. A journal that holds an epoch
is restored only under a newer one: a restore without `epoch=` raises `epoch_required`, and
one with the same or an older epoch raises `stale_epoch` (two nodes would otherwise hold one
epoch, and neither fences the other). Host-supplied leases sit behind the adapter's `fence`:
a `BrokerAdapter` built with `lease=` runs the lease check inside `fence(epoch)`, so a node
that lost the lease is fenced by its next request or its `heartbeat()`, and the writes it had
queued but not sent come back as `rejected` reports with the reason `fenced` instead of
staying journaled as sent.

**Working orders after a restart.** The replay runs with the adapter muted, so an adapter
built after the restart has never seen the orders the node holds as working. After the replay
the node hands them to `adapter.restore_orders(...)` (a `BrokerAdapter` has it): the adapter
registers each with the node's own fill progress, finds it in the day's book by tag and terms,
and binds the venue's id. Fills the venue made meanwhile then arrive as the difference from
what the node booked, and a cancel reaches the order. An adapter that cannot account for one
(its symbol is not mapped) refuses the restore (`adapter_cannot_adopt`). A venue row the
adapter ignores raises an `ignored_update` event once per order and status.

## The adapter's namespace

The engine names its orders `s{strategy}-{n}` and starts again at `s0-0` in every run, so
two runs on one account send the same ids. An adapter's **namespace** (a non-empty string
attribute, `BrokerAdapter.namespace`) marks the tags of one node, and is how a rebuilt node
recognises its own working orders at the venue. A node refuses an adapter without one
(`adapter_namespace_missing`); the node journals it in `start`; `restore` refuses an adapter
whose namespace differs (`adapter_namespace_mismatch`), because that adapter could not decode
the node's orders, `reconcile` would call them missing, and `resend` would place second live
orders. With the namespace journaled and required, the repeating ids are safe.

## Corrections and busts

A `Correction` or `Bust` of an execution the book already took is applied through
the book, exactly: the cash, the margin lock, the position's booked cost and entry
fees come off in the minor units they were booked in, and a correction books the
execution again as restated (tested against a book that was given the restated fill
to begin with). Supported: an execution that opened or added to a position that has
not been reduced since. Refused with `correction_not_reversible`, before anything
changes: an exit (a trade was realised), or an entry whose position has since been
reduced or closed. A fill that **reversed** a position (it reduced one and opened the other
way) made two bookings; taking back only the opening would leave the reduction, so a
correction or bust of it is refused too, whatever the policy. Under hedging a fill that
names no position opens its own, so its bust is a single entry and is applied. The strategy hears no event for a restatement (the journal has a
`restatement` record); it sees the restated position at its next event. Protective
orders the venue holds for the original quantity are not resized: reconcile.

## Restore time

Restore replays the journal through the engine, so it costs about what running those
records cost, plus a few microseconds a record to compare what the replay writes with
what the journal holds. Measured on a laptop (Apple silicon, release build, a loaded
machine, two symbols, a paper venue, a strategy that orders every few bars, journal in
memory, nothing else running in the node):

| records | restore | per 100,000 | peak memory (original node, journal, replay) |
| --- | --- | --- | --- |
| 100,000 | 3.5 s | 3.5 s | 0.6 GB |
| 1,000,000 | 31 s | 3.1 s | 2.2 GB |

The time is linear in the journal. It was not before the order book stopped scanning
every order ever placed (19 s for 100,000 records and 4 minutes for 1,000,000): the
cost of a node's every event grew with its order count, in a live node as in a replay.
Of the 3 s, about 60 % is the engine pushing the bars and 25 % the replay driver in
Python. A heap snapshot is not worth building for this: a journal bounded to a session
restores in seconds, and a host that keeps days of journal starts a new one per session
(`seq_base`, `chain`) and carries state across with `state()` / `load_state()`.

## Known limits

* Restore replays the whole journal since `seq_base`, a few seconds for every 100,000
  records. A true snapshot of the engine's heap is not taken (the indicator banks,
  kernels and Python strategy objects are not serialisable), so keep sessions' journals
  bounded.
* A strategy that reads the wall clock, a random source or the environment is not
  deterministic and fails the replay check.
