# Event semantics

How RaptorBT orders events in time, what a strategy can see at each step, and
when its orders execute. This describes current behaviour; the code references
are the source of truth.

## Timestamps

- Every timestamp is an `int64` count of nanoseconds. Engine code carries it
  through unchanged: it is never rounded, shifted or rewritten.
- `session_tz_offset_ns` only decides where calendar-aligned windows (daily
  composite bars, `DAY` order expiry) start. It does not alter any timestamp
  (`src/data/aggregate.rs`, `window_of` / `window_end`).
- Input must be sorted **ascending**. A timestamp lower than the one before it
  is refused with a `ValueError` naming the instrument and stream, the index of
  the offending element and the pair of timestamps, for example
  `A bars: timestamps must not decrease, but index 2 is 1 after 2`. Equal
  timestamps are allowed and keep their input order.
- Where it is checked:
  - the event session when data is attached (`set_bars`, `set_ticks`,
    `set_depth`) and on every live `push_tick` / `push_bar` / `push_depth`
    (`src/portfolio/session.rs`, `src/data/feed.rs::ensure_time_sorted`);
    this covers `run_portfolio_strategy`, `run_tick_strategy` and
    `TickStrategyStream`;
  - every array runner, at the binding boundary (`src/python/bindings.rs`):
    `run_single_backtest`, `run_basket_backtest`, `run_portfolio_backtest`,
    `run_multi_backtest`, `run_pairs_backtest` (each leg),
    `run_options_backtest`, `run_spread_backtest`, `batch_spread_backtest`,
    `batch_single_backtest` and `run_tick_backtest`; the error names the
    instrument, leg or series, for example `PAIR leg 2 bars`;
  - `run_strategy_backtest` (`python/raptorbt/strategy/runner.py`).
- A live feed is checked per instrument, against that instrument's latest
  event (including any warmup history). Different instruments may interleave
  freely.

## Merge order

Several instruments, and the trade / quote / depth / bar streams of one
instrument, are merged into one schedule. The canonical key, ascending, is

    (timestamp, phase, stream registration order, sequence)

- **timestamp**: the event's own timestamp.
- **phase**, lowest first: trade (0), quote (1), depth (2), bar (3)
  (`MarketEvent::phase`, `src/data/events.rs`). Intra-bar data therefore comes
  before the bar that summarises it: a bar stamped `t` is delivered after
  every tick at `t`.
- **stream registration order**: streams are numbered in the order the
  session registers them: all bar streams by instrument, then each tick
  instrument's trade and quote streams, then depth streams
  (`EventSession::seal`).
- **sequence**: position within the stream.

The key is a total order, so identical inputs always produce the identical
schedule (`src/data/feed.rs`, `Entry::key`). Live pushes are the exception by
design: they are appended in arrival order after the sealed batch data, and
arrival order is the schedule.

## Equity sampling

The event session records **one equity point per distinct timestamp**, taken
after every event at that timestamp has been applied
(`EventSession::apply_current`, `src/portfolio/session.rs`).

- If two instruments print at the same instant, the second event replaces the
  sample taken after the first. N simultaneous instruments are one observation,
  not N, so returns and annualisation do not depend on how many instruments
  share a clock.
- Quotes and depth updates never sample equity.
- Risk checks (the drawdown kill-switch, the margin-call test) still run after
  every trade or bar event, not once per timestamp, so they react as soon as
  equity moves.
- Periods per year is inferred from the median spacing of the sampled
  timestamps (`src/metrics/annualization.rs`), so it follows the same
  one-sample-per-timestamp curve.

## Composite bars

Bars that the engine builds from finer data (`primary_bars`, `subscribe_bars`,
`BarAggregator`) are stamped at the **end** of their window. A bar labelled `t`
contains only data strictly before `t` (`src/data/aggregate.rs`). The bar is
emitted when the first record of a later window arrives, or when the stream is
flushed at the end of data.

## Lifecycle: corporate actions and expiries

Corporate actions are data the run is given (`CorporateAction`), never
inferred from prices. At a timestamp `t` the order is:

1. corporate actions with `ex <= t`, in effective order (ties keep input
   order): a split or bonus rescales the position, a dividend fixes its
   entitlement on the position held *before* anything at `t` trades;
2. dividends whose pay date is `<= t` are credited;
3. contracts expired by `t` that have no bar of their own at `t` settle (see
   below);
4. the market event at `t`, with its hooks and its orders.

Steps 1 to 3 run before the strategy's hooks for the event, so `on_bar` at the
ex-date sees the position on the new basis and the price after the action. The
strategy hears each as an event first: `on_corporate_action`,
`on_contract_event`. A bar-stamped action is therefore effective from the first
bar at or after its timestamp, never mid-bar.

A contract settles on the first event of its own at or after its expiry (the
kernel's expired branch), or, when it has no later bar, on the first event of
any instrument at or after the expiry (the book's timer). A contract whose
terms read its underlying (`settlement_price="underlying"`, or physical
settlement) and whose underlying has not printed at the instant the expiry
was noticed waits for that print and settles on it; if the underlying does not
print at that instant it settles on the underlying's last price at the next
event. A live host that only moves the clock calls `advance_to(ts)`. Orders
submitted on a contract at or after its expiry are rejected with reason
`expired`.

See `docs/architecture/lifecycle.md`.

## What a strategy sees, and when orders execute

At a bar event the strategy sees that whole bar, close included. Execution
timing (`fill_timing` in `BacktestConfig`, `src/core/types.rs::FillTiming`)
says when a decision made on bar `i` can execute:

| `fill_timing` | Market order or signal from bar `i` fills | Notes |
| --- | --- | --- |
| `same_bar_close` (default) | at bar `i`'s close | decision and price coincide |
| `next_bar_open` | at bar `i+1`'s open | a signal on the last bar never fills |
| `same_bar_open_lookahead` | at bar `i`'s own open | a price that traded before the data the decision used; kept only to reproduce old results |

The deprecated `upon_bar_close` flag maps to the first two values when
`fill_timing` is not set (`BacktestConfig::resolved_fill_timing`).

Resting orders (limit, stop, stop-limit, trailing, if-touched) are different:
they are matched against a bar only after the bar they were submitted on, so a
resting order submitted on bar `i` can first fill on bar `i+1`
(`OrderEngine`, `src/execution/orders/matching.rs`: orders with
`submitted_idx >= idx` are skipped). A stop-limit whose trigger is touched
rests as a limit that becomes marketable from the next bar.

When one bar touches both legs of a bracket (stop and target), the leg that
was submitted first wins, and the bracket helpers submit the stop first. Its
fill cancels the other leg, whose queued outcome for that bar is dropped
(`EngineKernel::apply_match_outcome`): an outcome computed for an order that
is no longer working is never applied. `bar_path_adaptive` does not reorder
bracket legs; it picks the first-touched level only for a position's own
`stop`/`target` attachments (`try_exit_position`).

In tick runs:

- Orders match against trade prints. Quotes are observation only: an order
  submitted from `on_quote` rests and matches on the next print.
- Within one timestamp a trade is delivered before the quote of the same row,
  and both before any bar stamped at that timestamp, so a handler reading the
  book during a trade sees the book from before that print.
- `submitted_idx` is a per-instrument event ordinal, not a bar index
  (`ScheduleEntry::local_idx`).

## Bar stamps

An input bar's timestamp can mean its start (`open`) or its end (`close`).
The strategy entry points that take bars (`run_strategy_backtest`,
`run_portfolio_strategy`, and the warmup bars and `push_bar` of
`TickStrategyStream`) accept `bar_stamp="open" | "close" | None` and
`bar_ns` (an int, or a per-symbol dict) and normalise to one internal
convention: **visible time is bar close time**
(`python/raptorbt/strategy/_stamps.py`).

- `"close"`: visible time is the stamp.
- `"open"`: visible time is `stamp + bar_ns`. A missing, non-positive or
  fractional `bar_ns`, or a dict missing a symbol, is a `ValueError`.
- Everything time-sensitive reads visible time: the merged schedule (a bar
  after trades at equal time is right for a close time), `DAY` order expiry,
  the strategy clock, fills and order timestamps, and `ctx.datetime`.
- Composite bars are built from visible time with right-closed windows
  (`BarAggregator(..., stamp="close")`): a bar at `t` joins the window ending
  at `t`, and the composite is stamped at that window end.
- `ctx.timestamp` and `ctx.bar.timestamp` stay the caller's own stamp;
  `ctx.bar_close_ts` is visible time. They are equal for `"close"`.
- `None` warns (`DeprecationWarning`, to become an error in 1.0) and keeps the
  old behaviour unchanged: composites are built as if stamps were open stamps
  (`BarAggregator(stamp="open")`) and the schedule treats the stamp as the
  event time. Tests: `tests/python/test_bar_stamp_runners.py` runs one dataset
  stamped both ways and requires identical decisions, orders, fills, composite
  bars, equity and metrics.

Scope:

- `run_tick_strategy` has no bar input. The bars it builds from prints
  (`primary_bars`) are stamped at window end, a close time, and its composite
  subscriptions are built from prints, which are instants.
- The vectorised array runners (`run_single_backtest`, `run_portfolio_backtest`,
  basket, pairs, options, spreads) take no `bar_stamp`. They read the
  timestamp as the time the engine acts at, which is a close time. The one
  place this changes a result is `run_spread_backtest`'s session square-off,
  which flattens on the first bar whose stamp is at or after the cutoff
  (`squareoff_flags`, `src/core/session.rs`); open stamps would flatten one
  bar width late. The event session ignores `squareoff_time`: a strategy
  that squares off does so itself, from `ctx.bar_close_ts` or `ctx.datetime`.

## Three times of an event

Every event has three times, kept apart and never rewritten:

| time | what it is | where it comes from |
| --- | --- | --- |
| event time | when the exchange (or the venue, for a report) says it happened | the data: a row's timestamp, a report's `ts_ns` |
| arrival time | when the event reached the engine | the caller: the `arrival_ns` of a push; the event's own time when none is given |
| processing time | when the engine acted on it | arrival time plus the node's declared `processing_latency_ns` (default 0) |

Nothing here reads a wall clock, so a replay of the same pushes (with the same
arrival times) reproduces all three. In a batch run (`run_strategies`) they are
equal to the event's time. A strategy reads them in any hook as
`ctx.event_ns`, `ctx.arrival_ns` and `ctx.processing_ns`; for a hook run by a
venue report they are the report's `ts_ns`, the `arrival_ns` it was pushed
with and that plus the latency.

Event time is what the engine orders by and what clocks, bars, order expiry,
fills and trades carry. Arrival and processing time are for audit and latency
analysis: a journal that records them can replay a live session exactly. The
order-latency model (`order_latency_ns`, the delay before an order reaches a
venue's book) is a separate setting, applied in event time.

## Execution reports in the stream

When orders go to an execution adapter (`LiveNode(execution=...)`), what
became of them comes back as execution reports, and a report is an event in
the same single-threaded stream as market data. The canonical key gains one
phase, after the market data it can follow:

    (timestamp, phase, ...)    phase: trade 0, quote 1, depth 2, bar 3, report 4

* A report handed to `LiveNode.push_report` is applied when it is pushed, at
  its place in the order of pushes, like a pushed tick. Live pushes are
  ordered by arrival (see "Merge order"), so a journal of pushes, market data
  and reports alike, replays to the same state.
* The reports a simulated venue (`PaperExecution`) makes for an event are
  phase 4 at that event's timestamp: they are applied after the strategy's
  hooks for the event and the orders those hooks queued (the venue sees
  them), after the event has stepped the strategy's book, and before the next
  event. The fills they book are restated into the equity sample the event
  took, and the account's limits (drawdown, margin, daily loss) are checked
  on it again, so the book reads as if the matcher had filled in the step.
* A report whose own time is earlier than the latest market event is applied
  when it arrives. The fills and trades it books carry its time; the equity
  sample of the latest event is left alone, and the next event marks the book.

Reports may repeat, arrive late and arrive out of order; applying them again
or in another order leaves the same orders, positions and cash
(`order-lifecycle.md`, "Convergence"). Only the order a venue's reports are
*pushed* in can differ between runs, never what they mean.
