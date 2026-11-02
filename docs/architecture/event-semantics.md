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

## Not yet specified

- Separate exchange, arrival and processing timestamps. Each event carries a
  single timestamp, which serves as both its ordering key and the time at which
  the engine acts on it.
