# Published benchmark harness

Every performance figure in this repository is produced by these scripts. They exist so a
claim can be re-run rather than taken on trust.

```bash
uv run maturin develop --release      # measure the release build, never a dev one
uv run python benches/python/run_all.py
```

## Why the 0.7.0 numbers are larger than the 0.6.4 ones

They were measured under a different harness, not a slower engine. The 0.6.4
figures came from a one-off script that was not kept, over synthetic data whose
exact shape could not be reconstructed. Re-running the published 0.6.4 wheel and
the 0.7.0 build side by side on *this* harness, on identical inputs:

    0.6.4 (PyPI wheel)   min 68.9 us   p50 71.0 us   1,000 bars
    0.7.0 (this build)   min 68.4 us   p50 70.5 us   1,000 bars

Identical results (`total_return_pct = -1.964369`, 16 trades) from both. 0.7.0
is marginally faster. Treat the version-over-version difference in the published
tables as a change of ruler, and compare only within one harness.

## Measured on raptorbt 1.0.0

Published 1.0.0 PyPI wheel (cp312, macOS arm64), `run_all.py`, 2026-11-20. Apple M4, 10 cores, 24 GB, macOS 27.2,
Python 3.12.10, numpy 2.5.5. Median of 3 runs of the best-of-N time; the spread and sweep sections ran standalone,
as the harness requires. Other machines will differ; compare only within one harness and one machine.

| Bars | Best | M bars/s | Trades |
|---|---|---|---|
| 1,000 | 137.5 µs | 7.27 | 16 |
| 5,000 | 723.3 µs | 6.91 | 92 |
| 10,000 | 1.45 ms | 6.91 | 189 |
| 50,000 | 7.26 ms | 6.89 | 965 |
| 93,750 | 13.79 ms | 6.80 | 1,765 |
| 1,875,000 | 288.1 ms | 6.51 | 31,316 |
| 25,000,000 | 3.447 s | 7.25 | 98,822 |

| Ticks | Best | M ticks/s | Trades | Reached the end |
|---|---|---|---|---|
| 10,000 | 1.44 ms | 6.93 | 62 | yes |
| 100,000 | 14.72 ms | 6.79 | 542 | yes |
| 1,000,000 | 157.3 ms | 6.36 | 5,703 | yes |
| 10,000,000 | 1.908 s | 5.24 | 35,700 | no |

Spread batch: 394.8 ms serial, 58.3 ms parallel, 6.77x, 8,571 per second, 500 distinct returns, matches the serial run.
Sweep (184 combinations of 93,750 bars, 17.25M bar evaluations): 2.74 s wall, 2.52 s in the engine, peak RSS
71.9 MB. Determinism: 20 runs, 1 distinct result hash. Metrics: 48 attributes, 31 keys in `to_dict()`.
Engine extension: 6.4 MB.

## Slower than 0.13.x

On this harness and this machine, 1.0.0 is slower on the bar and tick paths than the published 0.13.4 wheel
(the 1.0.0 harness run against both; the 0.13.4 tag's own harness agrees with the 0.13.4 figures):

| Run | 0.13.4 | 1.0.0 |
|---|---|---|
| 1,000 bars | 72.6 µs | 137.5 µs |
| 5,000 bars | 370.2 µs | 723.3 µs |
| 1,875,000 bars | 151.1 ms | 288.1 ms |
| 25,000,000 bars | 2.089 s | 3.447 s |
| 10,000 ticks | 201.3M ticks/s | 6.9M ticks/s |
| 100,000 ticks | 182.8M ticks/s | 6.8M ticks/s |
| 1,000,000 ticks | 145.6M ticks/s | 6.4M ticks/s |
| 10,000,000 ticks | 135.3M ticks/s | 5.2M ticks/s |

The 0.13.4 column is a single run; the 1.0.0 column is the median of 3. A build of the 1.0.0 source made locally
measures the same as the wheel, so this is a change in the engine, not in the measurement or the packaging. Do
not read the 0.7.0 note above as covering it: that one was a change of harness.

## What is measured

`run_all.py` writes a JSON summary and prints the table that goes into the docs.
Two rules the scripts enforce, because breaking either produces a flattering
number that is wrong:

- **Release build only.** A `cargo build` without `--release` is several times
  slower and not what ships.
- **Tick runs must traverse the whole array.** `run_tick_backtest` defaults to
  `max_trades=50` and *stops early* once it hits the cap, so a naive timing
  measures a partial run and reports an inflated ticks/sec. The harness raises
  the cap and asserts the last trade's `exit_idx` lands at the end of the input.

## Live path, memory and concurrency benchmarks

`run_all.py` times the signal-array runners. These scripts time what a strategy written
against the `Strategy` API costs: on a `LiveNode` fed one event at a time, and in
`run_strategies`. They share `_live.py` (seeded synthetic data, the strategies, the
timing and memory helpers) and need nothing beyond numpy and the standard library
(`psutil` is used for RSS if it happens to be installed, `ps -o rss=` otherwise).

```bash
uv run maturin develop --release
uv run python benches/python/live_latency.py
uv run python benches/python/live_memory.py
uv run python benches/python/live_concurrency.py
uv run python benches/python/live_multi.py
uv run python benches/python/strategy_backtest.py
```

Each script prints the raptorbt, Python and numpy versions, the platform, machine and
CPU count, a table, and writes `<script>_results.json` next to itself (`--json PATH`
to write elsewhere). `--quick` runs a tiny configuration in a few seconds, to check a
script works; its numbers mean nothing. A case that fails, or whose checks fail, is
reported as an error entry with whatever it measured, and the remaining cases still run.

The data: ticks and 1-minute bars around a mean-reverting random walk on a 0.05 price
grid, timestamps strictly increasing inside a 09:00-15:30 UTC session on consecutive
weekdays. No session schedule or square-off is configured. Every strategy trades a fixed
10 units with no fees on an account of 1,000,000,000, so it cannot run out of cash. The
trading strategies carry a fixed stop on every entry and a trailing stop from the config,
so a position is open much of the time and stops are checked on most events.

What one number means:

- **`live_latency.py`**: one `LiveNode.push_tick` call, `perf_counter_ns` immediately
  before the call and after it returns. Inside: the Python wrapper's checks, the core
  applying the print and the quote, the 1-minute aggregation, every strategy hook the tick
  fires, the orders those hooks submit being matched in-process, stops and trailing stops,
  and the journal record when there is a journal. Not inside: preparing the arguments,
  storing the sample, the loop. The first 5 percent of ticks are excluded as warm-up. The
  bare timer overhead (two `perf_counter_ns` calls and a store) is measured and printed:
  subtract its p50 from a per-call p50. Cases are hook (`bars`: no tick hook, `noop`: a
  tick hook that only counts, `momentum`: a tick hook that reads an indicator and trades)
  x journal (`none`, `file` with fsync policy `--fsync`, default `orders`, `memory`
  keeping the latest `--journal-memory` records) x 1 / 10 / 50 strategies on one
  symbol, 1,000,000 ticks each. Each case checks that every tick produced its two events,
  that every strategy saw every print (or, for `bars`, a session of bars) and that every
  momentum strategy traded. `ticks/s` is the whole loop, timing included. Equity curves
  are unbounded (the node's default) unless `--curve-samples` is given: at 50 strategies
  x 1,000,000 ticks they hold about 1.6 GB, and their growth is inside the timed region.
- **`live_memory.py`**: one configuration per fresh interpreter. RSS over the baseline of
  an interpreter with raptorbt and numpy imported, after the node was built and fed;
  peak RSS (`getrusage`) over the same baseline; `node.memory_report()` (the Rust heap
  by category: rings, curves, books, indicators); and, in a second fresh interpreter,
  what `tracemalloc` sees Python allocate (never in the same pass as the RSS reading,
  because tracing changes RSS). The tape is generated in chunks as it is pushed. Matrix:
  1 / 10 / 50 / 200 symbols x `history` default / 512 (20,000 ticks per symbol), the
  same symbol counts with a 5-level depth snapshot after every 10th tick, and 100,000 /
  1,000,000 / 5,000,000 ticks on one symbol with `curve_samples` unset / 1,000, which
  shows what grows per million ticks.
- **`live_concurrency.py`**: `inproc`, 1 / 10 / 50 / 100 / 200 nodes in one process fed
  round-robin (one number is one push to one node; RSS per node is the RSS over the
  baseline divided by the node count); `procs`, 1 / 2 / 4 / 8 / `os.cpu_count()`
  processes with one node each, started with `multiprocessing` spawn and released by a
  barrier once built (aggregate ticks/s is every tick over the span from the first start
  to the last finish; RSS is the sum over the children, so shared pages count once per
  child); `threads`, 1 / 2 / 4 / 8 threads, each building and feeding its own node, to
  show what the interpreter lock allows.
- **`live_multi.py`**: one node, one strategy over 1 / 5 / 20 / 50 symbols, with
  timeframe sets `1m`, `1m+5m`, `1m+5m+15m+1h+1d` (composite streams through
  `subscribe(timeframe=)`, an SMA on each) and `1m+at` (one `ctx.indicators.expr` per
  symbol reading `at("5m", ...)` through `at("1d", ...)`). Fed 1,000,000 ticks over five
  sessions (one number is one `push_tick`) and twenty sessions of closed 1-minute bars
  through `push_group`, one bar per symbol per timestamp (one number is one
  `push_group` of K bars). Pushes are split into three distributions: those on which a
  coarser bar reached the strategy (for `1m+at`, the 1-minute bar that opens a new
  5-minute window, which is when an `at` value can move), those on which only a 1-minute
  bar closed, and the rest. A `chain` case runs an underlying and 50 option contracts
  given as `instruments=`, every contract receiving ticks, with the strategy reading
  `ctx.option_chain` on every 100th print of the underlying.
- **`strategy_backtest.py`**: one `run_strategies` call, call to return (runtime build,
  hooks, matching, results), for 1,000 / 100,000 / 1,000,000 bars and 100,000 /
  1,000,000 ticks; best and median of several runs (one run at the largest sizes);
  `ns/event` is the best time over the input length. Each run checks that the strategy
  saw every bar or print and traded.

Discipline for published figures:

- **Release build only** (`maturin develop --release`), as for `run_all.py`.
- **Run each script standalone on a quiet machine**: nothing else building or testing,
  one script at a time. `live_concurrency.py` and `live_memory.py` start their own
  processes; do not run anything beside them.
- **Three runs, report the medians**, and state the machine (CPU, cores, memory, OS,
  Python, numpy), as each script prints it.
- Compare only within one harness and one machine.
