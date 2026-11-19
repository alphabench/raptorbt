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
