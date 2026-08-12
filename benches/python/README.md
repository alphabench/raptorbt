# Published benchmark harness

Every performance figure in `README.md`, the frontend `/raptorbt` page, and
`frontend/public/raptorbt-doc.md` is produced by these scripts. They exist so a
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
